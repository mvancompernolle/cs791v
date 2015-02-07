#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "reduce.h"

// used to take the average run time on the gpu
#define NUM_ITERATIONS 5

struct Range{
  unsigned int start;
  unsigned int end;

  Range(){
    start = 1;
    end = 1;
  }

  void printRange(){
    if(end > start)
      std::cout << start << " - " << end;
    else
      std::cout << start;
  }
};

void getCudaInfo(cudaDeviceProp props[], int& count);
void runTest(Range numElements, Range numBlocks, Range numThreads);
bool write_image(unsigned char *image, int size);

int main(int argc, char *argv[]) {

  cudaDeviceProp props[2];
  int count = 1;
  char input;
  bool running = true; 
  Range size, numIterations, numThreads, numBlocks;
  unsigned int currentDevice = 0;
  char inputc;
  int inputi;

  srand(time(NULL));

  // get cuda info
  getCudaInfo(props, count);

  // if in sli, set cuda to use device 2 (more memory available)
  if(count == 2){
    currentDevice = 1;
    cudaSetDevice(currentDevice);
  }

  do{
    // print menu options
    std::cout << std::endl << "-------------- MANDELBROT MENU ---------------" << std::endl;
    std::cout << "1. Select the vector input size" << std::endl;
    std::cout << "2. Select the number of threads" << std::endl;
    std::cout << "3. Select the number of blocks" << std::endl;
    std::cout << "4. Select the max number of iterations" << std::endl;
    std::cout << "5. Display run settings" << std::endl;
    std::cout << "6. Run vector reduction" << std::endl;
    std::cout << "Q. Quit Program" << std::endl;
    std::cout << "Select a menu option: ";
    std::cin >> input;
    std::cout << std::endl;

    switch(input){
      case '1':
        std::cout << "Select an input size: ";
        std::cin >> size.start;
        std::cout << "Do you want the input size to loop by powers of two? (Y/N): ";
        std::cin >> inputc;
        if(inputc == 'y' || inputc == 'Y'){
          std::cout << "Select an input size to stop looping at: ";
          std::cin >> inputi;
          size.end = inputi;
        }
        else if(inputc == 'n' || inputc == 'N'){
          size.end = size.start;
        }
        break;

      case '2':
        std::cout << "Select the number of threads: ";
        std::cin >> numThreads.start;
        std::cout << "Do you want the the number of threads to loop by powers of two? (Y/N): ";
        std::cin >> inputc;
        if(inputc == 'y' || inputc == 'Y'){
          std::cout << "Select a number of threads to stop looping at: ";
          std::cin >> inputi;
          numThreads.end = inputi;
        }
        else if(inputc == 'n' || inputc == 'N'){
          numThreads.end = numThreads.start;
        }
        break;

      case '3':
        std::cout << "Select the number of blocks: ";
        std::cin >> numBlocks.start;
        std::cout << "Do you want the the number of blocks to loop by powers of two? (Y/N): ";
        std::cin >> inputc;
        if(inputc == 'y' || inputc == 'Y'){
          std::cout << "Select a number of blocks to stop looping at: ";
          std::cin >> inputi;
          numBlocks.end = inputi;
        }
        else if(inputc == 'n' || inputc == 'N'){
          numBlocks.end = numBlocks.start;
        }
        break;

      case '4':
        std::cout << "Select the number of max iterations: ";
        std::cin >> numIterations.start;
        std::cout << "Do you want the the number of max iterations to loop? (Y/N): ";
        std::cin >> inputc;
        if(inputc == 'y' || inputc == 'Y'){
          std::cout << "Select a number of max iterations to stop looping at: ";
          std::cin >> inputi;
          numIterations.end = inputi;
        }
        else if(inputc == 'n' || inputc == 'N'){
          numIterations.end = numIterations.start;
        }
        break;

      case '5':
        std::cout << "------------- RUN INFO --------------" << std::endl;
        std::cout << "Vector Size:\t\t"; 
        size.printRange();
        std::cout << std::endl;
        std::cout << "Number of threads:\t"; 
        numThreads.printRange();
        std::cout << std::endl;
        std::cout << "Number of blocks:\t"; 
        numBlocks.printRange();
        std::cout << "Number of max iterations:\t"; 
        numIterations.printRange();
        std::cout << std::endl;
        break;
      case '6':
        runTest(size, numBlocks, numThreads);
        break;
      case 'Q': 
      case 'q':
        running = false;
        break;
    }

  }while(running);
}

void getCudaInfo(cudaDeviceProp props[], int& count){

  cudaError_t err;
  err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }

  for(int i=0; i<count; i++){
    err = cudaGetDeviceProperties(&props[i], i);
    if (err != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
      exit(1);
    }
  }
}

void runTest(Range size, Range numBlocks, Range numThreads){
  float elapsedTime, totalTime;
  cudaEvent_t start, end;
  std::ofstream fout;
  unsigned char *img, *devImg;

  //fout.open("results.txt", std::ios::app);
  //fout << std::endl << "Size, Blocks, Threads, Time" << std::endl;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // loop for the number of elements
  for(int n = size.start; n <= size.end; n*=2){

    // Arrays on the host (CPU)
    img = new unsigned char[n*n];

          /*for(int i=0; i<n/2; i++){
            for(int j=0; j<n; j++){
              img[i*n + j ] = 0;
            }
          }
          for(int i=n/2; i<n; i++){
            for(int j=0; j<n; j++){
              img[i*n + j ] = 255;
            }
          }*/

    // arrays on device (GPU)
    cudaError_t err = cudaMalloc( (void**) &devImg, n * n * sizeof(unsigned char));
    if (err != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
      exit(1);
    }

    // loop for the number of blocks
    for(int b = numBlocks.start; b <= numBlocks.end; b*=2){

      if(b == 65536)
        b = 65535;

      // loop for the number of threads
      for(int t = numThreads.start; t <= numThreads.end; t*= 2){

        totalTime = 0;

        // loop for number of iterations
        for(int i = 0; i < NUM_ITERATIONS; i++){
          cudaEventRecord( start, 0 );

          err = cudaMemcpy(devImg, img, n * n * sizeof(unsigned char), cudaMemcpyHostToDevice);
          if (err != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
            exit(1);
          }

          dim3 grid(n,n);
          mandelbrotKernel<<<grid, 1>>>(devImg);

          cudaMemcpy(img, devImg, n * n * sizeof(unsigned char), cudaMemcpyDeviceToHost);

          for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
              //std::cout << i << " " << j << " " << img[i * n + j] << " " << n << std::endl;
            }
          }

          cudaEventRecord( end, 0 );
          cudaEventSynchronize( end );
          cudaEventElapsedTime( &elapsedTime, start, end );
          totalTime += elapsedTime;

          /*if(dev_result != correctSum){
            std::cout << "Results did not match!" << std::endl;

            // clean up events - we should check for error codes here.
            cudaEventDestroy( start );
            cudaEventDestroy( end );

            // clean up device pointers
            cudaFree(devInput);
            cudaFree(devPartialSums);
            exit(1);
          }*/
        } // end of iterations loop

        // print results to screen
        std::cout << "Size: " << n << " Blocks: " << b << " Threads: " << t << std::endl;
        //std::cout << totalCPUTime/1000 << " " << totalTime << std::endl;
        std::cout << "Your program took: " << totalTime/NUM_ITERATIONS << " ms (I/O). " << std::endl;

        // output results to file
        //fout << n << ", " << b << ", " << t << ", " << (totalTime + (totalCPUTime/1000))/NUM_ITERATIONS << std::endl;

      } // end of threads loop

    } // end of blocks loop

    write_image(img, n);

    delete []img;
    cudaFree(devImg);

  } // end of vector size loop

  // Cleanup in the event of success.
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  fout.close();
}

bool write_image(unsigned char *image, int size){
    /* http://stackoverflow.com/questions/4346831/saving-numerical-2d-array-to-image */
    FILE *f = fopen("mandelbrot.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", size, size);

  unsigned int r = 0, g = 0, b = 0;
    for (int y=0; y<size; y++)
        for (int x=0; x<size; x++)
        {
      // shade mandelbrot as red
      b = image[y * size + x];
      r = image[y * size + x];
      g = image[y * size + x];
            fputc(r, f);   // 0 .. 255
            fputc(g, f); // 0 .. 255
            fputc(b, f);  // 0 .. 255
        }
    fclose(f);
    
    return true;
}
