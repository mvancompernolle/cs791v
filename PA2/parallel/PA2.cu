#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>

#include "reduce.h"

// used to take the average run time on the gpu
#define NUM_ITERATIONS 5

struct Range{
  unsigned int start;
  unsigned int end;

  Range(){
    start = 2;
    end = 2;
  }

  void printRange(){
    if(end > start)
      std::cout << start << " - " << end;
    else
      std::cout << start;
  }
};


long long int todiff(struct timeval *tod1, struct timeval *tod2)
{
  long long t1, t2;
  t1 = tod1->tv_sec * 1000000 + tod1->tv_usec;
  t2 = tod2->tv_sec * 1000000 + tod2->tv_usec;
  return t1 - t2;
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

void runTest(Range numElements, Range numBlocks, Range numThreads){
  float elapsedTime, totalTime;
  float totalCPUTime;
  struct timeval tod1, tod2;
  cudaEvent_t start, end;
  std::ofstream fout;
  int correctSum = 0;
  int *input, *partialSums;
  int *devInput, *devPartialSums;

  fout.open("results.txt", std::ios::app);
  fout << std::endl << "Size, Blocks, Threads, Time" << std::endl;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // loop for the number of elements
  for(int n = numElements.start; n <= numElements.end; n*=2){

    // Arrays on the host (CPU)
    input = new int[n];

    // arrays on device (GPU)
    cudaError_t err = cudaMalloc( (void**) &devInput, n * sizeof(int));
    if (err != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
      exit(1);
    }

    correctSum = 0;
    for (int i = 0; i < n; ++i) {
      input[i] = rand() % 1 + 1;
      correctSum += input[i];
    }
    std::cout << "Correct Reduced Sum: " << correctSum << std::endl;

    // loop for the number of blocks
    for(int b = numBlocks.start; b <= numBlocks.end; b*=2){

      if(b == 65536)
        b = 65535;

      partialSums = new int[b];

      err = cudaMalloc( (void**) &devPartialSums, (b) * sizeof(int));
      if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
      }

      // loop for the number of threads
      for(int t = numThreads.start; t <= numThreads.end; t*= 2){

        totalTime = 0;
        totalCPUTime = 0;

        // loop for number of iterations
        for(int i = 0; i < NUM_ITERATIONS; i++){
          cudaEventRecord( start, 0 );

          err = cudaMemcpy(devInput, input, n * sizeof(int), cudaMemcpyHostToDevice);
          if (err != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
            exit(1);
          }

          int memorySize = t * sizeof(int);
          reduce<<<b, t, memorySize>>>(devInput, devPartialSums, n, t);

          // check to see if sum is correct
          cudaMemcpy(partialSums, devPartialSums, b * sizeof(int), cudaMemcpyDeviceToHost);

          gettimeofday(&tod1, NULL);
          int dev_result = 0;
          for(int i = 0; i < b; i++){
            dev_result += partialSums[i];
          }
          gettimeofday(&tod2, NULL);
          totalCPUTime = todiff(&tod2, &tod1);

          cudaEventRecord( end, 0 );
          cudaEventSynchronize( end );
          cudaEventElapsedTime( &elapsedTime, start, end );
          totalTime += elapsedTime;

          std::cout << "Device sum: " << dev_result << std::endl;

          if(dev_result != correctSum){
            std::cout << "Results did not match!" << std::endl;

            // clean up events - we should check for error codes here.
            cudaEventDestroy( start );
            cudaEventDestroy( end );

            // clean up device pointers
            cudaFree(devInput);
            cudaFree(devPartialSums);
            exit(1);
          }
        } // end of iterations loop

        // print results to screen
        std::cout << "Size: " << n << " Blocks: " << b << " Threads: " << t << std::endl;
        //std::cout << totalCPUTime/1000 << " " << totalTime << std::endl;
        std::cout << "Your program took: " << (totalTime + (totalCPUTime/1000))/NUM_ITERATIONS << " ms (I/O). " << std::endl;

        // output results to file
        fout << n << ", " << b << ", " << t << ", " << (totalTime + (totalCPUTime/1000))/NUM_ITERATIONS << std::endl;

      } // end of threads loop

      delete []partialSums;
      cudaFree(devPartialSums);
    } // end of blocks loop

    delete []input;
    cudaFree(devInput);

  } // end of vector size loop

  // Cleanup in the event of success.
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  fout.close();
}


int main(int argc, char *argv[]) {

  cudaDeviceProp props[2];
  int count = 1;
  char input;
  bool running = true; 
  Range numBlocks, numThreads, numElements;
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
    std::cout << std::endl << "-------------- VECTOR REDUCTION MENU ---------------" << std::endl;
    std::cout << "1. Select the vector input size" << std::endl;
    std::cout << "2. Select the number of threads" << std::endl;
    std::cout << "3. Select the number of blocks" << std::endl;
    std::cout << "4. Display run settings" << std::endl;
    std::cout << "5. Run vector reduction" << std::endl;
    std::cout << "Q. Quit Program" << std::endl;
    std::cout << "Select a menu option: ";
    std::cin >> input;
    std::cout << std::endl;

    switch(input){
      case '1':
        std::cout << "Select an input size: ";
        std::cin >> numElements.start;
        std::cout << "Do you want the input size to loop by powers of two? (Y/N): ";
        std::cin >> inputc;
        if(inputc == 'y' || inputc == 'Y'){
          std::cout << "Select an input size to stop looping at: ";
          std::cin >> inputi;
          numElements.end = inputi;
        }
        else if(inputc == 'n' || inputc == 'N'){
          numElements.end = numElements.start;
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
        std::cout << "------------- RUN INFO --------------" << std::endl;
        std::cout << "Vector Size:\t\t"; 
        numElements.printRange();
        std::cout << std::endl;
        std::cout << "Number of threads:\t"; 
        numThreads.printRange();
        std::cout << std::endl;
        std::cout << "Number of blocks:\t"; 
        numBlocks.printRange();
        std::cout << std::endl;
        break;
      case '5':
        runTest(numElements, numBlocks, numThreads);
        break;
      case 'Q': 
      case 'q':
        running = false;
        break;
    }

  }while(running);
}
