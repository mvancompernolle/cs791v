#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>

#include "reduce.h"

// used to take the average run time on the gpu
#define NUM_ITERATIONS 5

enum RUN_TYPE {CPU, RECURSIVE_HOST, RECURSIVE_DEVICE};

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

void runTest(Range numElements, Range numThreads, RUN_TYPE reductionType){
  float elapsedTime, totalTime;
  struct timeval tod1, tod2;
  cudaEvent_t start, end;
  std::ofstream fout;
  int correctSum = 0;
  int *input, *partialSums;
  int *devInput, *devPartialSums;
  int b = 0;
  int dev_result = 0;
  int currentSize;

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

    // loop for the number of threads
    for(int t = numThreads.start; t <= numThreads.end; t*= 2){

      b = (n + (t*2 - 1))/(t*2);
      if(b > 65535)
        b = 65535;

      partialSums = new int[b];

      err = cudaMalloc( (void**) &devPartialSums, (b) * sizeof(int));
      if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
      }

      totalTime = 0;

      // loop for number of iterations
      for(int i = 0; i < NUM_ITERATIONS; i++){
        cudaEventRecord( start, 0 );

        err = cudaMemcpy(devInput, input, n * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
          std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
          exit(1);
        }

        currentSize = n;
        dev_result = 0;

        int memorySize = (t <= 32) ? 2 * t * sizeof(int) : t * sizeof(int);

        // run normal reduction for CPU final reduction versions
        if(reductionType == CPU || reductionType == RECURSIVE_HOST)
          reduce<<<b, t, memorySize>>>(devInput, devPartialSums, n, t);
        // call the recursive kernel for device version
        else if(reductionType == RECURSIVE_DEVICE)
        {
          reduceRecursive<<<b, t, memorySize>>>(devInput, devPartialSums, n, t);
        }

        if(reductionType == RECURSIVE_HOST){
          // call reduce until completely reduced
          currentSize = b;
          while(currentSize > 1){
            reduce<<<currentSize, t, memorySize>>>(devPartialSums, devPartialSums, currentSize, t);
            currentSize = (currentSize + (t*2 - 1))/(t*2);
            std::cout << "currSize: " << currentSize << std::endl;
          }
        }

        // get result back
        cudaMemcpy(partialSums, devPartialSums, b * sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventRecord( end, 0 );
        cudaEventSynchronize( end );
        cudaEventElapsedTime( &elapsedTime, start, end );
        totalTime += elapsedTime;

        // get time to reduce on CPU
        if(reductionType == CPU){
          gettimeofday(&tod1, NULL);
          for(int i = 0; i < b; i++){
            dev_result += partialSums[i];
          }
          gettimeofday(&tod2, NULL);
          totalTime += todiff(&tod2, &tod1)/1000;
        }
        // perform final reduction for recursive host
        else if(reductionType == RECURSIVE_HOST){
          for(int index=0; index<currentSize; index++){
            dev_result += partialSums[index];
          }
        }
        // perfor final reduction for recursive device
        else{
          /*for(int index=0; index<b; index++){
            std::cout << "num " << partialSums[index] << std::endl;
          }*/
          dev_result = partialSums[0];
        }

        std::cout << "blocks: " << b << " t: " << t << " n: " << n << std::endl;
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
      std::cout << "Your program took: " << totalTime/NUM_ITERATIONS << " ms (I/O). " << std::endl;

      // output results to file
      fout << n << ", " << b << ", " << t << ", " << totalTime/NUM_ITERATIONS << std::endl;

      delete []partialSums;
      cudaFree(devPartialSums);

    } // end of threads loop

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
  Range numThreads, numElements;
  unsigned int currentDevice = 0;
  char inputc;
  int inputi;
  RUN_TYPE reductionType = CPU;
  numThreads.start = 1024;
  numThreads.end = 1024;
  numElements.start = 10000;
  numElements.end = 10000;

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
    std::cout << "3. Select Reduction type" << std::endl;
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
        std::cout << "Select the type of reduction you want (1. CPU | 2. Recursive Host | 3. Recursive Device)" << std::endl;
        std::cin >> inputi;
        switch(inputi){
          case 1:
            reductionType = CPU;
            break;
          case 2:
            reductionType = RECURSIVE_HOST;
            break;
          case 3:
            reductionType = RECURSIVE_DEVICE;
            break;
          default:
            reductionType = CPU;
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
        std::cout << "Reduction Type:\t\t"; 
        switch(reductionType){
          case CPU:
            std::cout << "CPU Reduction";
            break;
          case RECURSIVE_HOST:
            std::cout << "Recursive Host Reduction";
            break;
          case RECURSIVE_DEVICE:
            std::cout << "Recursive Device Reduction";
            break;
        }
        std::cout << std::endl;
        break;
      case '5':
        runTest(numElements, numThreads, reductionType);
        break;
      case 'Q': 
      case 'q':
        running = false;
        break;
    }

  }while(running);
}
