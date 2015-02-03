#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sys/time.h>

#include "reduce.h"

// used to take the average run time on the gpu
#define NUM_ITERATIONS 5

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

void processFlags(int argc, char **argv, unsigned int& N){
  if(argc > 2){

    // loop over input for commands
    for(int i=1; i<argc-1; i++){
      std::string arg = argv[i];

      // input for vector size
      if(arg == "-s"){

        // error checking
        if(atoi(argv[i+1]) <= 0){
          std::cout << "Improper command. Follow the format: -s (unsigned int)" << std::endl;
          exit(1); 
        }
        N = atoi(argv[i+1]);
        i++;
      }
    }

  }
}

int main(int argc, char *argv[]) {

  cudaDeviceProp props[2];
  float elapsedTime, totalTime;
  int count = 1, correctSum = 0;
  unsigned int N = 1000000;
  bool timeIO = true;
  cudaEvent_t start, end;
  std::ofstream fout;

  int numBlocks, numThreads, currentDevice = 0;

  // get cuda info
  getCudaInfo(props, count);

  // if in sli, set cuda to use device 2 (more memory available)
  if(count == 2){
    currentDevice = 1;
    cudaSetDevice(currentDevice);
  }

  // process command line flags
  processFlags(argc, argv, N);

  // get max number of threads
  numThreads = props[currentDevice].maxThreadsPerBlock;
  //numThreads = 1024;
  numBlocks = (N/numThreads) + ((N%numThreads) ? 1 : 0);
  //numBlocks = numBlocks = imin(32, (N+numThreads-1) / numThreads);

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Arrays on the host (CPU)
  int *input, *partialSums;
  input = new int[N];
  partialSums = new int[numBlocks];
  
  // arrays on device (GPU)
  int *dev_input, *dev_partial_sums;

  cudaError_t err = cudaMalloc( (void**) &dev_input, N * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  err = cudaMalloc( (void**) &dev_partial_sums, (numBlocks) * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }

  srand(time(NULL));
  for (int i = 0; i < N; ++i) {
    input[i] = rand() % 10;
    correctSum += input[i];
  }
  std::cout << "Correct Reduced Sum: " << correctSum << std::endl;

  // loop through the number of blocks
  for(int bSize = 1; bSize < 2; bSize += 1){
    // loop through the number of threads
    for(int tSize = 1; tSize < 2; tSize += 1){

      // reset times
      totalTime = 0;

      // loop to get an average run time
      for(int i=0; i<NUM_ITERATIONS; i++){

        if(timeIO)
          cudaEventRecord( start, 0 );

        err = cudaMemcpy(dev_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
          std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
          exit(1);
        }

        if(!timeIO)
          cudaEventRecord( start, 0 );

        reduce<<<numBlocks, numThreads>>>(dev_input, dev_partial_sums, N);

        if(!timeIO){
          cudaEventRecord( end, 0 );
          cudaEventSynchronize( end );
          cudaEventElapsedTime( &elapsedTime, start, end );
          totalTime += elapsedTime;
        }

        // check to see if sum is correct
        cudaMemcpy(partialSums, dev_partial_sums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

        int dev_result = 0;
        for(int i = 0; i < numBlocks; i++){
          dev_result += partialSums[i];
        }
        
        if(timeIO){
          cudaEventRecord( end, 0 );
          cudaEventSynchronize( end );
          cudaEventElapsedTime( &elapsedTime, start, end );
          totalTime += elapsedTime;
        }

        std::cout << "Device sum: " << dev_result << std::endl;


        if(dev_result != correctSum){
          std::cout << "Results did not match!" << std::endl;

          // clean up events - we should check for error codes here.
          cudaEventDestroy( start );
          cudaEventDestroy( end );

          // clean up device pointers
          cudaFree(dev_input);
          cudaFree(dev_partial_sums);
          exit(1);
        }
      } // end of iterations loop

      std::cout << "Size: " << N << " Blocks: " << bSize << " Threads: " << tSize << std::endl;
      std::cout << "Your program took: " << totalTime/NUM_ITERATIONS << " ms." << std::endl;

      // output to file
      fout << bSize << ", " << tSize << ", " << totalTime/NUM_ITERATIONS << std::endl;

    } // end of threads loop
  } // end of blocks loop

  // Cleanup in the event of success.
  cudaEventDestroy( start );
  cudaEventDestroy( end );

  delete []input;
  delete []partialSums;

  cudaFree(dev_input);
  cudaFree(dev_partial_sums);

}
