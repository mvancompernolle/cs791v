/*
  This program demonstrates the basics of working with cuda. We use
  the GPU to add two arrays. We also introduce cuda's approach to
  error handling and timing using cuda Events.

  This is the main program. You should also look at the header add.h
  for the important declarations, and then look at add.cu to see how
  to define functions that execute on the GPU.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "add.h"

// struct to contain loop info
struct LoopInfo{
  int min, max, step;
};

// used to take the average run time on the gpu
#define NUM_ITERATIONS 5

// create enumeration for normal and striding execution
enum RunType {NORMAL, STRIDING};

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
    std::cout << std::endl;
    std::cout << "General Info for device " << i << std::endl;
    std::cout << "Name: " << props[i].name << std::endl;
    std::cout << "Compute capability: " << props[i].major << "." << props[i].minor << std::endl;
    std::cout << "Clock rate: " << props[i].clockRate << std::endl;
    std::cout << "Device copy overlap: ";
    if(props[i].deviceOverlap)
      std::cout << "Enabled" << std::endl;
    else
      std::cout << "Disabled" << std::endl;
    std::cout << "Kernel execution timeout: ";
    if(props[i].kernelExecTimeoutEnabled)
      std::cout << "Enabled" << std::endl;
    else
      std::cout << "Disabled" << std::endl;

    std::cout << "  --- Memory Information for device " << i << std::endl;
    std::cout << "Total global mem: " << props[i].totalGlobalMem << std::endl;
    std::cout << "Total constant Mem: " << props[i].totalConstMem << std::endl;
    std::cout << "Max mem pitch: " << props[i].memPitch << std::endl;
    std::cout << "Texture Alignment: " << props[i].textureAlignment << std::endl;

    std::cout << "  --- MP Info for device " << i << std::endl;
    std::cout << "Multiprocessor count: " << props[i].multiProcessorCount << std::endl;
    std::cout << "Shared mem per mp: " << props[i].sharedMemPerBlock << std::endl;
    std::cout << "Registers per mp: " << props[i].regsPerBlock << std::endl;
    std::cout << "Thread in warp: " << props[i].warpSize << std::endl;
    std::cout << "Max threads per block: " << props[i].maxThreadsPerBlock << std::endl;
    std::cout << "Max thread dimensions: " << props[i].maxThreadsDim[0] << " " << props[i].maxThreadsDim[1] << " " << props[i].maxThreadsDim[2] << std::endl;
    std::cout << "Max grid dimensions: " << props[i].maxGridSize[0] << " " << props[i].maxGridSize[1] << " " << props[i].maxGridSize[2] << std::endl << std::endl;  
  }
}

void processFlags(int argc, char **argv, unsigned int& N, RunType& type, bool& timeIO, LoopInfo& bLoopInfo, LoopInfo& tLoopInfo){
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
      // input for run type (normal or striding)
      else if(arg == "-i"){
        if(strcmp(argv[i+1],"normal") == 0)
          type = NORMAL;
        else if(strcmp(argv[i+1],"striding") == 0)
          type = STRIDING;
        else{
          std::cout << "Improper command. Follow the format: -t (normal / striding)" << std::endl;
          exit(1);          
        }
        i++;
      }
      // input to time IO or not
      else if(arg == "-io"){
        if(strcmp(argv[i+1],"false") == 0){
          timeIO = false;
        }
        else
          timeIO = true;
        i++;
      }
      // input for block loop info
      else if(arg == "-bi"){
        if(argc > i+3){
          if(atoi(argv[i+1]) <= 0 || atoi(argv[i+2]) <= 0 || atoi(argv[i+3]) <= 0){
            std::cout << "Improper command. Follow the format: -bi (unsigned int) (unsigned int) (unsigned int). See ReadMe for more info." << std::endl;
            exit(1); 
          }
          if(atoi(argv[i+1]) >= atoi(argv[i+2])){
            std::cout << "Improper command. Value 2 must be greater than Value 1" << std::endl;
            exit(1);             
          }
          bLoopInfo.min = atoi(argv[i+1]);
          bLoopInfo.max = atoi(argv[i+2]);
          bLoopInfo.step = atoi(argv[i+3]);
          i+=3;
        }
        else{
          std::cout << "Improper command. Follow the format: -bi minBlockSize maxBlockSize blockStepSize." << std::endl;
          exit(1);
        }
      }
      // input for thread loop info
      else if(arg == "-ti"){
        if(argc > i+3){
          if(atoi(argv[i+1]) <= 0 || atoi(argv[i+2]) <= 0 || atoi(argv[i+3]) <= 0){
            std::cout << "Improper command. Follow the format: -ti (unsigned int) (unsigned int) (unsigned int). See ReadMe for more info." << std::endl;
            exit(1); 
          }
          if(atoi(argv[i+1]) >= atoi(argv[i+2])){
            std::cout << "Improper command. Value 2 must be greater than Value 1" << std::endl;
            exit(1);             
          }
          tLoopInfo.min = atoi(argv[i+1]);
          tLoopInfo.max = atoi(argv[i+2]);
          tLoopInfo.step = atoi(argv[i+3]);
          i+=3;
        }
        else{
          std::cout << "Improper command. Follow the format: -bi minThreadSize maxThreadSize threadStepSize." << std::endl;
          exit(1);
        }
      }
      // input for single block value
      else if(arg == "-b"){
          if(atoi(argv[i+1]) <= 0){
            std::cout << "Improper command. Follow the format: -b (unsigned int). See ReadMe for more info." << std::endl;
            exit(1); 
          }
          bLoopInfo.min = atoi(argv[i+1]);
          bLoopInfo.max = atoi(argv[i+1])+1;
          bLoopInfo.step = 1;
          i++;
      }
      // input for single thread value
      else if(arg == "-t"){
          if(atoi(argv[i+1]) <= 0){
            std::cout << "Improper command. Follow the format: -t (unsigned int). See ReadMe for more info." << std::endl;
            exit(1); 
          }
          tLoopInfo.min = atoi(argv[i+1]);
          tLoopInfo.max = atoi(argv[i+1])+1;
          tLoopInfo.step = 1;
          i++;
      }

    }

  }
}

int main(int argc, char *argv[]) {

  cudaDeviceProp props[2];
  float elapsedTime, totalTime;
  RunType type = STRIDING;
  int count = 1;
  unsigned int N = 1024;
  bool timeIO = true;
  cudaEvent_t start, end;
  LoopInfo tLoopInfo, bLoopInfo;
  std::ofstream fout;

  // get cuda info
  getCudaInfo(props, count);

  // if in sli, set cuda to use device 2 (more memory available)
  if(count == 2)
    cudaSetDevice(1);

  // set default min, max, and step sizes
  tLoopInfo.min = 1;
  tLoopInfo.max = props[0].maxThreadsPerBlock;
  tLoopInfo.step = 1;
  bLoopInfo.min = 1;
  bLoopInfo.max = 65535;
  bLoopInfo.step = 1;

  // process command line flags
  processFlags(argc, argv, N, type, timeIO, bLoopInfo, tLoopInfo);

  // open file to write results to
  std::string fileName;
  std::string ioStr;
  if(timeIO)
    ioStr = "io_";
  else
    ioStr = "no_io_";

  if(type == STRIDING){
    std::string result;
    std::ostringstream convert;
    convert << "results/Striding_results_" << ioStr << N << ".txt";
    fileName = convert.str();
    fout.open(fileName.c_str());
  }
  else{
    std::string result;
    std::ostringstream convert;
    convert << "results/Normal_results_" << ioStr << N << ".txt";
    fileName = convert.str();
    fout.open(fileName.c_str());
  }
  fout << "Blocks\t\tThreads\t\tTime" << std::endl;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Arrays on the host (CPU)
  int *a, *b, *c;
  a = new int[N];
  b = new int[N];
  c = new int[N];
  
  // arrays on device (GPU)
  int *dev_a, *dev_b, *dev_c;

  cudaError_t err = cudaMalloc( (void**) &dev_a, N * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  err = cudaMalloc( (void**) &dev_b, N * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  err = cudaMalloc( (void**) &dev_c, N * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }

  // These lines just fill the host arrays with some data so we can do
  // something interesting. Well, so we can add two arrays.
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i;
  }

  // loop through the number of blocks
  for(int bSize = bLoopInfo.min; bSize < bLoopInfo.max; bSize += bLoopInfo.step){
    // loop through the number of threads
    for(int tSize = tLoopInfo.min; tSize < tLoopInfo.max; tSize += tLoopInfo.step){

      // if not doing striding, make sure blocks * threads is atleast as big as the vector size
      if(bSize * tSize < N && type == NORMAL)
        continue;

      // reset times
      totalTime = 0;

      // loop to get an average run time
      for(int i=0; i<NUM_ITERATIONS; i++){

        if(timeIO)
          cudaEventRecord( start, 0 );

        cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

        if(!timeIO)
          cudaEventRecord( start, 0 );
        
        if(type == NORMAL){
          add<<<bSize, tSize>>>(dev_a, dev_b, dev_c, N);
        }
        else{
          addStriding<<<bSize, tSize>>>(dev_a, dev_b, dev_c, N);
        }

        if(!timeIO){
          cudaEventRecord( end, 0 );
          cudaEventSynchronize( end );
          cudaEventElapsedTime( &elapsedTime, start, end );
          totalTime += elapsedTime;
        }

        cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

        if(timeIO){
          cudaEventRecord( end, 0 );
          cudaEventSynchronize( end );
          cudaEventElapsedTime( &elapsedTime, start, end );
          totalTime += elapsedTime;
        }

        for (int i = 0; i < N; ++i) {
          if (c[i] != a[i] + b[i]) {
            std::cerr << "Oh no! Something went wrong. You should check your cuda install and your GPU. :(" << std::endl;

            // clean up events - we should check for error codes here.
            cudaEventDestroy( start );
            cudaEventDestroy( end );

            // clean up device pointers
            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_c);
            exit(1);
          }
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

  delete []a;
  delete []b;
  delete []c;

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

}
