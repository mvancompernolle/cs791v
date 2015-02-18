
#include "reduce.h"

// REFERENCE : http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

__global__ void reduce(const int *input, int *blockResults, const unsigned int N, const unsigned int blockSize) {

  extern __shared__ int sharedData[];
  int i = blockIdx.x*(blockSize*2) + threadIdx.x;
  int gridStep = blockSize*2*gridDim.x;
  sharedData[threadIdx.x] = 0;

  int sum = 0;

  // attempt to perform an initial reduce on loading for performance
  // stride so that an input vector larger than the total shared memory between blocks still fits
  while(i < N){

    sum += input[i];

    if(i + blockSize < N)
      sum += input[i + blockSize];

    i += gridStep;
  }
  sharedData[threadIdx.x] = sum;

  // sync to make sure all shared memory has been initialized
  __syncthreads();

  // loop version
  int size = blockSize;
  for(unsigned int s=blockDim.x/2; s>0; s/=2){
    __syncthreads();

    if(threadIdx.x < s){
      sharedData[threadIdx.x] += sharedData[threadIdx.x+s];

      if(size%2 == 1 && threadIdx.x == 0)
        sharedData[threadIdx.x] += sharedData[size-1];
    }

    size /= 2;
  }

  __syncthreads();

  // save block's partial sum in results
  if(threadIdx.x == 0){
    blockResults[blockIdx.x] = sharedData[0];
  }
}

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
__global__ void reduceRecursive(const int *input, int *blockResults, const unsigned int N, const unsigned int blockSize) {

  extern __shared__ int sharedData[];
  int i = blockIdx.x*(blockSize*2) + threadIdx.x;
  int gridStep = blockSize*2*gridDim.x;
  sharedData[threadIdx.x] = 0;

  int sum = 0;

  // attempt to perform an initial reduce on loading for performance
  // stride so that an input vector larger than the total shared memory between blocks still fits
  while(i < N){

    sum += input[i];

    if(i + blockSize < N)
      sum += input[i + blockSize];

    i += gridStep;
  }
  sharedData[threadIdx.x] = sum;

  // sync to make sure all shared memory has been initialized
  __syncthreads();

  // loop version
  int size = blockSize;
  for(unsigned int s=blockDim.x/2; s>0; s/=2){
    __syncthreads();

    if(threadIdx.x < s){
      sharedData[threadIdx.x] += sharedData[threadIdx.x+s];

      if(size%2 == 1 && threadIdx.x == 0)
        sharedData[threadIdx.x] += sharedData[size-1];
    }

    size /= 2;
  }

  __syncthreads();

  // save block's partial sum in results
  if(threadIdx.x == 0){
    blockResults[blockIdx.x] = sharedData[0];

    __threadfence();

    unsigned int value = atomicInc(&count, gridDim.x);
    isLastBlockDone = (value == (gridDim.x-1));
  }

  __syncthreads();

  if(isLastBlockDone && threadIdx.x == 0 && N > 1){
    isLastBlockDone = false;
    count = 0;
    int b = (gridDim.x + (blockSize*2 - 1))/(blockSize*2);
    int memSize = (blockSize <= 32) ? 2 * blockSize * sizeof(int) : blockSize * sizeof(int);
    reduceRecursive<<<b, blockSize, memSize>>>(blockResults, blockResults, gridDim.x, blockSize);
  }

}
