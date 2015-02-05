
#include "reduce.h"

// REFERENCE : http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

__device__ void warpWiseReduce(volatile int* sdata, const unsigned int tid, const unsigned int blockSize){
  // reduce without sync (threads in warp all execute at same time)
  if(blockSize >= 64)
    sdata[tid] += sdata[tid + 32];

  if(blockSize >= 32)
    sdata[tid] += sdata[tid + 16];

  if(blockSize >= 16)
    sdata[tid] += sdata[tid + 8];

  if(blockSize >= 8)
    sdata[tid] += sdata[tid + 4];

  if(blockSize >= 4)
    sdata[tid] += sdata[tid + 2];

  if(blockSize >= 2)
    sdata[tid] += sdata[tid + 1];
}

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

  // repeatedly sum the left half of shared memory with the right half
  // effectively reduces by half of current size each time and leaves half of threads idle
  // make sure to sync between reducitons
  if(blockSize >= 1024 && threadIdx.x < 512){
    sharedData[threadIdx.x] += sharedData[threadIdx.x + 512];
  }
  __syncthreads();

  if(blockSize >= 512 && threadIdx.x < 256){
    sharedData[threadIdx.x] += sharedData[threadIdx.x + 256];
  }
  __syncthreads();

  if(blockSize >= 256 && threadIdx.x < 128){
    sharedData[threadIdx.x] += sharedData[threadIdx.x + 128];
  }
  __syncthreads();

  if(blockSize >= 128 && threadIdx.x < 64){
    sharedData[threadIdx.x] += sharedData[threadIdx.x + 64];
  }
  __syncthreads();

  // when small enough for a warp to reduce
  if(threadIdx.x < 32){
    warpWiseReduce(sharedData, threadIdx.x, blockSize);
  }

  // save block's partial sum in results
  if(threadIdx.x == 0){
    blockResults[blockIdx.x] = sharedData[0];
  }
}


__global__ void reduce2(const int *input, int *blockResults, const unsigned int N, const unsigned int blockSize) {

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

  // repeatedly sum the left half of shared memory with the right half
  // effectively reduces by half of current size each time and leaves half of threads idle
  // make sure to sync between reducitons
  if(blockSize >= 1024 && threadIdx.x < 512){
    sharedData[threadIdx.x] += sharedData[threadIdx.x + 512];
  }
  __syncthreads();

  if(blockSize >= 512 && threadIdx.x < 256){
    sharedData[threadIdx.x] += sharedData[threadIdx.x + 256];
  }
  __syncthreads();

  if(blockSize >= 256 && threadIdx.x < 128){
    sharedData[threadIdx.x] += sharedData[threadIdx.x + 128];
  }
  __syncthreads();

  if(blockSize >= 128 && threadIdx.x < 64){
    sharedData[threadIdx.x] += sharedData[threadIdx.x + 64];
  }
  __syncthreads();

  // when small enough for a warp to reduce
  if(threadIdx.x < 32){
    warpWiseReduce(sharedData, threadIdx.x, blockSize);
  }

  // save block's partial sum in results
  if(threadIdx.x == 0){
    blockResults[blockIdx.x] = sharedData[0];
  }
}


  /*__shared__ int cache[1024];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  /*int x = 0;
  for(int i=0; i<4; i++){
    x = 0;
    if((tid + (i * blockDim.x * gridDim.x)) < N){
      x = input[tid + (i * blockDim.x * gridDim.x)];
    }
    cache[threadIdx.x + (i*1024)] = x;
  }

  int x = 0;
  if(tid < N){
    x = input[tid];
  }
  cache[threadIdx.x] = x;

  __syncthreads();

  int i = blockDim.x/2;
  while(i != 0){
    if(threadIdx.x < i){
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }

  if(threadIdx.x == 0){
    block_results[blockIdx.x] = cache[0];
  }*/
