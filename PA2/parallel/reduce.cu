
#include "reduce.h"

__global__ void reduce(const int *input, int *block_results, const int N) {

  __shared__ int cache[4096];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  /*int x = 0;
  for(int i=0; i<4; i++){
    x = 0;
    if((tid + (i * blockDim.x * gridDim.x)) < N){
      x = input[tid + (i * blockDim.x * gridDim.x)];
    }
    cache[threadIdx.x + (i*1024)] = x;
  }*/

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
  }
}

