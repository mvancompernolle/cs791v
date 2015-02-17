/*
  This header demonstrates how we build cuda programs spanning
  multiple files. 
 */

#ifndef REDUCE_H_
#define REDUCE_H_

// This is the declaration of the function that will execute on the GPU.
__global__ void reduce(const int *input, int *block_results, const unsigned int N, const unsigned int blockSize);
__device__ void warpWideReduce(volatile int* sdata, const unsigned int tid, const unsigned int blockSize);
__global__ void reduceRecursive(const int *input, int *block_results, const unsigned int N, const unsigned int blockSize);

#endif // REDUCE_H_
