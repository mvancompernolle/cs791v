/*
  This header demonstrates how we build cuda programs spanning
  multiple files. 
 */

#ifndef REDUCE_H_
#define REDUCE_H_

// This is the declaration of the function that will execute on the GPU.
__global__ void reduce(const int *input, int *block_results, const int N);

#endif // REDUCE_H_
