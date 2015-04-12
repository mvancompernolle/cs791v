/*
  This header demonstrates how we build cuda programs spanning
  multiple files. 
 */

#ifndef ADD_H_
#define ADD_H_

__constant__ int numV;

// This is the declaration of the function that will execute on the GPU.
__global__ void add(unsigned int* N, unsigned int* invN);
__device__ void printBitSet(unsigned int* bitSet, int size);

#endif // ADD_H_
