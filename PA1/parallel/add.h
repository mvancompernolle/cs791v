/*
  This header demonstrates how we build cuda programs spanning
  multiple files. 
 */

#ifndef ADD_H_
#define ADD_H_

// This is the declaration of the function that will execute on the GPU.
__global__ void add(int*, int*, int*, int);
__global__ void addStriding(int*, int*, int*, int);

#endif // ADD_H_
