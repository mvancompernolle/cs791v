/*
  This header demonstrates how we build cuda programs spanning
  multiple files. 
 */

#ifndef REDUCE_H_
#define REDUCE_H_

#define DIM 1000

// This is the declaration of the function that will execute on the GPU.
__global__ void mandelbrotKernel(unsigned char *img);
__device__ int calcPixel(int x, int y, int size);
__global__ void kernel( unsigned char *ptr );

#endif // REDUCE_H_
