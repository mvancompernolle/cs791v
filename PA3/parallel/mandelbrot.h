/*
  This header demonstrates how we build cuda programs spanning
  multiple files. 
 */

#ifndef REDUCE_H_
#define REDUCE_H_

// This is the declaration of the function that will execute on the GPU.
__global__ void mandelbrotKernel(unsigned char *img, int size, int maxIterations);
__device__ int calcPixel(int x, int y, int size, int maxIterations);

#endif // REDUCE_H_
