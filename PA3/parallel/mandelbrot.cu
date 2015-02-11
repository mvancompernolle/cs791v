
#include "mandelbrot.h"

__global__ void mandelbrotKernel(unsigned char *img, int size, int maxIterations) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  while (thread_id < size*size) {
    img[thread_id] = (unsigned char) calcPixel(thread_id%size, thread_id/size, size, maxIterations);
    thread_id += blockDim.x * gridDim.x;
  }

}

__device__ int calcPixel(int x, int y, int size, int maxIterations){
  float real_max = 2.0f, real_min = -2.0f, imag_min = -2.0f, imag_max = 2.0f;

  // calculate the scaling for the image
  float scale_real = (real_max - real_min)/size;
  float scale_imag = (imag_max - imag_min)/size;
  float cReal = real_min + ((float) x * scale_real);
  float cImag = imag_min + ((float) y * scale_imag);

  int count;
  float zReal, zImag;
  float temp, lengthsq;
  zReal = 0;
  zImag = 0;
  count = 0;

  do{
    temp = zReal * zReal - zImag * zImag + cReal;
    zImag = 2 * zReal * zImag + cImag;
    zReal = temp;
    lengthsq = zReal * zReal + zImag * zImag;
    count++;
  }while ((lengthsq < 4.0) && (count < maxIterations));

  return count;
}

/*__global__ void mandelbrotKernel(unsigned char *img, int size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int offset = x + y * blockDim.x * gridDim.x;
  int offset = x + y * size;
  int pixelVal;
  if(offset < size*size){
    pixelVal = calcPixel( x, y, size);
    img[offset] = (unsigned char) pixelVal;
  }

  while(x < size && y < size){
    img[offset] = (unsigned char) calcPixel(x, y, size);
    x += blockIdx.x * blockDim.x;
    y += blockIdx.y * blockDim.y;
    offset = x + y * size;
  }
  while(offset < size*size){
    pixelVal = calcPixel( x, y, size);
    img[offset] = (unsigned char) pixelVal;
    x += blockIdx.x * blockDim.x;
    y += blockIdx.y * blockDim.y;
    offset = x + y * blockDim.x * gridDim.x;
  }

}*/