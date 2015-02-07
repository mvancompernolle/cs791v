
#include "reduce.h"

__global__ void mandelbrotKernel(unsigned char *img) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * gridDim.x;
  int pixelVal = calcPixel( x, y, gridDim.x);
  img[offset] = (unsigned char) pixelVal;
}

__device__ int calcPixel(int x, int y, int size){
  float real_max = 2.0f, real_min = -2.0f, imag_min = -2.0f, imag_max = 2.0f;

  // calculate the scaling for the image
  float scale_real = (real_max - real_min)/size;
  float scale_imag = (imag_max - imag_min)/size;
  float cReal = real_min + ((float) x * scale_real);
  float cImag = imag_min + ((float) y * scale_imag);

  int count, max_iter;
  float zReal, zImag;
  float temp, lengthsq;
  max_iter = 1024;
  zReal = 0;
  zImag = 0;
  count = 0;

  do{
    temp = zReal * zReal - zImag * zImag + cReal;
    zImag = 2 * zReal * zImag + cImag;
    zReal = temp;
    lengthsq = zReal * zReal + zImag * zImag;
    count++;
  }while ((lengthsq < 4.0) && (count < max_iter));

  return count;
}

struct cuComplex {
float r;
float i;
__device__ cuComplex( float a, float b ) : r(a), i(b)
  {}
  __device__ float magnitude2( void ) {
  return r * r + i * i;
  }
  __device__ cuComplex operator*(const cuComplex& a) {
  return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
  }
  __device__ cuComplex operator+(const cuComplex& a) {
  return cuComplex(r+a.r, i+a.i);
}
};

__device__ int julia( int x, int y ) {
  const float scale = 1.5;
  float jx = scale * (float)(DIM/2 - x)/(DIM/2);
  float jy = scale * (float)(DIM/2 - y)/(DIM/2);
  cuComplex c(-0.8, 0.156);
  cuComplex a(jx, jy);
  int i = 0;
  for (i=0; i<200; i++) {
    a = a * a + c;
    if (a.magnitude2() > 1000)
      return 0;
  }
  return 1;
}

__global__ void kernel( unsigned char *ptr ) {
  // map from threadIdx/BlockIdx to pixel position
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * gridDim.x;
  // now calculate the value at that position
  int juliaValue = julia( x, y );
  ptr[offset*4 + 0] = 255 * juliaValue;
  ptr[offset*4 + 1] = 0;
  ptr[offset*4 + 2] = 0;
  ptr[offset*4 + 3] = 255;
}
