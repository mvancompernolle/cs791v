
#include "cuda.h"
#include "stdio.h"

/*
  This is the function that each thread will execute on the GPU. The
  fact that it executes on the device is indicated by the __global__
  modifier in front of the return type of the function. After that,
  the signature of the function isn't special - in particular, the
  pointers we pass in should point to memory on the device, but this
  is not indicated by the function's signature.
 */
__global__ void add(unsigned int* N, unsigned int* invN) {

  unsigned int* tmp = N;
  if(threadIdx.x == 0 && blockIdx.x == 0){
    for(int i=0; i<40; i++){
      printBitSet(tmp, 2);
      printf("\n");
      tmp+=2;
    }
    //printf("\nNumber of Vertices: %d\n", numV);
  }
  //printf("Hello from block %d, thread %d, %d\n", blockIdx.x, threadIdx.x, Adj[blockIdx.x * threadIdx.x]);

}

__device__ void printBitSet(unsigned int* bitSet, int size){

  // loop over each int
  for(int currInt = 0; currInt < size; currInt++){
    // loop over each bit in the int
    for(int b=0; b<sizeof(unsigned int)*8; b++){
      int shift = 1 << b;
      int val = bitSet[currInt] & shift;
      if(val != 0)
        val = 1;
      printf("%u ", val);
    }
    printf(" | ");
  }

}
