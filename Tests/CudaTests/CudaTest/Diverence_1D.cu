#include "Tests.h"


__global__ 
void Divert1dKernal(float* in, float* out, int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) out[i] = 2 * in[i];
}

void Divergence_1D()
{
  float* baseValue{ new float[1000]{1.0f} };
  size_t arraySize{ sizeof(float) * 1000 };

  float* dIn{};
  cudaMalloc((void**)&dIn, arraySize);
  cudaMemcpy(dIn, baseValue, arraySize, cudaMemcpyHostToDevice);

  dim3 gridDim{ 32,1,1 };
  dim3 blockDim{ 32,1,1 };
  Divert1dKernal <<< gridDim, blockDim >>> (dIn, dIn, 1000);
}
