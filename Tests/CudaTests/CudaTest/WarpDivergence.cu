#include "Tests.h"


__global__
void PictureMultKernal(float* in, float* out, int matrixWidth, int matrixHeight)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < matrixHeight && col < matrixWidth)
  {
    out[row * matrixWidth + col] = 2 * in[row * matrixWidth];
  }
}


void WarpDivergence()
{
  float* picture{ new float[800 * 600] };

  size_t matrixSize{ sizeof(float) * 800 * 600 };
  
  float *dIn{}, *dOut{};
  cudaMalloc((void**)&dIn, matrixSize);
  cudaMalloc((void**)&dOut, matrixSize);

  cudaMemcpy(dIn, picture, matrixSize, cudaMemcpyHostToDevice);

  dim3 gridSize{ (unsigned)ceil(600 / 32), (unsigned)ceil(800 / 32), 1 };
  dim3 blockSize{ 32,32,1 };
  PictureMultKernal <<< gridSize, blockSize >>> (dIn, dOut, 800, 600);

}