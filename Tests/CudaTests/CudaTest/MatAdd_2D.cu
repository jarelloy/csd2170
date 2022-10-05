#include "Tests.h"

__global__
void MatAddKernel(float* matA, float* matB, float* output, int nx, int ny)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < nx && y < ny)
  {
    int id = y * nx + x;
    output[id] = matA[id] + matB[id];
  }
}

void MatAdd_2D()
{
  constexpr int xSize{ 100 };
  constexpr int ySize{ 50 };
  constexpr int elemCt{ xSize * ySize };
  constexpr size_t matSize{ elemCt * sizeof(float) };
  
  std::vector<float> matA(elemCt, 5.0f);
  std::vector<float> matB(elemCt, 10.0f);
  std::vector<float> output(elemCt, 0.0f);

  float* dA{}, * dB{}, * dOut{};
  cudaMalloc((void**)&dA, matSize);
  cudaMalloc((void**)&dB, matSize);
  cudaMalloc((void**)&dOut, matSize);

  cudaMemcpy(dA, matA.data(), matSize, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, matB.data(), matSize, cudaMemcpyHostToDevice);
  
  dim3 gridSize{ 100/32 + 1, 100/32 + 1, 1 };
  dim3 blockSize{ 32,32,1 };
  MatAddKernel <<< gridSize, blockSize >>> (dA, dB, dOut, xSize, ySize);
  cudaDeviceSynchronize();

  cudaMemcpy(output.data(), dOut, matSize, cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);

  std::ofstream ofs{ "./MatAdd_2D.txt" };
  for (int y{}; y < ySize; ++y)
  {
    for (int x{}; x < xSize; ++x)
    {
      ofs << output[y * xSize + x] << " ";
    }
    ofs << '\n';
  }
}