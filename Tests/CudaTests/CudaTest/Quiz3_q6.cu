#include "Tests.h"


__global__ 
void MatrixMulKernel(float* M, float* N, float* P, int Width) 
{
  int Row = blockIdx.y * blockDim.y + threadIdx.y; 
  int Col = blockIdx.x * blockDim.x + threadIdx.x; 
  
  if ((Row < Width) && (Col < Width)) 
  {
    float Pvalue = 0;
    
    for (int k = 0; k < Width; ++k) 
    {
      Pvalue += M[Row * Width + k] * N[k * Width + Col];
    } 
    
    P[Row * Width + Col] = Pvalue;
  }
}



void Q3_qn6()
{
  std::vector<float> matA(1000*1000, 1.0f);
  std::vector<float> matB( 1000 * 1000, 2.0f) ;
  std::vector<float> output( 1000 * 1000, 5.0f) ;

  //std::fill(matA.begin(), matA.end(), 1.0f);
  //std::fill(matB.begin(), matB.end(), 2.0f);

  float* dA{}, *dB{}, *dOut{};

  size_t size{ 1000 * 1000 * sizeof(float) };
  cudaMalloc((void**)&dA, size);
  cudaMalloc((void**)&dB, size);
  cudaMalloc((void**)&dOut, size);

  cudaMemcpy(dA, matA.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(dA, matB.data(), size, cudaMemcpyHostToDevice);

  dim3 gridSize{ 63,63,1 };
  dim3 blockSize{ 16,16,1 };
  MatrixMulKernel <<< gridSize, blockSize >>> (dA, dB, dOut, 1000);
  cudaDeviceSynchronize();

  cudaMemcpy(output.data(), dOut, size, cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dOut);

  std::cout << output[0] << '\n';
}
