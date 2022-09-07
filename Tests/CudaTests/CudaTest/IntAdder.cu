#include "Tests.h"


__global__
void CudaIntAdder(int* lhs, int* rhs, int* res)
{
  res[threadIdx.x] = lhs[threadIdx.x] + rhs[threadIdx.x];
}

void IntAdder()
{
  std::vector<int> lhs{ 1,3 };
  std::vector<int> rhs{ 3,1 };
  std::vector<int> result{ 0,0 };

  int* dLHS, * dRHS, * dRes;

  size_t size{ sizeof(int) * lhs.size() };
  cudaMalloc((void**)&dLHS, size);
  cudaMalloc((void**)&dRHS, size);
  cudaMalloc((void**)&dRes, size);

  cudaMemcpy(dLHS, lhs.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(dRHS, rhs.data(), size, cudaMemcpyHostToDevice);

  CudaIntAdder <<< 1, 2 >>> (dLHS, dRHS, dRes);

  cudaMemcpy(result.data(), dRes, size, cudaMemcpyDeviceToHost);

  cudaFree(dLHS);
  cudaFree(dRHS);
  cudaFree(dRes);

  std::cout << result[0] << ", " << result[1] << '\n';
}
