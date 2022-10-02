/* Start Header *****************************************************************/

/*! \file kernal.cu

     \author Derwin Yan Hong Rui 2000579

     \par h.yn@digipen.edu

     \date 2 Oct 2022

     \brief Copyright (C) 2022 DigiPen Institute of Technology.

  Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited. */

  /* End Header *******************************************************************/

#include <helper_cuda.h>
#include "helper.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__
void matrixMultiply(FLOAT_TYPE* output, const FLOAT_TYPE* input1, const FLOAT_TYPE* input2,
  const int m, const int n, const int k)
{
  __shared__ FLOAT_TYPE shared[TILE_WIDTH_RATIO_K][TILE_WIDTH_N];
  FLOAT_TYPE partialOutput[TILE_WIDTH_N]{};

  for (int iter{}; iter < (k - 1) / TILE_WIDTH_RATIO_K + 1; ++iter)
  {
    //Load into shared memory
    int smX = threadIdx.x % TILE_WIDTH_N;
    int smY = (threadIdx.x / TILE_WIDTH_N);

    int tgtX = smX + blockIdx.y * TILE_WIDTH_N;
    int tgtY = smY + iter * TILE_WIDTH_RATIO_K;

    shared[smY][smX] = (tgtX < n&& tgtY < k) ? input2[tgtY * n + tgtX] : (FLOAT_TYPE)0.0f;
    __syncthreads();


    for (int y{}; y < TILE_WIDTH_RATIO_K; ++y)
    {
      int regY = iter * TILE_WIDTH_RATIO_K + y;
      int regX = blockIdx.x * TILE_WIDTH_M + threadIdx.x;

      for (int x{}; x < TILE_WIDTH_N; ++x)
      {
        partialOutput[x] += input1[regY * m + regX] * shared[y][x];
      }
    }
    __syncthreads();
  }

  //store into output
  for (int i{}; i < TILE_WIDTH_N; ++i)
  {
    int outY = blockIdx.y * TILE_WIDTH_N + i;
    int outX = threadIdx.x + TILE_WIDTH_M * blockIdx.x;

    if (outX < m && outY < n)
    {
      output[outY * m + outX] = partialOutput[i];
    }
  }
}

void matrixMultiplyGPU(FLOAT_TYPE* C,
  FLOAT_TYPE* A,
  FLOAT_TYPE* B,
  int numARows,
  int numBColumns,
  int numAColumns)
{
  //@@ Initialize the grid and block dimensions here

  dim3 dimGrid((numARows - 1) / TILE_WIDTH_M + 1, (numBColumns - 1) / TILE_WIDTH_N + 1);
  dim3 dimBlock(TILE_WIDTH_M, 1);

  matrixMultiply << <dimGrid, dimBlock >> > (C,
    A,
    B,
    numARows,
    numBColumns,
    numAColumns);

  getLastCudaError("matrixMultiply failed\n");
  cudaDeviceSynchronize();
}
