/*
* Copyright 2022 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*
*/
// Utility and system includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "helper.h"

//P and M column-major, N row-major
__global__ 
void matrixMultiply(FLOAT_TYPE* output, const FLOAT_TYPE* input1, const FLOAT_TYPE* input2,
  const int m, const int n, const int k) 
{
  __shared__ FLOAT_TYPE shared[TILE_WIDTH_RATIO_K][TILE_WIDTH_N];
  FLOAT_TYPE partialOutput[TILE_WIDTH_N]{};

  for (int iter{}; iter < (k - 1) / TILE_WIDTH_RATIO_K + 1; ++iter)
  {
    //Load into shared memory
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int finalID = y * n + x;

    int smX = finalID % TILE_WIDTH_N;
    int smY = (finalID / TILE_WIDTH_N) % TILE_WIDTH_RATIO_K;

    //int currentBlock = blockIdx.y * gridDim.x + blockIdx.x;
    int tgtX = smX + blockIdx.y * TILE_WIDTH_N;
    int tgtY = smY + iter * TILE_WIDTH_RATIO_K;

    shared[smY][smX] = input2[tgtY * n + tgtX];
    __syncthreads();

    //Perform multiplication
    for (int perm{}; perm < TILE_WIDTH_N; ++perm)   //row permutation
    {
      //Load into register
      int regY = iter * TILE_WIDTH_RATIO_K + perm * TILE_WIDTH_N;
      int regX = finalID % TILE_WIDTH_M + blockIdx.x * TILE_WIDTH_M;
      __syncthreads();

      //Multiply with shared memory cols
      for (int mulCt{}; mulCt < TILE_WIDTH_N; ++mulCt)  //column permutation
      {
        

        //if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        //{
          printf(
            "Block ID(%d,%d)\n" 
            " Thread ID(%d,%d)\n"
            " Element ID, element value: %d, %f\n "
            " Iter : %d\tPerm : %d\tMulCt : %d\n "
            " SM[0][0] = %f\t SM[0][1] = %f\n "
            " SM[1][0] = %f\t SM[1][1] = %f\n "
            " SM[2][0] = %f\t SM[2][1] = %f\n "
            " SM[3][0] = %f\t SM[3][1] = %f\n "
            " Register = %f\tShared = %f\n"
            " \tPartial output[0]: before, after = %f, %f\n "
            " Register = %f\tShared = %f\n"
            " \tPartial output[1]: before, after = %f, %f\n",
            blockIdx.x, blockIdx.y, 
            threadIdx.x, threadIdx.y, 
            finalID, input2[finalID], 
            iter, perm, mulCt, 
            shared[0][0], shared[0][1], shared[1][0], shared[1][1], shared[2][0], shared[2][1], shared[3][0], shared[3][1], 
            input1[regY * n + regX], shared[regY % TILE_WIDTH_RATIO_K][mulCt], 
            partialOutput[0], partialOutput[mulCt] + input1[regY * n + regX] * shared[regY % TILE_WIDTH_RATIO_K][mulCt],
            input1[(regY + 1) * n + regX], shared[(regY + 1) % TILE_WIDTH_RATIO_K][mulCt], 
            partialOutput[1] , partialOutput[mulCt] = input1[(regY + 1) * n + regX] * shared[(regY + 1) % TILE_WIDTH_RATIO_K][mulCt]);
        //}
          partialOutput[mulCt] += input1[regY * n + regX] * shared[regY % TILE_WIDTH_RATIO_K][mulCt];
          __syncthreads();
          partialOutput[mulCt] += input1[(regY + 1) * n + regX] * shared[(regY + 1) % TILE_WIDTH_RATIO_K][mulCt];
          __syncthreads();
      }

      //Store into output
      int outCol = threadIdx.x + TILE_WIDTH_RATIO_K * blockIdx.x;
      int outRow = regY;

      atomicAdd(&output[outRow * n + outCol], partialOutput[0]);
      atomicAdd(&output[(outRow + 1) * n + outCol], partialOutput[1]);
      __syncthreads();
      //output[outRow * n + outCol] += partialOutput[0];
      //output[(outRow+1) * n + outCol] += partialOutput[1];
    }
  }
}

void matrixMultiplyGPU(FLOAT_TYPE* output, FLOAT_TYPE* input1, FLOAT_TYPE* input2,
  int numARows, int numBColumns, int numAColumns)
{
  //Transpose input1 matrix -- save space, store into output* and kernel will output back to output*

  FLOAT_TYPE* aTranspose{ new FLOAT_TYPE[numARows * numAColumns]{} };
  convertRowColumn(aTranspose, input1, numARows, numAColumns);

  FLOAT_TYPE* dInA{}, *dInB{}, *dOut{};
  size_t matAsize{ sizeof(FLOAT_TYPE) * numARows * numAColumns };
  size_t matBsize{ sizeof(FLOAT_TYPE) * numAColumns * numBColumns };
  size_t matOutSize{ sizeof(FLOAT_TYPE) * numARows * numBColumns };

  checkCudaErrors(cudaMalloc((void**)&dInA, matAsize));
  getLastCudaError("Error cuda malloc!");
  checkCudaErrors(cudaMalloc((void**)&dInB, matBsize));
  getLastCudaError("Error cuda malloc!");
  checkCudaErrors(cudaMalloc((void**)&dOut, matOutSize));
  getLastCudaError("Error cuda malloc!");

  checkCudaErrors(cudaMemcpy(dInA, aTranspose, matAsize, cudaMemcpyHostToDevice));
  getLastCudaError("Error cuda memcpy H2D!");
  checkCudaErrors(cudaMemcpy(dInB, input2, matBsize, cudaMemcpyHostToDevice));
  getLastCudaError("Error cuda memcpy H2D!");


  //@@ Initialize the grid and block dimensions here

  dim3 gridSize((numARows - 1) / TILE_WIDTH_M + 1, (numBColumns - 1) / TILE_WIDTH_N + 1);
  dim3 blockSize(TILE_WIDTH_M, 1);

  matrixMultiply <<< gridSize, blockSize >>> (dOut,dInA,dInB,numARows,numBColumns,numAColumns);

  getLastCudaError("matrixMultiply failed\n");
  cudaDeviceSynchronize();

  checkCudaErrors(cudaMemcpy(output, dOut, matOutSize, cudaMemcpyDeviceToHost));
  getLastCudaError("Error cuda memcpy D2H");

  cudaFree(dInA);
  cudaFree(dInB);
  cudaFree(dOut);
  delete aTranspose;
}
