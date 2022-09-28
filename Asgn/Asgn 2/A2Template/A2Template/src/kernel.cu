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
  // Shared memory for tiling input N array
  __shared__ FLOAT_TYPE B_s[TILE_WIDTH_RATIO_K][TILE_WIDTH_N];

  //your code here
}

void matrixMultiplyGPU(FLOAT_TYPE* output, FLOAT_TYPE* input1, FLOAT_TYPE* input2,
  int numARows, int numBColumns, int numAColumns)
{
  //Transpose input1 matrix -- save space, store into output* and kernel will output back to output*
  convertRowColumn(output, input1, numARows, numBColumns);

  FLOAT_TYPE* dInA{}, *dInB{}, *dOut{};
  size_t matAsize{ sizeof(float) * numARows * numAColumns };
  size_t matBsize{ sizeof(float) * numAColumns * numBColumns };
  size_t matOutSize{ sizeof(float) * numARows * numBColumns };

  checkCudaErrors(cudaMalloc((void**)&dInA, matAsize));
  getLastCudaError("Error cuda malloc!");
  checkCudaErrors(cudaMalloc((void**)&dInB, matBsize));
  getLastCudaError("Error cuda malloc!");
  checkCudaErrors(cudaMalloc((void**)&dOut, matOutSize));
  getLastCudaError("Error cuda malloc!");

  checkCudaErrors(cudaMemcpy(dInA, output, matAsize, cudaMemcpyHostToDevice));
  getLastCudaError("Error cuda memcpy H2D!");
  checkCudaErrors(cudaMemcpy(dInB, input2, matBsize, cudaMemcpyHostToDevice));
  getLastCudaError("Error cuda memcpy H2D!");


  //@@ Initialize the grid and block dimensions here

  dim3 gridSize((numARows - 1) / TILE_WIDTH_M + 1, (numBColumns - 1) / TILE_WIDTH_N + 1);
  dim3 blockSize(TILE_WIDTH_M, 1);

  matrixMultiply <<< gridSize, blockSize >>> (dOut,dInA,dInB,numARows,numBColumns,numAColumns);

  getLastCudaError("matrixMultiply failed\n");
  cudaDeviceSynchronize();

  checkCudaErrors(cudaMemcpy(output, dOut, matOutSize, cudaMemcpyHostToDevice));
  getLastCudaError("Error cuda memcpy D2H");

  cudaFree(dInA);
  cudaFree(dInB);
  cudaFree(dOut);

}
