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
  //Transpose input1 matrix -- save space, store into output* and kernel output back to output*
  convertRowColumn(output, input1, numARows, numBColumns);

  FLOAT_TYPE* dInA{}, dInB{}, dOut{};


  ////@@ Initialize the grid and block dimensions here

  //dim3 dimGrid((numARows - 1) / TILE_WIDTH_M + 1, (numBColumns - 1) / TILE_WIDTH_N + 1);
  //dim3 dimBlock(TILE_WIDTH_M, 1);

  //matrixMultiply <<< dimGrid, dimBlock >>> (C,A,B,numARows,numBColumns,numAColumns);

  //getLastCudaError("matrixMultiply failed\n");
  //cudaDeviceSynchronize();
}
