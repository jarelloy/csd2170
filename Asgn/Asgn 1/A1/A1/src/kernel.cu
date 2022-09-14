/*
* Copyright 2022 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 32
typedef unsigned int uint;

__global__ void heatDistrCalc(float* in, float* out, uint nRowPoints)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  int i = row * nRowPoints + col;
  if (row < nRowPoints-1 && row > 0 && col > 0 && col < nRowPoints-1)
  {
    out[row * nRowPoints + col] =
			in[(row-1) * nRowPoints + col] +
			in[(row+1) * nRowPoints + col] +
			in[row * nRowPoints + col+1] +
			in[row * nRowPoints + col-1];
		out[row * nRowPoints + col] *= 0.25f;
  }
}

///not required in A1
///Shared memory kernel function for heat distribution calculation
__global__ void heatDistrCalcShm(float* in, float* out, uint nRowPoints)
{

}

__global__ void heatDistrUpdate(float* in, float* out, uint nRowPoints)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < nRowPoints && col < nRowPoints)
  {
    out[row * nRowPoints + col] = in[row * nRowPoints + col];
  }
}

extern "C"
void heatDistrGPU(float* d_DataIn, float* d_DataOut, uint nRowPoints, uint nIter)
{
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 DimGrid2(ceil(((float)nRowPoints) / BLOCK_SIZE), ceil(((float)nRowPoints) / BLOCK_SIZE), 1);

	for (uint k = 0; k < nIter; k++) {
		heatDistrCalc << <DimGrid2, DimBlock >> > ((float*)d_DataIn,
			(float*)d_DataOut,
			nRowPoints);
		getLastCudaError("heatDistrCalc failed\n");
		cudaDeviceSynchronize();
		heatDistrUpdate << < DimGrid2, DimBlock >> > ((float*)d_DataOut,
			(float*)d_DataIn,
			nRowPoints);
		getLastCudaError("heatDistrUpdate failed\n");
		cudaDeviceSynchronize();
	}
}
