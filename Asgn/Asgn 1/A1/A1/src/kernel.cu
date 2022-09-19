/* Start Header *****************************************************************/

/*! \file kernal.cu

     \author Derwin Yan Hong Rui 2000579

     \par h.yn@digipen.edu

     \date 16 Sept 2022

     \brief Copyright (C) 2022 DigiPen Institute of Technology.

  Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited. */

/* End Header *******************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 32
typedef unsigned int uint;

/**
 * @brief    Kernal code that performs heat distribution
 * @param    in - input array
 * @param    out - output array
 * @param    nRowPoints - width and height of sqaure array
 * @return   void
 */
__global__ void heatDistrCalc(float* in, float* out, uint nRowPoints)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < nRowPoints - 1 && row > 0 && col > 0 && col < nRowPoints - 1)
  {
    out[row * nRowPoints + col] =
      in[(row - 1) * nRowPoints + col] +
      in[(row + 1) * nRowPoints + col] +
      in[row * nRowPoints + col + 1] +
      in[row * nRowPoints + col - 1];
    out[row * nRowPoints + col] *= 0.25f;
  }
}

///not required in A1
///Shared memory kernel function for heat distribution calculation
__global__ void heatDistrCalcShm(float* in, float* out, uint nRowPoints)
{

}

/**
 * @brief    Update input array values from output array
 * @param    in - input array
 * @param    out - output array
 * @param    nRowPoints - width and height of square array
 * @return   void
 */
__global__ void heatDistrUpdate(float* in, float* out, uint nRowPoints)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < nRowPoints && col < nRowPoints)
  {
    out[row * nRowPoints + col] = in[row * nRowPoints + col];
  }
}

/**
 * @brief    CPU function that calls GPU kernal for heat distribution
 * @param    d_DataIn - input array
 * @param    d_DataOut - output array
 * @param    nRowPoints - width and height of square array
 * @param    nIter - number of iterations to simulate
 */
extern "C"
void heatDistrGPU(float* d_DataIn, float* d_DataOut, uint nRowPoints, uint nIter)
{
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 DimGrid2((uint)ceil(((float)nRowPoints) / (float)BLOCK_SIZE), (uint)ceil(((float)nRowPoints) / (float)BLOCK_SIZE), 1);

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
