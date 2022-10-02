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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
/*
* This sample implements Matrix Multiplication
*/

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include "helper.h"

#include <stdint.h>

#define epsilon 1.0e-3

void printMtx(FLOAT_TYPE* mtx, int row, int col)
{
  std::cout << "\n";
  for (int i{ 0 }; i < row; ++i)
  {
    for (int j{ 0 }; j < col; ++j)
    {
      std::cout << mtx[i * col + j] << "\t";
    }

    std::cout << "\n";
  }
}

void correctness_test(int nRun,
  int numARows,
  int numACols,
  int numBCols)
{
  for (int i = 0; i < nRun; i++) {
    FLOAT_TYPE* h_A = createData(numARows, numACols);
    FLOAT_TYPE* h_B{ createData(numACols, numBCols) };
    FLOAT_TYPE* h_C{ (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numARows * numBCols) }; //CPU mtx
    FLOAT_TYPE* h_C_2{ (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numARows * numBCols) };//GPU mtx
    FLOAT_TYPE* h_A_conv{ (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numARows * numACols) };
    FLOAT_TYPE* h_C_conv{ (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numARows * numBCols) }; //GPU mtx converted 

    matrixMultiplyCPU(h_C, h_A, h_B, numARows, numACols, numBCols);

    FLOAT_TYPE* d_A, * d_B, * d_C;

    convertRowColumn(h_A_conv, h_A, numARows, numACols);
    checkCudaErrors(cudaMalloc((void**)&d_A, sizeof(FLOAT_TYPE) * numARows * numACols));
    checkCudaErrors(cudaMalloc((void**)&d_B, sizeof(FLOAT_TYPE) * numACols * numBCols));
    checkCudaErrors(cudaMalloc((void**)&d_C, sizeof(FLOAT_TYPE) * numARows * numBCols));

    checkCudaErrors(cudaMemcpy(d_A, h_A_conv, sizeof(FLOAT_TYPE) * numARows * numACols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, sizeof(FLOAT_TYPE) * numACols * numBCols, cudaMemcpyHostToDevice));
    matrixMultiplyGPU(d_C, d_A, d_B, numARows, numBCols, numACols);
    checkCudaErrors(cudaMemcpy(h_C_2, d_C, sizeof(FLOAT_TYPE) * numARows * numBCols, cudaMemcpyDeviceToHost));
    convertRowColumn(h_C_conv, h_C_2, numBCols, numARows);

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    //printMtx(h_A, numARows, numACols);
    //printMtx(h_B, numACols, numBCols);

    //printMtx(h_C, numACols, numBCols);
    //printMtx(h_C_conv, numARows, numBCols);

    for (int j{ 0 }; j < numARows * numBCols; ++j)
    {
      if (abs(h_C_conv[j] - h_C[j]) > epsilon)
      {
        std::cout << "no matching\n";
        break;
      }
    }

    free(h_C_conv);
    free(h_A_conv);
    free(h_C_2);
    free(h_C);
    free(h_B);
    free(h_A);
  }
}

void efficiency_test(int nRun,
  int numARows,
  int numACols,
  int numBCols)
{
  StopWatchInterface* hTimer = NULL;
  sdkCreateTimer(&hTimer);

  float cpuAvg{ 0.f };
  float gpuAvg{ 0.f };
  for (int i = 0; i < nRun; i++) {
    FLOAT_TYPE* h_A = createData(numARows, numACols);
    FLOAT_TYPE* h_B{ createData(numACols, numBCols) };
    FLOAT_TYPE* h_C{ (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numARows * numBCols) }; //CPU mtx
    FLOAT_TYPE* h_C_2{ (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numARows * numBCols) };//GPU mtx
    FLOAT_TYPE* h_A_conv{ (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numARows * numACols) };
    FLOAT_TYPE* h_C_conv{ (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numARows * numBCols) }; //GPU mtx converted 

    FLOAT_TYPE* d_A, * d_B, * d_C;

    convertRowColumn(h_A_conv, h_A, numARows, numACols);
    checkCudaErrors(cudaMalloc((void**)&d_A, sizeof(FLOAT_TYPE) * numARows * numACols));
    checkCudaErrors(cudaMalloc((void**)&d_B, sizeof(FLOAT_TYPE) * numACols * numBCols));
    checkCudaErrors(cudaMalloc((void**)&d_C, sizeof(FLOAT_TYPE) * numARows * numBCols));
    checkCudaErrors(cudaMemcpy(d_A, h_A_conv, sizeof(FLOAT_TYPE) * numARows * numACols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, sizeof(FLOAT_TYPE) * numACols * numBCols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A, h_A_conv, sizeof(FLOAT_TYPE) * numARows * numACols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, sizeof(FLOAT_TYPE) * numACols * numBCols, cudaMemcpyHostToDevice));

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    matrixMultiplyCPU(h_C, h_A, h_B, numARows, numACols, numBCols);
    sdkStopTimer(&hTimer);

    float dAvgSecs = 1.0e-3 * (float)sdkGetTimerValue(&hTimer);
    cpuAvg += dAvgSecs;

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    matrixMultiplyGPU(d_C, d_A, d_B, numARows, numBCols, numACols);
    sdkStopTimer(&hTimer);

    dAvgSecs = 1.0e-3 * (float)sdkGetTimerValue(&hTimer);
    gpuAvg += dAvgSecs;
    checkCudaErrors(cudaMemcpy(h_C_2, d_C, sizeof(FLOAT_TYPE) * numARows * numBCols, cudaMemcpyDeviceToHost));
    convertRowColumn(h_C_conv, h_C_2, numBCols, numARows);

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    for (int j{ 0 }; j < numARows * numBCols; ++j)
    {
      if (abs(h_C_conv[j] - h_C[j]) > epsilon)
      {
        std::cout << "no matching\n";
        break;
      }
    }

    free(h_C_conv);
    free(h_A_conv);
    free(h_C_2);
    free(h_C);
    free(h_B);
    free(h_A);

    //call createData() to generate random matrix as inputs
    //matrix multiply cpu results
    //measure the time for matrix multiplication cpu version
    //add to total latency for cpu version
    //matrix multiply gpu results
    //measure the time for matrix multiplication gpu version 
    //add to total latency for gpu version
  }

  printf("CPU average time taken: %fs\n", cpuAvg / nRun);
  printf("GPU average time taken: %fs\n", gpuAvg / nRun);
  sdkDeleteTimer(&hTimer);
  //average total latency for cpu version over nRun
  //average total latency for gpu version over nRun
}

int main(int argc, char** argv)
{
#if defined(DEBUG) || defined(_DEBUG)
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
  //int numARows = 191;
  //int numACols = 19;
  //int numBCols = 241;
  //int numBRows = numACols;

  correctness_test(200, 2, 2, 2);
  correctness_test(10, 101 - rand() % 10, 101 - rand() % 10, 101 - rand() % 10);
  correctness_test(10, 200 + rand() % 100, 200 + rand() % 100, 200 + rand() % 100);
  correctness_test(10, 500 + rand() % 500, 500 + rand() % 500, 500 + rand() % 500);
  //correctness_test(1, 2000, 2000, 2000);
  
  //efficiency_test(10, 100, 100, 100);
  //efficiency_test(10, 500, 500, 500);
  //efficiency_test(10, 1000, 1000, 1000);

  return 0;
}

