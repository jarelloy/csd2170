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
#include <fstream>
#include <iomanip>
/*
* This sample implements Matrix Multiplication
*/

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include "helper.h"

#include <stdint.h>

#define epsilon 1.0e-3


void correctness_test(int nRun,int numARows, int numACols, int numBCols)
{
	for (int i=0; i<nRun; i++) 
  {
    //Matrix A
    float* matA = createData(numARows, numACols);

    //Matrix B
    float* matB = createData(numACols, numBCols);


#define DEBUG_INPUT
#ifdef DEBUG_INPUT
    for (int i{}; i < numARows * numACols; ++i) matA[i] = (float)i;
    for (int i{}; i < numBCols * numACols; ++i) matB[i] = (float)i;
#endif // DEBUG_INPUT


    //CPU code
    float* cpuOut{ new float[numARows * numBCols]{} };
    matrixMultiplyCPU(cpuOut, matA, matB, numARows, numACols, numBCols);

    //GPU code
    float* gpuOut{ new float[numARows * numBCols]{} };
    matrixMultiplyGPU(gpuOut, matA, matB, numARows, numBCols, numACols);


#define DEBUG_OUTPUT
#ifdef  DEBUG_OUTPUT
    //Output CPU + GPU
    std::ofstream cpuOFS{ "cpu.txt" };
    std::ofstream gpuOFS{ "gpu.txt" };

    for (int y{}; y < numBCols; ++y) 
    {
      for (int x{}; x < numARows; ++x)
      {
        int i{ y * numARows + x };

        cpuOFS.width(5);
        cpuOFS << std::fixed << std::setprecision(2) << cpuOut[i] << ' ';
        
        gpuOFS.width(5);
        gpuOFS << std::fixed << std::setprecision(2) << gpuOut[i] << ' ';
      }
      cpuOFS << '\n';
      gpuOFS << '\n';
    }
    gpuOFS.close();
    cpuOFS.close();
#endif // DEBUG_OUTPUT


    //Check to see if match
    for (int i{}; i < numARows * numBCols; ++i)
    {
      assert(std::abs(cpuOut[i] - gpuOut[i]) <= epsilon);
    }

    delete[] cpuOut;
    delete[] gpuOut;
	}
}

void efficiency_test(int nRun, int numARows, int numACols, int numBCols)
{
	for (int i = 0; i < nRun; i++) 
  {
		//call createData() to generate random matrix as inputs
		//matrix multiply cpu results
		//measure the time for matrix multiplication cpu version
		//add to total latency for cpu version
		//matrix multiply gpu results
		//measure the time for matrix multiplication gpu version 
		//add to total latency for gpu version
	}
	//average total latency for cpu version over nRun
	//average total latency for gpu version over nRun
}

int main(int argc, char** argv)
{
  //correctness_test(1, 1, 4, 1); //Dot prod simulation
  //correctness_test(1, 4, 4, 1); //Mat * vec simulation
  correctness_test(1, 4, 4, 4); //Mat * mat simulation

	//correctness_test(1, 101 - rand() % 10, 101 - rand() % 10, 101 - rand() % 10);
	//correctness_test(1, 200 + rand() % 100, 200 + rand() % 100, 200 + rand() % 100);
	//correctness_test(1, 500 + rand() % 500, 500 + rand() % 500, 500 + rand() % 500);
	//correctness_test(1, 2000, 2000, 2000);

	//efficiency_test(10, 100, 100, 100);
	//efficiency_test(10, 500, 500, 500);
	//efficiency_test(10, 1000, 1000, 1000);

	return 0;
}

