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

void correctness_test(int nRun,
	int numARows,
	int numACols,
	int numBCols)
{
	for (int i=0; i<nRun; i++) {
		//call createData() to generate random matrix as inputs
		//matrix multiply cpu results
		//matrix multiply gpu results
		//cross-checking cpu and gpu results
	}
}

void efficiency_test(int nRun,
	int numARows,
	int numACols,
	int numBCols)
{
	for (int i = 0; i < nRun; i++) {
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
	int numARows = 191;
	int numACols = 19;
	int numBCols = 241;
	int numBRows = numACols;

	correctness_test(1, 101 - rand() % 10, 101 - rand() % 10, 101 - rand() % 10);
	correctness_test(1, 200 + rand() % 100, 200 + rand() % 100, 200 + rand() % 100);
	correctness_test(1, 500 + rand() % 500, 500 + rand() % 500, 500 + rand() % 500);
	correctness_test(1, 2000, 2000, 2000);

	efficiency_test(10, 100, 100, 100);
	efficiency_test(10, 500, 500, 500);
	efficiency_test(10, 1000, 1000, 1000);

	return 0;
}

