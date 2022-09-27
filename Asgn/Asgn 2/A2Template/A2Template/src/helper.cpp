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
#include <stdlib.h>
#include "helper.h"
FLOAT_TYPE* createData(int nRows, int nCols)
{
	FLOAT_TYPE* data = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * nCols * nRows);
	int i;
	for (i = 0; i < nCols * nRows; i++) 
  {
		data[i] = ((FLOAT_TYPE)(rand() % 10) - 5) / 5.0f;
	}
	return data;
}

void matrixMultiplyCPU(FLOAT_TYPE* output, FLOAT_TYPE* input0, FLOAT_TYPE* input1,
  int numARows, int numAColumns, int numBColumns)
{
  for (int y{}; y < numARows; ++y)
  {
    for (int x{}; x < numBColumns; ++x)
    {
      float sum{};
      for (int iter{}; iter < numAColumns; ++iter)
      {
        float in1{ input0[y * numAColumns + x + iter] };
        float in2{ input1[(y + iter) * numBColumns + x] };
        sum += in1 * in2;
      }
      output[y * numBColumns + x] = sum;
    }
  }
}

void convertRowColumn(FLOAT_TYPE* dst, FLOAT_TYPE* src, int numRows, int numCols)
{
	//your code here
}