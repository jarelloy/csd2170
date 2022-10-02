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
  for (i = 0; i < nCols * nRows; i++) {
    data[i] = ((FLOAT_TYPE)(rand() % 10) - 5) / 5.0f;
  }
  return data;
}


void matrixMultiplyCPU(FLOAT_TYPE* output, FLOAT_TYPE* input0, FLOAT_TYPE* input1,
  int numARows, int numAColumns, int numBColumns)
{ //  m             k                n

  for (int i{ 0 }; i < numARows; ++i)
  {
    for (int j{ 0 }; j < numBColumns; ++j)
    {
      FLOAT_TYPE tmpRes{ static_cast<FLOAT_TYPE>(0) };
      for (int k{ 0 }; k < numAColumns; ++k)
      {
        tmpRes += input0[numAColumns * i + k] * input1[numBColumns * k + j];
      }
      output[numBColumns * i + j] = tmpRes;
    }
  }

  // assumes input 1 is converted to column major since no need to upload
  //for (int i{ 0 }; i < numBColumns; ++i)
  //{
  //  for (int j{ 0 }; j < numAColumns; ++j)
  //  {
  //    FLOAT_TYPE tmpRes{ static_cast<FLOAT_TYPE>(0) };
  //    for (int k{ 0 }; k < numARows; ++k)
  //    {
  //      tmpRes += input0[numARows * i + k] * input1[numAColumns * j + k];
  //    }
  //    output[numAColumns * i + j] = tmpRes;
  //  }
  //}
}

void convertRowColumn(FLOAT_TYPE* dst, FLOAT_TYPE* src, int numRows, int numCols)
{ // is actually transpose, so to get it back flip rows and cols.
  for (int i{ 0 }, t{ numRows * numCols }; i < t; ++i)
  { // release single div instruction stores / in EAX and % in EDX registers
    dst[i] = src[numCols * (i % numRows) + (i / numRows)];
  }
}