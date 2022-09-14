/*
* Copyright 2022 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms 
* is strictly prohibited.
*/
#include "heat.h"
#include <memory>

extern "C" void initPoints(
  float *pointIn,
  float *pointOut,
  uint nRowPoints
)
{
  for (uint y = 0; y < nRowPoints; ++y)
  {
    for (uint x = 0; x < nRowPoints; ++x)
    {
      if (y == 0 && x >= 10 && x <= 30)       
        pointIn[y * nRowPoints + x] = pointOut[y * nRowPoints + x] = 65.56f;
      else if (x == 0 || x == nRowPoints-1 || y == 0 || y == nRowPoints-1)          
        pointIn[y * nRowPoints + x] = pointOut[y * nRowPoints + x] = 26.67f;
      else                                    
        pointIn[y * nRowPoints + x] = pointOut[y * nRowPoints + x] = 0.0f;
    }
  }
}

extern "C" void heatDistrCPU(
  float *pointIn,
  float *pointOut,
  uint nRowPoints,
  uint nIter
)
{
  for (uint n = 0; n < nIter; ++n)
  {
    for (uint y = 1; y < nRowPoints-1; ++ y)
    {
      for (uint x = 1; x < nRowPoints-1; ++x)
      {
        float total = 0.0f;
        if (x != 0) { total += pointIn[y * nRowPoints + x - 1]; }
        if (x != nRowPoints - 1) { total += pointIn[y * nRowPoints + x + 1]; }
        if (y != 0) { total += pointIn[(y - 1) * nRowPoints + x]; }
        if (y != nRowPoints-1) { total += pointIn[(y + 1) * nRowPoints + x]; }

        pointOut[y * nRowPoints + x] = total * 0.25f;
      }
    }

    std::memcpy(pointIn, pointOut, (size_t)nRowPoints * (size_t)nRowPoints * (size_t)sizeof(float));
  }
}
