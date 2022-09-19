/* Start Header *****************************************************************/

/*! \file cpu.cpp

     \author Derwin Yan Hong Rui 2000579

     \par h.yn@digipen.edu

     \date 16 Sept 2022

     \brief Copyright (C) 2022 DigiPen Institute of Technology.

  Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited. */

/* End Header *******************************************************************/
#include "heat.h"
#include <memory>

/**
 * @brief    Initialize default heat values in array
 * @param    pointIn - input array
 * @param    pointOut - output array
 * @param    nRowPoints - width and height of array
 */
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

/**
 * @brief    Simulate heat distribution on CPU side
 * @param    pointIn - input array
 * @param    pointOut - output array
 * @param    nRowPoints - width and height of array
 * @param    nIter - number of times to simulate
 */
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
