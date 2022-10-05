#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Vendor/cuda11.7/helper_cuda.h"
#include <vector>
#include <array>
#include <fstream>

void IntAdder();
void MatrixMult();
void WarpDivergence();
void Divergence_1D();
void Q3_qn6();
void MatAdd_2D();