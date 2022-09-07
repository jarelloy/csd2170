#include "Tests.h"



__global__
void CudaMatrixMult(float* inMat, float* inVec, float* outVec, int dim)
{
	int tID = threadIdx.x;
	int matID = tID * dim;
	for (int i = 0; i < dim; ++i)
	{
		outVec[tID] += inMat[matID + i] * inVec[tID];
	}
}


void MatrixMult()
{
	std::vector<float> mat{ 1.0f, 2.0f, 3.0f, 4.0f,
													5.0f, 6.0f, 7.0f, 8.0f,
													9.0f, 10.f, 11.f, 12.f,
													13.f, 14.f, 15.f, 16.f };

	std::vector<float> vec{ 2.0f, 4.0f, 6.0f, 8.0f };
	std::vector<float> res{ 0.0f, 0.0f, 0.0f, 0.0f };

	int dim{ (int)vec.size() };
	size_t matSize = sizeof(float) * dim * dim;
	size_t vecSize = sizeof(float) * dim;
	float* dVec, * dMat, * dRes;

	cudaMalloc((void**)&dVec, sizeof(float) * vec.size());
	cudaMalloc((void**)&dVec, vecSize);
	cudaMalloc((void**)&dRes, vecSize);
	cudaMalloc((void**)&dMat, matSize);

	cudaMemcpy(dVec, vec.data(), vecSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dMat, mat.data(), matSize, cudaMemcpyHostToDevice);

	CudaMatrixMult <<< 1, dim >>> (dMat, dVec, dRes, dim);

	cudaMemcpy(res.data(), dRes, vecSize, cudaMemcpyDeviceToHost);

	cudaFree(dVec);
	cudaFree(dMat);
	cudaFree(dRes);

	for (auto x : res) std::cout << x << ", ";
	std::cout << '\n';
}
