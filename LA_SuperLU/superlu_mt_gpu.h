#ifndef DAE_SUPERLU_MT_GPU_H
#define DAE_SUPERLU_MT_GPU_H

#ifdef daeSuperLU_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace daetools
{
namespace solver
{
class superlu_mt_gpu_solver
{
public:
    superlu_mt_gpu_solver();
    virtual ~superlu_mt_gpu_solver();
	
	cudaError_t Initialize(int nnz, int n, int* IA, int* JA);
	cudaError_t Reinitialize(int nnz, int n, int* IA, int* JA);
	cudaError_t SetMatrixValues(double* A);
	cudaError_t FreeMemory(void);
	
	cudaError_t Factorize(int& info);
	cudaError_t Solve(double** b, int& info);
	
	void SetDimensions(size_t nBlocks, size_t nThreadsInBlock);
	
protected:
	void PrintDeviceProperties(void);
	int _ConvertSMVer2Cores(int major, int minor);
	
protected:
	void* m_pData;
};

}
}

#endif


#endif
