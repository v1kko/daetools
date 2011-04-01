#ifdef daeSuperLU_CUDA

#include "superlu_mt_gpu.h"
#include "superlumt-gpu.cu"

#define CUDACheck(ce) if(ce != cudaSuccess) {return ce;}

namespace dae
{
namespace solver
{
class daeSuperLU_MT_GPU_Data
{
public:
	daeSuperLU_MT_GPU_Data()
	{
		m_nBlocks = 1;
		m_nThreadsInBlock = 1;
		m_matA = NULL;
		m_matB = NULL;
		m_matL = NULL;
		m_matU = NULL;
		m_matAC = NULL;
	
		m_Options = NULL;
		m_Stats = NULL;
		m_perm_c = NULL;
		m_perm_r = NULL;
		m_A = NULL;
		m_IA = NULL;
		m_JA = NULL;
		m_vecB = NULL;
		m_lwork = 0;
		m_work = NULL;
		m_pGlu = NULL;
		m_pxgstrf_shared = NULL;
		cu_info = NULL;
		m_threadarg = NULL;
	}
	
	int N;
	int NNZ;

	int						m_nBlocks;
	int						m_nThreadsInBlock;
	SuperMatrix*			m_matA;
	SuperMatrix*			m_matB;
	SuperMatrix*			m_matL;	
	SuperMatrix*			m_matU;	
	SuperMatrix*			m_matAC;
	
	GlobalLU_t*             m_pGlu;
	pxgstrf_shared_t*       m_pxgstrf_shared;
	int*                    cu_info;
	
	superlumt_options_t*	m_Options;
	Gstat_t*				m_Stats;
	
	double*					m_vecB;
    int*					m_perm_c;
    int*					m_perm_r;
	double*					m_A;
	int*					m_IA;
	int*					m_JA;
	int						m_lwork;
	void*					m_work;
	
    pdgstrf_threadarg_t*	m_threadarg;
	
	bool					m_bFactorizationDone;
};

superlu_mt_gpu_solver::superlu_mt_gpu_solver(void)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0)
	{
		printf("No CUDA devices found. Consider buying one :-)\n");
		return;
	}
	
	PrintDeviceProperties();
	
	size_t LimitMallocHeapSize;
	cudaThreadGetLimit(&LimitMallocHeapSize, cudaLimitMallocHeapSize);
	printf("LimitMallocHeapSize: %d MB\n", int(LimitMallocHeapSize / (1024. * 1024.)));
	
	cudaThreadSetLimit(cudaLimitMallocHeapSize, 100 * 1024 * 1024);
	cudaThreadGetLimit(&LimitMallocHeapSize, cudaLimitMallocHeapSize);
	printf("LimitMallocHeapSize: %d MB\n", int(LimitMallocHeapSize / (1024. * 1024.)));
	
	m_pData = new daeSuperLU_MT_GPU_Data;
} 

superlu_mt_gpu_solver::~superlu_mt_gpu_solver(void)
{

} 

void superlu_mt_gpu_solver::SetDimensions(size_t nBlocks, size_t nThreadsInBlock)
{
	daeSuperLU_MT_GPU_Data& data = *(daeSuperLU_MT_GPU_Data*)m_pData;
	data.m_nBlocks         = (int)nBlocks;
	data.m_nThreadsInBlock = (int)nThreadsInBlock;
}

cudaError_t superlu_mt_gpu_solver::Initialize(int nnz, int n, int* IA, int* JA)
{
	cudaError_t ce;	
	
	daeSuperLU_MT_GPU_Data& data = *(daeSuperLU_MT_GPU_Data*)m_pData;

	int N    = n;
	int NNZ  = nnz;
	data.N   = n;
	data.NNZ = nnz;
	
//	Allocate storage on the device
    ce = cudaMalloc((void**)&data.cu_info,  sizeof(int));
    CUDACheck(ce);

    ce = cudaMalloc((void**)&data.m_pGlu,  sizeof(GlobalLU_t));
    CUDACheck(ce);
    ce = cudaMalloc((void**)&data.m_pxgstrf_shared, sizeof(pxgstrf_shared_t));
    CUDACheck(ce);

	ce = cudaMalloc((void**)&data.m_matA,  sizeof(SuperMatrix));
    CUDACheck(ce);
	ce = cudaMalloc((void**)&data.m_matB,  sizeof(SuperMatrix));
    CUDACheck(ce);
	ce = cudaMalloc((void**)&data.m_matL,  sizeof(SuperMatrix));
    CUDACheck(ce);
	ce = cudaMalloc((void**)&data.m_matU,  sizeof(SuperMatrix));
    CUDACheck(ce);
	ce = cudaMalloc((void**)&data.m_matAC, sizeof(SuperMatrix));
    CUDACheck(ce);

	ce = cudaMalloc((void**)&data.m_A,   NNZ  * sizeof(double));
    CUDACheck(ce);
	ce = cudaMalloc((void**)&data.m_IA, (N+1) * sizeof(int));
    CUDACheck(ce);
	ce = cudaMalloc((void**)&data.m_JA,	NNZ  * sizeof(int));
    CUDACheck(ce);

	ce = cudaMalloc((void**)&data.m_Options, sizeof(superlumt_options_t));
    CUDACheck(ce);
	ce = cudaMalloc((void**)&data.m_Stats,   sizeof(Gstat_t));
    CUDACheck(ce);

	ce = cudaMalloc((void**)&data.m_perm_r,   N * sizeof(int));
    CUDACheck(ce);
	ce = cudaMalloc((void**)&data.m_perm_c,   N * sizeof(int));
    CUDACheck(ce);
	ce = cudaMalloc((void**)&data.m_vecB,     N * sizeof(double));
    CUDACheck(ce);

    int nThreads = data.m_nBlocks * data.m_nThreadsInBlock / 32;

    ce = cudaMalloc((void**)&data.m_threadarg, nThreads * sizeof(pdgstrf_threadarg_t));
    CUDACheck(ce);

	ce = cudaMemcpy(data.m_IA, IA, (N+1) * sizeof(int),    cudaMemcpyHostToDevice);
	CUDACheck(ce);
	ce = cudaMemcpy(data.m_JA, JA,  NNZ  * sizeof(int),    cudaMemcpyHostToDevice);
	CUDACheck(ce);

	gpu_dCreate_CompCol_Matrix<<<1, 1>>>(data.m_matA, N, N, NNZ, data.m_A, data.m_JA, data.m_IA, SLU_NC, SLU_D, SLU_GE);
	ce = cudaThreadSynchronize();
	CUDACheck(ce);
	
	gpu_dCreate_Dense_Matrix<<<1, 1>>>(data.m_matB, N, 1, data.m_vecB, N, SLU_DN, SLU_D, SLU_GE);
	ce = cudaThreadSynchronize();
	CUDACheck(ce);
	
    gpu_set_default_options<<<1, 1>>>(data.m_Options);
    ce = cudaThreadSynchronize();
    CUDACheck(ce);

    gpu_set_perm_r<<<1, 1>>>(data.m_Options, data.m_perm_r);
    ce = cudaThreadSynchronize();
    CUDACheck(ce);

    gpu_set_perm_c<<<1, 1>>>(data.m_Options, data.m_perm_c);
    ce = cudaThreadSynchronize();
    CUDACheck(ce);

    /* Achtung Achtung!! Above matrix is CRS therefore use TRANS!! */
    gpu_set_trans<<<1, 1>>>(data.m_Options, TRANS);
    ce = cudaThreadSynchronize();
    CUDACheck(ce);

    gpu_set_nprocs<<<1, 1>>>(data.m_Options, nThreads);
    ce = cudaThreadSynchronize();
    CUDACheck(ce);

	data.m_bFactorizationDone = false;

	return cudaSuccess;
} 

cudaError_t superlu_mt_gpu_solver::FreeMemory(void)
{
	cudaError_t ce;	
	
	daeSuperLU_MT_GPU_Data& data = *(daeSuperLU_MT_GPU_Data*)m_pData;
	
	ce = cudaFree(data.cu_info);
	CUDACheck(ce);
	ce = cudaFree(data.m_pGlu);
	CUDACheck(ce);
	ce = cudaFree(data.m_pxgstrf_shared);
	CUDACheck(ce);
	
	ce = cudaFree(data.m_perm_r);
	CUDACheck(ce);
	ce = cudaFree(data.m_perm_c);
	CUDACheck(ce);
	ce = cudaFree(data.m_vecB);
	CUDACheck(ce);
	
	ce = cudaFree(data.m_A);
	CUDACheck(ce);
	ce = cudaFree(data.m_IA);
	CUDACheck(ce);
	ce = cudaFree(data.m_JA);
	CUDACheck(ce);
	
	ce = cudaFree(data.m_matA);
	CUDACheck(ce);
	ce = cudaFree(data.m_matB);
	CUDACheck(ce);
	ce = cudaFree(data.m_matL);
	CUDACheck(ce);
	ce = cudaFree(data.m_matU);
	CUDACheck(ce);
	ce = cudaFree(data.m_matAC);
	CUDACheck(ce);
	
	ce = cudaFree(data.m_Stats);
	CUDACheck(ce);
	ce = cudaFree(data.m_Options);
	CUDACheck(ce);
	
	if(data.m_bFactorizationDone)
	{
		gpu_pxgstrf_finalize<<<1, 1>>>(data.m_Options, data.m_matAC);
		ce = cudaThreadSynchronize();
		CUDACheck(ce);
		
		gpu_Destroy_SuperNode_SCP<<<1, 1>>>(data.m_matL);
		ce = cudaThreadSynchronize();
		CUDACheck(ce);
		
		gpu_Destroy_CompCol_NCP<<<1, 1>>>(data.m_matU);
		ce = cudaThreadSynchronize();
		CUDACheck(ce);
	}
	
    gpu_Destroy_SuperMatrix_Store<<<1, 1>>>(data.m_matA);
    ce = cudaThreadSynchronize();
    CUDACheck(ce);

    gpu_Destroy_SuperMatrix_Store<<<1, 1>>>(data.m_matB);
    ce = cudaThreadSynchronize();
    CUDACheck(ce);
	
    gpu_StatFree<<<1, 1>>>(data.m_Stats);
    ce = cudaThreadSynchronize();
    CUDACheck(ce);
	
	return cudaSuccess;
}

cudaError_t superlu_mt_gpu_solver::Reinitialize(int nnz, int n, int* IA, int* JA)
{
	cudaError_t ce;	
	
	daeSuperLU_MT_GPU_Data& data = *(daeSuperLU_MT_GPU_Data*)m_pData;
	
	int N    = n;
	int NNZ  = nnz;
	
// If NNZ/N changed then recreate data structures and arrays
	if(data.N != n || data.NNZ != nnz)
	{
	// First free what has to be freed
		ce = cudaFree(data.m_A);
		CUDACheck(ce);
		ce = cudaFree(data.m_IA);
		CUDACheck(ce);
		ce = cudaFree(data.m_JA);
		CUDACheck(ce);
	
		if(data.m_bFactorizationDone)
		{
			gpu_pxgstrf_finalize<<<1, 1>>>(data.m_Options, data.m_matAC);
			ce = cudaThreadSynchronize();
			CUDACheck(ce);
			
			gpu_Destroy_SuperNode_SCP<<<1, 1>>>(data.m_matL);
			ce = cudaThreadSynchronize();
			CUDACheck(ce);
			
			gpu_Destroy_CompCol_NCP<<<1, 1>>>(data.m_matU);
			ce = cudaThreadSynchronize();
			CUDACheck(ce);
		}
		
		gpu_Destroy_SuperMatrix_Store<<<1, 1>>>(data.m_matA);
		ce = cudaThreadSynchronize();
		CUDACheck(ce);
		
	// Recreate matrix A
		data.N   = n;
		data.NNZ = nnz;

		ce = cudaMalloc((void**)&data.m_A,   NNZ  * sizeof(double));
		CUDACheck(ce);
		ce = cudaMalloc((void**)&data.m_IA, (N+1) * sizeof(int));
		CUDACheck(ce);
		ce = cudaMalloc((void**)&data.m_JA,	NNZ  * sizeof(int));
		CUDACheck(ce);
	
		gpu_dCreate_CompCol_Matrix<<<1, 1>>>(data.m_matA, N, N, NNZ, data.m_A, data.m_JA, data.m_IA, SLU_NC, SLU_D, SLU_GE);
		ce = cudaThreadSynchronize();
		CUDACheck(ce);
	}

// Now copy IA, JA into matrix A	
	ce = cudaMemcpy(data.m_IA, IA, (N+1) * sizeof(int),    cudaMemcpyHostToDevice);
	CUDACheck(ce);
	ce = cudaMemcpy(data.m_JA, JA,  NNZ  * sizeof(int),    cudaMemcpyHostToDevice);
	CUDACheck(ce);

	data.m_bFactorizationDone = false;

	return cudaSuccess;
} 

cudaError_t superlu_mt_gpu_solver::SetMatrixValues(double* A)
{
	cudaError_t ce;	
	
	daeSuperLU_MT_GPU_Data& data = *(daeSuperLU_MT_GPU_Data*)m_pData;
	
	ce = cudaMemcpy(data.m_A, A,  data.NNZ  * sizeof(double), cudaMemcpyHostToDevice);
	CUDACheck(ce);

	return cudaSuccess;
} 

cudaError_t superlu_mt_gpu_solver::Factorize(int& info)
{
	cudaError_t ce;	
	
	daeSuperLU_MT_GPU_Data& data = *(daeSuperLU_MT_GPU_Data*)m_pData;
	
	yes_no_t refact;
	
	info = -1000;

	if(data.m_bFactorizationDone)
	{
	// During the subsequent calls re-use what is possible (Pc, U, L, etree, colcnt_h, part_super_h)
		refact = YES;
		
	// Matrix AC has to be destroyed to avoid memory leaks in sp_colorder()
		gpu_Destroy_CompCol_Permuted<<<1, 1>>>(data.m_matAC);
		ce = cudaThreadSynchronize();
		CUDACheck(ce);
	}
	else
	{
	// At the first call do the fresh factorization (Pr, Pc, etree, L, U and AC will be computed)
		refact = NO;
		
		gpu_StatAlloc<<<1, 1>>>(data.N, data.m_Options, data.m_Stats);
		ce = cudaThreadSynchronize();
		CUDACheck(ce);
	
		gpu_get_perm_c<<<1, 1>>>(COLAMD, data.m_matA, data.m_perm_c);
		ce = cudaThreadSynchronize();
		CUDACheck(ce);
	}

	gpu_StatInit<<<1, 1>>>(data.N, data.m_Options, data.m_Stats);
	ce = cudaThreadSynchronize();
	CUDACheck(ce);
	
// This will allocate memory for AC. 
// If I call it repeatedly then I will have memory leaks! (double check it)
// If that is true before each call to pdgstrf_init I have to call: pxgstrf_finalize(&m_Options, &m_matAC)
// which destroys AC, options->etree, options->colcnt_h and options->part_super_h
// but in that case I still need options->etree, options->colcnt_h and options->part_super_h
// Perhaps the best idea is to call Destroy_CompCol_Permuted(&AC) before each pdgstrf_init() call
	gpu_pdgstrf_init<<<1, 1>>>(data.m_matA, data.m_matAC, refact, data.m_Options, data.m_Stats);
	ce = cudaThreadSynchronize();
	CUDACheck(ce);
	
	ce = gpu_pdgstrf(data.m_nBlocks, data.m_nThreadsInBlock, data.m_pGlu, data.m_threadarg, data.m_pxgstrf_shared, data.m_Options, data.m_matAC, data.m_perm_r, data.m_matL, data.m_matU, data.m_Stats, &info);
	
	if(info == 0 && ce == cudaSuccess)
		data.m_bFactorizationDone = true;
		
//	gpu_Print_CompCol_Matrix<<<1, 1>>>(data.m_matU);
//	ce = cudaThreadSynchronize();
//	CUDACheck(ce);
	
	return ce;
} 

cudaError_t superlu_mt_gpu_solver::Solve(double** b, int& info)
{
	cudaError_t ce;	
	
	daeSuperLU_MT_GPU_Data& data = *(daeSuperLU_MT_GPU_Data*)m_pData;
	
	info = -1000;
	if(!data.m_bFactorizationDone)
		return cudaSuccess;
		
    /* Copy the RHS to the device */
    ce = cudaMemcpy(data.m_vecB, *b,  data.N * sizeof(double), cudaMemcpyHostToDevice);
    CUDACheck(ce);

    /* Call dgstrs() on the device */
    gpu_dgstrs<<<1, 1>>>(data.m_Options, data.m_matL, data.m_matU, data.m_matB, data.m_Stats, data.cu_info);
    ce = cudaThreadSynchronize();
    CUDACheck(ce);

    /* Get info from the device */
    ce = cudaMemcpy(&info, data.cu_info,  sizeof(int), cudaMemcpyDeviceToHost);
    CUDACheck(ce);
    
	if(info != 0)
        return ce;

    /* Get the solution from the device */
    ce = cudaMemcpy(*b, data.m_vecB,  data.N * sizeof(double), cudaMemcpyDeviceToHost);
    CUDACheck(ce);
	
	return ce;
}

int superlu_mt_gpu_solver::_ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    { { 0x10,  8 },
      { 0x11,  8 },
      { 0x12,  8 },
      { 0x13,  8 },
      { 0x20, 32 },
      { 0x21, 48 },
      {   -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
    return -1;
}

void superlu_mt_gpu_solver::PrintDeviceProperties(void)
{
    int deviceCount;
    cudaDeviceProp deviceProp;
    size_t LimitPrintfFifoSize, LimitMallocHeapSize;

    cudaGetDeviceProperties(&deviceProp, 0);

    int dev;
    int driverVersion = 0, runtimeVersion = 0;
#if CUDART_VERSION >= 2020
    // Console log
    cudaDriverGetVersion(&driverVersion);
    printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
#endif
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

    char msg[256];
    sprintf(msg, "  Total amount of global memory:                 %d MB\n", int(deviceProp.totalGlobalMem / 1048576.));
    printf(msg);
#if CUDART_VERSION >= 2000
    printf("  Multiprocessors x Cores/MP = Cores:            %d (MP) x %d (Cores/MP) = %d\n",
        deviceProp.multiProcessorCount,
        _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
        _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
#endif
    printf("  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
            deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
            deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
    printf("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
    printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
    printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 2020
    printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 3000
    printf("  Concurrent kernel execution:                   %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 3010
    printf("  Device has ECC support enabled:                %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 3020
    printf("  Device is using TCC driver mode:               %s\n", deviceProp.tccDriver ? "Yes" : "No");
#endif

    cudaThreadGetLimit(&LimitPrintfFifoSize, cudaLimitPrintfFifoSize);
    cudaThreadGetLimit(&LimitMallocHeapSize, cudaLimitMallocHeapSize);
    printf("  LimitPrintfFifoSize:                           %d MB\n", int(LimitPrintfFifoSize / 1048576.));
    printf("  LimitMallocHeapSize:                           %d MB\n", int(LimitMallocHeapSize / 1048576.));
}

}
}

#endif
