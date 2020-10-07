#include "cusp_solver.h"

#include <cusp/precond/smoothed_aggregation.h>
#include <cusp/precond/ainv.h>

#include <cusp/krylov/bicgstab.h>
#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/print.h>

#include <iostream>


namespace dae
{
namespace solver
{
template <typename Monitor>
void report_status(Monitor& monitor)
{
    if (monitor.converged())
    {
        std::cout << "  Solver converged to " << monitor.tolerance() << " tolerance";
        std::cout << " after " << monitor.iteration_count() << " iterations";
        std::cout << " (" << monitor.residual_norm() << " final residual)" << std::endl;
    }
    else
    {
        std::cout << "  Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
        std::cout << " to " << monitor.tolerance() << " tolerance ";
        std::cout << " (" << monitor.residual_norm() << " final residual)" << std::endl;
    }
}

class daeCUSPData
{
public:
	cusp::csr_matrix<int, double, cusp::host_memory> A;
	cusp::array1d<double, cusp::host_memory> x;
	cusp::array1d<double, cusp::host_memory> b;
};

cusp_solver::cusp_solver()
{
	m_pData = new daeCUSPData;
}

void cusp_solver::ResizeMatrixAndVectors(int NNZ, int N, int* IA, int* JA)
{
	size_t i;
	daeCUSPData* pData = (daeCUSPData*)m_pData;
	
	pData->A.resize(N, N, NNZ);
	pData->b.resize(N, 0);
	pData->x.resize(N, 0);

	for(i = 0; i < N+1; i++)		
	    pData->A.row_offsets[i] = IA[i];

	for(i = 0; i < NNZ; i++)		
	    pData->A.column_indices[i] = JA[i];
}

void cusp_solver::SetMatrix_A(int NNZ, double* A)
{
	daeCUSPData* pData = (daeCUSPData*)m_pData;
	for(size_t i = 0; i < NNZ; i++)		
		pData->A.values[i] = A[i] * 1e-6; 

	//cusp::print(pData->A);
    cusp::io::write_matrix_market_file(pData->A, "/home/ciroki/A.mtx");
}

void cusp_solver::SetVector_b(int N, double* b)
{
	daeCUSPData* pData = (daeCUSPData*)m_pData;
	for(size_t i = 0; i < N; i++)		
		pData->b[i] = b[i]; 
}

void cusp_solver::GetVector_x(int N, double** x)
{
	daeCUSPData* pData = (daeCUSPData*)m_pData;
	for(size_t i = 0; i < N; i++)		
		(*x)[i] = pData->x[i]; 
}

bool cusp_solver::BuildPreconditioner()
{
	return true;
}

bool cusp_solver::Solve()
{
	daeCUSPData* pData = (daeCUSPData*)m_pData;

	cusp::verbose_monitor<double> monitor(pData->b, 1000, 1e-5);
	
	//cusp::precond::nonsym_bridson_ainv<double, cusp::host_memory> M(pData->A);
	cusp::precond::smoothed_aggregation<int, double, cusp::host_memory> M(pData->A);
	
	cusp::krylov::bicgstab(pData->A, pData->x, pData->b, monitor, M);
	
	report_status(monitor);
	
	std::cout << "\nPreconditioner statistics" << std::endl;
	M.print();
	
	return true;
}


}
}
