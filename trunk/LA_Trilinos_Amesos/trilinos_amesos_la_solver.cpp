#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include <idas/idas_impl.h>
#include "trilinos_amesos_la_solver.h"

#ifdef DAE_SINGLE_PRECISION
#error Trilinos Amesos LA Solver does not support single precision floating point values
#endif

namespace dae
{
namespace solver
{
int init_la(IDAMem ida_mem);
int setup_la(IDAMem ida_mem,
			  N_Vector	vectorVariables, 
			  N_Vector	vectorTimeDerivatives, 
			  N_Vector	vectorResiduals,
			  N_Vector	vectorTemp1, 
			  N_Vector	vectorTemp2, 
			  N_Vector	vectorTemp3);
int solve_la(IDAMem ida_mem,
			  N_Vector	b, 
			  N_Vector	weight, 
			  N_Vector	vectorVariables,
			  N_Vector	vectorTimeDerivatives, 
			  N_Vector	vectorResiduals);
int free_la(IDAMem ida_mem);

daeIDALASolver_t* daeCreateTrilinosAmesosSolver(const std::string& strSolverName)
{
	return new daeTrilinosAmesosSolver(strSolverName);
}

std::vector<string> daeTrilinosAmesosSupportedSolvers(void)
{
	std::vector<string> strarrSolvers;

	Amesos Factory;
	if(Factory.Query("Amesos_Klu")) 
		strarrSolvers.push_back("Amesos_Klu");
	if(Factory.Query("Amesos_Lapack")) 
		strarrSolvers.push_back("Amesos_Lapack");
	if(Factory.Query("Amesos_Scalapack")) 
		strarrSolvers.push_back("Amesos_Scalapack");
	if(Factory.Query("Amesos_Umfpack")) 
		strarrSolvers.push_back("Amesos_Umfpack");
	if(Factory.Query("Amesos_Pardiso")) 
		strarrSolvers.push_back("Amesos_Pardiso");
	if(Factory.Query("Amesos_Taucs")) 
		strarrSolvers.push_back("Amesos_Taucs");
	if(Factory.Query("Amesos_Superlu")) 
		strarrSolvers.push_back("Amesos_Superlu");
	if(Factory.Query("Amesos_Superludist")) 
		strarrSolvers.push_back("Amesos_Superludist");
	if(Factory.Query("Amesos_Dscpack")) 
		strarrSolvers.push_back("Amesos_Dscpack");
	if(Factory.Query("Amesos_Mumps")) 
		strarrSolvers.push_back("Amesos_Mumps");
	
	return strarrSolvers;
}

daeTrilinosAmesosSolver::daeTrilinosAmesosSolver(const std::string& strSolverName)
#ifdef HAVE_MPI
	: m_Comm(MPI_COMM_WORLD)
#endif
{
	m_pBlock				= NULL;
	m_strSolverName			= strSolverName;
	m_nNoEquations			= 0;
	m_nJacobianEvaluations	= 0;
}

daeTrilinosAmesosSolver::~daeTrilinosAmesosSolver(void)
{
	FreeMemory();
}

int daeTrilinosAmesosSolver::Create(void* ida, size_t n, daeDAESolver_t* pDAESolver)
{
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	
	m_pBlock = pDAESolver->GetBlock();
	if(!m_pBlock)
		return IDA_MEM_NULL;
	
	m_nNoEquations = n;

	AllocateMemory();
	if(!CheckData())
		return IDA_MEM_NULL;

// Initialize sparse matrix
	m_matJacobian.ResetCounters();
	m_pBlock->FillSparseMatrix(&m_matJacobian);
	m_matJacobian.Sort();
	//m_matJacobian.Print();

	ida_mem->ida_linit	= init_la;
	ida_mem->ida_lsetup = setup_la;
	ida_mem->ida_lsolve = solve_la;
	ida_mem->ida_lperf	= NULL;
	ida_mem->ida_lfree	= free_la;

	ida_mem->ida_lmem         = this;
	ida_mem->ida_setupNonNull = TRUE;

	return IDA_SUCCESS;
}

int daeTrilinosAmesosSolver::Reinitialize(void* ida)
{
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	
	if(!m_pBlock)
		return IDA_MEM_NULL;
	
// Re-initialize sparse matrix
// ACHTUNG, ACHTUNG!!! It doesn't work, should be checked
	m_matJacobian.ResetCounters();
	m_pBlock->FillSparseMatrix(&m_matJacobian);
	m_matJacobian.Sort();
	//m_matJacobian.Print();

	return IDA_SUCCESS;
}

int daeTrilinosAmesosSolver::SaveAsPBM(const std::string& strFileName)
{
	m_matJacobian.SaveMatrixAsPBM(strFileName);
	return IDA_SUCCESS;
}

int daeTrilinosAmesosSolver::SaveAsXPM(const std::string& strFileName)
{
	m_matJacobian.SaveMatrixAsXPM(strFileName);
	return IDA_SUCCESS;
}

bool daeTrilinosAmesosSolver::CheckData() const
{
	if(m_matEPETRA && m_vecB && m_vecX && m_nNoEquations > 0)
		return true;
	else
		return false;
}

void daeTrilinosAmesosSolver::AllocateMemory(void)
{
//    * Amesos_Lapack  - Interface to LAPACK's serial dense solver DGETRF.
//    * Amesos_Scalapack - Interface to ScaLAPACK's parallel dense solver PDGETRF.
//    * Amesos_Klu - Interface to Tim Davis serial solver KLU (distributed within Amesos).
//    * Amesos_Umfpack - Interface to Tim Davis's UMFPACK (version 4.3 or later)
//    * Amesos_Pardiso - Interface to PARDISO (prototype)
//    * Amesos_Taucs - Interface to TAUCS
//    * Amesos_Superlu - Interface to Xiaoye Li's SuperLU serial memory code with serial input interface (version 3.0 or later).
//    * Amesos_Superludist - Interface to Xiaoye Li's SuperLU Distributed memory code with serial input interface (version 2.0 or later).
//    * Amesos_Dscpack - Interface to Padma Raghavan's DSCPACK
//    * Amesos_Mumps - Interface to CERFACS' MUMPS (version 4.3.1 or later)

	Amesos Factory;
	if(!Factory.Query(m_strSolverName.c_str()))
	{
		std::cout << "Warning: the solver: [" << m_strSolverName << "] is not supported!" << std:: endl;
		std::cout << "Selected default one: [Amesos_Klu]" << std::endl;
		std::cout << "Supported Amesos solvers: " << std::endl;
		std::cout << "   * Amesos_Klu: "         << (Factory.Query("Amesos_Klu")         ? "true" : "false") << std::endl;
		std::cout << "   * Amesos_Lapack: "      << (Factory.Query("Amesos_Lapack")      ? "true" : "false") << std::endl;
		std::cout << "   * Amesos_Scalapack: "   << (Factory.Query("Amesos_Scalapack")   ? "true" : "false") << std::endl;
		std::cout << "   * Amesos_Umfpack: "     << (Factory.Query("Amesos_Umfpack")     ? "true" : "false") << std::endl;
		std::cout << "   * Amesos_Pardiso: "     << (Factory.Query("Amesos_Pardiso")     ? "true" : "false") << std::endl;
		std::cout << "   * Amesos_Taucs: "       << (Factory.Query("Amesos_Taucs")       ? "true" : "false") << std::endl;
		std::cout << "   * Amesos_Superlu: "     << (Factory.Query("Amesos_Superlu")     ? "true" : "false") << std::endl;
		std::cout << "   * Amesos_Superludist: " << (Factory.Query("Amesos_Superludist") ? "true" : "false") << std::endl;
		std::cout << "   * Amesos_Dscpack: "     << (Factory.Query("Amesos_Dscpack")     ? "true" : "false") << std::endl;
		std::cout << "   * Amesos_Mumps: "       << (Factory.Query("Amesos_Mumps")       ? "true" : "false") << std::endl;

		m_strSolverName = "Amesos_Klu";
	}
	
	m_map.reset(new Epetra_Map(m_nNoEquations, 0, m_Comm));
	m_matEPETRA.reset(new Epetra_CrsMatrix(Copy, *m_map, 0));
	m_matJacobian.InitMatrix(m_nNoEquations, m_matEPETRA.get());
	m_vecB.reset(new Epetra_Vector(*m_map));
	m_vecX.reset(new Epetra_Vector(*m_map));
	
	m_Problem.reset(new Epetra_LinearProblem(m_matEPETRA.get(), m_vecX.get(), m_vecB.get())); 

	m_pSolver.reset(Factory.Create(m_strSolverName.c_str(), *m_Problem));
}

void daeTrilinosAmesosSolver::FreeMemory(void)
{
	m_map.reset();
	m_matEPETRA.reset();
	m_vecB.reset();
	m_vecX.reset();
	m_Problem.reset(); 
	m_pSolver.reset();
}

int daeTrilinosAmesosSolver::Init(void* ida)
{
	return IDA_SUCCESS;
}

int daeTrilinosAmesosSolver::Setup(void*	ida,
								   N_Vector	vectorVariables, 
								   N_Vector	vectorTimeDerivatives, 
								   N_Vector	vectorResiduals,
								   N_Vector	vectorTemp1, 
								   N_Vector	vectorTemp2, 
								   N_Vector	vectorTemp3)
{
	int info;
	realtype *pdValues, *pdTimeDerivatives, *pdResiduals;
		
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	if(!m_pBlock)
		return IDA_MEM_NULL;

	size_t Neq              = m_nNoEquations;
	double time             = ida_mem->ida_tn;
	double dInverseTimeStep = ida_mem->ida_cj;
	
	pdValues			= NV_DATA_S(vectorVariables); 
	pdTimeDerivatives	= NV_DATA_S(vectorTimeDerivatives); 
	pdResiduals			= NV_DATA_S(vectorResiduals);

	m_nJacobianEvaluations++;
	
	m_arrValues.InitArray(Neq, pdValues);
	m_arrTimeDerivatives.InitArray(Neq, pdTimeDerivatives);
	m_arrResiduals.InitArray(Neq, pdResiduals);

	m_matJacobian.ClearMatrix();
	m_pBlock->CalculateJacobian(time, 
		                        m_arrValues, 
							    m_arrResiduals, 
							    m_arrTimeDerivatives, 
							    m_matJacobian, 
							    dInverseTimeStep);
	//m_matJacobian.Print();

	info = m_pSolver->SymbolicFactorization();
	if(info != 0)
	{
		std::cout << "[setup_linear]: SymbolicFactorization failed: " << info << std::endl;
		return IDA_LSETUP_FAIL;
	}
	
	info = m_pSolver->NumericFactorization();
	if(info != 0)
	{
		std::cout << "[setup_linear]: NumericFactorization failed: " << info << std::endl;
		return IDA_LSETUP_FAIL;
	}

	return IDA_SUCCESS;
}

int daeTrilinosAmesosSolver::Solve(void*	ida,
								   N_Vector	vectorB, 
								   N_Vector	vectorWeight, 
								   N_Vector	vectorVariables,
								   N_Vector	vectorTimeDerivatives, 
								   N_Vector	vectorResiduals)
{
	int info;
	realtype* pdB;
		
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	if(!m_pBlock)
		return IDA_MEM_NULL;
	
	size_t Neq	= m_nNoEquations;
	pdB     	= NV_DATA_S(vectorB); 
	
	double *b, *x;
	m_vecB->ExtractView(&b);
	m_vecX->ExtractView(&x);

	::memcpy(b, pdB, Neq*sizeof(double));
	
	info = m_pSolver->Solve();
	if(info != 0)
	{
		std::cout << "[solve_linear]: Solve Failed: " << info << std::endl;
		return IDA_LSOLVE_FAIL;
	}

	::memcpy(pdB, x, Neq*sizeof(double));

	if(ida_mem->ida_cjratio != 1.0)
	{
		for(size_t i = 0; i < Neq; i++)
			pdB[i] *= 2.0 / (1.0 + ida_mem->ida_cjratio);
		//N_VScale(2.0 / (1.0 + ida_mem->ida_cjratio), vectorB, vectorB);
	}
	
	return IDA_SUCCESS;
}

int daeTrilinosAmesosSolver::Free(void* ida)
{
	return IDA_SUCCESS;
}

int init_la(IDAMem ida_mem)
{
	daeTrilinosAmesosSolver* pSolver = (daeTrilinosAmesosSolver*)ida_mem->ida_lmem;
	if(!pSolver)
		return IDA_MEM_NULL;
	
	return pSolver->Init(ida_mem);
}

int setup_la(IDAMem	ida_mem,
			  N_Vector	vectorVariables, 
			  N_Vector	vectorTimeDerivatives, 
			  N_Vector	vectorResiduals,
			  N_Vector	vectorTemp1, 
			  N_Vector	vectorTemp2, 
			  N_Vector	vectorTemp3)
{
	daeTrilinosAmesosSolver* pSolver = (daeTrilinosAmesosSolver*)ida_mem->ida_lmem;
	if(!pSolver)
		return IDA_MEM_NULL;
	
	return pSolver->Setup(ida_mem,
						  vectorVariables, 
						  vectorTimeDerivatives, 
						  vectorResiduals,
						  vectorTemp1, 
						  vectorTemp2, 
						  vectorTemp3);
	
}

int solve_la(IDAMem	ida_mem,
			 N_Vector	vectorB, 
			 N_Vector	vectorWeight, 
			 N_Vector	vectorVariables,
			 N_Vector	vectorTimeDerivatives, 
			 N_Vector	vectorResiduals)
{
	daeTrilinosAmesosSolver* pSolver = (daeTrilinosAmesosSolver*)ida_mem->ida_lmem;
	if(!pSolver)
		return IDA_MEM_NULL;
	
	return pSolver->Solve(ida_mem,
						  vectorB, 
						  vectorWeight, 
						  vectorVariables,
						  vectorTimeDerivatives, 
						  vectorResiduals); 
}

int free_la(IDAMem ida_mem)
{
	daeTrilinosAmesosSolver* pSolver = (daeTrilinosAmesosSolver*)ida_mem->ida_lmem;
	if(!pSolver)
		return IDA_MEM_NULL;
	
	int ret = pSolver->Free(ida_mem);

	delete pSolver;
	ida_mem->ida_lmem = NULL;

	return ret;
}


}
}



