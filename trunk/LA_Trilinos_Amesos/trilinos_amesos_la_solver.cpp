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
	
	m_nNumIters				= 200;
	m_dTolerance			= 1e-6;
	m_bIsAmesos				= true;
	m_bIsPreconditionerCreated = false;
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

int daeTrilinosAmesosSolver::SaveAsXPM(const std::string& strFileName)
{
	m_matJacobian.SaveMatrixAsXPM(strFileName);
	return IDA_SUCCESS;
}

int daeTrilinosAmesosSolver::SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
{
	m_matJacobian.SaveAsMatrixMarketFile(strFileName, strMatrixName, strMatrixDescription);
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
	if(m_strSolverName == "AztecOO")
	{
		m_bIsAmesos = false;
	}
	else
	{
		Amesos Factory;
		if(!Factory.Query(m_strSolverName.c_str()))
		{
			daeDeclareException(exRuntimeCheck);
			e << "Error: the solver: [" << m_strSolverName << "] is not supported!" << "\n"
			  << "Supported Amesos solvers: " << "\n"
			  << "   * Amesos_Klu: "         << (Factory.Query("Amesos_Klu")         ? "true" : "false") << "\n"
			  << "   * Amesos_Lapack: "      << (Factory.Query("Amesos_Lapack")      ? "true" : "false") << "\n"
			  << "   * Amesos_Scalapack: "   << (Factory.Query("Amesos_Scalapack")   ? "true" : "false") << "\n"
			  << "   * Amesos_Umfpack: "     << (Factory.Query("Amesos_Umfpack")     ? "true" : "false") << "\n"
			  << "   * Amesos_Pardiso: "     << (Factory.Query("Amesos_Pardiso")     ? "true" : "false") << "\n"
			  << "   * Amesos_Taucs: "       << (Factory.Query("Amesos_Taucs")       ? "true" : "false") << "\n"
			  << "   * Amesos_Superlu: "     << (Factory.Query("Amesos_Superlu")     ? "true" : "false") << "\n"
			  << "   * Amesos_Superludist: " << (Factory.Query("Amesos_Superludist") ? "true" : "false") << "\n"
			  << "   * Amesos_Dscpack: "     << (Factory.Query("Amesos_Dscpack")     ? "true" : "false") << "\n"
			  << "   * Amesos_Mumps: "       << (Factory.Query("Amesos_Mumps")       ? "true" : "false") << "\n";
	
			throw e;
		}
		m_bIsAmesos = true;
	}
	
	m_map.reset(new Epetra_Map(m_nNoEquations, 0, m_Comm));
	
	m_matEPETRA.reset(new Epetra_CrsMatrix(Copy, *m_map, 0));

	m_matJacobian.InitMatrix(m_nNoEquations, m_matEPETRA.get());
	m_vecB.reset(new Epetra_Vector(*m_map));
	m_vecX.reset(new Epetra_Vector(*m_map));
	
	m_Problem.reset(new Epetra_LinearProblem(m_matEPETRA.get(), m_vecX.get(), m_vecB.get())); 
	
// Now I can create the solver (because of some implementation details in AztecOO)
	if(m_bIsAmesos)
	{
		Amesos Factory;
		m_pSolver.reset(Factory.Create(m_strSolverName.c_str(), *m_Problem));
	}
	else
	{
		Ifpack Factory;
		m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));		
		m_pPreconditioner.reset(Factory.Create("Ifpack_ILUT", m_matEPETRA.get(), 0));
	}
}

void daeTrilinosAmesosSolver::SetAztecOption(int Option, int Value)
{
	if(m_bIsAmesos || !m_pAztecOOSolver)
	{
		daeDeclareException(exInvalidCall);
		e << "Cannot set AztecOO option";
		throw e;
	}
	m_pAztecOOSolver->SetAztecOption(Option, Value);
}

void daeTrilinosAmesosSolver::SetAztecParameter(int Option, double Value)
{
	if(m_bIsAmesos || !m_pAztecOOSolver)
	{
		daeDeclareException(exInvalidCall);
		e << "Cannot set AztecOO parameter";
		throw e;
	}
	m_pAztecOOSolver->SetAztecParam(Option, Value);
}

void daeTrilinosAmesosSolver::FreeMemory(void)
{
	m_map.reset();
	m_matEPETRA.reset();
	m_vecB.reset();
	m_vecX.reset();
	m_Problem.reset(); 
	m_pSolver.reset();
	m_pAztecOOSolver.reset();
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

/* AMESOS */
	if(m_bIsAmesos)
	{
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
	}
	else
	{
/* AZTECOO */
		double condest;
		int nResult;
		
		if(!m_bIsPreconditionerCreated)
		{	
			Teuchos::ParameterList paramList;
			paramList.set("amesos: solver type", "Amesos_Superlu");

			m_pPreconditioner->SetParameters(paramList);
			m_pPreconditioner->PrintUnused();
			
/*
			m_pAztecOOSolver->SetMatrixName(12345);
			
			cout << "Setup before: " << endl;
			cout << "AZ_precond         = " << m_pAztecOOSolver->GetAztecOption(AZ_precond) << endl;
			cout << "AZ_subdomain_solve = " << m_pAztecOOSolver->GetAztecOption(AZ_subdomain_solve) << endl;
			cout << "AZ_kspace          = " << m_pAztecOOSolver->GetAztecOption(AZ_kspace) << endl;
			cout << "AZ_overlap         = " << m_pAztecOOSolver->GetAztecOption(AZ_overlap) << endl;
			cout << "AZ_scaling         = " << m_pAztecOOSolver->GetAztecOption(AZ_scaling) << endl;
			cout << "AZ_reorder         = " << m_pAztecOOSolver->GetAztecOption(AZ_reorder) << endl;
			cout << "AZ_graph_fill      = " << m_pAztecOOSolver->GetAztecOption(AZ_graph_fill) << endl;
			cout << "AZ_type_overlap    = " << m_pAztecOOSolver->GetAztecOption(AZ_type_overlap) << endl;
			cout << "AZ_drop            = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_drop] << endl;
			cout << "AZ_ilut_fill       = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_ilut_fill] << endl;
			cout.flush();
			
			m_pAztecOOSolver->SetAztecOption(AZ_pre_calc,        AZ_calc);
			m_pAztecOOSolver->SetAztecOption(AZ_solver,          AZ_gmres);
			m_pAztecOOSolver->SetAztecOption(AZ_precond,         AZ_dom_decomp);
			m_pAztecOOSolver->SetAztecOption(AZ_subdomain_solve, AZ_ilut);
			m_pAztecOOSolver->SetAztecParam(AZ_ilut_fill,        3.0);
			m_pAztecOOSolver->SetAztecOption(AZ_kspace,          m_nNumIters);
			m_pAztecOOSolver->SetAztecOption(AZ_overlap,         1);
			m_pAztecOOSolver->SetAztecParam(AZ_athresh,          1e8);
			m_pAztecOOSolver->SetAztecParam(AZ_rthresh,          0.0);
		
			nResult = m_pAztecOOSolver->ConstructPreconditioner(condest);
			if(nResult < 0)
			{
				daeDeclareException(exMiscellanous);
				e << "Failed to create the preconditioner";
				throw e;
			}
			
			double norminf = m_matEPETRA->NormInf();
			double normone = m_matEPETRA->NormOne();
			cout << "\n Condition estimate = "           << condest 
				 << "\n Inf-norm of A before scaling = " << norminf 
				 << "\n One-norm of A before scaling = " << normone << endl << endl;
			
			cout << "Setup after: " << endl;
			cout << "AZ_precond         = " << m_pAztecOOSolver->GetAztecOption(AZ_precond) << endl;
			cout << "AZ_subdomain_solve = " << m_pAztecOOSolver->GetAztecOption(AZ_subdomain_solve) << endl;
			cout << "AZ_kspace          = " << m_pAztecOOSolver->GetAztecOption(AZ_kspace) << endl;
			cout << "AZ_overlap         = " << m_pAztecOOSolver->GetAztecOption(AZ_overlap) << endl;
			cout << "AZ_scaling         = " << m_pAztecOOSolver->GetAztecOption(AZ_scaling) << endl;
			cout << "AZ_reorder         = " << m_pAztecOOSolver->GetAztecOption(AZ_reorder) << endl;
			cout << "AZ_graph_fill      = " << m_pAztecOOSolver->GetAztecOption(AZ_graph_fill) << endl;
			cout << "AZ_type_overlap    = " << m_pAztecOOSolver->GetAztecOption(AZ_type_overlap) << endl;
			cout << "AZ_drop            = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_drop] << endl;
			cout << "AZ_ilut_fill       = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_ilut_fill] << endl;
			cout.flush();
*/			
			m_bIsPreconditionerCreated = true;
		}
		else
		{
			m_pMLPreconditioner->DestroyPreconditioner();
			m_pMLPreconditioner->ReComputePreconditioner();
		}
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

//	daeDenseArray arr;
//	arr.InitArray(Neq, pdB);
//	std::cout << "b vector" << std::endl;
//	arr.Print();

	::memcpy(b, pdB, Neq*sizeof(double));

/* AMESOS */
	if(m_bIsAmesos)
	{
		info = m_pSolver->Solve();
		if(info != 0)
		{
			std::cout << "[solve_linear]: Solve Failed: " << info << std::endl;
			return IDA_LSOLVE_FAIL;
		}
	}
	else
	{	
/* AZTECOO */
		int nResult;
		double condest;
	
//		cout << "Solve before: " << endl;
//		cout << "AZ_precond         = " << m_pAztecOOSolver->GetAztecOption(AZ_precond) << endl;
//		cout << "AZ_subdomain_solve = " << m_pAztecOOSolver->GetAztecOption(AZ_subdomain_solve) << endl;
//		cout << "AZ_kspace          = " << m_pAztecOOSolver->GetAztecOption(AZ_kspace) << endl;
//		cout << "AZ_overlap         = " << m_pAztecOOSolver->GetAztecOption(AZ_overlap) << endl;
//		cout << "AZ_scaling         = " << m_pAztecOOSolver->GetAztecOption(AZ_scaling) << endl;
//		cout << "AZ_reorder         = " << m_pAztecOOSolver->GetAztecOption(AZ_reorder) << endl;
//		cout << "AZ_graph_fill      = " << m_pAztecOOSolver->GetAztecOption(AZ_graph_fill) << endl;
//		cout << "AZ_type_overlap    = " << m_pAztecOOSolver->GetAztecOption(AZ_type_overlap) << endl;
//		cout << "AZ_drop            = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_drop] << endl;
//		cout << "AZ_ilut_fill       = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_ilut_fill] << endl;
//		cout.flush();
		
		m_pAztecOOSolver->SetLHS(m_vecX.get());
		m_pAztecOOSolver->SetRHS(m_vecB.get());

//		m_pAztecOOSolver->SetAztecOption(AZ_solver,          AZ_gmres);
//		m_pAztecOOSolver->SetAztecOption(AZ_precond,         AZ_dom_decomp);
//		m_pAztecOOSolver->SetAztecOption(AZ_subdomain_solve, AZ_ilut);
//		m_pAztecOOSolver->SetAztecParam(AZ_ilut_fill,        3.0);
//		m_pAztecOOSolver->SetAztecOption(AZ_kspace,          m_nNumIters);
//		m_pAztecOOSolver->SetAztecOption(AZ_overlap,         1);
//		m_pAztecOOSolver->SetAztecParam(AZ_athresh,          1e8);
//		m_pAztecOOSolver->SetAztecParam(AZ_rthresh,          0.0);
	
		nResult = m_pAztecOOSolver->Iterate(m_nNumIters, m_dTolerance);
		cout << "m_nJacobianEvaluations = " << m_nJacobianEvaluations << endl;
		
		if(nResult == 1)
		{
			// Max. Iterations reached
			return IDA_CONV_FAIL;
			//Or return this: return IDA_LSOLVE_FAIL;
		}
		else if(nResult < 0)
		{
			string strError;
			daeDeclareException(exMiscellanous);
			
			if(nResult == -1) 
				strError = "Aztec status AZ_param: option not implemented";
			else if(nResult == -2) 
				strError = "Aztec status AZ_breakdown: numerical breakdown";
			else if(nResult == -3) 
				strError = "Aztec status AZ_loss: loss of precision";
			else if(nResult == -4) 
				strError = "Aztec status AZ_ill_cond: GMRES hessenberg ill-conditioned";
				
			e << "Trilinos AztecOO error: " + strError;
			throw e;
			
			return IDA_LSOLVE_FAIL;
		}
	}	
//	cout << "Solve after: " << endl;
//	cout << "AZ_precond         = " << m_pAztecOOSolver->GetAztecOption(AZ_precond) << endl;
//	cout << "AZ_subdomain_solve = " << m_pAztecOOSolver->GetAztecOption(AZ_subdomain_solve) << endl;
//	cout << "AZ_kspace          = " << m_pAztecOOSolver->GetAztecOption(AZ_kspace) << endl;
//	cout << "AZ_overlap         = " << m_pAztecOOSolver->GetAztecOption(AZ_overlap) << endl;
//	cout << "AZ_scaling         = " << m_pAztecOOSolver->GetAztecOption(AZ_scaling) << endl;
//	cout << "AZ_reorder         = " << m_pAztecOOSolver->GetAztecOption(AZ_reorder) << endl;
//	cout << "AZ_graph_fill      = " << m_pAztecOOSolver->GetAztecOption(AZ_graph_fill) << endl;
//	cout << "AZ_type_overlap    = " << m_pAztecOOSolver->GetAztecOption(AZ_type_overlap) << endl;
//	cout << "AZ_drop            = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_drop] << endl;
//	cout << "AZ_ilut_fill       = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_ilut_fill] << endl;
//	cout.flush();

	::memcpy(pdB, x, Neq*sizeof(double));
	if(ida_mem->ida_cjratio != 1.0)
	{
		for(size_t i = 0; i < Neq; i++)
			pdB[i] *= 2.0 / (1.0 + ida_mem->ida_cjratio);
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



