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

daeIDALASolver_t* daeCreateTrilinosSolver(const std::string& strSolverName)
{
	return new daeTrilinosSolver(strSolverName);
}

std::vector<string> daeTrilinosSupportedSolvers(void)
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
	
	strarrSolvers.push_back("AztecOO");
	strarrSolvers.push_back("AztecOO_Ifpack_IC");
	strarrSolvers.push_back("AztecOO_Ifpack_ICT");
	strarrSolvers.push_back("AztecOO_Ifpack_ILU");
	strarrSolvers.push_back("AztecOO_Ifpack_ILUT");

	return strarrSolvers;
}

daeTrilinosSolver::daeTrilinosSolver(const std::string& strSolverName)
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
	m_bIsPreconditionerCreated = false;
	m_bMatrixStructureChanged  = false;
}

daeTrilinosSolver::~daeTrilinosSolver(void)
{
	FreeMemory();
}

int daeTrilinosSolver::Create(void* ida, size_t n, daeDAESolver_t* pDAESolver)
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
	m_bMatrixStructureChanged = true;
	
	ida_mem->ida_linit	= init_la;
	ida_mem->ida_lsetup = setup_la;
	ida_mem->ida_lsolve = solve_la;
	ida_mem->ida_lperf	= NULL;
	ida_mem->ida_lfree	= free_la;

	ida_mem->ida_lmem         = this;
	ida_mem->ida_setupNonNull = TRUE;

	return IDA_SUCCESS;
}

int daeTrilinosSolver::Reinitialize(void* ida)
{
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	
	if(!m_pBlock)
		return IDA_MEM_NULL;
	
// Re-initialize sparse matrix
// ACHTUNG, ACHTUNG!!! It doesn't work, should be checked
// If the matrix structure changes I cannot change Epetra_Crs_Matrix - I have to re-built it!
	m_matJacobian.ResetCounters();
	m_pBlock->FillSparseMatrix(&m_matJacobian);
	m_matJacobian.Sort();
	//m_matJacobian.Print();
	m_bMatrixStructureChanged = true;

	return IDA_SUCCESS;
}

int daeTrilinosSolver::SaveAsXPM(const std::string& strFileName)
{
	m_matJacobian.SaveMatrixAsXPM(strFileName);
	return IDA_SUCCESS;
}

int daeTrilinosSolver::SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
{
	m_matJacobian.SaveAsMatrixMarketFile(strFileName, strMatrixName, strMatrixDescription);
	return IDA_SUCCESS;
}

bool daeTrilinosSolver::CheckData() const
{
	if(m_matEPETRA && m_vecB && m_vecX && m_nNoEquations > 0)
		return true;
	else
		return false;
}

void daeTrilinosSolver::AllocateMemory(void)
{
	if(m_strSolverName == "AztecOO")
	{
		m_eTrilinosSolver = eAztecOO;
	}
	else if(m_strSolverName == "AztecOO_Ifpack_ILU"  ||
			m_strSolverName == "AztecOO_Ifpack_ILUT" ||
			m_strSolverName == "AztecOO_Ifpack_IC"   ||
			m_strSolverName == "AztecOO_Ifpack_ICT ")
	{
		m_eTrilinosSolver = eAztecOO_Ifpack;
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
			  << "   * Amesos_Mumps: "       << (Factory.Query("Amesos_Mumps")       ? "true" : "false") << "\n"
 			  << "Supported AztecOO solvers: " << "\n"
			  << "   * AztecOO (with built-in preconditioners) \n"
			  << "   * AztecOO_Ifpack_IC \n"
			  << "   * AztecOO_Ifpack_ICT \n"
			  << "   * AztecOO_Ifpack_ILU \n"
			  << "   * AztecOO_Ifpack_ILUT \n";
	
			throw e;
		}
		m_eTrilinosSolver = eAmesos;
	}
	
	m_map.reset(new Epetra_Map(m_nNoEquations, 0, m_Comm));
	
	m_matEPETRA.reset(new Epetra_CrsMatrix(Copy, *m_map, 0));

	m_matJacobian.InitMatrix(m_nNoEquations, m_matEPETRA.get());
	m_vecB.reset(new Epetra_Vector(*m_map));
	m_vecX.reset(new Epetra_Vector(*m_map));
	
	m_Problem.reset(new Epetra_LinearProblem(m_matEPETRA.get(), m_vecX.get(), m_vecB.get())); 
	
// Now I can create the solver (because of some implementation details in AztecOO)
	if(m_eTrilinosSolver == eAmesos)
	{
		Amesos Factory;
		m_pSolver.reset(Factory.Create(m_strSolverName.c_str(), *m_Problem));
	}
	else if(m_eTrilinosSolver == eAztecOO)
	{
		m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));
	}
	else
	{
		//Ifpack Factory;
		//m_pPreconditioner.reset(Factory.Create("Amesos", m_matEPETRA.get(), 2));
		
		m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));

		if(m_strSolverName == "AztecOO_Ifpack_ILU")
			m_pPreconditioner.reset(new Ifpack_ILU(m_matEPETRA.get()));
		else if(m_strSolverName == "AztecOO_Ifpack_ILUT")
			m_pPreconditioner.reset(new Ifpack_ILUT(m_matEPETRA.get()));
		else if(m_strSolverName == "AztecOO_Ifpack_IC")
			m_pPreconditioner.reset(new Ifpack_IC(m_matEPETRA.get()));
		else if(m_strSolverName == "AztecOO_Ifpack_ICT")
			m_pPreconditioner.reset(new Ifpack_ICT(m_matEPETRA.get()));
		else
			daeDeclareAndThrowException(exNotImplemented);
		
		m_bIsPreconditionerCreated = false;
	}
}

void daeTrilinosSolver::SetAztecOptions(Teuchos::ParameterList& paramList)
{
	m_parameterListAztec = paramList;
	m_parameterListAztec.print(cout, 0, true, true);
}

void daeTrilinosSolver::SetIfpackOptions(Teuchos::ParameterList& paramList)
{
	m_parameterListIfpack = paramList;
	m_parameterListIfpack.print(cout, 0, true, true);
}

void daeTrilinosSolver::SetAmesosOptions(Teuchos::ParameterList& paramList)
{
	m_parameterListAmesos = paramList;
	m_parameterListAmesos.print(cout, 0, true, true);
}

//void daeTrilinosSolver::SetAztecOption(int Option, int Value)
//{
//	if(m_eTrilinosSolver != eAztecOO || m_eTrilinosSolver != eAztecOO_Ifpack)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot set AztecOO option";
//		throw e;
//	}
//	m_pAztecOOSolver->SetAztecOption(Option, Value);
//}
//
//void daeTrilinosSolver::SetAztecParameter(int Option, double Value)
//{
//	if(m_eTrilinosSolver != eAztecOO || m_eTrilinosSolver != eAztecOO_Ifpack)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot set AztecOO parameter";
//		throw e;
//	}
//	m_pAztecOOSolver->SetAztecParam(Option, Value);
//}

void daeTrilinosSolver::FreeMemory(void)
{
	m_map.reset();
	m_matEPETRA.reset();
	m_vecB.reset();
	m_vecX.reset();
	m_Problem.reset(); 
	m_pSolver.reset();
	m_pAztecOOSolver.reset();
	m_pPreconditioner.reset();
}

int daeTrilinosSolver::Init(void* ida)
{
	return IDA_SUCCESS;
}

int daeTrilinosSolver::Setup(void*	ida,
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

	if(m_eTrilinosSolver == eAmesos)
	{
		m_pSolver->SetParameters(m_parameterListAmesos);
		
	// This does not have to be called each time (but should be called when the nonzero structure changes)
		if(m_bMatrixStructureChanged)
		{
			info = m_pSolver->SymbolicFactorization();
			if(info != 0)
			{
				std::cout << "[setup_linear]: SymbolicFactorization failed: " << info << std::endl;
				return IDA_LSETUP_FAIL;
			}
			m_bMatrixStructureChanged = false;
		}
		
		info = m_pSolver->NumericFactorization();
		if(info != 0)
		{
			std::cout << "[setup_linear]: NumericFactorization failed: " << info << std::endl;
			return IDA_LSETUP_FAIL;
		}
	}
	else if(m_eTrilinosSolver == eAztecOO)
	{
		double condest;
		int nResult;
		
		if(!m_bIsPreconditionerCreated)
		{	
			m_pAztecOOSolver->SetParameters(m_parameterListAztec);
			
//			cout << "Setup before: " << endl;
//			cout << "AZ_precond         = " << m_pAztecOOSolver->GetAztecOption(AZ_precond) << endl;
//			cout << "AZ_subdomain_solve = " << m_pAztecOOSolver->GetAztecOption(AZ_subdomain_solve) << endl;
//			cout << "AZ_kspace          = " << m_pAztecOOSolver->GetAztecOption(AZ_kspace) << endl;
//			cout << "AZ_overlap         = " << m_pAztecOOSolver->GetAztecOption(AZ_overlap) << endl;
//			cout << "AZ_scaling         = " << m_pAztecOOSolver->GetAztecOption(AZ_scaling) << endl;
//			cout << "AZ_reorder         = " << m_pAztecOOSolver->GetAztecOption(AZ_reorder) << endl;
//			cout << "AZ_graph_fill      = " << m_pAztecOOSolver->GetAztecOption(AZ_graph_fill) << endl;
//			cout << "AZ_type_overlap    = " << m_pAztecOOSolver->GetAztecOption(AZ_type_overlap) << endl;
//			cout << "AZ_drop            = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_drop] << endl;
//			cout << "AZ_ilut_fill       = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_ilut_fill] << endl;
//			cout.flush();
			
//			m_pAztecOOSolver->SetAztecOption(AZ_pre_calc,        AZ_calc);
//			m_pAztecOOSolver->SetAztecOption(AZ_solver,          AZ_gmres);
//			m_pAztecOOSolver->SetAztecOption(AZ_precond,         AZ_dom_decomp);
//			m_pAztecOOSolver->SetAztecOption(AZ_subdomain_solve, AZ_ilut);
//			m_pAztecOOSolver->SetAztecParam(AZ_ilut_fill,        3.0);
//			m_pAztecOOSolver->SetAztecOption(AZ_kspace,          m_nNumIters);
//			m_pAztecOOSolver->SetAztecOption(AZ_overlap,         1);
//			m_pAztecOOSolver->SetAztecParam(AZ_athresh,          1e8);
//			m_pAztecOOSolver->SetAztecParam(AZ_rthresh,          0.0);
		
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
			
//			cout << "Setup after: " << endl;
//			cout << "AZ_precond         = " << m_pAztecOOSolver->GetAztecOption(AZ_precond) << endl;
//			cout << "AZ_subdomain_solve = " << m_pAztecOOSolver->GetAztecOption(AZ_subdomain_solve) << endl;
//			cout << "AZ_kspace          = " << m_pAztecOOSolver->GetAztecOption(AZ_kspace) << endl;
//			cout << "AZ_overlap         = " << m_pAztecOOSolver->GetAztecOption(AZ_overlap) << endl;
//			cout << "AZ_scaling         = " << m_pAztecOOSolver->GetAztecOption(AZ_scaling) << endl;
//			cout << "AZ_reorder         = " << m_pAztecOOSolver->GetAztecOption(AZ_reorder) << endl;
//			cout << "AZ_graph_fill      = " << m_pAztecOOSolver->GetAztecOption(AZ_graph_fill) << endl;
//			cout << "AZ_type_overlap    = " << m_pAztecOOSolver->GetAztecOption(AZ_type_overlap) << endl;
//			cout << "AZ_drop            = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_drop] << endl;
//			cout << "AZ_ilut_fill       = " << m_pAztecOOSolver->GetAllAztecParams()[AZ_ilut_fill] << endl;
//			cout.flush();

			m_bIsPreconditionerCreated = true;
		}
	}
	else if(m_eTrilinosSolver == eAztecOO_Ifpack)
	{
		double condest;
		int nResult;
		
		if(m_bIsPreconditionerCreated)
		{
			m_pPreconditioner->Compute();
			if(!m_pPreconditioner->IsComputed())
			{
				std::cout << "[setup_linear]: preconditioner compute failed" << std::endl;
				return IDA_LSETUP_FAIL;
			}
		}
		else
		{	
//			//level_fill, drop_tolerance, absolute_threshold, relative_threshold and overlap_mode
//			Teuchos::ParameterList paramList;
//			paramList.set("fact: level-of-fill", 5);
//			paramList.set("fact: ilut level-of-fill", 3.0);
//			//paramList.set("fact: relax value", 10.0);
//			paramList.set("fact: absolute threshold", 1e8);
//			paramList.set("fact: relative threshold", 0.01);
//			m_pPreconditioner->SetParameters(paramList);

			m_pAztecOOSolver->SetParameters(m_parameterListAztec, true);
			m_pPreconditioner->SetParameters(m_parameterListIfpack);
			
			m_pPreconditioner->Initialize();
			if(!m_pPreconditioner->IsInitialized())
			{
				std::cout << "[setup_linear]: preconditioner initialize failed" << std::endl;
				return IDA_LSETUP_FAIL;
			}
				
			m_pPreconditioner->Compute();
			if(!m_pPreconditioner->IsComputed())
			{
				std::cout << "[setup_linear]: preconditioner compute failed" << std::endl;
				return IDA_LSETUP_FAIL;
			}

			std::cout << "[setup_linear]: preconditioner info:" << std::endl;
			std::cout << *m_pPreconditioner.get() << std::endl;
			Ifpack_Analyze(*m_matEPETRA.get(), false, 1);
			//Ifpack_PrintSparsity(*m_matEPETRA.get(), "/home/ciroki/matrix.ps", 1);
			
			m_pAztecOOSolver->SetPrecOperator(m_pPreconditioner.get());
			
			m_bIsPreconditionerCreated = true;
		}
	}
	
	return IDA_SUCCESS;
}

int daeTrilinosSolver::Solve(void*	ida,
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

	if(m_eTrilinosSolver == eAmesos)
	{
		info = m_pSolver->Solve();
		if(info != 0)
		{
			std::cout << "[solve_linear]: Solve Failed: " << info << std::endl;
			return IDA_LSOLVE_FAIL;
		}
	}
	else if(m_eTrilinosSolver == eAztecOO || m_eTrilinosSolver == eAztecOO_Ifpack)
	{	
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
		
//		m_pAztecOOSolver->SetLHS(m_vecX.get());
//		m_pAztecOOSolver->SetRHS(m_vecB.get());

//		m_pAztecOOSolver->SetAztecOption(AZ_solver,          AZ_gmres);
//		m_pAztecOOSolver->SetAztecOption(AZ_precond,         AZ_dom_decomp);
//		m_pAztecOOSolver->SetAztecOption(AZ_subdomain_solve, AZ_ilut);
//		m_pAztecOOSolver->SetAztecParam(AZ_ilut_fill,        3.0);
//		m_pAztecOOSolver->SetAztecOption(AZ_kspace,          m_nNumIters);
//		m_pAztecOOSolver->SetAztecOption(AZ_overlap,         1);
//		m_pAztecOOSolver->SetAztecParam(AZ_athresh,          1e8);
//		m_pAztecOOSolver->SetAztecParam(AZ_rthresh,          0.0);
	
		m_pAztecOOSolver->SetParameters(m_parameterListAztec);
		nResult = m_pAztecOOSolver->Iterate(m_nNumIters, m_dTolerance);
		
//		cout << "m_nJacobianEvaluations = " << m_nJacobianEvaluations << endl;
//		cout << "Solve after: " << endl;
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

	::memcpy(pdB, x, Neq*sizeof(double));
	if(ida_mem->ida_cjratio != 1.0)
	{
		for(size_t i = 0; i < Neq; i++)
			pdB[i] *= 2.0 / (1.0 + ida_mem->ida_cjratio);
	}
	
	return IDA_SUCCESS;
}

int daeTrilinosSolver::Free(void* ida)
{
	return IDA_SUCCESS;
}

int init_la(IDAMem ida_mem)
{
	daeTrilinosSolver* pSolver = (daeTrilinosSolver*)ida_mem->ida_lmem;
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
	daeTrilinosSolver* pSolver = (daeTrilinosSolver*)ida_mem->ida_lmem;
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
	daeTrilinosSolver* pSolver = (daeTrilinosSolver*)ida_mem->ida_lmem;
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
	daeTrilinosSolver* pSolver = (daeTrilinosSolver*)ida_mem->ida_lmem;
	if(!pSolver)
		return IDA_MEM_NULL;
	
	int ret = pSolver->Free(ida_mem);

	delete pSolver;
	ida_mem->ida_lmem = NULL;

	return ret;
}


}
}



