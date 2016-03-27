#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include <idas/idas_impl.h>
#include "trilinos_amesos_la_solver.h"

using namespace std;

#ifdef DAE_USE_OPEN_BLAS
extern "C" void openblas_set_num_threads(int);
#endif

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

daeIDALASolver_t* daeCreateTrilinosSolver(const std::string& strSolverName, const std::string& strPreconditionerName)
{
	return new daeTrilinosSolver(strSolverName, strPreconditionerName);
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
	strarrSolvers.push_back("AztecOO_Ifpack");
	strarrSolvers.push_back("AztecOO_ML");

//	strarrSolvers.push_back("AztecOO_Ifpack_IC");
//	strarrSolvers.push_back("AztecOO_Ifpack_ICT");
//	strarrSolvers.push_back("AztecOO_Ifpack_ILU");
//	strarrSolvers.push_back("AztecOO_Ifpack_ILUT");
//	
//	strarrSolvers.push_back("AztecOO_ML_SA");
//	strarrSolvers.push_back("AztecOO_ML_DD");
//	strarrSolvers.push_back("AztecOO_ML_DD-ML");
//	strarrSolvers.push_back("AztecOO_ML_maxwell");
//	strarrSolvers.push_back("AztecOO_ML_NSSA");

	return strarrSolvers;
}

daeTrilinosSolver::daeTrilinosSolver(const std::string& strSolverName, const std::string& strPreconditionerName)
#ifdef HAVE_MPI
	: m_Comm(MPI_COMM_WORLD)
#endif
{
// Set OpenBLAS to use only one thread (OpenBLAS can't decide based on the matrix size)
// It can be changed later on by the user
    SetOpenBLASNoThreads(1);

    m_pBlock				= NULL;
	m_strSolverName			= strSolverName;
	m_strPreconditionerName = strPreconditionerName;
	m_nNoEquations			= 0;
	m_nJacobianEvaluations	= 0;
	m_eTrilinosSolver		= eAmesos;
	
	m_nNumIters						= 200;
	m_dTolerance					= 1e-6;
	m_bIsPreconditionerConstructed	= false;
	m_bMatrixStructureChanged		= false;
	
	if(m_strSolverName == "AztecOO")
	{
		m_eTrilinosSolver = eAztecOO;
	}
	else if(m_strSolverName == "AztecOO_Ifpack")
	{
		m_eTrilinosSolver = eAztecOO_Ifpack;
		
		if(m_strPreconditionerName != "ILU"  &&
		   m_strPreconditionerName != "ILUT" &&
		   m_strPreconditionerName != "PointRelaxation"   &&
		   m_strPreconditionerName != "BlockRelaxation"   &&
		   m_strPreconditionerName != "IC"   &&
		   m_strPreconditionerName != "ICT")
		{
			daeDeclareException(exRuntimeCheck);
			e << "Error: Ifpack preconditioner: [" << m_strPreconditionerName << "] is not supported!";
			throw e;			
		}
	}
	else if(m_strSolverName == "AztecOO_ML")
	{
		m_eTrilinosSolver = eAztecOO_ML;
		
		if(m_strPreconditionerName != "SA"       &&
		   m_strPreconditionerName != "DD"       &&
		   m_strPreconditionerName != "DD-ML"    &&
		   m_strPreconditionerName != "DD-ML-LU" &&
		   m_strPreconditionerName != "maxwell"  &&
		   m_strPreconditionerName != "NSSA")
		{
			daeDeclareException(exRuntimeCheck);
			e << "Error: ML preconditioner: [" << m_strPreconditionerName << "] is not supported!";
			throw e;			
		}
		
		if(m_strPreconditionerName == "SA")
			ML_Epetra::SetDefaults("SA", m_parameterListML);
		else if(m_strPreconditionerName == "DD")
			ML_Epetra::SetDefaults("DD", m_parameterListML);
		else if(m_strPreconditionerName == "DD-ML")
			ML_Epetra::SetDefaults("DD-ML", m_parameterListML);
		else if(m_strPreconditionerName == "DD-ML-LU")
			ML_Epetra::SetDefaults("DD-ML-LU", m_parameterListML);
		else if(m_strPreconditionerName == "DD-LU")
			ML_Epetra::SetDefaults("DD-LU", m_parameterListML);
		else if(m_strPreconditionerName == "maxwell")
			ML_Epetra::SetDefaults("maxwell", m_parameterListML);
		else if(m_strPreconditionerName == "NSSA")
			ML_Epetra::SetDefaults("NSSA", m_parameterListML);
		else
			daeDeclareAndThrowException(exNotImplemented);
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
			  << "   * AztecOO_Ifpack \n"
			  << "   * AztecOO_ML \n";
//			  << "   * AztecOO_Ifpack_IC \n"
//			  << "   * AztecOO_Ifpack_ICT \n"
//			  << "   * AztecOO_Ifpack_ILU \n"
//			  << "   * AztecOO_Ifpack_ILUT \n"
//			  << "   * AztecOO_ML_SA \n"
//			  << "   * AztecOO_ML_DD \n"
//			  << "   * AztecOO_ML_DD-ML \n"
//			  << "   * AztecOO_ML_maxwell \n"
//			  << "   * AztecOO_ML_NSSA \n";
			throw e;
		}
		m_eTrilinosSolver = eAmesos;
	}
}

void daeTrilinosSolver::SetOpenBLASNoThreads(int n)
{
#ifdef DAE_USE_OPEN_BLAS
    openblas_set_num_threads(n);
#endif
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
	
// ACHTUNG, ACHTUNG!!! 
// If the matrix structure changes I cannot modify Epetra_Crs_Matrix - only to re-built it!
// Therefore, here we create a new Epetra_CrsMatrix and use it to re-init the Jacobian matrix
	m_matEPETRA.reset(new Epetra_CrsMatrix(Copy, *m_map, 0));
	m_matJacobian.InitMatrix(m_nNoEquations, m_matEPETRA.get());	
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

std::string daeTrilinosSolver::GetName(void) const
{
	return m_strSolverName + (m_strPreconditionerName.empty() ? "" : " (" + m_strPreconditionerName + ")");
}

std::string daeTrilinosSolver::GetPreconditionerName(void) const
{
	return m_strPreconditionerName;
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
	m_map.reset(new Epetra_Map(m_nNoEquations, 0, m_Comm));
	
	m_matEPETRA.reset(new Epetra_CrsMatrix(Copy, *m_map, 0));

	m_matJacobian.InitMatrix(m_nNoEquations, m_matEPETRA.get());
	m_vecB.reset(new Epetra_Vector(*m_map));
	m_vecX.reset(new Epetra_Vector(*m_map));
	
/*	
	m_Problem.reset(new Epetra_LinearProblem(m_matEPETRA.get(), m_vecX.get(), m_vecB.get())); 
	
	if(m_eTrilinosSolver == eAmesos)
	{
		Amesos Factory;
		m_pAmesosSolver.reset(Factory.Create(m_strSolverName.c_str(), *m_Problem));
	}
	else if(m_eTrilinosSolver == eAztecOO)
	{
		m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));
		m_pAztecOOSolver->SetAztecDefaults();
	}
	else if(m_eTrilinosSolver == eAztecOO_ML)
	{
		m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));
		m_pAztecOOSolver->SetAztecDefaults();

		if(m_strPreconditionerName == "SA")
			ML_Epetra::SetDefaults("SA", m_parameterListML);
		else if(m_strPreconditionerName == "DD")
			ML_Epetra::SetDefaults("DD", m_parameterListML);
		else if(m_strPreconditionerName == "DD-ML")
			ML_Epetra::SetDefaults("DD-ML", m_parameterListML);
		else if(m_strPreconditionerName == "DD-ML-LU")
			ML_Epetra::SetDefaults("DD-ML-LU", m_parameterListML);
		else if(m_strPreconditionerName == "maxwell")
			ML_Epetra::SetDefaults("maxwell", m_parameterListML);
		else if(m_strPreconditionerName == "NSSA")
			ML_Epetra::SetDefaults("NSSA", m_parameterListML);
		else
			daeDeclareAndThrowException(exNotImplemented);

		m_pPreconditionerML.reset(new ML_Epetra::MultiLevelPreconditioner(*m_matEPETRA.get(), m_parameterListML, false));
		m_bIsPreconditionerConstructed = false;
	}
	else if(m_eTrilinosSolver == eAztecOO_Ifpack)
	{
		m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));
		m_pAztecOOSolver->SetAztecDefaults();

		if(m_strPreconditionerName == "ILU")
			m_pPreconditionerIfpack.reset(new Ifpack_ILU(m_matEPETRA.get()));
		else if(m_strPreconditionerName == "ILUT")
			m_pPreconditionerIfpack.reset(new Ifpack_ILUT(m_matEPETRA.get()));
		else if(m_strPreconditionerName == "IC")
			m_pPreconditionerIfpack.reset(new Ifpack_IC(m_matEPETRA.get()));
		else if(m_strPreconditionerName == "ICT")
			m_pPreconditionerIfpack.reset(new Ifpack_ICT(m_matEPETRA.get()));
		else
			daeDeclareAndThrowException(exNotImplemented);
		
		m_bIsPreconditionerConstructed = false;
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
*/
}

void daeTrilinosSolver::FreeMemory(void)
{
	m_map.reset();
	m_matEPETRA.reset();
	m_vecB.reset();
	m_vecX.reset();
	m_Problem.reset(); 
	m_pAmesosSolver.reset();
	m_pAztecOOSolver.reset();
	m_pPreconditionerML.reset();
	m_pPreconditionerIfpack.reset();
}

int daeTrilinosSolver::Init(void* ida)
{
	return IDA_SUCCESS;
}

bool daeTrilinosSolver::SetupLinearProblem(void)
{
	int nResult;
	
// At this point I should have the structure of A and the numerical values of A, x and b computed
	
	m_Problem.reset(new Epetra_LinearProblem(m_matEPETRA.get(), m_vecX.get(), m_vecB.get())); 
	
	if(m_eTrilinosSolver == eAmesos)
	{
		Amesos Factory;
		m_pAmesosSolver.reset(Factory.Create(m_strSolverName.c_str(), *m_Problem));
		
		nResult = m_pAmesosSolver->SymbolicFactorization();
		if(nResult != 0)
		{
			std::cout << "Symbolic factorization failed: " << nResult << std::endl;
			return false;
		}
	}
	else if(m_eTrilinosSolver == eAztecOO)
	{
		m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));
		// SetAztecDefaults has been called in the AztecOO constructor
		m_parameterListAztec.set("AZ_keep_info", 1);
		m_pAztecOOSolver->SetParameters(m_parameterListAztec, true);
	}
	else if(m_eTrilinosSolver == eAztecOO_ML)
	{
		m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));
		m_pAztecOOSolver->SetParameters(m_parameterListAztec, true);

		if(m_strPreconditionerName == "SA")
			ML_Epetra::SetDefaults("SA", m_parameterListML);
		else if(m_strPreconditionerName == "DD")
			ML_Epetra::SetDefaults("DD", m_parameterListML);
		else if(m_strPreconditionerName == "DD-ML")
			ML_Epetra::SetDefaults("DD-ML", m_parameterListML);
		else if(m_strPreconditionerName == "DD-ML-LU")
			ML_Epetra::SetDefaults("DD-ML-LU", m_parameterListML);
		else if(m_strPreconditionerName == "maxwell")
			ML_Epetra::SetDefaults("maxwell", m_parameterListML);
		else if(m_strPreconditionerName == "NSSA")
			ML_Epetra::SetDefaults("NSSA", m_parameterListML);
		else
			daeDeclareAndThrowException(exNotImplemented);

		m_parameterListML.set("reuse: enable", true);
		
		m_pPreconditionerML.reset(new ML_Epetra::MultiLevelPreconditioner(*m_matEPETRA.get(), m_parameterListML, false));
		m_bIsPreconditionerConstructed = false;

		nResult = m_pAztecOOSolver->SetPrecOperator(m_pPreconditionerML.get());		
		if(nResult != 0)
		{
			std::cout << "Failed to set tht ML preconditioner" << std::endl;
			return false;
		}
	}
	else if(m_eTrilinosSolver == eAztecOO_Ifpack)
	{
		m_pAztecOOSolver.reset(new AztecOO(*m_Problem.get()));
		m_pAztecOOSolver->SetParameters(m_parameterListAztec, true);

		if(m_strPreconditionerName == "ILU")
			m_pPreconditionerIfpack.reset(new Ifpack_ILU(m_matEPETRA.get()));
		else if(m_strPreconditionerName == "ILUT")
			m_pPreconditionerIfpack.reset(new Ifpack_ILUT(m_matEPETRA.get()));
		else if(m_strPreconditionerName == "IC")
			m_pPreconditionerIfpack.reset(new Ifpack_IC(m_matEPETRA.get()));
		else if(m_strPreconditionerName == "ICT")
			m_pPreconditionerIfpack.reset(new Ifpack_ICT(m_matEPETRA.get()));
		else if(m_strPreconditionerName == "PointRelaxation")
			m_pPreconditionerIfpack.reset(new Ifpack_PointRelaxation(m_matEPETRA.get()));
		else
			daeDeclareAndThrowException(exNotImplemented);
		
		m_pPreconditionerIfpack->SetParameters(m_parameterListIfpack);
		m_bIsPreconditionerConstructed = false;
		
		nResult = m_pPreconditionerIfpack->Initialize();
		if(nResult != 0)
		{
			std::cout << "Ifpack preconditioner initialize failed" << std::endl;
			return false;
		}
		
		nResult = m_pPreconditionerIfpack->Compute();
		if(nResult != 0)
		{
			std::cout << "Ifpack preconditioner compute failed: " << std::endl;
			return IDA_LSETUP_FAIL;
		}
		
		nResult = m_pAztecOOSolver->SetPrecOperator(m_pPreconditionerIfpack.get());
		if(nResult != 0)
		{
			std::cout << "Failed to set the Ifpack preconditioner" << std::endl;
			return false;
		}
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	
	m_bMatrixStructureChanged = false;
	return true;
}

void daeTrilinosSolver::PrintPreconditionerInfo(void)
{
	if(m_eTrilinosSolver == eAztecOO)
	{
	}
	else if(m_eTrilinosSolver == eAztecOO_ML)
	{
	}
	else if(m_eTrilinosSolver == eAztecOO_Ifpack)
	{
		std::cout << *m_pPreconditionerIfpack.get() << std::endl;
		//Ifpack_Analyze(*m_matEPETRA.get(), false, 1);
	}
}

int daeTrilinosSolver::Setup(void*		ida,
							 N_Vector	vectorVariables, 
							 N_Vector	vectorTimeDerivatives, 
							 N_Vector	vectorResiduals,
							 N_Vector	vectorTemp1, 
							 N_Vector	vectorTemp2, 
							 N_Vector	vectorTemp3)
{
	int nResult;
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

	if(m_bMatrixStructureChanged)
	{
		if(!SetupLinearProblem())
		{
			std::cout << "SetupLinearProblem failed" << std::endl;
			return IDA_LSETUP_FAIL;
		}
	}
	
	if(m_eTrilinosSolver == eAmesos)
	{
		if(!m_pAmesosSolver)
			daeDeclareAndThrowException(exInvalidCall);
/*		
	// This does not have to be called each time (but should be called when the nonzero structure changes)
		if(m_bMatrixStructureChanged)
		{
			//m_pAmesosSolver->SetParameters(m_parameterListAmesos);
			
			nResult = m_pAmesosSolver->SymbolicFactorization();
			if(nResult != 0)
			{
				std::cout << "[setup_linear]: SymbolicFactorization failed: " << info << std::endl;
				return IDA_LSETUP_FAIL;
			}
			m_bMatrixStructureChanged = false;
		}
*/		
		nResult = m_pAmesosSolver->NumericFactorization();
		if(nResult != 0)
		{
			std::cout << "Amesos: numeric factorization failed: " << nResult << std::endl;
			return IDA_LSETUP_FAIL;
		}
	}
	else if(m_eTrilinosSolver == eAztecOO)
	{
		double condest;
		
		if(!m_pAztecOOSolver)
			daeDeclareAndThrowException(exInvalidCall);

		nResult = m_pAztecOOSolver->ConstructPreconditioner(condest);
		if(nResult < 0)
		{
			std::cout << "AztecOO: construct preconditioner failed: " << nResult << std::endl;
			return IDA_LSETUP_FAIL;
		}
		
		cout << "Condition estimate = " << condest << endl << endl;
		
/*		
		if(!m_bIsPreconditionerConstructed)
		{	
			//m_pAztecOOSolver->SetParameters(m_parameterListAztec);
			
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
				e << "Failed to construct the preconditioner";
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

			m_bIsPreconditionerConstructed = true;
		}
*/
	}
	else if(m_eTrilinosSolver == eAztecOO_Ifpack)
	{
		if(!m_pAztecOOSolver || !m_pPreconditionerIfpack)
			daeDeclareAndThrowException(exInvalidCall);

		nResult = m_pPreconditionerIfpack->Compute();
		if(nResult != 0)
		{
			std::cout << "Ifpack: compute preconditioner failed: " << nResult << std::endl;
			return IDA_LSETUP_FAIL;
		}
		
/*
		if(m_bIsPreconditionerConstructed)
		{
			m_pPreconditionerIfpack->Compute();
			if(!m_pPreconditionerIfpack->IsComputed())
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
//			m_pPreconditionerIfpack->SetParameters(paramList);

//			m_pAztecOOSolver->SetPrecOperator(m_pPreconditionerIfpack.get());		
//			m_pAztecOOSolver->SetParameters(m_parameterListAztec, true);
//			m_pPreconditionerIfpack->SetParameters(m_parameterListIfpack);
			
			m_pPreconditionerIfpack->Initialize();
			if(!m_pPreconditionerIfpack->IsInitialized())
			{
				std::cout << "[setup_linear]: preconditioner initialize failed" << std::endl;
				return IDA_LSETUP_FAIL;
			}
				
			m_pPreconditionerIfpack->Compute();
			if(!m_pPreconditionerIfpack->IsComputed())
			{
				std::cout << "[setup_linear]: preconditioner compute failed" << std::endl;
				return IDA_LSETUP_FAIL;
			}

			std::cout << "[setup_linear]: preconditioner info:" << std::endl;
			std::cout << *m_pPreconditionerIfpack.get() << std::endl;
			Ifpack_Analyze(*m_matEPETRA.get(), false, 1);
			//Ifpack_PrintSparsity(*m_matEPETRA.get(), "/home/ciroki/matrix.ps", 1);
			
			m_bIsPreconditionerConstructed = true;
		}
*/
	}
	else if(m_eTrilinosSolver == eAztecOO_ML)
	{
		double condest;
		int nResult;
		
		if(!m_pAztecOOSolver || !m_pPreconditionerML)
			daeDeclareAndThrowException(exInvalidCall);

		nResult = m_pPreconditionerML->ComputePreconditioner();
		if(nResult != 0)
		{
			std::cout << "ML: compute preconditioner failed: " << nResult << std::endl;
			return IDA_LSETUP_FAIL;
		}
		
/*
		if(m_bIsPreconditionerConstructed)
		{
			nResult = m_pPreconditionerML->ComputePreconditioner();
			if(nResult != 0)
			{
				std::cout << "[setup_linear]: preconditioner compute failed" << std::endl;
				return IDA_LSETUP_FAIL;
			}
		}
		else
		{
			m_pAztecOOSolver->SetPrecOperator(m_pPreconditionerML.get());		
			m_pAztecOOSolver->SetParameters(m_parameterListAztec, true);
			m_pPreconditionerML->SetParameterList(m_parameterListML);
			m_bIsPreconditionerConstructed = true;
		}
*/
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
	int nResult;
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
		nResult = m_pAmesosSolver->Solve();
		if(nResult != 0)
		{
			std::cout << "Amesos: solve failed: " << nResult << std::endl;
			return IDA_LSOLVE_FAIL;
		}
	}
	else if(m_eTrilinosSolver == eAztecOO       || 
			m_eTrilinosSolver == eAztecOO_ML    || 
			m_eTrilinosSolver == eAztecOO_Ifpack)
	{	
/*	
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
	
		//m_pAztecOOSolver->SetParameters(m_parameterListAztec);
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
*/
		try
		{
			nResult = m_pAztecOOSolver->Iterate(m_nNumIters, m_dTolerance);
			if(nResult == 1)
			{
				std::cout << "AztecOO solve failed: Max. iterations reached" << std::endl;
				return IDA_CONV_FAIL;
				//Or return this: return IDA_LSOLVE_FAIL;
			}
			else if(nResult < 0)
			{
				string strError;
				
				if(nResult == -1) 
					strError = "Aztec status AZ_param: option not implemented";
				else if(nResult == -2) 
					strError = "Aztec status AZ_breakdown: numerical breakdown";
				else if(nResult == -3) 
					strError = "Aztec status AZ_loss: loss of precision";
				else if(nResult == -4) 
					strError = "Aztec status AZ_ill_cond: GMRES hessenberg ill-conditioned";
				
				std::cout << "AztecOO solve failed: " << strError << std::endl;
				return IDA_LSOLVE_FAIL;
			}
		}
		catch(std::exception& e)
		{
			std::cout << "AztecOO solve failed (exception thrown): " << e.what() << std::endl;
			return IDA_LSOLVE_FAIL;
		}
		catch(...)
		{
			std::cout << "AztecOO solve failed (unknown exception thrown)" << std::endl;
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

Teuchos::ParameterList& daeTrilinosSolver::GetAmesosOptions(void)
{
//	if(m_eTrilinosSolver != eAmesos || !m_pAmesosSolver)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot get Amesos options before the solver is created";
//		throw e;
//	}
	return m_parameterListAmesos;
}

Teuchos::ParameterList& daeTrilinosSolver::GetAztecOOOptions(void)
{
//	if(m_eTrilinosSolver != eAztecOO || !m_pAztecOOSolver)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot get AztecOO options before the solver is created";
//		throw e;
//	}
	return m_parameterListAztec;
}

Teuchos::ParameterList& daeTrilinosSolver::GetIfpackOptions(void)
{
//	if(m_eTrilinosSolver != eAztecOO_Ifpack || !m_pAztecOOSolver)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot get Ifpack options before the solver is created";
//		throw e;
//	}
	return m_parameterListIfpack;
}

Teuchos::ParameterList& daeTrilinosSolver::GetMLOptions(void)
{
//	if(m_eTrilinosSolver != eAztecOO_ML || !m_pAztecOOSolver)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot get ML options before the solver is created";
//		throw e;
//	}
	return m_parameterListML;
}

void daeTrilinosSolver::SetAmesosOptions(Teuchos::ParameterList& paramList)
{
//	if(m_eTrilinosSolver != eAmesos || !m_pAmesosSolver)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot set Amesos options before the solver is created";
//		throw e;
//	}
	m_parameterListAmesos = paramList;
//	m_parameterListAmesos.print(cout, 0, true, true);
//	m_pAmesosSolver->SetParameters(m_parameterListAztec);
}

void daeTrilinosSolver::SetAztecOOOptions(Teuchos::ParameterList& paramList)
{
//	if(m_eTrilinosSolver != eAztecOO || !m_pAztecOOSolver)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot set AztecOO options before the solver is created";
//		throw e;
//	}
	m_parameterListAztec = paramList;
//	m_parameterListAztec.print(cout, 0, true, true);
//	m_pAztecOOSolver->SetParameters(m_parameterListAztec, true);
}

void daeTrilinosSolver::SetIfpackOptions(Teuchos::ParameterList& paramList)
{
//	if(m_eTrilinosSolver != eAztecOO_Ifpack || !m_pAztecOOSolver)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot set Ifpack options before the solver is created";
//		throw e;
//	}
	m_parameterListIfpack = paramList;
//	m_parameterListIfpack.print(cout, 0, true, true);
//	m_pPreconditionerIfpack->SetParameters(m_parameterListIfpack);
}

void daeTrilinosSolver::SetMLOptions(Teuchos::ParameterList& paramList)
{
//	if(m_eTrilinosSolver != eAztecOO_ML || !m_pAztecOOSolver)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Cannot get ML options before the solver is created";
//		throw e;
//	}
	m_parameterListML = paramList;
//	m_parameterListML.print(cout, 0, true, true);
//	m_pPreconditionerML->SetParameterList(m_parameterListML);
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

// ACHTUNG, ACHTUNG!!
// It is the responsibility of the user to delete LA solver pointer!!
//	delete pSolver;

	ida_mem->ida_lmem = NULL;

	return ret;
}


}
}



