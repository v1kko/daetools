#include "stdafx.h"
using namespace std;
#include "ida_solver.h"
#include "../Core/helpers.h"
#include "dae_solvers.h"
#include <ida/ida.h>
#include <ida/ida_impl.h>
#include <ida/ida_dense.h>
#include <ida/ida_lapack.h>
#include <ida/ida_spgmr.h>

#define JACOBIAN(A) (A->cols)
 
namespace dae
{
namespace solver
{
int residuals(realtype	time, 
			  N_Vector	vectorVariables, 
			  N_Vector	vectorTimeDerivatives, 
			  N_Vector	vectorResiduals, 
			  void*		pUserData);

int roots(realtype	time, 
		  N_Vector	vectorVariables, 
		  N_Vector	vectorTimeDerivatives, 
		  realtype*	gout, 
		  void*		pUserData);

int jacobian(int	    Neq, 
			 realtype	time, 
			 realtype	dInverseTimeStep, 
			 N_Vector	vectorVariables, 
			 N_Vector	vectorTimeDerivatives,
			 N_Vector	vectorResiduals, 
			 DlsMat		dense_matrixJacobian,
			 void*		pUserData, 
			 N_Vector	vectorTemp1, 
			 N_Vector	vectorTemp2, 
			 N_Vector	vectorTemp3);

int setup_preconditioner(realtype	time, 
						 N_Vector	vectorVariables, 
						 N_Vector	vectorTimeDerivatives,
						 N_Vector	vectorResiduals, 
						 realtype	dInverseTimeStep, 
						 void*		pUserData, 
						 N_Vector	vectorTemp1, 
						 N_Vector	vectorTemp2, 
						 N_Vector	vectorTemp3);

int solve_preconditioner(realtype	time, 
						 N_Vector	vectorVariables, 
						 N_Vector	vectorTimeDerivatives,
						 N_Vector	vectorResiduals, 
						 N_Vector	vectorR, 
						 N_Vector	vectorZ, 
						 realtype	dInverseTimeStep,
						 realtype	delta,
						 void*		pUserData, 
						 N_Vector	vectorTemp);

daeIDASolver::daeIDASolver(void)
{
	m_pLog					         = NULL;
	m_pBlock				         = NULL;
	m_pIDA					         = NULL;
	m_eLASolver				         = eSundialsLU;
	m_pLASolver				         = NULL;
	m_dCurrentTime			         = 0;
	m_nNumberOfEquations	         = 0;
	m_dRelTolerance			         = 0;
	m_timeStart				         = 0;
	m_dTargetTime			         = 0;
	m_dNextTimeAfterReinitialization = 0;
	m_eInitialConditionMode          = eAlgebraicValuesProvided;
	m_pIDASolverData.reset(new daeIDASolverData);

	daeConfig& cfg = daeConfig::GetConfig();
	m_dRelTolerance                  = cfg.Get<real_t>("daetools.solver.relativeTolerance", 1e-5);
	m_dNextTimeAfterReinitialization = cfg.Get<real_t>("daetools.solver.nextTimeAfterReinitialization", 1e-2);
}

daeIDASolver::~daeIDASolver(void)
{
	if(m_pIDA) 
		IDAFree(&m_pIDA);		
}

void daeIDASolver::SetLASolver(daeIDALASolver_t* pLASolver)
{
	m_eLASolver = eThirdParty;
	m_pLASolver = pLASolver;
}

void daeIDASolver::SetRelativeTolerance(real_t relTol)
{
	m_dRelTolerance = relTol;
}

real_t daeIDASolver::GetRelativeTolerance(void) const
{
	return m_dRelTolerance;
}

void daeIDASolver::Initialize(daeBlock_t* pBlock, daeLog_t* pLog, daeeInitialConditionMode eMode)
{
	if(!pBlock)
		daeDeclareAndThrowException(exInvalidPointer);

	m_pLog					= pLog;
	m_pBlock				= pBlock;
	m_eInitialConditionMode = eMode;
	
	m_nNumberOfEquations = m_pBlock->GetNumberOfEquations();

// Create data IDA vectors etc
	CreateArrays();

// Setting initial conditions and initial values, and set rel. and abs. tolerances
	Set_InitialConditions_InitialGuesses_AbsRelTolerances();

// Create IDA structure 
	CreateIDA();

// Create linear solver 
	CreateLinearSolver();
	
// Set root function
	RefreshRootFunctions();
}

void daeIDASolver::CreateArrays(void)
{
	m_pIDASolverData->CreateSerialArrays(m_nNumberOfEquations);
}

void daeIDASolver::Set_InitialConditions_InitialGuesses_AbsRelTolerances(void)
{
	realtype *pVariableValues, *pTimeDerivatives, *pInitialConditionsTypes, *pAbsoluteTolerances;

	pVariableValues         = NV_DATA_S(m_pIDASolverData->m_vectorVariables);
	pTimeDerivatives        = NV_DATA_S(m_pIDASolverData->m_vectorTimeDerivatives);
	pInitialConditionsTypes = NV_DATA_S(m_pIDASolverData->m_vectorInitialConditionsTypes);

	memset(pVariableValues,         0, sizeof(real_t) * m_nNumberOfEquations);
	memset(pTimeDerivatives,        0, sizeof(real_t) * m_nNumberOfEquations);
	memset(pInitialConditionsTypes, 0, sizeof(real_t) * m_nNumberOfEquations);

	m_arrValues.InitArray(m_nNumberOfEquations, pVariableValues);
	m_arrTimeDerivatives.InitArray(m_nNumberOfEquations, pTimeDerivatives);

/* I have to fill initial values of:
             - Variables
			 - Time derivatives
			 - Variable IC types (that is whether the initial conditions are algebraic or differential)
   from the model.
   To do so first I set arrays into which I have to copy and then actually copy values,
   by the function SetInitialConditionsAndInitialGuesses().
*/
	daeDenseArray arrInitialConditionsTypes;
	arrInitialConditionsTypes.InitArray(m_nNumberOfEquations, pInitialConditionsTypes);

	m_pBlock->SetInitialConditionsAndInitialGuesses(m_arrValues, m_arrTimeDerivatives, arrInitialConditionsTypes);

// Absolute tolerances
	pAbsoluteTolerances = NV_DATA_S(m_pIDASolverData->m_vectorAbsTolerances);
	memset(pAbsoluteTolerances, 0, sizeof(real_t) * m_nNumberOfEquations);

	daeDenseArray arrAbsoluteTolerances;
	arrAbsoluteTolerances.InitArray(m_nNumberOfEquations, pAbsoluteTolerances);
	m_pBlock->FillAbsoluteTolerancesArray(arrAbsoluteTolerances);
}

void daeIDASolver::CreateIDA(void)
{
	int	retval;

	m_pIDA = IDACreate();
	if(!m_pIDA) 
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to allocate m_pIDA pointer";
		throw e;
	}

	if(m_eInitialConditionMode == eAlgebraicValuesProvided)
	{
		retval = IDASetId(m_pIDA, m_pIDASolverData->m_vectorInitialConditionsTypes);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Unable to set initial condition types; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	}
	
	retval = IDAInit(m_pIDA, 
					 residuals, 
					 m_timeStart, 
					 m_pIDASolverData->m_vectorVariables, 
					 m_pIDASolverData->m_vectorTimeDerivatives);
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to alloc IDA struct; " << CreateIDAErrorMessage(retval);
		throw e;
	}
	
	retval = IDASVtolerances(m_pIDA, m_dRelTolerance, m_pIDASolverData->m_vectorAbsTolerances);
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to set tolerances; " << CreateIDAErrorMessage(retval);
		throw e;
	}

	retval = IDASetUserData(m_pIDA, this);
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to set residual data; " << CreateIDAErrorMessage(retval);
		throw e;
	}
}

void daeIDASolver::CreateLinearSolver(void)
{
	int	retval;

	if(m_eLASolver == eSundialsLU)
	{
	// Sundials dense LU LA Solver	
		retval = IDADense(m_pIDA, (long)m_nNumberOfEquations);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exRuntimeCheck);
			e << "Unable to create Sundials dense linear solver; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	
		retval = IDADlsSetDenseJacFn(m_pIDA, jacobian);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exRuntimeCheck);
			e << "Unable to set Jacobian function for Sundials dense linear solver; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	}
	else if(m_eLASolver == eSundialsGMRES)
	{
		daeDeclareAndThrowException(exNotImplemented)
		
	// Sundials dense GMRES LA Solver	
		retval = IDASpgmr(m_pIDA, 20);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exRuntimeCheck);
			e << "Unable to create Sundials GMRES linear solver; " << CreateIDAErrorMessage(retval);
			throw e;
		}
		
		retval = IDASpilsSetPreconditioner(m_pIDA, setup_preconditioner, solve_preconditioner);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exRuntimeCheck);
			e << "Unable to set preconditioner functions for Sundials GMRES linear solver; " << CreateIDAErrorMessage(retval);
			throw e;
		}

		m_pIDASolverData->CreatePreconditionerArrays(m_nNumberOfEquations);
	}
	else if(m_eLASolver == eThirdParty)
	{
		if(m_pLASolver == NULL)
		{
			daeDeclareException(exRuntimeCheck);
			e << "Third party has not been specified";
			throw e;
		}
	// Third party LA Solver	
		retval = m_pLASolver->Create(m_pIDA, m_nNumberOfEquations, this);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exRuntimeCheck);
			e << "Unable to create third party linear solver; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	}
	else
	{
		daeDeclareException(exRuntimeCheck);
		e << "Unspecified linear solver";
		throw e;		
	}
}

void daeIDASolver::RefreshRootFunctions(void)
{
	int	retval;
	size_t nNoRoots;

	if(!m_pBlock)
		daeDeclareAndThrowException(exInvalidPointer);

	nNoRoots = m_pBlock->GetNumberOfRoots();

	retval = IDARootInit(m_pIDA, (int)nNoRoots, roots);
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to initialize roots; " << CreateIDAErrorMessage(retval);
		throw e;
	}
}

void daeIDASolver::SolveInitial(void)
{
	int retval = -1;

	if(!m_pLog || !m_pBlock || m_nNumberOfEquations == 0)
	{
		daeDeclareException(exMiscellanous);
		e << "IDA Solver has not been initialized";
		throw e;
	}
	
	if(m_eInitialConditionMode == eAlgebraicValuesProvided)
		retval = IDACalcIC(m_pIDA, IDA_YA_YDP_INIT, m_dNextTimeAfterReinitialization);
	else if(m_eInitialConditionMode == eSteadyState)
		retval = IDACalcIC(m_pIDA, IDA_Y_INIT, m_dNextTimeAfterReinitialization);
	else
		daeDeclareAndThrowException(exNotImplemented);
	
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to initialize the system at TIME = 0; " << CreateIDAErrorMessage(retval);
		throw e;
	}
}

void daeIDASolver::Reset(void)
{
	ResetIDASolver(true, 0);
}

void daeIDASolver::Reinitialize(bool bCopyDataFromBlock)
{
	string strMessage = "Reinitializing at time: " + toString<real_t>(m_dCurrentTime);
	m_pLog->Message(strMessage, 0);

	ResetIDASolver(bCopyDataFromBlock, m_dCurrentTime);	
}

void daeIDASolver::ResetIDASolver(bool bCopyDataFromBlock, real_t t0)
{
	int retval;
	size_t N;
	realtype *pdValues, *pdTimeDerivatives;

	if(!m_pLog || !m_pBlock || m_nNumberOfEquations == 0)
	{
		daeDeclareException(exMiscellanous);
		e << "IDA Solver has not been initialized";
		throw e;
	}
	
// Set the current time
	m_dCurrentTime = t0;
	
// Copy data from the block if requested
	if(bCopyDataFromBlock)
	{
		N = m_pBlock->GetNumberOfEquations();
	
		pdValues			= NV_DATA_S(m_pIDASolverData->m_vectorVariables); 
		pdTimeDerivatives	= NV_DATA_S(m_pIDASolverData->m_vectorTimeDerivatives); 
	
		m_arrValues.InitArray(N, pdValues);
		m_arrTimeDerivatives.InitArray(N, pdTimeDerivatives);
	
		m_pBlock->CopyValuesToSolver(m_arrValues);
		m_pBlock->CopyTimeDerivativesToSolver(m_arrTimeDerivatives);
	}
	
// I should reset LA solver, since (the most likely) 
// sparse matrix pattern has been changed after the discontinuity
	if(m_eLASolver == eThirdParty)
		m_pLASolver->Reinitialize(m_pIDA);
	
// ReInit IDA
	retval = IDAReInit(m_pIDA,
					   m_dCurrentTime, 
					   m_pIDASolverData->m_vectorVariables, 
					   m_pIDASolverData->m_vectorTimeDerivatives);
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to re-init IDA solver at TIME = " << m_dCurrentTime << "; " 
		  << CreateIDAErrorMessage(retval);
		throw e;
	}
	
	m_pBlock->SetCurrentTime(t0);
	
// Calculate initial conditions again
// Here we should not use eSteadyState but ONLY eAlgebraicValuesProvided since we already have results
// Check this !!!!
	if(m_eInitialConditionMode == eAlgebraicValuesProvided)
		retval = IDACalcIC(m_pIDA, IDA_YA_YDP_INIT, m_dCurrentTime + m_dNextTimeAfterReinitialization);
	else if(m_eInitialConditionMode == eSteadyState)
		retval = IDACalcIC(m_pIDA, IDA_Y_INIT, m_dCurrentTime + m_dNextTimeAfterReinitialization);
	else
		daeDeclareAndThrowException(exNotImplemented);
	
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to re-initialize the system at TIME = " << m_dCurrentTime << "; " 
		  << CreateIDAErrorMessage(retval);
		throw e;
	}
}

real_t daeIDASolver::Solve(real_t dTime, daeeStopCriterion eCriterion)
{
 	int retval, retvalr;
	int* rootsfound;
	size_t nNoRoots;
	daeeDiscontinuityType eDiscontinuityType;

	if(!m_pLog || !m_pBlock || m_nNumberOfEquations == 0)
	{
		daeDeclareException(exMiscellanous);
		e << "IDA Solver has not been initialized";
		throw e;
	}
	
	if(dTime <= m_dCurrentTime)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid time horizon specified [" << dTime << "]; it must be higher than [" << m_dCurrentTime << "]";
		throw e;
	}
	m_dTargetTime = dTime;

	for(;;)
	{
		retval = IDASetStopTime(m_pIDA, m_dTargetTime);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Unable to set time horizon (the stop time) at TIME = " << m_dCurrentTime << "; " << CreateIDAErrorMessage(retval);
			throw e;
		}

		retval = IDASolve(m_pIDA, m_dTargetTime, &m_dCurrentTime, m_pIDASolverData->m_vectorVariables, m_pIDASolverData->m_vectorTimeDerivatives, IDA_TSTOP_RETURN);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Integration failed at TIME = " << m_dCurrentTime << "; time horizon [" << m_dTargetTime << "]; " << CreateIDAErrorMessage(retval);
			throw e;
		}

		if(retval == IDA_ROOT_RETURN) 
		{
			nNoRoots = m_pBlock->GetNumberOfRoots();
			rootsfound = new int[nNoRoots];		
			retvalr = IDAGetRootInfo(m_pIDA, rootsfound);
			if(!CheckFlag(retval)) 
			{
				daeDeclareException(exMiscellanous);
				e << "Get root info failed; " << CreateIDAErrorMessage(retval);
				throw e;
			}
			delete[] rootsfound;

			eDiscontinuityType = m_pBlock->CheckDiscontinuities();
			if(eDiscontinuityType == eModelDiscontinuity)
			{
				RefreshRootFunctions();
				Reinitialize(false);
				if(eCriterion == eStopAtModelDiscontinuity)
					return m_dCurrentTime;
			}
			else if(eDiscontinuityType == eGlobalDiscontinuity)
			{
				if(eCriterion == eStopAtGlobalDiscontinuity)
					return m_dCurrentTime;
			}
		}

		if(m_dCurrentTime == m_dTargetTime) 
			break;
	}
	
	return m_dCurrentTime;
}

daeBlock_t* daeIDASolver::GetBlock(void) const
{
	return m_pBlock;
}

daeLog_t* daeIDASolver::GetLog(void) const
{
	return m_pLog;
}

bool daeIDASolver::CheckFlag(int flag)
{
	if(flag < 0)
		return false;
	else
		return true;
}

string daeIDASolver::CreateIDAErrorMessage(int flag)
{
	string strError;

	switch(flag)
	{
		case IDA_MEM_NULL:
			strError = "IDA_MEM_NULL";
			break;
		case IDA_ILL_INPUT:
			strError = "IDA_ILL_INPUT";
			break;
		case IDA_NO_MALLOC:
			strError = "IDA_NO_MALLOC";
			break;
		case IDA_TOO_MUCH_WORK:
			strError = "IDA_TOO_MUCH_WORK";
			break;
		case IDA_TOO_MUCH_ACC:
			strError = "IDA_TOO_MUCH_ACC";
			break;
		case IDA_ERR_FAIL:
			strError = "IDA_ERR_FAIL";
			break;
		case IDA_CONV_FAIL:
			strError = "IDA_CONV_FAIL";
			break;
		case IDA_LINIT_FAIL:
			strError = "IDA_LINIT_FAIL";
			break;
		case IDA_LSETUP_FAIL:
			strError = "IDA_LSETUP_FAIL";
			break;
		case IDA_LSOLVE_FAIL:
			strError = "IDA_LSOLVE_FAIL";
			break;
		case IDA_RES_FAIL:
			strError = "IDA_RES_FAIL";
			break;
		case IDA_CONSTR_FAIL:
			strError = "IDA_CONSTR_FAIL";
			break;
		case IDA_REP_RES_ERR:
			strError = "IDA_REP_RES_ERR";
			break;
		case IDA_MEM_FAIL:
			strError = "IDA_MEM_FAIL";
			break;
		case IDA_BAD_T:
			strError = "IDA_BAD_T";
			break;
		case IDA_BAD_EWT:
			strError = "IDA_BAD_EWT";
			break;
		case IDA_FIRST_RES_FAIL:
			strError = "IDA_FIRST_RES_FAIL";
			break;
		case IDA_LINESEARCH_FAIL:
			strError = "IDA_LINESEARCH_FAIL";
			break;
		case IDA_NO_RECOVERY:
			strError = "IDA_NO_RECOVERY";
			break;
		case IDA_RTFUNC_FAIL:
			strError = "IDA_RTFUNC_FAIL";
			break;
	default:
		strError = "Unknown error";
	}

	return strError;
} 


//void daeIDASolver::GetSparseMatrixData(int& nnz, int** ia, int** ja)
//{
//	nnz = 0;
//	*ia = NULL;
//	*ja = NULL;
//	
//	if(m_eLASolver != eSparse_MKL_PARDISO_LU)
//		return;
//	
//#ifdef HAS_INTEL_MKL
//	IDA_sparse_MKL_PARDISO_LU_Get_Matrix_Data(m_pIDA, nnz, ia, ja);
//#endif
//}


void daeIDASolver::SaveMatrixAsXPM(const std::string& strFilename)
{
	if(!m_pLASolver)
		return;
	m_pLASolver->SaveAsXPM(strFilename);
}

void daeIDASolver::SaveMatrixAsPBM(const std::string& strFilename)
{
	if(!m_pLASolver)
		return;
	m_pLASolver->SaveAsPBM(strFilename);
}

daeeInitialConditionMode daeIDASolver::GetInitialConditionMode(void) const
{
	return m_eInitialConditionMode;
}

void daeIDASolver::SetInitialConditionMode(daeeInitialConditionMode eMode)
{
	m_eInitialConditionMode = eMode;
}


int residuals(realtype	time, 
			  N_Vector	vectorVariables, 
			  N_Vector	vectorTimeDerivatives, 
			  N_Vector	vectorResiduals, 
			  void*		pUserData)
{
	realtype *pdValues, *pdTimeDerivatives, *pdResiduals;

	daeIDASolver* pSolver = (daeIDASolver*)pUserData;
	if(!pSolver || !pSolver->m_pIDASolverData)
		return -1;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return -1;

	size_t N = pBlock->GetNumberOfEquations();

	pdValues			= NV_DATA_S(vectorVariables); 
	pdTimeDerivatives	= NV_DATA_S(vectorTimeDerivatives); 
	pdResiduals			= NV_DATA_S(vectorResiduals);

	pSolver->m_arrValues.InitArray(N, pdValues);
	pSolver->m_arrTimeDerivatives.InitArray(N, pdTimeDerivatives);
	pSolver->m_arrResiduals.InitArray(N, pdResiduals);

	if(time == 0)
		pBlock->CheckDiscontinuities();

	pBlock->CalculateResiduals(time, 
		                       pSolver->m_arrValues, 
							   pSolver->m_arrResiduals, 
							   pSolver->m_arrTimeDerivatives);

	return 0;
}

int roots(realtype	time, 
		  N_Vector	vectorVariables, 
		  N_Vector	vectorTimeDerivatives, 
		  realtype*	gout, 
		  void*		pUserData)
{
	realtype *pdValues, *pdTimeDerivatives;
	vector<real_t> arrResults;
	
	daeIDASolver* pSolver = (daeIDASolver*)pUserData;
	if(!pSolver || !pSolver->m_pIDASolverData)
		return -1;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return -1;

	size_t Nroots = pBlock->GetNumberOfRoots();
	arrResults.resize(Nroots);

	size_t N = pBlock->GetNumberOfEquations();

	pdValues			= NV_DATA_S(vectorVariables); 
	pdTimeDerivatives	= NV_DATA_S(vectorTimeDerivatives); 

	pSolver->m_arrValues.InitArray(N, pdValues);
	pSolver->m_arrTimeDerivatives.InitArray(N, pdTimeDerivatives);

	pBlock->CalculateConditions(time, 
								pSolver->m_arrValues, 
								pSolver->m_arrTimeDerivatives, 
								arrResults);

	for(size_t i = 0; i < Nroots; i++)
		gout[i] = arrResults[i];

	return 0;
}

int jacobian(int	    Neq, 
			 realtype	time, 
			 realtype	dInverseTimeStep, 
			 N_Vector	vectorVariables, 
			 N_Vector	vectorTimeDerivatives,
			 N_Vector	vectorResiduals, 
			 DlsMat		dense_matrixJacobian,
			 void*		pUserData, 
			 N_Vector	vectorTemp1, 
			 N_Vector	vectorTemp2, 
			 N_Vector	vectorTemp3)
{
	realtype *pdValues, *pdTimeDerivatives, *pdResiduals, **ppdJacobian;

	daeIDASolver* pSolver = (daeIDASolver*)pUserData;
	if(!pSolver || !pSolver->m_pIDASolverData)
		return -1;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return -1;

	size_t N = pBlock->GetNumberOfEquations();

	pdValues			= NV_DATA_S(vectorVariables); 
	pdTimeDerivatives	= NV_DATA_S(vectorTimeDerivatives); 
	pdResiduals			= NV_DATA_S(vectorResiduals);
	ppdJacobian			= JACOBIAN(dense_matrixJacobian);

	pSolver->m_arrValues.InitArray(N, pdValues);
	pSolver->m_arrTimeDerivatives.InitArray(N, pdTimeDerivatives);
	pSolver->m_arrResiduals.InitArray(N, pdResiduals);
	pSolver->m_matJacobian.InitMatrix(N, ppdJacobian, eColumnWise);

	pBlock->CalculateJacobian(time, 
		                      pSolver->m_arrValues, 
							  pSolver->m_arrResiduals, 
							  pSolver->m_arrTimeDerivatives, 
							  pSolver->m_matJacobian, 
							  dInverseTimeStep);
	
	return 0;
}

int setup_preconditioner(realtype	time, 
						 N_Vector	vectorVariables, 
						 N_Vector	vectorTimeDerivatives,
						 N_Vector	vectorResiduals, 
						 realtype	dInverseTimeStep, 
						 void*		pUserData, 
						 N_Vector	vectorTemp1, 
						 N_Vector	vectorTemp2, 
						 N_Vector	vectorTemp3)
{
	realtype *pdValues, *pdTimeDerivatives, *pdResiduals, **ppdJacobian;

	daeIDASolver* pSolver = (daeIDASolver*)pUserData;
	if(!pSolver || !pSolver->m_pIDASolverData)
		return -1;
	if(!pSolver->m_pIDASolverData->m_matKrylov || !pSolver->m_pIDASolverData->m_vectorPivot)
		return -1;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return -1;

	size_t Neq = pBlock->GetNumberOfEquations();

	pdValues			= NV_DATA_S(vectorVariables); 
	pdTimeDerivatives	= NV_DATA_S(vectorTimeDerivatives); 
	pdResiduals			= NV_DATA_S(vectorResiduals);
	ppdJacobian			= JACOBIAN(pSolver->m_pIDASolverData->m_matKrylov);

	pSolver->m_arrValues.InitArray(Neq, pdValues);
	pSolver->m_arrTimeDerivatives.InitArray(Neq, pdTimeDerivatives);
	pSolver->m_arrResiduals.InitArray(Neq, pdResiduals);
	pSolver->m_matJacobian.InitMatrix(Neq, ppdJacobian, eColumnWise);
	
	SetToZero(pSolver->m_pIDASolverData->m_matKrylov);

	pBlock->CalculateJacobian(time, 
		                      pSolver->m_arrValues, 
							  pSolver->m_arrResiduals, 
							  pSolver->m_arrTimeDerivatives, 
							  pSolver->m_matJacobian, 
							  dInverseTimeStep);
	pSolver->m_pIDASolverData->SetMaxElements();
	pSolver->m_matJacobian.Print();

	daeDenseArray arr;
	arr.InitArray(Neq, pSolver->m_pIDASolverData->m_vectorInvMaxElements);
	std::cout << "setup_preconditioner" << std::endl;
	arr.Print();
	
	return 0;
	
	//return DenseGETRF(pSolver->m_pIDASolverData->m_matKrylov, pSolver->m_pIDASolverData->m_vectorPivot);
}

int solve_preconditioner(realtype	time, 
						 N_Vector	vectorVariables, 
						 N_Vector	vectorTimeDerivatives,
						 N_Vector	vectorResiduals, 
						 N_Vector	vectorR, 
						 N_Vector	vectorZ, 
						 realtype	dInverseTimeStep,
						 realtype	delta,
						 void*		pUserData, 
						 N_Vector	vectorTemp)
{
	realtype *pdR, *pdZ;

	daeIDASolver* pSolver = (daeIDASolver*)pUserData;
	if(!pSolver || !pSolver->m_pIDASolverData)
		return -1;
	if(!pSolver->m_pIDASolverData->m_matKrylov || !pSolver->m_pIDASolverData->m_vectorPivot)
		return -1;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return -1;

	size_t Neq = pBlock->GetNumberOfEquations();

	pdR			= NV_DATA_S(vectorR); 
	pdZ			= NV_DATA_S(vectorZ);

	daeDenseArray r, z;
	r.InitArray(Neq, pdR);
	std::cout << "r" << std::endl;
	r.Print();

	for(size_t i = 0; i < Neq; i++)
		pdZ[i] = pdR[i] * pSolver->m_pIDASolverData->m_vectorInvMaxElements[i];
	std::cout << "z" << std::endl;
	z.InitArray(Neq, pdZ);
	z.Print();
//	
//	::memcpy(pdZ, pdR, Neq*sizeof(realtype));
//	
//	DenseGETRS(pSolver->m_pIDASolverData->m_matKrylov, pSolver->m_pIDASolverData->m_vectorPivot, pdZ);
	
	return 0;
}


}
}

