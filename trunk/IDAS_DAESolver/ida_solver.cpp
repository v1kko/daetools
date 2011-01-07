#include "stdafx.h"
using namespace std;
#include "ida_solver.h"
#include "../Core/helpers.h"
#include "dae_solvers.h"
#include <idas/idas.h>
#include <idas/idas_impl.h>
#include <idas/idas_dense.h>
//#include <idas/idas_lapack.h>
#include <idas/idas_spgmr.h>

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

int sens_residuals(int			Ns, 
				   realtype		time, 
				   N_Vector		vectorVariables, 
				   N_Vector		vectorTimeDerivatives, 
				   N_Vector		vectorResVal, /* WTF ?? */ 
				   N_Vector*	pvectorSVariables, 
				   N_Vector*	pvectorSTimeDerivatives, 
				   N_Vector*	pvectorResidualValues,
				   void*		pUserData,
				   N_Vector		vectorTemp1, 
				   N_Vector		vectorTemp2, 
				   N_Vector		vectorTemp3);

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
	m_timeStart				         = 0;
	m_dTargetTime			         = 0;
	m_bCalculateSensitivities        = false;
	m_bIsModelDynamic				 = false;
	m_eInitialConditionMode          = eAlgebraicValuesProvided;
	
	m_pIDASolverData.reset(new daeIDASolverData);

	daeConfig& cfg = daeConfig::GetConfig();
	m_dRelTolerance                  = cfg.Get<real_t>("daetools.IDAS.relativeTolerance",             1e-5);
	m_dNextTimeAfterReinitialization = cfg.Get<real_t>("daetools.IDAS.nextTimeAfterReinitialization", 1e-2);
	m_strSensitivityMethod           = cfg.Get<string>("daetools.IDAS.sensitivityMethod",             "Simultaneous");
	m_bErrorControl                  = cfg.Get<bool>  ("daetools.IDAS.errorControl",                  false);
	m_bPrintInfo                     = cfg.Get<bool>  ("daetools.IDAS.printInfo",                     false);
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

size_t daeIDASolver::GetNumberOfVariables(void) const
{
	return m_nNumberOfEquations;
}

void daeIDASolver::SetRelativeTolerance(real_t relTol)
{
	m_dRelTolerance = relTol;
}

real_t daeIDASolver::GetRelativeTolerance(void) const
{
	return m_dRelTolerance;
}

void daeIDASolver::Initialize(daeBlock_t* pBlock, 
							  daeLog_t* pLog, 
							  daeeInitialConditionMode eMode, 
							  bool bCalculateSensitivities,
							  const std::vector<size_t>& narrParametersIndexes)
{
	if(!pBlock)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pLog)
		daeDeclareAndThrowException(exInvalidPointer);

	m_pLog					= pLog;
	m_pBlock				= pBlock;
	m_eInitialConditionMode = eMode;
	
	m_nNumberOfEquations      = m_pBlock->GetNumberOfEquations();

// Create data IDA vectors etc
	CreateArrays();

// Setting initial conditions and initial values, and set rel. and abs. tolerances
// and determine if the model is dynamic or steady-state
	Set_InitialConditions_InitialGuesses_AbsRelTolerances();

// Create IDA structure 
	CreateIDA();

// Sensitivity related data	
	m_bCalculateSensitivities = bCalculateSensitivities;	
	m_narrParametersIndexes   = narrParametersIndexes;

	if(bCalculateSensitivities && narrParametersIndexes.empty())
	{
		daeDeclareException(exInvalidPointer);
		e << "In order to calculate sensitivities the list of parameters must be given.";
		throw e;
	}

	if(m_bCalculateSensitivities)
		SetupSensitivityCalculation();
	
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

// Determine if the model is dynamic or steady-state
	m_bIsModelDynamic = m_pBlock->IsModelDynamic();
	m_pLog->Message(string("The model is ") + string(m_bIsModelDynamic ? "dynamic" : "steady-state"), 0);

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
		e << "Sundials IDAS solver faint-heartedly failed to allocate a memory for itself";
		throw e;
	}

	retval = IDASetId(m_pIDA, m_pIDASolverData->m_vectorInitialConditionsTypes);
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Sundials IDAS solver faint-heartedly failed to set initial condition types; " << CreateIDAErrorMessage(retval);
		throw e;
	}
		
	retval = IDAInit(m_pIDA, 
					 residuals, 
					 m_timeStart, 
					 m_pIDASolverData->m_vectorVariables, 
					 m_pIDASolverData->m_vectorTimeDerivatives);
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Sundials IDAS solver faint-heartedly failed to allocate IDA structure; " << CreateIDAErrorMessage(retval);
		throw e;
	}
	
	retval = IDASVtolerances(m_pIDA, m_dRelTolerance, m_pIDASolverData->m_vectorAbsTolerances);
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Sundials IDAS solver faint-heartedly refused to set tolerances; " << CreateIDAErrorMessage(retval);
		throw e;
	}

	retval = IDASetUserData(m_pIDA, this);
	if(!CheckFlag(retval)) 
	{
		daeDeclareException(exMiscellanous);
		e << "Sundials IDAS solver faint-heartedly refused to set residual data; " << CreateIDAErrorMessage(retval);
		throw e;
	}
}

void daeIDASolver::SetupSensitivityCalculation(void)
{
	int	retval;
	int iSensMethod;
	
	size_t Ns = m_narrParametersIndexes.size();

	if(!m_bCalculateSensitivities || Ns == 0)
	{
		daeDeclareException(exInvalidCall);
		e << "Sundials IDAS solver abjectly refused to enable sensitivity calculation: it has not been enabled";
		throw e;
	}
		
// If the model is dynamic then matrixes S, SD and Sres will be created
// Otherwise only the S matrix will be created
	m_pIDASolverData->CreateSensitivityArrays(Ns, m_bIsModelDynamic);
	
// Only if the model is dynamic setup IDAS to create sensitivities
// Otherwise the gradients will be calculated after the call tof SolveInitial()
	if(m_bIsModelDynamic)
	{
		if(m_strSensitivityMethod == "Simultaneous")
			iSensMethod = IDA_SIMULTANEOUS;
		else
			iSensMethod = IDA_STAGGERED;
		
		retval = IDASensInit(m_pIDA, Ns, iSensMethod, sens_residuals, m_pIDASolverData->m_pvectorSVariables, m_pIDASolverData->m_pvectorSTimeDerivatives);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Sundials IDAS solver abjectly refused to setup sensitivity calculation; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	
		retval = IDASensEEtolerances(m_pIDA);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Sundials IDAS solver abjectly refused to setup sensitivity error tolerances; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	
		retval = IDASetSensErrCon(m_pIDA, m_bErrorControl);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Sundials IDAS solver abjectly refused to setup sensitivity error control; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	
	//  retval = IDASetSensParams(m_pIDA, data->p, pbar, NULL);
	//	if(!CheckFlag(retval)) 
	//	{
	//		daeDeclareException(exMiscellanous);
	//		e << "Unable to setup sensitivity parameters; " << CreateIDAErrorMessage(retval);
	//		throw e;
	//	}
	}
}


daeMatrix<real_t>& daeIDASolver::GetSensitivities(void)
{
	int	retval;

	if(!m_bCalculateSensitivities)
		daeDeclareAndThrowException(exInvalidCall)
		
	if(m_bCalculateSensitivities && m_bIsModelDynamic)
	{
		retval = IDAGetSens(m_pIDA, &m_dCurrentTime, m_pIDASolverData->m_pvectorSVariables);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Sundials IDAS solver spinelessly failed to return sensitivities; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	}
	
	if(!m_pIDASolverData->ppdSValues)
		daeDeclareAndThrowException(exInvalidPointer)
	if(!m_pIDASolverData->m_pvectorSVariables)
		daeDeclareAndThrowException(exInvalidPointer)
	
	size_t N  = m_pIDASolverData->m_N;
	size_t Ns = m_pIDASolverData->m_Ns;

	for(int i = 0; i < Ns; i++)
		m_pIDASolverData->ppdSValues[i] = NV_DATA_S(m_pIDASolverData->m_pvectorSVariables[i]);
	
	m_matSValues.InitMatrix(Ns, N, m_pIDASolverData->ppdSValues, eRowWise);

/*	
	realtype* pdSValues;
	if(m_bIsModelDynamic)
		std::cout << "Sensitivities at the time: " << m_dCurrentTime << std::endl;
	else
		std::cout << "Gradients: " << std::endl;
		
	for(size_t i = 0; i < Ns; i++)
	{
		pdSValues = NV_DATA_S(m_pIDASolverData->m_pvectorSVariables[i]);
		if(!pdSValues)
			daeDeclareAndThrowException(exInvalidCall);
		
		for(size_t k = 0; k < m_nNumberOfEquations; k++)
			std::cout << toStringFormatted<real_t>(pdSValues[k], 14, 4, true);
		std::cout << std::endl;
	}
*/	
	return m_matSValues;
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
			e << "Sundials IDAS solver ignobly refused to create Sundials dense linear solver; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	
		retval = IDADlsSetDenseJacFn(m_pIDA, jacobian);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exRuntimeCheck);
			e << "Sundials IDAS solver ignobly refused to set Jacobian function for Sundials dense linear solver; " << CreateIDAErrorMessage(retval);
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
			e << "Sundials IDAS solver ignobly refused to create Sundials GMRES linear solver; " << CreateIDAErrorMessage(retval);
			throw e;
		}
		
		retval = IDASpilsSetPreconditioner(m_pIDA, setup_preconditioner, solve_preconditioner);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exRuntimeCheck);
			e << "Sundials IDAS solver ignobly refused to set preconditioner functions for Sundials GMRES linear solver; " << CreateIDAErrorMessage(retval);
			throw e;
		}

		m_pIDASolverData->CreatePreconditionerArrays(m_nNumberOfEquations);
	}
	else if(m_eLASolver == eThirdParty)
	{
		if(m_pLASolver == NULL)
		{
			daeDeclareException(exRuntimeCheck);
			e << "Sundials IDAS solver ignobly refused to set the third party linear solver: the solver object is null";
			throw e;
		}
	// Third party LA Solver	
		retval = m_pLASolver->Create(m_pIDA, m_nNumberOfEquations, this);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exRuntimeCheck);
			e << "The third party linear solver chickenly failed to initialize it self; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	}
	else
	{
		daeDeclareException(exRuntimeCheck);
		e << "Sundials IDAS solver abjectly refused to set an unspecified linear solver";
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
		e << "Sundials IDAS solver timidly failed to initialize roots' functions; " << CreateIDAErrorMessage(retval);
		throw e;
	}
}

void daeIDASolver::SolveInitial(void)
{
	int retval = -1;

	if(!m_pLog || !m_pBlock || m_nNumberOfEquations == 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Sundials IDAS solver timidly refuses to solve initial: the solver has not been initialized";
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
		e << "Sundials IDAS solver cowardly failed to initialize the system at TIME = 0; " << CreateIDAErrorMessage(retval);
		throw e;
	}

// Here I have a problem. If the model is steady-state then IDACalcIC does nothing!
// The system is not solved!! I have to do something?
// I can run the system for a small period of time and then return (here only for one step: IDA_ONE_STEP).
// But what if IDASolve fails?
	if(!m_bIsModelDynamic)
	{
		retval = IDASolve(m_pIDA, 
						  m_dNextTimeAfterReinitialization, 
						  &m_dCurrentTime, 
						  m_pIDASolverData->m_vectorVariables, 
						  m_pIDASolverData->m_vectorTimeDerivatives, 
						  IDA_ONE_STEP);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Sundials IDAS solver cowardly failed to initialize the system at TIME = 0; " << CreateIDAErrorMessage(retval);
			throw e;
		}
	}
	
// Only if sensitivities are enabled and the model is steady-state calculate gradients
// and store the results in the same matrix: SValues[Ns][Neq]
	if(m_bCalculateSensitivities && !m_bIsModelDynamic)
		CalculateGradients();
}

void daeIDASolver::Reset(void)
{
	ResetIDASolver(true, 0);
}

void daeIDASolver::Reinitialize(bool bCopyDataFromBlock)
{
	int retval;
	string strMessage = "Reinitializing at time: " + toString<real_t>(m_dCurrentTime);
	m_pLog->Message(strMessage, 0);

	ResetIDASolver(bCopyDataFromBlock, m_dCurrentTime);	
	
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
		e << "Sundials IDAS solver dastardly failed re-initialize the system at TIME = " << m_dCurrentTime << "; " 
		  << CreateIDAErrorMessage(retval);
		throw e;
	}
}

void daeIDASolver::ResetIDASolver(bool bCopyDataFromBlock, real_t t0)
{
	int retval;
	size_t N;
	realtype *pdValues, *pdTimeDerivatives;

	if(!m_pLog || !m_pBlock || m_nNumberOfEquations == 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Sundials IDAS solver timidly refuses to reset: the solver has not been initialized";
		throw e;
	}
	
// Set the current time
	m_dCurrentTime = t0;
	m_pBlock->SetTime(t0);
	
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
		e << "Sundials IDAS solver timidly refused to re-init at TIME = " << m_dCurrentTime << "; " 
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
		e << "Sundials IDAS solver cravenly refuses to solve the system: the solver has not been initialized";
		throw e;
	}
	
	if(dTime <= m_dCurrentTime)
	{
		daeDeclareException(exInvalidCall);
		e << "Sundials IDAS solver cravenly refuses to solve the system: an invalid time horizon specified [" 
		  << dTime << "]; it must be higher than [" << m_dCurrentTime << "]";
		throw e;
	}
	m_dTargetTime = dTime;

	for(;;)
	{
		retval = IDASetStopTime(m_pIDA, m_dTargetTime);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Sundials IDAS solver cravenly refused to set the stop time at TIME = " << m_dCurrentTime << "; " << CreateIDAErrorMessage(retval);
			throw e;
		}

		retval = IDASolve(m_pIDA, m_dTargetTime, &m_dCurrentTime, m_pIDASolverData->m_vectorVariables, m_pIDASolverData->m_vectorTimeDerivatives, IDA_TSTOP_RETURN);
		if(!CheckFlag(retval)) 
		{
			daeDeclareException(exMiscellanous);
			e << "Sundials IDAS solver cowardly failed to solve the system at TIME = " << m_dCurrentTime 
			  << "; time horizon [" << m_dTargetTime << "]; " << CreateIDAErrorMessage(retval);
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
				e << "Sundials IDAS solver cravenly failed to get roots' info; " << CreateIDAErrorMessage(retval);
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
		case IDA_BAD_EWT:
			strError = "IDA_BAD_EWT";
			break;
		case IDA_BAD_K:
			strError = "IDA_BAD_K";
			break;
		case IDA_BAD_T:
			strError = "IDA_BAD_T";
			break;
		case IDA_BAD_DKY:
			strError = "IDA_BAD_DKY";
			break;
		case IDA_NO_SENS:
			strError = "IDA_NO_SENS";
			break;
		case IDA_SRES_FAIL:
			strError = "IDA_SRES_FAIL";
			break;
		case IDA_REP_SRES_ERR:
			strError = "IDA_REP_SRES_ERR";
			break;
		case IDA_BAD_IS:
			strError = "IDA_BAD_IS";
			break;

		default:
			strError = "Unknown error";
	}

	return strError;
} 

void daeIDASolver::SaveMatrixAsXPM(const std::string& strFilename)
{
	if(!m_pLASolver)
		return;
	m_pLASolver->SaveAsXPM(strFilename);
}

daeeInitialConditionMode daeIDASolver::GetInitialConditionMode(void) const
{
	return m_eInitialConditionMode;
}

void daeIDASolver::SetInitialConditionMode(daeeInitialConditionMode eMode)
{
	m_eInitialConditionMode = eMode;
}

int daeIDASolver::CalculateGradients(void)
{
	realtype *pdValues, *pdTimeDerivatives;

	if(!m_bCalculateSensitivities || m_bIsModelDynamic)
		return -1;
	if(!m_pIDASolverData)
		return -1;
	if(!m_pBlock)
		return -1;

	size_t N  = m_pIDASolverData->m_N;
	size_t Ns = m_pIDASolverData->m_Ns;
	if(N == 0 || Ns == 0)
		return -1;
	
	pdValues			= NV_DATA_S(m_pIDASolverData->m_vectorVariables); 
	pdTimeDerivatives	= NV_DATA_S(m_pIDASolverData->m_vectorTimeDerivatives); 
	
	if(!m_pIDASolverData->ppdSValues)
		return -1;
	if(!m_pIDASolverData->m_pvectorSVariables)
		return -1;
		
	for(int i = 0; i < Ns; i++)
		m_pIDASolverData->ppdSValues[i] = NV_DATA_S(m_pIDASolverData->m_pvectorSVariables[i]);
	
	m_arrValues.InitArray(N, pdValues);
	m_arrTimeDerivatives.InitArray(N, pdTimeDerivatives);
	m_matSValues.InitMatrix(Ns, N, m_pIDASolverData->ppdSValues, eRowWise);
	
	m_pBlock->CalculateGradients(m_narrParametersIndexes,
							     m_arrValues, 
							     m_matSValues);

	if(m_bPrintInfo)
	{
		cout << "CalculateGradients function:" << endl;
		cout << "Gradients matrix:" << endl;
		m_matSValues.Print();
	}
	
	return 0;
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

	if(pSolver->m_bPrintInfo)
	{
		cout << "Residuals function:" << endl;
		cout << "Variable values:" << endl;
		pSolver->m_arrValues.Print();
	}
	
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
	pSolver->m_matJacobian.InitMatrix(N, N, ppdJacobian, eColumnWise);

	pBlock->CalculateJacobian(time, 
		                      pSolver->m_arrValues, 
							  pSolver->m_arrResiduals, 
							  pSolver->m_arrTimeDerivatives, 
							  pSolver->m_matJacobian, 
							  dInverseTimeStep);
	if(pSolver->m_bPrintInfo)
	{
		cout << "Jacobian function:" << endl;
		cout << "Variable values:" << endl;
		pSolver->m_arrValues.Print();
		cout << "Jacobian matrix:" << endl;
		pSolver->m_matJacobian.Print();
	}
	return 0;
}

int sens_residuals(int		 Ns, 
				   realtype	 time, 
				   N_Vector	 vectorVariables, 
				   N_Vector	 vectorTimeDerivatives, 
				   N_Vector	 vectorResVal, /* WTF ?? */ 
				   N_Vector* pvectorSVariables, 
				   N_Vector* pvectorSTimeDerivatives, 
				   N_Vector* pvectorResidualValues,
				   void*	 pUserData,
				   N_Vector	 vectorTemp1, 
				   N_Vector	 vectorTemp2, 
				   N_Vector	 vectorTemp3)
{
	realtype *pdValues, *pdTimeDerivatives;

	daeIDASolver* pSolver = (daeIDASolver*)pUserData;
	if(!pSolver || !pSolver->m_pIDASolverData)
		return -1;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return -1;

	size_t N  = pBlock->GetNumberOfEquations();
	if(N == 0 || Ns == 0)
		return -1;
	
	pdValues			= NV_DATA_S(vectorVariables); 
	pdTimeDerivatives	= NV_DATA_S(vectorTimeDerivatives); 
	
	if(!pSolver->m_pIDASolverData->ppdSValues      ||
	   !pSolver->m_pIDASolverData->ppdSDValues     ||
	   !pSolver->m_pIDASolverData->ppdSensResiduals )
		return -1;
		
	for(int i = 0; i < Ns; i++)
	{
		pSolver->m_pIDASolverData->ppdSValues[i]       = NV_DATA_S(pvectorSVariables[i]);
		pSolver->m_pIDASolverData->ppdSDValues[i]      = NV_DATA_S(pvectorSTimeDerivatives[i]);
		pSolver->m_pIDASolverData->ppdSensResiduals[i] = NV_DATA_S(pvectorResidualValues[i]);
	}
	
	pSolver->m_arrValues.InitArray         (N, pdValues);
	pSolver->m_arrTimeDerivatives.InitArray(N, pdTimeDerivatives);
	
	pSolver->m_matSValues.InitMatrix         (Ns, N, pSolver->m_pIDASolverData->ppdSValues,       eRowWise);
	pSolver->m_matSTimeDerivatives.InitMatrix(Ns, N, pSolver->m_pIDASolverData->ppdSDValues,      eRowWise);
	pSolver->m_matSResiduals.InitMatrix      (Ns, N, pSolver->m_pIDASolverData->ppdSensResiduals, eRowWise);
	
	pBlock->CalculateSensitivities(time, 
								   pSolver->m_narrParametersIndexes,
								   pSolver->m_arrValues, 
								   pSolver->m_arrTimeDerivatives,
								   pSolver->m_matSValues,
								   pSolver->m_matSTimeDerivatives,
								   pSolver->m_matSResiduals);

	if(pSolver->m_bPrintInfo)
	{
		cout << "Sensitivity residuals function:" << endl;
		cout << "S values:" << endl;
		pSolver->m_matSValues.Print();
		cout << "SD values:" << endl;
		pSolver->m_matSTimeDerivatives.Print();
		cout << "S residuals:" << endl;
		pSolver->m_matSResiduals.Print();
	}

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
	pSolver->m_matJacobian.InitMatrix(Neq, Neq, ppdJacobian, eColumnWise);
	
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

