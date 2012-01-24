#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{

daeBlock::daeBlock(void)
{
	m_bInitializeMode					= false;
	m_pDataProxy						= NULL;
	m_parrResidual						= NULL; 
	m_pmatJacobian						= NULL; 
	m_dCurrentTime						= 0;
	m_dInverseTimeStep					= 0;
	m_nNumberOfEquations                = 0;
	m_nCurrentVariableIndexForJacobianEvaluation = ULONG_MAX;

#if defined(DAE_MPI)
	m_nEquationIndexesStart = ULONG_MAX;
	m_nEquationIndexesEnd   = ULONG_MAX;
	m_nVariableIndexesStart = ULONG_MAX;
	m_nVariableIndexesEnd   = ULONG_MAX;
#endif
}

daeBlock::~daeBlock(void)
{
}

void daeBlock::Open(io::xmlTag_t* pTag)
{
	io::daeSerializable::Open(pTag);
}

void daeBlock::Save(io::xmlTag_t* pTag) const
{
	io::daeSerializable::Save(pTag);
}

void daeBlock::CalculateConditions(real_t				dTime, 
								   daeArray<real_t>&	arrValues, 
								   daeArray<real_t>&	arrTimeDerivatives, 
								   daeArray<real_t>&	arrResults)
{
	map<size_t, daeExpressionInfo>::iterator iter;

	SetTime(dTime);
	m_pDataProxy->SetCurrentTime(dTime);
	CopyDataFromSolver(arrValues, arrTimeDerivatives);

	daeExecutionContext EC;
	EC.m_pBlock						= this;
	EC.m_pDataProxy					= m_pDataProxy;
	EC.m_eEquationCalculationMode	= eCalculate;

	size_t nFnCounter = 0;
	for(iter = m_mapExpressionInfos.begin(); iter != m_mapExpressionInfos.end(); iter++)
	{
		arrResults[nFnCounter] = (*iter).second.m_pExpression->Evaluate(&EC).getValue();
		nFnCounter++;
	}
}

void daeBlock::CalculateResiduals(real_t			dTime, 
								  daeArray<real_t>& arrValues, 
								  daeArray<real_t>& arrResiduals, 
								  daeArray<real_t>& arrTimeDerivatives)
{
	size_t i;
	daeSTN* pSTN;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	SetTime(dTime);
	m_pDataProxy->SetCurrentTime(dTime);
	SetResidualArray(&arrResiduals);
	CopyDataFromSolver(arrValues, arrTimeDerivatives);

	daeExecutionContext EC;
	EC.m_pBlock						= this;
	EC.m_pDataProxy					= m_pDataProxy;
	EC.m_dInverseTimeStep			= GetInverseTimeStep();
	EC.m_pEquationExecutionInfo		= NULL;
	EC.m_eEquationCalculationMode	= eCalculate;
	
// First calculate normal equations (non-STN)
	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		pEquationExecutionInfo->Residual(EC);
	}

// Now calculate STN equations
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->CalculateResiduals(EC);
	}
}

void daeBlock::CalculateJacobian(real_t				dTime, 
								 daeArray<real_t>&	arrValues, 
								 daeArray<real_t>&	arrResiduals, 
								 daeArray<real_t>&	arrTimeDerivatives, 
								 daeMatrix<real_t>&	matJacobian, 
								 real_t				dInverseTimeStep)
{
	size_t i;
	daeSTN* pSTN;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	SetTime(dTime);
	m_pDataProxy->SetCurrentTime(dTime);
	SetResidualArray(&arrResiduals);
	SetJacobianMatrix(&matJacobian); 
	SetInverseTimeStep(dInverseTimeStep);
	CopyDataFromSolver(arrValues, arrTimeDerivatives);

	daeExecutionContext EC;
	EC.m_pBlock						= this;
	EC.m_pDataProxy					= m_pDataProxy;
	EC.m_dInverseTimeStep			= GetInverseTimeStep();
	EC.m_pEquationExecutionInfo		= NULL;
	EC.m_eEquationCalculationMode	= eCalculateJacobian;
	
// First calculate normal equations (non-STN)
	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		pEquationExecutionInfo->Jacobian(EC);
	}

// Now calculate STN equations
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->CalculateJacobian(EC);
	}
}

// For dynamic models
void daeBlock::CalculateSensitivityResiduals(real_t						dTime, 
											 const std::vector<size_t>& narrParameterIndexes,
											 daeArray<real_t>&			arrValues, 
											 daeArray<real_t>&			arrTimeDerivatives, 
											 daeMatrix<real_t>&			matSValues, 
											 daeMatrix<real_t>&			matSTimeDerivatives, 
											 daeMatrix<real_t>&			matSResiduals)
{
	size_t i;
	daeSTN* pSTN;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	SetTime(dTime);
	m_pDataProxy->SetCurrentTime(dTime);
	CopyDataFromSolver(arrValues, arrTimeDerivatives);
	
	m_pDataProxy->SetSensitivityMatrixes(&matSValues,
										 &matSTimeDerivatives,
										 &matSResiduals);
	
	daeExecutionContext EC;
	EC.m_pBlock						= this;
	EC.m_pDataProxy					= m_pDataProxy;
	EC.m_dInverseTimeStep			= GetInverseTimeStep();
	EC.m_pEquationExecutionInfo		= NULL;
	EC.m_eEquationCalculationMode	= eCalculateSensitivityResiduals;

	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		pEquationExecutionInfo->SensitivityResiduals(EC, narrParameterIndexes);
	}

// In general, neither objective function nor constraints can be within an STN
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->CalculateSensitivityResiduals(EC, narrParameterIndexes);
	}
	
	m_pDataProxy->ResetSensitivityMatrixes();
}

// For steady-state models
void daeBlock::CalculateSensitivityParametersGradients(const std::vector<size_t>& narrParameterIndexes,
													   daeArray<real_t>&		  arrValues, 
                                                       daeArray<real_t>&		  arrTimeDerivatives, 
													   daeMatrix<real_t>&		  matSResiduals)
{
	size_t i;
	daeSTN* pSTN;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	CopyDataFromSolver(arrValues, arrTimeDerivatives);
	m_pDataProxy->SetSensitivityMatrixes(NULL, NULL, &matSResiduals);
	
	daeExecutionContext EC;
	EC.m_pBlock						= this;
	EC.m_pDataProxy					= m_pDataProxy;
	EC.m_dInverseTimeStep			= GetInverseTimeStep();
	EC.m_pEquationExecutionInfo		= NULL;
	EC.m_eEquationCalculationMode	= eCalculateSensitivityParametersGradients;

	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		pEquationExecutionInfo->SensitivityParametersGradients(EC, narrParameterIndexes);
	}

// In general, neither objective function nor constraints can be within an STN
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->CalculateSensitivityParametersGradients(EC, narrParameterIndexes);
	}
	
	m_pDataProxy->ResetSensitivityMatrixes();
}

void daeBlock::CalcNonZeroElements(int& NNZ)
{
	size_t i;
	daeSTN* pSTN;
	daeEquationExecutionInfo* pEquationExecutionInfo;

// First find in normal equations (non-STN)
	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		NNZ += pEquationExecutionInfo->m_mapIndexes.size();
	}

// Then in STN equations
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->CalcNonZeroElements(NNZ);
	}	
}

void daeBlock::FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix)
{
	size_t i;
	daeSTN* pSTN;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		pMatrix->AddRow(pEquationExecutionInfo->m_mapIndexes);
	}
	
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->FillSparseMatrix(pMatrix);
	}
}
void daeBlock::FillAbsoluteTolerancesArray(daeArray<real_t>& arrAbsoluteTolerances)
{
	map<size_t, size_t>::iterator iter;

	if(GetNumberOfEquations() != m_mapVariableIndexes.size())
	{	
		daeDeclareException(exMiscellanous); 
		e << "Number of equations is not equal to number of variables";
		throw e;
	}
	if(!m_pDataProxy)
	{	
		daeDeclareException(exInvalidPointer); 
		e << "m_pDataProxy";
		throw e;
	}
	
	const real_t* pBlockAbsoluteTolerance = m_pDataProxy->GetAbsoluteTolerance(0);

// m_mapVariableIndexes<nOverallIndex, nBlockIndex>
	for(iter = m_mapVariableIndexes.begin(); iter != m_mapVariableIndexes.end(); iter++)
		arrAbsoluteTolerances[iter->second] = pBlockAbsoluteTolerance[iter->first];
}

void daeBlock::SetInitialConditionsAndInitialGuesses(daeArray<real_t>& arrValues, 
		                                             daeArray<real_t>& arrTimeDerivatives, 
													 daeArray<real_t>& arrInitialConditionsTypes)
{
	map<size_t, size_t>::iterator iter;

	if(GetNumberOfEquations() != m_mapVariableIndexes.size())
	{	
		daeDeclareException(exInvalidCall); 
		e << "Number of equation is not equal to number of variables";
		throw e;
	}
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer); 

	const real_t* pBlockValues          = m_pDataProxy->GetValue(0);
	const real_t* pBlockTimeDerivatives = m_pDataProxy->GetTimeDerivative(0);
	const real_t* pBlockIDs             = m_pDataProxy->GetVariableTypes();
	
	for(iter = m_mapVariableIndexes.begin(); iter != m_mapVariableIndexes.end(); iter++)
	{
		arrValues[iter->second]                 = pBlockValues[iter->first];
		arrTimeDerivatives[iter->second]        = pBlockTimeDerivatives[iter->first];
		arrInitialConditionsTypes[iter->second] = pBlockIDs[iter->first];
	} 
}

void daeBlock::SetAllInitialConditions(real_t value)
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t n = m_pDataProxy->GetTotalNumberOfVariables();
	for(size_t i = 0; i < n; i++)
	{
		if(m_pDataProxy->GetVariableTypeGathered(i) == cnDifferential)
		{
			m_pDataProxy->SetTimeDerivative(i, value);
			m_pDataProxy->SetVariableType(i, cnDifferential);
		}
	}
}

size_t daeBlock::FindVariableBlockIndex(size_t nVariableOverallIndex) const
{
	map<size_t, size_t>::const_iterator iter = m_mapVariableIndexes.find(nVariableOverallIndex);
	if(iter != m_mapVariableIndexes.end()) // if found
		return iter->second;
	else
		return ULONG_MAX;
}

bool daeBlock::IsModelDynamic() const
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer); 

	return m_pDataProxy->IsModelDynamic();
}

real_t* daeBlock::GetValuesPointer()
{
#ifdef DAE_DEBUG
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	return m_pDataProxy->GetValue(0);
}

real_t* daeBlock::GetTimeDerivativesPointer()
{
#ifdef DAE_DEBUG
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	return m_pDataProxy->GetTimeDerivative(0);
}

real_t* daeBlock::GetAbsoluteTolerancesPointer()
{
#ifdef DAE_DEBUG
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	return m_pDataProxy->GetAbsoluteTolerance(0);
}

real_t* daeBlock::GetVariableTypesPointer()
{
#ifdef DAE_DEBUG
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	return m_pDataProxy->GetVariableTypes();
}

void daeBlock::ClearAbsoluteTolerancesAndIDs()
{
#ifdef DAE_DEBUG
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	m_pDataProxy->ClearAbsoluteTolerancesAndIDs();
}

void daeBlock::CopyDataFromSolver(daeArray<real_t>& arrValues, daeArray<real_t>& arrTimeDerivatives)
{
#ifdef DAE_DEBUG
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
// m_mapVariableIndexes<nOverallIndex, nBlockIndex>
	real_t* pBlockValues          = m_pDataProxy->GetValue(0);
	real_t* pBlockTimeDerivatives = m_pDataProxy->GetTimeDerivative(0);

	for(map<size_t, size_t>::iterator iter = m_mapVariableIndexes.begin(); iter != m_mapVariableIndexes.end(); iter++)
	{
		pBlockValues         [iter->first] = arrValues         [iter->second];
		pBlockTimeDerivatives[iter->first] = arrTimeDerivatives[iter->second];
	}
}

void daeBlock::CopyDataToSolver(daeArray<real_t>& arrValues, daeArray<real_t>& arrTimeDerivatives) const
{
#ifdef DAE_DEBUG
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
// m_mapVariableIndexes<nOverallIndex, nBlockIndex>
	const real_t* pBlockValues          = m_pDataProxy->GetValue(0);
	const real_t* pBlockTimeDerivatives = m_pDataProxy->GetTimeDerivative(0);

	for(map<size_t, size_t>::const_iterator iter = m_mapVariableIndexes.begin(); iter != m_mapVariableIndexes.end(); iter++)
	{
		arrValues         [iter->second] = pBlockValues         [iter->first];
		arrTimeDerivatives[iter->second] = pBlockTimeDerivatives[iter->first];
	}
}

void daeBlock::Initialize(void)
{
	size_t i;
	pair<size_t, size_t> uintPair;
	map<size_t, size_t>::iterator iter;
	daeSTN* pSTN;

	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	if(GetNumberOfEquations() != m_mapVariableIndexes.size())
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of equations [" << GetNumberOfEquations() << "] is not equal to number of variables [" << m_mapVariableIndexes.size() << "]";
		throw e;
	}

	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer);

		pSTN->BuildExpressions(this);
	}

	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer);

		pSTN->CheckDiscontinuities();
	}
	RebuildExpressionMap();
}

bool daeBlock::CheckForDiscontinuities(void)
{
	size_t i;
	daeSTN* pSTN;

	if(m_dCurrentTime > 0 && m_pDataProxy->PrintInfo())
		m_pDataProxy->LogMessage(string("Checking state transitions at time [") + toStringFormatted<real_t>(m_dCurrentTime, -1, 15) + string("]..."), 0);

// First check the global stopping condition from the DataProxy (Simulation)
/*
	daeModel* model = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
	if(!model)
	   daeDeclareAndThrowException(exInvalidPointer);
	daeCondition* pCondition = model->GetGlobalCondition();
	if(pCondition)
	{
		daeExecutionContext EC;
		EC.m_pDataProxy					= m_pDataProxy;
		EC.m_eEquationCalculationMode	= eCalculate;
	
		if(pCondition->Evaluate(&EC))
		{
			m_pDataProxy->LogMessage(string("The global condition: ") + pCondition->SaveNodeAsPlainText() + string(" is satisfied"), 0);
			return eGlobalDiscontinuity;
		}
	}
*/
	
// Then check conditions from STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(pSTN->CheckDiscontinuities())
			return true;
	}
	
	return false;
}

daeeDiscontinuityType daeBlock::ExecuteOnConditionActions(void)
{
	size_t i;
	daeSTN* pSTN;
	daeeDiscontinuityType eResult;

	m_pDataProxy->SetReinitializationFlag(false);
	m_pDataProxy->SetCopyDataFromBlock(false);

	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->ExecuteOnConditionActions();
	}
	
// If any of the actions changed the state it has to be indicated in those flags
	if(m_pDataProxy->GetReinitializationFlag() && m_pDataProxy->GetCopyDataFromBlock())
	{
		eResult = eModelDiscontinuityWithDataChange;
		RebuildExpressionMap();
	}
	else if(m_pDataProxy->GetReinitializationFlag())
	{
		eResult = eModelDiscontinuity;
		RebuildExpressionMap();
	}
	else
	{
		eResult = eNoDiscontinuity;
	}
	
	return eResult;
}

void daeBlock::RebuildExpressionMap()
{
	size_t i;
	daeSTN* pSTN;

	m_mapExpressionInfos.clear();

// First add the global stopping condition from daeDataProxy
//	daeModel* model = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
//	if(!model)
//	   daeDeclareAndThrowException(exInvalidPointer);
//	daeCondition* pCondition = model->GetGlobalCondition();
//	if(pCondition)
//	{
//		daeExpressionInfo ei;
//		pair<size_t, daeExpressionInfo> pairExprInfo;
//		map<size_t, daeExpressionInfo>::iterator iter;
//
//		for(size_t i = 0; i < pCondition->m_ptrarrExpressions.size(); i++)
//		{
//			ei.m_pExpression      = pCondition->m_ptrarrExpressions[i];
//			ei.m_pStateTransition = NULL;
//			
//			pairExprInfo.first	= m_mapExpressionInfos.size();				
//			pairExprInfo.second	= ei;				
//			m_mapExpressionInfos.insert(pairExprInfo);
//		}
//	}
	
// Then for all othe STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->AddExpressionsToBlock(this);
	}
}

bool daeBlock::CheckOverlappingAndAddVariables(const vector<size_t>& narrVariablesInEquation)
{
	size_t i, k;
	pair<size_t, size_t> uintPair;
	map<size_t, size_t>::iterator iter;

	for(i = 0; i < narrVariablesInEquation.size(); i++)
	{
		iter = m_mapVariableIndexes.find(narrVariablesInEquation[i]);
		if(iter != m_mapVariableIndexes.end()) //if found
		{
			for(k = 0; k < narrVariablesInEquation.size(); k++)
			{
				uintPair.first  = narrVariablesInEquation[k];  // overall block
				uintPair.second = m_mapVariableIndexes.size(); // index in block
				m_mapVariableIndexes.insert(uintPair);
			}
			return true;
		}
	}

	return false;
}

void daeBlock::AddVariables(const map<size_t, size_t>& mapIndexes)
{
	pair<size_t, size_t> uintPair;
	map<size_t, size_t>::const_iterator iter;

	for(iter = mapIndexes.begin(); iter != mapIndexes.end(); iter++)
	{
		uintPair.first  = (*iter).first;				// overall index
		uintPair.second = m_mapVariableIndexes.size();	// index in block
		m_mapVariableIndexes.insert(uintPair);
	}
}

string daeBlock::GetCanonicalName(void) const
{
	return m_strName;
}

string daeBlock::GetName(void) const
{
	return m_strName;
}

void daeBlock::SetName(const string& strName)
{
	m_strName = strName;
}

size_t daeBlock::GetNumberOfRoots() const
{
	size_t nNoRoots = 0;
	
// First check the global stopping condition
//	daeModel* model = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
//	if(!model)
//	   daeDeclareAndThrowException(exInvalidPointer);
//	daeCondition* pCondition = model->GetGlobalCondition();
//	if(pCondition)
//		nNoRoots = 1;

	return (nNoRoots + m_mapExpressionInfos.size());
}

daeDataProxy_t* daeBlock::GetDataProxy(void) const
{
	return m_pDataProxy;
}

void daeBlock::SetDataProxy(daeDataProxy_t* pDataProxy)
{
	if(!pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	m_pDataProxy = pDataProxy;
}

map<size_t, size_t>& daeBlock::GetVariableIndexesMap()
{
	return m_mapVariableIndexes;
}

void daeBlock::AddEquationExecutionInfo(daeEquationExecutionInfo* pEquationExecutionInfo)
{
	if(!pEquationExecutionInfo)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
}

void daeBlock::GetEquationExecutionInfo(vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfos)
{
	ptrarrEquationExecutionInfos = m_ptrarrEquationExecutionInfos;
}

size_t daeBlock::GetNumberOfEquations() const
{
	return m_nNumberOfEquations;
}

real_t daeBlock::GetResidual(size_t nIndex) const
{
#ifdef DAE_DEBUG
	if(!m_parrResidual)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	return (*m_parrResidual)[nIndex];
}

void daeBlock::SetResidual(size_t nIndex, real_t dResidual)
{
#ifdef DAE_DEBUG
	if(!m_parrResidual)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	(*m_parrResidual)[nIndex] = dResidual;
}

real_t daeBlock::GetJacobian(size_t nEquationIndex, size_t nVariableindexInBlock) const
{
#ifdef DAE_DEBUG
	if(!m_pmatJacobian)
		daeDeclareAndThrowException(exInvalidPointer); 
#endif
	return m_pmatJacobian->GetItem(nEquationIndex, nVariableindexInBlock);
}

void daeBlock::SetJacobian(size_t nEquationIndex, size_t nVariableindexInBlock, real_t dJacobianItem)
{
#ifdef DAE_DEBUG
	if(!m_pmatJacobian)
		daeDeclareAndThrowException(exInvalidPointer); 
#endif
	m_pmatJacobian->SetItem(nEquationIndex, nVariableindexInBlock, dJacobianItem);
}

daeMatrix<real_t>* daeBlock::GetJacobianMatrix() const
{
	return m_pmatJacobian;
}

void daeBlock::SetJacobianMatrix(daeMatrix<real_t>* pJacobian)
{
#ifdef DAE_DEBUG
	if(!pJacobian)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	m_pmatJacobian = pJacobian;
}

daeArray<real_t>* daeBlock::GetResidualArray() const
{
	return m_parrResidual;
}

void daeBlock::SetResidualArray(daeArray<real_t>* pResidual)
{
#ifdef DAE_DEBUG
	if(!pResidual)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	m_parrResidual = pResidual;
}

real_t daeBlock::GetTime() const
{
	return m_dCurrentTime;
}

void daeBlock::SetTime(real_t dTime)
{
	m_dCurrentTime = dTime;
}

real_t daeBlock::GetInverseTimeStep() const
{
	return m_dInverseTimeStep;
}

void daeBlock::SetInverseTimeStep(real_t dInverseTimeStep)
{
	m_dInverseTimeStep = dInverseTimeStep;
}

bool daeBlock::GetInitializeMode() const
{
	return m_bInitializeMode;
}

void daeBlock::SetInitializeMode(bool bMode)
{
	m_bInitializeMode = bMode;
}

bool daeBlock::CheckObject(vector<string>& strarrErrors) const
{
	dae_capacity_check(m_ptrarrEquationExecutionInfos);
	dae_capacity_check(m_ptrarrSTNs);

	return true;
}


}
}

