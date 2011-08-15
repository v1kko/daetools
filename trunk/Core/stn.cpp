#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	daeSTN
**********************************************************************************************/
daeSTN::daeSTN()
{
	m_pModel		= NULL;
	m_pParentState	= NULL;
	m_pActiveState	= NULL;
	m_eSTNType		= eSTN;
}

daeSTN::~daeSTN()
{
}

void daeSTN::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	m_ptrarrStates.EmptyAndFreeMemory();
	m_ptrarrStates.SetOwnershipOnPointers(true);

	daeObject::Open(pTag);

	strName = "Type";
	OpenEnum(pTag, strName, m_eSTNType);

	strName = "States";
	pTag->OpenObjectArray<daeState, daeState>(strName, m_ptrarrStates);

	strName = "ParentState";
	daeFindStateByID del(m_pModel);
	m_pParentState = pTag->OpenObjectRef(strName, &del);

	//ReconnectStateTransitionsAndStates();
}

void daeSTN::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "Type";
	SaveEnum(pTag, strName, m_eSTNType);

	strName = "States";
	pTag->SaveObjectArray(strName, m_ptrarrStates);
	
	strName = "ParentState";
	pTag->SaveObjectRef(strName, m_pParentState);
}

void daeSTN::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport, strStates;
	boost::format fmtFile;

	if(c.m_bExportDefinition)
	{
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.%1% = self.STN(\"%2%\")\n\n" + 
						"%3%"+
						c.CalculateIndent(c.m_nPythonIndentLevel) + "self.END_STN()\n\n";
			ExportObjectArray(m_ptrarrStates, strStates, eLanguage, c);
			
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% m_strShortName 
					% strStates;
		}
		else if(eLanguage == eCDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "daeSTN* %1% = STN(\"%2%\");\n\n" + 
						"%3%"+
						c.CalculateIndent(c.m_nPythonIndentLevel) + "END_STN();\n\n";
			ExportObjectArray(m_ptrarrStates, strStates, eLanguage, c);

			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% m_strShortName 
					% strStates;
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += fmtFile.str();
}

void daeSTN::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daeSTN::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::SaveRuntime(pTag);

	strName = "Type";
	SaveEnum(pTag, strName, m_eSTNType);

	strName = "States";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrStates);
}

void daeSTN::ReconnectStateTransitionsAndStates()
{
	daeDeclareAndThrowException(exInvalidCall);
	
//	size_t i, k;
//	daeState* pState;
//	daeStateTransition *pST;
//
//	for(i = 0; i < m_ptrarrStates.size(); i++)
//	{
//		pState = m_ptrarrStates[i];
//		if(!pState)
//			daeDeclareAndThrowException(exInvalidPointer);
//
//		for(k = 0; k < pState->m_ptrarrStateTransitions.size(); k++)
//		{
//			pST = pState->m_ptrarrStateTransitions[k];
//			if(!pST)
//				daeDeclareAndThrowException(exInvalidPointer);
//
//			pST->m_pStateFrom = FindState(pST->m_nStateFromID);
//			if(!pST->m_pStateFrom)
//			{	
//				daeDeclareException(exInvalidCall); 
//				e << "Illegal start state in STN [" << m_strCanonicalName << "]";
//				throw e;
//			}
//
//			pST->m_pStateTo = FindState(pST->m_nStateToID);
//			if(!pST->m_pStateTo)
//			{	
//				daeDeclareException(exInvalidCall); 
//				e << "Illegal end state in STN [" << m_strCanonicalName << "]";
//				throw e;
//			}
//		}
//	}
}

daeState* daeSTN::FindState(long nID)
{
	size_t i;
	daeState* pState;

	for(i = 0; i < m_ptrarrStates.size(); i++)
	{
		pState = m_ptrarrStates[i];
		if(pState && pState->m_nID == (size_t)nID)
			return pState;
	}
	return NULL;
}
	
daeState* daeSTN::FindState(const string& strName)
{
	size_t i;
	daeState* pState;

	for(i = 0; i < m_ptrarrStates.size(); i++)
	{
		pState = m_ptrarrStates[i];
		if(pState && pState->GetName() == strName)
			return pState;
	}
	return NULL;
}

void daeSTN::FinalizeDeclaration()
{
	daeState *pState;

// This function is called at the end of creation of states so remove the last state from stack
	m_pModel->RemoveStateFromStack();

// Set the active state (default is the first)
	if(!GetActiveState())
	{
		if(m_ptrarrStates.size() > 0)
		{
			pState = m_ptrarrStates[0];
			SetActiveState(pState);
		}
	}

	m_bInitialized = true;
}

void daeSTN::InitializeStateTransitions(void)
{
	size_t i;
	daeState *pState;
	
	for(i = 0; i < m_ptrarrStates.size(); i++)
	{
		pState = m_ptrarrStates[i];
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer);

		pState->InitializeStateTransitions();
	}
}

void daeSTN::InitializeDEDIs(void)
{
	size_t i;
	daeState *pState;
	
	for(i = 0; i < m_ptrarrStates.size(); i++)
	{
		pState = m_ptrarrStates[i];
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer);

		pState->InitializeDEDIs();
	}
}

void daeSTN::CreateEquationExecutionInfo(void)
{
	size_t i, k, m;
	daeSTN* pSTN;
	daeState* pState;
	daeEquation* pEquation;
	vector<daeEquationExecutionInfo*> ptrarrEqnExecutionInfosCreated;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(m_ptrarrStates.size() == 0)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Number of states is 0; there must be at least two states in STN [" << m_strCanonicalName << "]";
		throw e;
	}

	for(k = 0; k < m_ptrarrStates.size(); k++)
	{
		pState = m_ptrarrStates[k];
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer); 

		for(m = 0; m < pState->m_ptrarrEquations.size(); m++)
		{
			pEquation = pState->m_ptrarrEquations[m];
			if(!pEquation)
				daeDeclareAndThrowException(exInvalidPointer); 

		// Create EqnExecInfos, call GatherInfo for each of them, but DON'T add them to the model
		// They are added to the vector which belongs to the state
			ptrarrEqnExecutionInfosCreated.clear();
			m_pModel->CreateEquationExecutionInfo(pEquation, ptrarrEqnExecutionInfosCreated, false);

		// Now add all of them to the state
			pState->m_ptrarrEquationExecutionInfos.insert(pState->m_ptrarrEquationExecutionInfos.end(),
				                                          ptrarrEqnExecutionInfosCreated.begin(),
														  ptrarrEqnExecutionInfosCreated.end());
		}

		for(i = 0; i < pState->m_ptrarrSTNs.size(); i++)
		{
			pSTN = pState->m_ptrarrSTNs[i];
			if(!pSTN)
				daeDeclareAndThrowException(exInvalidPointer);

			pSTN->CreateEquationExecutionInfo();
		}
	}

}

void daeSTN::CollectEquationExecutionInfos(vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo)
{
	size_t i, k;
	daeSTN* pSTN;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	if(!m_pActiveState)
		daeDeclareAndThrowException(exInvalidPointer); 

	for(k = 0; k < m_pActiveState->m_ptrarrEquationExecutionInfos.size(); k++)
	{
		pEquationExecutionInfo = m_pActiveState->m_ptrarrEquationExecutionInfos[k];
		if(!pEquationExecutionInfo)
			daeDeclareAndThrowException(exInvalidPointer); 

		ptrarrEquationExecutionInfo.push_back(pEquationExecutionInfo);
	}

// Nested STNs
	for(i = 0; i < m_pActiveState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_pActiveState->m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer); 

		pSTN->CollectEquationExecutionInfos(ptrarrEquationExecutionInfo);
	}
}

void daeSTN::CollectVariableIndexes(map<size_t, size_t>& mapVariableIndexes)
{
	size_t i, k, m;
	daeSTN* pSTN;
	daeState* pState;
	daeStateTransition* pStateTransition;
	daeEquationExecutionInfo* pEquationExecutionInfo;
	pair<size_t, size_t> uintPair;
	map<size_t, size_t>::iterator iter;

// Collect indexes from the states
	for(i = 0; i < m_ptrarrStates.size(); i++)
	{
		pState = m_ptrarrStates[i];
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer); 

		for(k = 0; k < pState->m_ptrarrEquationExecutionInfos.size(); k++)
		{
			pEquationExecutionInfo = pState->m_ptrarrEquationExecutionInfos[k];
			if(!pEquationExecutionInfo)
				daeDeclareAndThrowException(exInvalidPointer); 

			for(iter = pEquationExecutionInfo->m_mapIndexes.begin(); iter != pEquationExecutionInfo->m_mapIndexes.end(); iter++)
			{
				uintPair.first  = (*iter).first;
				uintPair.second = mapVariableIndexes.size(); // doesn't matter what value - it is not used anywhere
				mapVariableIndexes.insert(uintPair);
			}
		}

	// Collect indexes from the nested STNs
		for(m = 0; m < pState->m_ptrarrSTNs.size(); m++)
		{
			pSTN = pState->m_ptrarrSTNs[m];
			if(!pSTN)
				daeDeclareAndThrowException(exInvalidPointer); 

			pSTN->CollectVariableIndexes(mapVariableIndexes);
		}
	}

// Collect indexes from the conditions
	for(k = 0; k < m_ptrarrStates.size(); k++)
	{
		pState = m_ptrarrStates[k];
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer); 

		for(m = 0; m < pState->m_ptrarrStateTransitions.size(); m++)
		{
			pStateTransition = pState->m_ptrarrStateTransitions[m];
			if(!pStateTransition)
				daeDeclareAndThrowException(exInvalidPointer); 

			pStateTransition->m_Condition.m_pConditionNode->AddVariableIndexToArray(mapVariableIndexes, false);
		}
	}
}

void daeSTN::SetIndexesWithinBlockToEquationExecutionInfos(daeBlock* pBlock, size_t& nEquationIndex)
{
	size_t i, k, m, nTempEquationIndex;
	daeSTN* pSTN;
	daeState* pState;
	daeEquationExecutionInfo* pEquationExecutionInfo;
	map<size_t, size_t>::iterator iter, iterIndexInBlock;

	for(i = 0; i < m_ptrarrStates.size(); i++)
	{
		pState = m_ptrarrStates[i];
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer); 

		// Indexes are the same in each state;
		// State1: 15, 16, 17
		// State2: 15, 16, 17
		// etc ...
		nTempEquationIndex = nEquationIndex;
		for(k = 0; k < pState->m_ptrarrEquationExecutionInfos.size(); k++)
		{
			pEquationExecutionInfo = pState->m_ptrarrEquationExecutionInfos[k];
			if(!pEquationExecutionInfo)
				daeDeclareAndThrowException(exInvalidPointer); 

			pEquationExecutionInfo->m_nEquationIndexInBlock = nTempEquationIndex;
			pEquationExecutionInfo->m_pBlock = pBlock;
//------------------->
		// Here I have to associate overall variable indexes in equation to corresponding indexes in the block
		// m_mapIndexes<OverallIndex, BlockIndex>
			for(iter = pEquationExecutionInfo->m_mapIndexes.begin(); iter != pEquationExecutionInfo->m_mapIndexes.end(); iter++)
			{
			// Try to find OverallIndex in the map of BlockIndexes
				iterIndexInBlock = pBlock->m_mapVariableIndexes.find((*iter).first);
				if(iterIndexInBlock == pBlock->m_mapVariableIndexes.end())
				{
					daeDeclareException(exInvalidCall);
					e << "Cannot find overall variable index [" << toString<size_t>((*iter).first) << "] in equation " << pEquationExecutionInfo->m_pEquation->m_strCanonicalName;
					throw e;
				}
				(*iter).second = (*iterIndexInBlock).second;
			}
//------------------->
			nTempEquationIndex++;
		}
	// Nested STNs
		for(m = 0; m < pState->m_ptrarrSTNs.size(); m++)
		{
			pSTN = pState->m_ptrarrSTNs[m];
			if(!pSTN)
				daeDeclareAndThrowException(exInvalidPointer);
		// Here I use nTempEquationIndex since I continue counting equations in the same state
			pSTN->SetIndexesWithinBlockToEquationExecutionInfos(pBlock, nTempEquationIndex);
		}
	}

	nEquationIndex = nTempEquationIndex;
}

void daeSTN::AddExpressionsToBlock(daeBlock* pBlock)
{
	size_t i;
	daeSTN* pSTN;
	daeState* pState;
	daeStateTransition* pStateTransition;
	pair<size_t, daeExpressionInfo> pairExprInfo;
	map<size_t, daeExpressionInfo>::iterator iter;

	pState = m_pActiveState;
	if(!pState)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Active state does not exist in STN [" << m_strCanonicalName << "]";
		throw e;
	}

	for(i = 0; i < pState->m_ptrarrStateTransitions.size(); i++) 
	{
		pStateTransition = pState->m_ptrarrStateTransitions[i];
		if(!pStateTransition)
			daeDeclareAndThrowException(exInvalidPointer); 

		for(iter = pStateTransition->m_mapExpressionInfos.begin(); iter != pStateTransition->m_mapExpressionInfos.end(); iter++)
		{
			pairExprInfo		= *iter;
			pairExprInfo.first	= pBlock->m_mapExpressionInfos.size();				
			pBlock->m_mapExpressionInfos.insert(pairExprInfo);
		}
	}
	
// Nested STNs
	for(i = 0; i < pState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = pState->m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer); 

		pSTN->AddExpressionsToBlock(pBlock);
	}
}

void daeSTN::BuildExpressions(daeBlock* pBlock)
{
	size_t i, k, m;
	daeSTN* pSTN;
	daeState* pState;
	daeStateTransition* pStateTransition;
	pair<size_t, daeExpressionInfo> pairExprInfo;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeExecutionContext EC;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_eEquationCalculationMode	= eCreateFunctionsIFsSTNs;

// I have to set this since Create_adouble called from adSetup nodes needs it
	m_pModel->PropagateGlobalExecutionContext(&EC);

	for(i = 0; i < m_ptrarrStates.size(); i++) 
	{
		pState = m_ptrarrStates[i];
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer); 

		size_t nIndexInState = 0;
		for(k = 0; k < pState->m_ptrarrStateTransitions.size(); k++) 
		{
			pStateTransition = pState->m_ptrarrStateTransitions[k];
			if(!pStateTransition)
				daeDeclareAndThrowException(exInvalidPointer); 

		// Fill array with expressions of the form: left - right, 
		// made of the conditional expressions, like: left >= right ... etc etc
			m_pModel->m_pDataProxy->SetGatherInfo(true);
			pBlock->SetInitializeMode(true);
				pStateTransition->m_Condition.BuildExpressionsArray(&EC);
				for(m = 0; m < pStateTransition->m_Condition.m_ptrarrExpressions.size(); m++)
				{
					pairExprInfo.first                     = nIndexInState;
					pairExprInfo.second.m_pExpression      = pStateTransition->m_Condition.m_ptrarrExpressions[m];
					pairExprInfo.second.m_pStateTransition = pStateTransition;

					pStateTransition->m_mapExpressionInfos.insert(pairExprInfo);
					nIndexInState++;
				}
			m_pModel->m_pDataProxy->SetGatherInfo(false);
			pBlock->SetInitializeMode(false);
		}

	// Nested STNs
		for(m = 0; m < pState->m_ptrarrSTNs.size(); m++)
		{
			pSTN = pState->m_ptrarrSTNs[m];
			if(!pSTN)
				daeDeclareAndThrowException(exInvalidPointer); 

			pSTN->BuildExpressions(pBlock);
		}
	}

// Restore it to NULL
	m_pModel->PropagateGlobalExecutionContext(NULL);
}

bool daeSTN::CheckDiscontinuities()
{
	size_t i;
	daeSTN* pSTN;
	daeStateTransition* pStateTransition;

	if(!m_pActiveState)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Active state does not exist in STN [" << m_strCanonicalName << "]";
		throw e;
	}

	daeExecutionContext EC;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_eEquationCalculationMode	= eCalculate;

	// Save the current active state to check if it has been changed by one of the actions
	daeState* pFormerActiveState = m_pActiveState;
	
	bool bResult = false;
	for(i = 0; i < m_pActiveState->m_ptrarrStateTransitions.size(); i++)
	{
		pStateTransition = m_pActiveState->m_ptrarrStateTransitions[i];
		if(!pStateTransition)
			daeDeclareAndThrowException(exInvalidPointer); 

		if(pStateTransition->m_Condition.Evaluate(&EC))
		{
			LogMessage(string("The condition: ") + pStateTransition->GetConditionAsString() + string(" is satisfied"), 0);
		
		// Execute the actions
			pStateTransition->ExecuteActions();
			
		// If the active state has changed then set the flag and break
			if(pFormerActiveState != m_pActiveState)
			{
				bResult = true;
				break;
			}
			//return CheckState(pStateTransition->m_pStateTo);
		}
	}

// Now I have to check state transitions in the nested STNs of the current active state
// m_pActiveState might point now to the new state (if the state-change occured in actions above)
	for(i = 0; i < m_pActiveState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_pActiveState->m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer); 

		if(pSTN->CheckDiscontinuities())
			bResult = true;
	}
	return bResult;
}

// Not used anymore
bool daeSTN::CheckState(daeState* pState)
{
/*
	daeSTN* pSTN;
	
	if(!pState)
		daeDeclareAndThrowException(exInvalidPointer);

// Only if the current state is not equal to the active state change the state (no reinitialization)
// but continue searching for state change in the nested IF/STNs
	bool bResult = false;
	if(m_pActiveState != pState)
	{
		bResult = true;
		SetActiveState(pState);
	}
	else
	{
		LogMessage(string("Current state unchanged"), 0);
	}

// Check nested STNs no matter if the active state has or has not been changed
	for(size_t i = 0; i < pState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = pState->m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer);

		if(pSTN->CheckDiscontinuities())
			bResult = true;
	}

	return bResult;
*/
	return true;
}

size_t daeSTN::GetNumberOfEquations() const
{
	daeState* pState;

	if(m_ptrarrStates.empty())
		daeDeclareException(exInvalidCall);

	pState = m_ptrarrStates[0];
	if(!pState)
		daeDeclareException(exInvalidPointer); 

	return GetNumberOfEquationsInState(pState);
}

size_t daeSTN::GetNumberOfStates(void) const
{
	return m_ptrarrStates.size();
}

size_t daeSTN::GetNumberOfEquationsInState(daeState* pState) const
{
	size_t i, nNoEqns;
	daeSTN* pSTN;
	daeEquation* pEquation;

	if(!pState)
		daeDeclareAndThrowException(exInvalidPointer); 

	nNoEqns = 0;
	for(i = 0; i < pState->m_ptrarrEquations.size(); i++)
	{
		pEquation = pState->m_ptrarrEquations[i];
		if(!pEquation)
			daeDeclareAndThrowException(exInvalidPointer); 
		nNoEqns += pEquation->GetNumberOfEquations();
	}

	for(i = 0; i < pState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = pState->m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer); 

		nNoEqns += pSTN->GetNumberOfEquations();
	}

	return nNoEqns;
}

daeState* daeSTN::GetParentState(void) const
{
	return m_pParentState;
}

void daeSTN::SetParentState(daeState* pParentState)
{
	if(!pParentState)
		daeDeclareAndThrowException(exInvalidPointer); 
	m_pParentState = pParentState;
}

daeState* daeSTN::AddState(string strName)
{
// Remove previous state (if any) from the stack
	m_pModel->RemoveStateFromStack();

// Instantiate a new state and add it to STN
// Ignore the name
	daeState* pState = new daeState();
	m_ptrarrStates.push_back(pState);
	pState->Create(strName, this);
	pState->m_strCanonicalName = m_strCanonicalName + "." + pState->m_strShortName;

// Put it on the stack
	m_pModel->PutStateToStack(pState);
	return pState;
}

void daeSTN::GetStates(vector<daeState_t*>& ptrarrStates)
{
	ptrarrStates.clear();
	for(size_t i = 0; i < m_ptrarrStates.size(); i++)
		ptrarrStates.push_back(m_ptrarrStates[i]);
}

void daeSTN::SetActiveState2(const string& strStateName)
{
	daeState* pState = FindState(strStateName);

	if(!pState)
	{
		daeDeclareException(exInvalidCall); 
		e << "The state [" << strStateName << "] does not exist in STN [" << m_strCanonicalName << "]";
		throw e;
	}
	
	SetActiveState(pState);
}

string daeSTN::GetActiveState2(void) const
{
	if(!m_pActiveState)
		daeDeclareAndThrowException(exInvalidPointer); 
	
	return m_pActiveState->GetName();
}

void daeSTN::SetActiveState(daeState* pState)
{
	if(!pState)
		daeDeclareAndThrowException(exInvalidPointer); 
	
	LogMessage(string("The state: ") + pState->m_strShortName + string(" is active now"), 0);
	
	m_pModel->m_pDataProxy->SetReinitializationFlag(true);
	
// Disconnect old OnEventActions
	if(m_pActiveState)
		m_pActiveState->DisconnectOnEventActions();

// Connect new OnEventActions
	pState->ConnectOnEventActions();
	
// Set the new active state
	m_pActiveState = pState;
}

daeState_t* daeSTN::GetActiveState()
{
	return m_pActiveState;
}

void daeSTN::CalculateResiduals(void)
{
	size_t i;
	daeSTN* pSTN;
	daeState* pState;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	pState = m_pActiveState;
	if(!pState)
		daeDeclareAndThrowException(exInvalidPointer); 

	for(i = 0; i < pState->m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = pState->m_ptrarrEquationExecutionInfos[i];
		if(!pEquationExecutionInfo)
			daeDeclareAndThrowException(exInvalidPointer); 

		pEquationExecutionInfo->Residual();
	}
// Nested STNs
	for(i = 0; i < pState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = pState->m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer); 

		pSTN->CalculateResiduals();
	}
}

void daeSTN::CalculateJacobian(void)
{
	size_t i;
	daeSTN* pSTN;
	daeState* pState;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	pState = m_pActiveState;
	if(!pState)
		daeDeclareAndThrowException(exInvalidPointer); 

	for(i = 0; i < pState->m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = pState->m_ptrarrEquationExecutionInfos[i];
		if(!pEquationExecutionInfo)
			daeDeclareAndThrowException(exInvalidPointer); 

		pEquationExecutionInfo->Jacobian();
	}
// Nested STNs
	for(i = 0; i < pState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = pState->m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer); 

		pSTN->CalculateJacobian();
	}
}

void daeSTN::CalculateSensitivityResiduals(const std::vector<size_t>& narrParameterIndexes)
{
	size_t i;
	daeSTN* pSTN;
	daeState* pState;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	pState = m_pActiveState;
	if(!pState)
		daeDeclareAndThrowException(exInvalidPointer); 

	for(i = 0; i < pState->m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = pState->m_ptrarrEquationExecutionInfos[i];
		if(!pEquationExecutionInfo)
			daeDeclareAndThrowException(exInvalidPointer); 

		pEquationExecutionInfo->SensitivityResiduals(narrParameterIndexes);
	}
// Nested STNs
	for(i = 0; i < pState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = pState->m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer); 

		pSTN->CalculateSensitivityResiduals(narrParameterIndexes);
	}
}

void daeSTN::CalculateSensitivityParametersGradients(const std::vector<size_t>& narrParameterIndexes)
{
	size_t i;
	daeSTN* pSTN;
	daeState* pState;
	daeEquationExecutionInfo* pEquationExecutionInfo;

	pState = m_pActiveState;
	if(!pState)
		daeDeclareAndThrowException(exInvalidPointer); 

	for(i = 0; i < pState->m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = pState->m_ptrarrEquationExecutionInfos[i];
		if(!pEquationExecutionInfo)
			daeDeclareAndThrowException(exInvalidPointer); 

		pEquationExecutionInfo->SensitivityParametersGradients(narrParameterIndexes);
	}
// Nested STNs
	for(i = 0; i < pState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = pState->m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer); 

		pSTN->CalculateSensitivityParametersGradients(narrParameterIndexes);
	}
}

void daeSTN::CalcNonZeroElements(int& NNZ)
{
	daeState* pState = m_pActiveState;	
	pState->CalcNonZeroElements(NNZ);
}

void daeSTN::FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix)
{
	daeState* pState = m_pActiveState;	
	pState->FillSparseMatrix(pMatrix);	
}

bool daeSTN::CheckObject(vector<string>& strarrErrors) const
{
	string strError;
	size_t nNoEquationsInEachState = 0;

	bool bCheck = true;

// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// If parent state is not null then it is nested stn	
//	if(m_pParentState);
	
// Check the active state	
	if(!m_pActiveState)
	{
		strError = "Invalid active state in state transition network [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
// Check the type	
	if(m_eSTNType == eSTNTUnknown)
	{
		strError = "Invalid type in state transition network [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
// Check whether the stn is initialized	
	if(!m_bInitialized)
	{
		strError = "Uninitialized state transition network [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check number of states	
	if(m_ptrarrStates.size() == 0)
	{
		strError = "Invalid number of states in state transition network [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check states	
	if(m_ptrarrStates.size() > 0)
	{
		daeState* pState;	
		for(size_t i = 0; i < m_ptrarrStates.size(); i++)
		{
			pState = m_ptrarrStates[i];
			if(!pState)
			{
				strError = "Invalid state in state transition network [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
			
		// Number of equations must be the same in all states
			if(i == 0)
			{
				nNoEquationsInEachState = GetNumberOfEquationsInState(pState);
			}
			else
			{
				if(nNoEquationsInEachState != GetNumberOfEquationsInState(pState))
				{	
					strError = "Number of equations must be the same in all states in state transition network [" + GetCanonicalName() + "]";
					strarrErrors.push_back(strError);
					bCheck = false;
				}
			}

			if(!pState->CheckObject(strarrErrors))
				bCheck = false;
		}
	}

	return bCheck;
}


}
}

