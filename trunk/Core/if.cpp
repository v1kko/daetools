#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{
/*************************************************************
	daeIF
**************************************************************/
daeIF::daeIF(void)
{
	m_bInitialized = false;
	m_eSTNType     = eIF;
}

daeIF::~daeIF(void)
{
}

void daeIF::Clone(const daeIF& rObject)
{
	for(size_t i = 0; i < rObject.m_ptrarrStates.size(); i++)
	{
		daeState* pState = AddState(rObject.m_ptrarrStates[i]->m_strShortName);
		pState->Clone(*rObject.m_ptrarrStates[i]);
	}
}

void daeIF::Open(io::xmlTag_t* pTag)
{
	daeSTN::Open(pTag);
}

void daeIF::Save(io::xmlTag_t* pTag) const
{
	daeSTN::Save(pTag);
}

void daeIF::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	daeState* pState;
	string strExport, strState, strSTNs, strStateTransition, strEquations;
	boost::format fmtState;

	if(c.m_bExportDefinition)
	{
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			for(size_t i = 0; i < m_ptrarrStates.size(); i++)
			{
				pState = m_ptrarrStates[i];
				
				strEquations.clear();
				strStateTransition.clear();
				if(i == 0) // The first one: IF
				{
					strState = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.IF(%1%)\n\n" + "%2%%3%\n";

					pState->m_ptrarrStateTransitions[0]->m_Condition.Export(strStateTransition, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrEquations, strEquations, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrSTNs, strSTNs, eLanguage, c);
				
					fmtState.parse(strState);
					fmtState % strStateTransition % strEquations % strSTNs;
				}
				else if(i == m_ptrarrStates.size()-1)  // The last one: ELSE
				{
					strState = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.ELSE()\n\n" + "%1%%2%\n";
					
					ExportObjectArray(pState->m_ptrarrEquations, strEquations, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrSTNs, strSTNs, eLanguage, c);
				
					fmtState.parse(strState);
					fmtState % strEquations % strSTNs;
				}
				else // ELSE_IF
				{
					strState = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.ELSE_IF(%1%)\n\n" + "%2%3%\n";
					
					pState->m_ptrarrStateTransitions[0]->m_Condition.Export(strStateTransition, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrEquations, strEquations, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrSTNs, strSTNs, eLanguage, c);
				
					fmtState.parse(strState);
					fmtState % strStateTransition % strEquations % strSTNs;
				}
				strExport += fmtState.str();
			}
			
			strExport += c.CalculateIndent(c.m_nPythonIndentLevel) + "self.END_IF()\n\n";
		}
		else if(eLanguage == eCDAE)
		{
			for(size_t i = 0; i < m_ptrarrStates.size(); i++)
			{
				pState = m_ptrarrStates[i];
				
				strEquations.clear();
				strStateTransition.clear();
				if(i == 0) // The first one: IF
				{
					strState = c.CalculateIndent(c.m_nPythonIndentLevel) + "IF(%1%);\n\n" + "%2%%3%\n";

					pState->m_ptrarrStateTransitions[0]->m_Condition.Export(strStateTransition, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrEquations, strEquations, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrSTNs, strSTNs, eLanguage, c);
				
					fmtState.parse(strState);
					fmtState % strStateTransition % strEquations % strSTNs;
				}
				else if(i == m_ptrarrStates.size()-1)  // The last one: ELSE
				{
					strState = c.CalculateIndent(c.m_nPythonIndentLevel) + "ELSE();\n\n" + "%1%%2%\n";
					
					ExportObjectArray(pState->m_ptrarrEquations, strEquations, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrSTNs, strSTNs, eLanguage, c);
				
					fmtState.parse(strState);
					fmtState % strEquations % strSTNs;
				}
				else // ELSE_IF
				{
					strState = c.CalculateIndent(c.m_nPythonIndentLevel) + "ELSE_IF(%1%);\n\n" + "%2%3%\n";
					
					pState->m_ptrarrStateTransitions[0]->m_Condition.Export(strStateTransition, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrEquations, strEquations, eLanguage, c);
					ExportObjectArray(pState->m_ptrarrSTNs, strSTNs, eLanguage, c);
				
					fmtState.parse(strState);
					fmtState % strStateTransition % strEquations % strSTNs;
				}
				strExport += fmtState.str();
			}
			
			strExport += c.CalculateIndent(c.m_nPythonIndentLevel) + "END_IF();\n\n";
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += strExport;
}

void daeIF::OpenRuntime(io::xmlTag_t* pTag)
{
	daeSTN::OpenRuntime(pTag);
}

void daeIF::SaveRuntime(io::xmlTag_t* pTag) const
{
	daeSTN::SaveRuntime(pTag);
}

void daeIF::AddExpressionsToBlock(daeBlock* pBlock)
{
	size_t i, k, m;
	daeSTN* pSTN;
	daeState* pState;
	daeStateTransition* pStateTransition;
	pair<size_t, daeExpressionInfo> pairExprInfo;
	map<size_t, daeExpressionInfo>::iterator iter;

	for(i = 0; i < m_ptrarrStates.size(); i++) 
	{
		pState = m_ptrarrStates[i];
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer); 

		for(k = 0; k < pState->m_ptrarrStateTransitions.size(); k++) 
		{
			pStateTransition = pState->m_ptrarrStateTransitions[k];
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
		for(m = 0; m < pState->m_ptrarrSTNs.size(); m++)
		{
			pSTN = pState->m_ptrarrSTNs[m];
			if(!pSTN)
				daeDeclareAndThrowException(exInvalidPointer); 

			pSTN->AddExpressionsToBlock(pBlock);
		}
	}
}

void daeIF::FinalizeDeclaration()
{
	daeState* pState;

// This function is called at the end of creation of states so remove the last state from stack
	m_pModel->RemoveStateFromStack();

// Set the active state (just temporary, until the next check)
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

/* Old CheckDiscontinuities() function
bool daeIF::CheckDiscontinuities()
{
	bool bResult, bResultTemp;
	size_t i;
	daeState* pState, *pElseState;
	daeStateTransition* pStateTransition;

	if(!m_pActiveState)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Active state does not exist in IF [" << m_strCanonicalName << "]";
		throw e;
	}

	daeExecutionContext EC;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_eEquationCalculationMode	= eCalculate;

	bResult = false;
	for(i = 0; i < m_ptrarrStates.size(); i++) // For all States in the STN
	{
		pState = m_ptrarrStates[i];
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer); 

		if(i != m_ptrarrStates.size() - 1) // All states except the last
		{
			pStateTransition = pState->m_ptrarrStateTransitions[0];
			if(!pStateTransition)
				daeDeclareAndThrowException(exInvalidPointer); 

			bResultTemp = pStateTransition->m_Condition.Evaluate(&EC);
			if(bResultTemp)
				return CheckState(pState);
		}
	}
	
// If no conditions are satisfied, then I have to activate ELSE
	if(!bResult)
	{
		pElseState = m_ptrarrStates[m_ptrarrStates.size() - 1];
		if(!pElseState)
			daeDeclareAndThrowException(exInvalidPointer);

		return CheckState(pElseState);
	}
	
	return false;
}
*/

bool daeIF::CheckDiscontinuities(void)
{
	size_t i;
	daeSTN* pSTN;
	daeState* pState;
	daeStateTransition* pStateTransition;

	if(!m_pActiveState)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Active state does not exist in IF [" << m_strCanonicalName << "]";
		throw e;
	}

	daeExecutionContext EC;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_eEquationCalculationMode	= eCalculate;

	for(i = 0; i < m_ptrarrStates.size(); i++)
	{
		pState = m_ptrarrStates[i];

		if(i != m_ptrarrStates.size() - 1) // All states except the last (for ELSE does not have conditions)
		{
			pStateTransition = pState->m_ptrarrStateTransitions[0];

			if(pStateTransition->m_Condition.Evaluate(&EC))
			{
			// If this state is not the active one then there is state change - therefore return true
			// If it is equal then there is no state change so break the for loop
				if(m_pActiveState != pState) // There is a state change, thus return true; otherwise break
					return true;
				else
					break;
			}
		}
		else if(i == m_ptrarrStates.size() - 1) // ELSE state (with no conditions)
		{
		// If I reached this point then none of the above conditions was satisfied, so that ELSE should be the active state
			if(m_pActiveState != pState) // There is a state change, thus return true
				return true;
		}
	}
	
// If I reached this point then there was no discontinuity in this IF block so check for discontinuities in the nested STNs
	for(i = 0; i < m_pActiveState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_pActiveState->m_ptrarrSTNs[i];
		if(pSTN->CheckDiscontinuities())
			return true;
	}

	return false;
}

void daeIF::ExecuteOnConditionActions(void)
{
	size_t i;
	daeState* pState;
	daeSTN* pSTN;
	daeStateTransition* pStateTransition;

	if(!m_pActiveState)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Active state does not exist in IF [" << m_strCanonicalName << "]";
		throw e;
	}

	daeExecutionContext EC;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_eEquationCalculationMode	= eCalculate;

	for(i = 0; i < m_ptrarrStates.size(); i++) // For all states in the STN
	{
		pState = m_ptrarrStates[i];

		if(i != m_ptrarrStates.size() - 1) // All states except the last
		{
			pStateTransition = pState->m_ptrarrStateTransitions[0];

			if(pStateTransition->m_Condition.Evaluate(&EC))
			{
				if(m_pActiveState != pState)
				{
					LogMessage(string("The condition: ") + pStateTransition->GetConditionAsString() + string(" is satisfied"), 0);
					SetActiveState(pState);
				}
				break;
			}
		}
		else if(i == m_ptrarrStates.size() - 1) // ELSE state (with no conditions)
		{
			if(m_pActiveState != pState)
			{
				SetActiveState(pState);
			}
			break;
		}
	}
	
	for(i = 0; i < m_pActiveState->m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_pActiveState->m_ptrarrSTNs[i];

		pSTN->ExecuteOnConditionActions();
	}
}

void daeIF::CheckState(daeState* pState)
{
//	daeStateTransition* pStateTransition;
//	
//	if(!pState)
//		daeDeclareAndThrowException(exInvalidPointer);
//
//// Only if the current state is not equal to the active state change the state (no reinitialization)
//// but continue searching for state change in the nested IF/STNs
//	if(m_pActiveState != pState)
//	{
//		pStateTransition = pState->m_ptrarrStateTransitions[0];
//		LogMessage(string("The condition: ") + pStateTransition->GetConditionAsString() + string(" is satisfied"), 0);
//		SetActiveState(pState);
//	}
}

daeState* daeIF::AddState(string strName)
{
// Remove previous state (if any) from the stack
	m_pModel->RemoveStateFromStack();

// Instantiate a new state and add it to STN
	daeState* pState = new daeState();

	dae_push_back(m_ptrarrStates, pState);
	
	string strStateName = "State_" + toString<size_t>(m_ptrarrStates.size());
	pState->Create(strStateName, this);
	pState->m_strCanonicalName = m_strCanonicalName + "." + pState->m_strShortName;

// Put it on the stack
	m_pModel->PutStateToStack(pState);
	return pState;
}

daeState* daeIF::CreateElse(void)
{
	string strName = "ELSE";
	daeState* pELSE = daeSTN::AddState(strName);
// Override default State_XXX name with ELSE
	pELSE->SetName(strName);
	pELSE->SetCanonicalName(m_strCanonicalName + "." + strName);
	return pELSE;
}

bool daeIF::CheckObject(vector<string>& strarrErrors) const
{
	size_t i;
	string strError;

	bool bCheck = true;

// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// If parent state is not null then it is nested stn	
//	if(m_pParentState);
	
// Check the active state	
	if(!m_pActiveState)
	{
		strError = "Invalid active state in if block [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
// Check the type	
	if(m_eSTNType == eSTNTUnknown)
	{
		strError = "Invalid type in if block [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}
	
// Check whether the stn is initialized	
	if(!m_bInitialized)
	{
		strError = "Uninitialized if block [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}

// Check number of states	
	if(m_ptrarrStates.size() < 2)
	{
		strError = "Invalid number of states if block [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}

// Check states	
	daeState* pState;	
	size_t nNoEquationsInEachState = 0;

	for(i = 0; i < m_ptrarrStates.size(); i++)
	{
		pState = m_ptrarrStates[i];
		if(!pState)
		{
			strError = "Invalid state in if block [" + GetCanonicalName() + "]";
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
				strError = "Number of equations must be the same in all states in if block [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
			}
		}

		if(i != m_ptrarrStates.size()-1)
		{
		// Number of state transitions in each state except the last must be 1
			if(pState->m_ptrarrStateTransitions.size() != 1)
			{	
				strError = "States in if block must contain only one state transition in if block [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
			}
		}
		else
		{
		// Number of state transitions in the last state (ELSE) has to be 0
			if(pState->m_ptrarrStateTransitions.size() != 0)
			{	
				strError = "The last in if block mustn't contain any state transitions in if block [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
			}
				
		// Name must be ELSE
			if(pState->GetName() != string("ELSE"))
			{	
				strError = "The last state must be ELSE in if block [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
			}
		}

		if(!pState->CheckObject(strarrErrors))
			bCheck = false;
	}
	
	return bCheck;
}



}
}

