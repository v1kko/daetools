#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	daeState
**********************************************************************************************/
daeState::daeState()
{
	m_pModel  = NULL;
	m_pSTN    = NULL;
}

daeState::~daeState()
{
}

void daeState::Clone(const daeState& rObject)
{
	for(size_t i = 0; i < rObject.m_ptrarrStateTransitions.size(); i++)
	{
		daeStateTransition* pStateTransition = new daeStateTransition();
		pStateTransition->SetName(rObject.m_ptrarrStateTransitions[i]->m_strShortName);
		AddStateTransition(pStateTransition);		
		pStateTransition->Clone(*rObject.m_ptrarrStateTransitions[i]);
	}

	for(size_t i = 0; i < rObject.m_ptrarrEquations.size(); i++)
	{
		daeEquation* pEquation = m_pModel->CreateEquation(rObject.m_ptrarrEquations[i]->m_strShortName,
														  rObject.m_ptrarrEquations[i]->m_strDescription,
		                                                  rObject.m_ptrarrEquations[i]->m_dScaling);
		pEquation->Clone(*rObject.m_ptrarrEquations[i]);
	}

	for(size_t i = 0; i < rObject.m_ptrarrSTNs.size(); i++)
	{
		daeSTN* pSTN = m_pModel->AddSTN(rObject.m_ptrarrSTNs[i]->m_strShortName);
		pSTN->Clone(*rObject.m_ptrarrSTNs[i]);
	}

	for(size_t i = 0; i < rObject.m_ptrarrOnEventActions.size(); i++)
	{
		daeOnEventActions* pOnEventActions = new daeOnEventActions();
		dae_push_back(m_ptrarrOnEventActions, pOnEventActions);
		pOnEventActions->Clone(*rObject.m_ptrarrOnEventActions[i]);
	}
}

void daeState::CleanUpSetupData()
{
	clean_vector(m_ptrarrEquations);
	
	for(size_t i = 0; i < m_ptrarrSTNs.size(); i++)
		m_ptrarrSTNs[i]->CleanUpSetupData();
}

void daeState::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 

	m_ptrarrEquations.EmptyAndFreeMemory();
	m_ptrarrStateTransitions.EmptyAndFreeMemory();
	m_ptrarrSTNs.EmptyAndFreeMemory();

	m_ptrarrEquations.SetOwnershipOnPointers(true);
	m_ptrarrStateTransitions.SetOwnershipOnPointers(true);
	m_ptrarrSTNs.SetOwnershipOnPointers(true);

	daeObject::Open(pTag);

	daeSetModelAndCanonicalNameDelegate<daeObject> del(this, m_pModel);

	strName = "Equations";
	pTag->OpenObjectArray(strName, m_ptrarrEquations, &del);

	strName = "STNs";
	pTag->OpenObjectArray(strName, m_ptrarrSTNs, &del);

	strName = "StateTransitions";
	pTag->OpenObjectArray(strName, m_ptrarrStateTransitions, &del);

	strName = "OnEventActions";
	pTag->OpenObjectArray(strName, m_ptrarrOnEventActions, &del);

	strName = "STN";
	daeFindSTNByID stndel(m_pModel);
	m_pSTN = pTag->OpenObjectRef<daeSTN>(strName, &stndel);
}

void daeState::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "Equations";
	pTag->SaveObjectArray(strName, m_ptrarrEquations);

	strName = "STNs";
	pTag->SaveObjectArray(strName, m_ptrarrSTNs);

	strName = "StateTransitions";
	pTag->SaveObjectArray(strName, m_ptrarrStateTransitions);

	strName = "OnEventActions";
	pTag->SaveObjectArray(strName, m_ptrarrOnEventActions);

	strName = "STN";
	pTag->SaveObjectRef(strName, m_pSTN);
}

void daeState::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport, strEquations, strStateTransitions, strSTNs;
	boost::format fmtFile;

	if(c.m_bExportDefinition)
	{
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.STATE(\"%1%\")\n" + 
						"%2%"+
						"%3%"+
						"%4%\n";
			ExportObjectArray(m_ptrarrEquations,        strEquations,        eLanguage, c);
			ExportObjectArray(m_ptrarrSTNs,             strSTNs,             eLanguage, c);
			ExportObjectArray(m_ptrarrStateTransitions, strStateTransitions, eLanguage, c);
			
			fmtFile.parse(strExport);
			fmtFile % m_strShortName
					% strEquations 
					% strSTNs 
					% strStateTransitions;
		}
		else if(eLanguage == eCDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "STATE(\"%1%\");\n" + 
						"%2%"+
						"%3%"+
						"%4%\n";
			ExportObjectArray(m_ptrarrEquations,        strEquations,        eLanguage, c);
			ExportObjectArray(m_ptrarrSTNs,             strSTNs,             eLanguage, c);
			ExportObjectArray(m_ptrarrStateTransitions, strStateTransitions, eLanguage, c);
			
			fmtFile.parse(strExport);
			fmtFile % m_strShortName
					% strEquations 
					% strSTNs 
					% strStateTransitions;
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += fmtFile.str();
}

void daeState::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daeState::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::SaveRuntime(pTag);
	
	strName = "Equations";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrEquations);

	strName = "STNs";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrSTNs);

	strName = "StateTransitions";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrStateTransitions);

	strName = "STN";
	pTag->SaveObjectRef(strName, m_pSTN);
}

string daeState::GetCanonicalName(void) const
{
	if(m_pSTN)
		return m_pSTN->GetCanonicalName() + '.' + m_strShortName;
	else
		return m_strShortName;
}

void daeState::Create(const string& strName, daeSTN* pSTN)
{
	m_pModel		= pSTN->m_pModel;
	m_pSTN			= pSTN;
	m_strShortName	= strName;
}

void daeState::InitializeStateTransitions(void)
{
	size_t i;
	daeSTN *pSTN;
	daeStateTransition *pST;
	
	for(i = 0; i < m_ptrarrStateTransitions.size(); i++)
	{
		pST = m_ptrarrStateTransitions[i];
		if(!pST)
			daeDeclareAndThrowException(exInvalidPointer);
	
		pST->Initialize();
	}
	
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer);
		pSTN->InitializeStateTransitions();
	}
	
// Initialize OnEventActions
	daeOnEventActions* pOnEventActions;
	for(i = 0; i < m_ptrarrOnEventActions.size(); i++)
	{
		pOnEventActions = m_ptrarrOnEventActions[i];
		if(!pOnEventActions)
			daeDeclareAndThrowException(exInvalidPointer);

		pOnEventActions->Initialize();
	}
}

void daeState::InitializeDEDIs(void)
{
	size_t i;
	daeSTN *pSTN;
	daeEquation* pEquation;
	
	for(i = 0; i < m_ptrarrEquations.size(); i++)
	{
		pEquation = m_ptrarrEquations[i];
		if(!pEquation)
			daeDeclareAndThrowException(exInvalidPointer);

		pEquation->InitializeDEDIs();
	}
	
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer);
		pSTN->InitializeDEDIs();
	}
}

daeSTN* daeState::GetSTN(void) const
{
	return m_pSTN;
}

void daeState::SetSTN(daeSTN* pSTN)
{
	m_pSTN = pSTN;
}

void daeState::AddEquation(daeEquation* pEquation)
{
    pEquation->SetModel(m_pModel);
	//SetModelAndCanonicalName(pEquation);
	pEquation->m_pParentState = this;
	dae_push_back(m_ptrarrEquations, pEquation);
}

void daeState::AddOnEventAction(daeOnEventActions& rOnEventAction, const string& strName, string strDescription)
{
	rOnEventAction.SetName(strName);
	rOnEventAction.SetDescription(strDescription);
    rOnEventAction.SetModel(m_pModel);
    //SetModelAndCanonicalName(&rOnEventAction);
	rOnEventAction.m_pParentState = this;
	
	if(CheckName(m_ptrarrOnEventActions, strName))
	{
		daeDeclareException(exInvalidCall); 
		e << "OnEventAction [" << strName << "] already exists in the state [" << GetCanonicalName() << "]";
		throw e;
	}
	dae_push_back(m_ptrarrOnEventActions, &rOnEventAction);
}

void daeState::ConnectOnEventActions(void)
{
	daeEventPort* pEventPort;
	daeOnEventActions* pOnEventActions;
	
	for(size_t i = 0; i < m_ptrarrOnEventActions.size(); i++)
	{
		pOnEventActions = m_ptrarrOnEventActions[i];
		pEventPort = pOnEventActions->GetEventPort();
		pEventPort->Attach(pOnEventActions);
	}
}

void daeState::DisconnectOnEventActions(void)
{
	daeEventPort* pEventPort;
	daeOnEventActions* pOnEventActions;
	
	for(size_t i = 0; i < m_ptrarrOnEventActions.size(); i++)
	{
		pOnEventActions = m_ptrarrOnEventActions[i];
		pEventPort = pOnEventActions->GetEventPort();
		pEventPort->Detach(pOnEventActions);
	}
}

//daeEquation* daeState::AddEquation(const string& strEquationExpression)
//{
//	daeEquation* pEquation = new daeEquation();
//	string strName = "Equation_" + toString<size_t>(m_ptrarrEquations.size());
//	pEquation->SetName(strName);
//	AddEquation(pEquation);
//	return pEquation;
//}

size_t daeState::GetNumberOfEquations(void) const
{
	return m_ptrarrEquations.size();
}

size_t daeState::GetNumberOfStateTransitions(void) const
{
	return m_ptrarrStateTransitions.size();
}

size_t daeState::GetNumberOfSTNs(void) const
{
	return m_ptrarrSTNs.size();
}

void daeState::AddNestedSTN(daeSTN* pSTN)
{
	//SetModelAndCanonicalName(pSTN);
    pSTN->SetModel(m_pModel);
	pSTN->SetParentState(this);
	dae_push_back(m_ptrarrSTNs, pSTN);
}

void daeState::AddStateTransition(daeStateTransition* pStateTransition)
{
	//SetModelAndCanonicalName(pStateTransition);
    pStateTransition->SetModel(m_pModel);
	dae_push_back(m_ptrarrStateTransitions, pStateTransition);
}
	
void daeState::GetStateTransitions(vector<daeStateTransition_t*>& ptrarrStateTransitions)
{
	dae_set_vector(m_ptrarrStateTransitions, ptrarrStateTransitions);
}
	
void daeState::GetEquations(vector<daeEquation_t*>& ptrarrEquations)
{
	dae_set_vector(m_ptrarrEquations, ptrarrEquations);
}
	
void daeState::GetNestedSTNs(vector<daeSTN_t*>& ptrarrSTNs)
{
	dae_set_vector(m_ptrarrSTNs, ptrarrSTNs);
}

void daeState::CalcNonZeroElements(int& NNZ)
{
	size_t i;
	daeSTN* pSTN;
	daeState* pState;
	daeEquationExecutionInfo* pEquationExecutionInfo;

// First find in normal equations (non-STN)
	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		NNZ += pEquationExecutionInfo->m_mapIndexes.size();
	}

// Then in nested STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->CalcNonZeroElements(NNZ);
	}
}

void daeState::FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix)
{
	size_t i;
	daeSTN* pSTN;
	daeEquationExecutionInfo* pEquationExecutionInfo;

// First find in normal equations (non-STN)
	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		pMatrix->AddRow(pEquationExecutionInfo->m_mapIndexes);
	}

// Then in nested STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->FillSparseMatrix(pMatrix);
	}
}

void daeState::AddIndexesFromAllEquations(std::vector< std::map<size_t, size_t> >& arrIndexes, size_t& nCurrentEquaton)
{
	size_t i;
	daeSTN* pSTN;
	daeEquationExecutionInfo* pEquationExecutionInfo;

// First find in normal equations (non-STN)
	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		arrIndexes[nCurrentEquaton].insert(pEquationExecutionInfo->m_mapIndexes.begin(), pEquationExecutionInfo->m_mapIndexes.end());
		nCurrentEquaton++;
	}

// Then in nested STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		pSTN->AddIndexesFromAllEquations(arrIndexes, nCurrentEquaton);
	}
}

bool daeState::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

	dae_capacity_check(m_ptrarrEquations);
	dae_capacity_check(m_ptrarrStateTransitions);
	dae_capacity_check(m_ptrarrSTNs);
	dae_capacity_check(m_ptrarrOnEventActions);
	dae_capacity_check(m_ptrarrEquationExecutionInfos);
	
// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check parent STN pointer	
	if(!m_pSTN)
	{
		strError = "Invalid parent state transition network in state [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
// Check number of equations	
	if(m_ptrarrEquations.size() == 0)
	{
		strError = "Invalid number of equations in state [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check equations	
	if(m_ptrarrEquations.size() > 0)
	{
		daeEquation* pEquation;	
		for(size_t i = 0; i < m_ptrarrEquations.size(); i++)
		{
			pEquation = m_ptrarrEquations[i];
			if(!pEquation)
			{
				strError = "Invalid equation in state [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
			
			if(!pEquation->CheckObject(strarrErrors))
				bCheck = false;
		}
	}

// Check state transitions	
	if(m_ptrarrStateTransitions.size() > 0)
	{
		daeStateTransition* pStateTransition;	
		for(size_t i = 0; i < m_ptrarrStateTransitions.size(); i++)
		{
			pStateTransition = m_ptrarrStateTransitions[i];
			if(!pStateTransition)
			{
				strError = "Invalid equation in state [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
			
			if(!pStateTransition->CheckObject(strarrErrors))
				bCheck = false;
		}
	}
	
// Check OnEventActions	
	if(m_ptrarrOnEventActions.size() > 0)
	{
		daeOnEventActions* pOnEventActions;
		for(size_t i = 0; i < m_ptrarrOnEventActions.size(); i++)
		{
			pOnEventActions = m_ptrarrOnEventActions[i];
			if(!pOnEventActions)
			{
				strError = "Invalid on event actions in state [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
	
			if(!pOnEventActions->CheckObject(strarrErrors))
				bCheck = false;
		}
	}
	
// Check nested STNs	
	if(m_ptrarrSTNs.size() > 0)
	{
		daeSTN* pSTN;	
		for(size_t i = 0; i < m_ptrarrSTNs.size(); i++)
		{
			pSTN = m_ptrarrSTNs[i];
			if(!pSTN)
			{
				strError = "Invalid equation in state [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
			
			if(!pSTN->CheckObject(strarrErrors))
				bCheck = false;
		}
	}

	return bCheck;
}

}
}
