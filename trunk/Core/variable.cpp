#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "limits.h"

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	daeVariable
**********************************************************************************************/
daeVariable::daeVariable()
{
	m_bReportingOn		= false;
	m_nOverallIndex		= ULONG_MAX;
	m_pModel			= NULL;
}

daeVariable::daeVariable(string strName, const daeVariableType& varType, daeModel* pModel, string strDescription)
{
	m_bReportingOn		= false;
	m_nOverallIndex		= ULONG_MAX;
	m_pModel			= NULL;

	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	pModel->AddVariable(*this, strName, varType, strDescription);
}

daeVariable::daeVariable(string strName, const daeVariableType& varType, daePort* pPort, string strDescription)
{
	m_bReportingOn		= false;
	m_nOverallIndex		= ULONG_MAX;
	m_pModel			= NULL;

	if(!pPort)
		daeDeclareAndThrowException(exInvalidPointer);
	pPort->AddVariable(*this, strName, varType, strDescription);
}

daeVariable::~daeVariable()
{
}

real_t daeVariable::GetValueAt(size_t nIndex) const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer); 

	return *m_pModel->m_pDataProxy->GetValue(nIndex);
}

// This function shouldn't be called during eCalculate, eCalculateIFSTN and eGatherInfo
real_t daeVariable::GetADValueAt(size_t nIndex) const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeExecutionContext* pExecutionContext = m_pModel->m_pDataProxy->GetExecutionContext(nIndex);
	if(!pExecutionContext)
	{	
		daeDeclareException(exInvalidPointer); 
		e << "pExecutionContext";
		throw e;
	}

	if(pExecutionContext->m_nCurrentVariableIndexForJacobianEvaluation == nIndex)
		return 1.0;
	else
		return 0.0;
}

void daeVariable::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);	
	
	m_ptrDomains.clear();

	daeObject::Open(pTag);

	strName = "Type";
	string strVarType;
	pTag->Open(strName, strVarType);

	strName = "DomainRefs";
	daeFindDomainByID del(m_pModel);
	pTag->OpenObjectRefArray(strName, m_ptrDomains, &del);
}

void daeVariable::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "VariableType";
	pTag->Save(strName, m_VariableType.m_strName);

	strName = "DomainRefs";
	pTag->SaveObjectRefArray(strName, m_ptrDomains);
}

void daeVariable::OpenRuntime(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);	
	
	m_ptrDomains.clear();

	daeObject::Open(pTag);

	strName = "Type";
	string strVarType;
	pTag->Open(strName, strVarType);

	strName = "DomainRefs";
	daeFindDomainByID del(m_pModel);
	pTag->OpenObjectRefArray(strName, m_ptrDomains, &del);

	strName = "OverallIndex";
	pTag->Open(strName, m_nOverallIndex);

// I should recreate all Runtime nodes here
}

void daeVariable::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::SaveRuntime(pTag);

	strName = "VariableType";
	pTag->Save(strName, m_VariableType.m_strName);

	strName = "DomainRefs";
	pTag->SaveObjectRefArray(strName, m_ptrDomains);

	strName = "OverallIndex";
	pTag->Save(strName, m_nOverallIndex);
}

real_t daeVariable::GetInitialCondition(const size_t* indexes, const size_t N)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, N);
	return *m_pModel->m_pDataProxy->GetInitialCondition(nIndex);
}

void daeVariable::SetAbsoluteTolerances(real_t dAbsTolerances)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nTotalNumberOfVariables = GetNumberOfPoints();
	for(size_t i = 0; i < nTotalNumberOfVariables; i++)
	{
		m_pModel->m_pDataProxy->SetAbsoluteTolerance(m_nOverallIndex + i, dAbsTolerances);
	}	
}

void daeVariable::SetInitialGuesses(real_t dInitialGuesses)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nTotalNumberOfVariables = GetNumberOfPoints();
	for(size_t i = 0; i < nTotalNumberOfVariables; i++)
	{
		m_pModel->m_pDataProxy->SetValue(m_nOverallIndex + i, dInitialGuesses);
	}	
}

void daeVariable::SetInitialGuess(real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	m_pModel->m_pDataProxy->SetValue(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	m_pModel->m_pDataProxy->SetValue(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	m_pModel->m_pDataProxy->SetValue(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	m_pModel->m_pDataProxy->SetValue(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	m_pModel->m_pDataProxy->SetValue(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	m_pModel->m_pDataProxy->SetValue(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	m_pModel->m_pDataProxy->SetValue(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	m_pModel->m_pDataProxy->SetValue(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	m_pModel->m_pDataProxy->SetValue(nIndex, dInitialGuess);
}

void daeVariable::SetValue(real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::SetValue(size_t nD1, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::AssignValue(real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnFixed);
}

void daeVariable::AssignValue(size_t nD1, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnFixed);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnFixed);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnFixed);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnFixed);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnFixed);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnFixed);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnFixed);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnFixed);
}

void daeVariable::ReAssignValue(real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnFixed)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reassign the value of the state variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnFixed)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reassign the value of the state variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnFixed)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reassign the value of the state variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnFixed)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reassign the value of the state variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnFixed)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reassign the value of the state variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnFixed)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reassign the value of the state variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnFixed)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reassign the value of the state variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnFixed)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reassign the value of the state variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnFixed)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reassign the value of the state variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}







real_t daeVariable::GetValue(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	return (*m_pModel->m_pDataProxy->GetValue(nIndex));
}

real_t daeVariable::GetValue(size_t nD1)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	return (*m_pModel->m_pDataProxy->GetValue(nIndex));
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	return (*m_pModel->m_pDataProxy->GetValue(nIndex));
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	return (*m_pModel->m_pDataProxy->GetValue(nIndex));
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	return (*m_pModel->m_pDataProxy->GetValue(nIndex));
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	return (*m_pModel->m_pDataProxy->GetValue(nIndex));
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	return (*m_pModel->m_pDataProxy->GetValue(nIndex));
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	return (*m_pModel->m_pDataProxy->GetValue(nIndex));
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	return (*m_pModel->m_pDataProxy->GetValue(nIndex));
}

void daeVariable::SetInitialCondition(real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialCondition(size_t nD1, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::ReSetInitialCondition(real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

void daeVariable::ReSetInitialCondition(size_t nD1, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << m_strCanonicalName << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

daeVariableType_t* daeVariable::GetVariableType(void)
{
	return &m_VariableType;
}
	
void daeVariable::SetVariableType(const daeVariableType& VariableType)
{
	m_VariableType = VariableType;
}

void daeVariable::DistributeOnDomain(daeDomain& rDomain)
{
	if(!(&rDomain))
		daeDeclareAndThrowException(exInvalidPointer);
	m_ptrDomains.push_back(&rDomain);
}

void daeVariable::GetDomains(vector<daeDomain_t*>& ptrarrDomains)
{
	ptrarrDomains.clear();
	for(size_t i = 0; i < m_ptrDomains.size(); i++)
		ptrarrDomains.push_back(m_ptrDomains[i]);
}

real_t* daeVariable::GetValuePointer() const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	return m_pModel->m_pDataProxy->GetValue(m_nOverallIndex);
}

bool daeVariable::GetReportingOn(void) const
{
	return m_bReportingOn;
}

void daeVariable::SetReportingOn(bool bOn)
{
	m_bReportingOn = bOn;
}

real_t daeVariable::TimeDerivative(void)
{
	return dt().getValue();
}

real_t daeVariable::TimeDerivative(size_t nD1)
{
	return dt(nD1).getValue();
}
	
real_t daeVariable::TimeDerivative(size_t nD1, size_t nD2)
{
	return dt(nD1, nD2).getValue();
}
	
real_t daeVariable::TimeDerivative(size_t nD1, size_t nD2, size_t nD3)
{
	return dt(nD1, nD2, nD3).getValue();
}

real_t daeVariable::TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4)
{
	return dt(nD1, nD2, nD3, nD4).getValue();
}

real_t daeVariable::TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5)
{
	return dt(nD1, nD2, nD3, nD4, nD5).getValue();
}

real_t daeVariable::TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6)
{
	return dt(nD1, nD2, nD3, nD4, nD5, nD6).getValue();
}

real_t daeVariable::TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7)
{
	return dt(nD1, nD2, nD3, nD4, nD5, nD6, nD7).getValue();
}

real_t daeVariable::TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8)
{
	return dt(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8).getValue();
}

real_t daeVariable::PartialDerivative1(const daeDomain_t& rDomain, size_t nD1)
{
	return d(rDomain, nD1).getValue();
}
	
real_t daeVariable::PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2)
{
	return d(rDomain, nD1, nD2).getValue();
}
	
real_t daeVariable::PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3)
{
	return d(rDomain, nD1, nD2, nD3).getValue();
}
	
real_t daeVariable::PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4)
{
	return d(rDomain, nD1, nD2, nD3, nD4).getValue();
}

real_t daeVariable::PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5)
{
	return d(rDomain, nD1, nD2, nD3, nD4, nD5).getValue();
}

real_t daeVariable::PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6)
{
	return d(rDomain, nD1, nD2, nD3, nD4, nD5, nD6).getValue();
}

real_t daeVariable::PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7)
{
	return d(rDomain, nD1, nD2, nD3, nD4, nD5, nD6, nD7).getValue();
}

real_t daeVariable::PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8)
{
	return d(rDomain, nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8).getValue();
}


real_t daeVariable::PartialDerivative2(const daeDomain_t& rDomain, size_t nD1)
{
	return d2(rDomain, nD1).getValue();
}
	
real_t daeVariable::PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2)
{
	return d2(rDomain, nD1, nD2).getValue();
}
	
real_t daeVariable::PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3)
{
	return d2(rDomain, nD1, nD2, nD3).getValue();
}
	
real_t daeVariable::PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4)
{
	return d2(rDomain, nD1, nD2, nD3, nD4).getValue();
}
	
real_t daeVariable::PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5)
{
	return d2(rDomain, nD1, nD2, nD3, nD4, nD5).getValue();
}

real_t daeVariable::PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6)
{
	return d2(rDomain, nD1, nD2, nD3, nD4, nD5, nD6).getValue();
}

real_t daeVariable::PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7)
{
	return d2(rDomain, nD1, nD2, nD3, nD4, nD5, nD6, nD7).getValue();
}

real_t daeVariable::PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8)
{
	return d2(rDomain, nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8).getValue();
}

bool daeVariable::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check variable type	
	if(!m_VariableType.CheckObject(strarrErrors))
	{
		strError = "Invalid variable type in variable [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
// Check index	
	if(m_nOverallIndex == ULONG_MAX)
	{
		strError = "Invalid variable index in variable [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check domains	
	if(m_ptrDomains.size() == 0)
	{
		daeDomain* pDomain;	
		for(size_t i = 0; i < m_ptrDomains.size(); i++)
		{
			pDomain = m_ptrDomains[i];
			if(!pDomain)
			{
				strError = "Invalid domain in variable [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
			
			if(!pDomain->CheckObject(strarrErrors))
				bCheck = false;
		}
	}

	return bCheck;
}


}
}
