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
	m_pParentPort       = NULL;
}

daeVariable::daeVariable(string strName, 
						 const daeVariableType& varType, 
						 daeModel* pModel, 
						 string strDescription, 
						 daeDomain* d1, daeDomain* d2, daeDomain* d3, daeDomain* d4, daeDomain* d5, daeDomain* d6, daeDomain* d7, daeDomain* d8)
{
	m_bReportingOn		= false;
	m_nOverallIndex		= ULONG_MAX;
	m_pModel			= pModel;
	m_pParentPort       = NULL;

	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	pModel->AddVariable(*this, strName, varType, strDescription);
	
	m_ptrDomains = dae::makeVector<daeDomain*>(d1, d2, d3, d4, d5, d6, d7, d8);
}

daeVariable::daeVariable(string strName, 
						 const daeVariableType& varType, 
						 daePort* pPort, 
						 string strDescription, 
						 daeDomain* d1, daeDomain* d2, daeDomain* d3, daeDomain* d4, daeDomain* d5, daeDomain* d6, daeDomain* d7, daeDomain* d8)
{
	m_bReportingOn		= false;
	m_nOverallIndex		= ULONG_MAX;
	m_pModel			= NULL;
	m_pParentPort       = pPort;

	if(!pPort)
		daeDeclareAndThrowException(exInvalidPointer);
	pPort->AddVariable(*this, strName, varType, strDescription);
	
	m_ptrDomains = dae::makeVector<daeDomain*>(d1, d2, d3, d4, d5, d6, d7, d8);
}

daeVariable::~daeVariable()
{
}

void daeVariable::Clone(const daeVariable& rObject)
{
	m_bReportingOn  = rObject.m_bReportingOn;
	m_nOverallIndex = ULONG_MAX;
	m_VariableType  = rObject.m_VariableType;
	FindDomains(rObject.m_ptrDomains, m_ptrDomains, m_pModel);
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

void daeVariable::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport, strDomains;
	boost::format fmtFile(strExport);

	if(c.m_bExportDefinition)
	{
		if(eLanguage == ePYDAE)
		{
		}
		else if(eLanguage == eCDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "daeVariable %1%;\n";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName();
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}		
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			if(!m_ptrDomains.empty())
				strDomains = ", [" + toString_StrippedRelativeNames<daeDomain*, daeModel*>(m_ptrDomains, m_pModel, "self.") + "]";
			
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.%1% = daeVariable(\"%2%\", %3%, self, \"%4%\"%5%)\n";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% m_strShortName 
					% m_VariableType.GetName() 
					% m_strDescription
					% strDomains;
		}
		else if(eLanguage == eCDAE)
		{
			if(!m_ptrDomains.empty())
				strDomains = ", " + toString_StrippedRelativeNames<daeDomain*, daeModel*>(m_ptrDomains, m_pModel, "&");
			
			strExport = ",\n" + c.CalculateIndent(c.m_nPythonIndentLevel) + "%1%(\"%2%\", %3%, this, \"%4%\"%5%)";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% m_strShortName 
					% m_VariableType.GetName() 
					% m_strDescription
					% strDomains;
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += fmtFile.str();
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

string daeVariable::GetCanonicalName(void) const
{
	if(m_pParentPort)
		return m_pParentPort->GetCanonicalName() + '.' + m_strShortName;
	else
		return daeObject::GetCanonicalName();
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

void daeVariable::SetInitialGuess(const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuess(value);
}

void daeVariable::SetInitialGuess(size_t nD1, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuess(nD1, value);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuess(nD1, nD2, value);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuess(nD1, nD2, nD3, value);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuess(nD1, nD2, nD3, nD4, value);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuess(nD1, nD2, nD3, nD4, nD5, value);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuess(nD1, nD2, nD3, nD4, nD5, nD6, value);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuess(nD1, nD2, nD3, nD4, nD5, nD6, nD7, value);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuess(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8, value);
}

void daeVariable::SetInitialGuesses(const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuesses(value);
}

void daeVariable::SetValue(const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetValue(value);
}

void daeVariable::SetValue(size_t nD1, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetValue(nD1, value);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetValue(nD1, nD2, value);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetValue(nD1, nD2, nD3, value);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetValue(nD1, nD2, nD3, nD4, value);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetValue(nD1, nD2, nD3, nD4, nD5, value);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetValue(nD1, nD2, nD3, nD4, nD5, nD6, value);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, value);
}

void daeVariable::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8, value);
}

quantity daeVariable::GetQuantity(void)
{
	real_t value = GetValue();
	return quantity(value, m_VariableType.GetUnits());
}

quantity daeVariable::GetQuantity(size_t nD1)
{
	real_t value = GetValue(nD1);
	return quantity(value, m_VariableType.GetUnits());
}

quantity daeVariable::GetQuantity(size_t nD1, size_t nD2)
{
	real_t value = GetValue(nD1, nD2);
	return quantity(value, m_VariableType.GetUnits());
}

quantity daeVariable::GetQuantity(size_t nD1, size_t nD2, size_t nD3)
{
	real_t value = GetValue(nD1, nD2, nD3);
	return quantity(value, m_VariableType.GetUnits());
}

quantity daeVariable::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4);
	return quantity(value, m_VariableType.GetUnits());
}

quantity daeVariable::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4, nD5);
	return quantity(value, m_VariableType.GetUnits());
}

quantity daeVariable::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4, nD5, nD6);
	return quantity(value, m_VariableType.GetUnits());
}

quantity daeVariable::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7);
	return quantity(value, m_VariableType.GetUnits());
}

quantity daeVariable::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8);
	return quantity(value, m_VariableType.GetUnits());
}

void daeVariable::AssignValues(real_t dValues)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nTotalNumberOfVariables = GetNumberOfPoints();
	for(size_t i = 0; i < nTotalNumberOfVariables; i++)
	{
		m_pModel->m_pDataProxy->SetValue(m_nOverallIndex + i, dValues);
		m_pModel->m_pDataProxy->SetVariableType(m_nOverallIndex + i, cnFixed);
	}	
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

void daeVariable::AssignValues(const quantity& q)
{
	real_t values = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValues(values);
}

void daeVariable::AssignValue(const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValue(value);
}

void daeVariable::AssignValue(size_t nD1, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValue(nD1, value);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValue(nD1, nD2, value);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValue(nD1, nD2, nD3, value);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValue(nD1, nD2, nD3, nD4, value);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValue(nD1, nD2, nD3, nD4, nD5, value);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValue(nD1, nD2, nD3, nD4, nD5, nD6, value);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, value);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8, value);
}

void daeVariable::ReAssignValues(real_t dValues)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nTotalNumberOfVariables = GetNumberOfPoints();
	for(size_t i = 0; i < nTotalNumberOfVariables; i++)
	{
		if(m_pModel->m_pDataProxy->GetVariableType(m_nOverallIndex + i) != cnFixed)
		{
			daeDeclareException(exInvalidCall);
			e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
			throw e;
		}		
		m_pModel->m_pDataProxy->SetValue(m_nOverallIndex + i, dValues);
	}
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
		e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reassign the value of the state variable for [" << GetCanonicalName() << "] variable";
		throw e;
	}		
	m_pModel->m_pDataProxy->SetValue(nIndex, dValue);
}

void daeVariable::ReAssignValues(const quantity& q)
{
	real_t values = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValues(values);
}

void daeVariable::ReAssignValue(const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValue(value);
}

void daeVariable::ReAssignValue(size_t nD1, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValue(nD1, value);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValue(nD1, nD2, value);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValue(nD1, nD2, nD3, value);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValue(nD1, nD2, nD3, nD4, value);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValue(nD1, nD2, nD3, nD4, nD5, value);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValue(nD1, nD2, nD3, nD4, nD5, nD6, value);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, value);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8, value);
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

void daeVariable::SetInitialConditions(real_t dInitialConditions)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nTotalNumberOfVariables = GetNumberOfPoints();
	for(size_t i = 0; i < nTotalNumberOfVariables; i++)
	{
		if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
			m_pModel->m_pDataProxy->SetValue(m_nOverallIndex + i, dInitialConditions);
		else
			m_pModel->m_pDataProxy->SetTimeDerivative(m_nOverallIndex + i, dInitialConditions);
		
		m_pModel->m_pDataProxy->SetVariableType(m_nOverallIndex + i, cnDifferential);
	}
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

void daeVariable::SetInitialConditions(const quantity& q)
{
	real_t values = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialConditions(values);
}

void daeVariable::SetInitialCondition(const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialCondition(value);
}

void daeVariable::SetInitialCondition(size_t nD1, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialCondition(nD1, value);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialCondition(nD1, nD2, value);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialCondition(nD1, nD2, nD3, value);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialCondition(nD1, nD2, nD3, nD4, value);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialCondition(nD1, nD2, nD3, nD4, nD5, value);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialCondition(nD1, nD2, nD3, nD4, nD5, nD6, value);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialCondition(nD1, nD2, nD3, nD4, nD5, nD6, nD7, value);
}

void daeVariable::SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialCondition(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8, value);
}

void daeVariable::ReSetInitialConditions(real_t dInitialConditions)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nTotalNumberOfVariables = GetNumberOfPoints();
	for(size_t i = 0; i < nTotalNumberOfVariables; i++)
	{
		if(m_pModel->m_pDataProxy->GetVariableType(m_nOverallIndex + i) != cnDifferential)
		{
			daeDeclareException(exInvalidCall);
			e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
			throw e;
		}		

		if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
			m_pModel->m_pDataProxy->SetValue(m_nOverallIndex + i, dInitialConditions);
		else
			m_pModel->m_pDataProxy->SetTimeDerivative(m_nOverallIndex + i, dInitialConditions);
	}
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
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
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
		e << "Invalid call: you cannot reset initial condition of the non-differential variable for [" << GetCanonicalName() << "] variable";
		throw e;
	}		

	if(m_pModel->GetInitialConditionMode() == eAlgebraicValuesProvided)
		m_pModel->m_pDataProxy->SetValue(nIndex, dInitialCondition);
	else
		m_pModel->m_pDataProxy->SetTimeDerivative(nIndex, dInitialCondition);
}

void daeVariable::ReSetInitialConditions(const quantity& q)
{
	real_t values = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialConditions(values);
}

void daeVariable::ReSetInitialCondition(const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialCondition(value);
}

void daeVariable::ReSetInitialCondition(size_t nD1, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialCondition(nD1, value);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialCondition(nD1, nD2, value);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialCondition(nD1, nD2, nD3, value);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialCondition(nD1, nD2, nD3, nD4, value);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialCondition(nD1, nD2, nD3, nD4, nD5, value);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialCondition(nD1, nD2, nD3, nD4, nD5, nD6, value);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialCondition(nD1, nD2, nD3, nD4, nD5, nD6, nD7, value);
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialCondition(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8, value);
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
	dae_push_back(m_ptrDomains, &rDomain);
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

/*********************************************************************************************
	daeVariableWrapper
**********************************************************************************************/
daeVariableWrapper::daeVariableWrapper() 
{
	m_nOverallIndex = ULONG_MAX;
	m_pVariable     = NULL;
}

daeVariableWrapper::daeVariableWrapper(daeVariable& variable, std::string strName) 
{
	std::vector<size_t>	narrDomainIndexes;
	Initialize(&variable, strName, narrDomainIndexes);
}

daeVariableWrapper::daeVariableWrapper(adouble& a, std::string strName) 
{
	daeVariable* pVariable;
	std::vector<size_t>	narrDomainIndexes;
	
	daeGetVariableAndIndexesFromNode(a, &pVariable, narrDomainIndexes);
	if(!pVariable)
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid expression for daeVariableWrapper (cannot get variable)";
		throw e;
	}
	
	Initialize(pVariable, strName, narrDomainIndexes);
}

daeVariableWrapper::~daeVariableWrapper(void)
{
}

void daeVariableWrapper::Initialize(daeVariable* pVariable, std::string strName, const std::vector<size_t>& narrDomainIndexes)
{
	if(!pVariable)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pVariable->m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pVariable->m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_pVariable     = pVariable;
	m_pDataProxy    = pVariable->m_pModel->m_pDataProxy;
	m_nOverallIndex = pVariable->m_nOverallIndex + pVariable->CalculateIndex(narrDomainIndexes);
	
	//std::cout << "daeVariableWrapper::m_nOverallIndex = " << m_nOverallIndex << std::endl;
	
	if(strName.empty())
	{
		m_strName = pVariable->GetName();
		if(!narrDomainIndexes.empty())
			m_strName += "(" + toString(narrDomainIndexes, string(",")) + ")";
	}
	else
		m_strName = strName;
}

string daeVariableWrapper::GetName(void) const
{
	return m_strName;
}

real_t daeVariableWrapper::GetValue(void) const
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
	return *m_pDataProxy->GetValue(m_nOverallIndex);
}

void daeVariableWrapper::SetValue(real_t value)
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_pDataProxy->SetValue(m_nOverallIndex, value);
}


}
}
