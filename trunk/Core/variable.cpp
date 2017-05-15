#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "limits.h"
#include <limits>

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

string daeVariable::GetCanonicalNameAndPrepend(const std::string& prependToName) const
{
    if(m_pParentPort)
        return m_pParentPort->GetCanonicalName() + '.' + prependToName + m_strShortName;
    else
        return daeObject::GetCanonicalNameAndPrepend(prependToName);
}

daePort* daeVariable::GetParentPort(void) const
{
    return m_pParentPort;
}

const std::vector<daeDomain*>& daeVariable::Domains(void) const
{
    return m_ptrDomains;
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

void daeVariable::GetDomainsIndexesMap(std::map<size_t, std::vector<size_t> >& mapDomainsIndexes, size_t nIndexBase) const
{
    std::vector<size_t> narrDomainIndexes;
    size_t d1, d2, d3, d4, d5, d6, d7, d8;
	daeDomain *pDomain1, *pDomain2, *pDomain3, 
		      *pDomain4, *pDomain5, *pDomain6,
			  *pDomain7, *pDomain8;

    size_t nNoDomains    = m_ptrDomains.size();
    size_t nIndexCounter = 0;
    
    if(nNoDomains == 0)
    {
        narrDomainIndexes.clear();
        mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
    }
    else if(nNoDomains == 1)
    {
        pDomain1 = m_ptrDomains[0];
        if(!pDomain1)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(1);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 2)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        if(!pDomain1 || !pDomain2)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(2);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 3)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        if(!pDomain1 || !pDomain2 || !pDomain3)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(3);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 4)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(4);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 5)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        pDomain5 = m_ptrDomains[4];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4 || !pDomain5)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(5);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        for(d5 = 0; d5 < pDomain5->m_nNumberOfPoints; d5++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            narrDomainIndexes[4] = nIndexBase + d5;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 6)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        pDomain5 = m_ptrDomains[4];
        pDomain6 = m_ptrDomains[5];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4 || 
           !pDomain5 || !pDomain6)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(6);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        for(d5 = 0; d5 < pDomain5->m_nNumberOfPoints; d5++)
        for(d6 = 0; d6 < pDomain6->m_nNumberOfPoints; d6++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            narrDomainIndexes[4] = nIndexBase + d5;
            narrDomainIndexes[5] = nIndexBase + d6;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 7)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        pDomain5 = m_ptrDomains[4];
        pDomain6 = m_ptrDomains[5];
        pDomain7 = m_ptrDomains[6];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4 || 
           !pDomain5 || !pDomain6 || !pDomain7)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(7);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        for(d5 = 0; d5 < pDomain5->m_nNumberOfPoints; d5++)
        for(d6 = 0; d6 < pDomain6->m_nNumberOfPoints; d6++)
        for(d7 = 0; d7 < pDomain7->m_nNumberOfPoints; d7++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            narrDomainIndexes[4] = nIndexBase + d5;
            narrDomainIndexes[5] = nIndexBase + d6;
            narrDomainIndexes[6] = nIndexBase + d7;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 8)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        pDomain5 = m_ptrDomains[4];
        pDomain6 = m_ptrDomains[5];
        pDomain7 = m_ptrDomains[6];
        pDomain8 = m_ptrDomains[7];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4 || 
           !pDomain5 || !pDomain6 || !pDomain7 || !pDomain8)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(8);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        for(d5 = 0; d5 < pDomain5->m_nNumberOfPoints; d5++)
        for(d6 = 0; d6 < pDomain6->m_nNumberOfPoints; d6++)
        for(d7 = 0; d7 < pDomain7->m_nNumberOfPoints; d7++)
        for(d8 = 0; d8 < pDomain8->m_nNumberOfPoints; d8++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            narrDomainIndexes[4] = nIndexBase + d5;
            narrDomainIndexes[5] = nIndexBase + d6;
            narrDomainIndexes[6] = nIndexBase + d7;
            narrDomainIndexes[7] = nIndexBase + d8;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
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
		m_pModel->m_pDataProxy->SetInitialGuess(m_nOverallIndex + i, dInitialGuesses);
	}	
}

void daeVariable::SetInitialGuesses(const std::vector<real_t>& initialGuesses)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nTotalNumberOfVariables = GetNumberOfPoints();
    if(initialGuesses.size() != nTotalNumberOfVariables)
        daeDeclareAndThrowException(exInvalidCall);

    for(size_t i = 0; i < nTotalNumberOfVariables; i++)
    {
        // Skip if it is equal to cnUnsetValue (by design: that means unset value, with value equal to DOUBLE_MAX)
        if(initialGuesses[i] == cnUnsetValue)
            continue;

        m_pModel->m_pDataProxy->SetInitialGuess(m_nOverallIndex + i, initialGuesses[i]);
    }
}

void daeVariable::SetInitialGuess(real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialGuess)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
}

void daeVariable::SetInitialGuess(const std::vector<size_t>& narrDomainIndexes, real_t dInitialGuess)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nIndex = m_nOverallIndex + CalculateIndex(narrDomainIndexes);
    m_pModel->m_pDataProxy->SetInitialGuess(nIndex, dInitialGuess);
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

void daeVariable::SetInitialGuess(const std::vector<size_t>& narrDomainIndexes, const quantity& q)
{
    real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
    SetInitialGuess(narrDomainIndexes, value);
}

void daeVariable::SetInitialGuesses(const quantity& q)
{
	real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialGuesses(value);
}

void daeVariable::SetInitialGuesses(const std::vector<quantity>& initialGuesses)
{
    std::vector<real_t> r_initialGuesses;
    r_initialGuesses.resize(initialGuesses.size());
    for(size_t i = 0; i < initialGuesses.size(); i++)
    {
        // If the value is equal to cnUnsetValue set real_t value to cnUnsetValue as well (by design: that means unset value)
        if(initialGuesses[i].getValue() == cnUnsetValue)
            r_initialGuesses[i] = cnUnsetValue;
        else
            r_initialGuesses[i] = initialGuesses[i].scaleTo(m_VariableType.GetUnits()).getValue();
    }
    SetInitialGuesses(r_initialGuesses);
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

void daeVariable::SetValue(const std::vector<size_t>& narrDomainIndexes, const quantity& q)
{
    real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
    SetValue(narrDomainIndexes, value);
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

quantity daeVariable::GetQuantity(const std::vector<size_t>& narrDomainIndexes)
{
    real_t value = GetValue(narrDomainIndexes);
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
		m_pModel->m_pDataProxy->AssignValue(m_nOverallIndex + i, dValues);
		m_pModel->m_pDataProxy->SetVariableType(m_nOverallIndex + i, cnAssigned);
	}	
}

void daeVariable::AssignValues(const std::vector<real_t>& values)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nTotalNumberOfVariables = GetNumberOfPoints();
    if(values.size() != nTotalNumberOfVariables)
        daeDeclareAndThrowException(exInvalidCall);

    for(size_t i = 0; i < nTotalNumberOfVariables; i++)
    {
        // Skip if it is equal to cnUnsetValue (by design: that means unset value)
        if(values[i] == cnUnsetValue)
            continue;

        m_pModel->m_pDataProxy->AssignValue(m_nOverallIndex + i, values[i]);
        m_pModel->m_pDataProxy->SetVariableType(m_nOverallIndex + i, cnAssigned);
    }
}

void daeVariable::AssignValue(real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValue(size_t nD1, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValue(const std::vector<size_t>& narrDomainIndexes, real_t dValue)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nIndex = m_nOverallIndex + CalculateIndex(narrDomainIndexes);
    m_pModel->m_pDataProxy->AssignValue(nIndex, dValue);
    m_pModel->m_pDataProxy->SetVariableType(nIndex, cnAssigned);
}

void daeVariable::AssignValues(const quantity& q)
{
	real_t values = q.scaleTo(m_VariableType.GetUnits()).getValue();
	AssignValues(values);
}

void daeVariable::AssignValues(const std::vector<quantity>& values)
{
    std::vector<real_t> r_values;
    r_values.resize(values.size());
    for(size_t i = 0; i < values.size(); i++)
    {
        // If the value is equal to cnUnsetValue set real_t value to cnUnsetValue as well (by design: that means unset value)
        if(values[i].getValue() == cnUnsetValue)
            r_values[i] = cnUnsetValue;
        else
            r_values[i] = values[i].scaleTo(m_VariableType.GetUnits()).getValue();
    }
    AssignValues(r_values);
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

void daeVariable::AssignValue(const std::vector<size_t>& narrDomainIndexes, const quantity& q)
{
    real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
    AssignValue(narrDomainIndexes, value);
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
        if(m_pModel->m_pDataProxy->GetVariableType(m_nOverallIndex + i) != cnAssigned)
		{
			daeDeclareException(exInvalidCall);
            e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
			throw e;
		}		
		m_pModel->m_pDataProxy->ReAssignValue(m_nOverallIndex + i, dValues);
	}
}

void daeVariable::ReAssignValues(const std::vector<real_t>& values)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nTotalNumberOfVariables = GetNumberOfPoints();
    if(values.size() != nTotalNumberOfVariables)
        daeDeclareAndThrowException(exInvalidCall);

    for(size_t i = 0; i < nTotalNumberOfVariables; i++)
    {
        // Skip if it is equal to cnUnsetValue (by design: that means unset value)
        if(values[i] == cnUnsetValue)
            continue;

        if(m_pModel->m_pDataProxy->GetVariableType(m_nOverallIndex + i) != cnAssigned)
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
            throw e;
        }
        m_pModel->m_pDataProxy->ReAssignValue(m_nOverallIndex + i, values[i]);
    }
}

void daeVariable::ReAssignValue(real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
	{
		daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
		throw e;
	}		
	m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
	{
		daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
		throw e;
	}		
	m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
	{
		daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
		throw e;
	}		
	m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
	{
		daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
		throw e;
	}		
	m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
	{
		daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
		throw e;
	}		
	m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
	{
		daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
		throw e;
	}		
	m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
	{
		daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
		throw e;
	}		
	m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
	{
		daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
		throw e;
	}		
	m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dValue)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
	{
		daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
		throw e;
	}		
	m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValue(const std::vector<size_t>& narrDomainIndexes, real_t dValue)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nIndex = m_nOverallIndex + CalculateIndex(narrDomainIndexes);
    if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reassign the value of the state variable [" << GetCanonicalName() << "]";
        throw e;
    }
    m_pModel->m_pDataProxy->ReAssignValue(nIndex, dValue);
}

void daeVariable::ReAssignValues(const quantity& q)
{
	real_t values = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReAssignValues(values);
}

void daeVariable::ReAssignValues(const std::vector<quantity>& values)
{
    std::vector<real_t> r_values;
    r_values.resize(values.size());
    for(size_t i = 0; i < values.size(); i++)
    {
        // If the value is equal to cnUnsetValue set real_t value to cnUnsetValue as well (by design: that means unset value)
        if(values[i].getValue() == cnUnsetValue)
            r_values[i] = cnUnsetValue;
        else
            r_values[i] = values[i].scaleTo(m_VariableType.GetUnits()).getValue();
    }
    ReAssignValues(r_values);
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

void daeVariable::ReAssignValue(const std::vector<size_t>& narrDomainIndexes, const quantity& q)
{
    real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
    ReAssignValue(narrDomainIndexes, value);
}

void daeVariable::GetValues(std::vector<real_t>& values) const
{
    if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

    size_t noPoints = GetNumberOfPoints();
    values.resize(noPoints);

    for(size_t i = 0; i < noPoints; i++)
        values[i] = m_pModel->m_pDataProxy->GetValue(m_nOverallIndex + i);
}

void daeVariable::GetValues(std::vector<quantity>& quantities) const
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t noPoints = GetNumberOfPoints();
    quantities.resize(noPoints);

    for(size_t i = 0; i < noPoints; i++)
        quantities[i] = quantity(m_pModel->m_pDataProxy->GetValue(m_nOverallIndex + i), m_VariableType.GetUnits());
}

void daeVariable::SetValues(const std::vector<quantity>& quantities)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t noPoints = GetNumberOfPoints();
    if(noPoints != quantities.size())
        daeDeclareAndThrowException(exInvalidCall);

    for(size_t i = 0; i < noPoints; i++)
    {
        real_t val = quantities[i].scaleTo(m_VariableType.GetUnits()).getValue();
        m_pModel->m_pDataProxy->SetValue(m_nOverallIndex + i, val);
    }
}

void daeVariable::SetValues(const std::vector<real_t>& values)
{
    if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

    size_t noPoints = GetNumberOfPoints();
    if(noPoints != values.size())
		daeDeclareAndThrowException(exInvalidCall);

    for(size_t i = 0; i < noPoints; i++)
        m_pModel->m_pDataProxy->SetValue(m_nOverallIndex + i, values[i]);
}

real_t daeVariable::GetValue(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
	return m_pModel->m_pDataProxy->GetValue(nIndex);
}

real_t daeVariable::GetValue(size_t nD1)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
	return m_pModel->m_pDataProxy->GetValue(nIndex);
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
	return m_pModel->m_pDataProxy->GetValue(nIndex);
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
	return m_pModel->m_pDataProxy->GetValue(nIndex);
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
	return m_pModel->m_pDataProxy->GetValue(nIndex);
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
	return m_pModel->m_pDataProxy->GetValue(nIndex);
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
	return m_pModel->m_pDataProxy->GetValue(nIndex);
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
	return m_pModel->m_pDataProxy->GetValue(nIndex);
}

real_t daeVariable::GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
	return m_pModel->m_pDataProxy->GetValue(nIndex);
}

real_t daeVariable::GetValue(const std::vector<size_t>& narrDomainIndexes)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nIndex = m_nOverallIndex + CalculateIndex(narrDomainIndexes);
    return m_pModel->m_pDataProxy->GetValue(nIndex);
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

void daeVariable::SetValue(const std::vector<size_t>& narrDomainIndexes, real_t dValue)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nIndex = m_nOverallIndex + CalculateIndex(narrDomainIndexes);
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
		m_pModel->m_pDataProxy->SetInitialCondition(m_nOverallIndex + i, dInitialConditions, m_pModel->GetInitialConditionMode());
		m_pModel->m_pDataProxy->SetVariableType(m_nOverallIndex + i, cnDifferential);
	}
}

void daeVariable::SetInitialConditions(const std::vector<real_t>& initialConditions)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nTotalNumberOfVariables = GetNumberOfPoints();
    if(initialConditions.size() != nTotalNumberOfVariables)
        daeDeclareAndThrowException(exInvalidCall);

    for(size_t i = 0; i < nTotalNumberOfVariables; i++)
    {
        // Skip if it is equal to cnUnsetValue (by design: that means unset value)
        if(initialConditions[i] == cnUnsetValue)
            continue;

        m_pModel->m_pDataProxy->SetInitialCondition(m_nOverallIndex + i, initialConditions[i], m_pModel->GetInitialConditionMode());
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

	m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
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

	m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
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

	m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
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

	m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
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

	m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
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

	m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
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

	m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
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

	m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
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

	m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
	m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialCondition(const std::vector<size_t>& narrDomainIndexes, real_t dInitialCondition)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nIndex = m_nOverallIndex + CalculateIndex(narrDomainIndexes);
    m_pModel->m_pDataProxy->SetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
    m_pModel->m_pDataProxy->SetVariableType(nIndex, cnDifferential);
}

void daeVariable::SetInitialConditions(const quantity& q)
{
	real_t values = q.scaleTo(m_VariableType.GetUnits()).getValue();
	SetInitialConditions(values);
}

void daeVariable::SetInitialConditions(const std::vector<quantity>& initialConditions)
{
    std::vector<real_t> r_initialConditions;
    r_initialConditions.resize(initialConditions.size());
    for(size_t i = 0; i < initialConditions.size(); i++)
    {
        // If the value is equal to cnUnsetValue set real_t value to cnUnsetValue as well (by design: that means unset value)
        if(initialConditions[i].getValue() == cnUnsetValue)
            r_initialConditions[i] = cnUnsetValue;
        else
            r_initialConditions[i] = initialConditions[i].scaleTo(m_VariableType.GetUnits()).getValue();
    }
    SetInitialConditions(r_initialConditions);
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

void daeVariable::SetInitialCondition(const std::vector<size_t>& narrDomainIndexes, const quantity& q)
{
    real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
    SetInitialCondition(narrDomainIndexes, value);
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
        if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(m_nOverallIndex + i) != cnDifferential)
		{
            // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
            // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
            daeDeclareException(exInvalidCall);
            e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
			throw e;
		}		

		m_pModel->m_pDataProxy->ReSetInitialCondition(m_nOverallIndex + i, dInitialConditions, m_pModel->GetInitialConditionMode());
	}
}

void daeVariable::ReSetInitialConditions(const std::vector<real_t>& initialConditions)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nTotalNumberOfVariables = GetNumberOfPoints();
    if(initialConditions.size() != nTotalNumberOfVariables)
        daeDeclareAndThrowException(exInvalidCall);

    for(size_t i = 0; i < nTotalNumberOfVariables; i++)
    {
        // Skip if it is equal to cnUnsetValue (by design: that means unset value)
        if(initialConditions[i] == cnUnsetValue)
            continue;

        if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(m_nOverallIndex + i) != cnDifferential)
        {
            // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
            // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
            daeDeclareException(exInvalidCall);
            e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
            throw e;
        }
        m_pModel->m_pDataProxy->ReSetInitialCondition(m_nOverallIndex + i, initialConditions[i], m_pModel->GetInitialConditionMode());
    }
}

void daeVariable::ReSetInitialCondition(real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nIndex = m_nOverallIndex + CalculateIndex(NULL, 0);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
		throw e;
	}		

	m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialCondition(size_t nD1, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[1] = {nD1};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 1);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
		throw e;
	}		

	m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[2] = {nD1, nD2};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 2);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
		throw e;
	}		

	m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[3] = {nD1, nD2, nD3};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 3);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
		throw e;
	}		

	m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[4] = {nD1, nD2, nD3, nD4};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 4);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
		throw e;
	}		

	m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[5] = {nD1, nD2, nD3, nD4, nD5};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 5);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
		throw e;
	}		

	m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[6] = {nD1, nD2, nD3, nD4, nD5, nD6};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 6);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
		throw e;
	}		

	m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[7] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 7);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
		throw e;
	}		

	m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialCondition)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t indexes[8] = {nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8};
	size_t nIndex = m_nOverallIndex + CalculateIndex(indexes, 8);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
	{
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
		throw e;
	}		

	m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialCondition(const std::vector<size_t>& narrDomainIndexes, real_t dInitialCondition)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    size_t nIndex = m_nOverallIndex + CalculateIndex(narrDomainIndexes);
    if(m_pModel->m_pDataProxy->GetInitialConditionMode() == eAlgebraicValuesProvided && m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnDifferential)
    {
        // Raise an exception if the mode is eAlgebraicValuesProvided and we attempt to reset init. condition of the non-diff variable
        // because if we are in the eQuasySteadyState mode we do not have VariableTypes set thus we do not know which variables are differential.
        daeDeclareException(exInvalidCall);
        e << "Invalid call: cannot reset initial condition of the non-differential variable [" << GetCanonicalName() << "]";
        throw e;
    }

    m_pModel->m_pDataProxy->ReSetInitialCondition(nIndex, dInitialCondition, m_pModel->GetInitialConditionMode());
}

void daeVariable::ReSetInitialConditions(const quantity& q)
{
	real_t values = q.scaleTo(m_VariableType.GetUnits()).getValue();
	ReSetInitialConditions(values);
}

void daeVariable::ReSetInitialConditions(const std::vector<quantity>& initialConditions)
{
    std::vector<real_t> r_initialConditions;
    r_initialConditions.resize(initialConditions.size());
    for(size_t i = 0; i < initialConditions.size(); i++)
    {
        // If the value is equal to cnUnsetValue set real_t value to cnUnsetValue as well (by design: that means unset value)
        if(initialConditions[i].getValue() == cnUnsetValue)
            r_initialConditions[i] = cnUnsetValue;
        else
            r_initialConditions[i] = initialConditions[i].scaleTo(m_VariableType.GetUnits()).getValue();
    }
    ReSetInitialConditions(r_initialConditions);
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

void daeVariable::ReSetInitialCondition(const std::vector<size_t>& narrDomainIndexes, const quantity& q)
{
    real_t value = q.scaleTo(m_VariableType.GetUnits()).getValue();
    ReSetInitialCondition(narrDomainIndexes, value);
}

const daeVariableType_t* daeVariable::GetVariableType(void) const
{
	return &m_VariableType;
}
	
void daeVariable::SetVariableType(const daeVariableType& VariableType)
{
	m_VariableType = VariableType;
}

size_t daeVariable::GetNumberOfDomains() const
{
    return m_ptrDomains.size();
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

//real_t* daeVariable::GetValuePointer() const
//{
//	if(!m_pModel)
//		daeDeclareAndThrowException(exInvalidPointer);
//	if(!m_pModel->m_pDataProxy)
//		daeDeclareAndThrowException(exInvalidPointer);
//	return m_pModel->m_pDataProxy->GetValue(m_nOverallIndex);
//}

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

daeDomain* daeVariable::GetDomain(size_t nIndex) const
{
	if(nIndex >= m_ptrDomains.size())
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid domain index [" << nIndex << "] in variable [" << GetCanonicalName() << "]";
		throw e;
	}
	
	return m_ptrDomains[nIndex];
}

/*********************************************************************************************
	daeVariableWrapper
**********************************************************************************************/
daeVariableWrapper::daeVariableWrapper() 
{
//	m_nOverallIndex = ULONG_MAX;
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
	
	m_pVariable         = pVariable;
	m_narrDomainIndexes = narrDomainIndexes;
/*	
	m_pDataProxy        = pVariable->m_pModel->m_pDataProxy;
	m_nOverallIndex     = pVariable->m_nOverallIndex + pVariable->CalculateIndex(narrDomainIndexes);
	
	//std::cout << "daeVariableWrapper::m_nOverallIndex = " << m_nOverallIndex << std::endl;
	
	if(m_nOverallIndex == ULONG_MAX)
		daeDeclareAndThrowException(exInvalidCall);
*/	
	if(strName.empty())
	{
        m_strName = ReplaceAll(pVariable->GetCanonicalName(), '.', '_');
        
		if(!narrDomainIndexes.empty())
			m_strName += "(" + toString(narrDomainIndexes, string(",")) + ")";
	}
	else
	{
		m_strName = strName;
	}
}

string daeVariableWrapper::GetName(void) const
{
	return m_strName;
}

/*
  Old code that does not work when used as an InputVariable for parameter estimation.
  
real_t daeVariableWrapper::GetValue(void) const
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
// This accesses the ValuesReference; therefore the system must be initialized before using it!
	return m_pDataProxy->GetValue(m_nOverallIndex);
}

void daeVariableWrapper::SetValue(real_t value)
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
// This accesses the ValuesReference; therefore the system must be initialized before using it!
	m_pDataProxy->SetValue(m_nOverallIndex, value);
}
*/

size_t daeVariableWrapper::GetOverallIndex(void) const
{
    size_t index = m_pVariable->CalculateIndex(m_narrDomainIndexes);
    return m_pVariable->m_nOverallIndex + index;
}

int daeVariableWrapper::GetVariableType(void) const
{
    size_t nOverallIndex = GetOverallIndex();
    boost::shared_ptr<daeDataProxy_t> pDataProxy = m_pVariable->m_pModel->GetDataProxy();
    return pDataProxy->GetVariableType(nOverallIndex);
}

real_t daeVariableWrapper::GetValue(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidPointer);
	
	size_t n = m_narrDomainIndexes.size();
	
	if(n == 0)
		return m_pVariable->GetValue();
	else if(n == 1)
		return m_pVariable->GetValue(m_narrDomainIndexes[0]);
	else if(n == 2)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1]);
	else if(n == 3)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2]);
	else if(n == 4)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3]);
	else if(n == 5)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3],
									 m_narrDomainIndexes[4]);
	else if(n == 6)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3],
									 m_narrDomainIndexes[4],
									 m_narrDomainIndexes[5]);
	else if(n == 7)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3],
									 m_narrDomainIndexes[4],
									 m_narrDomainIndexes[5],
									 m_narrDomainIndexes[6]);
	else if(n == 8)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3],
									 m_narrDomainIndexes[4],
									 m_narrDomainIndexes[5],
									 m_narrDomainIndexes[6],
									 m_narrDomainIndexes[7]);
	else
		daeDeclareAndThrowException(exInvalidCall);
				
	return 0;
}

void daeVariableWrapper::SetValue(real_t value)
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidPointer);

    size_t n = m_narrDomainIndexes.size();

    if(n == 0)
        m_pVariable->SetValue(value);
    else if(n == 1)
        m_pVariable->SetValue(m_narrDomainIndexes[0],
                              value);
    else if(n == 2)
        m_pVariable->SetValue(m_narrDomainIndexes[0],
                              m_narrDomainIndexes[1],
                              value);
    else if(n == 3)
        m_pVariable->SetValue(m_narrDomainIndexes[0],
                              m_narrDomainIndexes[1],
                              m_narrDomainIndexes[2],
                              value);
    else if(n == 4)
        m_pVariable->SetValue(m_narrDomainIndexes[0],
                              m_narrDomainIndexes[1],
                              m_narrDomainIndexes[2],
                              m_narrDomainIndexes[3],
                              value);
    else if(n == 5)
        m_pVariable->SetValue(m_narrDomainIndexes[0],
                              m_narrDomainIndexes[1],
                              m_narrDomainIndexes[2],
                              m_narrDomainIndexes[3],
                              m_narrDomainIndexes[4],
                              value);
    else if(n == 6)
        m_pVariable->SetValue(m_narrDomainIndexes[0],
                              m_narrDomainIndexes[1],
                              m_narrDomainIndexes[2],
                              m_narrDomainIndexes[3],
                              m_narrDomainIndexes[4],
                              m_narrDomainIndexes[5],
                              value);
    else if(n == 7)
        m_pVariable->SetValue(m_narrDomainIndexes[0],
                              m_narrDomainIndexes[1],
                              m_narrDomainIndexes[2],
                              m_narrDomainIndexes[3],
                              m_narrDomainIndexes[4],
                              m_narrDomainIndexes[5],
                              m_narrDomainIndexes[6],
                              value);
    else if(n == 8)
        m_pVariable->SetValue(m_narrDomainIndexes[0],
                              m_narrDomainIndexes[1],
                              m_narrDomainIndexes[2],
                              m_narrDomainIndexes[3],
                              m_narrDomainIndexes[4],
                              m_narrDomainIndexes[5],
                              m_narrDomainIndexes[6],
                              m_narrDomainIndexes[7],
                              value);
    else
        daeDeclareAndThrowException(exInvalidCall)
}

}
}
