#include "stdafx.h"
#include "datareporters.h"

namespace daetools
{
namespace datareporting
{
/*********************************************************************
    daeDataReporterLocal
*********************************************************************/
daeDataReporterLocal::daeDataReporterLocal()
{
    m_strName = "DataReporterLocal";
    m_dCurrentTime = -1;
}

daeDataReporterLocal::~daeDataReporterLocal()
{
}

std::string daeDataReporterLocal::GetName() const
{
    return m_strName;
}

string daeDataReporterLocal::GetConnectString() const
{
    return m_strConnectString;
}

string daeDataReporterLocal::GetProcessName() const
{
    return m_strProcessName;
}

void daeDataReporterLocal::SetName(const std::string& strName)
{
    m_strName = strName;
}

void daeDataReporterLocal::SetConnectString(const std::string& strConnectString)
{
    m_strConnectString = strConnectString;
}

void daeDataReporterLocal::SetProcessName(const std::string& strProcessName)
{
    m_strProcessName = strProcessName;
}

bool daeDataReporterLocal::StartRegistration(void)
{
    return true;
}

bool daeDataReporterLocal::EndRegistration(void)
{
    return true;
}

bool daeDataReporterLocal::RegisterDomain(const daeDataReporterDomain* pDomain)
{
    daeDataReceiverDomain* pDom = new daeDataReceiverDomain;

    pDom->m_strName			= pDomain->m_strName;
    pDom->m_strUnits		= pDomain->m_strUnits;
    pDom->m_eType			= pDomain->m_eType;
    pDom->m_nNumberOfPoints = pDomain->m_nNumberOfPoints;
    if(pDomain->m_eType == eUnstructuredGrid)
        pDom->m_arrCoordinates = pDomain->m_arrCoordinates;
    else
        pDom->m_arrPoints = pDomain->m_arrPoints;

    m_drProcess.m_ptrarrRegisteredDomains.push_back(pDom);
    return true;
}

bool daeDataReporterLocal::RegisterVariable(const daeDataReporterVariable* pVariable)
{
    bool bFound;
    size_t i, j;
    daeDataReceiverDomain* pDomain;

    daeDataReceiverVariable* pVar = new daeDataReceiverVariable;

    pVar->m_strName			= pVariable->m_strName;
    pVar->m_strUnits		= pVariable->m_strUnits;
    pVar->m_nNumberOfPoints = pVariable->m_nNumberOfPoints;

    for(i = 0; i < pVariable->m_strarrDomains.size(); i++)
    {
        bFound = false;
        for(j = 0; j < m_drProcess.m_ptrarrRegisteredDomains.size(); j++)
        {
            pDomain = m_drProcess.m_ptrarrRegisteredDomains[j];
            if(pDomain->m_strName == pVariable->m_strarrDomains[i])
            {
                pVar->m_ptrarrDomains.push_back(pDomain);
                bFound = true;
                break;
            }
        }
        if(!bFound)
        {
            delete pVar;
            return false;
        }
    }
    size_t noPoints = 1;
    for(i = 0; i < pVar->m_ptrarrDomains.size(); i++)
        noPoints *= pVar->m_ptrarrDomains[i]->m_nNumberOfPoints;

    if(noPoints != pVar->m_nNumberOfPoints)
    {
        delete pVar;
        return false;
    }

    std::pair<string, daeDataReceiverVariable*> p(pVariable->m_strName, pVar);
    m_drProcess.m_ptrmapRegisteredVariables.insert(p);

    return true;
}

bool daeDataReporterLocal::StartNewResultSet(real_t dTime)
{
    m_dCurrentTime = dTime;
    return true;
}

bool daeDataReporterLocal::EndOfData()
{
    m_dCurrentTime = -1;
    return true;
}

bool daeDataReporterLocal::SendVariable(const daeDataReporterVariableValue* pVariableValue)
{
    if(m_dCurrentTime < 0)
        return false;

    std::map<string, daeDataReceiverVariable*>::iterator iter = m_drProcess.m_ptrmapRegisteredVariables.find(pVariableValue->m_strName);
    if(iter == m_drProcess.m_ptrmapRegisteredVariables.end())
        return false;
    daeDataReceiverVariable* pVariable = (*iter).second;
    if(!pVariable)
        return false;

    daeDataReceiverVariableValue* pVar = new daeDataReceiverVariableValue;
    pVariable->m_ptrarrValues.push_back(pVar);
    pVar->m_dTime	= m_dCurrentTime;
    pVar->m_pValues	= new real_t[pVariableValue->m_nNumberOfPoints];
    for(size_t i = 0; i < pVariableValue->m_nNumberOfPoints; i++)
        pVar->m_pValues[i] = pVariableValue->m_pValues[i];

    return true;
}

daeDataReceiverProcess* daeDataReporterLocal::GetProcess(void)
{
    return &m_drProcess;
}


/*********************************************************************
    daeNoOpDataReporter
*********************************************************************/
daeNoOpDataReporter::daeNoOpDataReporter()
{
    m_strName = "NoOpDataReporter";
}

daeNoOpDataReporter::~daeNoOpDataReporter()
{
}

bool daeNoOpDataReporter::Connect(const string& strConnectString, const string& strProcessName)
{
    m_strConnectString = strConnectString;
    m_strProcessName   = strProcessName;
    return true;
}

bool daeNoOpDataReporter::IsConnected()
{
    return true;
}

bool daeNoOpDataReporter::Disconnect()
{
    return true;
}


}
}
