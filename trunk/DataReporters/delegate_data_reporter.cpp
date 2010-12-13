#include "stdafx.h"
#include "datareporters.h"

namespace dae
{
namespace datareporting
{
/*********************************************************************
	daeDelegateDataReporter
*********************************************************************/
daeDelegateDataReporter::daeDelegateDataReporter(void)
{
}

daeDelegateDataReporter::~daeDelegateDataReporter(void)
{
}

void daeDelegateDataReporter::AddDataReporter(daeDataReporter_t* pDataReporter)
{
	if(!pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer); 
		
	m_ptrarrDataReporters.push_back(pDataReporter);
}

bool daeDelegateDataReporter::StartRegistration(void)
{
	vector<daeDataReporter_t*>::iterator it;
	bool bResult = true;
	for(it = m_ptrarrDataReporters.begin(); it != m_ptrarrDataReporters.end(); it++)
		if((*it)->StartRegistration() == false)
			bResult = false;
	return bResult;
}

bool daeDelegateDataReporter::EndRegistration(void)
{
	vector<daeDataReporter_t*>::iterator it;
	bool bResult = true;
	for(it = m_ptrarrDataReporters.begin(); it != m_ptrarrDataReporters.end(); it++)
		if((*it)->EndRegistration() == false)
			bResult = false;
	return bResult;
}

bool daeDelegateDataReporter::RegisterDomain(const daeDataReporterDomain* pDomain)
{
	vector<daeDataReporter_t*>::iterator it;
	bool bResult = true;
	for(it = m_ptrarrDataReporters.begin(); it != m_ptrarrDataReporters.end(); it++)
		if((*it)->RegisterDomain(pDomain) == false)
			bResult = false;
	return bResult;
}

bool daeDelegateDataReporter::RegisterVariable(const daeDataReporterVariable* pVariable)
{
	vector<daeDataReporter_t*>::iterator it;
	bool bResult = true;
	for(it = m_ptrarrDataReporters.begin(); it != m_ptrarrDataReporters.end(); it++)
		if((*it)->RegisterVariable(pVariable) == false)
			bResult = false;
	return bResult;
}

bool daeDelegateDataReporter::StartNewResultSet(real_t dTime)
{
	vector<daeDataReporter_t*>::iterator it;
	bool bResult = true;
	for(it = m_ptrarrDataReporters.begin(); it != m_ptrarrDataReporters.end(); it++)
		if((*it)->StartNewResultSet(dTime) == false)
			bResult = false;
	return bResult;
}

bool daeDelegateDataReporter::EndOfData()
{
	vector<daeDataReporter_t*>::iterator it;
	bool bResult = true;
	for(it = m_ptrarrDataReporters.begin(); it != m_ptrarrDataReporters.end(); it++)
		if((*it)->EndOfData() == false)
			bResult = false;
	return bResult;
}

bool daeDelegateDataReporter::SendVariable(const daeDataReporterVariableValue* pVariableValue)
{
	vector<daeDataReporter_t*>::iterator it;
	bool bResult = true;
	for(it = m_ptrarrDataReporters.begin(); it != m_ptrarrDataReporters.end(); it++)
		if((*it)->SendVariable(pVariableValue) == false)
			bResult = false;
	return bResult;
}

bool daeDelegateDataReporter::Connect(const string& strConnectString, const string& strProcessName)
{
	return true;
}

bool daeDelegateDataReporter::IsConnected()
{
	vector<daeDataReporter_t*>::iterator it;
	bool bResult = true;
	for(it = m_ptrarrDataReporters.begin(); it != m_ptrarrDataReporters.end(); it++)
		if((*it)->IsConnected() == false)
			bResult = false;
	return bResult;
}

bool daeDelegateDataReporter::Disconnect()
{
	vector<daeDataReporter_t*>::iterator it;
	bool bResult = true;
	for(it = m_ptrarrDataReporters.begin(); it != m_ptrarrDataReporters.end(); it++)
		if((*it)->Disconnect() == false)
			bResult = false;
	return bResult;
}

}
}



