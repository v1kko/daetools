#include "stdafx.h"
#include "datareporters.h"

namespace dae
{
namespace datareporting
{
daeDataReporterRemote::daeDataReporterRemote()
{
}

daeDataReporterRemote::~daeDataReporterRemote()
{
	Disconnect();
}

bool daeDataReporterRemote::SendProcessName(const string& strProcessName)
{
	m_strProcessName = strProcessName;
	
	return SendMessage(
						m_msgFormatter.SendProcessName(strProcessName)
					  );
}

bool daeDataReporterRemote::Connect(const string& strConnectString, const string& strProcessName)
{
	return true;
}

bool daeDataReporterRemote::IsConnected()
{
	return false;
}

bool daeDataReporterRemote::Disconnect()
{
	return true;
}

bool daeDataReporterRemote::StartRegistration(void)
{
	if(!IsConnected())
		return false;

	return SendMessage(
						m_msgFormatter.StartRegistration()
					  );
}
	
bool daeDataReporterRemote::EndRegistration(void)
{
	if(!IsConnected())
		return false;

	return SendMessage(
						m_msgFormatter.EndRegistration()
					  );
}

bool daeDataReporterRemote::RegisterDomain(const daeDataReporterDomain* pDomain)
{
	if(!IsConnected())
		return false;

	return SendMessage(
						m_msgFormatter.RegisterDomain(pDomain)
					  );
}

bool daeDataReporterRemote::RegisterVariable(const daeDataReporterVariable* pVariable)
{
	if(!IsConnected())
		return false;

	return SendMessage(
						m_msgFormatter.RegisterVariable(pVariable)
					  );
}

bool daeDataReporterRemote::StartNewResultSet(real_t dTime)
{
	if(!IsConnected())
		return false;

	m_dCurrentTime = dTime;

	return SendMessage(
						m_msgFormatter.StartNewResultSet(dTime)
					  );
}

bool daeDataReporterRemote::EndOfData()
{
	if(!IsConnected())
		return false;

	m_dCurrentTime = -1;

	return SendMessage(
						m_msgFormatter.EndOfData()
					  );
}

bool daeDataReporterRemote::SendVariable(const daeDataReporterVariableValue* pVariableValue)
{
	if(!IsConnected())
		return false;
	if(m_dCurrentTime < 0)
		return false;

	return SendMessage(
						m_msgFormatter.SendVariable(pVariableValue)
					  );
}

bool daeDataReporterRemote::SendMessage(const string& strMessage)
{
	return false;
}


	


}
}
