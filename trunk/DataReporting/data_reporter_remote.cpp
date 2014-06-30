#include "stdafx.h"
#include "datareporters.h"

namespace dae
{
namespace datareporting
{
daeDataReporterRemote::daeDataReporterRemote()
{
    m_strName = "DataReporterRemote";
}

daeDataReporterRemote::~daeDataReporterRemote()
{
	Disconnect();
}

string daeDataReporterRemote::GetName() const
{
    return m_strName;
}

string daeDataReporterRemote::GetConnectString() const
{
    return m_strConnectString;
}

string daeDataReporterRemote::GetProcessName() const
{
    return m_strProcessName;
}

void daeDataReporterRemote::SetName(const std::string& strName)
{
    m_strName = strName;
}

void daeDataReporterRemote::SetConnectString(const std::string& strConnectString)
{
    m_strConnectString = strConnectString;
}

void daeDataReporterRemote::SetProcessName(const std::string& strProcessName)
{
    m_strProcessName = strProcessName;
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
    m_strConnectString = strConnectString;
    m_strProcessName   = strProcessName;
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
