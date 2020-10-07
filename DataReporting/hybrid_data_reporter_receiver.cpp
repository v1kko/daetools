#include "stdafx.h"
#include "datareporters.h"

namespace daetools
{
namespace datareporting
{
/*********************************************************************
	daeHybridDataReporterReceiver
*********************************************************************/
//daeHybridDataReporterReceiver::daeHybridDataReporterReceiver(void)
//{
//	m_dCurrentTime = -1;
//}

//daeHybridDataReporterReceiver::~daeHybridDataReporterReceiver(void)
//{
//}

//bool daeHybridDataReporterReceiver::Connect(const string& /*strConnectString*/, const string& strProcessName)
//{
//	m_drProcess.m_strName = strProcessName;
//	return true;
//}

//bool daeHybridDataReporterReceiver::IsConnected()
//{
//	return true;
//}

//bool daeHybridDataReporterReceiver::Disconnect()
//{
//	return true;
//}

//void daeHybridDataReporterReceiver::GetProcessName(string& strProcessName)
//{
//	strProcessName = m_drProcess.m_strName;
//}

//bool daeHybridDataReporterReceiver::Start(void)
//{
//	return true;
//}

//bool daeHybridDataReporterReceiver::Stop(void)
//{
//	return true;
//}

//void daeHybridDataReporterReceiver::GetDomains(vector<const daeDataReceiverDomain*>& ptrarrDomains) const
//{
//	for(size_t i = 0; i < m_drProcess.m_ptrarrRegisteredDomains.size(); i++)
//		ptrarrDomains.push_back(m_drProcess.m_ptrarrRegisteredDomains[i]);
//}
	
//void daeHybridDataReporterReceiver::GetVariables(map<string, const daeDataReceiverVariable*>& ptrmappVariables) const
//{
//	map<string, daeDataReceiverVariable*>::const_iterator iter;
//	for(iter = m_drProcess.m_ptrmapRegisteredVariables.begin(); iter != m_drProcess.m_ptrmapRegisteredVariables.end(); iter++)
//	{
//		std::pair<string, const daeDataReceiverVariable*> p((*iter).first, (*iter).second);
//		ptrmappVariables.insert(p);
//	}
//}

//daeDataReceiverProcess*	daeHybridDataReporterReceiver::GetProcess(void)
//{
//	return &m_drProcess;
//}

}
}



