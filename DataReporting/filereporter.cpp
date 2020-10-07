#include "stdafx.h"
#include "datareporters.h"

namespace daetools
{
namespace datareporting
{
daeFileDataReporter::daeFileDataReporter()
{
    m_strName = "FileDataReporter";
}

daeFileDataReporter::~daeFileDataReporter()
{
	if(IsConnected())
		Disconnect();
}

bool daeFileDataReporter::Connect(const string& strConnectString, const string& strProcessName)
{
    m_strConnectString    = strConnectString;
    m_strProcessName      = strProcessName;
	m_strFilename         = strConnectString;
	m_drProcess.m_strName = strProcessName;

	of.open(m_strFilename.c_str());
	if(of.is_open())
		return true;
	else
		return false;
}

bool daeFileDataReporter::IsConnected()
{
	if(of.is_open())
		return true;
	else
		return false;
}

bool daeFileDataReporter::Disconnect()
{
	if(!IsConnected())
		return false;
	WriteDataToFile();
	of.flush();
	of.close();
	return true;
}

void daeFileDataReporter::WriteDataToFile(void)
{
}


}
}
