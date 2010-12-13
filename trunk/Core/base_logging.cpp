#include "stdafx.h"
#include "base_logging.h"
#include "tcpiplog.h"

namespace dae
{
namespace logging
{
daeLog_t* daeCreateFileLog(const string& strFileName)
{
	return new daeFileLog(strFileName);
}

daeLog_t* daeCreateStdOutLog(void)
{
	return new daeStdOutLog;
}

daeLog_t* daeCreateTCPIPLog(const string& strIPAddress, int nPort)
{
	return new daeTCPIPLog(strIPAddress, nPort);
}

//daeLog_t* daeCreateTCPIPLogServer(int nPort)
//{
//	return new daeTCPIPLogServer(nPort);
//}

}
}

