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

daeLog_t* daeCreateTCPIPLog()
{
	return new daeTCPIPLog();
}

}
}

