#include "stdafx.h"
#include "base_logging.h"
#include "logs.h"
#include "tcpiplog.h"

namespace daetools
{
namespace logging
{
daeLog_t* daeCreateFileLog(const string& strFileName)
{
    return new daeFileLog(strFileName);
}

daeLog_t* daeCreateStdOutLog()
{
    return new daeStdOutLog;
}
}
}

