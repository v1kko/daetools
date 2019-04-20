/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2015
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_BASE_LOGGING_H
#define DAE_BASE_LOGGING_H

#include <string>
#include "log.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

#ifdef DAE_DLL_INTERFACE
#ifdef MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllexport)
#else // MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllimport)
#endif // MODEL_EXPORTS
#else // DAE_DLL_INTERFACE
#define DAE_CORE_API
#endif // DAE_DLL_INTERFACE

#else // WIN32
#define DAE_CORE_API
#endif // WIN32

namespace daetools
{
namespace logging
{
DAE_CORE_API daeLog_t* daeCreateFileLog(const string& strFileName);
DAE_CORE_API daeLog_t* daeCreateStdOutLog();
DAE_CORE_API daeLog_t* daeCreateTCPIPLog();
}
}

#endif

