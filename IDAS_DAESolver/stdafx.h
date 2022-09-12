#include <cfloat>

#ifndef DAE_SOLVERS_STD_AFX_H
#define DAE_SOLVERS_STD_AFX_H

#ifndef __MINGW32__
#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifndef _MSC_VER
#define _MSC_VER
#endif

// Modify the following defines if you have to target a platform prior to the ones specified below.
// Refer to MSDN for the latest info on corresponding values for different platforms.
#ifndef WINVER				// Allow use of features specific to Windows XP or later.
#define WINVER 0x0501		// Change this to the appropriate value to target other versions of Windows.
#endif

#ifndef _WIN32_WINNT		// Allow use of features specific to Windows XP or later.                   
#define _WIN32_WINNT 0x0501	// Change this to the appropriate value to target other versions of Windows.
#endif						

#ifndef _WIN32_WINDOWS		// Allow use of features specific to Windows 98 or later.
#define _WIN32_WINDOWS 0x0410 // Change this to the appropriate value to target Windows Me or later.
#endif

#ifndef _WIN32_IE			// Allow use of features specific to IE 6.0 or later.
#define _WIN32_IE 0x0600	// Change this to the appropriate value to target other versions of IE.
#endif

#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>
#include <tchar.h>

#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#pragma warning(disable: 4267)

#endif //Win32
#endif

#include <stdio.h>
#include <stdarg.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <iomanip>
#include <memory>

//#include "../Core/xmlfile.h"
//using namespace daetools::xml;

#include <math.h>

#include <boost/smart_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
using namespace boost;

#endif
