/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include <string>

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#ifdef OpenCS_Simulators_EXPORTS
#define OPENCS_SIMULATORS_API __declspec(dllexport)
#else
#define OPENCS_SIMULATORS_API __declspec(dllimport)
#endif
#else
#define OPENCS_SIMULATORS_API
#endif

namespace cs
{
OPENCS_SIMULATORS_API void csSimulate_DAE(const std::string& inputDirectory);
OPENCS_SIMULATORS_API void csSimulate_ODE(const std::string& inputDirectory);
}
