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
#ifndef DAE_COOL_PROP_H
#define DAE_COOL_PROP_H

#include "../Core/thermo_package.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

#ifdef DAE_DLL_INTERFACE
#ifdef COOL_PROP_EXPORTS
#define DAE_COOL_PROP_API __declspec(dllexport)
#else
#define DAE_COOL_PROP_API __declspec(dllimport)
#endif
#else
#define DAE_COOL_PROP_API
#endif

#else // WIN32
#define DAE_COOL_PROP_API
#endif // WIN32

DAE_COOL_PROP_API daetools::tpp::daeThermoPhysicalPropertyPackage_t* daeCreateCoolPropPropertyPackage();

#endif
