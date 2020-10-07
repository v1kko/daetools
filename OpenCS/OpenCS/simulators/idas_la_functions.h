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
#ifndef CS_SIMULATOR_IDAS_LA_FUNCTIONS_H
#define CS_SIMULATOR_IDAS_LA_FUNCTIONS_H

#include <idas/idas.h>
#include <idas/idas_impl.h>
#include <nvector/nvector_parallel.h>

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#ifdef OpenCS_SIMULATORS_EXPORTS
#define OPENCS_SIMULATORS_API __declspec(dllexport)
#else
#define OPENCS_SIMULATORS_API __declspec(dllimport)
#endif
#else
#define OPENCS_SIMULATORS_API
#endif

namespace cs_dae_simulator
{
// DAE systems
OPENCS_SIMULATORS_API int init_la(IDAMem ida_mem);
OPENCS_SIMULATORS_API int setup_la(IDAMem ida_mem,
                                   N_Vector	vectorVariables,
                                   N_Vector	vectorTimeDerivatives,
                                   N_Vector	vectorResiduals,
                                   N_Vector	vectorTemp1,
                                   N_Vector	vectorTemp2,
                                   N_Vector	vectorTemp3);
OPENCS_SIMULATORS_API int solve_la(IDAMem ida_mem,
                                   N_Vector	b,
                                   N_Vector	weight,
                                   N_Vector	vectorVariables,
                                   N_Vector	vectorTimeDerivatives,
                                   N_Vector	vectorResiduals);
OPENCS_SIMULATORS_API int free_la(IDAMem ida_mem);
}

#endif

