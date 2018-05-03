/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_IDAS_LA_FUNCTIONS_H
#define DAE_IDAS_LA_FUNCTIONS_H

#include <idas/idas.h>
#include <idas/idas_impl.h>
#include <nvector/nvector_parallel.h>

int init_la(IDAMem ida_mem);
int setup_la(IDAMem ida_mem,
             N_Vector	vectorVariables,
             N_Vector	vectorTimeDerivatives,
             N_Vector	vectorResiduals,
             N_Vector	vectorTemp1,
             N_Vector	vectorTemp2,
             N_Vector	vectorTemp3);
int solve_la(IDAMem ida_mem,
             N_Vector	b,
             N_Vector	weight,
             N_Vector	vectorVariables,
             N_Vector	vectorTimeDerivatives,
             N_Vector	vectorResiduals);
int free_la(IDAMem ida_mem);


#endif

