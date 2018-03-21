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
#ifndef DAE_LA_SOLVER_H
#define DAE_LA_SOLVER_H

#include "typedefs.h"

int laInitialize(daeLASolver_t* lasolver,
                 void*          model,
                 size_t         numberOfVariables);
int laSetup(daeLASolver_t*  lasolver,
            real_t          time,
            real_t          inverseTimeStep,
            real_t*         values,
            real_t*         timeDerivatives,
            real_t*         residuals);
int laSolve(daeLASolver_t*  lasolver,
            real_t          time,
            real_t          inverseTimeStep,
            real_t          tolerance,
            real_t*         values,
            real_t*         timeDerivatives,
            real_t*         residuals,
            real_t*         r,
            real_t*         z);
int laFree(daeLASolver_t* lasolver);

#endif
