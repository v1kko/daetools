/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic, 2013
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "auxiliary.h"
#include "daetools_model.h"
#include "simulation.h"

int main(int argc, char *argv[])
{
    daeModel_t       model;
    daeIDASolver_t   dae_solver;
    daeSimulation_t  simulation;

    memset(&model,      0, sizeof(daeModel_t));
    memset(&dae_solver, 0, sizeof(daeIDASolver_t));
    memset(&simulation, 0, sizeof(daeSimulation_t));

    initialize_model(&model);
    simInitialize(&simulation, &model, &dae_solver, false);

    model.values          = dae_solver.yval;
    model.timeDerivatives = dae_solver.ypval;
    initialize_values_references(&model);

    simSolveInitial(&simulation);
    simRun(&simulation);
    simFinalize(&simulation);

    return 0;
}


