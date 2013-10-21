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
#include "model.h"
#include "simulation.h"
#include "daesolver.h"

int main(int argc, char *argv[])
{
    daeModel_t       model;
    daeIDASolver_t   daesolver;
    daeSimulation_t  simulation;

    /* Initialize all data structures to zero */
    memset(&model,      0, sizeof(daeModel_t));
    memset(&daesolver,  0, sizeof(daeIDASolver_t));
    memset(&simulation, 0, sizeof(daeSimulation_t));

    /* Initialize the model */
    modInitialize(&model);
    
    /* Initialize the simulation and the dae solver */
    simInitialize(&simulation, &model, &daesolver, false);
    
    /* Solve the system at time = 0.0 */
    simSolveInitial(&simulation);
    
    /* Run the simulation */
    simRun(&simulation);
    
    /* Finalize the simulation */
    simFinalize(&simulation);

    return 0;
}


