/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with the
OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include <mpi.h>
#include "roberts.h"
#include <OpenCS/models/cs_model_builder.h>
#include <OpenCS/evaluators/cs_evaluator_sequential.h>
#include <OpenCS/simulators/cs_simulators.h>
using namespace cs;

const char* simulation_options_json =
#include "simulation_options-ode.json"
;

/* Reimplementation of CVodes cvsRoberts_dns example.
 * The Roberts chemical kinetics problem with 3 rate equations:
 *
 *    dy1/dt = -0.04*y1 + 1.e4*y2*y3
 *    dy2/dt =  0.04*y1 - 1.e4*y2*y3 - 3.e7*(y2)^2
 *    dy3/dt =  3.e7*(y2)^2
 *
 * The problem is solved on the time interval from 0.0 <= t <= 4.e10,
 * with initial conditions:
 *   y1 = 1.0
 *   y2 = y3 = 0.
 * The problem is stiff.*/
int main(int argc, char *argv[])
{
    Roberts rob;

    uint32_t    Ndofs                         = 0;
    uint32_t    Nvariables                    = rob.Nequations;
    real_t      defaultVariableValue          = 0.0;
    real_t      defaultAbsoluteTolerance      = 1e-7;
    std::string defaultVariableName           = "x";

    printf("Nvariables: %u\n", Nvariables);

    /* 1. Initialise model builder with the number of variables/equations. */
    csModelBuilder_t mb;
    mb.Initialize_ODE_System(Nvariables,
                             Ndofs,
                             defaultVariableValue,
                             defaultAbsoluteTolerance,
                             defaultVariableName);
    printf("Model builder initialised\n");

    /* 2. Create and set model equations. */
    const std::vector<csNumber_t>& y_vars  = mb.GetVariables();

    std::vector<csNumber_t> equations(Nvariables);

    rob.CreateEquations(y_vars, equations);

    printf("Model equations generated\n");

    mb.SetModelEquations(equations);
    printf("Model equations set\n");

    printf("Equations expresions:\n");
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        std::string expression = equations[i].node->ToLatex();
        printf(" $$%5d: %s $$\n", i, expression.c_str());
    }

    /* 3. Generate a sequential model. */
    // Set initial conditions
    std::vector<real_t> y0(Nvariables, 0.0);
    rob.SetInitialConditions(y0);
    mb.SetVariableValues(y0);

    real_t startTime         = 0.0;
    real_t timeHorizon       = 4E3;
    real_t reportingInterval = 10;
    real_t relativeTolerance = 1e-5;

    const size_t bsize = 8192;
    char buffer[bsize];
    std::snprintf(buffer, bsize, simulation_options_json, startTime, timeHorizon, reportingInterval, relativeTolerance);
    std::string simulationOptions = buffer;

    uint32_t Npe = 1;
    std::vector<std::string> balancingConstraints;
    std::string inputFilesDirectory = "ode_example_1-Roberts";
    csGraphPartitioner_Simple partitioner;
    std::vector<csModelPtr> models_sequential = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
    mb.ExportModels(models_sequential, inputFilesDirectory, simulationOptions);
    printf("Generated model for the Roberts chemical kinetics model\n");

    /* 4. Simulate the exported model using the libOpenCS_Simulators. */
    printf("Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());
    MPI_Init(&argc, &argv);
    csSimulate_ODE(inputFilesDirectory);
    MPI_Finalize();

    return 0;
}

/* Results from CVodes Roberts_dns:
At t = 2.6391e-01      y =  9.899653e-01    3.470564e-05    1.000000e-02
At t = 4.0000e-01      y =  9.851641e-01    3.386242e-05    1.480205e-02
At t = 4.0000e+00      y =  9.055097e-01    2.240338e-05    9.446793e-02
At t = 4.0000e+01      y =  7.158010e-01    9.185084e-06    2.841898e-01
At t = 4.0000e+02      y =  4.504693e-01    3.222627e-06    5.495274e-01
At t = 4.0000e+03      y =  1.832126e-01    8.943459e-07    8.167865e-01
At t = 4.0000e+04      y =  3.897839e-02    1.621552e-07    9.610214e-01
At t = 4.0000e+05      y =  4.940533e-03    1.985905e-08    9.950594e-01
At t = 4.0000e+06      y =  5.170046e-04    2.069075e-09    9.994830e-01
At t = 2.0803e+07      y =  1.000000e-04    4.000395e-10    9.999000e-01
At t = 4.0000e+07      y =  5.199610e-05    2.079951e-10    9.999480e-01
At t = 4.0000e+08      y =  5.200133e-06    2.080064e-11    9.999948e-01
At t = 4.0000e+09      y =  5.131179e-07    2.052473e-12    9.999995e-01
At t = 4.0000e+10      y =  5.470287e-08    2.188115e-13    9.999999e-01

*/
