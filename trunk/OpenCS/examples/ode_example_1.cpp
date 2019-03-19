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
#include <OpenCS/simulators/cs_simulators.h>
using namespace cs;

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
 * The problem is stiff.
 * The original results are in ode_example_1.csv file. */
int main(int argc, char *argv[])
{
    printf("##########################################################################################\n");
    printf(" Roberts chemical kinetics problem with 3 rate equations\n");
    printf("##########################################################################################\n");

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

    /* 2. Specify the OpenCS model. */
    // Create and set model equations using the provided time/variable/dof objects.
    // The ODE system is defined as:
    //     x' = f(x,y,t)
    // where x' are derivatives of state variables, x are state variables,
    // y are fixed variables (degrees of freedom) and t is the current simulation time.
    const std::vector<csNumber_t>& y_vars  = mb.GetVariables();

    std::vector<csNumber_t> equations(Nvariables);

    rob.CreateEquations(y_vars, equations);

    printf("Model equations generated\n");

    mb.SetModelEquations(equations);
    printf("Model equations set\n");

    // Set variable names
    std::vector<std::string> names;
    rob.GetVariableNames(names);
    mb.SetVariableNames(names);

    /*
    printf("Equations expresions:\n");
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        std::string expression = equations[i].node->ToLatex();
        printf(" $$%5d: %s $$\n", i, expression.c_str());
    }
    */

    /* 3. Generate a sequential model. */
    // Set initial conditions
    std::vector<real_t> y0(Nvariables, 0.0);
    rob.SetInitialConditions(y0);
    mb.SetVariableValues(y0);

    // Set the simulation options.
    csSimulationOptionsPtr options = mb.GetSimulationOptions();
    options->SetDouble("Simulation.TimeHorizon",              4000);
    options->SetDouble("Simulation.ReportingInterval",          10);
    options->SetDouble("Solver.Parameters.RelativeTolerance", 1e-5);
    std::string simulationOptions = options->ToString();

    // Generate a single model (no graph partitioner required).
    uint32_t Npe = 1;
    std::vector<std::string> balancingConstraints;
    std::string inputFilesDirectory = "ode_example_1-sequential";
    csGraphPartitioner_t* gp = NULL;
    std::vector<csModelPtr> models_sequential = mb.PartitionSystem(Npe, gp, balancingConstraints, true);
    csModelBuilder_t::ExportModels(models_sequential, inputFilesDirectory, simulationOptions);
    printf("Generated model for Npe = 1\n");

    /* 4. Simulate the sequential model using the libOpenCS_Simulators. */
    printf("Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());
    MPI_Init(&argc, &argv);

    // (a) Run simulation using the input files from the specified directory:
    csSimulate(inputFilesDirectory);
    // (b) Run simulation using the generated csModel_t, string with JSON options and the directory for simulation outputs:
    //csModelPtr model = models_sequential[0];
    //csSimulate(model, simulationOptions, inputFilesDirectory);

    MPI_Finalize();

    return 0;
}
