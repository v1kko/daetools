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
#include "brusselator_2d.h"
#include <OpenCS/models/cs_model_builder.h>
#include <OpenCS/evaluators/cs_evaluator_sequential.h>
#include <OpenCS/simulators/cs_simulators.h>
using namespace cs;

const char* simulation_options_json =
#include "simulation_options-ode.json"
;

/* Reimplementation of IDAS idasBruss_kry_bbd_p example.
 * The PDE system is a two-species time-dependent PDE known as
 * Brusselator PDE and models a chemically reacting system:
 *
 *  du/dt = eps1(d2u/dx2  + d2u/dy2) + u^2 v - (B+1)u + A
 *  dv/dt = eps2(d2v/dx2  + d2v/dy2) - u^2 v + Bu
 *
 *  BC: Homogenous Neumann
 *  IC: u(x,y,t0) = u0(x,y) =  1  - 0.5*cos(pi*y/L)
 *      v(x,y,t0) = v0(x,y) = 3.5 - 2.5*cos(pi*x/L)
 *
 * The PDEs are discretized by central differencing on a Nx by Ny. */
int main(int argc, char *argv[])
{
    uint32_t Nx = 42;
    uint32_t Ny = 42;

    // Homogenous Neumann BCs at all four edges.
    csNumber_t bc_u_flux(0.0);
    csNumber_t bc_v_flux(0.0);
    Brusselator_2D bruss(Nx, Ny, bc_u_flux, bc_v_flux);

    uint32_t    Ndofs                         = 0;
    uint32_t    Nvariables                    = bruss.Nequations;
    real_t      defaultVariableValue          = 0.0;
    real_t      defaultVariableTimeDerivative = 0.0;
    real_t      defaultAbsoluteTolerance      = 1e-5;
    std::string defaultVariableName           = "x";

    printf("Nvariables: %u\n", Nvariables);

    /* 1. Initialise model builder with the number of variables/equations. */
    csModelBuilder_t mb;
    mb.Initialize_DAE_System(Nvariables,
                             Ndofs,
                             defaultVariableValue,
                             defaultVariableTimeDerivative,
                             defaultAbsoluteTolerance,
                             defaultVariableName);
    printf("Model builder initialised\n");

    /* 2. Create and set model equations. */
    const std::vector<csNumber_t>& x_vars  = mb.GetVariables();
    const std::vector<csNumber_t>& xt_vars = mb.GetTimeDerivatives();

    std::vector<csNumber_t> equations(Nvariables);
    bruss.CreateEquations(x_vars, xt_vars, equations);
    printf("Model equations generated\n");

    mb.SetModelEquations(equations);
    printf("Model equations set\n");

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
    std::vector<real_t> uv0   (Nvariables, 0.0);
    std::vector<real_t> uv0_dt(Nvariables, 0.0);

    bruss.SetInitialConditions(uv0, uv0_dt);

    mb.SetVariableValues(uv0);
    mb.SetVariableTimeDerivatives(uv0_dt);

    real_t startTime         = 0.0;
    real_t timeHorizon       = 20.0;
    real_t reportingInterval = 0.1;
    real_t relativeTolerance = 1e-5;

    const size_t bsize = 8192;
    char buffer[bsize];
    std::snprintf(buffer, bsize, simulation_options_json, startTime, timeHorizon, reportingInterval, relativeTolerance);
    std::string simulationOptions = buffer;

    uint32_t Npe = 1;
    std::vector<std::string> balancingConstraints;
    std::string inputFilesDirectory = "dae_example_3-Brusselator";
    {
        csGraphPartitioner_Simple partitioner;
        std::vector<csModelPtr> models_sequential = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
        mb.ExportModels(models_sequential, inputFilesDirectory, simulationOptions);
        printf("Generated model for the Brusselator model\n");
    }

    /* 4. Simulate the exported model using the libOpenCS_Simulators. */
    printf("Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());
    MPI_Init(&argc, &argv);
    csSimulate_DAE(inputFilesDirectory);
    MPI_Finalize();

    return 0;
}
