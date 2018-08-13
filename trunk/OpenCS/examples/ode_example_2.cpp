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
#include "advection_diffusion_2d.h"
#include <OpenCS/models/cs_model_builder.h>
#include <OpenCS/evaluators/cs_evaluator_sequential.h>
#include <OpenCS/simulators/cs_simulators.h>
using namespace cs;

const char* simulation_options_json =
#include "simulation_options-ode.json"
;

/* Reimplementation of CVodes cvsAdvDiff_bnd example.
 * The problem is simple advection-diffusion in 2-D:
 *
 *   du/dt = d2u/dx2 + 0.5 du/dx + d2u/dy2
 *
 * on the rectangle 0 <= x <= 2, 0 <= y <= 1, and the time interval 0 <= t <= 1.
 *
 * Homogeneous Dirichlet boundary conditions are imposed, with the initial conditions:
 *
 *   u(x,y,t=0) = x(2-x)y(1-y)exp(5xy).
 *
 * The PDE is discretized on a uniform Nx+2 by Ny+2 grid with central differencing.
 * The boundary points are eliminated leaving an ODE system of size Nx*Ny. */
int main(int argc, char *argv[])
{
    uint32_t    Nx = 10;
    uint32_t    Ny = 5;

    // Homogenous Dirichlet BCs at all four edges: u|boundary = 0.0.
    // Boundaries are excluded from the system of equations, that's why the system is ODE (should be a DAE system).
    csNumber_t bc_at_all_four_edges(0.0);
    AdvectionDiffusion_2D adv_diff(Nx, Ny, bc_at_all_four_edges);

    uint32_t    Ndofs                         = 0;
    uint32_t    Nvariables                    = adv_diff.Nequations;
    real_t      defaultVariableValue          = 0.0;
    real_t      defaultAbsoluteTolerance      = 1e-6;
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
    const std::vector<csNumber_t>& u_vars = mb.GetVariables();

    std::vector<csNumber_t> equations(Nvariables);

    adv_diff.CreateEquations(u_vars, equations);
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
    std::vector<real_t> u0(Nvariables, 0.0);
    adv_diff.SetInitialConditions(u0);
    mb.SetVariableValues(u0);

    real_t startTime         = 0.0;
    real_t timeHorizon       = 1.0;
    real_t reportingInterval = 0.1;
    real_t relativeTolerance = 1e-5;

    const size_t bsize = 8192;
    char buffer[bsize];
    std::snprintf(buffer, bsize, simulation_options_json, startTime, timeHorizon, reportingInterval, relativeTolerance);
    std::string simulationOptions = buffer;

    uint32_t Npe = 1;
    std::vector<std::string> balancingConstraints;
    std::string inputFilesDirectory = "ode_example_2-Advection-Diffusion";
    {
        csGraphPartitioner_Simple partitioner;
        std::vector<csModelPtr> models_sequential = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
        mb.ExportModels(models_sequential, inputFilesDirectory, simulationOptions);
        printf("Generated model for the Advection-Diffusion model\n");
    }

    /* 4. Simulate the exported model using the libOpenCS_Simulators. */
    printf("Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());
    MPI_Init(&argc, &argv);
    csSimulate_ODE(inputFilesDirectory);
    MPI_Finalize();

    return 0;
}

/* Results from CVodes cvsAdvDiff_bnd:
At t = 0      max.norm(u) =  8.954716e+01
At t = 0.10   max.norm(u) =  4.132889e+00   nst =   85
At t = 0.20   max.norm(u) =  1.039294e+00   nst =  103
At t = 0.30   max.norm(u) =  2.979829e-01   nst =  113
At t = 0.40   max.norm(u) =  8.765774e-02   nst =  120
At t = 0.50   max.norm(u) =  2.625637e-02   nst =  126
At t = 0.60   max.norm(u) =  7.830425e-03   nst =  130
At t = 0.70   max.norm(u) =  2.329387e-03   nst =  134
At t = 0.80   max.norm(u) =  6.953434e-04   nst =  137
At t = 0.90   max.norm(u) =  2.115983e-04   nst =  140
At t = 1.00   max.norm(u) =  6.556853e-05   nst =  142
*/
