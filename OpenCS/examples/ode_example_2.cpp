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
#include <OpenCS/opencs.h>
using namespace cs;

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
 * The boundary points are eliminated leaving an ODE system of size Nx*Ny.
 * The original results are in ode_example_2.csv file. */
int main(int argc, char *argv[])
{
    uint32_t Nx = 10;
    uint32_t Ny = 5;

    if(argc == 3)
    {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
    }
    else if(argc == 1)
    {
        // Use the default grid size.
    }
    else
    {
        printf("Usage:\n");
        printf("  %s (using the default grid: %u x %u)\n", argv[0], Nx, Ny);
        printf("  %s Nx Ny\n", argv[0]);
        return -1;
    }

    printf("############################################################\n");
    printf(" Simple advection-diffusion in 2D: %u x %u grid\n", Nx, Ny);
    printf("############################################################\n");

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

    // Set variable names
    std::vector<std::string> names;
    adv_diff.GetVariableNames(names);
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
    std::vector<real_t> u0(Nvariables, 0.0);
    adv_diff.SetInitialConditions(u0);
    mb.SetVariableValues(u0);

    // Set the simulation options.
    csSimulationOptionsPtr options = mb.GetSimulationOptions();
    options->SetDouble("Simulation.TimeHorizon",               1.0);
    options->SetDouble("Simulation.ReportingInterval",         0.1);
    options->SetDouble("Solver.Parameters.RelativeTolerance", 1e-5);
    std::string simulationOptions = options->ToString();

    // Generate a single model (no graph partitioner required).
    uint32_t Npe = 1;
    std::vector<std::string> balancingConstraints;
    std::string inputFilesDirectory = "ode_example_2-sequential";
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
