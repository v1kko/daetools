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
#include "diurnal_kinetics_2d.h"
#include <OpenCS/models/cs_model_builder.h>
#include <OpenCS/models/cs_partitioners.h>
#include <OpenCS/simulators/cs_simulators.h>
using namespace cs;

/* Reimplementation of CVodes cvsDiurnal_kry example.
 * 2-species diurnal kinetics advection-diffusion PDE system in 2D:
 *
 *   dc(i)/dt = Kh*(d/dx)^2 c(i) + V*dc(i)/dx + (d/dy)(Kv(y)*dc(i)/dy) + Ri(c1,c2,t), i = 1,2
 *
 * where
 *   R1(c1,c2,t) = -q1*c1*c3 - q2*c1*c2 + 2*q3(t)*c3 + q4(t)*c2
 *   R2(c1,c2,t) =  q1*c1*c3 - q2*c1*c2 - q4(t)*c2
 *   Kv(y) = Kv0*exp(y/5)
 *
 * Kh, V, Kv0, q1, q2, and c3 are constants, and q3(t) and q4(t) vary diurnally.
 * The problem is posed on the square:
 *    0 <= x <= 20 (km)
 *   30 <= y <= 50 (km)
 * with homogeneous Neumann boundary conditions, and integrated for time t in
 *   0 <= t <= 86400 sec (1 day)
 * The PDE system is discretised using the central differences on a uniform 10 x 10 mesh.
 * The original results are in ode_example_3.csv file. */
int main(int argc, char *argv[])
{
    uint32_t Nx = 80;
    uint32_t Ny = 80;

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

    printf("##########################################################################################\n");
    printf(" 2-species diurnal kinetics advection-diffusion PDE system in 2D: %u x %u grid\n", Nx, Ny);
    printf("##########################################################################################\n");

    DiurnalKinetics_2D dk(Nx, Ny);

    uint32_t    Ndofs                         = 0;
    uint32_t    Nvariables                    = dk.Nequations;
    real_t      defaultVariableValue          = 0.0;
    real_t      defaultAbsoluteTolerance      = 1e-5; // 100*1E-5 in cvodes cvsDiurnal_kry example
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
    const csNumber_t&              time     = mb.GetTime();
    const std::vector<csNumber_t>& C_values = mb.GetVariables();

    std::vector<csNumber_t> equations(Nvariables);
    dk.CreateEquations(C_values, time, equations);
    printf("Model equations generated\n");

    mb.SetModelEquations(equations);
    printf("Model equations set\n");

    // Set variable names
    std::vector<std::string> names;
    dk.GetVariableNames(names);
    mb.SetVariableNames(names);

    /* 3. Generate a sequential model. */
    // Set initial conditions
    std::vector<real_t> C0(Nvariables, 0.0);
    dk.SetInitialConditions(C0);
    mb.SetVariableValues(C0);

    // Set the simulation options.
    csSimulationOptionsPtr options = mb.GetSimulationOptions();
    options->SetDouble("Simulation.TimeHorizon",              86400);
    options->SetDouble("Simulation.ReportingInterval",          100);
    options->SetDouble("Solver.Parameters.RelativeTolerance",  1e-5);

    // Shared memory systems (Npe = 1, a single model).
    // In this case, a graph partitioner is not required.

    // For Ncpu = 1: k = 1, rho = 1.0, alpha = 1e-5, w = 0.0
    options->SetInteger("LinearSolver.Preconditioner.Parameters.fact: level-of-fill",         1);
    options->SetDouble ("LinearSolver.Preconditioner.Parameters.fact: relax value",         0.0);
    options->SetDouble ("LinearSolver.Preconditioner.Parameters.fact: absolute threshold", 1e-5);
    options->SetDouble ("LinearSolver.Preconditioner.Parameters.fact: relative threshold",  1.0);
    std::string simulationOptions_seq = options->ToString();

    std::string inputFilesDirectory = "ode_example_3-sequential";
    {
        uint32_t Npe = 1;
        std::vector<std::string> balancingConstraints;
        csGraphPartitioner_t* gp = NULL;
        std::vector<csModelPtr> models_sequential = mb.PartitionSystem(Npe, gp, balancingConstraints, true);
        csModelBuilder_t::ExportModels(models_sequential, inputFilesDirectory, simulationOptions_seq);
        printf("Generated model for Npe = 1\n");
    }

    // For Ncpu = 8: k = 1, rho = 1.0, alpha = 1e-1, w = 0.0
    options->SetInteger("LinearSolver.Preconditioner.Parameters.fact: level-of-fill",         1);
    options->SetDouble ("LinearSolver.Preconditioner.Parameters.fact: relax value",         0.0);
    options->SetDouble ("LinearSolver.Preconditioner.Parameters.fact: absolute threshold", 1e-1);
    options->SetDouble ("LinearSolver.Preconditioner.Parameters.fact: relative threshold",  1.0);
    std::string simulationOptions_par = options->ToString();
    {
        // 2D_Npde graph partitioner (for uniform 2D meshes).
        uint32_t Npe = 8;
        std::vector<std::string> balancingConstraints;
        std::string inputFilesDirectory = "ode_example_3-Npe=8-2D_Npde";
        csGraphPartitionerPtr partitioner = createGraphPartitioner_2D_Npde(Nx, Ny, 2, 0.5 /* Npex:Npey = 1 : 2 */);
        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe, partitioner.get(), balancingConstraints, true);
        csModelBuilder_t::ExportModels(models_parallel, inputFilesDirectory, simulationOptions_par);
        printf("Generated models for Npe = %u %s\n", Npe, partitioner->GetName().c_str());
    }

    /* 4. Simulate the sequential model using the libOpenCS_Simulators. */
    printf("Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());
    MPI_Init(&argc, &argv);
    csSimulate(inputFilesDirectory);
    MPI_Finalize();

    return 0;
}
