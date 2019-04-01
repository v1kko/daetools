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
#include <OpenCS/models/cs_partitioners.h>
#include <OpenCS/models/partitioner_metis.h>
#include <OpenCS/simulators/cs_simulators.h>
using namespace cs;

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
 * The PDEs are discretized by central differencing on a uniform (Nx, Ny) grid.
 * The original results are in dae_example_3.csv file.
 *
 * The model is described in:
 *  - R. Serban and A. C. Hindmarsh. CVODES, the sensitivity-enabled ODE solver in SUNDIALS.
 *    In Proceedings of the 5th International Conference on Multibody Systems,
 *    Nonlinear Dynamics and Control, Long Beach, CA, 2005. ASME.
 *  - M. R. Wittman. Testing of PVODE, a Parallel ODE Solver.
 *    Technical Report UCRL-ID-125562, LLNL, August 1996. */
int main(int argc, char *argv[])
{
    uint32_t Nx = 82;
    uint32_t Ny = 82;

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

    printf("########################################################\n");
    printf(" Brusselator model in 2D: %u x %u grid\n", Nx, Ny);
    printf("########################################################\n");

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

    // Set variable names
    std::vector<std::string> names;
    bruss.GetVariableNames(names);
    mb.SetVariableNames(names);

    /* 3. Export model(s) to a specified directory for sequential and parallel simulations. */
    // Set initial conditions.
    std::vector<real_t> uv0(Nvariables, 0.0);
    bruss.SetInitialConditions(uv0);
    mb.SetVariableValues(uv0);

    // Set the simulation options.
    csSimulationOptionsPtr options = mb.GetSimulationOptions();
    options->SetDouble("Simulation.TimeHorizon",              10.0);
    options->SetDouble("Simulation.ReportingInterval",         0.1);
    options->SetDouble("Solver.Parameters.RelativeTolerance", 1e-5);

    // For Ncpu = 1: k = 3, rho = 1.0, alpha = 1e-1, w = 0.5
    options->SetInteger("LinearSolver.Preconditioner.Parameters.fact: level-of-fill",         3);
    options->SetDouble ("LinearSolver.Preconditioner.Parameters.fact: relax value",         0.5);
    options->SetDouble ("LinearSolver.Preconditioner.Parameters.fact: absolute threshold", 1e-1);
    options->SetDouble ("LinearSolver.Preconditioner.Parameters.fact: relative threshold",  1.0);
    std::string simulationOptions_seq = options->ToString();

    /* Model export requires partitioning of the system and simulation options (in JSON format).
     * For Npe = 1, graph partitioner is not required (the whole system of equations is used).
     * For distributed memory systems a graph partitioner must be specified.
     * Available partitioners:
     *   - Simple
     *   - 2D_Npde (for discretisation of Npde equations on uniform 2D grids)
     *   - Metis:
     *     - PartGraphKway:      'Multilevel k-way partitioning' algorithm
     *     - PartGraphRecursive: 'Multilevel recursive bisectioning' algorithm
     * Metis partitioner can additionally balance specified quantities in all partitions using the balancing constraints.
     * Available balancing constraints:
     *  - Ncs:      balance number of compute stack items in equations
     *  - Nnz:      balance number of non-zero items in the incidence matrix
     *  - Nflops:   balance number of FLOPS required to evaluate equations
     *  - Nflops_j: balance number of FLOPS required to evaluate derivatives (Jacobian) matrix */

    // Generate a single model (no graph partitioner required).
    std::string inputFilesDirectory = "dae_example_3-sequential";
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
        // csGraphPartitioner_2D_Npde graph partitioner.
        uint32_t Npe = 8;
        std::vector<std::string> balancingConstraints;
        std::string inputFilesDirectory = "dae_example_3-Npe=8-2D_Npde";
        csGraphPartitionerPtr partitioner = createGraphPartitioner_2D_Npde(Nx, Ny, 2, 2.0);
        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe, partitioner.get(), balancingConstraints, true);
        csModelBuilder_t::ExportModels(models_parallel, inputFilesDirectory, simulationOptions_par);
        printf("Generated models for Npe = %u %s\n", Npe, partitioner->GetName().c_str());
    }

    // Use the Metis graph partitioner.
    if(false)
    {
        uint32_t Npe = 8;
        std::string inputFilesDirectory = "dae_example_3-Npe=8-[Nflops,Nflops_j]";

        // Set the load balancing constraints
        std::vector<std::string> balancingConstraints = {"Nflops", "Nflops_j"};

        // Create METIS graph partitioner (instantiate directly to be able to set the options).
        csGraphPartitioner_Metis partitioner(PartGraphRecursive);

        // Partitioner options can be set using the SetOptions function.
        // First, obtain the array with options already initialised to default values.
        std::vector<int32_t> metis_options = partitioner.GetOptions();
        // Then, set the options (as described in the Section 5.4 of the METIS manual; requires <metis.h> included).
        // metis_options[METIS_OPTION_DBGLVL]  = METIS_DBG_INFO | METIS_DBG_TIME;
        // metis_options[METIS_OPTION_NITER]   = 10;
        // metis_options[METIS_OPTION_UFACTOR] = 30;
        // Finally, set the updated options to the Metis partitioner.
        partitioner.SetOptions(metis_options);

        // Graph partitioners can optionally use dictionaries with a number of FLOPs required for individual mathematical operations.
        // This way, the total number of FLOPs can be precisely estimated for every equation.
        // Number of FLOPs are specified using two dictionaries:
        //  1. unaryOperationsFlops for:
        //      - unary operators (+, -)
        //      - unary functions (sqrt, log, log10, exp, sin, cos, tan, asin, acos, atan, sinh, cosh,
        //                         tanh, asinh, acosh, atanh, erf, floor, ceil, and abs)
        //  2. binaryOperationsFlops for:
        //      - binary operators (+, -, *, /, ^)
        //      - binary functions (pow, min, max, atan2)
        // For instance:
        //   unaryOperationsFlops[eSqrt] = 12
        //   unaryOperationsFlops[eExp]   = 5
        //   binaryOperationsFlops[eMulti]  = 4
        //   binaryOperationsFlops[eDivide] = 6
        // If a mathematical operation is not in the dictionary, it is assumed that it requires a single FLOP.
        // In this example the dictionaries are not used (all operations require a single FLOP).
        std::map<csUnaryFunctions,uint32_t>  unaryOperationsFlops;
        std::map<csBinaryFunctions,uint32_t> binaryOperationsFlops;

        // Partition the system to generate Npe models (one per processing element).
        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe,
                                                                     &partitioner,
                                                                     balancingConstraints,
                                                                     true,
                                                                     unaryOperationsFlops,
                                                                     binaryOperationsFlops);

        // Export the models into the specified directory.
        csModelBuilder_t::ExportModels(models_parallel, inputFilesDirectory, simulationOptions_par);
        printf("Generated models for Npe = %u %s-[%s]\n", Npe, partitioner.GetName().c_str(), "Nflops,Nflops_j");
    }

    /* 4. Simulate the sequential model using the libOpenCS_Simulators. */
    printf("Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());
    MPI_Init(&argc, &argv);
    csSimulate(inputFilesDirectory);
    MPI_Finalize();

    return 0;
}
