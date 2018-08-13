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
#include "heat_conduction_2d.h"
#include <OpenCS/models/cs_model_builder.h>
#include <OpenCS/evaluators/cs_evaluator_sequential.h>
#include <OpenCS/simulators/cs_simulators.h>
using namespace cs;

const char* simulation_options_json =
#include "simulation_options-dae.json"
;

/* Reimplementation of DAE Tools tutorial1.py example.
 * A simple heat conduction problem: conduction through a very thin, rectangular copper plate.
 * Two-dimensional Cartesian grid (x,y) of 20 x 20 elements. */
int main(int argc, char *argv[])
{
    /*************************************************************************************************
     * A) Specification of models
     *************************************************************************************************/
    /* Use a uniform mesh with 401x101 grid points to avoid symmetry and a more thorough check of
     * partitioning/simulation results since the number of equations cannot be uniformly distributed
     * among processing elements. */
    uint32_t Nx = 20;
    uint32_t Ny = 20;
    HeatConduction_2D hc(Nx, Ny);

    uint32_t    Ndofs                         = 0;
    uint32_t    Nvariables                    = hc.Nequations;
    real_t      defaultVariableValue          = 0.0;
    real_t      defaultVariableTimeDerivative = 0.0;
    real_t      defaultAbsoluteTolerance      = 1e-5;
    std::string defaultVariableName           = "T";

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

    /* 2. Create and set model equations using the provided time/variable/derivative/dof objects. */
    const csNumber_t&              time      = mb.GetTime();
    const std::vector<csNumber_t>& T_vars    = mb.GetVariables();
    const std::vector<csNumber_t>& dTdt_vars = mb.GetTimeDerivatives();

    std::vector<csNumber_t> equations(Nvariables);

    hc.CreateEquations(T_vars, dTdt_vars, equations);

    mb.SetModelEquations(equations);
    printf("Model equations generated and set\n");

    /*
    printf("Equations expresions:\n");
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        std::string expression = equations[i].node->ToLatex();
        printf(" $$%5d: %s $$\n", i, expression.c_str());
    }
    */

    /*************************************************************************************************
     * B) Use Cases
     *************************************************************************************************/
    real_t epsilon = 1e-15;
    std::vector<real_t> dof_values;
    std::vector<real_t> T    (Nvariables, 0.0);
    std::vector<real_t> dT_dt(Nvariables, 0.0);

    hc.SetInitialConditions(T, dT_dt);

    mb.SetVariableValues         (T);
    mb.SetVariableTimeDerivatives(dT_dt);

    /*************************************************************************************************
     * Use Case 1: Export model(s) to a specified directory for simulation using csSimulator_DAE
     *************************************************************************************************/
    real_t startTime         =   0.0;
    real_t timeHorizon       = 500.0;
    real_t reportingInterval =   5.0;
    real_t relativeTolerance = 1e-5;

    /* Model export requires partitioning of the system and simulation options (in JSON format).
     * a) Available partitioners:
     *    - Simple
     *    - Metis:
     *      - PartGraphKway:      'Multilevel k-way partitioning' algorithm
     *      - PartGraphRecursive: 'Multilevel recursive bisectioning' algorithm
     *    Metis partitioner can additionally balance specified quantities in all partitions using the balancing constraints.
     *    Available balancing constraints:
     *    - Ncs:      balance number of compute stack items in equations
     *    - Nnz:      balance number of non-zero items in the incidence matrix
     *    - Nflops:   balance number of FLOPS required to evaluate equations
     *    - Nflops_j: balance number of FLOPS required to evaluate derivatives (Jacobian) matrix
     *
     * b) Sample simulation options are given in "simulation_options-Sundials_Ifpack_ILU.json" file
     *    loaded in "heat_conduction_2d.h" as "simulation_options_json" variable.
     *    simulation_options_json variable serves as a template for setting the values of
     *    StartTime, TimeHorizon, ReportingInterval and RelativeTolerance that must be set.
     *    Here, use snprintf to format the simulation options string. */
    const size_t bsize = 8192;
    char buffer[bsize];
    std::snprintf(buffer, bsize, simulation_options_json, startTime, timeHorizon, reportingInterval, relativeTolerance);
    std::string simulationOptions = buffer;

    /* 1. For Npe = 1 the partitioner creates a single model for a sequential simulation (a single MPI node).
     *    Here, Simple partitioner can be used to simply split equations into Npe parts (with no analysis). */
    uint32_t Npe = 1;
    std::vector<std::string> balancingConstraints;
    std::string inputFilesDirectory = "dae_example_2-Npe_1";
    csGraphPartitioner_Simple partitioner;
    std::vector<csModelPtr> models_sequential = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
    mb.ExportModels(models_sequential, inputFilesDirectory, simulationOptions);
    printf("Case 1.1 Generated models for Npe = %u\n", Npe);

    /* 2. For Npe > 1 the partitioner creates Npe models for a parallel simulation.
     * 2.1 Use the Simple partitioner (does not use balancing constraints). */
    Npe = 4;
    {
        std::vector<std::string> balancingConstraints;
        std::string inputFilesDirectory = "dae_example_2-Npe_4-Simple";
        csGraphPartitioner_Simple partitioner;
        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
        mb.ExportModels(models_parallel, inputFilesDirectory, simulationOptions);
        printf("Case 1.2 Generated models for Npe = %u %s\n", Npe, partitioner.GetName().c_str());
    }

    /* 2.2 Use the Metis partitioner and 'Multilevel k-way partitioning' algorithm. No additional balancing constraints. */
    {
        std::string inputFilesDirectory = "dae_example_2-Npe_4-PartGraphKway";
        std::vector<std::string> balancingConstraints;

        csGraphPartitioner_Metis partitioner(PartGraphKway);

        // Options can be set using the SetOptions options.
        // First, obtain the array with options already initialised to default values.
        std::vector<int32_t> options = partitioner.GetOptions();
        // Then, set the options (as describe in the Section 5.4 of the METIS manual; requires <metis.h> included).
        // options[METIS_OPTION_DBGLVL]  = METIS_DBG_INFO | METIS_DBG_TIME;
        // options[METIS_OPTION_NITER]   = 10;
        // options[METIS_OPTION_UFACTOR] = 30;
        // Finally, set the updated options to the Metis partitioner.
        partitioner.SetOptions(options);

        // Graph partitioners can optionally use dictionaries with a number of FLOPS required for individual mathematical operations.
        // This way, the total number of FLOPS can be precisely estimated for every equation.
        // Number of FLOPS are specified using two dictionaries:
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

        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe,
                                                                     &partitioner,
                                                                     balancingConstraints,
                                                                     true,
                                                                     unaryOperationsFlops,
                                                                     binaryOperationsFlops);

        mb.ExportModels(models_parallel, inputFilesDirectory, simulationOptions);
        printf("Case 1.2 Generated models for Npe = %u %s\n", Npe, partitioner.GetName().c_str());
    }

    /* 2.3 Use the Metis partitioner and 'Multilevel recursive bisectioning' algorithm. No additional balancing constraints. */
    {
        std::vector<std::string> balancingConstraints;
        std::string inputFilesDirectory = "dae_example_2-Npe_4-PartGraphRecursive";
        csGraphPartitioner_Metis partitioner(PartGraphRecursive);
        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
        mb.ExportModels(models_parallel, inputFilesDirectory, simulationOptions);
        printf("Case 1.3 Generated models for Npe = %u %s\n", Npe, partitioner.GetName().c_str());
    }

    /* 2.4 Use the Metis partitioner and 'Multilevel recursive bisectioning' algorithm. Balance 'Ncs'. */
    {
        std::vector<std::string> balancingConstraints = {"Ncs"};
        std::string inputFilesDirectory = "dae_example_2-Npe_4-[Ncs]";
        csGraphPartitioner_Metis partitioner(PartGraphRecursive);
        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe, &partitioner, balancingConstraints);
        mb.ExportModels(models_parallel, inputFilesDirectory, simulationOptions);
        printf("Case 1.4 Generated models for Npe = %u %s-[%s]\n", Npe, partitioner.GetName().c_str(), "Ncs");
    }

    /* 2.5 Use the Metis partitioner and 'Multilevel recursive bisectioning' algorithm. Balance 'Ncs' and 'Nflops'. */
    {
        std::vector<std::string> balancingConstraints = {"Ncs", "Nflops"};
        std::string inputFilesDirectory = "dae_example_2-Npe_4-[Ncs,Nflops]";
        csGraphPartitioner_Metis partitioner(PartGraphRecursive);
        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
        mb.ExportModels(models_parallel, inputFilesDirectory, simulationOptions);
        printf("Case 1.5 Generated models for Npe = %u %s-[%s]\n", Npe, partitioner.GetName().c_str(), "Ncs,Nflops");
    }

    /* 2.6 Use the Metis partitioner and 'Multilevel recursive bisectioning' algorithm. Balance all available constraints. */
    {
        std::vector<std::string> balancingConstraints = {"Ncs", "Nnz", "Nflops", "Nflops_j"};
        std::string inputFilesDirectory = "dae_example_2-Npe_4-[Ncs,Nnz,Nflops,Nflops_j]";
        csGraphPartitioner_Metis partitioner(PartGraphRecursive);
        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
        mb.ExportModels(models_parallel, inputFilesDirectory, simulationOptions);
        printf("Case 1.6 Generated models for Npe = %u %s-[%s]\n", Npe, partitioner.GetName().c_str(), "Ncs,Nnz,Nflops,Nflops_j");
    }

    /* 2.7 Try the Metis partitioner for a large number of processing elements. */
    {
        uint32_t Npe = 100;
        std::vector<std::string> balancingConstraints = {"Ncs"};
        std::string inputFilesDirectory = "dae_example_2-Npe_100-[Ncs]";
        csGraphPartitioner_Metis partitioner(PartGraphRecursive);
        std::vector<csModelPtr> models_parallel = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
        mb.ExportModels(models_parallel, inputFilesDirectory, simulationOptions);
        printf("Case 1.7 Generated models for Npe = %u %s-[%s]\n", Npe, partitioner.GetName().c_str(), "Ncs");
    }

    /* 3. Simulation in GNU/Linux.
     * 3.1 The sequential simulation (for Npe = 1) can be started using:
     *   $ csSimulator_DAE "inputFilesDirectory" */
    //std::system("./../bin/csSimulator_DAE dae_example_2-Npe_1 > dae_example_2-sequential.out");

    /* 3.2 The parallel simulation (for Npe > 1) can be started using:
     *   $ mpirun -np Npe csSimulator_DAE "inputFilesDirectory"
     * or
     *   $ mpirun -np Npe konsole --hold -e csSimulator_DAE "inputFilesDirectory"
     * where the arguments "konsole --hold -e" are used to start each process in a separate shell
     * and to keep it open after the simulation ends. */
    //std::system("mpirun -np 4 konsole --hold -e ./../bin/csSimulator_DAE dae_example_2-Npe_4 > dae_example_2-parallel.out");

    /* 4. Simulation in Windows.
     * 4.1 The sequential simulation (for Npe = 1) can be started using:
     *   $ csSimulator_DAE "inputFilesDirectory" */
    //std::system("..\bin\csSimulator_DAE.exe dae_example_2-Npe_1 > dae_example_2-sequential.out");

    /* 4.2 The parallel simulation (for Npe > 1) can be started using:
     *   $ mpiexec -n Npe csSimulator_DAE "inputFilesDirectory" */
    //std::system("mpiexec -n 4 ..\bin\csSimulator_DAE.exe dae_example_2-Npe_4 > dae_example_2-parallel.out");

    /*************************************************************************************************
     * Use Case 2: Evaluation of equations
     *************************************************************************************************/
    // Select a model to use for evaluations.
    csModelPtr model = models_sequential[0];

    {
        // 1. Declare arrays for the results.
        std::vector<real_t> residuals_nodes;
        std::vector<real_t> residuals_cse;
        residuals_nodes.resize(Nvariables);
        residuals_cse.resize(Nvariables);

        // 2. Set current time set. */
        real_t currentTime = 1.0;

        // 3.1 Use csModelBuilder_t::EvaluateEquations function to evaluate equations.
        mb.EvaluateEquations(currentTime, residuals_nodes);

        // 3.2 Use the Compute Stack Evaluator to evaluate equations.
        std::shared_ptr<csComputeStackEvaluator_t> csEvaluator;
        csEvaluator.reset( new cs::csComputeStackEvaluator_Sequential() );
        csEvaluator->Initialize(false,
                                model->structure.Nequations,
                                model->structure.Nequations,
                                model->structure.Ndofs,
                                model->equations.computeStacks.size(),
                                model->sparsityPattern.incidenceMatrixItems.size(),
                                model->sparsityPattern.incidenceMatrixItems.size(),
                                &model->equations.computeStacks[0],
                                &model->equations.activeEquationSetIndexes[0],
                                &model->sparsityPattern.incidenceMatrixItems[0]);

        real_t* pdofs            = (dof_values.size() > 0 ? &dof_values[0] : NULL);
        real_t* pvalues          = &T[0];
        real_t* ptimeDerivatives = &dT_dt[0];

        csEvaluationContext_t EC;
        EC.equationEvaluationMode       = cs::eEvaluateEquation;
        EC.sensitivityParameterIndex    = -1;
        EC.jacobianIndex                = -1;
        EC.numberOfVariables            = model->structure.Nequations;
        EC.numberOfEquations            = model->structure.Nequations;
        EC.numberOfDOFs                 = dof_values.size();
        EC.numberOfComputeStackItems    = model->equations.computeStacks.size();
        EC.numberOfIncidenceMatrixItems = 0;
        EC.valuesStackSize              = 5;
        EC.lvaluesStackSize             = 20;
        EC.rvaluesStackSize             = 5;
        EC.currentTime                  = currentTime;
        EC.inverseTimeStep              = 0;
        EC.startEquationIndex           = 0;
        EC.startJacobianIndex           = 0;

        csEvaluator->EvaluateEquations(EC, pdofs, pvalues, ptimeDerivatives, &residuals_cse[0]);

        // 4. Compare the results from csNode_t::Evaluate and csComputeStackEvaluator_t::Evaluate functions.
        //    Print only the items that are different (if any). */
        printf("Case 2.  Evaluation of equation residuals\n");
        printf("         Comparison of residuals (only equations where fabs(F_cse[i] - F_nodes[i]) > %.0e are printed, if any):\n", epsilon);
        for(uint32_t i = 0; i < Nvariables; i++)
        {
            real_t difference = residuals_cse[i] - residuals_nodes[i];
            if(std::fabs(difference) > epsilon)
                printf("           [%5d] %.2e (%20.15f, %20.15f)\n", i, difference, residuals_cse[i], residuals_nodes[i]);
        }
        printf("\n");
    }

    /*************************************************************************************************
     * Use Case 3: Evaluation of derivatives (the Jacobian matrix).
     *************************************************************************************************/
    {
        // 1. Set the current time and the time step
        real_t currentTime = 1.0;
        real_t timeStep    = 1e-5;

        // 2. Generate the incidence matrix (it can also be populated during the first call to EvaluateDerivatives).
        //    The Jacobian matrix is stored in the CRS format.
        //    Declare arrays for the results.
        std::vector<uint32_t> IA, JA;
        std::vector<real_t> A_nodes;
        std::vector<real_t> A_cse;

        mb.GetSparsityPattern(IA, JA);

        uint32_t Nnz = JA.size();
        A_nodes.resize(Nnz);
        A_cse.resize(Nnz);

        // 3.1 Use csModelBuilder_t::EvaluateDerivatives function to evaluate derivatives.
        //     The incidence matrix will be populated during the first call to EvaluateDerivatives.
        bool generateIncidenceMatrix = false;
        mb.EvaluateDerivatives(currentTime, timeStep, IA, JA, A_nodes, generateIncidenceMatrix);

        // 3.2 Use the Compute Stack Evaluator to evaluate equations.
        std::shared_ptr<csComputeStackEvaluator_t> csEvaluator;
        csEvaluator.reset( new cs::csComputeStackEvaluator_Sequential() );
        csEvaluator->Initialize(false,
                                model->structure.Nequations,
                                model->structure.Nequations,
                                model->structure.Ndofs,
                                model->equations.computeStacks.size(),
                                model->sparsityPattern.incidenceMatrixItems.size(),
                                model->sparsityPattern.incidenceMatrixItems.size(),
                                &model->equations.computeStacks[0],
                                &model->equations.activeEquationSetIndexes[0],
                                &model->sparsityPattern.incidenceMatrixItems[0]);

        real_t* pdofs            = (dof_values.size() > 0 ? &dof_values[0] : NULL);
        real_t* pvalues          = &T[0];
        real_t* ptimeDerivatives = &dT_dt[0];

        csEvaluationContext_t EC;
        EC.equationEvaluationMode       = cs::eEvaluateDerivative;
        EC.sensitivityParameterIndex    = -1;
        EC.jacobianIndex                = -1;
        EC.numberOfVariables            = model->structure.Nequations;
        EC.numberOfEquations            = model->structure.Nequations;
        EC.numberOfDOFs                 = dof_values.size();
        EC.numberOfComputeStackItems    = model->equations.computeStacks.size();
        EC.numberOfIncidenceMatrixItems = model->sparsityPattern.incidenceMatrixItems.size();
        EC.valuesStackSize              = 5;
        EC.lvaluesStackSize             = 20;
        EC.rvaluesStackSize             = 5;
        EC.currentTime                  = currentTime;
        EC.inverseTimeStep              = 1.0 / timeStep;
        EC.startEquationIndex           = 0;
        EC.startJacobianIndex           = 0;

        csEvaluator->EvaluateDerivatives(EC, pdofs, pvalues, ptimeDerivatives, &A_cse[0]);

        // 4. Compare the results from csNode_t::Evaluate and csComputeStackEvaluator_t::Evaluate functions.
        //    Print only the items that are different (if any). */
        printf("Case 3.  Evaluation of equation derivatives\n");
        printf("         Comparison of derivatives (only Jacobian items where fabs(J_cse[i,j] - J_nodes[i,j]) > %.0e are printed, if any):\n", epsilon);
        for(uint32_t i = 0; i < Nnz; i++)
        {
            real_t difference = A_cse[i] - A_nodes[i];
            if(std::fabs(difference) > epsilon)
                printf("           [%5d] %.2e (%20.15f, %20.15f)\n", i, difference, A_cse[i], A_nodes[i]);
        }
        printf("\n");
    }

    /*************************************************************************************************
     * Use Case 4: Simulate the exported model using the DAE simulator from libOpenCS_Simulators
     *************************************************************************************************/
    printf("Case 4.  Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());
    MPI_Init(&argc, &argv);
    csSimulate_DAE(inputFilesDirectory);
    MPI_Finalize();

    return 0;
}

/* Results from DAE Tools tutorial1.py:

"Time (s)","tutorial1.T(*, 0.0, 0.0)"
0.00000000000000e+00,3.00831255195860e+02
5.00000000000000e+00,3.06837451188856e+02
1.00000000000000e+01,3.09631631862243e+02
1.50000000000000e+01,3.11776479684250e+02
2.00000000000000e+01,3.13561375251053e+02
2.50000000000000e+01,3.15087894848071e+02
3.00000000000000e+01,3.16405209652843e+02
3.50000000000000e+01,3.17545869627578e+02
4.00000000000000e+01,3.18533721627763e+02
4.50000000000000e+01,3.19389033832041e+02
5.00000000000000e+01,3.20130019723842e+02
5.50000000000000e+01,3.20772153204060e+02
6.00000000000000e+01,3.21328685311137e+02
6.50000000000000e+01,3.21811042591061e+02
7.00000000000000e+01,3.22229085929137e+02
7.50000000000000e+01,3.22590969989233e+02
8.00000000000000e+01,3.22904487889583e+02
8.50000000000000e+01,3.23175792270656e+02
9.00000000000000e+01,3.23410728619497e+02
9.50000000000000e+01,3.23614136403836e+02
1.00000000000000e+02,3.23790528925655e+02
1.05000000000000e+02,3.23943478546197e+02
1.10000000000000e+02,3.24075972416751e+02
1.15000000000000e+02,3.24190997743872e+02
1.20000000000000e+02,3.24290746258327e+02
1.25000000000000e+02,3.24377072886934e+02
1.30000000000000e+02,3.24451643770463e+02
1.35000000000000e+02,3.24516464088734e+02
1.40000000000000e+02,3.24572661049033e+02
1.45000000000000e+02,3.24621040843894e+02
1.50000000000000e+02,3.24663122138114e+02
1.55000000000000e+02,3.24700111995483e+02
1.60000000000000e+02,3.24731828777483e+02
1.65000000000000e+02,3.24759168100337e+02
1.70000000000000e+02,3.24783045653244e+02
1.75000000000000e+02,3.24803856620891e+02
1.80000000000000e+02,3.24821680808070e+02
1.85000000000000e+02,3.24837195495440e+02
1.90000000000000e+02,3.24850972713202e+02
1.95000000000000e+02,3.24862673502914e+02
2.00000000000000e+02,3.24872536111687e+02
2.05000000000000e+02,3.24881457148674e+02
2.10000000000000e+02,3.24889199353647e+02
2.15000000000000e+02,3.24895861816951e+02
2.20000000000000e+02,3.24901579580994e+02
2.25000000000000e+02,3.24906513595854e+02
2.30000000000000e+02,3.24910894200529e+02
2.35000000000000e+02,3.24914775625652e+02
2.40000000000000e+02,3.24918104345982e+02
2.45000000000000e+02,3.24920932880687e+02
2.50000000000000e+02,3.24923506745139e+02
2.55000000000000e+02,3.24925668843272e+02
2.60000000000000e+02,3.24927419175085e+02
2.65000000000000e+02,3.24928962842201e+02
2.70000000000000e+02,3.24930349884284e+02
2.75000000000000e+02,3.24931483464292e+02
2.80000000000000e+02,3.24932370609579e+02
2.85000000000000e+02,3.24933239011289e+02
2.90000000000000e+02,3.24933957821262e+02
2.95000000000000e+02,3.24934527039499e+02
3.00000000000000e+02,3.24934982673785e+02
3.05000000000000e+02,3.24935378627018e+02
3.10000000000000e+02,3.24935702345441e+02
3.15000000000000e+02,3.24935969185557e+02
3.20000000000000e+02,3.24936183478552e+02
3.25000000000000e+02,3.24936361181539e+02
3.30000000000000e+02,3.24936516592071e+02
3.35000000000000e+02,3.24936651286492e+02
3.40000000000000e+02,3.24936740909997e+02
3.45000000000000e+02,3.24936808285029e+02
3.50000000000000e+02,3.24936858521826e+02
3.55000000000000e+02,3.24936896730627e+02
3.60000000000000e+02,3.24936928021671e+02
3.65000000000000e+02,3.24936957505199e+02
3.70000000000000e+02,3.24936992221172e+02
3.75000000000000e+02,3.24937041584628e+02
3.80000000000000e+02,3.24937090537838e+02
3.85000000000000e+02,3.24937139080803e+02
3.90000000000000e+02,3.24937187213522e+02
3.95000000000000e+02,3.24937234935995e+02
4.00000000000000e+02,3.24937282248223e+02
4.05000000000000e+02,3.24937329150205e+02
4.10000000000000e+02,3.24937345141902e+02
4.15000000000000e+02,3.24937360435058e+02
4.20000000000000e+02,3.24937375728214e+02
4.25000000000000e+02,3.24937391021369e+02
4.30000000000000e+02,3.24937406314525e+02
4.35000000000000e+02,3.24937421607681e+02
4.40000000000000e+02,3.24937436900836e+02
4.45000000000000e+02,3.24937452193992e+02
4.50000000000000e+02,3.24937467487147e+02
4.55000000000000e+02,3.24937482780303e+02
4.60000000000000e+02,3.24937498073459e+02
4.65000000000000e+02,3.24937513366614e+02
4.70000000000000e+02,3.24937528659770e+02
4.75000000000000e+02,3.24937543952925e+02
4.80000000000000e+02,3.24937555217247e+02
4.85000000000000e+02,3.24937564387100e+02
4.90000000000000e+02,3.24937573556953e+02
4.95000000000000e+02,3.24937582726806e+02
5.00000000000000e+02,3.24937591896659e+02
*/
