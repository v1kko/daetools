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
#include <OpenCS/opencs.h>
#include <OpenCS/simulators/daesimulator.h>
using namespace cs;

/* Reimplementation of DAE Tools tutorial1.py example.
 * A simple heat conduction problem: conduction through a very thin, rectangular copper plate.
 * Two-dimensional Cartesian grid (x,y) of 20 x 20 elements.
 * The original results are in dae_example_2.csv file. */
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

    printf("###################################################################\n");
    printf(" Simple heat conduction problem in 2D: %u x %u grid\n", Nx, Ny);
    printf("###################################################################\n");

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

    // Set variable names
    std::vector<std::string> names;
    hc.GetVariableNames(names);
    mb.SetVariableNames(names);

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

    hc.SetInitialConditions(T);

    mb.SetVariableValues         (T);
    mb.SetVariableTimeDerivatives(dT_dt);

    /*************************************************************************************************
     * Use Case 1: Export model(s) to a specified directory for simulation using the csSimulator
     *************************************************************************************************/
    /* Model export requires partitioning of the system and simulation options (in JSON format). */
    /* 1. Set the model options. */
    csSimulationOptionsPtr options = mb.GetSimulationOptions();
    options->SetDouble("Simulation.TimeHorizon",              500.0);
    options->SetDouble("Simulation.ReportingInterval",          5.0);
    options->SetDouble("Solver.Parameters.RelativeTolerance",  1e-5);
    std::string simulationOptions = options->ToString();

    /* 2. Partition the system to create a single model for a sequential simulation (a single MPI node).
     *    For Npe = 1, graph partitioner is not required. */
    uint32_t Npe = 1;
    std::vector<std::string> balancingConstraints;
    std::string inputFilesDirectory = "dae_example_2-sequential";
    csGraphPartitioner_t* gp = NULL;
    std::vector<csModelPtr> models_sequential = mb.PartitionSystem(Npe, gp, balancingConstraints);
    csModelBuilder_t::ExportModels(models_sequential, inputFilesDirectory, simulationOptions);
    printf("Case 1. Generated model for Npe = 1\n");

    /* 3. Simulation in GNU/Linux.
     * 3.1 The sequential simulation (for Npe = 1) can be started using:
     *   $ csSimulator "inputFilesDirectory" */
    //std::system("./../bin/csSimulator dae_example_2-sequential);

    /* 3.2 The parallel simulation (for Npe > 1) can be started using:
     *   $ mpirun -np Npe csSimulator "inputFilesDirectory" */
    //std::system("mpirun -np 4 ./../bin/csSimulator dae_example_2-Npe_4");

    /* 4. Simulation in Windows.
     * 4.1 The sequential simulation (for Npe = 1) can be started using:
     *   $ csSimulator "inputFilesDirectory" */
    //std::system("..\bin\csSimulator.exe dae_example_2-Npe_1");

    /* 4.2 The parallel simulation (for Npe > 1) can be started using:
     *   $ mpiexec -n Npe csSimulator "inputFilesDirectory" */
    //std::system("mpiexec -n 4 ..\bin\csSimulator.exe dae_example_2-Npe_4");

    /*************************************************************************************************
     * Use Case 2: Evaluation of equations - low level method)
     *************************************************************************************************/
    // Select a model to use for evaluations.
    csModelPtr model = models_sequential[0];

    std::vector<real_t> residuals_nodes;
    std::vector<real_t> residuals_cse;
    {
        // 1. Declare arrays for the results.
        residuals_nodes.resize(Nvariables);
        residuals_cse.resize(Nvariables);

        // 2. Set current time set. */
        real_t currentTime = 1.0;

        // 3.1 Use csModelBuilder_t::EvaluateEquations function to evaluate equations.
        mb.EvaluateEquations(currentTime, residuals_nodes);

        // 3.2 Use the Compute Stack Evaluator to evaluate equations.
        std::shared_ptr<csComputeStackEvaluator_t> csEvaluator(createEvaluator_Sequential());
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
        printf("Case 2. Evaluation of equation residuals (low-level method)\n");
        printf("        Comparison of residuals (only equations where fabs(F_cse[i] - F_nodes[i]) > %.0e are printed, if any):\n", epsilon);
        for(uint32_t i = 0; i < Nvariables; i++)
        {
            real_t difference = residuals_cse[i] - residuals_nodes[i];
            if(std::fabs(difference) > epsilon)
                printf("           [%5d] %.2e (%20.15f, %20.15f)\n", i, difference, residuals_cse[i], residuals_nodes[i]);
        }
        printf("\n");
    }

    /*************************************************************************************************
     * Use Case 3: Evaluation of derivatives (the Jacobian matrix) - low level method.
     *************************************************************************************************/
    std::vector<real_t> J_nodes;
    std::vector<real_t> J_cse;
    {
        // 1. Set the current time and the time step
        real_t currentTime = 1.0;
        real_t timeStep    = 1e-5;

        // 2. Generate the incidence matrix (it can also be populated during the first call to EvaluateDerivatives).
        //    The Jacobian matrix is stored in the CRS format.
        //    Declare arrays for the results.
        std::vector<uint32_t> IA, JA;

        mb.GetSparsityPattern(IA, JA);

        uint32_t Nnz = JA.size();
        J_nodes.resize(Nnz);
        J_cse.resize(Nnz);

        // 3.1 Use csModelBuilder_t::EvaluateDerivatives function to evaluate derivatives.
        //     The incidence matrix will be populated during the first call to EvaluateDerivatives.
        bool generateIncidenceMatrix = false;
        mb.EvaluateDerivatives(currentTime, timeStep, IA, JA, J_nodes, generateIncidenceMatrix);

        // 3.2 Use the Compute Stack Evaluator to evaluate equations.
        std::shared_ptr<csComputeStackEvaluator_t> csEvaluator(createEvaluator_Sequential());
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

        csEvaluator->EvaluateDerivatives(EC, pdofs, pvalues, ptimeDerivatives, &J_cse[0]);

        // 4. Compare the results from csNode_t::Evaluate and csComputeStackEvaluator_t::Evaluate functions.
        //    Print only the items that are different (if any). */
        printf("Case 3. Evaluation of equation derivatives (low-level method)\n");
        printf("        Comparison of derivatives (only Jacobian items where fabs(J_cse[i,j] - J_nodes[i,j]) > %.0e are printed, if any):\n", epsilon);
        for(uint32_t i = 0; i < Nnz; i++)
        {
            real_t difference = J_cse[i] - J_nodes[i];
            if(std::fabs(difference) > epsilon)
                printf("           [%5d] %.2e (%20.15f, %20.15f)\n", i, difference, J_cse[i], J_nodes[i]);
        }
        printf("\n");
    }

    /*************************************************************************************************
     * Use Case 4: Model exchange
     *************************************************************************************************/
    // 1. Initialise MPI
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    {
        /* 2. Instantiate the Compute Stack model implementation of the
         *    csDifferentialEquationModel_t interface.
         *    A reference implementation (csDifferentialEquationModel) is provided
         *    in the libOpenCS_Models shared library. */
        csDifferentialEquationModel de_model;

        /* 3. Load the model from the specified directory with input files
         *    or use the model generated by Model Builder. */
        de_model.Load(rank, inputFilesDirectory);

        /* 4. Instantiate and set the Compute Stack Evaluator.
         *    For simplicity, the sequential one is used here. */
        std::shared_ptr<csComputeStackEvaluator_t> csEvaluator(createEvaluator_Sequential());
        de_model.SetComputeStackEvaluator(csEvaluator);

        /* 5. Obtain the information from the model such as
         *    number of variables, variable names, types,
         *    absolute tolerances, initial conditions
         *    and the sparsity pattern. */
        int              N, Nnz;
        std::vector<int> IA, JA;
        de_model.GetSparsityPattern(N, Nnz, IA, JA);

        /* 6. In a loop: evaluation of equations and derivatives. */
        /*    6.1 Set the current values of state variables and derivatives
         *        using the time, x and dx_dt values from the ODE/DAE solver.
         *        In parallel simulations, the MPI C interface will be used
         *        to exchange the values and derivatives of adjacent variables
         *        between the processing elements. */
        real_t time = 1.01;
        de_model.SetAndSynchroniseData(time, &T[0], &dT_dt[0]);

        /*    6.2 Evaluate residuals. */
        std::vector<real_t> residuals(Nvariables, 0.0);
        de_model.EvaluateEquations(time, &residuals[0]);
        {
            // Compare the results from csNode_t::Evaluate and csComputeStackEvaluator_t::Evaluate functions.
            // Print only the items that are different (if any). */
            printf("Case 4. Model exchange for the model in %s\n", inputFilesDirectory.c_str());
            printf("        Comparison of residuals (only equations where fabs(F_cs_model[i] - F_nodes[i]) > %.0e are printed, if any):\n", epsilon);
            for(uint32_t i = 0; i < Nvariables; i++)
            {
                real_t difference = residuals[i] - residuals_nodes[i];
                if(std::fabs(difference) > epsilon)
                    printf("           [%5d] %.2e (%20.15f, %20.15f)\n", i, difference, residuals[i], residuals_nodes[i]);
            }
            printf("\n");
        }

        /*    6.2 Evaluate derivatives (the Jacobian matrix).
         *        csMatrixAccess_t is used as a generic interface for sparse matrix storage.
         *        inverseTimeStep is an inverse of the current step taken by the solver (1/h).
         *        It is assumed that a call to SetAndSynchroniseData has already been performed
         *        (therefore the current values set and exchanged between processing elements)
         *        which is a typical procedure in ODE/DAE solvers where the residuals/RHS are
         *        always evaluated first and then, if necessary, the derivatives evaluated and
         *        a preconditioner recomputed (in iterative methods) or a matrix re-factored
         *        (in direct methods). */
        class testMatrixAccess : public csMatrixAccess_t
        {
        public:
            testMatrixAccess(uint32_t Nnz)
            {
                J.resize(Nnz);
            }

            ~testMatrixAccess()
            {
            }

            virtual void SetItem(size_t row, size_t col, real_t value)
            {
                //printf("  J(%u,%u) = %10.3e\n", row, col, value);
            }

            std::vector<real_t> J;
        };

        testMatrixAccess ma(Nnz);
        real_t inverseTimeStep = 1E5;
        de_model.EvaluateJacobian(time, inverseTimeStep, &ma);

        /* 7. Free the resources allocated in the model and the evaluator. */
        de_model.Free();

        /* 8. Finalise MPI. */
        //MPI_Finalize(); // will be called later at the end of program
    }

    /*************************************************************************************************
     * Use Case 5: Simulate the sequential model using the DAE simulator from libOpenCS_Simulators
     *************************************************************************************************/
    printf("Case 5. Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());

    // (a) Run simulation using the input files from the specified directory:
    csSimulate(inputFilesDirectory);
    // (b) Run simulation using the generated csModel_t, string with JSON options and the directory for simulation outputs:
    //Simulate(model, simulationOptions, inputFilesDirectory);

    // Finalise MPI
    MPI_Finalize();

    return 0;
}
