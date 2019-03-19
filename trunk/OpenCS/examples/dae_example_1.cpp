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
#include "chemical_kinetics.h"
#include <OpenCS/models/cs_model_builder.h>
#include <OpenCS/simulators/cs_simulators.h>
using namespace cs;

/* Reimplementation of IDAS idasAkzoNob_dns example.
 * The chemical kinetics problem with 6 non-linear diff. equations.
 * The system is stiff.
 * The original results are in dae_example_1.csv file. */
int main(int argc, char *argv[])
{
    printf("######################################################\n");
    printf(" Chemical kinetics problem with 6 diff. equations\n");
    printf("######################################################\n");

    // Instantiate the model being simulated.
    ChemicalKinetics chem_kin;

    uint32_t    Ndofs                         = 0;
    uint32_t    Nvariables                    = chem_kin.Nequations;
    real_t      defaultVariableValue          = 0.0;
    real_t      defaultVariableTimeDerivative = 0.0;
    real_t      defaultAbsoluteTolerance      = 1e-10;
    std::string defaultVariableName           = "x";

    printf("Nvariables: %u\n", Nvariables);

    /* 1. Initialise the DAE system with the number of variables/equations. */
    csModelBuilder_t mb;
    mb.Initialize_DAE_System(Nvariables,
                             Ndofs,
                             defaultVariableValue,
                             defaultVariableTimeDerivative,
                             defaultAbsoluteTolerance,
                             defaultVariableName);
    printf("Model builder initialised\n");

    /* 2. Specify the OpenCS model. */
    // Create and set model equations using the provided time/variable/timeDerivative/dof objects.
    // The DAE system is defined as:
    //     F(x',x,y,t) = 0
    // where x' are derivatives of state variables, x are state variables,
    // y are fixed variables (degrees of freedom) and t is the current simulation time.
    const std::vector<csNumber_t>& y     = mb.GetVariables();
    const std::vector<csNumber_t>& dy_dt = mb.GetTimeDerivatives();

    std::vector<csNumber_t> equations(Nvariables);
    chem_kin.CreateEquations(y, dy_dt, equations);

    mb.SetModelEquations(equations);
    printf("Model equations generated and set\n");

    // Set variable names
    std::vector<std::string> names;
    chem_kin.GetVariableNames(names);
    mb.SetVariableNames(names);

    // Set initial conditions
    std::vector<real_t> y0(Nvariables, 0.0);
    chem_kin.SetInitialConditions(y0);
    mb.SetVariableValues(y0);


    printf("Equations expresions:\n");
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        std::string expression = equations[i].node->ToLatex(&mb);
        printf(" $$%5d: %s $$\n", i, expression.c_str());
    }


    /* 3. Generate a model for single CPU simulations. */
    // Set simulation options (specified as a string in JSON format).
    // The default options are returned by ModelBuiler.GetSimulationOptions function
    // and can be changed using the API from csSimulationOptions_t class.
    // The TimeHorizon and the ReportingInterval must be set.
    csSimulationOptionsPtr options = mb.GetSimulationOptions();
    options->SetDouble("Simulation.TimeHorizon",              180.0);
    options->SetDouble("Simulation.ReportingInterval",          1.0);
    options->SetDouble("Solver.Parameters.RelativeTolerance",  1e-8);
    std::string simulationOptions = options->ToString();

    // Partition the system to create the OpenCS model for a single CPU simulation.
    // In this case (Npe = 1) the graph partitioner is not required.
    uint32_t Npe = 1;
    std::string inputFilesDirectory = "dae_example_1-sequential";
    csGraphPartitioner_t* gp = NULL;
    std::vector<csModelPtr> models_sequential = mb.PartitionSystem(Npe, gp);
    csModelBuilder_t::ExportModels(models_sequential, inputFilesDirectory, simulationOptions);
    printf("Generated model for Npe = 1\n");

    /* 4. Simulate the exported model using the libOpenCS_Simulators. */
    printf("Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());
    MPI_Init(&argc, &argv);

    // (a) Run simulation using the input files from the specified directory:
    //csSimulate(inputFilesDirectory);
    // (b) Run simulation using the generated csModel_t, string with JSON options and the directory for simulation outputs:
    csModelPtr model = models_sequential[0];
    csSimulate(model, simulationOptions, inputFilesDirectory);

    MPI_Finalize();

    return 0;
}
