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
#include <OpenCS/evaluators/cs_evaluator_sequential.h>
#include <OpenCS/simulators/cs_simulators.h>
using namespace cs;

const char* simulation_options_json =
#include "simulation_options-dae.json"
;

/* Reimplementation of IDAS idasAkzoNob_dns example.
 * The chemical kinetics problem with 6 non-linear diff. equations.
 * The system is stiff. */
int main(int argc, char *argv[])
{
    ChemicalKinetics chem_kin;

    uint32_t    Ndofs                         = 0;
    uint32_t    Nvariables                    = chem_kin.Nequations;
    real_t      defaultVariableValue          = 0.0;
    real_t      defaultVariableTimeDerivative = 0.0;
    real_t      defaultAbsoluteTolerance      = 1e-10;
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

    /* 2. Create and set model equations using the provided time/variable/derivative/dof objects. */
    const std::vector<csNumber_t>& y     = mb.GetVariables();
    const std::vector<csNumber_t>& dy_dt = mb.GetTimeDerivatives();

    std::vector<csNumber_t> equations(Nvariables);
    chem_kin.CreateEquations(y, dy_dt, equations);

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

    /* 3. Generate a sequential model. */
    // Set initial conditions
    std::vector<real_t> y0    (Nvariables, 0.0);
    std::vector<real_t> dy0_dt(Nvariables, 0.0);

    chem_kin.SetInitialConditions(y0, dy0_dt);

    mb.SetVariableValues         (y0);
    mb.SetVariableTimeDerivatives(dy0_dt);

    real_t startTime         = 0.0;
    real_t timeHorizon       = 180.0;
    real_t reportingInterval = 1.0;
    real_t relativeTolerance = 1e-8;

    const size_t bsize = 8192;
    char buffer[bsize];
    std::snprintf(buffer, bsize, simulation_options_json, startTime, timeHorizon, reportingInterval, relativeTolerance);
    std::string simulationOptions = buffer;

    uint32_t Npe = 1;
    std::vector<std::string> balancingConstraints;
    std::string inputFilesDirectory = "dae_example_1-ChemicalKinetics";
    csGraphPartitioner_Simple partitioner;
    std::vector<csModelPtr> models_sequential = mb.PartitionSystem(Npe, &partitioner, balancingConstraints, true);
    mb.ExportModels(models_sequential, inputFilesDirectory, simulationOptions);
    printf("Generated models for dae_example_1-ChemicalKinetics\n");

    /* 4. Simulate the exported model using the libOpenCS_Simulators. */
    printf("Case 4.  Simulation of '%s' (using libOpenCS_Simulators)\n\n", inputFilesDirectory.c_str());
    MPI_Init(&argc, &argv);
    csSimulate_DAE(inputFilesDirectory);
    MPI_Finalize();

    return 0;
}

/* Results from IDAS idasAkzoNob_dns:
---------------------------------------------------------------------------------
   t        y1        y2       y3       y4       y5      y6    | nst  k      h
---------------------------------------------------------------------------------
0.00e+00 4.44e-01 1.23e-03 0.00e+00 7.00e-03 0.00e+00 3.60e-01 |   0  0 0.00e+00
1.00e-08 4.44e-01 1.23e-03 2.55e-10 7.00e-03 1.91e-11 3.60e-01 |  15  1 9.23e-09
2.57e-08 4.44e-01 1.23e-03 6.55e-10 7.00e-03 4.91e-11 3.60e-01 |  16  1 1.85e-08
6.61e-08 4.44e-01 1.23e-03 1.69e-09 7.00e-03 1.26e-10 3.60e-01 |  17  1 3.69e-08
1.70e-07 4.44e-01 1.23e-03 4.33e-09 7.00e-03 3.25e-10 3.60e-01 |  19  1 1.48e-07
4.37e-07 4.44e-01 1.23e-03 1.11e-08 7.00e-03 8.35e-10 3.60e-01 |  20  1 2.95e-07
1.12e-06 4.44e-01 1.23e-03 2.87e-08 7.00e-03 2.15e-09 3.60e-01 |  21  1 5.90e-07
2.89e-06 4.44e-01 1.23e-03 7.37e-08 7.00e-03 5.52e-09 3.60e-01 |  23  1 2.36e-06
7.44e-06 4.44e-01 1.23e-03 1.90e-07 7.00e-03 1.42e-08 3.60e-01 |  25  1 2.36e-06
1.91e-05 4.44e-01 1.23e-03 4.88e-07 7.00e-03 3.65e-08 3.60e-01 |  28  2 4.72e-06
4.92e-05 4.44e-01 1.23e-03 1.25e-06 7.00e-03 9.39e-08 3.60e-01 |  30  2 1.89e-05
1.27e-04 4.44e-01 1.23e-03 3.22e-06 7.00e-03 2.41e-07 3.60e-01 |  32  2 7.56e-05
3.25e-04 4.44e-01 1.23e-03 8.28e-06 7.00e-03 6.20e-07 3.60e-01 |  34  2 1.51e-04
8.37e-04 4.44e-01 1.22e-03 2.13e-05 7.00e-03 1.59e-06 3.60e-01 |  37  3 3.02e-04
2.15e-03 4.44e-01 1.20e-03 5.45e-05 7.00e-03 4.08e-06 3.60e-01 |  41  3 6.05e-04
5.53e-03 4.44e-01 1.16e-03 1.39e-04 7.00e-03 1.04e-05 3.60e-01 |  45  3 8.45e-04
1.42e-02 4.43e-01 1.05e-03 3.47e-04 7.00e-03 2.61e-05 3.59e-01 |  52  4 3.38e-03
3.66e-02 4.42e-01 8.07e-04 8.35e-04 7.00e-03 6.29e-05 3.59e-01 |  58  5 3.38e-03
9.41e-02 4.40e-01 4.01e-04 1.81e-03 7.00e-03 1.37e-04 3.57e-01 |  69  5 6.76e-03
2.42e-01 4.37e-01 1.18e-04 3.26e-03 7.00e-03 2.50e-04 3.55e-01 |  91  5 6.76e-03
6.22e-01 4.32e-01 1.08e-04 5.87e-03 6.99e-03 4.59e-04 3.50e-01 | 132  5 1.35e-02
1.60e+00 4.19e-01 1.30e-04 1.24e-02 6.96e-03 1.02e-03 3.38e-01 | 150  4 1.08e-01
4.12e+00 3.87e-01 2.07e-04 2.81e-02 6.77e-03 2.59e-03 3.04e-01 | 165  4 1.42e-01
1.06e+01 3.21e-01 4.82e-04 6.10e-02 5.86e-03 6.80e-03 2.18e-01 | 194  5 2.84e-01
2.72e+01 2.32e-01 9.50e-04 1.04e-01 3.07e-03 1.34e-02 8.27e-02 | 279  4 1.52e-01
7.00e+01 1.62e-01 1.15e-03 1.38e-01 7.40e-04 1.66e-02 1.39e-02 | 375  4 3.60e-01
1.80e+02 1.15e-01 1.20e-03 1.61e-01 3.66e-04 1.71e-02 4.87e-03 | 500  4 1.03e+00
*/
