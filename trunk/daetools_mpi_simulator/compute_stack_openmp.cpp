/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2017
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "compute_stack_openmp.h"
#include <omp.h>

#include <string>
#include <vector>
#include <map>
#include <typeinfo>
using namespace std;
#include "../Core/helpers.h"

double evaluateTime_Residuals = 0;
double evaluateTime_Jacobian  = 0;
int numEvaluate_Residuals = 0;
int numEvaluate_Jacobian = 0;

daeComputeStackEvaluator_OpenMP::daeComputeStackEvaluator_OpenMP()
{

}

daeComputeStackEvaluator_OpenMP::~daeComputeStackEvaluator_OpenMP()
{
}

void daeComputeStackEvaluator_OpenMP::FreeResources()
{
    setlocale(LC_NUMERIC, ""); // allow thousands separator
    printf("Residuals ComputeStack evaluation time  = %.4f s\n",  evaluateTime_Residuals);
    printf("Jacobian ComputeStack evaluation time   = %.4f s\n",  evaluateTime_Jacobian);
    printf("Total ComputeStack evaluation time      = %.4f s\n",  evaluateTime_Residuals+evaluateTime_Jacobian);
    printf("Number of CS residuals evaluation       = %'d\n",     numEvaluate_Residuals);
    printf("Number of CS Jacobian evaluation        = %'d\n",     numEvaluate_Jacobian);
    printf("Average time for a single CS evaluation = %.4f us\n", 1E6*(evaluateTime_Residuals+evaluateTime_Jacobian)/(numEvaluate_Residuals+numEvaluate_Jacobian+1));
}

void daeComputeStackEvaluator_OpenMP::Initialize(bool                    calculateSensitivities,
                                                 size_t                  numberOfVariables,
                                                 size_t                  numberOfEquationsToProcess,
                                                 size_t                  numberOfDOFs,
                                                 size_t                  numberOfComputeStackItems,
                                                 size_t                  numberOfJacobianItems,
                                                 size_t                  numberOfJacobianItemsToProcess,
                                                 adComputeStackItem_t*   computeStacks,
                                                 uint32_t*               activeEquationSetIndexes,
                                                 adJacobianMatrixItem_t* computeStackJacobianItems)
{
    m_numberOfEquations         = numberOfEquationsToProcess;
    m_numberOfComputeStackItems = numberOfComputeStackItems;
    m_numberOfJacobianItems     = numberOfJacobianItemsToProcess;
    m_computeStacks             = computeStacks;
    m_activeEquationSetIndexes  = activeEquationSetIndexes;
    m_computeStackJacobianItems = computeStackJacobianItems;
}

void daeComputeStackEvaluator_OpenMP::EvaluateResiduals(daeComputeStackEvaluationContext_t EC,
                                                        real_t*                            dofs,
                                                        real_t*                            values,
                                                        real_t*                            timeDerivatives,
                                                        real_t*                            residuals)
{
    //omp_set_num_threads(1);

    //#pragma omp parallel for firstprivate(EC)
    for(int ei = 0; ei < EC.numberOfEquations; ei++)
    {
        /* Locate the current equation stack in the array of all compute stacks. */
        uint32_t firstIndex                      = m_activeEquationSetIndexes[ei];
        const adComputeStackItem_t* computeStack = &m_computeStacks[firstIndex];

    numEvaluate_Residuals++;
    double start = dae::GetTimeInSeconds();
        /* Evaluate the compute stack (scaling is included). */
        adouble_t res_cs = evaluateComputeStack(computeStack,
                                                EC,
                                                dofs,
                                                values,
                                                timeDerivatives,
                                                NULL,
                                                NULL);
    evaluateTime_Residuals += (dae::GetTimeInSeconds() - start);

        /* Set the value in the residuals array. */
        residuals[ei] = res_cs.m_dValue;
    }

    //printf("   r evaluateComputeStackNo = %d\n", evaluateComputeStackNo);
    //printf("   r Cumulative evaluateComputeStack time = %.14f\n", evaluateComputeStackTime);
    //printf("   r Average evaluateComputeStack time    = %.14f\n", evaluateComputeStackTime/evaluateComputeStackNo);
}

void daeComputeStackEvaluator_OpenMP::EvaluateJacobian(daeComputeStackEvaluationContext_t EC,
                                                       real_t*                            dofs,
                                                       real_t*                            values,
                                                       real_t*                            timeDerivatives,
                                                       real_t*                            jacobianItems)
{
    //omp_set_num_threads(1);

    //#pragma omp parallel for firstprivate(EC)
    for(int ji = 0; ji < EC.numberOfJacobianItems; ji++)
    {
        /* Locate the current equation stack in the array of all compute stacks. */
        adJacobianMatrixItem_t jacobianItem      = m_computeStackJacobianItems[ji];
        uint32_t firstIndex                      = m_activeEquationSetIndexes[jacobianItem.equationIndex];
        const adComputeStackItem_t* computeStack = &m_computeStacks[firstIndex];

        /* Set the overall index for jacobian evaluation. */
        EC.jacobianIndex = jacobianItem.overallIndex;

    numEvaluate_Jacobian++;
    double start = dae::GetTimeInSeconds();
        /* Evaluate the compute stack (scaling is included). */
        adouble_t jac_cs = evaluateComputeStack(computeStack,
                                                EC,
                                                dofs,
                                                values,
                                                timeDerivatives,
                                                NULL,
                                                NULL);
    evaluateTime_Jacobian += (dae::GetTimeInSeconds() - start);

        /* Set the value in the jacobian array. */
        jacobianItems[ji] = jac_cs.m_dDeriv;
    }

    //printf("   j evaluateComputeStackNo = %d\n", evaluateComputeStackNo);
    //printf("   j Cumulative evaluateComputeStack time = %.14f\n", evaluateComputeStackTime);
    //printf("   j Average evaluateComputeStack time    = %.14f\n", evaluateComputeStackTime/evaluateComputeStackNo);
}

void daeComputeStackEvaluator_OpenMP::EvaluateSensitivityResiduals(daeComputeStackEvaluationContext_t EC,
                                                                   real_t*                            dofs,
                                                                   real_t*                            values,
                                                                   real_t*                            timeDerivatives,
                                                                   real_t*                            svalues,
                                                                   real_t*                            sdvalues,
                                                                   real_t*                            sresiduals)
{

}

