/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "cs_evaluator_sequential.h"

namespace cs
{
csComputeStackEvaluator_Sequential::csComputeStackEvaluator_Sequential()
{

}

csComputeStackEvaluator_Sequential::~csComputeStackEvaluator_Sequential()
{
}

void csComputeStackEvaluator_Sequential::FreeResources()
{
}

void csComputeStackEvaluator_Sequential::Initialize(bool                    calculateSensitivities,
                                                    size_t                  numberOfVariables,
                                                    size_t                  numberOfEquationsToProcess,
                                                    size_t                  numberOfDOFs,
                                                    size_t                  numberOfComputeStackItems,
                                                    size_t                  numberOfJacobianItems,
                                                    size_t                  numberOfJacobianItemsToProcess,
                                                    csComputeStackItem_t*   computeStacks,
                                                    uint32_t*               activeEquationSetIndexes,
                                                    csJacobianMatrixItem_t* computeStackJacobianItems)
{
    m_numberOfEquations         = numberOfEquationsToProcess;
    m_numberOfComputeStackItems = numberOfComputeStackItems;
    m_numberOfJacobianItems     = numberOfJacobianItemsToProcess;
    m_computeStacks             = computeStacks;
    m_activeEquationSetIndexes  = activeEquationSetIndexes;
    m_computeStackJacobianItems = computeStackJacobianItems;
}

void csComputeStackEvaluator_Sequential::EvaluateResiduals(csEvaluationContext_t EC,
                                                           real_t*               dofs,
                                                           real_t*               values,
                                                           real_t*               timeDerivatives,
                                                           real_t*               residuals)
{
    for(int ei = 0; ei < EC.numberOfEquations; ei++)
    {
        /* Locate the current equation stack in the array of all compute stacks. */
        uint32_t firstIndex                      = m_activeEquationSetIndexes[ei];
        const csComputeStackItem_t* computeStack = &m_computeStacks[firstIndex];

        /* Evaluate the compute stack (scaling is included). */
        adouble_t res_cs = evaluateComputeStack(computeStack,
                                                EC,
                                                dofs,
                                                values,
                                                timeDerivatives,
                                                NULL,
                                                NULL);

        /* Set the value in the residuals array. */
        residuals[ei] = res_cs.m_dValue;
    }
}

void csComputeStackEvaluator_Sequential::EvaluateJacobian(csEvaluationContext_t EC,
                                                          real_t*               dofs,
                                                          real_t*               values,
                                                          real_t*               timeDerivatives,
                                                          real_t*               jacobianItems)
{
    for(int ji = 0; ji < EC.numberOfJacobianItems; ji++)
    {
        /* Locate the current equation stack in the array of all compute stacks. */
        csJacobianMatrixItem_t jacobianItem      = m_computeStackJacobianItems[ji];
        uint32_t firstIndex                      = m_activeEquationSetIndexes[jacobianItem.equationIndex];
        const csComputeStackItem_t* computeStack = &m_computeStacks[firstIndex];

        /* Set the overall index for jacobian evaluation. */
        EC.jacobianIndex = jacobianItem.overallIndex;

        /* Evaluate the compute stack (scaling is included). */
        adouble_t jac_cs = evaluateComputeStack(computeStack,
                                                EC,
                                                dofs,
                                                values,
                                                timeDerivatives,
                                                NULL,
                                                NULL);

        /* Set the value in the jacobian array. */
        jacobianItems[ji] = jac_cs.m_dDeriv;
    }
}

void csComputeStackEvaluator_Sequential::EvaluateSensitivityResiduals(csEvaluationContext_t EC,
                                                                      real_t*               dofs,
                                                                      real_t*               values,
                                                                      real_t*               timeDerivatives,
                                                                      real_t*               svalues,
                                                                      real_t*               sdvalues,
                                                                      real_t*               sresiduals)
{
    for(int ei = 0; ei < EC.numberOfEquations; ei++)
    {
        /* Locate the current equation stack in the array of all compute stacks. */
        uint32_t firstIndex                      = m_activeEquationSetIndexes[ei];
        const csComputeStackItem_t* computeStack = &m_computeStacks[firstIndex];

        /* Evaluate the compute stack (scaling is included). */
        adouble_t res_cs = evaluateComputeStack(computeStack,
                                                EC,
                                                dofs,
                                                values,
                                                timeDerivatives,
                                                svalues,
                                                sdvalues);

        /* Set the value in the sensitivity array matrix (here we access the data using its row pointers). */
        sresiduals[ei] = res_cs.m_dDeriv;
    }
}

}
