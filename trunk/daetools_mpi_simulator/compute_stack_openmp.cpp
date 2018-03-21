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

daeComputeStackEvaluator_OpenMP::daeComputeStackEvaluator_OpenMP()
{

}

daeComputeStackEvaluator_OpenMP::~daeComputeStackEvaluator_OpenMP()
{

}

void daeComputeStackEvaluator_OpenMP::FreeResources()
{

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
    for(int ei = 0; ei < EC.numberOfEquations; ei++)
    {
        /* Locate the current equation stack in the array of all compute stacks. */
        uint32_t firstIndex                      = m_activeEquationSetIndexes[ei];
        const adComputeStackItem_t* computeStack = &m_computeStacks[firstIndex];

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

void daeComputeStackEvaluator_OpenMP::EvaluateJacobian(daeComputeStackEvaluationContext_t EC,
                                                       real_t*                            dofs,
                                                       real_t*                            values,
                                                       real_t*                            timeDerivatives,
                                                       real_t*                            jacobianItems)
{
    for(int ji = 0; ji < EC.numberOfJacobianItems; ji++)
    {
        /* Locate the current equation stack in the array of all compute stacks. */
        adJacobianMatrixItem_t jacobianItem      = m_computeStackJacobianItems[ji];
        uint32_t firstIndex                      = m_activeEquationSetIndexes[jacobianItem.equationIndex];
        const adComputeStackItem_t* computeStack = &m_computeStacks[firstIndex];

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

void daeComputeStackEvaluator_OpenMP::EvaluateSensitivityResiduals(daeComputeStackEvaluationContext_t EC,
                                                                   real_t*                            dofs,
                                                                   real_t*                            values,
                                                                   real_t*                            timeDerivatives,
                                                                   real_t*                            svalues,
                                                                   real_t*                            sdvalues,
                                                                   real_t*                            sresiduals)
{

}

