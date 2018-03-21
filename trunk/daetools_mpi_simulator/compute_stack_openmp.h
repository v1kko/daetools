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
#ifndef DAE_COMPUTE_STACK_OPENMP_H
#define DAE_COMPUTE_STACK_OPENMP_H

#include "compute_stack.h"
using namespace computestack;

class daeComputeStackEvaluator_OpenMP : public adComputeStackEvaluator_t
{
public:
    daeComputeStackEvaluator_OpenMP();
    virtual ~daeComputeStackEvaluator_OpenMP();

    void FreeResources();

    /* Initialize function. */
    void Initialize(bool                    calculateSensitivities,
                    size_t                  numberOfVariables,
                    size_t                  numberOfEquationsToProcess,
                    size_t                  numberOfDOFs,
                    size_t                  numberOfComputeStackItems,
                    size_t                  numberOfJacobianItems,
                    size_t                  numberOfJacobianItemsToProcess,
                    adComputeStackItem_t*   computeStacks,
                    uint32_t*               activeEquationSetIndexes,
                    adJacobianMatrixItem_t* computeStackJacobianItems);

    /* Residual kernel function. */
    void EvaluateResiduals(daeComputeStackEvaluationContext_t EC,
                           real_t*                            dofs,
                           real_t*                            values,
                           real_t*                            timeDerivatives,
                           real_t*                            residuals);

    /* Jacobian kernel function (generic version). */
    void EvaluateJacobian(daeComputeStackEvaluationContext_t EC,
                          real_t*                            dofs,
                          real_t*                            values,
                          real_t*                            timeDerivatives,
                          real_t*                            jacobianItems);

    /* Sensitivity residual kernel function. */
    void EvaluateSensitivityResiduals(daeComputeStackEvaluationContext_t EC,
                                      real_t*                            dofs,
                                      real_t*                            values,
                                      real_t*                            timeDerivatives,
                                      real_t*                            svalues,
                                      real_t*                            sdvalues,
                                      real_t*                            sresiduals);

public:
    uint32_t                m_numberOfEquations;
    uint32_t                m_numberOfComputeStackItems;
    uint32_t                m_numberOfJacobianItems;
    uint32_t*               m_activeEquationSetIndexes;
    adComputeStackItem_t*   m_computeStacks;
    adJacobianMatrixItem_t* m_computeStackJacobianItems;
};

#endif
