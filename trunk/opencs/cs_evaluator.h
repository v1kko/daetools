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
#ifndef CS_COMPUTE_STACK_EVALUATOR_H
#define CS_COMPUTE_STACK_EVALUATOR_H

#include "cs_machine.h"

namespace cs
{
class csComputeStackEvaluator_t
{
public:
    virtual ~csComputeStackEvaluator_t(){}

    /* Initialisation function (can be called repeatedly after every change in the active equation set). */
    virtual void Initialize(bool                    calculateSensitivities,
                            size_t                  numberOfVariables,
                            size_t                  numberOfEquationsToProcess,
                            size_t                  numberOfDOFs,
                            size_t                  numberOfComputeStackItems,
                            size_t                  numberOfJacobianItems,
                            size_t                  numberOfJacobianItemsToProcess,
                            csComputeStackItem_t*   computeStacks,
                            uint32_t*               activeEquationSetIndexes,
                            csJacobianMatrixItem_t* computeStackJacobianItems) = 0;

    virtual void FreeResources() = 0;

    virtual void EvaluateResiduals(csEvaluationContext_t EC,
                                   real_t*               dofs,
                                   real_t*               values,
                                   real_t*               timeDerivatives,
                                   real_t*               residuals) = 0;

    virtual void EvaluateJacobian(csEvaluationContext_t EC,
                                  real_t*               dofs,
                                  real_t*               values,
                                  real_t*               timeDerivatives,
                                  real_t*               jacobianItems) = 0;

    virtual void EvaluateSensitivityResiduals(csEvaluationContext_t EC,
                                              real_t*               dofs,
                                              real_t*               values,
                                              real_t*               timeDerivatives,
                                              real_t*               svalues,
                                              real_t*               sdvalues,
                                              real_t*               sresiduals) = 0;

};

}

#endif
