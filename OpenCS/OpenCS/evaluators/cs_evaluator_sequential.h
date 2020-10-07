/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_EVALUATOR_SEQUENTIAL_H
#define CS_EVALUATOR_SEQUENTIAL_H

#include <string>
#include <vector>
#include <map>
#include "../cs_evaluator.h"

namespace cs
{
class OPENCS_EVALUATORS_API csComputeStackEvaluator_Sequential : public cs::csComputeStackEvaluator_t
{
public:
    csComputeStackEvaluator_Sequential();
    virtual ~csComputeStackEvaluator_Sequential();

    void FreeResources();

    void Initialize(bool                    calculateSensitivities,
                    size_t                  numberOfVariables,
                    size_t                  numberOfEquationsToProcess,
                    size_t                  numberOfDOFs,
                    size_t                  numberOfComputeStackItems,
                    size_t                  numberOfIncidenceMatrixItems,
                    size_t                  numberOfIncidenceMatrixItemsToProcess,
                    csComputeStackItem_t*   computeStacks,
                    uint32_t*               activeEquationSetIndexes,
                    csIncidenceMatrixItem_t* incidenceMatrixItems);

    /* Residual kernel function. */
    void EvaluateEquations(csEvaluationContext_t EC,
                           real_t*               dofs,
                           real_t*               values,
                           real_t*               timeDerivatives,
                           real_t*               residuals);

    /* Jacobian kernel function (generic version). */
    void EvaluateDerivatives(csEvaluationContext_t EC,
                             real_t*               dofs,
                             real_t*               values,
                             real_t*               timeDerivatives,
                             real_t*               jacobianItems);

    /* Sensitivity residual kernel function. */
    void EvaluateSensitivityDerivatives(csEvaluationContext_t EC,
                                        real_t*               dofs,
                                        real_t*               values,
                                        real_t*               timeDerivatives,
                                        real_t*               svalues,
                                        real_t*               sdvalues,
                                        real_t*               sresiduals);

public:
    uint32_t                m_numberOfEquations;
    uint32_t                m_numberOfComputeStackItems;
    uint32_t                m_numberOfJacobianItems;
    uint32_t*               m_activeEquationSetIndexes;
    csComputeStackItem_t*   m_computeStacks;
    csIncidenceMatrixItem_t* m_computeStackJacobianItems;
};

}

#endif
