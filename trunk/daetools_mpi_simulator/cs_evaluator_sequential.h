/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_EVALUATOR_SEQUENTIAL_H
#define CS_EVALUATOR_SEQUENTIAL_H

#include <string>
#include <vector>
#include <map>
#include <cs_evaluator.h>

namespace cs
{
class csComputeStackEvaluator_Sequential : public cs::csComputeStackEvaluator_t
{
public:
    csComputeStackEvaluator_Sequential();
    virtual ~csComputeStackEvaluator_Sequential();

    void FreeResources();

    /* Initialize function. */
    void Initialize(bool                    calculateSensitivities,
                    size_t                  numberOfVariables,
                    size_t                  numberOfEquationsToProcess,
                    size_t                  numberOfDOFs,
                    size_t                  numberOfComputeStackItems,
                    size_t                  numberOfJacobianItems,
                    size_t                  numberOfJacobianItemsToProcess,
                    csComputeStackItem_t*   computeStacks,
                    uint32_t*               activeEquationSetIndexes,
                    csJacobianMatrixItem_t* computeStackJacobianItems);

    /* Residual kernel function. */
    void EvaluateResiduals(csEvaluationContext_t EC,
                           real_t*               dofs,
                           real_t*               values,
                           real_t*               timeDerivatives,
                           real_t*               residuals);

    /* Jacobian kernel function (generic version). */
    void EvaluateJacobian(csEvaluationContext_t EC,
                          real_t*               dofs,
                          real_t*               values,
                          real_t*               timeDerivatives,
                          real_t*               jacobianItems);

    /* Sensitivity residual kernel function. */
    void EvaluateSensitivityResiduals(csEvaluationContext_t EC,
                                      real_t*               dofs,
                                      real_t*               values,
                                      real_t*               timeDerivatives,
                                      real_t*               svalues,
                                      real_t*               sdvalues,
                                      real_t*               sresiduals);

    std::map<std::string, call_stats::TimeAndCount> GetCallStats() const;

public:
    uint32_t                m_numberOfEquations;
    uint32_t                m_numberOfComputeStackItems;
    uint32_t                m_numberOfJacobianItems;
    uint32_t*               m_activeEquationSetIndexes;
    csComputeStackItem_t*   m_computeStacks;
    csJacobianMatrixItem_t* m_computeStackJacobianItems;

    std::map<std::string, call_stats::TimeAndCount> m_stats;
};

}

#endif
