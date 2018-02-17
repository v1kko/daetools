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
#include "compute_stack.h"
using namespace computestack;

namespace openmp_evaluator
{
/* Residual kernel function. */
void EvaluateResiduals(const adComputeStackItem_t*         computeStacks,
                       uint32_t                            equationIndex,
                       const uint32_t*                     activeEquationSetIndexes,
                       daeComputeStackEvaluationContext_t  EC,
                       const real_t*                       dofs,
                       const real_t*                       values,
                       const real_t*                       timeDerivatives,
                       real_t*                             residuals);

/* Jacobian kernel functions. */
void EvaluateJacobian(const adComputeStackItem_t*         computeStacks,
                      uint32_t                            jacobianItemIndex,
                      const uint32_t*                     activeEquationSetIndexes,
                      const adJacobianMatrixItem_t*       computeStackJacobianItems,
                      daeComputeStackEvaluationContext_t  EC,
                      const real_t*                       dofs,
                      const real_t*                       values,
                      const real_t*                       timeDerivatives,
                      real_t*                             jacobian);

/* Sensitivity residual kernel function. */
void EvaluateSensitivityResiduals(const adComputeStackItem_t*         computeStacks,
                                  uint32_t                            equationIndex,
                                  const uint32_t*                     activeEquationSetIndexes,
                                  daeComputeStackEvaluationContext_t  EC,
                                  const real_t*                       dofs,
                                  const real_t*                       values,
                                  const real_t*                       timeDerivatives,
                                  const real_t*                       svalues,
                                  const real_t*                       sdvalues,
                                        real_t*                       sresiduals);

/* Unused functions
void calculate_cs_jacobian_dns(const adComputeStackItem_t*         computeStacks,
                               uint32_t                            jacobianItemIndex,
                               const uint32_t*                     activeEquationSetIndexes,
                               const adJacobianMatrixItem_t*       computeStackJacobianItems,
                               daeComputeStackEvaluationContext_t  EC,
                               const real_t*                       dofs,
                               const real_t*                       values,
                               const real_t*                       timeDerivatives,
                               real_t**                            jacobian);

void calculate_cs_jacobian_csr(const adComputeStackItem_t*         computeStacks,
                               uint32_t                            jacobianItemIndex,
                               const uint32_t*                     activeEquationSetIndexes,
                               const adJacobianMatrixItem_t*       computeStackJacobianItems,
                               daeComputeStackEvaluationContext_t  EC,
                               const real_t*                       dofs,
                               const real_t*                       values,
                               const real_t*                       timeDerivatives,
                               const int*                          IA,
                               const int*                          JA,
                               real_t*                             A);

void calculate_cs_jacobian_gen(const adComputeStackItem_t*         computeStacks,
                               uint32_t                            jacobianItemIndex,
                               const uint32_t*                     activeEquationSetIndexes,
                               const adJacobianMatrixItem_t*       computeStackJacobianItems,
                               daeComputeStackEvaluationContext_t  EC,
                               const real_t*                       dofs,
                               const real_t*                       values,
                               const real_t*                       timeDerivatives,
                               void*                               jacobianMatrix,
                               jacobian_fn                         jacobian);
*/
}
