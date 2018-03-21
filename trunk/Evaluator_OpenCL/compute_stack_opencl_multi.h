/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#if !defined(COMPUTE_STACK_OPENCL_MULTI_H)
#define COMPUTE_STACK_OPENCL_MULTI_H

#include "compute_stack_opencl.h"

class daeComputeStackEvaluator_OpenCL_multi : public adComputeStackEvaluator_t
{
public:
    daeComputeStackEvaluator_OpenCL_multi(const std::vector<int>&    platforms,
                                          const std::vector<int>&    devices,
                                          const std::vector<double>& taskPortions,
                                          std::string                buildProgramOptions);
    virtual ~daeComputeStackEvaluator_OpenCL_multi();

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
    std::vector<int>                              m_platforms;
    std::vector<int>                              m_devices;
    std::vector<double>                           m_taskPortions;
    std::vector<daeComputeStackEvaluator_OpenCL*> m_evaluators;
    std::vector<int>                              m_startEquationIndexes;
    std::vector<int>                              m_noEquations;
    std::vector<int>                              m_startJacobianIndexes;
    std::vector<int>                              m_noJacobianIndexes;
};


#endif
