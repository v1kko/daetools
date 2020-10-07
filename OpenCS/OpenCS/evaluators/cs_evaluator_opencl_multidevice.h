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
#if !defined(CS_EVALUATOR_OPENCL_MULTIDEVICE_H)
#define CS_EVALUATOR_OPENCL_MULTIDEVICE_H

#include "cs_evaluator_opencl.h"

namespace cs
{
class OPENCS_EVALUATORS_API csComputeStackEvaluator_OpenCL_MultiDevice : public csComputeStackEvaluator_t
{
public:
    csComputeStackEvaluator_OpenCL_MultiDevice(const std::vector<int>&    platforms,
                                                const std::vector<int>&    devices,
                                                const std::vector<double>& taskPortions,
                                                std::string                buildProgramOptions);
    virtual ~csComputeStackEvaluator_OpenCL_MultiDevice();

    void FreeResources();

    void Initialize(bool                     calculateSensitivities,
                    size_t                   numberOfVariables,
                    size_t                   numberOfEquationsToProcess,
                    size_t                   numberOfDOFs,
                    size_t                   numberOfComputeStackItems,
                    size_t                   numberOfIncidenceMatrixItems,
                    size_t                   numberOfIncidenceMatrixItemsToProcess,
                    csComputeStackItem_t*    computeStacks,
                    uint32_t*                activeEquationSetIndexes,
                    csIncidenceMatrixItem_t* incidenceMatrixItems);

    /* Evaluate equations. */
    void EvaluateEquations(csEvaluationContext_t EC,
                           real_t*               dofs,
                           real_t*               values,
                           real_t*               timeDerivatives,
                           real_t*               residuals);

    /* Evaluate derivatives (Jacobian matrix). */
    void EvaluateDerivatives(csEvaluationContext_t EC,
                             real_t*               dofs,
                             real_t*               values,
                             real_t*               timeDerivatives,
                             real_t*               jacobianItems);

    /* Evaluate sensitivity derivatives. */
    void EvaluateSensitivityDerivatives(csEvaluationContext_t EC,
                                        real_t*               dofs,
                                        real_t*               values,
                                        real_t*               timeDerivatives,
                                        real_t*               svalues,
                                        real_t*               sdvalues,
                                        real_t*               sresiduals);

public:
    std::vector<int>                             m_platforms;
    std::vector<int>                             m_devices;
    std::vector<double>                          m_taskPortions;
    std::vector<csComputeStackEvaluator_OpenCL*> m_evaluators;
    std::vector<int>                             m_startEquationIndexes;
    std::vector<int>                             m_noEquations;
    std::vector<int>                             m_startJacobianIndexes;
    std::vector<int>                             m_noJacobianIndexes;
};

}
#endif
