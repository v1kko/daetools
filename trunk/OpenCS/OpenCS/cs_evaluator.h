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
#ifndef CS_COMPUTE_STACK_EVALUATOR_H
#define CS_COMPUTE_STACK_EVALUATOR_H

#include "cs_machine.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#ifdef OpenCS_Evaluators_EXPORTS
#define OPENCS_EVALUATORS_API __declspec(dllexport)
#else
#define OPENCS_EVALUATORS_API __declspec(dllimport)
#endif
#else
#define OPENCS_EVALUATORS_API
#endif

namespace cs
{
class csComputeStackEvaluator_t
{
public:
    virtual ~csComputeStackEvaluator_t(){}

    /* Initialisation function (can be called repeatedly after every change in the active equation set). */
    virtual void Initialize(bool                     calculateSensitivities,
                            size_t                   numberOfVariables,
                            size_t                   numberOfEquationsToProcess,
                            size_t                   numberOfDOFs,
                            size_t                   numberOfComputeStackItems,
                            size_t                   numberOfIncidenceMatrixItems,
                            size_t                   numberOfIncidenceMatrixItemsToProcess,
                            csComputeStackItem_t*    computeStacks,
                            uint32_t*                activeEquationSetIndexes,
                            csIncidenceMatrixItem_t* incidenceMatrixItems) = 0;

    virtual void FreeResources() = 0;

    virtual void EvaluateEquations(csEvaluationContext_t EC,
                                   real_t*               dofs,
                                   real_t*               values,
                                   real_t*               timeDerivatives,
                                   real_t*               equations) = 0;

    virtual void EvaluateDerivatives(csEvaluationContext_t EC,
                                     real_t*               dofs,
                                     real_t*               values,
                                     real_t*               timeDerivatives,
                                     real_t*               derivatives) = 0;

    virtual void EvaluateSensitivityDerivatives(csEvaluationContext_t EC,
                                                real_t*               dofs,
                                                real_t*               values,
                                                real_t*               timeDerivatives,
                                                real_t*               svalues,
                                                real_t*               sdvalues,
                                                real_t*               sderivatives) = 0;
};

}

#endif
