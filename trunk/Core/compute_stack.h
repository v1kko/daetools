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
#ifndef DAE_COMPUTE_STACK_H
#define DAE_COMPUTE_STACK_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef real_t
#define real_t double
#endif

#define CS_DECL static inline

// Copies of the enum types defined in "core.h" with the addition of eScaling
typedef enum
{
    eUFUnknown = 0,
    eSign,
    eSqrt,
    eExp,
    eLog,
    eLn,
    eAbs,
    eSin,
    eCos,
    eTan,
    eArcSin,
    eArcCos,
    eArcTan,
    eCeil,
    eFloor,
    eSinh,
    eCosh,
    eTanh,
    eArcSinh,
    eArcCosh,
    eArcTanh,
    eErf,
    eScaling
}daeeUnaryFunctions;

typedef enum
{
    eBFUnknown = 0,
    ePlus,
    eMinus,
    eMulti,
    eDivide,
    ePower,
    eMin,
    eMax,
    eArcTan2
}daeeBinaryFunctions;

typedef enum
{
    eECMUnknown = 0,
    eGatherInfo,
    eCalculate,
    eCreateFunctionsIFsSTNs,
    eCalculateJacobian,
    eCalculateSensitivityResiduals
}daeeEquationCalculationMode;

// Compute stack only related enums
typedef enum
{
    eOP_Unknown = 0,
    eOP_Constant,
    eOP_Time,
    eOP_InverseTimeStep,
    eOP_Variable,
    eOP_DegreeOfFreedom,
    eOP_TimeDerivative,
    eOP_Unary,
    eOP_Binary
}daeeOpCode;

typedef enum
{
    eOP_Result_Unknown = 0,
    eOP_Result_to_value,
    eOP_Result_to_lvalue,
    eOP_Result_to_rvalue,
}daeeOpResultLocation;

typedef struct adComputeStackItem_
{
    uint8_t  opCode;
    uint8_t  function;
    uint8_t  resultLocation;
    uint32_t size;
    union data_
    {
        double value;        // For constants

        struct dof_indexes_  // For degrees of freedom
        {
            uint32_t overallIndex;
            uint32_t dofIndex;
        }dof_indexes;

        struct indexes_      // For variables (algebraic and differential)
        {
            uint32_t overallIndex;
            uint32_t blockIndex;
        }indexes;
    }data;
}adComputeStackItem_t;

typedef struct adouble_cs_
{
    real_t m_dValue;
    real_t m_dDeriv;
} adouble_cs;

typedef struct daeComputeStackEvaluationContext_
{
    real_t   currentTime;
    real_t   inverseTimeStep;
    uint32_t equationCalculationMode;
    uint32_t currentParameterIndexForSensitivityEvaluation;
    uint32_t currentVariableIndexForJacobianEvaluation;
    uint32_t numberOfVariables;
    uint32_t valuesStackSize;
    uint32_t lvaluesStackSize;
    uint32_t rvaluesStackSize;
}daeComputeStackEvaluationContext_t;

typedef struct adJacobianMatrixItem_
{
    uint32_t equationIndex;
    uint32_t overallIndex;
    uint32_t blockIndex;
} adJacobianMatrixItem_t;

/* Stack evaluation function .*/
adouble_cs evaluateComputeStack(const adComputeStackItem_t*         computeStack,
                                daeComputeStackEvaluationContext_t  EC,
                                const real_t*                       dofs,
                                const real_t*                       values,
                                const real_t*                       timeDerivatives,
                                const real_t*                       svalues,
                                const real_t*                       sdvalues);

/* Residual kernel function. */
void calculate_cs_residual(const adComputeStackItem_t*         computeStacks,
                           uint32_t                            equationIndex,
                           const uint32_t*                     activeEquationSetIndexes,
                           daeComputeStackEvaluationContext_t  EC,
                           const real_t*                       dofs,
                           const real_t*                       values,
                           const real_t*                       timeDerivatives,
                           real_t*                             residuals);

/* Jacobian kernel function (dense and CSR sparse matrix versions). */
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

typedef void (*jacobian_fn)(void*, uint32_t, uint32_t, real_t);
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

/* Sensitivity residual kernel function. */
void calculate_cs_sens_residual(const adComputeStackItem_t*         computeStacks,
                                uint32_t                            equationIndex,
                                const uint32_t*                     activeEquationSetIndexes,
                                daeComputeStackEvaluationContext_t  EC,
                                const real_t*                       dofs,
                                const real_t*                       values,
                                const real_t*                       timeDerivatives,
                                const real_t*                       svalues,
                                const real_t*                       sdvalues,
                                      real_t*                       sresiduals);


#ifdef __cplusplus
}
#endif

#endif
