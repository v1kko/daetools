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
#include "compute_stack_adouble.h"
#include "compute_stack_lifo.h"

adouble_cs evaluateComputeStack(const adComputeStackItem_t*         computeStack,
                                daeComputeStackEvaluationContext_t  EC,
                                const real_t*                       dofs,
                                const real_t*                       values,
                                const real_t*                       timeDerivatives,
                                const real_t*                       svalues,
                                const real_t*                       sdvalues)
{
    lifo_stack_t  value, lvalue, rvalue;
    lifo_init(&value,  EC.valuesStackSize);
    lifo_init(&lvalue, EC.lvaluesStackSize);
    lifo_init(&rvalue, EC.rvaluesStackSize);

    /* Get the length of the compute stack (it is always in the 'size' member in the adComputeStackItem_t struct). */
    adComputeStackItem_t item = computeStack[0];
    uint32_t computeStackSize = item.size;

    adouble_cs result;
    for(uint32_t i = 0; i < computeStackSize; i++)
    {
        const adComputeStackItem_t item = computeStack[i];

        adouble_init(&result, 0.0, 0.0);

        if(item.opCode == eOP_Constant)
        {
            result.m_dValue = item.data.value;
        }
        else if(item.opCode == eOP_Time)
        {
            result.m_dValue = EC.currentTime;
        }
        else if(item.opCode == eOP_InverseTimeStep)
        {
            result.m_dValue = EC.inverseTimeStep;
        }
        else if(item.opCode == eOP_Variable)
        {
            /* Take the value from the values array. */
            result.m_dValue = values[item.data.indexes.blockIndex];

            if(EC.equationCalculationMode == eCalculateSensitivityResiduals)
            {
                if(EC.currentParameterIndexForSensitivityEvaluation == item.data.indexes.overallIndex)
                {
                    /* We should never reach this point, since the variable must be a degree of freedom. */
#ifdef __cplusplus
                    throw std::runtime_error("eOP_Variable invalid call (eCalculateSensitivityResiduals)");
#endif
                }
                else
                {
                    /* Get the derivative value based on the blockIndex. */
                    result.m_dDeriv = svalues[item.data.indexes.blockIndex];
                }
            }
            else /* eCalculate or eCalculateJacobian. */
            {
                result.m_dDeriv = (EC.currentVariableIndexForJacobianEvaluation == item.data.indexes.overallIndex ? 1 : 0 );
            }
        }
        else if(item.opCode == eOP_DegreeOfFreedom)
        {
            /* Take the value from the dofs array. */
            result.m_dValue = dofs[item.data.dof_indexes.dofIndex];

            /* DOFs can have derivatives only when calculating sensitivities. */
            if(EC.equationCalculationMode == eCalculateSensitivityResiduals)
            {
                /* The derivative is non-zero only if the DOF overall index is equal to the requested sensitivity parameter index. */
                if(EC.currentParameterIndexForSensitivityEvaluation == item.data.dof_indexes.overallIndex)
                {
                    result.m_dDeriv = 1;
                }
            }
        }
        else if(item.opCode == eOP_TimeDerivative)
        {
            /* Take the value from the time derivatives array. */
            result.m_dValue = timeDerivatives[item.data.indexes.blockIndex];

            if(EC.equationCalculationMode == eCalculateSensitivityResiduals)
            {
                /* Index for the sensitivity residual can never be equal to an overallIndex
                 * since it would be a degree of freedom. */
#ifdef __cplusplus
                if(EC.currentParameterIndexForSensitivityEvaluation == item.data.indexes.overallIndex)
                    throw std::runtime_error("eOP_TimeDerivative invalid call (eCalculateSensitivityResiduals)");
#endif

                result.m_dDeriv = sdvalues[item.data.indexes.blockIndex];
            }
            else /* eCalculate or eCalculateJacobian */
            {
                result.m_dDeriv = (EC.currentVariableIndexForJacobianEvaluation == item.data.indexes.overallIndex ? EC.inverseTimeStep : 0);
            }
        }
        else if(item.opCode == eOP_Unary)
        {
            adouble_cs arg = lifo_top(&value);

            switch(item.function)
            {
                case eSign:
                    result = _sign_(arg);
                    break;
                case eSin:
                    result = _sin_(arg);
                    break;
                case eCos:
                    result = _cos_(arg);
                    break;
                case eTan:
                    result = _tan_(arg);
                    break;
                case eArcSin:
                    result = _asin_(arg);
                    break;
                case eArcCos:
                    result = _acos_(arg);
                    break;
                case eArcTan:
                    result = _atan_(arg);
                    break;
                case eSqrt:
                    result = _sqrt_(arg);
                    break;
                case eExp:
                    result = _exp_(arg);
                    break;
                case eLn:
                    result = _log_(arg);
                    break;
                case eLog:
                    result = _log10_(arg);
                    break;
                case eAbs:
                    result = _abs_(arg);
                    break;
                case eCeil:
                    result = _ceil_(arg);
                    break;
                case eFloor:
                    result = _floor_(arg);
                    break;
                case eSinh:
                    result = _sinh_(arg);
                    break;
                case eCosh:
                    result = _cosh_(arg);
                    break;
                case eTanh:
                    result = _tanh_(arg);
                    break;
                case eArcSinh:
                    result = _asinh_(arg);
                    break;
                case eArcCosh:
                    result = _acosh_(arg);
                    break;
                case eArcTanh:
                    result = _atanh_(arg);
                    break;
                case eErf:
                    result = _erf_(arg);
                    break;
                case eScaling:
                    /* Scaling op code only exists in compute stacks to inlude the equation scaling
                     * in the compute stack (stored in the data.value member). */
                    adouble_cs scaling;
                    adouble_init(&scaling, item.data.value, 0.0);
                    result = _multi_(scaling, arg);
                    break;
                default:
#ifdef __cplusplus
                    throw std::runtime_error("Invalid unary function");
#endif
                    break;
            }

            lifo_pop(&value);
        }
        else if(item.opCode == eOP_Binary)
        {
            adouble_cs left  = lifo_top(&lvalue);
            adouble_cs right = lifo_top(&rvalue);

            switch(item.function)
            {
                case ePlus:
                    result = _plus_(left, right);
                    break;
                case eMinus:
                    result = _minus_(left, right);
                    break;
                case eMulti:
                    result = _multi_(left, right);
                    break;
                case eDivide:
                    result = _divide_(left, right);
                    break;
                case ePower:
                    result = _pow_(left, right);
                    break;
                case eMin:
                    result = _min_(left, right);
                    break;
                case eMax:
                    result = _max_(left, right);
                    break;
                case eArcTan2:
                    result = _atan2_(left, right);
                    break;
                default:
#ifdef __cplusplus
                    throw std::runtime_error("Invalid binary function");
#endif
                    break;
            }

            lifo_pop(&lvalue);
            lifo_pop(&rvalue);
        }
        else
        {
#ifdef __cplusplus
            throw std::runtime_error("Invalid op code");
#endif
        }

        /* At the end push the result into the requested stack. */
        if(item.resultLocation == eOP_Result_to_value)
        {
            lifo_push(&value, &result);

            /*adouble_cs ad_top = lifo_top(&value);
              printf("  ad_top value = %.14f deriv=%.14f \n", ad_top.m_dValue, ad_top.m_dDeriv); */
        }
        else if(item.resultLocation == eOP_Result_to_lvalue)
        {
            lifo_push(&lvalue, &result);

            /*adouble_cs ad_top = lifo_top(&lvalue);
              printf("  ad_top lvalue = %.14f deriv=%.14f \n", ad_top.m_dValue, ad_top.m_dDeriv); */
        }
        else if(item.resultLocation == eOP_Result_to_rvalue)
        {
            lifo_push(&rvalue, &result);

            /*adouble_cs ad_top = lifo_top(&rvalue);
              printf("  ad_top rvalue = %.14f deriv=%.14f \n", ad_top.m_dValue, ad_top.m_dDeriv); */
        }
        else
        {
#ifdef __cplusplus
            throw std::runtime_error("Invalid resultLocation code");
#endif
        }
    }

    adouble_cs result_final = lifo_top(&value);
    lifo_pop(&value);

    /* printf("  Result final val = %.14f deriv=%.14f \n", result_final.m_dValue, result_final.m_dDeriv); */

    /* Everything that has been put on stack must be removed during the evaluation. */
#ifdef __cplusplus
    if(!lifo_isempty(&value))
        throw std::runtime_error("Length of the value list is not zero");
    if(!lifo_isempty(&lvalue))
        throw std::runtime_error("Length of the lvalue list is not zero");
    if(!lifo_isempty(&rvalue))
        throw std::runtime_error("Length of the rvalue list is not zero");
#endif

    lifo_free(&value);
    lifo_free(&lvalue);
    lifo_free(&rvalue);

    return result_final;
}

/* Residual kernel function .*/
void calculate_cs_residual(const adComputeStackItem_t*         computeStacks,
                           uint32_t                            equationIndex,
                           const uint32_t*                     activeEquationSetIndexes,
                           daeComputeStackEvaluationContext_t  EC,
                           const real_t*                       dofs,
                           const real_t*                       values,
                           const real_t*                       timeDerivatives,
                           real_t*                             residuals)
{
    /* Locate the current equation stack in the array of all compute stacks. */
    uint32_t firstIndex                      = activeEquationSetIndexes[equationIndex];
    const adComputeStackItem_t* computeStack = &computeStacks[firstIndex];

    /* Evaluate the compute stack (scaling is included). */
    adouble_cs res_cs = evaluateComputeStack(computeStack,
                                             EC,
                                             dofs,
                                             values,
                                             timeDerivatives,
                                             NULL,
                                             NULL);

    /* Set the value in the residuals array. */
    residuals[equationIndex] = res_cs.m_dValue;
}

/* Jacobian kernel function (dense and CSR sparse matrix versions). */
void calculate_cs_jacobian_dns(const adComputeStackItem_t*         computeStacks,
                               uint32_t                            jacobianItemIndex,
                               const uint32_t*                     activeEquationSetIndexes,
                               const adJacobianMatrixItem_t*       computeStackJacobianItems,
                               daeComputeStackEvaluationContext_t  EC,
                               const real_t*                       dofs,
                               const real_t*                       values,
                               const real_t*                       timeDerivatives,
                               real_t**                            jacobian)
{
    /* Locate the current equation stack in the array of all compute stacks. */
    adJacobianMatrixItem_t jacobianItem      = computeStackJacobianItems[jacobianItemIndex];
    uint32_t firstIndex                      = activeEquationSetIndexes[jacobianItem.equationIndex];
    const adComputeStackItem_t* computeStack = &computeStacks[firstIndex];

    /* Set the overall index for jacobian evaluation. */
    EC.currentVariableIndexForJacobianEvaluation = jacobianItem.overallIndex;

    /* Evaluate the compute stack (scaling is included). */
    adouble_cs jac_cs = evaluateComputeStack(computeStack,
                                             EC,
                                             dofs,
                                             values,
                                             timeDerivatives,
                                             NULL,
                                             NULL);

    /* Set the value in the jacobian dense matrix.
     * IDA solver uses the column wise data format. */
    int row = jacobianItem.equationIndex;
    int col = jacobianItem.blockIndex;
    jacobian[col][row] = jac_cs.m_dDeriv;
}

CS_DECL void setCSRMatrixItem(int row, int col, real_t value, const int* IA, const int* JA, real_t* A)
{
    /* IA contains number of column indexes in the row: Ncol_indexes = IA[row+1] - IA[row]
     * Column indexes start at JA[ IA[row] ] and end at JA[ IA[row+1] ] */
    for(int k = IA[row]; k < IA[row+1]; k++)
    {
        if(col == JA[k])
        {
            A[k] = value;
            return;
        }
    }

#ifdef __cplusplus
    throw std::runtime_error("Invalid element in CRS matrix");
#endif
}

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
                               real_t*                             A)
{
    /* Locate the current equation stack in the array of all compute stacks. */
    adJacobianMatrixItem_t jacobianItem      = computeStackJacobianItems[jacobianItemIndex];
    uint32_t firstIndex                      = activeEquationSetIndexes[jacobianItem.equationIndex];
    const adComputeStackItem_t* computeStack = &computeStacks[firstIndex];

    /* Set the overall index for jacobian evaluation. */
    EC.currentVariableIndexForJacobianEvaluation = jacobianItem.overallIndex;

    /* Evaluate the compute stack (scaling is included). */
    adouble_cs jac_cs = evaluateComputeStack(computeStack,
                                             EC,
                                             dofs,
                                             values,
                                             timeDerivatives,
                                             NULL,
                                             NULL);

    /* Set the value in the jacobian sparse matrix (CSR format). */
    int row = jacobianItem.equationIndex;
    int col = jacobianItem.blockIndex;
    setCSRMatrixItem(row, col, jac_cs.m_dDeriv, IA, JA, A);
}

/* Version that accepts the function pointer for setting the results. */
void calculate_cs_jacobian_gen(const adComputeStackItem_t*         computeStacks,
                               uint32_t                            jacobianItemIndex,
                               const uint32_t*                     activeEquationSetIndexes,
                               const adJacobianMatrixItem_t*       computeStackJacobianItems,
                               daeComputeStackEvaluationContext_t  EC,
                               const real_t*                       dofs,
                               const real_t*                       values,
                               const real_t*                       timeDerivatives,
                               void*                               jacobianMatrix,
                               jacobian_fn                         jacobian)
{
    /* Locate the current equation stack in the array of all compute stacks. */
    adJacobianMatrixItem_t jacobianItem      = computeStackJacobianItems[jacobianItemIndex];
    uint32_t firstIndex                      = activeEquationSetIndexes[jacobianItem.equationIndex];
    const adComputeStackItem_t* computeStack = &computeStacks[firstIndex];

    /* Set the overall index for jacobian evaluation. */
    EC.currentVariableIndexForJacobianEvaluation = jacobianItem.overallIndex;

    /* Evaluate the compute stack (scaling is included). */
    adouble_cs jac_cs = evaluateComputeStack(computeStack,
                                             EC,
                                             dofs,
                                             values,
                                             timeDerivatives,
                                             NULL,
                                             NULL);

    /* Set the value in the jacobian dense matrix. */
    int row = jacobianItem.equationIndex;
    int col = jacobianItem.blockIndex;
    jacobian(jacobianMatrix, row, col, jac_cs.m_dDeriv);
}

/* Residual kernel function .*/
void calculate_cs_sens_residual(const adComputeStackItem_t*         computeStacks,
                                uint32_t                            equationIndex,
                                const uint32_t*                     activeEquationSetIndexes,
                                daeComputeStackEvaluationContext_t  EC,
                                const real_t*                       dofs,
                                const real_t*                       values,
                                const real_t*                       timeDerivatives,
                                const real_t*                       svalues,
                                const real_t*                       sdvalues,
                                      real_t*                       sresiduals)
{
    /* Locate the current equation stack in the array of all compute stacks. */
    uint32_t firstIndex                      = activeEquationSetIndexes[equationIndex];
    const adComputeStackItem_t* computeStack = &computeStacks[firstIndex];

    /* Evaluate the compute stack (scaling is included). */
    adouble_cs res_cs = evaluateComputeStack(computeStack,
                                             EC,
                                             dofs,
                                             values,
                                             timeDerivatives,
                                             svalues,
                                             sdvalues);

    /* Set the value in the sensitivity array matrix (here we access the data using its row pointers). */
    sresiduals[equationIndex] = res_cs.m_dDeriv;
}
