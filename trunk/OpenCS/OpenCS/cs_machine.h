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
#ifndef CS_COMPUTE_STACK_MACHINE_H
#define CS_COMPUTE_STACK_MACHINE_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
#include <stdexcept>
namespace cs
{
#endif

/* Start declarations for inclusion into kernel sources. */

#define OPENCS_VERSION       10000
#define OPENCS_VERSION_MAJOR 1
#define OPENCS_VERSION_MINOR 0
#define OPENCS_VERSION_BUILD 0

#ifdef __OPENCL_C_VERSION__
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int   uint32_t;
#define CS_KERNEL_FLAG __global
#define CS_DECL inline

#else
#define CS_KERNEL_FLAG
#define CS_DECL static inline
#endif

#ifndef real_t
#define real_t double
#endif

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
} csUnaryFunctions;

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
} csBinaryFunctions;

typedef enum
{
    eEEMUnknown = 0,
    eEvaluateEquation,
    eEvaluateDerivative,
    eEvaluateSensitivityDerivative
} csEquationEvaluationMode;

/* Compute stack only related enums. */
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
} csOpCode;

typedef enum
{
    eOP_Result_Unknown = 0,
    eOP_Result_to_value,
    eOP_Result_to_lvalue,
    eOP_Result_to_rvalue,
} csOpResultLocation;

typedef struct csComputeStackItem_
{
    uint8_t  opCode;
    uint8_t  function;
    uint8_t  resultLocation;
    uint32_t size;           /* Size of the compute stack array */
    union data_
    {
        real_t value;        /* For constants and scaling factors */

        struct dof_indexes_  /* For degrees of freedom */
        {
            uint32_t overallIndex;
            uint32_t dofIndex;
        }dof_indexes;

        struct indexes_      /* For variables (algebraic and differential) */
        {
            uint32_t overallIndex;
            uint32_t blockIndex;
        }indexes;
    }data;
} csComputeStackItem_t;

typedef struct adouble_
{
    real_t m_dValue;
    real_t m_dDeriv;
} adouble_t;

typedef struct csEvaluationContext_
{
    real_t   currentTime;
    real_t   inverseTimeStep;
    uint32_t equationEvaluationMode;
    /* Indexes used for evaluation of Jacobian/sensitivities. */
    uint32_t sensitivityParameterIndex;
    uint32_t jacobianIndex;
    /* The total number of variables in DAE system (same for single and multiple device setups). */
    uint32_t numberOfVariables;
    /* Accelerator can process all equations (single device setup) where numberOfEquations = numberOfVariables
     * and startEquationIndex is zero or only a range of equations where startEquationIndex defines
     * the first equation to process and numberOfEquations their number. */
    uint32_t startEquationIndex;
    uint32_t numberOfEquations;
    /* Accelerator can process all jacobian items (single device setup) where numberOfIncidenceMatrixItems is the total number
     * and startJacobianIndex is zero or only a range of jacobian items where startJacobianIndex defines
     * the first jacobian item to process and numberOfIncidenceMatrixItems their number. */
    uint32_t startJacobianIndex;
    uint32_t numberOfIncidenceMatrixItems;
    /* Number of degrees of freedom (fixed variables). */
    uint32_t numberOfDOFs;
    /* Total number of compute stack items. */
    uint32_t numberOfComputeStackItems;
    /* Compute stack sizes. */
    uint32_t valuesStackSize;
    uint32_t lvaluesStackSize;
    uint32_t rvaluesStackSize;
} csEvaluationContext_t;

typedef struct csIncidenceMatrixItem_
{
    uint32_t equationIndex;
    uint32_t overallIndex;
    uint32_t blockIndex;
} csIncidenceMatrixItem_t;

typedef struct lifo_stack_
{
    int       size;
    adouble_t data[20];
    int       top;
} lifo_stack_t;


/*************************************************************************************************
 *  adouble_t implementation
**************************************************************************************************/
CS_DECL adouble_t adouble_t_(real_t value, real_t derivative)
{
    adouble_t a;
    a.m_dValue = value;
    a.m_dDeriv = derivative;
    return a;
}

CS_DECL void adouble_init(adouble_t* a, real_t value, real_t derivative)
{
    a->m_dValue = value;
    a->m_dDeriv = derivative;
}

CS_DECL void adouble_copy(adouble_t* src, adouble_t* target)
{
    src->m_dValue = target->m_dValue;
    src->m_dDeriv = target->m_dDeriv;
}

CS_DECL real_t adouble_getValue(const adouble_t* a)
{
    return a->m_dValue;
}

CS_DECL void adouble_setValue(adouble_t* a, const real_t v)
{
    a->m_dValue = v;
}

CS_DECL real_t adouble_getDerivative(const adouble_t* a)
{
    return a->m_dDeriv;
}

CS_DECL void adouble_setDerivative(adouble_t* a, real_t v)
{
    a->m_dDeriv = v;
}

CS_DECL real_t _makeNaN_()
{
#ifdef NAN
    return NAN;
#else
    return 0.0/0.0;
#endif
}

CS_DECL adouble_t _sign_(adouble_t a)
{
    adouble_t tmp;

    tmp.m_dValue = -a.m_dValue;
    tmp.m_dDeriv = -a.m_dDeriv;
    return tmp;
}

CS_DECL adouble_t _plus_(const adouble_t a, const adouble_t b)
{
    adouble_t tmp;

    tmp.m_dValue = a.m_dValue + b.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv + b.m_dDeriv;
    return tmp;
}

CS_DECL adouble_t _minus_(const adouble_t a, const adouble_t b)
{
    adouble_t tmp;

    tmp.m_dValue = a.m_dValue - b.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv - b.m_dDeriv;
    return tmp;
}

CS_DECL adouble_t _multi_(const adouble_t a, const adouble_t b)
{
    adouble_t tmp;

    tmp.m_dValue = a.m_dValue * b.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv * b.m_dValue + a.m_dValue * b.m_dDeriv;
    return tmp;
}

CS_DECL adouble_t _divide_(const adouble_t a, const adouble_t b)
{
    adouble_t tmp;

    tmp.m_dValue = a.m_dValue / b.m_dValue;
    tmp.m_dDeriv = (a.m_dDeriv * b.m_dValue - a.m_dValue * b.m_dDeriv) / (b.m_dValue * b.m_dValue);
    return tmp;
}

CS_DECL adouble_t _pow_(const adouble_t a, const adouble_t b)
{
    adouble_t tmp;

    if(b.m_dDeriv == 0)
    {
        /* To avoid logarithm of a negative number assume we have pow(adouble, const). */
        tmp.m_dValue = pow(a.m_dValue, b.m_dValue);
        real_t tmp2 = b.m_dValue * pow(a.m_dValue, b.m_dValue-1);
        tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    }
    else if(a.m_dValue <= 0)
    {
        /* Power function called for a negative base. */
        tmp.m_dValue = _makeNaN_();
        tmp.m_dDeriv = _makeNaN_();
    }
    else
    {
        tmp.m_dValue = pow(a.m_dValue, b.m_dValue);
        real_t tmp2 = b.m_dValue * pow(a.m_dValue, b.m_dValue-1);
        real_t tmp3 = log(a.m_dValue) * tmp.m_dValue;
        tmp.m_dDeriv = tmp2 * a.m_dDeriv + tmp3 * b.m_dDeriv;
    }
    return tmp;
}

CS_DECL adouble_t _exp_(const adouble_t a)
{
    adouble_t tmp;

    tmp.m_dValue = exp(a.m_dValue);
    tmp.m_dDeriv = tmp.m_dValue * a.m_dDeriv;
    return tmp;
}

CS_DECL adouble_t _log_(const adouble_t a)
{
    adouble_t tmp;

    if(a.m_dValue <= 0)
    {
        /* log(number) = NaN if the number is <= 0 */
        tmp.m_dValue = _makeNaN_();
        tmp.m_dDeriv = _makeNaN_();
    }
    else
    {
        tmp.m_dValue = log(a.m_dValue);
        tmp.m_dDeriv = a.m_dDeriv / a.m_dValue;
    }
    return tmp;
}

CS_DECL adouble_t _log10_(const adouble_t a)
{
    adouble_t tmp;

    if(a.m_dValue <= 0)
    {
        /* log10(number) = NaN if the number is <= 0 */
        tmp.m_dValue = _makeNaN_();
        tmp.m_dDeriv = _makeNaN_();
    }
    else
    {
        tmp.m_dValue = log10(a.m_dValue);
        real_t tmp2 = log((real_t)10) * a.m_dValue;
        tmp.m_dDeriv = a.m_dDeriv / tmp2;
    }
    return tmp;
}

CS_DECL adouble_t _sqrt_(const adouble_t a)
{
    adouble_t tmp;

    /* ACHTUNG, ACHTUNG!!! */
    /* sqrt(number) = NaN if the number is < 0 */
    if(a.m_dValue > 0)
    {
        tmp.m_dValue = sqrt(a.m_dValue);
        tmp.m_dDeriv = a.m_dDeriv / tmp.m_dValue / 2;
    }
    else if(a.m_dValue == 0)
    {
        tmp.m_dValue = 0; /* sqrt(0) = 0 */
        tmp.m_dDeriv = 0; /* number/0 = 0 (Is it??) */
    }
    else
    {
        /* Sqrt function called for a negative argument */
        tmp.m_dValue = _makeNaN_();
        tmp.m_dDeriv = _makeNaN_();
    }
    return tmp;
}

CS_DECL adouble_t _abs_(const adouble_t a)
{
    adouble_t tmp;

    tmp.m_dValue = fabs(a.m_dValue);
    int as = 0;
    if(a.m_dValue > 0)
        as = 1;
    if(a.m_dValue < 0)
        as = -1;
    if(as != 0)
    {
        tmp.m_dDeriv = a.m_dDeriv * as;
    }
    else
    {
        as = 0;
        if(a.m_dDeriv > 0)
            as = 1;
        if(a.m_dDeriv < 0)
            as = -1;
        tmp.m_dDeriv = a.m_dDeriv * as;
    }
    return tmp;
}

CS_DECL adouble_t _sin_(const adouble_t a)
{
    adouble_t tmp;
    real_t tmp2;

    tmp.m_dValue = sin(a.m_dValue);
    tmp2         = cos(a.m_dValue);

    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

CS_DECL adouble_t _cos_(const adouble_t a)
{
    adouble_t tmp;
    real_t tmp2;

    tmp.m_dValue = cos(a.m_dValue);
    tmp2         = -sin(a.m_dValue);

    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

CS_DECL adouble_t _tan_(const adouble_t a)
{
    adouble_t tmp;
    real_t tmp2;

    tmp.m_dValue = tan(a.m_dValue);
    tmp2         = cos(a.m_dValue);
    tmp2 *= tmp2;

    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

CS_DECL adouble_t _asin_(const adouble_t a)
{
    adouble_t tmp;

    tmp.m_dValue = asin(a.m_dValue);
    real_t tmp2  = sqrt(1 - a.m_dValue * a.m_dValue);

    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

CS_DECL adouble_t _acos_(const adouble_t a)
{
    adouble_t tmp;

    tmp.m_dValue =  acos(a.m_dValue);
    real_t tmp2  = -sqrt(1 - a.m_dValue*a.m_dValue);

    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

CS_DECL adouble_t _atan_(const adouble_t a)
{
    adouble_t tmp;

    tmp.m_dValue = atan(a.m_dValue);
    real_t tmp2  = 1 + a.m_dValue * a.m_dValue;
    tmp2 = 1 / tmp2;
    if (tmp2 != 0)
        tmp.m_dDeriv = a.m_dDeriv * tmp2;
    else
        tmp.m_dDeriv = 0.0;
    return tmp;
}

CS_DECL adouble_t _sinh_(const adouble_t a)
{
    adouble_t tmp, tmp1, tmp2;
    const adouble_t c1  = {1.0, 0.0};
    const adouble_t c05 = {0.5, 0.0};

    if(a.m_dValue < 0.0)
    {
        tmp = _exp_(a);
        tmp1 = _divide_(c1, tmp);
        tmp2 = _minus_(tmp, tmp1);
        return _multi_(c05, tmp2);
        /*return 0.5*(tmp - 1.0/tmp);*/
    }
    else
    {
        tmp = _exp_(_sign_(a));
        tmp1 = _divide_(c1, tmp);
        tmp2 = _minus_(tmp1, tmp);
        return _multi_(c05, tmp2);
        /*return 0.5*(1.0/tmp - tmp);*/
    }
}

CS_DECL adouble_t _cosh_(const adouble_t a)
{
    adouble_t tmp, tmp1, tmp2;
    const adouble_t c1  = {1.0, 0.0};
    const adouble_t c05 = {0.5, 0.0};

    if(a.m_dValue < 0.0)
        tmp = _exp_(a);
    else
        tmp = _exp_(_sign_(a));
    tmp1 = _divide_(c1, tmp);
    tmp2 = _plus_(tmp, tmp1);
    return _multi_(c05, tmp2);
    /*return 0.5*(tmp + 1.0/tmp);*/
}

CS_DECL adouble_t _tanh_(const adouble_t a)
{
    adouble_t tmp, tmp1, tmp2;
    const adouble_t c1 = {1.0, 0.0};
    const adouble_t c2 = {2.0, 0.0};
    if(a.m_dValue < 0.0)
    {
        tmp = _exp_(_multi_(c2, a));
        /*tmp = exp(2.0*a);*/
        tmp1 = _minus_(tmp, c1);
        tmp2 = _plus_(tmp, c1);
        /*return (tmp - 1.0)/(tmp + 1.0);*/
    }
    else
    {
        tmp = _exp_(_multi_(_sign_(c2), a));
        /*tmp = exp(-2.0*a);*/
        tmp1 = _minus_(c1, tmp);
        tmp2 = _plus_(tmp, c1);
        /*return (1.0 - tmp)/(tmp + 1.0);*/
    }
    return _divide_(tmp1, tmp2);
}

CS_DECL adouble_t _asinh_(const adouble_t a)
{
    adouble_t tmp;
    tmp.m_dValue = asinh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / sqrt(a.m_dValue*a.m_dValue + 1);
    return tmp;
}

CS_DECL adouble_t _acosh_(const adouble_t a)
{
    adouble_t tmp;
    tmp.m_dValue = acosh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / sqrt(a.m_dValue*a.m_dValue - 1);
    return tmp;
}

CS_DECL adouble_t _atanh_(const adouble_t a)
{
    adouble_t tmp;
    tmp.m_dValue = atanh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / (1 - a.m_dValue*a.m_dValue);
    return tmp;
}

CS_DECL adouble_t _erf_(const adouble_t a)
{
    adouble_t tmp;
    tmp.m_dValue = erf(a.m_dValue);
    double pi = cos(-1.0);
    tmp.m_dDeriv = a.m_dDeriv * (2.0 / sqrt(pi)) * exp(-a.m_dValue*a.m_dValue);
    return tmp;
}

CS_DECL adouble_t _atan2_(const adouble_t a, const adouble_t b)
{
    adouble_t tmp;
    tmp.m_dValue = atan2(a.m_dValue, b.m_dValue);
    double tmp2 = a.m_dValue*a.m_dValue;
    double tmp3 = b.m_dValue*b.m_dValue;
    double tmp4 = tmp3 / (tmp2 + tmp3);
    if(tmp4 != 0)
        tmp.m_dDeriv = (a.m_dDeriv*b.m_dValue - a.m_dValue*b.m_dDeriv) / tmp3*tmp4;
    else
        tmp.m_dDeriv = 0.0;
    return tmp;
}

/* ceil is non-differentiable: should I remove it? */
CS_DECL adouble_t _ceil_(const adouble_t a)
{
    adouble_t tmp;

    tmp.m_dValue = ceil(a.m_dValue);
    tmp.m_dDeriv = 0.0;
    return tmp;
}

/* floor is non-differentiable: should I remove it? */
CS_DECL adouble_t _floor_(const adouble_t a)
{
    adouble_t tmp;

    tmp.m_dValue = floor(a.m_dValue);
    tmp.m_dDeriv = 0.0;
    return tmp;
}

CS_DECL adouble_t _max_(const adouble_t a, const adouble_t b)
{
    adouble_t tmp;

    real_t tmp2 = a.m_dValue - b.m_dValue;
    if(tmp2 < 0)
    {
        tmp.m_dValue = b.m_dValue;
        tmp.m_dDeriv = b.m_dDeriv;
    }
    else
    {
        tmp.m_dValue = a.m_dValue;
        if(tmp2 > 0)
        {
            tmp.m_dDeriv = a.m_dDeriv;
        }
        else
        {
            if(a.m_dDeriv < b.m_dDeriv)
                tmp.m_dDeriv = b.m_dDeriv;
            else
                tmp.m_dDeriv = a.m_dDeriv;
        }
    }
    return tmp;
}

CS_DECL adouble_t _min_(const adouble_t a, const adouble_t b)
{
    adouble_t tmp;

    real_t tmp2 = a.m_dValue - b.m_dValue;
    if(tmp2 < 0)
    {
        tmp.m_dValue = a.m_dValue;
        tmp.m_dDeriv = a.m_dDeriv;
    }
    else
    {
        tmp.m_dValue = b.m_dValue;
        if(tmp2 > 0)
        {
            tmp.m_dDeriv = b.m_dDeriv;
        }
        else
        {
            if(a.m_dDeriv < b.m_dDeriv)
                tmp.m_dDeriv = a.m_dDeriv;
            else
                tmp.m_dDeriv = b.m_dDeriv;
        }
    }
    return tmp;
}

CS_DECL int _neq_(const adouble_t a, const adouble_t b)
{
    return (a.m_dValue != b.m_dValue);
}

CS_DECL int _eq_(const adouble_t a, const adouble_t b)
{
    return (a.m_dValue == b.m_dValue);
}

CS_DECL int _lteq_(const adouble_t a, const adouble_t b)
{
    return (a.m_dValue <= b.m_dValue);
}

CS_DECL int _gteq_(const adouble_t a, const adouble_t b)
{
    return (a.m_dValue >= b.m_dValue);
}

CS_DECL int _gt_(const adouble_t a, const adouble_t b)
{
    return (a.m_dValue > b.m_dValue);
}

CS_DECL int _lt_(const adouble_t a, const adouble_t b)
{
    return (a.m_dValue < b.m_dValue);
}


/*************************************************************************************************
 *  LIFO stack implementation
**************************************************************************************************/
CS_DECL void lifo_init(lifo_stack_t* stack, int stack_size)
{
    if(stack_size > 0)
    {
        stack->size = stack_size;
        /*stack->data = (adouble_t*)malloc(stack_size * sizeof(adouble_t));*/
    }
    else
    {
        stack->size = 0;
        /*stack->data = NULL;*/
    }
    stack->top = -1;
}

CS_DECL void lifo_free(lifo_stack_t* stack)
{
    stack->size = 0;
    /*free(stack->data);*/
    stack->top = -1;
}

CS_DECL int lifo_isempty(lifo_stack_t* stack)
{
    if(stack->top == -1)
        return 1;
    else
        return 0;
}

CS_DECL int lifo_isfull(lifo_stack_t* stack)
{
    if(stack->top == stack->size - 1)
        return 1;
    else
        return 0;
}

CS_DECL adouble_t lifo_top(lifo_stack_t* stack)
{
#ifdef __cplusplus
    if(stack->top < 0)
        throw std::runtime_error("lifo_top: top < 0");
#endif
    return stack->data[stack->top];
}

CS_DECL void lifo_pop(lifo_stack_t* stack)
{
    if(!lifo_isempty(stack))
    {
        stack->top = stack->top - 1;
    }
    else
    {
#ifdef __cplusplus
        throw std::runtime_error("lifo_pop: stack is empty");
#endif
    }
}

CS_DECL void lifo_push(lifo_stack_t* stack, adouble_t* item)
{
    if(!lifo_isfull(stack))
    {
        stack->top = stack->top + 1;
        adouble_copy(&stack->data[stack->top], item);
    }
    else
    {
#ifdef __cplusplus
        throw std::runtime_error("lifo_pop: stack is full");
#endif
    }
}


/*************************************************************************************************
 *  Evaluate function
**************************************************************************************************/
CS_DECL adouble_t evaluateComputeStack(CS_KERNEL_FLAG const csComputeStackItem_t*  computeStack,
                                       csEvaluationContext_t                       EC,
                                       CS_KERNEL_FLAG const real_t*                dofs,
                                       CS_KERNEL_FLAG const real_t*                values,
                                       CS_KERNEL_FLAG const real_t*                timeDerivatives,
                                       CS_KERNEL_FLAG const real_t*                svalues,
                                       CS_KERNEL_FLAG const real_t*                sdvalues)
{
    lifo_stack_t  value, lvalue, rvalue;
    lifo_init(&value,  EC.valuesStackSize);
    lifo_init(&lvalue, EC.lvaluesStackSize);
    lifo_init(&rvalue, EC.rvaluesStackSize);

    /* Get the length of the compute stack (it is always in the 'size' member in the adComputeStackItem_t struct). */
    csComputeStackItem_t item0 = computeStack[0];
    uint32_t computeStackSize  = item0.size;

    adouble_t result, scaling;
    for(uint32_t i = 0; i < computeStackSize; i++)
    {
        const csComputeStackItem_t item = computeStack[i];

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

            if(EC.equationEvaluationMode == eEvaluateSensitivityDerivative)
            {
                if(EC.sensitivityParameterIndex == item.data.indexes.overallIndex)
                {
                    /* We should never reach this point, since the variable must be a degree of freedom. */
#ifdef __cplusplus
                    throw std::runtime_error("eOP_Variable invalid call (eEvaluateSensitivityDerivative)");
#endif
                }
                else
                {
                    /* Get the derivative value based on the blockIndex. */
                    result.m_dDeriv = svalues[item.data.indexes.blockIndex];
                }
            }
            else /* eEvaluateEquation or eEvaluateDerivative */
            {
                result.m_dDeriv = (EC.jacobianIndex == item.data.indexes.overallIndex ? 1 : 0 );
            }
        }
        else if(item.opCode == eOP_DegreeOfFreedom)
        {
            /* Take the value from the dofs array. */
            result.m_dValue = dofs[item.data.dof_indexes.dofIndex];

            /* DOFs can have derivatives only when calculating sensitivities. */
            if(EC.equationEvaluationMode == eEvaluateSensitivityDerivative)
            {
                /* The derivative is non-zero only if the DOF overall index is equal to the requested sensitivity parameter index. */
                if(EC.sensitivityParameterIndex == item.data.dof_indexes.overallIndex)
                {
                    result.m_dDeriv = 1;
                }
            }
        }
        else if(item.opCode == eOP_TimeDerivative)
        {
            /* Take the value from the time derivatives array. */
            result.m_dValue = timeDerivatives[item.data.indexes.blockIndex];

            if(EC.equationEvaluationMode == eEvaluateSensitivityDerivative)
            {
                /* Index for the sensitivity residual can never be equal to an overallIndex
                 * since it would be a degree of freedom. */
#ifdef __cplusplus
                if(EC.sensitivityParameterIndex == item.data.indexes.overallIndex)
                    throw std::runtime_error("eOP_TimeDerivative invalid call (eEvaluateSensitivityDerivative)");
#endif

                result.m_dDeriv = sdvalues[item.data.indexes.blockIndex];
            }
            else /* eEvaluateEquation or eEvaluateDerivative */
            {
                result.m_dDeriv = (EC.jacobianIndex == item.data.indexes.overallIndex ? EC.inverseTimeStep : 0);
            }
        }
        else if(item.opCode == eOP_Unary)
        {
            adouble_t arg = lifo_top(&value);

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
            adouble_t left  = lifo_top(&lvalue);
            adouble_t right = lifo_top(&rvalue);

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
        }
        else if(item.resultLocation == eOP_Result_to_lvalue)
        {
            lifo_push(&lvalue, &result);
        }
        else if(item.resultLocation == eOP_Result_to_rvalue)
        {
            lifo_push(&rvalue, &result);
        }
        else
        {
#ifdef __cplusplus
            throw std::runtime_error("Invalid resultLocation code");
#endif
        }
    }

    adouble_t result_final = lifo_top(&value);
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

/* End of declarations for inclusion into kernel sources. */

#ifdef __cplusplus
} /* End of namespace cs */
#endif

#endif
