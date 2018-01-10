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
#if !defined(ADOUBLE_CS_H)
#define ADOUBLE_CS_H

#include <math.h>
#include "compute_stack.h"

#ifdef __cplusplus
#include <stdexcept>
extern "C" {
#endif

CS_DECL adouble_cs adouble_cs_(real_t value, real_t derivative)
{
    adouble_cs a;
    a.m_dValue = value;
    a.m_dDeriv = derivative;
    return a;
}

CS_DECL void adouble_init(adouble_cs* a, real_t value, real_t derivative)
{
    a->m_dValue = value;
    a->m_dDeriv = derivative;
}

CS_DECL void adouble_copy(adouble_cs* src, adouble_cs* target)
{
    src->m_dValue = target->m_dValue;
    src->m_dDeriv = target->m_dDeriv;
}

CS_DECL real_t adouble_getValue(const adouble_cs* a)
{
    return a->m_dValue;
}

CS_DECL void adouble_setValue(adouble_cs* a, const real_t v)
{
    a->m_dValue = v;
}

CS_DECL real_t adouble_getDerivative(const adouble_cs* a)
{
    return a->m_dDeriv;
}

CS_DECL void adouble_setDerivative(adouble_cs* a, real_t v)
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

CS_DECL adouble_cs _sign_(adouble_cs a)
{
    adouble_cs tmp;

    tmp.m_dValue = -a.m_dValue;
    tmp.m_dDeriv = -a.m_dDeriv;
    return tmp;
}

CS_DECL adouble_cs _plus_(const adouble_cs a, const adouble_cs b)
{
    adouble_cs tmp;

    tmp.m_dValue = a.m_dValue + b.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv + b.m_dDeriv;
    return tmp;
}

CS_DECL adouble_cs _minus_(const adouble_cs a, const adouble_cs b)
{
    adouble_cs tmp;

    tmp.m_dValue = a.m_dValue - b.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv - b.m_dDeriv;
    return tmp;
}

CS_DECL adouble_cs _multi_(const adouble_cs a, const adouble_cs b)
{
    adouble_cs tmp;

    tmp.m_dValue = a.m_dValue * b.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv * b.m_dValue + a.m_dValue * b.m_dDeriv;
    return tmp;
}

CS_DECL adouble_cs _divide_(const adouble_cs a, const adouble_cs b)
{
    adouble_cs tmp;

    tmp.m_dValue = a.m_dValue / b.m_dValue;
    tmp.m_dDeriv = (a.m_dDeriv * b.m_dValue - a.m_dValue * b.m_dDeriv) / (b.m_dValue * b.m_dValue);
    return tmp;
}

CS_DECL adouble_cs _pow_(const adouble_cs a, const adouble_cs b)
{
    adouble_cs tmp;

    if(b.m_dDeriv == 0)
    {
        tmp.m_dValue = pow(a.m_dValue, b.m_dValue);
        real_t tmp2 = b.m_dValue * pow(a.m_dValue, b.m_dValue-1);
        tmp.m_dDeriv = tmp2 * a.m_dDeriv;
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

CS_DECL adouble_cs _log10_(const adouble_cs a)
{
    adouble_cs tmp;

    tmp.m_dValue = log10(a.m_dValue);
    real_t tmp2 = log((real_t)10) * a.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

CS_DECL adouble_cs _exp_(const adouble_cs a)
{
    adouble_cs tmp;

    tmp.m_dValue = exp(a.m_dValue);
    tmp.m_dDeriv = tmp.m_dValue * a.m_dDeriv;
    return tmp;
}

CS_DECL adouble_cs _log_(const adouble_cs a)
{
    adouble_cs tmp;

    tmp.m_dValue = log(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / a.m_dValue;
    return tmp;
}

CS_DECL adouble_cs _sqrt_(const adouble_cs a)
{
    adouble_cs tmp;

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
        tmp.m_dValue = _makeNaN_();
        tmp.m_dDeriv = _makeNaN_();
    }
    return tmp;
}

CS_DECL adouble_cs _abs_(const adouble_cs a)
{
    adouble_cs tmp;

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

CS_DECL adouble_cs _sin_(const adouble_cs a)
{
    adouble_cs tmp;
    real_t tmp2;

    tmp.m_dValue = sin(a.m_dValue);
    tmp2         = cos(a.m_dValue);

    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

CS_DECL adouble_cs _cos_(const adouble_cs a)
{
    adouble_cs tmp;
    real_t tmp2;

    tmp.m_dValue = cos(a.m_dValue);
    tmp2         = -sin(a.m_dValue);

    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

CS_DECL adouble_cs _tan_(const adouble_cs a)
{
    adouble_cs tmp;
    real_t tmp2;

    tmp.m_dValue = tan(a.m_dValue);
    tmp2         = cos(a.m_dValue);
    tmp2 *= tmp2;

    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

CS_DECL adouble_cs _asin_(const adouble_cs a)
{
    adouble_cs tmp;

    tmp.m_dValue = asin(a.m_dValue);
    real_t tmp2  = sqrt(1 - a.m_dValue * a.m_dValue);

    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

CS_DECL adouble_cs _acos_(const adouble_cs a)
{
    adouble_cs tmp;

    tmp.m_dValue =  acos(a.m_dValue);
    real_t tmp2  = -sqrt(1 - a.m_dValue*a.m_dValue);

    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

CS_DECL adouble_cs _atan_(const adouble_cs a)
{
    adouble_cs tmp;

    tmp.m_dValue = atan(a.m_dValue);
    real_t tmp2  = 1 + a.m_dValue * a.m_dValue;
    tmp2 = 1 / tmp2;
    if (tmp2 != 0)
        tmp.m_dDeriv = a.m_dDeriv * tmp2;
    else
        tmp.m_dDeriv = 0.0;
    return tmp;
}

CS_DECL adouble_cs _sinh_(const adouble_cs a)
{
    adouble_cs tmp;
    tmp.m_dValue = sinh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / sqrt(a.m_dValue*a.m_dValue + 1);
    return tmp;
}

CS_DECL adouble_cs _cosh_(const adouble_cs a)
{
    adouble_cs tmp;
    tmp.m_dValue = cosh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / sqrt(a.m_dValue*a.m_dValue - 1);
    return tmp;
}

CS_DECL adouble_cs _tanh_(const adouble_cs a)
{
    adouble_cs tmp;
    tmp.m_dValue = tanh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / (1 - a.m_dValue*a.m_dValue);
    return tmp;
}

CS_DECL adouble_cs _asinh_(const adouble_cs a)
{
    adouble_cs tmp;
    tmp.m_dValue = asinh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / sqrt(a.m_dValue*a.m_dValue + 1);
    return tmp;
}

CS_DECL adouble_cs _acosh_(const adouble_cs a)
{
    adouble_cs tmp;
    tmp.m_dValue = acosh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / sqrt(a.m_dValue*a.m_dValue - 1);
    return tmp;
}

CS_DECL adouble_cs _atanh_(const adouble_cs a)
{
    adouble_cs tmp;
    tmp.m_dValue = atanh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / (1 - a.m_dValue*a.m_dValue);
    return tmp;
}

CS_DECL adouble_cs _erf_(const adouble_cs a)
{
    adouble_cs tmp;
    tmp.m_dValue = erf(a.m_dValue);
    double pi = cos(-1);
    tmp.m_dDeriv = a.m_dDeriv * (2.0 / sqrt(pi)) * exp(-a.m_dValue*a.m_dValue);
    return tmp;
}

CS_DECL adouble_cs _atan2_(const adouble_cs a, const adouble_cs b)
{
    adouble_cs tmp;
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
CS_DECL adouble_cs _ceil_(const adouble_cs a)
{
    adouble_cs tmp;

    tmp.m_dValue = ceil(a.m_dValue);
    tmp.m_dDeriv = 0.0;
    return tmp;
}

/* floor is non-differentiable: should I remove it? */
CS_DECL adouble_cs _floor_(const adouble_cs a)
{
    adouble_cs tmp;

    tmp.m_dValue = floor(a.m_dValue);
    tmp.m_dDeriv = 0.0;
    return tmp;
}

CS_DECL adouble_cs _max_(const adouble_cs a, const adouble_cs b)
{
    adouble_cs tmp;

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

CS_DECL adouble_cs _min_(const adouble_cs a, const adouble_cs b)
{
    adouble_cs tmp;

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

CS_DECL int _neq_(const adouble_cs a, const adouble_cs b)
{
    return (a.m_dValue != b.m_dValue);
}

CS_DECL int _eq_(const adouble_cs a, const adouble_cs b)
{
    return (a.m_dValue == b.m_dValue);
}

CS_DECL int _lteq_(const adouble_cs a, const adouble_cs b)
{
    return (a.m_dValue <= b.m_dValue);
}

CS_DECL int _gteq_(const adouble_cs a, const adouble_cs b)
{
    return (a.m_dValue >= b.m_dValue);
}

CS_DECL int _gt_(const adouble_cs a, const adouble_cs b)
{
    return (a.m_dValue > b.m_dValue);
}

CS_DECL int _lt_(const adouble_cs a, const adouble_cs b)
{
    return (a.m_dValue < b.m_dValue);
}

#ifdef __cplusplus
}
#endif

#endif
