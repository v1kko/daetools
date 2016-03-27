/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic, 2016
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#if !defined(ADOLC_ADOUBLE_H)
#define ADOLC_ADOUBLE_H

#include "typedefs.h"
#include <iostream>

#define NO_INLINE __attribute__((noinline))

class adouble
{
public:
    adouble(real_t value = 0, real_t derivative = 0)
    {
        m_dValue = value;
        m_dDeriv = derivative;
    }

    real_t getValue() const
    {
        return m_dValue;
    }

    void setValue(const real_t v)
    {
        m_dValue = v;
    }

    real_t getDerivative() const
    {
        return m_dDeriv;
    }

    void setDerivative(const real_t v)
    {
        m_dDeriv = v;
    }

public:
    real_t m_dValue;
    real_t m_dDeriv;
};

adouble operator -(const adouble& a);

adouble operator +(const adouble& a, const adouble& b);
adouble operator -(const adouble& a, const adouble& b);
adouble operator *(const adouble& a, const adouble& b);
adouble operator /(const adouble& a, const adouble& b);
adouble operator ^(const adouble& a, const adouble& b);

adouble exp_(const adouble& a);
adouble log_(const adouble& a);
adouble log10_(const adouble& a);
adouble sqrt_(const adouble& a);
adouble sin_(const adouble& a);
adouble cos_(const adouble& a);
adouble tan_(const adouble& a);
adouble asin_(const adouble& a);
adouble acos_(const adouble& a);
adouble atan_(const adouble& a);

adouble sinh_(const adouble& a);
adouble cosh_(const adouble& a);
adouble tanh_(const adouble& a);
adouble asinh_(const adouble& a);
adouble acosh_(const adouble& a);
adouble atanh_(const adouble& a);
adouble erf_(const adouble& a);

adouble pow_(const adouble& a, const adouble& b);

/* ceil/floor are non-differentiable: should I remove them? */
adouble ceil_(const adouble& a);
adouble floor_(const adouble& a);

adouble abs_(const adouble& a);
adouble max_(const adouble& a, const adouble& b);
adouble min_(const adouble& a, const adouble& b);
adouble atan2_(const adouble& a, const adouble& b);

bool operator !=(const adouble& a, const adouble& b);
bool operator ==(const adouble& a, const adouble& b);
bool operator <=(const adouble& a, const adouble& b);
bool operator >=(const adouble& a, const adouble& b);
bool operator > (const adouble& a, const adouble& b);
bool operator < (const adouble& a, const adouble& b);

std::ostream& operator<<(std::ostream& os, const adouble& a);

real_t _makeNaN_();

#endif
