/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
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

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#ifdef DAEDLL
#ifdef MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllexport)
#else // MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllimport)
#endif // MODEL_EXPORTS
#else // DAEDLL
#define DAE_CORE_API
#endif // DAEDLL

#else // WIN32
#define DAE_CORE_API 
#endif // WIN32

// Some M$ macro crap
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <float.h>
#include <stack>

#ifndef real_t
#define real_t double
#endif

/*********************************************************************************************
	adouble
**********************************************************************************************/
class DAE_CORE_API adouble 
{
public:
    adouble(void);
    adouble(const real_t value);
    adouble(const real_t value, real_t derivative);
    adouble(const adouble& a);
    virtual ~adouble();

public:
    const adouble operator -(void) const;
    const adouble operator +(void) const;

    //const adouble operator +(const real_t v) const;
    const adouble operator +(const adouble& a) const;
    friend DAE_CORE_API const adouble operator +(const real_t v, const adouble& a);
    friend DAE_CORE_API const adouble operator +(const adouble& a, const real_t v);

    //const adouble operator -(const real_t v) const;
    const adouble operator -(const adouble& a) const;
    friend DAE_CORE_API const adouble operator -(const real_t v, const adouble& a);
    friend DAE_CORE_API const adouble operator -(const adouble& a, const real_t v);

    //const adouble operator *(const real_t v) const;
    const adouble operator *(const adouble& a) const;
    friend DAE_CORE_API const adouble operator *(const real_t v, const adouble& a);
    friend DAE_CORE_API const adouble operator *(const adouble& a, const real_t v);

    //const adouble operator /(const real_t v) const;
    const adouble operator /(const adouble& a) const;
    friend DAE_CORE_API const adouble operator /(const real_t v, const adouble& a);
    friend DAE_CORE_API const adouble operator /(const adouble& a, const real_t v);
    
    friend DAE_CORE_API const adouble exp(const adouble &a);
    friend DAE_CORE_API const adouble log(const adouble &a);
    friend DAE_CORE_API const adouble sqrt(const adouble &a);
    friend DAE_CORE_API const adouble sin(const adouble &a);
    friend DAE_CORE_API const adouble cos(const adouble &a);
    friend DAE_CORE_API const adouble tan(const adouble &a);
    friend DAE_CORE_API const adouble asin(const adouble &a);
    friend DAE_CORE_API const adouble acos(const adouble &a);
    friend DAE_CORE_API const adouble atan(const adouble &a);
    
	friend DAE_CORE_API const adouble sinh(const adouble &a);
    friend DAE_CORE_API const adouble cosh(const adouble &a);
    friend DAE_CORE_API const adouble tanh(const adouble &a);
	friend DAE_CORE_API const adouble asinh(const adouble &a);
    friend DAE_CORE_API const adouble acosh(const adouble &a);
    friend DAE_CORE_API const adouble atanh(const adouble &a);
    friend DAE_CORE_API const adouble atan2(const adouble &a, const adouble &b);

    friend DAE_CORE_API const adouble pow(const adouble &a, real_t v);
    friend DAE_CORE_API const adouble pow(const adouble &a, const adouble &b);
    friend DAE_CORE_API const adouble pow(real_t v, const adouble &a);
    friend DAE_CORE_API const adouble log10(const adouble &a);

// ceil/floor are non-differentiable: should I remove them?
    friend DAE_CORE_API const adouble ceil(const adouble &a);
    friend DAE_CORE_API const adouble floor(const adouble &a);

    friend DAE_CORE_API const adouble abs(const adouble &a);
    friend DAE_CORE_API const adouble max(const adouble &a, const adouble &b);
    friend DAE_CORE_API const adouble max(real_t v, const adouble &a);
    friend DAE_CORE_API const adouble max(const adouble &a, real_t v);
    friend DAE_CORE_API const adouble min(const adouble &a, const adouble &b);
    friend DAE_CORE_API const adouble min(real_t v, const adouble &a);
    friend DAE_CORE_API const adouble min(const adouble &a, real_t v);

    void operator =(const real_t v);
    void operator =(const adouble& a);

    bool operator !=(const adouble&) const;
    bool operator !=(const real_t) const;
    friend bool operator !=(const real_t, const adouble&);

    bool operator ==(const adouble&) const;
    bool operator ==(const real_t) const;
    friend bool operator ==(const real_t, const adouble&);

    bool operator <=(const adouble&) const;
    bool operator <=(const real_t) const;
    friend bool operator <=(const real_t, const adouble&);

    bool operator >=(const adouble&) const;
    bool operator >=(const real_t) const;
    friend bool operator >= (const real_t, const adouble&);

    bool operator >(const adouble&) const;
    bool operator >(const real_t) const;
    friend bool operator >(const real_t, const adouble&);

    bool operator <(const adouble&) const;
    bool operator <(const real_t) const;
    friend bool operator <(const real_t, const adouble&);
    
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
	
    void setDerivative(real_t v) 
	{
		m_dDeriv = v;
	}	   
	
   
private:
    real_t m_dValue;
    real_t m_dDeriv;
};



#endif
