/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic, 2010
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include <string>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "adouble.h"

inline real_t fmin(const real_t &x, const real_t &y) 
{
    if(x < y) 
		return x;
    else 
		return y;
}

inline real_t fmax(const real_t &x, const real_t &y ) 
{
    if(x > y) 
		return x;
    else 
		return y;
}

inline real_t makeNaN() 
{
    if(std::numeric_limits<real_t>::has_quiet_NaN)
		return std::numeric_limits<real_t>::quiet_NaN();
	else// (std::numeric_limits<real_t>::has_signaling_NaN)
		return std::numeric_limits<real_t>::signaling_NaN();
}

/*********************************************************************************************
	adouble
**********************************************************************************************/
adouble::adouble() 
{
	m_dValue = 0.0;
    m_dDeriv = 0.0;
}

adouble::adouble(const real_t value)
{
	m_dValue = value;
    m_dDeriv = 0.0;
}

adouble::adouble(const real_t value, real_t deriv)
{
	m_dValue = value;
    m_dDeriv = deriv;
}

adouble::adouble(const adouble& a)
{
    m_dValue = a.m_dValue;
    m_dDeriv = a.m_dDeriv;
}

adouble::~adouble()
{
}

const adouble adouble::operator - () const 
{
    adouble tmp;

    tmp.m_dValue = -m_dValue;
    tmp.m_dDeriv = -m_dDeriv;
    return tmp;
}

const adouble adouble::operator + () const 
{
	adouble tmp;

    tmp.m_dValue = m_dValue;
    tmp.m_dDeriv = m_dDeriv;
	return tmp;
}

const adouble operator +(const adouble& a, const real_t v) 
{
    adouble tmp;

    tmp.m_dValue = a.m_dValue + v;
    tmp.m_dDeriv = a.m_dDeriv;
	return tmp;
}

const adouble adouble::operator + (const adouble& a) const 
{
    adouble tmp;

    tmp.m_dValue = m_dValue + a.m_dValue;
    tmp.m_dDeriv = m_dDeriv + a.m_dDeriv;
    return tmp;
}

const adouble operator +(const real_t v, const adouble& a) 
{
    adouble tmp;

    tmp.m_dValue = v + a.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv;
	return tmp;
}

const adouble operator -(const adouble& a, const real_t v) 
{
    adouble tmp;

    tmp.m_dValue = a.m_dValue - v;
    tmp.m_dDeriv = a.m_dDeriv;
    return tmp;
}

const adouble adouble::operator - (const adouble& a) const 
{
    adouble tmp;

	tmp.m_dValue = m_dValue - a.m_dValue;
    tmp.m_dDeriv = m_dDeriv - a.m_dDeriv;
    return tmp;
}

const adouble operator - (const real_t v, const adouble& a) 
{
    adouble tmp;

	tmp.m_dValue = v - a.m_dValue;
    tmp.m_dDeriv =   - a.m_dDeriv;
    return tmp;
}

const adouble operator *(const adouble& a, const real_t v) 
{
    adouble tmp;

    tmp.m_dValue = a.m_dValue * v;
    tmp.m_dDeriv = a.m_dDeriv * v;
    return tmp;
}
  
const adouble adouble::operator * (const adouble& a) const 
{
    adouble tmp;

	tmp.m_dValue = m_dValue * a.m_dValue;
    tmp.m_dDeriv = m_dDeriv * a.m_dValue + m_dValue * a.m_dDeriv;
    return tmp;
}

const adouble operator * (const real_t v, const adouble& a) 
{
    adouble tmp;

    tmp.m_dValue = v * a.m_dValue;
    tmp.m_dDeriv = v * a.m_dDeriv;
    return tmp;
}

const adouble operator /(const adouble& a, const real_t v) 
{
    adouble tmp;

    tmp.m_dValue = a.m_dValue / v;
    tmp.m_dDeriv = a.m_dDeriv / v;
    return tmp;
}

const adouble adouble::operator / (const adouble& a) const 
{
    adouble tmp;

    tmp.m_dValue = m_dValue / a.m_dValue;
    tmp.m_dDeriv = (m_dDeriv * a.m_dValue - m_dValue * a.m_dDeriv) / (a.m_dValue * a.m_dValue);
    return tmp;
}

const adouble operator / (const real_t v, const adouble& a) 
{
    adouble tmp;

    tmp.m_dValue = v / a.m_dValue;
    tmp.m_dDeriv = (-v * a.m_dDeriv) / (a.m_dValue * a.m_dValue);
    return tmp;
}

const adouble pow(const adouble &a, real_t v) 
{
    adouble tmp;

	tmp.m_dValue = ::pow(a.m_dValue, v);
    real_t tmp2 = v * ::pow(a.m_dValue, v-1);
    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

const adouble pow(const adouble &a, const adouble &b) 
{
    adouble tmp;

	if(b.m_dDeriv == 0)
	{
	// In order to avoid logarithm of a negative number here I work as if I have: pow(adouble, const)!!
	// It is useful if I want (expression)^2 for instance, so that expression MAY take negative numbers
		return pow(a, b.m_dValue);
	}
	else if(a.m_dValue <= 0)
	{
        throw std::runtime_error("Power function called for a negative base");
    }
	
    tmp.m_dValue = ::pow(a.m_dValue, b.m_dValue);
    real_t tmp2 = b.m_dValue * ::pow(a.m_dValue, b.m_dValue-1);
    real_t tmp3 = ::log(a.m_dValue) * tmp.m_dValue;
    tmp.m_dDeriv = tmp2 * a.m_dDeriv + tmp3 * b.m_dDeriv;
    return tmp;
}

const adouble pow(real_t v, const adouble &a) 
{
    adouble tmp;
	
// ACHTUNG, ACHTUNG!!!	
// log(number) = NaN if the number is <= 0
	if(v <= 0)
	{
        throw std::runtime_error("Power function called for a negative base");
	}

	tmp.m_dValue = ::pow(v, a.m_dValue);
    real_t tmp2 = tmp.m_dValue * ::log(v);    
    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

const adouble log10(const adouble &a) 
{
    adouble tmp;
	
// ACHTUNG, ACHTUNG!!!	
// log10(number) = NaN if the number is <= 0
	if(a.m_dValue <= 0)
	{
        throw std::runtime_error("Log10 function called for a negative base");
	}

    tmp.m_dValue = ::log10(a.m_dValue);
    real_t tmp2 = ::log((real_t)10) * a.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

const adouble exp(const adouble &a) 
{
    adouble tmp;

    tmp.m_dValue = ::exp(a.m_dValue);
    tmp.m_dDeriv = tmp.m_dValue * a.m_dDeriv;
    return tmp;
}

const adouble log(const adouble &a) 
{
    adouble tmp;
	
// ACHTUNG, ACHTUNG!!!	
// log(number) = NaN if the number is <= 0
	if(a.m_dValue <= 0)
	{
        throw std::runtime_error("Log function called for a negative base");
	}

    tmp.m_dValue = ::log(a.m_dValue);
	tmp.m_dDeriv = a.m_dDeriv / a.m_dValue;
	return tmp;
	
// The original code:
//    if (a.m_dValue > 0 || a.m_dValue == 0 && a.m_dDeriv >= 0)
//		tmp.m_dDeriv = a.m_dDeriv / a.m_dValue;
//    else 
//		tmp.m_dDeriv = makeNaN();
}

const adouble sqrt(const adouble &a) 
{
    adouble tmp;
	
// ACHTUNG, ACHTUNG!!!	
// sqrt(number) = NaN if the number is < 0
	if(a.m_dValue > 0)
	{
		tmp.m_dValue = ::sqrt(a.m_dValue);
		tmp.m_dDeriv = a.m_dDeriv / tmp.m_dValue / 2;
	}
	else if(a.m_dValue == 0)
	{
		tmp.m_dValue = 0; // sqrt(0) = 0
		tmp.m_dDeriv = 0; // number/0 = 0 (Is it??) 
	}
	else
	{
        throw std::runtime_error("Sqrt function called with a negative argument");
	}
	return tmp;

// The original code:
//	if(a.m_dValue > 0 || a.m_dValue == 0 && a.m_dDeriv >= 0)  // ????????????????????
//		tmp.m_dDeriv = a.m_dDeriv / tmp.m_dValue / 2;
//	else 
//		tmp.m_dDeriv = makeNaN();
}

const adouble abs(const adouble &a) 
{
    adouble tmp;

    tmp.m_dValue = ::fabs(a.m_dValue);
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

const adouble sin(const adouble &a) 
{
    adouble tmp;
    real_t tmp2;

    tmp.m_dValue = ::sin(a.m_dValue);
    tmp2         = ::cos(a.m_dValue);
    
    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

const adouble cos(const adouble &a) 
{
    adouble tmp;
    real_t tmp2;

    tmp.m_dValue = ::cos(a.m_dValue);
    tmp2         = -::sin(a.m_dValue);
    
    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

const adouble tan(const adouble& a) 
{
    adouble tmp;
    real_t tmp2;

    tmp.m_dValue = ::tan(a.m_dValue);
    tmp2         = ::cos(a.m_dValue);
    tmp2 *= tmp2;
    
    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

const adouble asin(const adouble &a) 
{
    adouble tmp;

    tmp.m_dValue = ::asin(a.m_dValue);
    real_t tmp2  = ::sqrt(1 - a.m_dValue * a.m_dValue);
    
    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

const adouble acos(const adouble &a) 
{
    adouble tmp;

    tmp.m_dValue =  ::acos(a.m_dValue);
    real_t tmp2  = -::sqrt(1-a.m_dValue*a.m_dValue);
    
    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

const adouble atan(const adouble &a) 
{
    adouble tmp;

    tmp.m_dValue = ::atan(a.m_dValue);
    real_t tmp2  = 1 + a.m_dValue * a.m_dValue;
    tmp2 = 1 / tmp2;
    if (tmp2 != 0)
        tmp.m_dDeriv = a.m_dDeriv * tmp2;
    else
        tmp.m_dDeriv = 0.0;
    return tmp;
}

const adouble sinh(const adouble& a) 
{
    throw std::runtime_error("sinh function is not implemented");
    return adouble();
    
//    if (a.getValue() < 0.0) 
//	{
//        adouble temp = exp(a);
//        return  0.5*(temp - 1.0/temp);
//    }
//	else 
//	{
//        adouble temp = exp(-a);
//        return 0.5*(1.0/temp - temp);
//    }
}

const adouble cosh(const adouble& a) 
{
    throw std::runtime_error("cosh function is not implemented");
	return adouble();

//    adouble temp = (a.getValue() < 0.0) ? exp(a) : exp(-a);
//    return 0.5*(temp + 1.0/temp);
}

const adouble tanh(const adouble& a)
{
    throw std::runtime_error("tanh function is not implemented");
	return adouble();

//    if (a.getValue() < 0.0) 
//	{
//        adouble temp = exp(2.0*a);
//        return (temp - 1.0)/(temp + 1.0);
//    }
//	else
//	{
//        adouble temp = exp((-2.0)*a);
//        return (1.0 - temp)/(temp + 1.0);
//    }
}

const adouble asinh(const adouble &a)
{
    throw std::runtime_error("asinh function is not implemented");
	return adouble();
}

const adouble acosh(const adouble &a)
{
    throw std::runtime_error("acosh function is not implemented");
	return adouble();
}

const adouble atanh(const adouble &a)
{
    throw std::runtime_error("atanh function is not implemented");
	return adouble();
}

const adouble atan2(const adouble &a, const adouble &b)
{
    throw std::runtime_error("atan2 function is not implemented");
	return adouble();
}

// ceil is non-differentiable: should I remove it?
const adouble ceil(const adouble &a) 
{
    adouble tmp;

	tmp.m_dValue = ::ceil(a.m_dValue);    
    tmp.m_dDeriv = 0.0;
    return tmp;
}

// floor is non-differentiable: should I remove it?
const adouble floor(const adouble &a) 
{
    adouble tmp;

	tmp.m_dValue = ::floor(a.m_dValue);
    tmp.m_dDeriv = 0.0;
    return tmp;
}

const adouble max(const adouble &a, const adouble &b) 
{
    adouble tmp;

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

const adouble max(real_t v, const adouble &a) 
{
    adouble tmp;

	real_t tmp2 = v - a.m_dValue;
	if(tmp2 < 0)
	{
		tmp.m_dValue = a.m_dValue;
		tmp.m_dDeriv = a.m_dDeriv;
	} 
	else 
	{
		tmp.m_dValue = v;
		if(tmp2 > 0) 
		{
			tmp.m_dDeriv = 0.0;
		} 
		else 
		{
			if(a.m_dDeriv > 0)
				tmp.m_dDeriv = a.m_dDeriv;
			else 
				tmp.m_dDeriv = 0.0;
		}
	}
	return tmp;
}

const adouble max(const adouble &a, real_t v) 
{
    adouble tmp;

	real_t tmp2 = a.m_dValue - v;
	if(tmp2 < 0) 
	{
		tmp.m_dValue = v;
		tmp.m_dDeriv = 0.0;
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
			if(a.m_dDeriv > 0) 
				tmp.m_dDeriv = a.m_dDeriv;
			else 
				tmp.m_dDeriv = 0.0;
		}
	}
	return tmp;
}

const adouble min(const adouble &a, const adouble &b) 
{
    adouble tmp;

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

const adouble min(real_t v, const adouble &a) 
{
    adouble tmp;

	real_t tmp2 = v - a.m_dValue;
	if(tmp2 < 0) 
	{
		tmp.m_dValue = v;
		tmp.m_dDeriv = 0.0;
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
			if(a.m_dDeriv < 0) 
				tmp.m_dDeriv = a.m_dDeriv;
			else 
				tmp.m_dDeriv = 0.0;
		}
	}
	return tmp;
}

const adouble min(const adouble &a, real_t v) 
{
    adouble tmp;

	real_t tmp2 = a.m_dValue - v;
	if(tmp2 < 0) 
	{
		tmp.m_dValue = a.m_dValue;
		tmp.m_dDeriv = a.m_dDeriv;
	} 
	else 
	{
		tmp.m_dValue = v;
		if(tmp2 > 0) 
		{
			tmp.m_dDeriv = 0.0;
		} 
		else 
		{
			if(a.m_dDeriv < 0) 
				tmp.m_dDeriv = a.m_dDeriv;
			else 
				tmp.m_dDeriv = 0.0;
		}
	}
	return tmp;
}

void adouble::operator =(const real_t v) 
{
    m_dValue = v;
    m_dDeriv = 0.0;
}

void adouble::operator =(const adouble& a) 
{
    m_dValue = a.m_dValue;
    m_dDeriv = a.m_dDeriv;
}

bool adouble::operator != (const adouble &a) const 
{
    return (m_dValue != a.m_dValue);
}

bool adouble::operator != (const real_t v) const 
{
    return (m_dValue != v);
}

bool operator != (const real_t v, const adouble &a) 
{
    return (v != a.m_dValue);
}

bool adouble::operator == (const adouble &a) const 
{
    return (m_dValue == a.m_dValue);
}

bool adouble::operator == (const real_t v) const 
{
    return (m_dValue == v);
}

bool operator == (const real_t v, const adouble &a) 
{
    return (v == a.m_dValue);
}

bool adouble::operator <= (const adouble &a) const 
{
    return (m_dValue <= a.m_dValue);
}

bool adouble::operator <= (const real_t v) const 
{
    return (m_dValue <= v);
}

bool operator <= (const real_t v, const adouble &a) 
{
    return (v <= a.m_dValue);
}

bool adouble::operator >= (const adouble &a) const 
{
    return (m_dValue >= a.m_dValue);
}

bool adouble::operator >= (const real_t v) const 
{
    return (m_dValue >= v);
}

bool operator >= (const real_t v, const adouble &a) 
{
    return (v >= a.m_dValue);
}

bool adouble::operator > (const adouble &a) const 
{
    return (m_dValue > a.m_dValue);
}

bool adouble::operator > (const real_t v) const 
{
    return (m_dValue > v);
}

bool operator > (const real_t v, const adouble &a) 
{
    return (v > a.m_dValue);
}

bool adouble::operator <  (const adouble &a) const 
{
    return (m_dValue < a.m_dValue);
}

bool adouble::operator <  (const real_t v) const 
{
    return (m_dValue < v);
}

bool operator <  (const real_t v, const adouble &a)
{
    return (v < a.m_dValue);
}

std::ostream& operator<<(std::ostream& out, const adouble& a)
{
    return out << "(" << a.getValue() << ", " << a.getDerivative() << ")";
}

