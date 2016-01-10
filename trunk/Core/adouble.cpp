#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include <limits>

namespace dae
{
namespace core
{
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
adouble::adouble(real_t value/* = 0.0*/,
                 real_t derivative/* = 0.0*/,
                 bool gatherInfo/* = false*/,
                 adNode* node_/* = NULL*/)
{
    m_dValue      = value;
    m_dDeriv      = derivative;
    m_bGatherInfo = gatherInfo;

    if(gatherInfo)
        node = adNodePtr( (node_ ? node_ : new adConstantNode(value)) );
}

adouble::adouble(const adouble& a)
{
    m_bGatherInfo = false;
    if(a.getGatherInfo())
    {
        m_bGatherInfo = true;
        node  = adNodePtr( (a.node ? a.node->Clone() : new adConstantNode(a.m_dValue)) );
    }
    m_dValue = a.m_dValue;
    m_dDeriv = a.m_dDeriv;
}

adouble::~adouble()
{
}

const adouble adouble::operator - () const
{
    adouble tmp;
    if(m_bGatherInfo)
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode(eSign, CLONE_NODE(node, m_dValue)));
        return tmp;
    }
    tmp.m_dValue = -m_dValue;
    tmp.m_dDeriv = -m_dDeriv;
    return tmp;
}

const adouble adouble::operator + () const
{
    adouble tmp;
    if(m_bGatherInfo)
    {
        tmp.m_bGatherInfo = true;
        tmp.node = CLONE_NODE(node, m_dValue);
        return tmp;
    }
    tmp.m_dValue = m_dValue;
    tmp.m_dDeriv = m_dDeriv;
    return tmp;
}

const adouble operator +(const adouble& a, const real_t v)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(ePlus,
                                              CLONE_NODE(a.node, a.m_dValue),
                                              adNodePtr(new adConstantNode(v, UNITS(a.node))) ));
        return tmp;
    }

    tmp.m_dValue = a.m_dValue + v;
    tmp.m_dDeriv = a.m_dDeriv;
    return tmp;
}

const adouble adouble::operator + (const adouble& a) const
{
    adouble tmp;
    if(m_bGatherInfo || a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(ePlus,
                                              CLONE_NODE(node, m_dValue),
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = m_dValue + a.m_dValue;
    tmp.m_dDeriv = m_dDeriv + a.m_dDeriv;
    return tmp;
}

adouble& adouble::operator +=(const adouble& a)
{
    adouble tmp = (*this) + a;
    *this = tmp;
    return *this;
}

adouble& adouble::operator -=(const adouble& a)
{
    adouble tmp = (*this) - a;
    *this = tmp;
    return *this;
}

adouble& adouble::operator *=(const adouble& a)
{
    adouble tmp = (*this) * a;
    *this = tmp;
    return *this;
}

adouble& adouble::operator /=(const adouble& a)
{
    adouble tmp = (*this) / a;
    *this = tmp;
    return *this;
}

adouble& adouble::operator +=(const real_t v)
{
    adouble tmp = (*this) + v;
    *this = tmp;
    return *this;
}

adouble& adouble::operator -=(const real_t v)
{
    adouble tmp = (*this) - v;
    *this = tmp;
    return *this;
}

adouble& adouble::operator *=(const real_t v)
{
    adouble tmp = (*this) * v;
    *this = tmp;
    return *this;
}

adouble& adouble::operator /=(const real_t v)
{
    adouble tmp = (*this) / v;
    *this = tmp;
    return *this;
}

const adouble operator +(const real_t v, const adouble& a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(ePlus,
                                              adNodePtr(new adConstantNode(v, UNITS(a.node))),
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = v + a.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv;
    return tmp;
}

const adouble operator -(const adouble& a, const real_t v)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMinus,
                                              CLONE_NODE(a.node, a.m_dValue),
                                              adNodePtr(new adConstantNode(v, UNITS(a.node))) ));
        return tmp;
    }
    tmp.m_dValue = a.m_dValue - v;
    tmp.m_dDeriv = a.m_dDeriv;
    return tmp;
}

const adouble adouble::operator - (const adouble& a) const
{
    adouble tmp;
    if(m_bGatherInfo || a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMinus,
                                              CLONE_NODE(node, m_dValue),
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = m_dValue - a.m_dValue;
    tmp.m_dDeriv = m_dDeriv - a.m_dDeriv;
    return tmp;
}

const adouble operator - (const real_t v, const adouble& a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMinus,
                                              adNodePtr(new adConstantNode(v, UNITS(a.node))),
                                              CLONE_NODE(a.node, a.m_dValue)));
        return tmp;
    }

    tmp.m_dValue = v - a.m_dValue;
    tmp.m_dDeriv =   - a.m_dDeriv;
    return tmp;
}

const adouble operator *(const adouble& a, const real_t v)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMulti,
                                              CLONE_NODE(a.node, a.m_dValue),
                                              adNodePtr(new adConstantNode(v, UNITS(a.node))) ));
        return tmp;
    }

    tmp.m_dValue = a.m_dValue * v;
    tmp.m_dDeriv = a.m_dDeriv * v;
    return tmp;
}

const adouble adouble::operator * (const adouble& a) const
{
    adouble tmp;
    if(m_bGatherInfo || a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode( eMulti,
                                               CLONE_NODE(node,   m_dValue),
                                               CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = m_dValue * a.m_dValue;
    tmp.m_dDeriv = m_dDeriv * a.m_dValue + m_dValue * a.m_dDeriv;
    return tmp;
}

const adouble operator * (const real_t v, const adouble& a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode( eMulti,
                                               adNodePtr(new adConstantNode(v, UNITS(a.node))),
                                               CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
    tmp.m_dValue = v * a.m_dValue;
    tmp.m_dDeriv = v * a.m_dDeriv;
    return tmp;
}

const adouble operator /(const adouble& a, const real_t v)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode( eDivide,
                                               CLONE_NODE(a.node, a.m_dValue),
                                               adNodePtr(new adConstantNode(v, UNITS(a.node))) ));
        return tmp;
    }
    tmp.m_dValue = a.m_dValue / v;
    tmp.m_dDeriv = a.m_dDeriv / v;
    return tmp;
}

const adouble adouble::operator / (const adouble& a) const
{
    adouble tmp;
    if(m_bGatherInfo || a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode( eDivide,
                                               CLONE_NODE(node,   m_dValue),
                                               CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
    tmp.m_dValue = m_dValue / a.m_dValue;
    tmp.m_dDeriv = (m_dDeriv * a.m_dValue - m_dValue * a.m_dDeriv) / (a.m_dValue * a.m_dValue);
    return tmp;
}

const adouble operator / (const real_t v, const adouble& a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode( eDivide,
                                               adNodePtr(new adConstantNode(v, UNITS(a.node))),
                                               CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
    tmp.m_dValue = v / a.m_dValue;
    tmp.m_dDeriv = (-v * a.m_dDeriv) / (a.m_dValue * a.m_dValue);
    return tmp;
}

const adouble pow(const adouble &a, real_t v)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode( ePower,
                                               CLONE_NODE(a.node, a.m_dValue),
                                               adNodePtr(new adConstantNode(v)) ));
        return tmp;
    }

    tmp.m_dValue = ::pow(a.m_dValue, v);
    real_t tmp2 = v * ::pow(a.m_dValue, v-1);
    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

const adouble pow(const adouble &a, const adouble &b)
{
    adouble tmp;
    if(a.m_bGatherInfo || b.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode( ePower,
                                               CLONE_NODE(a.node, a.m_dValue),
                                               CLONE_NODE(b.node, b.m_dValue) ));
        return tmp;
    }

    if(b.m_dDeriv == 0)
    {
    // In order to avoid logarithm of a negative number here I work as if I have: pow(adouble, const)!!
    // It is useful if I want (expression)^2 for instance, so that expression MAY take negative numbers
        return pow(a, b.m_dValue);
    }
    else if(a.m_dValue <= 0)
    {
        daeDeclareException(exRuntimeCheck);
        e << "Power function called for a negative base: "
          << toStringFormatted<real_t>(a.m_dValue, -1, 5, true)
          << " ^ "
          << toStringFormatted<real_t>(b.m_dValue, -1, 5, true);
        throw e;
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
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode( ePower,
                                               adNodePtr(new adConstantNode(v)),
                                               CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

// ACHTUNG, ACHTUNG!!!
// log(number) = NaN if the number is <= 0
    if(v <= 0)
    {
        daeDeclareException(exRuntimeCheck);
        e << "Power function called for a negative base: "
          << toStringFormatted<real_t>(v, -1, 5, true)
          << " ^ "
          << toStringFormatted<real_t>(a.m_dValue, -1, 5, true);
        throw e;
    }

    tmp.m_dValue = ::pow(v, a.m_dValue);
    real_t tmp2 = tmp.m_dValue * ::log(v);
    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

const adouble log10(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eLog,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

// ACHTUNG, ACHTUNG!!!
// log10(number) = NaN if the number is <= 0
    if(a.m_dValue <= 0)
    {
        daeDeclareException(exRuntimeCheck);
        e << "Log10 function called for a negative base: log10("
          << toStringFormatted<real_t>(a.m_dValue, -1, 5, true)
          << ")";
        throw e;
    }

    tmp.m_dValue = ::log10(a.m_dValue);
    real_t tmp2 = ::log((real_t)10) * a.m_dValue;
    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

const adouble exp(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eExp,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
    tmp.m_dValue = ::exp(a.m_dValue);
    tmp.m_dDeriv = tmp.m_dValue * a.m_dDeriv;
    return tmp;
}

const adouble log(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eLn,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

// ACHTUNG, ACHTUNG!!!
// log(number) = NaN if the number is <= 0
    if(a.m_dValue <= 0)
    {
        daeDeclareException(exRuntimeCheck);
        e << "Log function called for a negative base: log("
          << toStringFormatted<real_t>(a.m_dValue, -1, 5, true)
          << ")";
        throw e;
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
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eSqrt,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

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
        daeDeclareException(exRuntimeCheck);
        e << "Sqrt function called with a negative argument: sqrt("
          << toStringFormatted<real_t>(a.m_dValue, -1, 15, true)
          << ")";
        throw e;
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
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eAbs,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
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
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eSin,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
    tmp.m_dValue = ::sin(a.m_dValue);
    tmp2         = ::cos(a.m_dValue);

    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

const adouble cos(const adouble &a)
{
    adouble tmp;
    real_t tmp2;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eCos,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
    tmp.m_dValue = ::cos(a.m_dValue);
    tmp2         = -::sin(a.m_dValue);

    tmp.m_dDeriv = tmp2 * a.m_dDeriv;
    return tmp;
}

const adouble tan(const adouble& a)
{
    adouble tmp;
    real_t tmp2;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eTan,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
    tmp.m_dValue = ::tan(a.m_dValue);
    tmp2         = ::cos(a.m_dValue);
    tmp2 *= tmp2;

    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

const adouble asin(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eArcSin,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
    tmp.m_dValue = ::asin(a.m_dValue);
    real_t tmp2  = ::sqrt(1 - a.m_dValue * a.m_dValue);

    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

const adouble acos(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eArcCos,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
    tmp.m_dValue =  ::acos(a.m_dValue);
    real_t tmp2  = -::sqrt(1-a.m_dValue*a.m_dValue);

    tmp.m_dDeriv = a.m_dDeriv / tmp2;
    return tmp;
}

const adouble atan(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eArcTan,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }
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
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eSinh,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    if(a.m_dValue < 0.0)
    {
        tmp = exp(a);
        return 0.5*(tmp - 1.0/tmp);
    }
    else
    {
        tmp = exp(-a);
        return 0.5*(1.0/tmp - tmp);
    }
}

const adouble cosh(const adouble& a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eCosh,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp = (a.m_dValue < 0.0) ? exp(a) : exp(-a);
    return 0.5*(tmp + 1.0/tmp);
}

const adouble tanh(const adouble& a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eTanh,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    if(a.m_dValue < 0.0)
    {
        tmp = exp(2.0*a);
        return (tmp - 1.0)/(tmp + 1.0);
    }
    else
    {
        tmp = exp((-2.0)*a);
        return (1.0 - tmp)/(tmp + 1.0);
    }
}

const adouble asinh(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eArcSinh,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = ::asinh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / ::sqrt(a.m_dValue*a.m_dValue + 1);

    return tmp;
}

const adouble acosh(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eArcCosh,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = ::acosh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / ::sqrt(a.m_dValue*a.m_dValue - 1);

    return tmp;
}

const adouble atanh(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eArcTanh,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = ::atanh(a.m_dValue);
    tmp.m_dDeriv = a.m_dDeriv / (1 - a.m_dValue*a.m_dValue);

    return tmp;
}

const adouble atan2(const adouble &a, const adouble &b)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode( eArcTan2,
                                               CLONE_NODE(a.node, a.m_dValue),
                                               CLONE_NODE(b.node, b.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = ::atan2(a.m_dValue, b.m_dValue);
    double tmp2 = a.m_dValue*a.m_dValue;
    double tmp3 = b.m_dValue*b.m_dValue;
    double tmp4 = tmp3 / (tmp2 + tmp3);
    if(tmp4 != 0)
        tmp.m_dDeriv = (a.m_dDeriv*b.m_dValue - a.m_dValue*b.m_dDeriv) / tmp3*tmp4;
    else
        tmp.m_dDeriv = 0.0;

    return tmp;
}

const adouble erf(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eErf,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = ::erf(a.m_dValue);
    const double pi = 3.14159265358979323846;
    tmp.m_dDeriv = a.m_dDeriv * (2.0 / ::sqrt(pi)) * ::exp(-a.m_dValue*a.m_dValue);

    return tmp;
}

// ceil is non-differentiable: should I remove it?
const adouble ceil(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eCeil,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = ::ceil(a.m_dValue);
    tmp.m_dDeriv = 0.0;
    return tmp;
}

// floor is non-differentiable: should I remove it?
const adouble floor(const adouble &a)
{
    adouble tmp;
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adUnaryNode( eFloor,
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

    tmp.m_dValue = ::floor(a.m_dValue);
    tmp.m_dDeriv = 0.0;
    return tmp;
}

const adouble max(const adouble &a, const adouble &b)
{
    adouble tmp;
    if(a.getGatherInfo() || b.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMax,
                                              CLONE_NODE(a.node, a.m_dValue),
                                              CLONE_NODE(b.node, b.m_dValue) ));
        return tmp;
    }

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
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMax,
                                              adNodePtr(new adConstantNode(v, UNITS(a.node))),
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

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
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMax,
                                              CLONE_NODE(a.node, a.m_dValue),
                                              adNodePtr(new adConstantNode(v, UNITS(a.node))) ));
        return tmp;
    }

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
    if(a.getGatherInfo() || b.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMin,
                                              CLONE_NODE(a.node, a.m_dValue),
                                              CLONE_NODE(b.node, b.m_dValue) ));
        return tmp;
    }

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
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMin,
                                              adNodePtr(new adConstantNode(v, UNITS(a.node))),
                                              CLONE_NODE(a.node, a.m_dValue) ));
        return tmp;
    }

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
    if(a.getGatherInfo())
    {
        tmp.m_bGatherInfo = true;
        tmp.node = adNodePtr(new adBinaryNode(eMin,
                                              CLONE_NODE(a.node, a.m_dValue),
                                              adNodePtr(new adConstantNode(v, UNITS(a.node))) ));
        return tmp;
    }

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

const adouble Time(void)
{
    adouble tmp;
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adTimeNode());
    return tmp;
}

const adouble InverseTimeStep(void)
{
    adouble tmp;
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adInverseTimeStepNode());
    return tmp;
}

const adouble Constant(const quantity& q)
{
    adouble tmp;
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adConstantNode(q));
    return tmp;
}

const adouble Constant(real_t c)
{
    quantity q(c, unit());
    return Constant(q);
}


/*********************************************************************************************
  d, d2, dt
**********************************************************************************************/
const adouble dt(const adouble& a)
{
    adouble tmp;
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adSetupExpressionDerivativeNode(CLONE_NODE(a.node, a.getValue()) ));
    return tmp;
}

const adouble d(const adouble&                            a,
                daeDomain&                                domain,
                daeeDiscretizationMethod                  eDiscretizationMethod,
                const std::map<std::string, std::string>& mapDiscretizationOptions)
{
    adouble tmp;
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adSetupExpressionPartialDerivativeNode(&domain,
                                                                    1,
                                                                    eDiscretizationMethod,
                                                                    mapDiscretizationOptions,
                                                                    CLONE_NODE(a.node, a.getValue()) ));
    return tmp;
}

const adouble d2(const adouble&                            a,
                 daeDomain&                                domain,
                 daeeDiscretizationMethod                  eDiscretizationMethod,
                 const std::map<std::string, std::string>& mapDiscretizationOptions)
{
    adouble tmp;
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adSetupExpressionPartialDerivativeNode(&domain,
                                                                    2,
                                                                    eDiscretizationMethod,
                                                                    mapDiscretizationOptions,
                                                                    CLONE_NODE(a.node, a.getValue()) ));
    return tmp;
}

adouble& adouble::operator =(const real_t v)
{
    //if(getGatherInfo())
    //{
        if(node)
            node.reset();
    //}
    m_dValue = v;
    m_dDeriv = 0.0;

    return *this;
}

adouble& adouble::operator =(const adouble& a)
{
    if(m_bGatherInfo || a.getGatherInfo())
    {
        m_bGatherInfo = true;
        node = adNodePtr(  (a.node ? a.node->Clone() : new adConstantNode(a.getValue()))  );
    }
    m_dValue = a.m_dValue;
    m_dDeriv = a.m_dDeriv;

    return *this;
}

daeCondition adouble::operator != (const adouble &a) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eNotEQ, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator != (const real_t v) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eNotEQ, v);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition operator != (const real_t v, const adouble &a)
{
    condExpressionNode* expr = new condExpressionNode(v, eNotEQ, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator == (const adouble &a) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eEQ, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator == (const real_t v) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eEQ, v);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition operator == (const real_t v, const adouble &a)
{
    condExpressionNode* expr = new condExpressionNode(v, eEQ, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator <= (const adouble &a) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eLTEQ, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator <= (const real_t v) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eLTEQ, v);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition operator <= (const real_t v, const adouble &a)
{
    condExpressionNode* expr = new condExpressionNode(v, eLTEQ, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator >= (const adouble &a) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eGTEQ, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator >= (const real_t v) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eGTEQ, v);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition operator >= (const real_t v, const adouble &a)
{
    condExpressionNode* expr = new condExpressionNode(v, eGTEQ, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator > (const adouble &a) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eGT, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator > (const real_t v) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eGT, v);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition operator > (const real_t v, const adouble &a)
{
    condExpressionNode* expr = new condExpressionNode(v, eGT, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator <  (const adouble &a) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eLT, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition adouble::operator <  (const real_t v) const
{
    condExpressionNode* expr = new condExpressionNode(*this, eLT, v);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

daeCondition operator <  (const real_t v, const adouble &a)
{
    condExpressionNode* expr = new condExpressionNode(v, eLT, a);
    condNodePtr node(expr);
    daeCondition cond(node);
    return cond;
}

std::ostream& operator<<(std::ostream& out, const adouble& a)
{
    return out << "(" << a.getValue() << ", " << a.getDerivative() << ")";
}

string adouble::NodeAsPlainText(void) const
{
    string strContent;
    daeModelExportContext c;

    c.m_pModel             = NULL;
    c.m_nPythonIndentLevel = 0;
    c.m_bExportDefinition  = true;

    if(node)
        node->Export(strContent, eCDAE, c);
    else
        strContent = toString(getValue());

    return strContent;
}

string adouble::NodeAsLatex(void) const
{
    daeNodeSaveAsContext c;
    c.m_pModel = NULL;

    if(node)
        return node->SaveAsLatex(&c);
    else
        return toString(getValue());

    return string();
}

}
}
