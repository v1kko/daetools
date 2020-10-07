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
#include "cs_number.h"

namespace cs
{
csNumber_t::csNumber_t()
{
}

csNumber_t::csNumber_t(real_t value)
{
    node.reset(new csConstantNode(value));
}


/* Unary operators. */
csNumber_t operator -(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eSign, n.node));
    return tmp;
}

csNumber_t operator +(const csNumber_t& n)
{
    return n;
}

/* Binary operators. */
csNumber_t operator +(const csNumber_t& l, const csNumber_t& r)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csBinaryNode(ePlus, l.node, r.node));
    return tmp;
}

csNumber_t operator -(const csNumber_t& l, const csNumber_t& r)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csBinaryNode(eMinus, l.node, r.node));
    return tmp;
}

csNumber_t operator *(const csNumber_t& l, const csNumber_t& r)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csBinaryNode(eMulti, l.node, r.node));
    return tmp;
}

csNumber_t operator /(const csNumber_t& l, const csNumber_t& r)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csBinaryNode(eDivide, l.node, r.node));
    return tmp;
}

/* Unary functions. */
csNumber_t exp(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eExp, n.node));
    return tmp;
}

csNumber_t log(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eLn, n.node));
    return tmp;
}

csNumber_t log10(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eLog, n.node));
    return tmp;
}

csNumber_t sqrt(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eSqrt, n.node));
    return tmp;
}

csNumber_t sin(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eSin, n.node));
    return tmp;
}

csNumber_t cos(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eCos, n.node));
    return tmp;
}

csNumber_t tan(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eTan, n.node));
    return tmp;
}

csNumber_t asin(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eArcSin, n.node));
    return tmp;
}

csNumber_t acos(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eArcCos, n.node));
    return tmp;
}

csNumber_t atan(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eArcTan, n.node));
    return tmp;
}

csNumber_t sinh(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eSinh, n.node));
    return tmp;
}

csNumber_t cosh(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eCosh, n.node));
    return tmp;
}

csNumber_t tanh(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eTanh, n.node));
    return tmp;
}

csNumber_t asinh(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eArcSinh, n.node));
    return tmp;
}

csNumber_t acosh(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eArcCosh, n.node));
    return tmp;
}

csNumber_t atanh(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eArcTanh, n.node));
    return tmp;
}

csNumber_t ceil(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eCeil, n.node));
    return tmp;
}

csNumber_t floor(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eFloor, n.node));
    return tmp;
}

csNumber_t fabs(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eAbs, n.node));
    return tmp;
}

csNumber_t erf(const csNumber_t& n)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csUnaryNode(eErf, n.node));
    return tmp;
}

/* Binary functions. */
csNumber_t atan2(const csNumber_t& l, const csNumber_t &r)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csBinaryNode(eArcTan2, l.node, r.node));
    return tmp;
}

csNumber_t pow(const csNumber_t& l, const csNumber_t &r)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csBinaryNode(ePower, l.node, r.node));
    return tmp;
}

csNumber_t min(const csNumber_t& l, const csNumber_t &r)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csBinaryNode(eMin, l.node, r.node));
    return tmp;
}

csNumber_t max(const csNumber_t& l, const csNumber_t &r)
{
    csNumber_t tmp;
    tmp.node = csNodePtr(new csBinaryNode(eMax, l.node, r.node));
    return tmp;
}

}
