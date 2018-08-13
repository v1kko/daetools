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
#ifndef CS_NUMBER_H
#define CS_NUMBER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "cs_nodes.h"

namespace cs
{
class OPENCS_MODELS_API csNumber_t
{
public:
    csNumber_t();
    csNumber_t(real_t value);
    virtual ~csNumber_t(){}

public:
    csNodePtr node;
};

/* Unary operators. */
OPENCS_MODELS_API csNumber_t operator -(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t operator +(const csNumber_t& n);

/* Binary operators. */
OPENCS_MODELS_API csNumber_t operator +(const csNumber_t& l, const csNumber_t& r);
OPENCS_MODELS_API csNumber_t operator -(const csNumber_t& l, const csNumber_t& r);
OPENCS_MODELS_API csNumber_t operator *(const csNumber_t& l, const csNumber_t& r);
OPENCS_MODELS_API csNumber_t operator /(const csNumber_t& l, const csNumber_t& r);

/* Unary functions. */
OPENCS_MODELS_API csNumber_t exp(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t log(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t log10(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t sqrt(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t sin(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t cos(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t tan(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t asin(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t acos(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t atan(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t sinh(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t cosh(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t tanh(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t asinh(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t acosh(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t atanh(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t ceil(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t floor(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t fabs(const csNumber_t& n);
OPENCS_MODELS_API csNumber_t erf(const csNumber_t& n);

/* Binary functions. */
OPENCS_MODELS_API csNumber_t atan2(const csNumber_t& l, const csNumber_t& r);
OPENCS_MODELS_API csNumber_t pow(const csNumber_t& l, const csNumber_t& r);
OPENCS_MODELS_API csNumber_t min(const csNumber_t& l, const csNumber_t& r);
OPENCS_MODELS_API csNumber_t max(const csNumber_t& l, const csNumber_t& r);
}

#endif
