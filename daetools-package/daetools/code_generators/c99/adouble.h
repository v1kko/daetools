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
#if !defined(ADOLC_ADOUBLE_H)
#define ADOLC_ADOUBLE_H

#include "typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct  
{
    real_t m_dValue;
    real_t m_dDeriv;
} adouble;

#define NO_INLINE __attribute__((noinline))

adouble _adouble_(real_t value, real_t derivative) NO_INLINE;

adouble _sign_(adouble a) NO_INLINE;

adouble _plus_(const adouble a, const adouble b) NO_INLINE;
adouble _minus_(const adouble a, const adouble b) NO_INLINE;
adouble _multi_(const adouble a, const adouble b) NO_INLINE;
adouble _divide_(const adouble a, const adouble b) NO_INLINE;

adouble _exp_(const adouble a) NO_INLINE;
adouble _log_(const adouble a) NO_INLINE;
adouble _log10_(const adouble a) NO_INLINE;
adouble _sqrt_(const adouble a) NO_INLINE;
adouble _sin_(const adouble a) NO_INLINE;
adouble _cos_(const adouble a) NO_INLINE;
adouble _tan_(const adouble a) NO_INLINE;
adouble _asin_(const adouble a) NO_INLINE;
adouble _acos_(const adouble a) NO_INLINE;
adouble _atan_(const adouble a) NO_INLINE;

adouble _sinh_(const adouble a) NO_INLINE;
adouble _cosh_(const adouble a) NO_INLINE;
adouble _tanh_(const adouble a) NO_INLINE;
adouble _asinh_(const adouble a) NO_INLINE;
adouble _acosh_(const adouble a) NO_INLINE;
adouble _atanh_(const adouble a) NO_INLINE;
adouble _erf_(const adouble a) NO_INLINE;

adouble _pow_(const adouble a, const adouble b) NO_INLINE;

/* ceil/floor are non-differentiable: should I remove them? */
adouble _ceil_(const adouble a) NO_INLINE;
adouble _floor_(const adouble a) NO_INLINE;

adouble _abs_(const adouble a) NO_INLINE;
adouble _max_(const adouble a, const adouble b) NO_INLINE;
adouble _min_(const adouble a, const adouble b) NO_INLINE;
adouble _atan2_(const adouble a, const adouble b) NO_INLINE;

bool _neq_(const adouble a, const adouble b) NO_INLINE;
bool _eq_(const adouble a, const adouble b) NO_INLINE;
bool _lteq_(const adouble a, const adouble b) NO_INLINE;
bool _gteq_(const adouble a, const adouble b) NO_INLINE;
bool _gt_(const adouble a, const adouble b) NO_INLINE;
bool _lt_(const adouble a, const adouble b) NO_INLINE;

real_t _getValue_(const adouble* a) NO_INLINE;
void   _setValue_(adouble* a, const real_t v) NO_INLINE; 
real_t _getDerivative_(const adouble* a) NO_INLINE;
void   _setDerivative_(adouble* a, real_t v) NO_INLINE; 

real_t _makeNaN_() NO_INLINE;

#ifdef __cplusplus
}
#endif

#endif
