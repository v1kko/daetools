#include "units.h"
#include <algorithm>

namespace units
{
/**************************************************************
		quantity
**************************************************************/
quantity::quantity(void)
{
	_value = 0.0;
}

quantity::quantity(double val, const unit& u)
{
	_units = u;
	_value = val;
}

double quantity::getValue(void) const
{
	return _value;
}

void quantity::setValue(double val)
{
	_value = val;
}

void quantity::setValue(const quantity& other)
{
	_value = other.scaleTo(*this).getValue();
}

unit quantity::getUnits(void) const
{
	return _units;
}

void quantity::setUnits(const unit& u)
{
	_units = u;
}

double quantity::getValueInSIUnits(void) const
{
	return _value * _units.getBaseUnit().multiplier;
}

quantity quantity::scaleTo(const quantity& referrer) const
{
	return scaleTo(referrer.getUnits());
}

quantity quantity::scaleTo(const unit& referrer) const
{
	if(not _units.getBaseUnit().areDimensionsEqual(referrer.getBaseUnit()))
		throw units_error((boost::format("Units not consistent: scale from %1% to %2%") % _units % referrer).str());

	quantity tmp;
	tmp.setUnits(referrer);
	tmp.setValue(_value * _units.getBaseUnit().multiplier / referrer.getBaseUnit().multiplier);
	return tmp;
}

std::string quantity::toString(void) const
{
	return (boost::format("%.10f %s") % _value % _units.toString()).str();
}

std::ostream& operator<<(std::ostream& out, const quantity& q)
{
	out << q.toString();
	return out;
}

bool quantity::operator==(const quantity& other) const
{
	if(_units != other.getUnits())
		throw units_error((boost::format("Units not consistent: %1% == %2%") % _units % other.getUnits()).str());
		
	return (_value == other.scaleTo(*this).getValue());	
}

bool quantity::operator==(double value) const
{
// Here assume that the value is in the same units as this quantity
// We need this for unit-consistency checks
	return (getValueInSIUnits() == value);
}

bool quantity::operator!=(const quantity& other) const
{
	return !(*this == other);
}

bool quantity::operator!=(double value) const
{
	return !(*this == value);
}

quantity quantity::operator+(const quantity& other) const
{
	if(_units != other.getUnits())
		throw units_error((boost::format("Units not consistent: %1% + %2%") % _units % other.getUnits()).str());
	
	quantity tmp(_value + other.scaleTo(*this).getValue(), _units);
	return tmp;	
}

quantity quantity::operator+(double value) const
{
	quantity q(value, unit());
	return (*this + q);
}

quantity quantity::operator+(void) const
{
	quantity tmp(_value, _units);
	return tmp;	
}

quantity quantity::operator-(const quantity& other) const
{
	if(_units != other.getUnits())
		throw units_error((boost::format("Units not consistent: %1% - %2%") % _units % other.getUnits()).str());
	
	quantity tmp(_value - other.scaleTo(*this).getValue(), _units);
	return tmp;	
}

quantity quantity::operator-(double value) const
{
	quantity q(value, unit());
	return (*this - q);
}

quantity quantity::operator-(void) const
{
	quantity tmp(-_value, _units);
	return tmp;	
}

quantity quantity::operator*(const quantity& other) const
{
	quantity tmp;
	tmp.setUnits(_units * other.getUnits());
	tmp.setValue(_value * other.getValue());
	return tmp;	
}

quantity quantity::operator*(const unit& other) const
{
	quantity tmp;
	tmp.setUnits(_units * other);
	tmp.setValue(_value);
	return tmp;	
}

quantity quantity::operator*(double value) const
{
	quantity q(value, unit());
	return (*this * q);
}

quantity quantity::operator/(const quantity& other) const
{
	quantity tmp;
	tmp.setUnits(_units / other.getUnits());
	tmp.setValue(_value / other.getValue());
	return tmp;	
}

quantity quantity::operator/(const unit& other) const
{
	quantity tmp;
	tmp.setUnits(_units / other);
	tmp.setValue(_value);
	return tmp;	
}

quantity quantity::operator/(double value) const
{
	quantity q(value, unit());
	return (*this / q);
}

quantity quantity::operator^(const quantity& other) const
{
	if(other.getUnits() != unit())
		throw units_error((boost::format("Exponent must be dimension-less in: %1% ^ %2%") % _units % other.getUnits()).str());
	return (*this ^ other.getValueInSIUnits());
}

quantity quantity::operator^(double exponent) const
{
	quantity tmp;
	tmp.setUnits(_units ^ exponent);
	tmp.setValue(::pow(_value, exponent));
	return tmp;	
}

bool quantity::operator <=(const quantity& other) const
{
	if(_units != other.getUnits())
		throw units_error((boost::format("Units not consistent: %1% <= %2%") % _units % other.getUnits()).str());
		
	return _value <= other.scaleTo(*this).getValue();	
}

bool quantity::operator <=(double value) const
{
	quantity q(value, unit());
	return (*this <= q);
}

bool quantity::operator >=(const quantity& other) const
{
	if(_units != other.getUnits())
		throw units_error((boost::format("Units not consistent: %1% >= %2%") % _units % other.getUnits()).str());
		
	return _value >= other.scaleTo(*this).getValue();	
}

bool quantity::operator >=(double value) const
{
	quantity q(value, unit());
	return (*this >= q);
}

bool quantity::operator >(const quantity& other) const
{
	if(_units != other.getUnits())
		throw units_error((boost::format("Units not consistent: %1% > %2%") % _units % other.getUnits()).str());
		
	return _value > other.scaleTo(*this).getValue();	
}

bool quantity::operator >(double value) const
{
	quantity q(value, unit());
	return (*this > q);
}

bool quantity::operator <(const quantity& other) const
{
	if(_units != other.getUnits())
		throw units_error((boost::format("Units not consistent: %1% < %2%") % _units % other.getUnits()).str());
		
	return _value < other.scaleTo(*this).getValue();	
}

bool quantity::operator <(double value) const
{
	quantity q(value, unit());
	return (*this < q);
}

quantity operator*(double value, const quantity& self)
{
	return self * value;	
}

quantity operator/(double value, const quantity& self)
{
	return quantity(value, self.getUnits() ^ (-1));	
}

quantity operator+(double value, const quantity& self)
{
	quantity q(value, unit());
	return q + self;
}

quantity operator-(double value, const quantity& self)
{
	quantity q(value, unit());
	return q - self;
}

quantity operator^(double value, const quantity& self)
{
	quantity q(value, unit());
	return q ^ self;	
}

quantity pow(double value, const quantity& exponent)
{
	quantity q(value, unit());
	return q ^ exponent;	
}

quantity pow(const quantity& self, double exponent)
{
	return (self ^ exponent);	
}

quantity pow(const quantity& self, const quantity& exponent)
{
	return (self ^ exponent);	
}

bool operator ==(double value, const quantity& self)
{
	quantity q(value, unit());
	return (q == self);
}

bool operator !=(double value, const quantity& self)
{
	quantity q(value, unit());
	return (q != self);
}

bool operator <=(double value, const quantity& self)
{
	quantity q(value, unit());
	return (q <= self);
}

bool operator >=(double value, const quantity& self)
{
	quantity q(value, unit());
	return (q >= self);
}

bool operator > (double value, const quantity& self)
{
	quantity q(value, unit());
	return (q > self);
}

bool operator < (double value, const quantity& self)
{
	quantity q(value, unit());
	return (q < self);
}

quantity exp(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: exp(%1%)") % q).str());
	return quantity(::exp(q.getValue()), unit());
}

quantity log(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: log(%1%)") % q).str());
	return quantity(::log(q.getValue()), unit());
}

quantity log10(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: log10(%1%)") % q).str());
	return quantity(::log10(q.getValue()), unit());
}

quantity sqrt(const quantity &q)
{
	return q ^ 0.5;
}

quantity sin(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: sin(%1%)") % q).str());
	return quantity(::sin(q.getValue()), unit());
}

quantity cos(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: cos(%1%)") % q).str());
	return quantity(::cos(q.getValue()), unit());
}

quantity tan(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: tan(%1%)") % q).str());
	return quantity(::tan(q.getValue()), unit());
}

quantity asin(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: asin(%1%)") % q).str());
	return quantity(::asin(q.getValue()), unit());
}

quantity acos(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: acos(%1%)") % q).str());
	return quantity(::acos(q.getValue()), unit());
}

quantity atan(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: atan(%1%)") % q).str());
	return quantity(::atan(q.getValue()), unit());
}

quantity sinh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: sinh(%1%)") % q).str());
	return quantity(::sinh(q.getValue()), unit());
}

quantity cosh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: cosh(%1%)") % q).str());
	return quantity(::cosh(q.getValue()), unit());
}

quantity tanh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: tanh(%1%)") % q).str());
	return quantity(::tanh(q.getValue()), unit());
}

quantity asinh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: asinh(%1%)") % q).str());
	return quantity(::asinh(q.getValue()), unit());
}

quantity acosh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: acosh(%1%)") % q).str());
	return quantity(::acosh(q.getValue()), unit());
}

quantity atanh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: atanh(%1%)") % q).str());
	return quantity(::atanh(q.getValue()), unit());
}

quantity atan2(const quantity &a, const quantity &b)
{
	if(a.getUnits() != unit() || b.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: atan2(%1%, %2%)") % a % b).str());
	return quantity(::atan2(a.getValue(), b.getValue()), unit());
}

quantity ceil(const quantity &q)
{
	return quantity(::ceil(q.getValue()), q.getUnits());
}

quantity floor(const quantity &q)
{
	return quantity(::floor(q.getValue()), q.getUnits());
}

quantity abs(const quantity &q)
{
	return quantity(::abs(q.getValue()), q.getUnits());
}

quantity max(const quantity &a, const quantity &b)
{
	if(a.getUnits() != b.getUnits())	
		throw units_error((boost::format("Invalid units in function: max(%1%, %2%)") % a % b).str());
	return quantity(std::max(a.getValue(), b.getValue()), a.getUnits());
}

quantity max(double v, const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: max(%1%, %2%)") % v % q).str());
	return quantity(std::max(v, q.getValue()), unit());
}

quantity max(const quantity &q, double v)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: max(%1%, %2%)") % v % q).str());
	return quantity(std::max(v, q.getValue()), unit());
}

quantity min(const quantity &a, const quantity &b)
{
	if(a.getUnits() != b.getUnits())	
		throw units_error((boost::format("Invalid units in function: min(%1%, %2%)") % a % b).str());
	return quantity(std::min(a.getValue(), b.getValue()), a.getUnits());
}

quantity min(double v, const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: min(%1%, %2%)") % v % q).str());
	return quantity(std::min(v, q.getValue()), unit());
}

quantity min(const quantity &q, double v)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: min(%1%, %2%)") % v % q).str());
	return quantity(std::min(v, q.getValue()), unit());
}
}
