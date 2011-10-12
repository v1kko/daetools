#include "units.h"

namespace units
{
/**************************************************************
		base_unit
**************************************************************/
base_unit operator*(double value, const base_unit& self)
{
	return self * value;
}

base_unit operator/(double value, const base_unit& self)
{
	return value * (self ^ (-1.0));
}

base_unit operator+(double value, const base_unit& self)
{
	throw units_error((boost::format("Invalid operation: %1% + %2%") % value % self).str());
	return base_unit();	
}

base_unit operator-(double value, const base_unit& self)
{
	throw units_error((boost::format("Invalid operation: %1% - %2%") % value % self).str());
	return base_unit();	
}

base_unit pow(const base_unit& self, double exponent)
{
	return self ^ exponent;
}

base_unit pow(const base_unit& self, const base_unit& exponent)
{
	return self ^ exponent;
}

/**************************************************************
		unit
**************************************************************/
inline void fill_base_units(std::map<std::string, base_unit>& __base_units__)
{
	using namespace base_units_pool;
	
	__base_units__["kg"] = kg;
	__base_units__["m"] = m;
	__base_units__["s"] = s;
	__base_units__["cd"] = cd;
	__base_units__["A"] = A;
	__base_units__["K"] = K;
	__base_units__["mol"] = mol;
	
	__base_units__["ms"] = ms;
	__base_units__["us"] = us;
	__base_units__["min"] = min;
	__base_units__["hour"] = hour;
	__base_units__["day"] = day;
	__base_units__["Hz"] = Hz;
	__base_units__["kHz"] = kHz;
	__base_units__["MHz"] = MHz;
	
	__base_units__["km"] = km;
	__base_units__["dm"] = dm;
	__base_units__["cm"] = cm;
	__base_units__["mm"] = mm;
	__base_units__["um"] = um;
	__base_units__["nm"] = nm;
	
	__base_units__["l"] = lit;
	__base_units__["dl"] = dl;
	
	__base_units__["N"] = N;
	__base_units__["J"] = J;
	__base_units__["kJ"] = kJ;
	__base_units__["W"] = W;
	__base_units__["kW"] = kW;
	
	__base_units__["C"] = C;
	__base_units__["F"] = F;
	__base_units__["Ohm"] = Ohm;
	__base_units__["V"] = V;
	__base_units__["mV"] = mV;
	__base_units__["mA"] = mA;
	
	__base_units__["Pa"] = Pa;
	__base_units__["kPa"] = kPa;
	__base_units__["MPa"] = MPa;
	
	__base_units__["P"] = P;
	__base_units__["St"] = St;
	//for(std::map<std::string, base_unit>::iterator iter = __base_units__.begin(); iter != __base_units__.end(); iter++)
	//	std::cout << (*iter).first << " = " << (*iter).second.toString() << std::endl;
}

std::map<std::string, base_unit>& unit::get_base_units(void)
{
	static std::map<std::string, base_unit> __base_units__;
	fill_base_units(__base_units__);
	return __base_units__;
}

quantity unit::operator*(double multi) const
{
    return quantity(multi, *this);
}

quantity operator*(double multi, const unit& self)
{
    return quantity(multi, self);
}

quantity unit::operator/(double multi) const
{
    return quantity(1.0 / multi, *this);
}

quantity operator/(double multi, const unit& self)
{
    return quantity(multi, self ^ (-1));
}

unit operator+(double value, const unit& self)
{
	throw units_error((boost::format("Invalid operation: %1% + %2%") % value % self).str());
	return unit();	
}

unit operator-(double value, const unit& self)
{
	throw units_error((boost::format("Invalid operation: %1% - %2%") % value % self).str());
	return unit();	
}

unit pow(const unit& self, const unit& other)
{
	return self ^ other;
}

unit pow(const unit& self, double exponent)
{
	return self ^ exponent;
}

/**************************************************************
		quantity
**************************************************************/
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



}
