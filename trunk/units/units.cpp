#include "units.h"

namespace units
{

base_unit operator*(double value, const base_unit& self)
{
	return self * value;
}

base_unit operator/(double value, const base_unit& self)
{
	throw std::runtime_error("");
	return base_unit();
}

base_unit operator+(double value, const base_unit& self)
{
	throw std::runtime_error("");
	return base_unit();	
}

base_unit operator-(double value, const base_unit& self)
{
	throw std::runtime_error("");
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


int unit::init_base_units(void)
{
	using namespace base_units_pool;
	
	__base_units__ = std::map<std::string, base_unit>();
	
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
	
	return 0;
}

std::map<std::string, base_unit> unit::__base_units__ = std::map<std::string, base_unit>();
int __dummy = unit::init_base_units();

quantity unit::operator*(double multi) const
{
    return quantity(multi, *this);
}

quantity unit::operator/(double multi) const
{
    return quantity(1.0 / multi, *this);
}

quantity operator*(double value, const unit& self)
{
	return self * value;
}

unit operator/(double value, const unit& self)
{
	throw std::runtime_error("");
	return unit();
}

unit operator+(double value, const unit& self)
{
	throw std::runtime_error("");
	return unit();	
}

unit operator-(double value, const unit& self)
{
	throw std::runtime_error("");
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




}
