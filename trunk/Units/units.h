#ifndef UNITS_H
#define UNITS_H

#include <boost/assign/list_of.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <math.h>

namespace units
{
/**************************************************************
		units_error
**************************************************************/
class units_error : public std::runtime_error
{
public:
	units_error(const std::string& error) : std::runtime_error(error)
	{		
	}
};

/**************************************************************
		Some handy functions
**************************************************************/
inline std::string to_string(const std::string& name, double exponent)
{
	if(exponent == 0)
		return std::string();
	if(exponent == 1)
		return name;
	if((int)exponent == exponent)
		return (boost::format("%1%^%2%") % name % (int)exponent).str();
	else	
		return (boost::format("%1%^%2%") % name % exponent).str();
}

inline void to_string_and_add(const std::string& name, double exponent, std::vector<std::string>& arrUnits)
{
	std::string res = to_string(name, exponent);
	if(!res.empty())
		arrUnits.push_back(res);
}

/**************************************************************
		base_unit
**************************************************************/
const std::string __string_unit_delimiter__  = " ";

class base_unit
{
public:
	base_unit(void)
	{
        L          = 0; //length, m
        M          = 0; //mass, kg
        T          = 0; //time, s
        C          = 0; //luminous intensity, cd
        I          = 0; //el. current, A
        O          = 0; //temperature, K
        N          = 0; //amount of a substance, mol
        multiplier = 1.0;	
	}

	base_unit(double multi, double _L, double _M, double _T, 
	          double _C, double _I, double _O, double _N)
	{
        L          = _L; //length, m
        M          = _M; //mass, kg
        T          = _T; //time, s
        C          = _C; //luminous intensity, cd
        I          = _I; //el. current, A
        O          = _O; //temperature, K
        N          = _N; //amount of a substance, mol
        multiplier = multi;	
	}

	bool operator==(const base_unit& other) const
	{
		return areDimensionsEqual(other);
	}

	bool operator!=(const base_unit& other) const
	{
		return !(*this == other);
	}

    bool isEqualTo(const base_unit& other) const
	{
        if((multiplier == other.multiplier) && areDimensionsEqual(other))
           return true;
        else
            return false;
	}
		    
	bool areDimensionsEqual(const base_unit& other) const
	{
        if ((M == other.M) && (L == other.L) && (T == other.T) && (C == other.C) &&
            (I == other.I) && (O == other.O) && (N == other.N))
           return true;
        else
            return false;
	}

	base_unit operator*(const base_unit& other) const
	{
		base_unit tmp;
        tmp.L = L + other.L;
        tmp.M = M + other.M;
        tmp.T = T + other.T;
        tmp.C = C + other.C;
        tmp.I = I + other.I;
        tmp.O = O + other.O;
        tmp.N = N + other.N;
        tmp.multiplier = multiplier * other.multiplier;
        return tmp;
	}

	base_unit operator*(double multi) const
	{
		base_unit tmp = *this;
        tmp.multiplier = multiplier * multi;
        return tmp;
	}
	
	base_unit operator/(const base_unit& other) const
	{
		base_unit tmp;
        tmp.L = L - other.L;
        tmp.M = M - other.M;
        tmp.T = T - other.T;
        tmp.C = C - other.C;
        tmp.I = I - other.I;
        tmp.O = O - other.O;
        tmp.N = N - other.N;
        tmp.multiplier = multiplier / other.multiplier;
        return tmp;
	}
	
	base_unit operator/(double multi) const
	{
		base_unit tmp = *this;
        tmp.multiplier = multiplier / multi;
        return tmp;
	}

	base_unit operator^(double exponent) const
	{
		base_unit tmp;
	    tmp.L = L * exponent;
	    tmp.M = M * exponent;
	    tmp.T = T * exponent;
	    tmp.C = C * exponent;
	    tmp.I = I * exponent;
	    tmp.O = O * exponent;
	    tmp.N = N * exponent;
	    tmp.multiplier = ::pow(multiplier, exponent);
	    return tmp;		
	}
	
	base_unit operator^(const base_unit& other) const
	{
		throw units_error((boost::format("Invalid operation: %1% ^ %2%") % *this % other).str());
		return base_unit();
	}

	base_unit operator+(const base_unit& other) const
	{
		throw units_error((boost::format("Invalid operation: %1% + %2%") % *this % other).str());
		return base_unit();	
	}
	
	base_unit operator+(double value) const
	{
		throw units_error((boost::format("Invalid operation: %1% + %2%") % *this % value).str());
		return base_unit();	
	}

	base_unit operator+(void) const
	{
		throw units_error((boost::format("Invalid operation: +%1%") % *this).str());
		return base_unit();	
	}
	
	base_unit operator-(const base_unit& other) const
	{
		throw units_error((boost::format("Invalid operation: %1% - %2%") % *this % other).str());
		return base_unit();	
	}
	
	base_unit operator-(double value) const
	{
		throw units_error((boost::format("Invalid operation: %1% - %2%") % *this % value).str());
		return base_unit();	
	}

	base_unit operator-(void) const
	{
		throw units_error((boost::format("Invalid operation: -%1%") % *this).str());
		return base_unit();	
	}
	
	std::string toString(bool bUnitsOnly = false) const
	{
		std::vector<std::string> arrUnits;
        
		to_string_and_add("m",   L, arrUnits);
        to_string_and_add("kg",  M, arrUnits);
        to_string_and_add("s",   T, arrUnits);
        to_string_and_add("cd",  C, arrUnits);
        to_string_and_add("A",   I, arrUnits);
        to_string_and_add("K",   O, arrUnits);
        to_string_and_add("mol", N, arrUnits);
		
		std::string _units = boost::algorithm::join(arrUnits, __string_unit_delimiter__);
		if(bUnitsOnly)
			return (boost::format("%s") % _units).str();
		else
			return (boost::format("%.10f [%s]") % multiplier % _units).str();
	}
	
	friend std::ostream& operator<<(std::ostream& out, const base_unit& u)
	{
	    out << u.toString();
	    return out;
	}

public:
	double L;
	double M;
	double T;
	double C;
	double I;
	double O;
	double N;
	double multiplier;	
};
base_unit operator*(double value, const base_unit& self);
base_unit operator/(double value, const base_unit& self);
base_unit operator+(double value, const base_unit& self);
base_unit operator-(double value, const base_unit& self);
base_unit pow      (const base_unit& self, double exponent);
base_unit pow      (const base_unit& self, const base_unit& exponent);

const double tera  = 1E+12;
const double giga  = 1E+9;
const double mega  = 1E+6;
const double kilo  = 1E+3;
const double deka  = 1E+1;
const double deci  = 1E-1;
const double centi = 1E-2;
const double mili  = 1E-3;
const double micro = 1E-6;
const double nano  = 1E-9;
const double pico  = 1E-12;

/**************************************************************
		base_units_pool
**************************************************************/
namespace base_units_pool
{
const base_unit dimless = base_unit(1.0, 0, 0, 0, 0, 0, 0, 0);

const base_unit m  = base_unit(1.0, 1, 0, 0, 0, 0, 0, 0);
const base_unit kg = base_unit(1.0, 0, 1, 0, 0, 0, 0, 0);
const base_unit s  = base_unit(1.0, 0, 0, 1, 0, 0, 0, 0);
const base_unit cd = base_unit(1.0, 0, 0, 0, 1, 0, 0, 0);
const base_unit A  = base_unit(1.0, 0, 0, 0, 0, 1, 0, 0);
const base_unit K  = base_unit(1.0, 0, 0, 0, 0, 0, 1, 0);
const base_unit mol= base_unit(1.0, 0, 0, 0, 0, 0, 0, 1);

const base_unit ms   = mili  * s;
const base_unit us   = micro * s;
const base_unit min  = 60    * s;
const base_unit hour = 3600  * s;
const base_unit day  = 43200 * s;
const base_unit Hz   = s^(-1);
const base_unit kHz  = kilo * Hz;
const base_unit MHz  = mega * Hz;

// Length related:
const base_unit km = kilo  * m;
const base_unit dm = deci  * m;
const base_unit cm = centi * m;
const base_unit mm = mili  * m;
const base_unit um = micro * m;
const base_unit nm = nano  * m;

// Volume:
const base_unit lit = 1E-3 * (m^3);
const base_unit dl  = deci * lit;

// Energy:
const base_unit N  = kg * m / (s^2);
const base_unit J  = N * m;
const base_unit kJ = kilo * J;
const base_unit W  = J / s;
const base_unit kW = kilo * W;

// Electrical:
const base_unit V   = kg * (m^2) / (A * (s^3)); // Volt
const base_unit C   = A * s;                    // Coulomb
const base_unit F   = C / V;                    // Farad
const base_unit Ohm = J * s / (C^2);
const base_unit mV  = mili * V;
const base_unit mA  = mili * A;

// Pressure:
const base_unit Pa  = N / (m^2);
const base_unit kPa = kilo * Pa;
const base_unit MPa = mega * Pa;

// Viscosity
const base_unit P  = Pa * s;       // Poise
const base_unit St = (cm^2) / s;   // Stoke
}

/**************************************************************
		unit
**************************************************************/
class quantity;
class unit
{
public:
	unit(std::string u1 = "", double exp1 = 0,
	     std::string u2 = "", double exp2 = 0,
	     std::string u3 = "", double exp3 = 0,
	     std::string u4 = "", double exp4 = 0,
	     std::string u5 = "", double exp5 = 0,
	     std::string u6 = "", double exp6 = 0,
	     std::string u7 = "", double exp7 = 0,
	     std::string u8 = "", double exp8 = 0)
	{
		addUnit(u1, exp1);
		addUnit(u2, exp2);
		addUnit(u3, exp3);
		addUnit(u4, exp4);
		addUnit(u5, exp5);
		addUnit(u6, exp6);
		addUnit(u7, exp7);
		addUnit(u8, exp8);
	}

	unit(const std::map<std::string, double>& mapUnits)
	{
		setUnits(mapUnits);
	}

	void addUnit(std::string name, double exp)
	{
		if(name.empty())
			return;
		std::map<std::string, base_unit>& __base_units__ = get_base_units();
        std::map<std::string, base_unit>::const_iterator iter = __base_units__.find(name);
        if(iter == __base_units__.end())
			throw units_error((boost::format("Cannot find the base_unit: %1%") % name).str());
        units[name] = exp;
	}

	void setUnits(const std::map<std::string, double>& mapUnits)
	{
		units.clear();
        for(std::map<std::string, double>::const_iterator iter = mapUnits.begin(); iter != mapUnits.end(); iter++)
			addUnit(iter->first, iter->second);
	}

	static std::map<std::string, base_unit>& get_base_units(void);
	
	base_unit getBaseUnit(void) const
	{
		base_unit tmp = base_units_pool::dimless;
		std::map<std::string, base_unit>& __base_units__ = get_base_units();
		
		for(std::map<std::string, double>::const_iterator iter = units.begin(); iter != units.end(); iter++)
		{
			std::string name = (*iter).first;
			double      exp  = (*iter).second;
			tmp = tmp * (__base_units__[name] ^ exp);
		}
		return tmp;
	}

	bool operator==(const unit& other) const
	{
		return (getBaseUnit() == other.getBaseUnit());
	}

	bool operator!=(const unit& other) const
	{
		return !(*this == other);
	}

	unit operator*(const unit& other) const
	{
		unit tmp(*this);
		for(std::map<std::string, double>::const_iterator iter = other.units.begin(); iter != other.units.end(); iter++)
		{
			std::string name = (*iter).first;
			double      exp  = (*iter).second;
			
			if(tmp.units.find(name) == tmp.units.end())
				tmp.units[name] = exp;
			else
				tmp.units[name] = tmp.units[name] + exp;
		}
        return tmp;
	}
	quantity operator*(double multi) const;
	
	unit operator/(const unit& other) const
	{
		unit tmp(*this);
		for(std::map<std::string, double>::const_iterator iter = other.units.begin(); iter != other.units.end(); iter++)
		{
			std::string name = (*iter).first;
			double      exp  = (*iter).second;
			
			if(tmp.units.find(name) == tmp.units.end())
				tmp.units[name] = -exp;
			else
				tmp.units[name] = tmp.units[name] - exp;
		}
        return tmp;
	}
	quantity operator/(double multi) const;

	unit operator^(double exponent) const
	{
		unit tmp;
		for(std::map<std::string, double>::const_iterator iter = units.begin(); iter != units.end(); iter++)
		{
			std::string name = (*iter).first;
			double      exp  = (*iter).second;
			tmp.units[name]  = exp * exponent;
		}
        return tmp;
	}
	
	unit operator^(const unit& other) const
	{
		throw units_error((boost::format("Invalid operation: %1% ^ %2%") % *this % other).str());
		return unit();	
	}

	unit operator+(const unit& other) const
	{
		throw units_error((boost::format("Invalid operation: %1% + %2%") % *this % other).str());
		return unit();	
	}
	
	unit operator+(double value) const
	{
		throw units_error((boost::format("Invalid operation: %1% + %2%") % *this % value).str());
		return unit();	
	}

	unit operator+(void) const
	{
		throw units_error((boost::format("Invalid operation: +%1%") % *this).str());
		return unit();	
	}

	unit operator-(const unit& other) const
	{
		throw units_error((boost::format("Invalid operation: %1% - %2%") % *this % other).str());
		return unit();	
	}
	
	unit operator-(double value) const
	{
		throw units_error((boost::format("Invalid operation: %1% - %2%") % *this % value).str());
		return unit();	
	}

	unit operator-(void) const
	{
		throw units_error((boost::format("Invalid operation: -%1%") % *this).str());
		return unit();	
	}

	std::string toString(void) const
	{
		std::vector<std::string> arrUnits;
		for(std::map<std::string, double>::const_iterator iter = units.begin(); iter != units.end(); iter++)
		{
			std::string name = (*iter).first;
			double      exp  = (*iter).second;
			to_string_and_add(name, exp, arrUnits);
		}
		return boost::algorithm::join(arrUnits, __string_unit_delimiter__);
	}

	friend std::ostream& operator<<(std::ostream& out, const unit& u)
	{
	    out << u.toString();
	    return out;
	}
	
public:
	std::map<std::string, double> units;
};
quantity operator*(double value, const unit& self);
quantity operator/(double value, const unit& self);
unit     operator+(double value, const unit& self);
unit     operator-(double value, const unit& self);
unit     pow      (const unit& self, double exponent);
unit     pow      (const unit& self, const unit& exponent);

/**************************************************************
		units_pool
**************************************************************/
namespace units_pool
{
const unit dimensionless = unit();

// Fundamental units:
const unit m  = unit("m",   1);
const unit kg = unit("kg",  1);
const unit s  = unit("s",   1);
const unit cd = unit("cd",  1);
const unit A  = unit("A",   1);
const unit K  = unit("K",   1);
const unit mol= unit("mol", 1);

// Time related:
const unit ms   = unit("ms",   1);
const unit us   = unit("us",   1);
const unit min  = unit("min",  1);
const unit hour = unit("hour", 1);
const unit day  = unit("day",  1);
const unit Hz   = unit("Hz",   1);
const unit kHz  = unit("kHz",  1);
const unit MHz  = unit("MHz",  1);

// Length related:
const unit km = unit("km", 1);
const unit dm = unit("dm", 1);
const unit cm = unit("cm", 1);
const unit mm = unit("mm", 1);
const unit um = unit("um", 1);
const unit nm = unit("nm", 1);

// Volume:
const unit lit = unit("l", 1);
const unit dl  = unit("dl", 1);

// Energy:
const unit N  = unit("N", 1);
const unit J  = unit("J", 1);
const unit kJ = unit("kJ", 1);
const unit W  = unit("W", 1);
const unit kW = unit("kW", 1);

// Electrical:
const unit V   = unit("V",   1);
const unit C   = unit("C",   1);
const unit F   = unit("F",   1);
const unit Ohm = unit("Ohm", 1);
const unit mV  = unit("mV",  1);
const unit mA  = unit("mA",  1);

// Pressure:
const unit Pa  = unit("Pa",  1);
const unit kPa = unit("kPa", 1);
const unit MPa = unit("MPa", 1);

// Viscosity
const unit P  = unit("P",  1);
const unit St = unit("St", 1);
}

/**************************************************************
		quantity
**************************************************************/
class quantity
{
public:
	quantity(void)
	{
		_value = 0.0;
	}
	
	quantity(double val, const unit& u)
	{
		_units = u;
		_value = val;
	}
	
	double getValue(void) const
	{
		return _value;
	}
	
	void setValue(double val)
	{
		_value = val;
	}
	
	void setValue(const quantity& other)
	{
		_value = other.scaleTo(*this).getValue();
	}

	unit getUnits(void) const
	{
		return _units;
	}
	
	void setUnits(const unit& u)
	{
		_units = u;
	}

    double getValueInSIUnits(void) const
	{
        return _value * _units.getBaseUnit().multiplier;
    }
	
    quantity scaleTo(const quantity& referrer) const
	{
		return scaleTo(referrer.getUnits());
	}

    quantity scaleTo(const unit& referrer) const
	{
        if(_units.getBaseUnit() != referrer.getBaseUnit())
            throw units_error((boost::format("Units not consistent: scale from %1% to %2%") % _units % referrer).str());

		quantity tmp;
        tmp.setUnits(referrer);
        tmp.setValue(_value * _units.getBaseUnit().multiplier / referrer.getBaseUnit().multiplier);
        return tmp;
	}
	
	std::string toString(void) const
	{
		return (boost::format("%.10f %s") % _value % _units.toString()).str();
	}

	friend std::ostream& operator<<(std::ostream& out, const quantity& q)
	{
	    out << q.toString();
	    return out;
	}
	
	bool operator==(const quantity& other) const
	{
		if(_units != other.getUnits())
			throw units_error((boost::format("Units not consistent: %1% == %2%") % _units % other.getUnits()).str());
			
		return (_value == other.scaleTo(*this).getValue());	
	}
	
	bool operator==(double value) const
	{
	// Here assume that the value is in the same units as this quantity
	// We need this for unit-consistency checks
		return (getValueInSIUnits() == value);
	}

	bool operator!=(const quantity& other) const
	{
		return !(*this == other);
	}
	
	bool operator!=(double value) const
	{
		return !(*this == value);
	}

	quantity operator+(const quantity& other) const
	{
        if(_units != other.getUnits())
            throw units_error((boost::format("Units not consistent: %1% + %2%") % _units % other.getUnits()).str());
		
		quantity tmp(_value + other.scaleTo(*this).getValue(), _units);
		return tmp;	
	}

	quantity operator+(double value) const
	{
		quantity q(value, unit());
		return (*this + q);
	}
	
	quantity operator+(void) const
	{
		quantity tmp(_value, _units);
		return tmp;	
	}

	quantity operator-(const quantity& other) const
	{
        if(_units != other.getUnits())
            throw units_error((boost::format("Units not consistent: %1% - %2%") % _units % other.getUnits()).str());
		
		quantity tmp(_value - other.scaleTo(*this).getValue(), _units);
		return tmp;	
	}
	
	quantity operator-(double value) const
	{
		quantity q(value, unit());
		return (*this - q);
	}

	quantity operator-(void) const
	{
		quantity tmp(-_value, _units);
		return tmp;	
	}

	quantity operator*(const quantity& other) const
	{
		quantity tmp;
        tmp.setUnits(_units * other.getUnits());
        tmp.setValue(_value * other.getValue());
		return tmp;	
	}

	quantity operator*(double value) const
	{
		quantity q(value, unit());
		return (*this * q);
	}

	quantity operator/(const quantity& other) const
	{
		quantity tmp;
        tmp.setUnits(_units / other.getUnits());
        tmp.setValue(_value / other.getValue());
		return tmp;	
	}
	
	quantity operator/(double value) const
	{
		quantity q(value, unit());
		return (*this / q);
	}

	quantity operator^(const quantity& other) const
	{
		if(other.getUnits() != unit())
			throw units_error((boost::format("Exponent must be dimension-less in: %1% ^ %2%") % _units % other.getUnits()).str());
		return (*this ^ other.getValueInSIUnits());
	}

	quantity operator^(double exponent) const
	{
		quantity tmp;
        tmp.setUnits(_units ^ exponent);
        tmp.setValue(::pow(_value, exponent));
		return tmp;	
	}

    bool operator <=(const quantity& other) const
	{
		if(_units != other.getUnits())
			throw units_error((boost::format("Units not consistent: %1% <= %2%") % _units % other.getUnits()).str());
			
		return _value <= other.scaleTo(*this).getValue();	
	}
   
	bool operator <=(double value) const
	{
		quantity q(value, unit());
		return (*this <= q);
	}

    bool operator >=(const quantity& other) const
	{
		if(_units != other.getUnits())
			throw units_error((boost::format("Units not consistent: %1% >= %2%") % _units % other.getUnits()).str());
			
		return _value >= other.scaleTo(*this).getValue();	
	}
   
    bool operator >=(double value) const
	{
		quantity q(value, unit());
		return (*this >= q);
	}

    bool operator >(const quantity& other) const
	{
		if(_units != other.getUnits())
			throw units_error((boost::format("Units not consistent: %1% > %2%") % _units % other.getUnits()).str());
			
		return _value > other.scaleTo(*this).getValue();	
	}
   
    bool operator >(double value) const
	{
		quantity q(value, unit());
		return (*this > q);
	}

    bool operator <(const quantity& other) const
	{
		if(_units != other.getUnits())
			throw units_error((boost::format("Units not consistent: %1% < %2%") % _units % other.getUnits()).str());
			
		return _value < other.scaleTo(*this).getValue();	
	}
   
    bool operator <(double value) const
	{
		quantity q(value, unit());
		return (*this < q);
	}
	
protected:
	unit   _units;
	double _value;
};

quantity operator*(double value, const quantity& self);
quantity operator/(double value, const quantity& self);
quantity operator+(double value, const quantity& self);
quantity operator-(double value, const quantity& self);
quantity operator^(double value, const quantity& self);
quantity pow      (double value, const quantity& exponent);
quantity pow      (const quantity& self, double exponent);
quantity pow      (const quantity& self, const quantity& exponent);

bool operator ==(double value, const quantity& self);
bool operator !=(double value, const quantity& self);
bool operator <=(double value, const quantity& self);
bool operator >=(double value, const quantity& self);
bool operator > (double value, const quantity& self);
bool operator < (double value, const quantity& self);


}

#endif // UNITS_H
