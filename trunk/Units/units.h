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
		base_unit
**************************************************************/
const std::string __string_unit_delimiter__  = " ";

class base_unit
{
public:
	base_unit(void);
	base_unit(double multi, double _L, double _M, double _T, 
	          double _C, double _I, double _O, double _N);

	bool operator==(const base_unit& other) const;
	bool operator!=(const base_unit& other) const;

	base_unit operator*(const base_unit& other) const;
	base_unit operator*(double multi) const;
	base_unit operator/(const base_unit& other) const;
	base_unit operator/(double multi) const;
	base_unit operator^(double exponent) const;
	base_unit operator^(const base_unit& other) const;
	base_unit operator+(const base_unit& other) const;
	base_unit operator+(double value) const;
	base_unit operator+(void) const;
	base_unit operator-(const base_unit& other) const;
	base_unit operator-(double value) const;
	base_unit operator-(void) const;
	
	std::string toString(bool bUnitsOnly = false) const;
	friend std::ostream& operator<<(std::ostream& out, const base_unit& u);
	
	bool areDimensionsEqual(const base_unit& other) const;

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

/**************************************************************
		base_units_pool
**************************************************************/
namespace base_units_pool
{
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
	     std::string u8 = "", double exp8 = 0);
	unit(const std::map<std::string, double>& mapUnits);

	void addUnit(std::string name, double exp);
	void setUnits(const std::map<std::string, double>& mapUnits);

	static std::map<std::string, base_unit>& get_base_units(void);
	base_unit getBaseUnit(void) const;

	bool operator==(const unit& other) const;
	bool operator!=(const unit& other) const;
	
	quantity operator*(double multi) const;
	quantity operator/(double multi) const;
	
	unit operator*(const unit& other) const;
	unit operator/(const unit& other) const;
	unit operator^(double exponent) const;
	unit operator^(const unit& other) const;
	unit operator+(const unit& other) const;
	unit operator+(double value) const;
	unit operator+(void) const;
	unit operator-(const unit& other) const;
	unit operator-(double value) const;
	unit operator-(void) const;

	std::string toString(void) const;
	friend std::ostream& operator<<(std::ostream& out, const unit& u);
	
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
	quantity(void);
	quantity(double val, const unit& u);
	
	double getValue(void) const;
	void setValue(double val);
	void setValue(const quantity& other);
    double getValueInSIUnits(void) const;
	
	unit getUnits(void) const;
	void setUnits(const unit& u);
    
	quantity scaleTo(const quantity& referrer) const;
    quantity scaleTo(const unit& referrer) const;
	
	std::string toString(void) const;
	friend std::ostream& operator<<(std::ostream& out, const quantity& q);
	
	bool operator==(const quantity& other) const;
	bool operator==(double value) const;
	bool operator!=(const quantity& other) const;
	bool operator!=(double value) const;
	
	quantity operator+(const quantity& other) const;
	quantity operator+(double value) const;
	quantity operator+(void) const;
	quantity operator-(const quantity& other) const;
	quantity operator-(double value) const;
	quantity operator-(void) const;
	quantity operator*(const quantity& other) const;
	quantity operator*(const unit& other) const;
	quantity operator*(double value) const;
	quantity operator/(const quantity& other) const;
	quantity operator/(const unit& other) const;
	quantity operator/(double value) const;
	quantity operator^(const quantity& other) const;
	quantity operator^(double exponent) const;
    
	bool operator <=(const quantity& other) const;
	bool operator <=(double value) const;
    bool operator >=(const quantity& other) const;
    bool operator >=(double value) const;
    bool operator >(const quantity& other) const;
    bool operator >(double value) const;
    bool operator <(const quantity& other) const;
    bool operator <(double value) const;
	
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

quantity exp(const quantity &q);
quantity log(const quantity &q);
quantity log10(const quantity &q);
quantity sqrt(const quantity &q);
quantity sin(const quantity &q);
quantity cos(const quantity &q);
quantity tan(const quantity &q);
quantity asin(const quantity &q);
quantity acos(const quantity &q);
quantity atan(const quantity &q);

quantity sinh(const quantity &q);
quantity cosh(const quantity &q);
quantity tanh(const quantity &q);
quantity asinh(const quantity &q);
quantity acosh(const quantity &q);
quantity atanh(const quantity &q);
quantity atan2(const quantity &a, const quantity &b);

quantity ceil(const quantity &q);
quantity floor(const quantity &q);

quantity abs(const quantity &q);
quantity max(const quantity &a, const quantity &b);
quantity max(double v, const quantity &q);
quantity max(const quantity &q, double v);
quantity min(const quantity &a, const quantity &b);
quantity min(double v, const quantity &q);
quantity min(const quantity &q, double v);

}

#endif // UNITS_H
