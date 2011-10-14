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
const double hecto = 1E+2;
const double deka  = 1E+1;
const double deci  = 1E-1;
const double centi = 1E-2;
const double mili  = 1E-3;
const double micro = 1E-6;
const double nano  = 1E-9;
const double pico  = 1E-12;

const base_unit dimless = base_unit(1.0, 0, 0, 0, 0, 0, 0, 0);

const base_unit m   = base_unit(1.0, 1, 0, 0, 0, 0, 0, 0);
const base_unit kg  = base_unit(1.0, 0, 1, 0, 0, 0, 0, 0);
const base_unit s   = base_unit(1.0, 0, 0, 1, 0, 0, 0, 0);
const base_unit cd  = base_unit(1.0, 0, 0, 0, 1, 0, 0, 0);
const base_unit A   = base_unit(1.0, 0, 0, 0, 0, 1, 0, 0);
const base_unit K   = base_unit(1.0, 0, 0, 0, 0, 0, 1, 0);
const base_unit mol = base_unit(1.0, 0, 0, 0, 0, 0, 0, 1);

// Angles
const base_unit rad = base_unit(1.0, 0, 0, 0, 0, 0, 0, 0); // Radian
const base_unit sr  = base_unit(1.0, 0, 0, 0, 0, 0, 0, 0); // Steradian

// Time
const base_unit ms   = mili  * s;
const base_unit us   = micro * s;
const base_unit min  = 60    * s;
const base_unit hour = 3600  * s;
const base_unit day  = 43200 * s;

// Frequency
const base_unit Hz   = s^(-1);

// Length related:
const base_unit km = kilo  * m;
const base_unit cm = centi * m;
const base_unit mm = mili  * m;

// Volume:
const base_unit l  = 1E-3 * (m^3);
const base_unit dl = deci * l;

// Energy:
const base_unit N  = kg * m / (s^2); // Newtown
const base_unit J  = N * m;          // Joule
const base_unit W  = J / s;          // Watt

// Electromagnetism:
const base_unit C   = A * s;                    // Coulomb
const base_unit Ohm = J * s / (C^2);            // Ohm
const base_unit V   = kg * (m^2) / (A * (s^3)); // Volt
const base_unit F   = C / V;                    // Farad
const base_unit T   = kg / ((s^2) * A);         // Tesla
const base_unit H   = V * s / A;                // Henry
const base_unit S   = A / V;                    // Siemens
const base_unit Wb  = T * (m^2);                // Weber

// Pressure:
const base_unit Pa  = N / (m^2); // Pascal

// Viscosity
const base_unit P  = Pa * s;       // Poise
const base_unit St = (cm^2) / s;   // Stoke

// Radioactivity
const base_unit Bq  = s^(-1);     // Radioactivity, Becquerel
const base_unit Gy  = J / kg;     // Absorbed Dose, Gray
const base_unit Sv  = J / kg;     // Equivalent Dose, Sievert

// Optics
const base_unit lx = cd * sr / (m^2); // Illuminance, Lux
const base_unit lm = cd * sr;         // Luminous Flux, Lumen

// Catalytic Activity
const base_unit kat = mol / s;  // Katal

// Temporary used SI units
const base_unit knot = (1852.0/3600.0) * m / s; // Knot
const base_unit bar  = 1E5 * Pa;                // Bar
const base_unit b    = 1E-28 * (m^2);           // Barn
const base_unit Ci   = 3.7E10 * Bq;             // Curie
const base_unit R    = 2.58E-4 * C / kg;        // Roentgen
const base_unit rd   = 0.01 * Gy;               // Rad (absorbed radiation dose)
const base_unit rem	 = 0.01 * Sv;               // Rem (Roentgen equivalent man)
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
#define DECLARE_UNIT(UNIT) \
	const unit UNIT     = unit(std::string(#UNIT), 1); \
	const unit T##UNIT  = unit(std::string("T")  + std::string(#UNIT), 1); \
	const unit G##UNIT  = unit(std::string("G")  + std::string(#UNIT), 1); \
	const unit M##UNIT  = unit(std::string("M")  + std::string(#UNIT), 1); \
	const unit k##UNIT  = unit(std::string("k")  + std::string(#UNIT), 1); \
	const unit h##UNIT  = unit(std::string("h")  + std::string(#UNIT), 1); \
	const unit da##UNIT = unit(std::string("da") + std::string(#UNIT), 1); \
	const unit d##UNIT  = unit(std::string("d")  + std::string(#UNIT), 1); \
	const unit c##UNIT  = unit(std::string("c")  + std::string(#UNIT), 1); \
	const unit m##UNIT  = unit(std::string("m")  + std::string(#UNIT), 1); \
	const unit u##UNIT  = unit(std::string("u")  + std::string(#UNIT), 1); \
	const unit n##UNIT  = unit(std::string("n")  + std::string(#UNIT), 1); \
	const unit p##UNIT  = unit(std::string("p")  + std::string(#UNIT), 1); 


const unit dimensionless = unit();

// Fundamental units:
DECLARE_UNIT(m);
DECLARE_UNIT(s);
DECLARE_UNIT(cd);
DECLARE_UNIT(A);
DECLARE_UNIT(mol);
const unit kg  = unit("kg",  1);
const unit K   = unit("K",   1);
const unit rad = unit("rad", 1);
const unit sr  = unit("sr",  1);

// Time related:
const unit min  = unit("min",  1);
const unit hour = unit("hour", 1);
const unit day  = unit("day",  1);

// Volume:
const unit l  = unit("l",  1);
const unit dl = unit("dl", 1);

// Electrical:
DECLARE_UNIT(V);
DECLARE_UNIT(C);
DECLARE_UNIT(F);
DECLARE_UNIT(Ohm);
DECLARE_UNIT(T);
DECLARE_UNIT(H);
DECLARE_UNIT(S);
DECLARE_UNIT(Wb);

// Pressure:
DECLARE_UNIT(Pa);

// Viscosity
DECLARE_UNIT(P);
DECLARE_UNIT(St);

// Radioactivity
DECLARE_UNIT(Bq);
DECLARE_UNIT(Gy);
DECLARE_UNIT(Sv);

// Optics
DECLARE_UNIT(lx);
DECLARE_UNIT(lm);

DECLARE_UNIT(kat);

DECLARE_UNIT(knot);
DECLARE_UNIT(bar);
DECLARE_UNIT(b);
DECLARE_UNIT(Ci);
DECLARE_UNIT(R);
DECLARE_UNIT(rd);
DECLARE_UNIT(rem);

/*
// Fundamental units:
const unit m   = unit("m",   1);
const unit s   = unit("s",   1);
const unit cd  = unit("cd",  1);
const unit A   = unit("A",   1);
const unit mol = unit("mol", 1);

// Time related:
const unit Hz   = unit("Hz",   1);

// Volume:
const unit lit = unit("l",  1);
const unit dl  = unit("dl", 1);

// Energy:
const unit N  = unit("N",  1);
const unit J  = unit("J",  1);
const unit kJ = unit("kJ", 1);
const unit W  = unit("W",  1);
const unit kW = unit("kW", 1);

// Electrical:
const unit V   = unit("V",   1);
const unit C   = unit("C",   1);
const unit F   = unit("F",   1);
const unit Ohm = unit("Ohm", 1);
const unit T   = unit("T",   1);
const unit H   = unit("H",   1);
const unit S   = unit("S",   1);
const unit Wb  = unit("Wb",  1);
const unit mV  = unit("mV",  1);
const unit mA  = unit("mA",  1);

// Pressure:
const unit Pa  = unit("Pa",  1);
const unit kPa = unit("kPa", 1);
const unit MPa = unit("MPa", 1);

// Viscosity
const unit P  = unit("P",  1);
const unit St = unit("St", 1);

// Radioactivity
const unit Bq  = unit("Bq",  1);
const unit Gy  = unit("Gy",  1);
const unit Sv  = unit("Sv",  1);
const unit mSv = unit("mSv", 1);
const unit uSv = unit("uSv", 1);

// Optics
const unit lx = unit("lx",  1);
const unit lm = unit("lm",  1);

const unit kat = unit("kat",  1);

const unit knot = unit("knot",1);
const unit bar  = unit("bar", 1);
const unit b    = unit("b",   1);
const unit Ci   = unit("Ci",  1);
const unit R    = unit("R",   1);
const unit rd   = unit("rd",  1);
const unit rem  = unit("rem", 1);
*/
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
