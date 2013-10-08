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
const std::string __string_unit_delimiter__  = " * ";

class base_unit
{
public:
	base_unit(void);
	base_unit(double multi, double _L, double _M, double _T, 
	          double _C, double _I, double _O, double _N);

	bool operator==(const base_unit& other) const;
	bool operator!=(const base_unit& other) const;

	const base_unit operator*(const base_unit& other) const;
	const base_unit operator*(double multi) const;
	const base_unit operator/(const base_unit& other) const;
	const base_unit operator/(double multi) const;
	const base_unit operator^(double exponent) const;
	const base_unit operator^(const base_unit& other) const;
	const base_unit operator+(const base_unit& other) const;
	const base_unit operator+(double value) const;
	const base_unit operator+(void) const;
	const base_unit operator-(const base_unit& other) const;
	const base_unit operator-(double value) const;
	const base_unit operator-(void) const;
	
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
const base_unit operator*(double value, const base_unit& self);
const base_unit operator/(double value, const base_unit& self);
const base_unit operator+(double value, const base_unit& self);
const base_unit operator-(double value, const base_unit& self);
const base_unit pow      (const base_unit& self, double exponent);
const base_unit pow      (const base_unit& self, const base_unit& exponent);

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
	const base_unit getBaseUnit(void) const;

	bool operator==(const unit& other) const;
	bool operator!=(const unit& other) const;
	
	const quantity operator*(double multi) const;
	const quantity operator/(double multi) const;
	
	const unit operator*(const unit& other) const;
	const unit operator/(const unit& other) const;
	const unit operator^(double exponent) const;
	const unit operator^(const unit& other) const;
	const unit operator+(const unit& other) const;
	const unit operator+(double value) const;
	const unit operator+(void) const;
	const unit operator-(const unit& other) const;
	const unit operator-(double value) const;
	const unit operator-(void) const;

	std::string toString(void) const;
	std::string toLatex(void) const;
	friend std::ostream& operator<<(std::ostream& out, const unit& u);
	
public:
	std::map<std::string, double> units;
};

inline std::string toString(const std::vector<unit>& uarray, const std::string& strDelimiter = std::string(", "))
{
	std::string result;
	for(size_t i = 0; i < uarray.size(); i++)
	{
		if(i != 0)
			result += strDelimiter;
		result += uarray[i].toString();
	}
	return result;
}

const quantity operator*(double value, const unit& self);
const quantity operator/(double value, const unit& self);
const unit     operator+(double value, const unit& self);
const unit     operator-(double value, const unit& self);
const unit     pow      (const unit& self, double exponent);
const unit     pow      (const unit& self, const unit& exponent);

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
    
	const quantity scaleTo(const quantity& referrer) const;
    const quantity scaleTo(const unit& referrer) const;
	
	std::string toString(void) const;
	std::string toLatex(void) const;
	friend std::ostream& operator<<(std::ostream& out, const quantity& q);
	
	bool operator==(const quantity& other) const;
	bool operator==(double value) const;
	bool operator!=(const quantity& other) const;
	bool operator!=(double value) const;
	
	const quantity operator+(const quantity& other) const;
	const quantity operator+(double value) const;
	const quantity operator+(void) const;
	const quantity operator-(const quantity& other) const;
	const quantity operator-(double value) const;
	const quantity operator-(void) const;
	const quantity operator*(const quantity& other) const;
	const quantity operator*(const unit& other) const;
	const quantity operator*(double value) const;
	const quantity operator/(const quantity& other) const;
	const quantity operator/(const unit& other) const;
	const quantity operator/(double value) const;
	const quantity operator^(const quantity& other) const;
	const quantity operator^(double exponent) const;
    
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

inline std::string toString(const std::vector<quantity>& qarray, const std::string& strDelimiter = std::string(", "))
{
	std::string result;
	for(size_t i = 0; i < qarray.size(); i++)
	{
		if(i != 0)
			result += strDelimiter;
		result += qarray[i].toString();
	}
	return result;
}

const quantity operator*(double value, const quantity& self);
const quantity operator/(double value, const quantity& self);
const quantity operator+(double value, const quantity& self);
const quantity operator-(double value, const quantity& self);
const quantity operator^(double value, const quantity& self);
const quantity pow      (double value, const quantity& exponent);
const quantity pow      (const quantity& self, double exponent);
const quantity pow      (const quantity& self, const quantity& exponent);

bool operator ==(double value, const quantity& self);
bool operator !=(double value, const quantity& self);
bool operator <=(double value, const quantity& self);
bool operator >=(double value, const quantity& self);
bool operator > (double value, const quantity& self);
bool operator < (double value, const quantity& self);

const quantity exp(const quantity &q);
const quantity log(const quantity &q);
const quantity log10(const quantity &q);
const quantity sqrt(const quantity &q);
const quantity sin(const quantity &q);
const quantity cos(const quantity &q);
const quantity tan(const quantity &q);
const quantity asin(const quantity &q);
const quantity acos(const quantity &q);
const quantity atan(const quantity &q);

const quantity sinh(const quantity &q);
const quantity cosh(const quantity &q);
const quantity tanh(const quantity &q);
const quantity asinh(const quantity &q);
const quantity acosh(const quantity &q);
const quantity atanh(const quantity &q);
const quantity atan2(const quantity &a, const quantity &b);

const quantity ceil(const quantity &q);
const quantity floor(const quantity &q);

const quantity abs(const quantity &q);
const quantity max(const quantity &a, const quantity &b);
const quantity max(double v, const quantity &q);
const quantity max(const quantity &q, double v);
const quantity min(const quantity &a, const quantity &b);
const quantity min(double v, const quantity &q);
const quantity min(const quantity &q, double v);

class create_base_units
{
public:
	create_base_units();

public:
	std::map<std::string, base_unit> __base_units__;
};

}

#endif // UNITS_H
