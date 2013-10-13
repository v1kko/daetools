#include "units.h"
#include "units_pool.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace units
{
inline std::string to_string(const std::string& name, double exponent)
{
	if(exponent == 0)
		return std::string();
	if(exponent == 1)
		return name;

    std::string strExponent;
    if((int)exponent == exponent)
    {
        if(exponent < 0)
            strExponent = (boost::format("(%1%)") % (int)exponent).str();
        else
            strExponent = (boost::format("%1%") % (int)exponent).str();
    }
    else
    {
        if(exponent < 0)
            strExponent = (boost::format("(%1%)") % exponent).str();
        else
            strExponent = (boost::format("%1%") % exponent).str();
    }

    return (boost::format("%1%%2%%3%") % name % __string_unit_power__ % strExponent).str();
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
base_unit::base_unit(void)
{
	L          = 0; // length, m
	M          = 0; // mass, kg
	T          = 0; // time, s
	C          = 0; // luminous intensity, cd
	I          = 0; // el. current, A
	O          = 0; // temperature, K
	N          = 0; // amount of a substance, mol
	multiplier = 1.0;	
}

base_unit::base_unit(double multi, double _L, double _M, double _T, 
		             double _C, double _I, double _O, double _N)
{
	L          = _L; // length, m
	M          = _M; // mass, kg
	T          = _T; // time, s
	C          = _C; // luminous intensity, cd
	I          = _I; // el. current, A
	O          = _O; // temperature, K
	N          = _N; // amount of a substance, mol
	multiplier = multi;	
}

bool base_unit::operator==(const base_unit& other) const
{
	bool multipliers_equal = false;
	if(multiplier == other.multiplier)
		multipliers_equal = true;
	else if(fabs(multiplier - other.multiplier) < fabs(other.multiplier)*1E-7)
		multipliers_equal = true;
	
	//std::cout << (boost::format("%.20f == %.20f") % multiplier % other.multiplier).str() << std::endl;
	return (multipliers_equal && areDimensionsEqual(other));
}

bool base_unit::operator!=(const base_unit& other) const
{
	return !(*this == other);
}

bool base_unit::areDimensionsEqual(const base_unit& other) const
{
	return ((M == other.M) && (L == other.L) && (T == other.T) && (C == other.C) &&
			(I == other.I) && (O == other.O) && (N == other.N));
}

const base_unit base_unit::operator*(const base_unit& other) const
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

const base_unit base_unit::operator*(double multi) const
{
	base_unit tmp = *this;
	tmp.multiplier = multiplier * multi;
	return tmp;
}

const base_unit base_unit::operator/(const base_unit& other) const
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

const base_unit base_unit::operator/(double multi) const
{
	base_unit tmp = *this;
	tmp.multiplier = multiplier / multi;
	return tmp;
}

const base_unit base_unit::operator^(double exponent) const
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

const base_unit base_unit::operator^(const base_unit& other) const
{
	throw units_error((boost::format("Invalid operation: %1% ^ %2%") % *this % other).str());
	return base_unit();
}

const base_unit base_unit::operator+(const base_unit& other) const
{
	throw units_error((boost::format("Invalid operation: %1% + %2%") % *this % other).str());
	return base_unit();	
}

const base_unit base_unit::operator+(double value) const
{
	throw units_error((boost::format("Invalid operation: %1% + %2%") % *this % value).str());
	return base_unit();	
}

const base_unit base_unit::operator+(void) const
{
	throw units_error((boost::format("Invalid operation: +%1%") % *this).str());
	return base_unit();	
}

const base_unit base_unit::operator-(const base_unit& other) const
{
	throw units_error((boost::format("Invalid operation: %1% - %2%") % *this % other).str());
	return base_unit();	
}

const base_unit base_unit::operator-(double value) const
{
	throw units_error((boost::format("Invalid operation: %1% - %2%") % *this % value).str());
	return base_unit();	
}

const base_unit base_unit::operator-(void) const
{
	throw units_error((boost::format("Invalid operation: -%1%") % *this).str());
	return base_unit();	
}

std::string base_unit::toString(bool bUnitsOnly) const
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
		return (boost::format("%.10g [%s]") % multiplier % _units).str();
}

std::ostream& operator<<(std::ostream& out, const base_unit& u)
{
	out << u.toString();
	return out;
}
	
const base_unit operator*(double value, const base_unit& self)
{
	return self * value;
}

const base_unit operator/(double value, const base_unit& self)
{
	return value * (self ^ (-1.0));
}

const base_unit operator+(double value, const base_unit& self)
{
	throw units_error((boost::format("Invalid operation: %1% + %2%") % value % self).str());
	return base_unit();	
}

const base_unit operator-(double value, const base_unit& self)
{
	throw units_error((boost::format("Invalid operation: %1% - %2%") % value % self).str());
	return base_unit();	
}

const base_unit pow(const base_unit& self, double exponent)
{
	return self ^ exponent;
}

const base_unit pow(const base_unit& self, const base_unit& exponent)
{
	return self ^ exponent;
}

/**************************************************************
		unit
**************************************************************/
unit::unit(std::string u1, double exp1,
	 std::string u2, double exp2,
	 std::string u3, double exp3,
	 std::string u4, double exp4,
	 std::string u5, double exp5,
	 std::string u6, double exp6,
	 std::string u7, double exp7,
	 std::string u8, double exp8)
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

unit::unit(const std::map<std::string, double>& mapUnits)
{
	setUnits(mapUnits);
}

void unit::addUnit(std::string name, double exp)
{
	if(name.empty())
		return;
	std::map<std::string, base_unit>& __base_units__ = get_base_units();
	std::map<std::string, base_unit>::const_iterator iter = __base_units__.find(name);
	if(iter == __base_units__.end())
		throw units_error((boost::format("Cannot find the base_unit: %1%") % name).str());
	units[name] = exp;
}

void unit::setUnits(const std::map<std::string, double>& mapUnits)
{
	units.clear();
	for(std::map<std::string, double>::const_iterator iter = mapUnits.begin(); iter != mapUnits.end(); iter++)
		addUnit(iter->first, iter->second);
}

const base_unit unit::getBaseUnit(void) const
{
	double exp;
	std::string name;
	base_unit tmp(1.0, 0, 0, 0, 0, 0, 0, 0);
	std::map<std::string, base_unit>& __base_units__ = get_base_units();
	
	for(std::map<std::string, double>::const_iterator iter = units.begin(); iter != units.end(); iter++)
	{
		name = (*iter).first;
		exp  = (*iter).second;
		if(__base_units__.find(name) == __base_units__.end())
			throw units_error((boost::format("Cannot find the base_unit: %1%") % name).str());
		tmp = tmp * (__base_units__[name] ^ exp);
	}
	return tmp;
}

bool unit::operator==(const unit& other) const
{
	return (getBaseUnit() == other.getBaseUnit());
}

bool unit::operator!=(const unit& other) const
{
	return !(*this == other);
}

const unit unit::operator*(const unit& other) const
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

const unit unit::operator/(const unit& other) const
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

const unit unit::operator^(double exponent) const
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

const unit unit::operator^(const unit& other) const
{
    throw units_error((boost::format("Invalid operation: %1% ** %2%") % *this % other).str());
	return unit();	
}

const unit unit::operator+(const unit& other) const
{
	throw units_error((boost::format("Invalid operation: %1% + %2%") % *this % other).str());
	return unit();	
}

const unit unit::operator+(double value) const
{
	throw units_error((boost::format("Invalid operation: %1% + %2%") % *this % value).str());
	return unit();	
}

const unit unit::operator+(void) const
{
	throw units_error((boost::format("Invalid operation: +%1%") % *this).str());
	return unit();	
}

const unit unit::operator-(const unit& other) const
{
	throw units_error((boost::format("Invalid operation: %1% - %2%") % *this % other).str());
	return unit();	
}

const unit unit::operator-(double value) const
{
	throw units_error((boost::format("Invalid operation: %1% - %2%") % *this % value).str());
	return unit();	
}

const unit unit::operator-(void) const
{
	throw units_error((boost::format("Invalid operation: -%1%") % *this).str());
	return unit();	
}

std::string unit::toString(void) const
{
    int noNegative, noPositive;
    std::map<std::string, double>::const_iterator iter;

    // If it is an empty return an empty string
    if(units.empty())
        return std::string("");

    noNegative = 0;
    noPositive = 0;
    for(iter = units.begin(); iter != units.end(); iter++)
	{
        if((*iter).second > 0)
            noPositive += 1;
		else
            noNegative += 1;
    }

    if(noPositive == 0 && noNegative == 0)
    {
        // If all are zero return an empty string
        return std::string("");
    }
    else if(noPositive == 0 && noNegative > 0)
    {
        // All exponents are negative
        std::vector<std::string> arrNegative;
        for(iter = units.begin(); iter != units.end(); iter++)
        {
            std::string name = (*iter).first;
            double      exp  = (*iter).second;
            to_string_and_add(name, exp, arrNegative);
        }
        return boost::algorithm::join(arrNegative, __string_unit_delimiter__);
    }
    else
    {
        // There are positive and (perhaps) negative exponents
        std::string strPositive, strNegative;
        std::vector<std::string> arrPositive, arrNegative;

        for(iter = units.begin(); iter != units.end(); iter++)
        {
            std::string name = (*iter).first;
            double      exp  = (*iter).second;
            if(exp > 0)
                to_string_and_add(name, exp, arrPositive);
            else
                to_string_and_add(name, ::abs(exp), arrNegative);
        }

        strPositive = boost::algorithm::join(arrPositive, __string_unit_delimiter__);
        strNegative = boost::algorithm::join(arrNegative, __string_unit_delimiter__);

    // If there are multiple units with negative exponents wrap them into ()
        if(arrNegative.size() > 1)
            strNegative = "(" + strNegative + ")";

    // If there is positive but not negative return only positive; otherwise pos/neg
        if(arrNegative.size() == 0)
            return (boost::format("%1%") % strPositive).str();
        else
            return (boost::format("%1%/%2%") % strPositive % strNegative).str();
    }
}

std::string unit::toLatex(void) const
{
	std::string strResult;
	
	for(std::map<std::string, double>::const_iterator iter = units.begin(); iter != units.end(); iter++)
	{
		std::string name = (*iter).first;
		double      exp  = (*iter).second;
		if(exp == 1)
			strResult += (boost::format("{%1%}") % name).str();
		else
			strResult += (boost::format("{%1% ^ %2%}") % name % exp).str();
	}
	return "{" + strResult + "}"; 
}

std::string unit::toJSON(void) const
{
    std::string strJSON = "{ ";
    for(std::map<std::string, double>::const_iterator iter = units.begin(); iter != units.end(); iter++)
        strJSON += ( boost::format("%s\"%s\" : %f") % (iter == units.begin() ? "" : ", ") % iter->first % iter->second ).str();
    strJSON += " }";
    return strJSON;
}

std::ostream& operator<<(std::ostream& out, const unit& u)
{
	out << u.toString();
	return out;
}

std::map<std::string, base_unit>& unit::get_base_units(void)
{
	static create_base_units cbu;
	return cbu.__base_units__;
}

const quantity unit::operator*(double multi) const
{
    return quantity(multi, *this);
}

const quantity operator*(double multi, const unit& self)
{
    return quantity(multi, self);
}

const quantity unit::operator/(double multi) const
{
    return quantity(1.0 / multi, *this);
}

const quantity operator/(double multi, const unit& self)
{
    return quantity(multi, self ^ (-1));
}

const unit operator+(double value, const unit& self)
{
	throw units_error((boost::format("Invalid operation: %1% + %2%") % value % self).str());
	return unit();	
}

const unit operator-(double value, const unit& self)
{
	throw units_error((boost::format("Invalid operation: %1% - %2%") % value % self).str());
	return unit();	
}

const unit pow(const unit& self, const unit& other)
{
	return self ^ other;
}

const unit pow(const unit& self, double exponent)
{
	return self ^ exponent;
}

}
