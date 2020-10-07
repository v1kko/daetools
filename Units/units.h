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
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

#ifdef DAE_DLL_INTERFACE
#ifdef UNITS_EXPORTS
#define DAE_UNITS_API __declspec(dllexport)
#else
#define DAE_UNITS_API __declspec(dllimport)
#endif
#else
#define DAE_UNITS_API
#endif

#else // WIN32
#define DAE_UNITS_API
#endif // WIN32

/**************************************************************
  units_error
**************************************************************/
class DAE_UNITS_API units_error : public std::runtime_error
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
const std::string __string_unit_power__      = "**";

class DAE_UNITS_API base_unit
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
DAE_UNITS_API std::ostream& operator<<(std::ostream& out, const base_unit& u);

DAE_UNITS_API const base_unit operator*(double value, const base_unit& self);
DAE_UNITS_API const base_unit operator/(double value, const base_unit& self);
DAE_UNITS_API const base_unit operator+(double value, const base_unit& self);
DAE_UNITS_API const base_unit operator-(double value, const base_unit& self);
DAE_UNITS_API const base_unit pow      (const base_unit& self, double exponent);
DAE_UNITS_API const base_unit pow      (const base_unit& self, const base_unit& exponent);

/**************************************************************
        unit
**************************************************************/
class quantity;
class DAE_UNITS_API unit
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

    size_t SizeOf() const;
    size_t GetHash() const;

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

    std::string toString() const;
    std::string toLatex() const;
    std::string toJSON() const;

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

DAE_UNITS_API std::ostream& operator<<(std::ostream& out, const unit& u);

DAE_UNITS_API const quantity operator*(double value, const unit& self);
DAE_UNITS_API const quantity operator/(double value, const unit& self);
DAE_UNITS_API const unit     operator+(double value, const unit& self);
DAE_UNITS_API const unit     operator-(double value, const unit& self);
DAE_UNITS_API const unit     pow      (const unit& self, double exponent);
DAE_UNITS_API const unit     pow      (const unit& self, const unit& exponent);

/**************************************************************
        quantity
**************************************************************/
class DAE_UNITS_API quantity
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

    std::string toString() const;
    std::string toLatex() const;
    std::string toJSON() const;

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

    size_t SizeOf() const;
    size_t GetHash() const;

protected:
    std::shared_ptr<unit> _units;
    double                  _value;
};

DAE_UNITS_API std::ostream& operator<<(std::ostream& out, const quantity& q);
DAE_UNITS_API std::size_t hash_value(unit const& u);
DAE_UNITS_API std::size_t hash_value(quantity const& q);

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

DAE_UNITS_API const quantity operator*(double value, const quantity& self);
DAE_UNITS_API const quantity operator/(double value, const quantity& self);
DAE_UNITS_API const quantity operator+(double value, const quantity& self);
DAE_UNITS_API const quantity operator-(double value, const quantity& self);
DAE_UNITS_API const quantity operator^(double value, const quantity& self);
DAE_UNITS_API const quantity pow      (double value, const quantity& exponent);
DAE_UNITS_API const quantity pow      (const quantity& self, double exponent);
DAE_UNITS_API const quantity pow      (const quantity& self, const quantity& exponent);

DAE_UNITS_API bool operator ==(double value, const quantity& self);
DAE_UNITS_API bool operator !=(double value, const quantity& self);
DAE_UNITS_API bool operator <=(double value, const quantity& self);
DAE_UNITS_API bool operator >=(double value, const quantity& self);
DAE_UNITS_API bool operator > (double value, const quantity& self);
DAE_UNITS_API bool operator < (double value, const quantity& self);

DAE_UNITS_API const quantity exp(const quantity &q);
DAE_UNITS_API const quantity log(const quantity &q);
DAE_UNITS_API const quantity log10(const quantity &q);
DAE_UNITS_API const quantity sqrt(const quantity &q);
DAE_UNITS_API const quantity sin(const quantity &q);
DAE_UNITS_API const quantity cos(const quantity &q);
DAE_UNITS_API const quantity tan(const quantity &q);
DAE_UNITS_API const quantity asin(const quantity &q);
DAE_UNITS_API const quantity acos(const quantity &q);
DAE_UNITS_API const quantity atan(const quantity &q);

DAE_UNITS_API const quantity sinh(const quantity &q);
DAE_UNITS_API const quantity cosh(const quantity &q);
DAE_UNITS_API const quantity tanh(const quantity &q);
DAE_UNITS_API const quantity asinh(const quantity &q);
DAE_UNITS_API const quantity acosh(const quantity &q);
DAE_UNITS_API const quantity atanh(const quantity &q);
DAE_UNITS_API const quantity atan2(const quantity &a, const quantity &b);
DAE_UNITS_API const quantity erf(const quantity &q);

DAE_UNITS_API const quantity ceil(const quantity &q);
DAE_UNITS_API const quantity floor(const quantity &q);

DAE_UNITS_API const quantity abs(const quantity &q);
DAE_UNITS_API const quantity max(const quantity &a, const quantity &b);
DAE_UNITS_API const quantity max(double v, const quantity &q);
DAE_UNITS_API const quantity max(const quantity &q, double v);
DAE_UNITS_API const quantity min(const quantity &a, const quantity &b);
DAE_UNITS_API const quantity min(double v, const quantity &q);
DAE_UNITS_API const quantity min(const quantity &q, double v);

class DAE_UNITS_API create_base_units
{
public:
    create_base_units();

public:
    std::map<std::string, base_unit> __base_units__;
};

}

#endif // UNITS_H
