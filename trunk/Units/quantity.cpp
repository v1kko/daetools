#include "units.h"
#include <algorithm>

namespace units
{
create_base_units::create_base_units()
{
	std::map<std::string, base_unit> __scaled_units__;
	
	const double yotta = 1E+24;
	const double zetta = 1E+21;
	const double exa   = 1E+18;
	const double peta  = 1E+15;
	const double tera  = 1E+12;
	const double giga  = 1E+9;
	const double mega  = 1E+6;
	const double kilo  = 1E+3;
	const double hecto = 1E+2;
	const double deka  = 1E+1;
	const double deci  = 1E-1;
	const double centi = 1E-2;
	const double milli = 1E-3;
	const double micro = 1E-6;
	const double nano  = 1E-9;
	const double pico  = 1E-12;
	const double femto = 1E-15;
	const double atto  = 1E-18;
	const double zepto = 1E-21;
	const double yocto = 1E-24;
	
	base_unit dimless = base_unit(1.0, 0, 0, 0, 0, 0, 0, 0);
	
	base_unit m   = base_unit(1.0, 1, 0, 0, 0, 0, 0, 0);
	base_unit kg  = base_unit(1.0, 0, 1, 0, 0, 0, 0, 0);
	base_unit s   = base_unit(1.0, 0, 0, 1, 0, 0, 0, 0);
	base_unit cd  = base_unit(1.0, 0, 0, 0, 1, 0, 0, 0);
	base_unit A   = base_unit(1.0, 0, 0, 0, 0, 1, 0, 0);
	base_unit K   = base_unit(1.0, 0, 0, 0, 0, 0, 1, 0);
	base_unit mol = base_unit(1.0, 0, 0, 0, 0, 0, 0, 1);
	
	// Angles
	base_unit rad = base_unit(1.0, 0, 0, 0, 0, 0, 0, 0); // Radian
	base_unit sr  = base_unit(1.0, 0, 0, 0, 0, 0, 0, 0); // Steradian
	
    // Mass
    base_unit g    = milli * kg;
    base_unit t    = kilo  * kg;

    // Time
	base_unit min  = 60    * s;
	base_unit hour = 3600  * s;
	base_unit day  = 43200 * s;
	
	// Frequency
	base_unit Hz   = s^(-1);
	
	// Length related:
	base_unit cm = centi * m;
	
	// Volume:
	base_unit l  = 1E-3 * (m^3);
	base_unit dl = deci * l;
    base_unit ml = milli * l;
	
	// Energy:
	base_unit N  = kg * m / (s^2); // Newtown
	base_unit J  = N * m;          // Joule
	base_unit W  = J / s;          // Watt
	
	// Electromagnetism:
	base_unit C   = A * s;                    // Coulomb
	base_unit Ohm = J * s / (C^2);            // Ohm
	base_unit V   = kg * (m^2) / (A * (s^3)); // Volt
	base_unit F   = C / V;                    // Farad
	base_unit T   = kg / ((s^2) * A);         // Tesla
	base_unit H   = V * s / A;                // Henry
	base_unit S   = A / V;                    // Siemens
	base_unit Wb  = T * (m^2);                // Weber
	
	// Pressure:
	base_unit Pa  = N / (m^2); // Pascal
	
	// Viscosity
	base_unit P  = Pa * s;       // Poise
	base_unit St = (cm^2) / s;   // Stoke
	
	// Radioactivity
	base_unit Bq  = s^(-1);     // Radioactivity, Becquerel
	base_unit Gy  = J / kg;     // Absorbed Dose, Gray
	base_unit Sv  = J / kg;     // Equivalent Dose, Sievert
	
	// Optics
	base_unit lx = cd * sr / (m^2); // Illuminance, Lux
	base_unit lm = cd * sr;         // Luminous Flux, Lumen
	
	// Catalytic Activity
	base_unit kat = mol / s;  // Katal
	
	// Temporary used SI units
	base_unit knot = (1852.0/3600.0) * m / s; // Knot
	base_unit bar  = 1E5 * Pa;                // Bar
	base_unit b    = 1E-28 * (m^2);           // Barn
	base_unit Ci   = 3.7E10 * Bq;             // Curie
	base_unit R    = 2.58E-4 * C / kg;        // Roentgen
	base_unit rd   = 0.01 * Gy;               // Rad (absorbed radiation dose)
	base_unit rem	 = 0.01 * Sv;               // Rem (Roentgen equivalent man)

	__base_units__["m"] = m;
	__base_units__["s"] = s;
	__base_units__["cd"] = cd;
	__base_units__["A"] = A;
	__base_units__["mol"] = mol;

	__base_units__["Hz"] = Hz;
	
	__base_units__["N"] = N;
	__base_units__["J"] = J;
	__base_units__["W"] = W;
	
	__base_units__["V"] = V;
	__base_units__["C"] = C;
	__base_units__["F"] = F;
	__base_units__["Ohm"] = Ohm;
	__base_units__["T"]  = T;
	__base_units__["H"]  = H;
	__base_units__["S"]  = S;
	__base_units__["Wb"] = Wb;
	
	__base_units__["Pa"] = Pa;
	
	__base_units__["P"]  = P;
	__base_units__["St"] = St;

	__base_units__["Bq"]  = Bq;
	__base_units__["Gy"]  = Gy;
	__base_units__["Sv"]  = Sv;
	
	__base_units__["lx"] = lx;
	__base_units__["lm"] = lm;
	
	__base_units__["kat"] = kat;
	
	__base_units__["knot"] = knot;
	__base_units__["bar"]  = bar;
	__base_units__["b"]    = b;
	__base_units__["Ci"]   = Ci;
	__base_units__["R"]    = R;
	__base_units__["rd"]   = rd;
	__base_units__["rem"]  = rem;
	
	for(std::map<std::string, base_unit>::iterator iter = __base_units__.begin(); iter != __base_units__.end(); iter++)
	{
		__scaled_units__["Y"  + (*iter).first] = yotta * base_unit((*iter).second);
		__scaled_units__["Z"  + (*iter).first] = zetta * base_unit((*iter).second);
		__scaled_units__["E"  + (*iter).first] = exa   * base_unit((*iter).second);
		__scaled_units__["P"  + (*iter).first] = peta  * base_unit((*iter).second);
		__scaled_units__["T"  + (*iter).first] = tera  * base_unit((*iter).second);
		__scaled_units__["G"  + (*iter).first] = giga  * base_unit((*iter).second);
		__scaled_units__["M"  + (*iter).first] = mega  * base_unit((*iter).second);
		__scaled_units__["k"  + (*iter).first] = kilo  * base_unit((*iter).second);
		__scaled_units__["h"  + (*iter).first] = hecto * base_unit((*iter).second);
		__scaled_units__["da" + (*iter).first] = deka  * base_unit((*iter).second);
		__scaled_units__["d"  + (*iter).first] = deci  * base_unit((*iter).second);
		__scaled_units__["c"  + (*iter).first] = centi * base_unit((*iter).second);
		__scaled_units__["m"  + (*iter).first] = milli * base_unit((*iter).second);
		__scaled_units__["u"  + (*iter).first] = micro * base_unit((*iter).second);
		__scaled_units__["n"  + (*iter).first] = nano  * base_unit((*iter).second);
		__scaled_units__["p"  + (*iter).first] = pico  * base_unit((*iter).second);
		__scaled_units__["f"  + (*iter).first] = femto * base_unit((*iter).second);
		__scaled_units__["a"  + (*iter).first] = atto  * base_unit((*iter).second);
		__scaled_units__["z"  + (*iter).first] = zepto * base_unit((*iter).second);
		__scaled_units__["y"  + (*iter).first] = yocto * base_unit((*iter).second);
	}
	__base_units__.insert(__scaled_units__.begin(), __scaled_units__.end());
	
	__base_units__["kg"] = kg;
    __base_units__["g"]  = g;
    __base_units__["t"]  = t;

    __base_units__["K"]  = K;
	
	__base_units__["rad"] = rad;
	__base_units__["sr"]  = sr;
	
	__base_units__["min"]  = min;
	__base_units__["hour"] = hour;
	__base_units__["day"]  = day;

	__base_units__["l"]  = l;
	__base_units__["dl"] = dl;
    __base_units__["ml"] = ml;
	
//	for(std::map<std::string, base_unit>::iterator iter = __base_units__.begin(); iter != __base_units__.end(); iter++)
//		std::cout << (*iter).first << " = " << (*iter).second << std::endl;
}

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

const quantity quantity::scaleTo(const quantity& referrer) const
{
	return scaleTo(referrer.getUnits());
}

const quantity quantity::scaleTo(const unit& referrer) const
{
	if(!_units.getBaseUnit().areDimensionsEqual(referrer.getBaseUnit()))
		throw units_error((boost::format("Units not consistent: scale from %1% to %2%") % _units % referrer).str());

	quantity tmp;
	tmp.setUnits(referrer);
	tmp.setValue(_value * _units.getBaseUnit().multiplier / referrer.getBaseUnit().multiplier);
	return tmp;
}

std::string quantity::toString(void) const
{
	return (boost::format("%.10g %s") % _value % _units.toString()).str();
}

std::string quantity::toLatex(void) const
{
	return (boost::format("{%.10g %s}") % _value % _units.toLatex()).str();
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

const quantity quantity::operator+(const quantity& other) const
{
	if(_units != other.getUnits())
		throw units_error((boost::format("Units not consistent: %1% + %2%") % _units % other.getUnits()).str());
	
	quantity tmp(_value + other.scaleTo(*this).getValue(), _units);
	return tmp;	
}

const quantity quantity::operator+(double value) const
{
	quantity q(value, unit());
	return (*this + q);
}

const quantity quantity::operator+(void) const
{
	quantity tmp(_value, _units);
	return tmp;	
}

const quantity quantity::operator-(const quantity& other) const
{
	if(_units != other.getUnits())
		throw units_error((boost::format("Units not consistent: %1% - %2%") % _units % other.getUnits()).str());
	
	quantity tmp(_value - other.scaleTo(*this).getValue(), _units);
	return tmp;	
}

const quantity quantity::operator-(double value) const
{
	quantity q(value, unit());
	return (*this - q);
}

const quantity quantity::operator-(void) const
{
	quantity tmp(-_value, _units);
	return tmp;	
}

const quantity quantity::operator*(const quantity& other) const
{
	quantity tmp;
	tmp.setUnits(_units * other.getUnits());
	tmp.setValue(_value * other.getValue());
	return tmp;	
}

const quantity quantity::operator*(const unit& other) const
{
	quantity tmp;
	tmp.setUnits(_units * other);
	tmp.setValue(_value);
	return tmp;	
}

const quantity quantity::operator*(double value) const
{
	quantity q(value, unit());
	return (*this * q);
}

const quantity quantity::operator/(const quantity& other) const
{
	quantity tmp;
	tmp.setUnits(_units / other.getUnits());
	tmp.setValue(_value / other.getValue());
	return tmp;	
}

const quantity quantity::operator/(const unit& other) const
{
	quantity tmp;
	tmp.setUnits(_units / other);
	tmp.setValue(_value);
	return tmp;	
}

const quantity quantity::operator/(double value) const
{
	quantity q(value, unit());
	return (*this / q);
}

const quantity quantity::operator^(const quantity& other) const
{
	if(other.getUnits() != unit())
		throw units_error((boost::format("Exponent must be dimension-less in: %1% ^ %2%") % _units % other.getUnits()).str());
	return (*this ^ other.getValueInSIUnits());
}

const quantity quantity::operator^(double exponent) const
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

const quantity operator*(double value, const quantity& self)
{
	return self * value;	
}

const quantity operator/(double value, const quantity& self)
{
	return quantity(value, self.getUnits() ^ (-1));	
}

const quantity operator+(double value, const quantity& self)
{
	quantity q(value, unit());
	return q + self;
}

const quantity operator-(double value, const quantity& self)
{
	quantity q(value, unit());
	return q - self;
}

const quantity operator^(double value, const quantity& self)
{
	quantity q(value, unit());
	return q ^ self;	
}

const quantity pow(double value, const quantity& exponent)
{
	quantity q(value, unit());
	return q ^ exponent;	
}

const quantity pow(const quantity& self, double exponent)
{
	return (self ^ exponent);	
}

const quantity pow(const quantity& self, const quantity& exponent)
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

const quantity exp(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: exp(%1%)") % q).str());
	return quantity(::exp(q.getValue()), unit());
}

const quantity log(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: log(%1%)") % q).str());
	return quantity(::log(q.getValue()), unit());
}

const quantity log10(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: log10(%1%)") % q).str());
	return quantity(::log10(q.getValue()), unit());
}

const quantity sqrt(const quantity &q)
{
	return q ^ 0.5;
}

const quantity sin(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: sin(%1%)") % q).str());
	return quantity(::sin(q.getValue()), unit());
}

const quantity cos(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: cos(%1%)") % q).str());
	return quantity(::cos(q.getValue()), unit());
}

const quantity tan(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: tan(%1%)") % q).str());
	return quantity(::tan(q.getValue()), unit());
}

const quantity asin(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: asin(%1%)") % q).str());
	return quantity(::asin(q.getValue()), unit());
}

const quantity acos(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: acos(%1%)") % q).str());
	return quantity(::acos(q.getValue()), unit());
}

const quantity atan(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: atan(%1%)") % q).str());
	return quantity(::atan(q.getValue()), unit());
}

const quantity sinh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: sinh(%1%)") % q).str());
	return quantity(::sinh(q.getValue()), unit());
}

const quantity cosh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: cosh(%1%)") % q).str());
	return quantity(::cosh(q.getValue()), unit());
}

const quantity tanh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: tanh(%1%)") % q).str());
	return quantity(::tanh(q.getValue()), unit());
}

// Some versions of stupid MS <math.h> do not have asinh, acosh, atanh
const quantity asinh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: asinh(%1%)") % q).str());

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
	return quantity(::log(q.getValue() + ::sqrt(q.getValue()*q.getValue() + 1)), unit());
#else
	return quantity(::asinh(q.getValue()), unit());
#endif
}

const quantity acosh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: acosh(%1%)") % q).str());

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
	return quantity(::log(q.getValue() + ::sqrt(q.getValue()*q.getValue() - 1)), unit());
#else
	return quantity(::acosh(q.getValue()), unit());
#endif
}
        
const quantity atanh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: atanh(%1%)") % q).str());

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
	return quantity((::log(1 + q.getValue()) - ::log(1 - q.getValue())) / 2.0 , unit());
#else
	return quantity(::atanh(q.getValue()), unit());
#endif
}

/*
const quantity asinh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: asinh(%1%)") % q).str());
	return quantity(::asinh(q.getValue()), unit());
}

const quantity acosh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: acosh(%1%)") % q).str());
	return quantity(::acosh(q.getValue()), unit());
}

const quantity atanh(const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: atanh(%1%)") % q).str());
	return quantity(::atanh(q.getValue()), unit());
}
*/
const quantity atan2(const quantity &a, const quantity &b)
{
	if(a.getUnits() != unit() || b.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: atan2(%1%, %2%)") % a % b).str());
	return quantity(::atan2(a.getValue(), b.getValue()), unit());
}

const quantity ceil(const quantity &q)
{
	return quantity(::ceil(q.getValue()), q.getUnits());
}

const quantity floor(const quantity &q)
{
	return quantity(::floor(q.getValue()), q.getUnits());
}

const quantity abs(const quantity &q)
{
	return quantity(::abs(q.getValue()), q.getUnits());
}

const quantity max(const quantity &a, const quantity &b)
{
	if(a.getUnits() != b.getUnits())	
		throw units_error((boost::format("Invalid units in function: max(%1%, %2%)") % a % b).str());
	return quantity(std::max(a.getValue(), b.getValue()), a.getUnits());
}

const quantity max(double v, const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: max(%1%, %2%)") % v % q).str());
	return quantity(std::max(v, q.getValue()), unit());
}

const quantity max(const quantity &q, double v)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: max(%1%, %2%)") % v % q).str());
	return quantity(std::max(v, q.getValue()), unit());
}

const quantity min(const quantity &a, const quantity &b)
{
	if(a.getUnits() != b.getUnits())	
		throw units_error((boost::format("Invalid units in function: min(%1%, %2%)") % a % b).str());
	return quantity(std::min(a.getValue(), b.getValue()), a.getUnits());
}

const quantity min(double v, const quantity &q)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: min(%1%, %2%)") % v % q).str());
	return quantity(std::min(v, q.getValue()), unit());
}

const quantity min(const quantity &q, double v)
{
	if(q.getUnits() != unit())	
		throw units_error((boost::format("Invalid units in function: min(%1%, %2%)") % v % q).str());
	return quantity(std::min(v, q.getValue()), unit());
}


}
