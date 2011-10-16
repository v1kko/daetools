#ifndef UNITS_POOL_H
#define UNITS_POOL_H

#include "units.h"

namespace units
{
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

}

#endif // UNITS_POOL_H
