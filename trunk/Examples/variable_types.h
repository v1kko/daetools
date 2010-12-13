#ifndef DAE_EXAMPLE_VARIABLE_TYPES_H
#define DAE_EXAMPLE_VARIABLE_TYPES_H

#include "../dae_develop.h"

const daeVariableType length			("length",				"m",         0.0,      1.0e+05,   0.0,      1e-05);
const daeVariableType molar_conc		("molar_conc",			"kmol/m3",  -1.0e+10,  1.0e+10,  10.0,      1e-05);
const daeVariableType molar_flowrate	("molar_flowrate",		"kmol/s",   -1.0e+20,  1.0e+20,   0.0,      1e-05);
const daeVariableType fraction			("fraction",			"-",        -1.0e-10,  1.01,      0.5,      1e-05);
const daeVariableType moles				("moles",				"mol",       0.0,      1.0e+20,   1.0,      1e-05);
const daeVariableType no_type			("no_type",				"-",        -1.0e+100, 1.0e+100,  1.0,      1e-05);
const daeVariableType volume			("volume",				"m3",        0.0,      1.0e+05,   0.1,      1e-05);
const daeVariableType temperature		("temperature",			"K",         0.0,      1.0e+05,   300,      1e-05);
const daeVariableType heat				("heat",				"J",		-1.0e+20,  1.0e+20,  1000,      1e-05);
const daeVariableType heat_flux			("heat flux",			"W/m2",     -1.0e+20,  1.0e+20,     0,      1e-05);
const daeVariableType heat_tr_coeff		("heat transfer coeff",	"W/m2K",     0,        1.0e+20,     100,    1e-05);
const daeVariableType power				("power",				"W",        -1.0e+20,  1.0e+20,     0,      1e-05);
const daeVariableType heat_capacity		("heat capacity",		"J/kgK",    -1.0e+20,  1.0e+20,  1000,      1e-05);
const daeVariableType density			("density",				"kg/m3",    -1.0e+20,  1.0e+20,  1000,      1e-05);
const daeVariableType conductivity		("conductivity",		"W/mK",     -1.0e+20,  1.0e+20,     1,      1e-05);

#endif
