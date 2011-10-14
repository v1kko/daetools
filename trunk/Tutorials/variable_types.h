#ifndef DAE_EXAMPLE_VARIABLE_TYPES_H
#define DAE_EXAMPLE_VARIABLE_TYPES_H

#include "../dae_develop.h"
namespace
{
using namespace units::units_pool;

const daeVariableType length_t                    ("length_t",                     m,              0.0, 1.0e+05,     1.0, 1e-05);
const daeVariableType area_t                      ("area_t",                       m^2,            0.0, 1.0e+05,     1.0, 1e-05);
const daeVariableType volume_t                    ("volume_t",                     m^3,            0.0, 1.0e+05,     1.0, 1e-06);
const daeVariableType velocity_t                  ("velocity_t",                   m/s,       -1.0e+10, 1.0e+10,     0.0, 1e-05);
const daeVariableType pressure_t                  ("pressure_t",                   Pa,         1.0e+02, 1.0e+10, 1.0e+05, 1e-05);
const daeVariableType temperature_t               ("temperature_t",                K,              0.0, 1.0e+05,     300, 1e-05);

const daeVariableType fraction_t                  ("fraction_t",                   unit(),    -1.0e-10, 1.1e+00,     0.1, 1e-06);
const daeVariableType no_t                        ("no_t",                         unit(),    -1.0e+20, 1.0e+20,     0.0, 1e-05);

const daeVariableType moles_t                     ("moles_t",                      mol,            0.0, 1.0e+20,     1.0, 1e-05);
const daeVariableType molar_flux_t                ("molar_flux_t",                 mol/(m^2), -1.0e+20, 1.0e+20,     0.0, 1e-05);
const daeVariableType molar_concentration_t       ("molar_concentration_t",        mol/(m^3),      0.0, 1.0e+10,     1.0, 1e-05);
const daeVariableType molar_flowrate_t            ("molar_flowrate_t",             mol/s,     -1.0e+10, 1.0e+10,     1.0, 1e-05);

const daeVariableType heat_t                      ("heat_t",                       J,         -1.0e+20, 1.0e+20, 1.0e+03, 1e-05);
const daeVariableType heat_flux_t                 ("heat_flux_t",                  W/(m^2),   -1.0e+20, 1.0e+20,     0.0, 1e-05);
const daeVariableType heat_transfer_coefficient_t ("heat_transfer_coefficient_t",  W/((m^2)*K),      0, 1.0e+10, 1.0e+02, 1e-05);
const daeVariableType power_t                     ("power_t",                      W,         -1.0e+20, 1.0e+20,     0.0, 1e-05);

const daeVariableType specific_heat_capacity_t    ("specific_heat_capacity_t",     J/(kg*K),       0.0, 1.0e+20, 1.0e+03, 1e-05);
const daeVariableType density_t                   ("density_t",                    kg/(m^3),       0.0, 1.0e+20, 1.0e+03, 1e-05);
const daeVariableType specific_heat_conductivity_t("specific_heat_conductivity_t", W/(m*K),        0.0, 1.0e+10,     1.0, 1e-05);
const daeVariableType dynamic_viscosity_t         ("dynamic_viscosity_t",          Pa*s,           0.0, 1.0e+10, 1.0e-03, 1e-06);
const daeVariableType diffusivity_t               ("diffusivity_t",                (m^2)/s,        0.0, 1.0e+05, 1.0e-03, 1e-08);
 
const daeVariableType amount_adsorbed_t           ("amount_adsorbed_t",            mol/kg,    -1.0e+20, 1.0e+20,     0.0, 1e-05);
 
}

#endif
