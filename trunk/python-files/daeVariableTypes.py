from pyCore import *

length_t                     = daeVariableType("length_t",                     "m",             0.0, 1.0e+05,     1.0, 1e-05)
area_t                       = daeVariableType("area_t",                       "m2",            0.0, 1.0e+05,     1.0, 1e-05)
volume_t                     = daeVariableType("volume_t",                     "m3",            0.0, 1.0e+05,     1.0, 1e-06)
velocity_t                   = daeVariableType("velocity_t",                   "m/s",      -1.0e+10, 1.0e+10,     0.0, 1e-05)
pressure_t                   = daeVariableType("pressure_t",                   "Pa",        1.0e+02, 1.0e+10, 1.0e+05, 1e-05)
temperature_t                = daeVariableType("temperature_t",                "K",             0.0, 1.0e+05,     300, 1e-05)

fraction_t                   = daeVariableType("fraction_t",                   "-",        -1.0e-10, 1.1e+00,     0.1, 1e-06)
no_t                         = daeVariableType("no_t",                         "-",        -1.0e+20, 1.0e+20,     0.0, 1e-05)

moles_t                      = daeVariableType("moles_t",                      "mol",           0.0, 1.0e+20,     1.0, 1e-05)
molar_flux_t                 = daeVariableType("molar_flux_t",                 "mol/m2",   -1.0e+20, 1.0e+20,     0.0, 1e-05)
molar_concentration_t        = daeVariableType("molar_concentration_t",        "mol/m3",        0.0, 1.0e+10,     1.0, 1e-05)
molar_flowrate_t             = daeVariableType("molar_flowrate_t",             "mol/s",    -1.0e+10, 1.0e+10,     1.0, 1e-05)

heat_t                       = daeVariableType("heat_t",                       "J",        -1.0e+20, 1.0e+20, 1.0e+03, 1e-05)
heat_flux_t                  = daeVariableType("heat_flux_t",                  "W/m2",     -1.0e+20, 1.0e+20,     0.0, 1e-05)
heat_transfer_coefficient_t  = daeVariableType("heat_transfer_coefficient_t",  "W/(m2.K)",        0, 1.0e+10, 1.0e+02, 1e-05)
power_t                      = daeVariableType("power_t",                      "W",        -1.0e+20, 1.0e+20,     0.0, 1e-05)

specific_heat_capacity_t     = daeVariableType("specific_heat_capacity_t",     "J/(kg.K)",      0.0, 1.0e+20, 1.0e+03, 1e-05)
density_t                    = daeVariableType("density_t",                    "kg/m3",         0.0, 1.0e+20, 1.0e+03, 1e-05)
specific_heat_conductivity_t = daeVariableType("specific_heat_conductivity_t", "W/(m.K)",       0.0, 1.0e+10,     1.0, 1e-05)
dynamic_viscosity_t          = daeVariableType("dynamic_viscosity_t",          "Pa.s",          0.0, 1.0e+10, 1.0e-03, 1e-06)
diffusivity_t                = daeVariableType("diffusivity_t",                "m2/s",          0.0, 1.0e+05, 1.0e-03, 1e-08)

amount_adsorbed_t            = daeVariableType("amount_adsorbed_t",            "mol/kg",   -1.0e+20, 1.0e+20,     0.0, 1e-05)
