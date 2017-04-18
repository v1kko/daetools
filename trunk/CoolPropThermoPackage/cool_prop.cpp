/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2015
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include <iostream>
#include "cool_prop.h"
#include "coolprop_thermo.h"

daeThermoPhysicalPropertyPackage_t* daeCreateCoolPropPropertyPackage()
{
    return new daeCoolPropThermoPhysicalPropertyPackage();
}

daeCoolPropThermoPhysicalPropertyPackage::daeCoolPropThermoPhysicalPropertyPackage()
{
}

daeCoolPropThermoPhysicalPropertyPackage::~daeCoolPropThermoPhysicalPropertyPackage()
{
}

void daeCoolPropThermoPhysicalPropertyPackage::LoadPackage(const std::string& strPackageManager,
                                                           const std::string& strPackageName,
                                                           const std::vector<std::string>& strarrCompoundIDs,
                                                           const std::vector<std::string>& strarrCompoundCASNumbers,
                                                           const std::map<std::string,daeeThermoPackagePhase>& mapAvailablePhases,
                                                           daeeThermoPackageBasis defaultBasis,
                                                           const std::map<std::string,std::string>& mapOptions)
{
/*
    Backends: "HEOS", "REFPROP", "INCOMP", "IF97", "TREND", "TTSE", "BICUBIC", "SRK", "PR", "VTPR"
    Reference states:  "IIR", "ASHRAE", "NBP", "DEF", "RESET"
*/

    m_strarrCompoundIDs        = strarrCompoundIDs;
    m_strarrCompoundCASNumbers = strarrCompoundCASNumbers;
    m_mapAvailablePhases       = mapAvailablePhases;
    m_defaultBasis             = defaultBasis;

    m_defaultBackend           = "";
    m_defaultReferenceState    = "";

    std::string key = "backend";
    if(mapOptions.find(key) != mapOptions.end())
        m_defaultBackend = mapOptions.at(key);

    key = "referenceState";
    if(mapOptions.find(key) != mapOptions.end())
        m_defaultReferenceState = mapOptions.at(key);

    key = "debugLevel";
    if(mapOptions.find(key) != mapOptions.end())
    {
        int dl = ::atoi(mapOptions.at(key).c_str());
        set_debug_level(dl);
    }

    m_mixture = shared_ptr<AbstractState>(AbstractState::factory(m_defaultBackend, m_strarrCompoundIDs));
}

std::string daeCoolPropThermoPhysicalPropertyPackage::GetTPPName()
{
    return std::string("CoolProp TPP");
}

// ICapeThermoCompounds interface
double daeCoolPropThermoPhysicalPropertyPackage::GetCompoundConstant(const std::string& capeOpenProperty, const std::string& compound)
{
    std::string msg = "CoolProp GetCompoundConstant function is not implemented";
    throw std::runtime_error(msg);
    return 0;
}

double daeCoolPropThermoPhysicalPropertyPackage::GetTDependentProperty(const std::string& capeOpenProperty, double T, const std::string& compound)
{
    std::string msg = "CoolProp PureCompoundTDProperty function is not implemented";
    throw std::runtime_error(msg);
    return 0;
}

double daeCoolPropThermoPhysicalPropertyPackage::GetPDependentProperty(const std::string& capeOpenProperty, double P, const std::string& compound)
{
    std::string msg = "CoolProp PureCompoundPDProperty function is not implemented";
    throw std::runtime_error(msg);
    return 0;
}

// ICapeThermoPropertyRoutine interface
double daeCoolPropThermoPhysicalPropertyPackage::CalcSinglePhaseScalarProperty(const std::string& capeOpenProperty,
                                                                               double P_,
                                                                               double T_,
                                                                               const std::vector<double>& x,
                                                                               const std::string& phase,
                                                                               daeeThermoPackageBasis basis)
{
/*
    std::vector< std::vector<double> > matResults;
    std::vector<double> T(1, T_), P(1, P_);
    std::vector<std::string> properties;
    std::string coolPropProperty;
    CapeOpen_to_CoolPropName(capeOpenProperty, basis, coolPropProperty);
    properties.push_back(coolPropProperty);

    matResults = PropsSImulti(properties, "T", T, "P", P, m_defaultBackend, m_strarrCompoundIDs, x);
    if(matResults.size() != 1 || matResults[0].size() != 1)
    {
        std::string msg = "Invalid number of results in CoolProp SinglePhaseScalarProperty function";
        throw std::runtime_error(msg);
    }
    return matResults[0][0];
*/
    std::map<std::string, daeeThermoPackagePhase>::iterator it = m_mapAvailablePhases.find(phase);
    if(it != m_mapAvailablePhases.end())
    {
        daeeThermoPackagePhase stateOfAggregation = it->second;
        if(stateOfAggregation == eVapor)
            m_mixture->specify_phase(iphase_gas);
        else if(stateOfAggregation == eLiquid)
            m_mixture->specify_phase(iphase_liquid);
    }
    m_mixture->set_mole_fractions(x);
    m_mixture->update(PT_INPUTS, P_, T_); // in SI units
    return GetScalarProperty(capeOpenProperty, basis);
}

void daeCoolPropThermoPhysicalPropertyPackage::CalcSinglePhaseVectorProperty(const std::string& capeOpenProperty,
                                                                             double P_,
                                                                             double T_,
                                                                             const std::vector<double>& x,
                                                                             const std::string& phase,
                                                                             std::vector<double>& results,
                                                                             daeeThermoPackageBasis basis)
{
    std::map<std::string, daeeThermoPackagePhase>::iterator it = m_mapAvailablePhases.find(phase);
    if(it != m_mapAvailablePhases.end())
    {
        daeeThermoPackagePhase stateOfAggregation = it->second;
        if(stateOfAggregation == eVapor)
            m_mixture->specify_phase(iphase_gas);
        else if(stateOfAggregation == eLiquid)
            m_mixture->specify_phase(iphase_liquid);
    }
    m_mixture->set_mole_fractions(x);
    m_mixture->update(PT_INPUTS, P_, T_); // in SI units
    GetVectorProperty(capeOpenProperty, basis, results);
}

double daeCoolPropThermoPhysicalPropertyPackage::CalcTwoPhaseScalarProperty(const std::string& capeOpenProperty,
                                                                            double P1,
                                                                            double T1,
                                                                            const std::vector<double>& x1,
                                                                            const std::string& phase1,
                                                                            double P2,
                                                                            double T2,
                                                                            const std::vector<double>& x2,
                                                                            const std::string& phase2,
                                                                            daeeThermoPackageBasis basis)
{
// CoolProp does not use T,P,x for both phases but calculates the equilibrium based on T,P,x of the mixture.
// To check: how to fit this into the Cape-Open interface?
//           temporary solution is to use only T1,P1,x1 (the data for the first phase).
    std::string msg = "CoolProp CalcTwoPhaseScalarProperty function is not implemented";
    throw std::runtime_error(msg);
    return 0;
}

void daeCoolPropThermoPhysicalPropertyPackage::CalcTwoPhaseVectorProperty(const std::string& capeOpenProperty,
                                                                          double P1,
                                                                          double T1,
                                                                          const std::vector<double>& x1,
                                                                          const std::string& phase1,
                                                                          double P2,
                                                                          double T2,
                                                                          const std::vector<double>& x2,
                                                                          const std::string& phase2,
                                                                          std::vector<double>& results,
                                                                          daeeThermoPackageBasis basis)
{
// CoolProp does not use T,P,x for both phases but calculates the equilibrium based on T,P,x of the mixture.
// To check: how to fit this into the Cape-Open interface?
//           temporary solution is to use only T1,P1,x1 (the data for the first phase).
    std::string msg = "CoolProp CalcTwoPhaseVectorProperty function is not implemented";
    throw std::runtime_error(msg);
}

/*
void daeCoolPropThermoPhysicalPropertyPackage::CapeOpen_to_CoolPropName(const std::string& capeOpenProperty, daeeThermoPackageBasis eBasis, std::string& coolPropProperty)
{
    std::string basis;
    if(eBasis == eMole)
        basis = "molar";
    else if(eBasis == eMass)
        basis = "mass";

    if(capeOpenProperty == "density")
        coolPropProperty = "D" + basis;
    else if(capeOpenProperty == "heatCapacityCp")
        coolPropProperty = "Cp" + basis;
    else if(capeOpenProperty == "heatCapacityCv")
        coolPropProperty = "Cv" + basis;
    else if(capeOpenProperty == "viscosity")
        coolPropProperty = "V";
    else if(capeOpenProperty == "entropy")
        coolPropProperty = "S" + basis;
    else if(capeOpenProperty == "gibbsEnergy")
        coolPropProperty = "G" + basis;
    else if(capeOpenProperty == "helmholtzEnergy")
        coolPropProperty = "Helmholtz" + basis;
    else if(capeOpenProperty == "conductivity")
        coolPropProperty = "L";
    else if(capeOpenProperty == "temperature")
        coolPropProperty = "T";
    else if(capeOpenProperty == "pressure")
        coolPropProperty = "P";
    else // if not found return the same name
        coolPropProperty = capeOpenProperty;

     std::cout << "coolPropProperty = " << coolPropProperty << std::endl;
}
*/

double daeCoolPropThermoPhysicalPropertyPackage::GetScalarProperty(const std::string& capeOpenProperty, daeeThermoPackageBasis eBasis)
{
    //std::cout << "capeOpenProperty = " << capeOpenProperty << ", basis = " << eBasis << std::endl;

    if(capeOpenProperty == "density")
    {
        if(eBasis == eMole)
            return m_mixture->rhomolar();
        else if(eBasis == eMass)
            return m_mixture->rhomass();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "thermalConductivity")
    {
        return m_mixture->conductivity();
    }
    else if(capeOpenProperty == "viscosity")
    {
        return m_mixture->viscosity();
    }
    else if(capeOpenProperty == "heatCapacityCp")
    {
        if(eBasis == eMole)
            return m_mixture->cpmolar();
        else if(eBasis == eMass)
            return m_mixture->cpmass();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "heatCapacityCv")
    {
        if(eBasis == eMole)
            return m_mixture->cvmolar();
        else if(eBasis == eMass)
            return m_mixture->cvmass();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "enthalpy")
    {
        if(eBasis == eMole)
            return m_mixture->hmolar();
        else if(eBasis == eMass)
            return m_mixture->hmass();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "entropy")
    {
        if(eBasis == eMole)
            return m_mixture->smolar();
        else if(eBasis == eMass)
            return m_mixture->smass();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "gibbsEnergy")
    {
        if(eBasis == eMole)
            return m_mixture->gibbsmolar();
        else if(eBasis == eMass)
            return m_mixture->gibbsmass();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "helmholtzEnergy")
    {
        if(eBasis == eMole)
            return m_mixture->helmholtzmolar();
        else if(eBasis == eMass)
            return m_mixture->helmholtzmass();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "internalEnergy")
    {
        if(eBasis == eMole)
            return m_mixture->umolar();
        else if(eBasis == eMass)
            return m_mixture->umass();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "excessEnthalpy")
    {
        if(eBasis == eMole)
            return m_mixture->hmolar_excess();
        else if(eBasis == eMass)
            return m_mixture->hmass_excess();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "excessEntropy")
    {
        if(eBasis == eMole)
            return m_mixture->smolar_excess();
        else if(eBasis == eMass)
            return m_mixture->smass_excess();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "excessGibbsEnergy")
    {
        if(eBasis == eMole)
            return m_mixture->gibbsmolar_excess();
        else if(eBasis == eMass)
            return m_mixture->gibbsmass_excess();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "excessHelmholtzEnergy")
    {
        if(eBasis == eMole)
            return m_mixture->helmholtzmolar_excess();
        else if(eBasis == eMass)
            return m_mixture->helmholtzmass_excess();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "excessHelmholtzEnergy")
    {
        if(eBasis == eMole)
            return m_mixture->volumemolar_excess();
        else if(eBasis == eMass)
            return m_mixture->volumemass_excess();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "excessInternalEnergy")
    {
        if(eBasis == eMole)
            return m_mixture->umolar_excess();
        else if(eBasis == eMass)
            return m_mixture->umass_excess();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "excessVolume")
    {
        if(eBasis == eMole)
            return m_mixture->volumemolar_excess();
        else if(eBasis == eMass)
            return m_mixture->volumemass_excess();
        else
            throw std::runtime_error("Invalid basis specified for the property " + capeOpenProperty);
    }
    else if(capeOpenProperty == "temperature")
    {
        return m_mixture->T();
    }
    else if(capeOpenProperty == "pressure")
    {
        return m_mixture->p();
    }
    else if(capeOpenProperty == "compressibilityFactor")
    {
        return m_mixture->compressibility_factor();
    }

    // Two phase properties
    else if(capeOpenProperty == "surfaceTension")
    {
        return m_mixture->surface_tension();
    }

    std::string msg = "Invalid single phase CoolProp property: " + capeOpenProperty;
    throw std::runtime_error(msg);

     return -INFINITY;
}

void daeCoolPropThermoPhysicalPropertyPackage::GetVectorProperty(const std::string& capeOpenProperty, daeeThermoPackageBasis eBasis, std::vector<double>& results)
{
    size_t Nc = m_strarrCompoundIDs.size();
    results.resize(Nc);

    if(capeOpenProperty == "fugacity")
    {
        for(size_t i = 0; i < Nc; i++)
            results[i] = m_mixture->fugacity(i);
    }
    else if(capeOpenProperty == "logFugacity")
    {
        for(size_t i = 0; i < Nc; i++)
            results[i] = ::log(m_mixture->fugacity(i));
    }
    else if(capeOpenProperty == "fugacityCoefficient")
    {
        for(size_t i = 0; i < Nc; i++)
            results[i] = m_mixture->fugacity_coefficient(i);
    }
    else if(capeOpenProperty == "logFugacityCofficient")
    {
        for(size_t i = 0; i < Nc; i++)
            results[i] = ::log(m_mixture->fugacity_coefficient(i));
    }
    else if(capeOpenProperty == "activity")
    {
        throw std::runtime_error("The property 'activity' has not been implemented in CoolProp");
    }
    else if(capeOpenProperty == "activityCoefficient")
    {
        throw std::runtime_error("The property 'activityCoefficient' has not been implemented in CoolProp");
    }
    else if(capeOpenProperty == "diffusionCoefficient") // tensor rank = 2
    {
        throw std::runtime_error("The property 'diffusionCoefficient' has not been implemented in CoolProp");
    }

    std::string msg = "Invalid single phase CoolProp property: " + capeOpenProperty;
    throw std::runtime_error(msg);
}

/*
    {iT,      "T",      "IO", "K",       "Temperature",                           false},
    {iP,      "P",      "IO", "Pa",      "Pressure",                              false},
    {iDmolar, "Dmolar", "IO", "mol/m^3", "Molar density",                         false},
    {iHmolar, "Hmolar", "IO", "J/mol",   "Molar specific enthalpy",               false},
    {iSmolar, "Smolar", "IO", "J/mol/K", "Molar specific entropy",                false},
    {iUmolar, "Umolar", "IO", "J/mol",   "Molar specific internal energy",        false},
    {iGmolar, "Gmolar", "O",  "J/mol",   "Molar specific Gibbs energy",           false},
    {iHelmholtzmolar, "Helmholtzmolar", "O",  "J/mol",   "Molar specific Helmholtz energy",           false},
    {iDmass,  "Dmass",  "IO", "kg/m^3",  "Mass density",                          false},
    {iHmass,  "Hmass",  "IO", "J/kg",    "Mass specific enthalpy",                false},
    {iSmass,  "Smass",  "IO", "J/kg/K",  "Mass specific entropy",                 false},
    {iUmass,  "Umass",  "IO", "J/kg",    "Mass specific internal energy",         false},
    {iGmass,  "Gmass",  "O",  "J/kg",    "Mass specific Gibbs energy",            false},
    {iHelmholtzmass,  "Helmholtzmass",  "O",  "J/kg",    "Mass specific Helmholtz energy",            false},
    {iQ,      "Q",      "IO", "mol/mol", "Mass vapor quality",                    false},
    {iDelta,  "Delta",  "IO", "-",       "Reduced density (rho/rhoc)",            false},
    {iTau,    "Tau",    "IO", "-",       "Reciprocal reduced temperature (Tc/T)", false},
    /// Output only
    {iCpmolar,           "Cpmolar",           "O", "J/mol/K", "Molar specific constant pressure specific heat", false},
    {iCpmass,            "Cpmass",            "O", "J/kg/K",  "Mass specific constant pressure specific heat",  false},
    {iCvmolar,           "Cvmolar",           "O", "J/mol/K", "Molar specific constant volume specific heat",    false},
    {iCvmass,            "Cvmass",            "O", "J/kg/K",  "Mass specific constant volume specific heat",     false},
    {iCp0molar,          "Cp0molar",          "O", "J/mol/K", "Ideal gas molar specific constant pressure specific heat",false},
    {iCp0mass,           "Cp0mass",           "O", "J/kg/K",  "Ideal gas mass specific constant pressure specific heat",false},
    {iSmolar_residual,   "Smolar_residual",   "O", "J/mol/K", "Residual molar entropy (sr/R = tau*dar_dtau-ar)",false},
    {iGWP20,             "GWP20",             "O", "-",       "20-year global warming potential",                 true},
    {iGWP100,            "GWP100",            "O", "-",       "100-year global warming potential",                true},
    {iGWP500,            "GWP500",            "O", "-",       "500-year global warming potential",                true},
    {iFH,                "FH",                "O", "-",       "Flammability hazard",                             true},
    {iHH,                "HH",                "O", "-",       "Health hazard",                                   true},
    {iPH,                "PH",                "O", "-",       "Physical hazard",                                 true},
    {iODP,               "ODP",               "O", "-",       "Ozone depletion potential",                       true},
    {iBvirial,           "Bvirial",           "O", "-",       "Second virial coefficient",                       false},
    {iCvirial,           "Cvirial",           "O", "-",       "Third virial coefficient",                        false},
    {idBvirial_dT,       "dBvirial_dT",       "O", "-",       "Derivative of second virial coefficient with respect to T",false},
    {idCvirial_dT,       "dCvirial_dT",       "O", "-",       "Derivative of third virial coefficient with respect to T",false},
    {igas_constant,      "gas_constant",      "O", "J/mol/K", "Molar gas constant",                              true},
    {imolar_mass,        "molar_mass",        "O", "kg/mol",  "Molar mass",                                      true},
    {iacentric_factor,   "acentric",          "O", "-",       "Acentric factor",                                 true},
    {idipole_moment,     "dipole_moment",     "O", "C-m",     "Dipole moment",                                   true},
    {irhomass_reducing,  "rhomass_reducing",  "O", "kg/m^3",  "Mass density at reducing point",                  true},
    {irhomolar_reducing, "rhomolar_reducing", "O", "mol/m^3", "Molar density at reducing point",                 true},
    {irhomolar_critical, "rhomolar_critical", "O", "mol/m^3", "Molar density at critical point",                 true},
    {irhomass_critical,  "rhomass_critical",  "O", "kg/m^3",  "Mass density at critical point",                  true},
    {iT_reducing,        "T_reducing",        "O", "K",       "Temperature at the reducing point",               true},
    {iT_critical,        "T_critical",        "O", "K",       "Temperature at the critical point",               true},
    {iT_triple,          "T_triple",          "O", "K",       "Temperature at the triple point",                 true},
    {iT_max,             "T_max",             "O", "K",       "Maximum temperature limit",                       true},
    {iT_min,             "T_min",             "O", "K",       "Minimum temperature limit",                       true},
    {iP_min,             "P_min",             "O", "Pa",      "Minimum pressure limit",                          true},
    {iP_max,             "P_max",             "O", "Pa",      "Maximum pressure limit",                          true},
    {iP_critical,        "p_critical",        "O", "Pa",      "Pressure at the critical point",                  true},
    {iP_reducing,        "p_reducing",        "O", "Pa",      "Pressure at the reducing point",                  true},
    {iP_triple,          "p_triple",          "O", "Pa",      "Pressure at the triple point (pure only)",        true},
    {ifraction_min,      "fraction_min",      "O", "-",       "Fraction (mole, mass, volume) minimum value for incompressible solutions",true},
    {ifraction_max,      "fraction_max",      "O", "-",       "Fraction (mole, mass, volume) maximum value for incompressible solutions",true},
    {iT_freeze,          "T_freeze",          "O", "K",       "Freezing temperature for incompressible solutions",true},

    {ispeed_sound,     "speed_of_sound",  "O", "m/s",   "Speed of sound",       false},
    {iviscosity,       "viscosity",       "O", "Pa-s",  "Viscosity",            false},
    {iconductivity,    "conductivity",    "O", "W/m/K", "Thermal conductivity", false},
    {isurface_tension, "surface_tension", "O", "N/m",   "Surface tension",      false},
    {iPrandtl,         "Prandtl",         "O", "-",     "Prandtl number",       false},

    {iisothermal_compressibility,             "isothermal_compressibility",             "O", "1/Pa", "Isothermal compressibility",false},
    {iisobaric_expansion_coefficient,         "isobaric_expansion_coefficient",         "O", "1/K",  "Isobaric expansion coefficient",false},
    {iZ,                                      "Z",                                      "O", "-",    "Compressibility factor",false},
    {ifundamental_derivative_of_gas_dynamics, "fundamental_derivative_of_gas_dynamics", "O", "-",    "Fundamental derivative of gas dynamics",false},
    {iPIP,                                    "PIP",                                    "O", "-",    "Phase identification parameter", false},

    {ialphar,                  "alphar",                  "O", "-", "Residual Helmholtz energy", false},
    {idalphar_dtau_constdelta, "dalphar_dtau_constdelta", "O", "-", "Derivative of residual Helmholtz energy with tau",false},
    {idalphar_ddelta_consttau, "dalphar_ddelta_consttau", "O", "-", "Derivative of residual Helmholtz energy with delta",false},

    {ialpha0,                  "alpha0",                  "O", "-", "Ideal Helmholtz energy", false},
    {idalpha0_dtau_constdelta, "dalpha0_dtau_constdelta", "O", "-", "Derivative of ideal Helmholtz energy with tau",false},
    {idalpha0_ddelta_consttau, "dalpha0_ddelta_consttau", "O", "-", "Derivative of ideal Helmholtz energy with delta",false},

    {iPhase, "Phase", "O", "-", "Phase index as a float", false},
*/
