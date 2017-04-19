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
    std::string msg = "CalcTwoPhaseScalarProperty function is not implemented in CoolProp";
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
    std::string msg = "CalcTwoPhaseVectorProperty function is not implemented in CoolProp";
    throw std::runtime_error(msg);
}

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

    else if(capeOpenProperty == "molecularWeight")
    {
        // Molar mass in CoolProp is in kg/mol. We should return dimensionless number
        // which should be multiplied by the 'molar mass constant' (1 g/mol)
        // to obtain the molar mass. Therefore, multiply the molar mass by 1000
        // to get the molar mass in g/mol and then divide by 1 g/mol.
        return m_mixture->molar_mass() * 1000; // MM kg/mol * (1000 g/kg) / (1 g/mol) -> dimensionless MW
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
