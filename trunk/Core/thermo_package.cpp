#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "nodes_array.h"

// Support for Cape-Open thermo physical property packages exist only in windows (it is COM technology)
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#include "../CapeOpenThermoPackage/cape_open_package.h"
#include <objbase.h>
#endif

using dae::tpp::eMole;
using dae::tpp::eMass;
using dae::tpp::eUndefinedBasis;
using dae::tpp::etppPhaseUnknown;

namespace dae
{
namespace core
{
daeCapeOpenThermoPhysicalPropertyPackage::daeCapeOpenThermoPhysicalPropertyPackage(const string& strName, daeModel* pModel, const string& strDescription)
{
    m_package = NULL;
    m_pModel = pModel;
    SetName(strName);
    m_strDescription = strDescription;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    ::CoInitialize(NULL);
#endif
}

daeCapeOpenThermoPhysicalPropertyPackage::~daeCapeOpenThermoPhysicalPropertyPackage(void)
{
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    daeDeleteCapeOpenPropertyPackage(m_package);
    ::CoUninitialize();
#endif
}

void daeCapeOpenThermoPhysicalPropertyPackage::LoadPackage(const std::string& strPackageManager,
                                                           const std::string& strPackageName,
                                                           const std::vector<std::string>& strarrCompoundIDs,
                                                           const std::vector<std::string>& strarrCompoundCASNumbers,
                                                           const std::map<std::string,daeeThermoPackagePhase>& mapAvailablePhases,
                                                           daeeThermoPackageBasis defaultBasis)
{
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    m_package = daeCreateCapeOpenPropertyPackage();
    if(!m_package)
        daeDeclareAndThrowException(exInvalidPointer);
    m_package->LoadPackage(strPackageManager, strPackageName, strarrCompoundIDs, strarrCompoundCASNumbers, mapAvailablePhases, defaultBasis);
#endif
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::PureCompoundConstantProperty(const std::string& property,
                                                                               const std::string& compound)
{
    adouble tmp;
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    daeeThermoPackageBasis basis = eUndefinedBasis;
    adThermoPhysicalPropertyPackageScalarNode* n = new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::ePureCompoundConstantProperty,
                                                                                                 property,
                                                                                                 basis,
                                                                                                 compound,
                                                                                                 GetUnits(property, basis),
                                                                                                 m_package);
     tmp.node = adNodePtr(n);
     n->pressure     = adNodePtr();
     n->temperature  = adNodePtr();
     n->composition  = adNodeArrayPtr();
     n->phase        = "Unknown";
     n->pressure2    = adNodePtr();
     n->temperature2 = adNodePtr();
     n->composition2 = adNodeArrayPtr();
     n->phase2       = "Unknown";
#endif

    return tmp;
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::PureCompoundTDProperty(const std::string& property,
                                                                         const adouble& T,
                                                                         const std::string& compound)
{
    adouble tmp;
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    daeeThermoPackageBasis basis = eUndefinedBasis;
    adThermoPhysicalPropertyPackageScalarNode* n = new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::ePureCompoundTDProperty,
                                                                                                 property,
                                                                                                 basis,
                                                                                                 compound,
                                                                                                 GetUnits(property, basis),
                                                                                                 m_package);
     tmp.node = adNodePtr(n);
     n->pressure     = adNodePtr();
     n->temperature  = T.node;
     n->composition  = adNodeArrayPtr();
     n->phase        = "Unknown";
     n->pressure2    = adNodePtr();
     n->temperature2 = adNodePtr();
     n->composition2 = adNodeArrayPtr();
     n->phase2       = "Unknown";
#endif

    return tmp;
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::PureCompoundPDProperty(const std::string& property,
                                                                         const adouble& P,
                                                                         const std::string& compound)
{
    adouble tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    daeeThermoPackageBasis basis = eUndefinedBasis;
    adThermoPhysicalPropertyPackageScalarNode* n = new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::ePureCompoundPDProperty,
                                                                                                 property,
                                                                                                 basis,
                                                                                                 compound,
                                                                                                 GetUnits(property, basis),
                                                                                                 m_package);
     tmp.node = adNodePtr(n);
     n->pressure     = P.node;
     n->temperature  = adNodePtr();
     n->composition  = adNodeArrayPtr();
     n->phase        = "Unknown";
     n->pressure2    = adNodePtr();
     n->temperature2 = adNodePtr();
     n->composition2 = adNodeArrayPtr();
     n->phase2       = "Unknown";
#endif

    return tmp;
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::SinglePhaseScalarProperty(const std::string& property,
                                                                            const adouble& P,
                                                                            const adouble& T,
                                                                            const adouble_array& x,
                                                                            const std::string& phase,
                                                                            daeeThermoPackageBasis basis)
{
    adouble tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    adThermoPhysicalPropertyPackageScalarNode* n = new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::eSinglePhaseScalarProperty,
                                                                                                 property,
                                                                                                 basis,
                                                                                                 std::string(""),
                                                                                                 GetUnits(property, basis),
                                                                                                 m_package);
    tmp.node = adNodePtr(n);
    n->pressure     = P.node;
    n->temperature  = T.node;
    n->composition  = x.node;
    n->phase        = phase;
    n->pressure2    = adNodePtr();
    n->temperature2 = adNodePtr();
    n->composition2 = adNodeArrayPtr();
    n->phase2       = "Unknown";
#endif

    return tmp;
}

adouble_array daeCapeOpenThermoPhysicalPropertyPackage::SinglePhaseVectorProperty(const std::string& property,
                                                                                  const adouble& P,
                                                                                  const adouble& T,
                                                                                  const adouble_array& x,
                                                                                  const std::string& phase,
                                                                                  daeeThermoPackageBasis basis)
{
    adouble_array tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    adThermoPhysicalPropertyPackageArrayNode* n = new adThermoPhysicalPropertyPackageArrayNode(dae::tpp::eSinglePhaseVectorProperty,
                                                                                               property,
                                                                                               basis,
                                                                                               GetUnits(property, basis),
                                                                                               m_package);
    tmp.node = adNodeArrayPtr(n);
    n->pressure     = P.node;
    n->temperature  = T.node;
    n->composition  = x.node;
    n->phase        = phase;
    n->pressure2    = adNodePtr();
    n->temperature2 = adNodePtr();
    n->composition2 = adNodeArrayPtr();
    n->phase2       = "Unknown";
#endif

    return tmp;
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::TwoPhaseScalarProperty(const std::string& property,
                                                                         const adouble& P1,
                                                                         const adouble& T1,
                                                                         const adouble_array& x1,
                                                                         const std::string& phase1,
                                                                         const adouble& P2,
                                                                         const adouble& T2,
                                                                         const adouble_array& x2,
                                                                         const std::string& phase2,
                                                                         daeeThermoPackageBasis basis)

{
    adouble tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    adThermoPhysicalPropertyPackageScalarNode* n = new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::eTwoPhaseScalarProperty,
                                                                                                 property,
                                                                                                 basis,
                                                                                                 std::string(""),
                                                                                                 GetUnits(property, basis),
                                                                                                 m_package);
    tmp.node = adNodePtr(n);
    n->pressure     = P1.node;
    n->temperature  = T1.node;
    n->composition  = x1.node;
    n->phase        = phase1;
    n->pressure2    = P2.node;
    n->temperature2 = T2.node;
    n->composition2 = x2.node;
    n->phase2       = phase2;
#endif

    return tmp;
}

adouble_array daeCapeOpenThermoPhysicalPropertyPackage::TwoPhaseVectorProperty(const std::string& property,
                                                                               const adouble& P1,
                                                                               const adouble& T1,
                                                                               const adouble_array& x1,
                                                                               const std::string& phase1,
                                                                               const adouble& P2,
                                                                               const adouble& T2,
                                                                               const adouble_array& x2,
                                                                               const std::string& phase2,
                                                                               daeeThermoPackageBasis basis)
{
    adouble_array tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    adThermoPhysicalPropertyPackageArrayNode* n = new adThermoPhysicalPropertyPackageArrayNode(dae::tpp::eTwoPhaseVectorProperty,
                                                                                               property,
                                                                                               basis,
                                                                                               GetUnits(property, basis),
                                                                                               m_package);
    tmp.node = adNodeArrayPtr(n);
    n->pressure     = P1.node;
    n->temperature  = T1.node;
    n->composition  = x1.node;
    n->phase        = phase1;
    n->pressure2    = P2.node;
    n->temperature2 = T2.node;
    n->composition2 = x2.node;
    n->phase2       = phase2;
#endif

    return tmp;
}

#include "../Units/units_pool.h"
using namespace units::units_pool;

unit daeCapeOpenThermoPhysicalPropertyPackage::GetUnits(const std::string& property, daeeThermoPackageBasis basis)
{
    unit basis_u;
    if(basis == eMole)
        basis_u = mol;
    else if(basis == eMass)
        basis_u = kg;

    if(property == "avogadroConstant")
        return mol^(-1);
    else if(property == "boltzmannConstant")
        return J/K;
    else if(property == "idealGasStateReferencePressure")
        return Pa;
    else if(property == "molarGasConstant")
        return J/(mol*K);
    else if(property == "speedOfLightInVacuum")
        return m/s;
    else if(property == "standardAccelerationOfGravity")
        return m/(s^2);
//    else if(property == "casRegistryNumber")
//        return unit();
//    else if(property == "chemicalFormula")
//        return unit();
//    else if(property == "iupacName")
//        return unit();
//    else if(property == "SMILESformula")
//        return unit();
//    else if(property == "acentricFactor")
//        return unit();
    else if(property == "associationParameter")
        return unit();
    else if(property == "bornRadius")
        return m;
    else if(property == "charge")
        return unit();
    else if(property == "criticalCompressibilityFactor")
        return unit();
    else if(property == "criticalDensity")
        return mol/(m^3);
    else if(property == "criticalPressure")
        return Pa;
    else if(property == "criticalTemperature")
        return K;
    else if(property == "criticalVolume")
        return (m^3)/mol;
    else if(property == "diffusionVolume")
        return m^3;
    else if(property == "dipoleMoment")
        return C*m;
    else if(property == "energyLennardJones")
        return K;
    else if(property == "gyrationRadius")
        return m;
    else if(property == "heatOfFusionAtNormalFreezingPoint")
        return J/mol;
    else if(property == "heatOfVaporizationAtNormalBoilingPoint")
        return J/mol;
    else if(property == "idealGasEnthalpyOfFormationAt25C")
        return J/mol;
    else if(property == "idealGasGibbsFreeEnergyOfFormationAt25C")
        return J/mol;
    else if(property == "liquidDensityAt25C")
        return mol/(m^3);
    else if(property == "liquidVolumeAt25C")
        return (m^3)/mol;
    else if(property == "lengthLennardJones")
        return m;
    else if(property == "molecularWeight")
        return unit();
    else if(property == "normalBoilingPoint")
        return K;
    else if(property == "normalFreezingPoint")
        return K;
    else if(property == "parachor")
        return (m^3) * (kg^0.25) / ((s^0.5) * mol);
    else if(property == "standardEntropyGas")
        return J/mol;
    else if(property == "standardEntropyLiquid")
        return J/mol;
    else if(property == "standardEntropySolid")
        return J/mol;
    else if(property == "standardEnthalpyAqueousDilution")
        return J/mol;
    else if(property == "standardFormationEnthalpyGas")
        return J/mol;
    else if(property == "standardFormationEnthalpyLiquid")
        return J/mol;
    else if(property == "standardFormationEnthalpySolid")
        return J/mol;
    else if(property == "standardFormationGibbsEnergyGas")
        return J/mol;
    else if(property == "standardFormationGibbsEnergyLiquid")
        return J/mol;
    else if(property == "standardFormationGibbsEnergySolid")
        return J/mol;
    else if(property == "standardGibbsAqueousDilution")
        return J/mol;
    else if(property == "triplePointPressure")
        return Pa;
    else if(property == "triplePointTemperature")
        return K;
    else if(property == "vanderwaalsArea")
        return (m^2)/mol;
    else if(property == "vanderwaalsVolume")
        return (m^3)/mol;

    // daeePureCompoundTDProperty
    if(property == "cpAqueousInfiniteDilution")
        return J/(mol*K);
    else if(property == "dielectricConstant")
        return unit();
    else if(property == "expansivity")
        return K^(-1);
    else if(property == "fugacityCoefficientOfVapor")
        return unit();
    else if(property == "glassTransitionPressure")
        return Pa;
    else if(property == "heatCapacityOfLiquid")
        return J/(mol*K);
    else if(property == "heatCapacityOfSolid")
        return J/(mol*K);
    else if(property == "heatOfFusion")
        return J/mol;
    else if(property == "heatOfSublimation")
        return J/mol;
    else if(property == "heatOfSolidSolidPhaseTransition")
        return J/mol;
    else if(property == "heatOfVaporization")
        return J/mol;
    else if(property == "idealGasEnthalpy")
        return J/mol;
    else if(property == "idealGasEntropy")
        return J/(mol*K);
    else if(property == "idealGasHeatCapacity")
        return J/(mol*K);
    else if(property == "meltingPressure")
        return Pa;
    else if(property == "selfDiffusionCoefficientGas")
        return (m^2)/s;
    else if(property == "selfDiffusionCoefficientLiquid")
        return (m^2)/s;
    else if(property == "solidSolidPhaseTransitionPressure")
        return Pa;
    else if(property == "sublimationPressure")
        return Pa;
    else if(property == "surfaceTensionSatLiquid")
        return N/m;
    else if(property == "thermalConductivityOfLiquid")
        return W/(m*K);
    else if(property == "thermalConductivityOfSolid")
        return W/(m*K);
    else if(property == "thermalConductivityOfVapor")
        return W/(m*K);
    else if(property == "vaporPressure")
        return Pa;
    else if(property == "virialCoefficient")
        return (m^3)/mol;
    else if(property == "viscosityOfLiquid")
        return Pa*s;
    else if(property == "viscosityOfVapor")
        return Pa*s;
    else if(property == "volumeChangeUponMelting")
        return (m^3)/mol;
    else if(property == "volumeChangeUponSolidSolidPhaseTransition")
        return (m^3)/mol;
    else if(property == "volumeChangeUponSublimation")
        return (m^3)/mol;
    else if(property == "volumeChangeUponVaporization")
        return (m^3)/mol;
    else if(property == "volumeOfLiquid")
        return (m^3)/mol;
    else if(property == "volumeOfSolid")
        return (m^3)/mol;

    // PureCompoundPDProperty
    if(property == "boilingPointTemperature")
        return K;
    else if(property == "glassTransitionTemperature")
        return K;
    else if(property == "meltingTemperature")
        return K;
    else if(property == "solidSolidPhaseTransitionTemperature")
        return K;

    // SinglePhaseScalarProperties
    if(property == "activity")
        return unit();
    else if(property == "activityCoefficient")
        return unit();
    else if(property == "compressibility")
        return Pa^(-1);
    else if(property == "compressibilityFactor")
        return unit();
    else if(property == "density")
        if(basis == eMole)
            return mol/(m^3);
        else if(basis == eMass)
            return kg/(m^3);
        else
            return unit();
    else if(property == "dissociationConstant")
        return unit();
    else if(property == "enthalpy") // extensive property
        return J / basis_u;
    else if(property == "enthalpyF") // extensive property
        return J / basis_u;
    else if(property == "enthalpyNF") // extensive property
        return J / basis_u;
    else if(property == "entropy") // extensive property
        return J/K / basis_u;
    else if(property == "entropyF") // extensive property
        return J/K / basis_u;
    else if(property == "entropyNF") // extensive property
        return J/K / basis_u;
    else if(property == "excessEnthalpy") // extensive property
        return J / basis_u;
    else if(property == "excessEntropy") // extensive property
        return J/K / basis_u;
    else if(property == "excessGibbsEnergy") // extensive property
        return J / basis_u;
    else if(property == "excessHelmholtzEnergy") // extensive property
        return J / basis_u;
    else if(property == "excessInternalEnergy") // extensive property
        return J / basis_u;
    else if(property == "excessVolume") // extensive property
        return (m^3) / basis_u;
    else if(property == "flow") // extensive property
        if(basis == eMole)
            return mol/s;
        else if(basis == eMass)
            return kg/s;
        else
            return unit();
    else if(property == "fraction")
        return unit();
    else if(property == "fugacity")
        return Pa;
    else if(property == "fugacityCoefficient")
        return unit();
    else if(property == "gibbsEnergy") // extensive property
        return J / basis_u;
    else if(property == "heatCapacityCp") // extensive property
        return J/K / basis_u;
    else if(property == "heatCapacityCv") // extensive property
        return J/K / basis_u;
    else if(property == "helmholtzEnergy") // extensive property
        return J / basis_u;
    else if(property == "internalEnergy") // extensive property
        return J / basis_u;
    else if(property == "jouleThomsonCoefficient")
        return K/Pa;
    else if(property == "logFugacity")
        return unit();
    else if(property == "logFugacityCoefficient")
        return unit();
    else if(property == "meanActivityCoefficient")
        return unit();
    else if(property == "osmoticCoefficient")
        return unit();
    else if(property == "pH")
        return unit();
    else if(property == "pOH")
        return unit();
    else if(property == "phaseFraction")
        return unit();
    else if(property == "pressure")
        return Pa;
    else if(property == "speedOfSound")
        return m/s;
    else if(property == "temperature")
        return K;
    else if(property == "thermalConductivity")
        return W/(m*K);
    else if(property == "totalFlow") // extensive property
        if(basis == eMole)
            return mol/s;
        else if(basis == eMass)
            return kg/s;
        else
            return unit();
    else if(property == "viscosity")
        return Pa*s;
    else if(property == "volume") // extensive property
        return (m^3) / basis_u;

    // SinglePhaseVectorProperties
    if(property == "diffusionCoefficient")
        return (m^2)/s;

    // TwoPhaseScalarProperties
    if(property == "kvalue")
        return unit();
    else if(property == "logKvalue")
        return unit();
    else if(property == "surfaceTension")
        return N/m;

    return unit();
}

}
}

