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
#ifndef DAE_THERMO_PACKAGE_H
#define DAE_THERMO_PACKAGE_H

#include <string>
#include <vector>

namespace dae
{
namespace tpp
{
enum daeeThermoPackagePropertyType
{
    ePureCompoundConstantProperty,
    ePureCompoundTDProperty,
    ePureCompoundPDProperty,
    eSinglePhaseScalarProperty,
    eSinglePhaseVectorProperty,
    eTwoPhaseScalarProperty,
    eTwoPhaseVectorProperty
};

enum daeeThermoPackageBasis
{
    eMole = 0,
    eMass,
    eUndefinedBasis
};

enum daeeThermoPackagePhase
{
    etppPhaseUnknown = 0,
    eVapor,
    eLiquid,
    eSolid
};

// These constants can help in identifying to which group the property belongs.
// i.e. scalar single phase properties are in the range:
//     [cSinglePhaseScalarProperties, cSinglePhaseScalarProperties + offset)
const int cPropertyTypeOffset = 1000;

const int cUniversalConstants                   = 0;
const int cPureCompoundConstantStringProperties = 1000;
const int cPureCompoundConstantFloatProperties  = 2000;
const int cPureCompoundTDProperties             = 3000;
const int cPureCompoundPDProperties             = 4000;
const int cSinglePhaseScalarProperties          = 5000;
const int cSinglePhaseVectorProperties          = 6000;
const int cTwoPhaseScalarProperties             = 7000;
const int cTwoPhaseVectorProperties             = 8000;

enum daeeThermoPhysicalProperty
{
// Universal constants
    avogadroConstant = cUniversalConstants,
    boltzmannConstant,
    idealGasStateReferencePressure,
    molarGasConstant,
    speedOfLightInVacuum,
    standardAccelerationOfGravity,

 // PureCompoundConstantStringProperty
    casRegistryNumber = cPureCompoundConstantStringProperties,
    chemicalFormula,
    iupacName,
    SMILESformula,

// PureCompoundConstantFloatProperty
    acentricFactor = cPureCompoundConstantFloatProperties,
    associationParameter,
    bornRadius,
    charge,
    criticalCompressibilityFactor,
    criticalDensity,
    criticalPressure,
    criticalTemperature,
    criticalVolume,
    diffusionVolume,
    dipoleMoment,
    energyLennardJones,
    gyrationRadius,
    heatOfFusionAtNormalFreezingPoint,
    heatOfVaporizationAtNormalBoilingPoint,
    idealGasEnthalpyOfFormationAt25C,
    idealGasGibbsFreeEnergyOfFormationAt25C,
    liquidDensityAt25C,
    liquidVolumeAt25C,
    lengthLennardJones,
    pureMolecularWeight, // was molecularWeight but clashes with the scalar single phase property
    normalBoilingPoint,
    normalFreezingPoint,
    parachor,
    standardEntropyGas,
    standardEntropyLiquid,
    standardEntropySolid,
    standardEnthalpyAqueousDilution,
    standardFormationEnthalpyGas,
    standardFormationEnthalpyLiquid,
    standardFormationEnthalpySolid,
    standardFormationGibbsEnergyGas,
    standardFormationGibbsEnergyLiquid,
    standardFormationGibbsEnergySolid,
    standardGibbsAqueousDilution,
    triplePointPressure,
    triplePointTemperature,
    vanderwaalsArea,
    vanderwaalsVolume,

// daeePureCompoundTDProperty
    cpAqueousInfiniteDilution = cPureCompoundTDProperties,
    dielectricConstant,
    expansivity,
    fugacityCoefficientOfVapor,
    glassTransitionPressure,
    heatCapacityOfLiquid,
    heatCapacityOfSolid,
    heatOfFusion,
    heatOfSublimation,
    heatOfSolidSolidPhaseTransition,
    heatOfVaporization,
    idealGasEnthalpy,
    idealGasEntropy,
    idealGasHeatCapacity,
    meltingPressure,
    selfDiffusionCoefficientGas,
    selfDiffusionCoefficientLiquid,
    solidSolidPhaseTransitionPressure,
    sublimationPressure,
    surfaceTensionSatLiquid,
    thermalConductivityOfLiquid,
    thermalConductivityOfSolid,
    thermalConductivityOfVapor,
    vaporPressure,
    virialCoefficient,
    viscosityOfLiquid,
    viscosityOfVapor,
    volumeChangeUponMelting,
    volumeChangeUponSolidSolidPhaseTransition,
    volumeChangeUponSublimation,
    volumeChangeUponVaporization,
    volumeOfLiquid,

// PureCompoundPDProperty
    boilingPointTemperature = cPureCompoundPDProperties,
    glassTransitionTemperature,
    meltingTemperature,
    solidSolidPhaseTransitionTemperature,

// SinglePhaseScalarProperties
    compressibility = cSinglePhaseScalarProperties,
    compressibilityFactor,
    density,
    dissociationConstant,
    enthalpy,
    enthalpyF,
    enthalpyNF,
    entropy,
    entropyF,
    entropyNF,
    excessEnthalpy,
    excessEntropy,
    excessGibbsEnergy,
    excessHelmholtzEnergy,
    excessInternalEnergy,
    excessVolume,
    gibbsEnergy,
    heatCapacityCp,
    heatCapacityCv,
    helmholtzEnergy,
    internalEnergy,
    jouleThomsonCoefficient,
    molecularWeight,
    osmoticCoefficient,
    pH,
    pOH,
    phaseFraction,
    pressure,
    speedOfSound,
    temperature,
    thermalConductivity,
    totalFlow,
    viscosity,
    volume,

// SinglePhaseVectorProperties
    activity = cSinglePhaseVectorProperties,
    activityCoefficient,
    diffusionCoefficient, // Tensor rank 2
    flow,
    fraction,
    fugacity,
    fugacityCoefficient,
    logFugacity,
    logFugacityCoefficient,
    meanActivityCoefficient,

// TwoPhaseScalarProperties
    surfaceTension = cTwoPhaseScalarProperties,

// TwoPhaseVectorProperties
    kvalue = cTwoPhaseVectorProperties,
    logKvalue
};

/*********************************************************************************************
    daeThermoPhysicalPropertyPackage_t
**********************************************************************************************/
class daeThermoPhysicalPropertyPackage_t
{
public:
    virtual ~daeThermoPhysicalPropertyPackage_t() {}

public:
    virtual void LoadPackage(const std::string& strPackageManager,
                             const std::string& strPackageName,
                             const std::vector<std::string>& strarrCompounds) = 0;

    virtual double PureCompoundConstantProperty(daeeThermoPhysicalProperty property, const std::string& compound) = 0;

    virtual double PureCompoundTDProperty(daeeThermoPhysicalProperty property, double T, const std::string& compound) = 0;

    virtual double PureCompoundPDProperty(daeeThermoPhysicalProperty property, double P, const std::string& compound) = 0;

    virtual double SinglePhaseScalarProperty(daeeThermoPhysicalProperty property,
                                             double P,
                                             double T,
                                             const std::vector<double>& x,
                                             daeeThermoPackagePhase phase,
                                             daeeThermoPackageBasis basis = eMole) = 0;

    virtual void SinglePhaseVectorProperty(daeeThermoPhysicalProperty property,
                                           double P,
                                           double T,
                                           const std::vector<double>& x,
                                           daeeThermoPackagePhase phase,
                                           std::vector<double>& results,
                                           daeeThermoPackageBasis basis = eMole) = 0;

    virtual double TwoPhaseScalarProperty(daeeThermoPhysicalProperty property,
                                          double P,
                                          double T,
                                          const std::vector<double>& x,
                                          daeeThermoPackageBasis basis = eMole) = 0;

    virtual void TwoPhaseVectorProperty(daeeThermoPhysicalProperty property,
                                        double P,
                                        double T,
                                        const std::vector<double>& x,
                                        std::vector<double>& results,
                                        daeeThermoPackageBasis basis = eMole) = 0;
};

}
}

#endif
