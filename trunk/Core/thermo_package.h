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
    eMass
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

const int cPureCompoundConstantProperties = 0;
const int cPureCompoundTDProperties       = 1000;
const int cPureCompoundPDProperties       = 2000;
const int cSinglePhaseScalarProperties    = 3000;
const int cSinglePhaseVectorProperties    = 4000;
const int cTwoPhaseScalarProperties       = 5000;
const int cTwoPhaseVectorProperties       = 6000;

enum daeeThermoPhysicalProperty
{
// PureCompoundConstantProperty
    acentricFactor = cPureCompoundConstantProperties,
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
    activity = cSinglePhaseScalarProperties,
    activityCoefficient,
    compressibility,
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
    flow,
    fraction,
    fugacity,
    fugacityCoefficient,
    gibbsEnergy,
    heatCapacityCp,
    heatCapacityCv,
    helmholtzEnergy,
    internalEnergy,
    jouleThomsonCoefficient,
    logFugacity,
    logFugacityCoefficient,
    meanActivityCoefficient,
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
    diffusionCoefficient = cSinglePhaseVectorProperties,

// TwoPhaseScalarProperties
    kvalue = cTwoPhaseScalarProperties,
    logKvalue,
    surfaceTension
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

    virtual double PureCompoundConstantProperty(daeeThermoPhysicalProperty property,
                                                daeeThermoPackageBasis basis = eMole) = 0;

    virtual double PureCompoundTDProperty(daeeThermoPhysicalProperty property,
                                          double T,
                                          daeeThermoPackageBasis basis = eMole) = 0;

    virtual double PureCompoundPDProperty(daeeThermoPhysicalProperty property,
                                          double P,
                                          daeeThermoPackageBasis basis = eMole) = 0;

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
};

}
}

#endif
