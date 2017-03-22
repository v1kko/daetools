#pragma once
#include "stdafx.h"
#include <comcat.h>
#include <atlcom.h>
#include <iostream>
#include <vector>
#include <map>
#include "cape_open_package.h"

using dae::tpp::daeeThermoPackagePropertyType;
using dae::tpp::daeeThermoPhysicalProperty;
using dae::tpp::daeeThermoPackagePhase;
using dae::tpp::daeeThermoPackageBasis;
using dae::tpp::daeThermoPhysicalPropertyPackage_t;
using dae::tpp::eMole;
using dae::tpp::eMass;
using dae::tpp::eVapor;
using dae::tpp::eLiquid;
using dae::tpp::eSolid;

#import "CAPE-OPENv1-1-0.tlb" rename_namespace("CO_COM")
using namespace CO_COM;

ICapeThermoMaterial* daeCreateThermoMaterial(const std::vector<BSTR>*                   compounds = NULL, 
                                             const std::map<ATL::CComBSTR, _variant_t>* overallProperties = NULL,
                                             const std::map<ATL::CComBSTR, _variant_t>* singleProperties = NULL);

//	CAPE - OPEN 1.1 Property Package Manager CF51E383 - 0110 - 4ed8 - ACB7 - B50CFDE6908E
//	CAPE - OPEN 1.1 Property Package CF51E384 - 0110 - 4ed8 - ACB7 - B50CFDE6908E
//	CAPE - OPEN 1.1 Physical Property Calculator CF51E385 - 0110 - 4ed8 - ACB7 - B50CFDE6908E
//	CAPE - OPEN 1.1 Equilibrium Calculator CF51E386 - 0110 - 4ed8 - ACB7 - B50CFDE6908E;

static const GUID CapeThermoPropertyPackageManager = { 0xCF51E383, 0x0110, 0x4ed8,{ 0xAC, 0xB7, 0xB5, 0x0C, 0xFD, 0xE6, 0x90, 0x8E } };
static const GUID CapeThermoPropertyPackage = { 0xCF51E384, 0x0110, 0x4ed8,{ 0xAC, 0xB7, 0xB5, 0x0C, 0xFD, 0xE6, 0x90, 0x8E } };
static const GUID CapeThermoPhysicalPropertyCalculator = { 0xCF51E385, 0x0110, 0x4ed8,{ 0xAC, 0xB7, 0xB5, 0x0C, 0xFD, 0xE6, 0x90, 0x8E } };
static const GUID CapeThermoEquilibriumCalculator = { 0xCF51E386, 0x0110, 0x4ed8,{ 0xAC, 0xB7, 0xB5, 0x0C, 0xFD, 0xE6, 0x90, 0x8E } };

struct daeCapeCreatableObject
{
    BSTR m_strName;
    BSTR m_strCreationString;
    BSTR m_strDescription;
    BSTR m_strVersion;
    BSTR m_strCopyright;
};

bool CreateDoubleArray(std::vector<double>& darrResult, _variant_t& varSource);
bool CreateStringArray(std::vector<BSTR>& strarrResult, _variant_t& varSource);
bool CreateSafeArray(std::vector<double>& darrSource, _variant_t& varResult);
bool CreateSafeArray(std::vector<BSTR>& strarrSource, _variant_t& varResult);

void GetCLSIDsForCategory(GUID catID, std::vector<daeCapeCreatableObject>& objects);
void CreateTPPManager(daeCapeCreatableObject* pObject, ICapeThermoPropertyPackageManagerPtr& manager, ICapeIdentificationPtr& identification);
void CreateMaterialContext(IDispatchPtr& dispatchPackage, ICapeThermoMaterialContextPtr& materialContext);
void CreatePropertyRoutine(IDispatchPtr& dispatchPackage, ICapeThermoPropertyRoutinePtr& propertyRoutine);
_bstr_t phase_to_bstr(daeeThermoPackagePhase phase);
_bstr_t basis_to_bstr(daeeThermoPackageBasis basis);
_bstr_t property_to_bstr(daeeThermoPhysicalProperty property);

_bstr_t phase_to_bstr(daeeThermoPackagePhase phase)
{
    using namespace dae::tpp;
    if (phase == eVapor)
        return _bstr_t("Vapor");
    else if (phase == eLiquid)
        return _bstr_t("Liquid");
    else if (phase == eSolid)
        return _bstr_t("Solid");
    else
        return _bstr_t("unknown-phase");
}

_bstr_t basis_to_bstr(daeeThermoPackageBasis basis)
{
    using namespace dae::tpp;
    if (basis == eMole)
        return _bstr_t("Mole");
    else if (basis == eMass)
        return _bstr_t("Mass");
    else
        return _bstr_t("unknown-phase");
}

_bstr_t property_to_bstr(daeeThermoPhysicalProperty property)
{
    using namespace dae::tpp;

    if(property == acentricFactor)
        return _bstr_t("acentricFactor");
    else if(property == associationParameter)
        return _bstr_t("associationParameter");
    else if(property == bornRadius)
        return _bstr_t("bornRadius");
    else if(property == charge)
        return _bstr_t("charge");
    else if(property == criticalCompressibilityFactor)
        return _bstr_t("criticalCompressibilityFactor");
    else if(property == criticalDensity)
        return _bstr_t("criticalDensity");
    else if(property == criticalPressure)
        return _bstr_t("criticalPressure");
    else if(property == criticalTemperature)
        return _bstr_t("criticalTemperature");
    else if(property == criticalVolume)
        return _bstr_t("criticalVolume");
    else if(property == diffusionVolume)
        return _bstr_t("diffusionVolume");
    else if(property == dipoleMoment)
        return _bstr_t("dipoleMoment");
    else if(property == energyLennardJones)
        return _bstr_t("energyLennardJones");
    else if(property == gyrationRadius)
        return _bstr_t("gyrationRadius");
    else if(property == heatOfFusionAtNormalFreezingPoint)
        return _bstr_t("heatOfFusionAtNormalFreezingPoint");
    else if(property == heatOfVaporizationAtNormalBoilingPoint)
        return _bstr_t("heatOfVaporizationAtNormalBoilingPoint");
    else if(property == idealGasEnthalpyOfFormationAt25C)
        return _bstr_t("idealGasEnthalpyOfFormationAt25C");
    else if(property == idealGasGibbsFreeEnergyOfFormationAt25C)
        return _bstr_t("idealGasGibbsFreeEnergyOfFormationAt25C");
    else if(property == liquidDensityAt25C)
        return _bstr_t("liquidDensityAt25C");
    else if(property == liquidVolumeAt25C)
        return _bstr_t("liquidVolumeAt25C");
    else if(property == lengthLennardJones)
        return _bstr_t("lengthLennardJones");
    else if(property == pureMolecularWeight) // was molecularWeight but clashes with the scalar single phase property
        return _bstr_t("molecularWeight");
    else if(property == normalBoilingPoint)
        return _bstr_t("normalBoilingPoint");
    else if(property == normalFreezingPoint)
        return _bstr_t("normalFreezingPoint");
    else if(property == parachor)
        return _bstr_t("parachor");
    else if(property == standardEntropyGas)
        return _bstr_t("standardEntropyGas");
    else if(property == standardEntropyLiquid)
        return _bstr_t("standardEntropyLiquid");
    else if(property == standardEntropySolid)
        return _bstr_t("standardEntropySolid");
    else if(property == standardEnthalpyAqueousDilution)
        return _bstr_t("standardEnthalpyAqueousDilution");
    else if(property == standardFormationEnthalpyGas)
        return _bstr_t("standardFormationEnthalpyGas");
    else if(property == standardFormationEnthalpyLiquid)
        return _bstr_t("standardFormationEnthalpyLiquid");
    else if(property == standardFormationEnthalpySolid)
        return _bstr_t("standardFormationEnthalpySolid");
    else if(property == standardFormationGibbsEnergyGas)
        return _bstr_t("standardFormationGibbsEnergyGas");
    else if(property == standardFormationGibbsEnergyLiquid)
        return _bstr_t("standardFormationGibbsEnergyLiquid");
    else if(property == standardFormationGibbsEnergySolid)
        return _bstr_t("standardFormationGibbsEnergySolid");
    else if(property == standardGibbsAqueousDilution)
        return _bstr_t("standardGibbsAqueousDilution");
    else if(property == triplePointPressure)
        return _bstr_t("triplePointPressure");
    else if(property == triplePointTemperature)
        return _bstr_t("triplePointTemperature");
    else if(property == vanderwaalsArea)
        return _bstr_t("vanderwaalsArea");
    else if(property == vanderwaalsVolume)
        return _bstr_t("vanderwaalsVolume");

    // daeePureCompoundTDProperty
    else if(property == cpAqueousInfiniteDilution)
        return _bstr_t("cpAqueousInfiniteDilution");
    else if(property == dielectricConstant)
        return _bstr_t("dielectricConstant");
    else if(property == expansivity)
        return _bstr_t("expansivity");
    else if(property == fugacityCoefficientOfVapor)
        return _bstr_t("fugacityCoefficientOfVapor");
    else if(property == glassTransitionPressure)
        return _bstr_t("glassTransitionPressure");
    else if(property == heatCapacityOfLiquid)
        return _bstr_t("heatCapacityOfLiquid");
    else if(property == heatCapacityOfSolid)
        return _bstr_t("heatCapacityOfSolid");
    else if(property == heatOfFusion)
        return _bstr_t("heatOfFusion");
    else if(property == heatOfSublimation)
        return _bstr_t("heatOfSublimation");
    else if(property == heatOfSolidSolidPhaseTransition)
        return _bstr_t("heatOfSolidSolidPhaseTransition");
    else if(property == heatOfVaporization)
        return _bstr_t("heatOfVaporization");
    else if(property == idealGasEnthalpy)
        return _bstr_t("idealGasEnthalpy");
    else if(property == idealGasEntropy)
        return _bstr_t("idealGasEntropy");
    else if(property == idealGasHeatCapacity)
        return _bstr_t("idealGasHeatCapacity");
    else if(property == meltingPressure)
        return _bstr_t("meltingPressure");
    else if(property == selfDiffusionCoefficientGas)
        return _bstr_t("selfDiffusionCoefficientGas");
    else if(property == selfDiffusionCoefficientLiquid)
        return _bstr_t("selfDiffusionCoefficientLiquid");
    else if(property == solidSolidPhaseTransitionPressure)
        return _bstr_t("solidSolidPhaseTransitionPressure");
    else if(property == sublimationPressure)
        return _bstr_t("sublimationPressure");
    else if(property == surfaceTensionSatLiquid)
        return _bstr_t("surfaceTensionSatLiquid");
    else if(property == thermalConductivityOfLiquid)
        return _bstr_t("thermalConductivityOfLiquid");
    else if(property == thermalConductivityOfSolid)
        return _bstr_t("thermalConductivityOfSolid");
    else if(property == thermalConductivityOfVapor)
        return _bstr_t("thermalConductivityOfVapor");
    else if(property == vaporPressure)
        return _bstr_t("vaporPressure");
    else if(property == virialCoefficient)
        return _bstr_t("virialCoefficient");
    else if(property == viscosityOfLiquid)
        return _bstr_t("viscosityOfLiquid");
    else if(property == viscosityOfVapor)
        return _bstr_t("viscosityOfVapor");
    else if(property == volumeChangeUponMelting)
        return _bstr_t("volumeChangeUponMelting");
    else if(property == volumeChangeUponSolidSolidPhaseTransition)
        return _bstr_t("volumeChangeUponSolidSolidPhaseTransition");
    else if(property == volumeChangeUponSublimation)
        return _bstr_t("volumeChangeUponSublimation");
    else if(property == volumeChangeUponVaporization)
        return _bstr_t("volumeChangeUponVaporization");
    else if(property == volumeOfLiquid)
        return _bstr_t("volumeOfLiquid");

    // PureCompoundPDProperty
    else if(property == boilingPointTemperature)
        return _bstr_t("boilingPointTemperature");
    else if(property == glassTransitionTemperature)
        return _bstr_t("glassTransitionTemperature");
    else if(property == meltingTemperature)
        return _bstr_t("meltingTemperature");
    else if(property == solidSolidPhaseTransitionTemperature)
        return _bstr_t("solidSolidPhaseTransitionTemperature");

    // SinglePhaseScalarProperties
    else if(property == activity)
        return _bstr_t("activity");
    else if(property == activityCoefficient)
        return _bstr_t("activityCoefficient");
    else if(property == compressibility)
        return _bstr_t("compressibility");
    else if(property == compressibilityFactor)
        return _bstr_t("compressibilityFactor");
    else if(property == density)
        return _bstr_t("density");
    else if(property == dissociationConstant)
        return _bstr_t("dissociationConstant");
    else if(property == enthalpy)
        return _bstr_t("enthalpy");
    else if(property == enthalpyF)
        return _bstr_t("enthalpyF");
    else if(property == enthalpyNF)
        return _bstr_t("enthalpyNF");
    else if(property == entropy)
        return _bstr_t("entropy");
    else if(property == entropyF)
        return _bstr_t("entropyF");
    else if(property == entropyNF)
        return _bstr_t("entropyNF");
    else if(property == excessEnthalpy)
        return _bstr_t("excessEnthalpy");
    else if(property == excessEntropy)
        return _bstr_t("excessEntropy");
    else if(property == excessGibbsEnergy)
        return _bstr_t("excessGibbsEnergy");
    else if(property == excessHelmholtzEnergy)
        return _bstr_t("excessHelmholtzEnergy");
    else if(property == excessInternalEnergy)
        return _bstr_t("excessInternalEnergy");
    else if(property == excessVolume)
        return _bstr_t("excessVolume");
    else if(property == flow)
        return _bstr_t("flow");
    else if(property == fraction)
        return _bstr_t("fraction");
    else if(property == fugacity)
        return _bstr_t("fugacity");
    else if(property == fugacityCoefficient)
        return _bstr_t("fugacityCoefficient");
    else if(property == gibbsEnergy)
        return _bstr_t("gibbsEnergy");
    else if(property == heatCapacityCp)
        return _bstr_t("heatCapacityCp");
    else if(property == heatCapacityCv)
        return _bstr_t("heatCapacityCv");
    else if(property == helmholtzEnergy)
        return _bstr_t("helmholtzEnergy");
    else if(property == internalEnergy)
        return _bstr_t("internalEnergy");
    else if(property == jouleThomsonCoefficient)
        return _bstr_t("jouleThomsonCoefficient");
    else if(property == logFugacity)
        return _bstr_t("logFugacity");
    else if(property == logFugacityCoefficient)
        return _bstr_t("logFugacityCoefficient");
    else if(property == meanActivityCoefficient)
        return _bstr_t("meanActivityCoefficient");
    else if(property == molecularWeight)
        return _bstr_t("molecularWeight");
    else if(property == osmoticCoefficient)
        return _bstr_t("osmoticCoefficient");
    else if(property == pH)
        return _bstr_t("pH");
    else if(property == pOH)
        return _bstr_t("pOH");
    else if(property == phaseFraction)
        return _bstr_t("phaseFraction");
    else if(property == pressure)
        return _bstr_t("pressure");
    else if(property == speedOfSound)
        return _bstr_t("speedOfSound");
    else if(property == temperature)
        return _bstr_t("temperature");
    else if(property == thermalConductivity)
        return _bstr_t("thermalConductivity");
    else if(property == totalFlow)
        return _bstr_t("totalFlow");
    else if(property == viscosity)
        return _bstr_t("viscosity");
    else if(property == volume)
        return _bstr_t("volume");

    // SinglePhaseVectorProperties
    else if(property == diffusionCoefficient)
        return _bstr_t("diffusionCoefficient");

    // TwoPhaseScalarProperties
    else if(property == kvalue)
        return _bstr_t("kvalue");
    else if(property == logKvalue)
        return _bstr_t("logKvalue");
    else if(property == surfaceTension)
        return _bstr_t("surfaceTension");
    else
        return _bstr_t("unknown-property");
}

void GetCLSIDsForCategory(GUID catID, std::vector<daeCapeCreatableObject>& objects)
{
    HRESULT hr;
    CLSID clsid;
    BSTR bstrClassName;
    BSTR bstrCLSID;
    ICatInformation *pCatInfo = NULL;
    //Create an instance of standard Component Category Manager
    hr = CoCreateInstance(CLSID_StdComponentCategoriesMgr, NULL, CLSCTX_INPROC_SERVER, IID_ICatInformation, (void **)&pCatInfo);
    if (FAILED(hr))
        return;

    //Increase ref count on interface
    pCatInfo->AddRef();

    //IEnumGUID interface provides enumerator for enumerating through
    //the collection of COM objects
    IEnumGUID *pEnumGUID = NULL;

    //We are intersted in finding out only controls so put CATID_Control
    //in the array
    CATID pcatidImpl[1];
    CATID pcatidReqd[1];
    pcatidImpl[0] = catID;

    // Now enumerate the classes i.e. COM objects of this type.
    hr = pCatInfo->EnumClassesOfCategories(1, pcatidImpl, 0, pcatidReqd, &pEnumGUID);
    if (FAILED(hr))
        return;

    hr = pEnumGUID->Next(1, &clsid, NULL);
    while (hr == S_OK)
    {
        /*
        USERCLASSTYPE_FULL     The full type name of the class.
        USERCLASSTYPE_SHORT    A short name (maximum of 15 characters) that is used for popup menus and the Links dialog box.
        USERCLASSTYPE_APPNAME  The name of the application servicing the class and is used in the Result text in dialog boxes.
        */
        OleRegGetUserType(clsid, USERCLASSTYPE_FULL, &bstrClassName);
        StringFromCLSID(clsid, &bstrCLSID);

        daeCapeCreatableObject object;
        object.m_strCreationString = bstrCLSID;
        object.m_strDescription = L"";
        object.m_strName = bstrClassName;
        object.m_strVersion = L"";
        object.m_strCopyright = L"";
        objects.push_back(object);

        hr = pEnumGUID->Next(1, &clsid, NULL);
    }
    pCatInfo->Release();
}

bool CreateSafeArray(std::vector<BSTR>& strarrSource, _variant_t& varResult)
{
    BSTR* pData;
    HRESULT hr;
    SAFEARRAY * pSafeArray;
    SAFEARRAYBOUND rgsabound[1];

    // Set the type of data it contains
    varResult.vt = VT_ARRAY | VT_BSTR;
    // Get the SAFEARRAY pointer
    rgsabound[0].lLbound = 0;
    rgsabound[0].cElements = strarrSource.size();
    pSafeArray = SafeArrayCreate(VT_BSTR, 1, rgsabound);
    if (!pSafeArray)
        return false;

    varResult.parray = pSafeArray;
    // Lock the SAFEARRAY
    hr = SafeArrayLock(pSafeArray);
    if (FAILED(hr))
        return false;

    pData = (BSTR*)pSafeArray->pvData;
    if (!pData)
        return false;

    for (size_t i = 0; i < strarrSource.size(); i++)
    {
        *pData = ::SysAllocString(strarrSource[i]);
        pData++;
    }

    // Unock the SAFEARRAY
    hr = SafeArrayUnlock(pSafeArray);
    if (FAILED(hr))
        return false;

    return true;
}

bool CreateSafeArray(std::vector<double>& darrSource, _variant_t& varResult)
{
    double* pData;
    HRESULT hr;
    SAFEARRAY * pSafeArray;
    SAFEARRAYBOUND rgsabound[1];

    // Set the type of data it contains
    varResult.vt = VT_ARRAY | VT_R8;
    // Get the SAFEARRAY pointer
    rgsabound[0].lLbound = 0;
    rgsabound[0].cElements = darrSource.size();
    pSafeArray = SafeArrayCreate(VT_R8, 1, rgsabound);
    if (!pSafeArray)
        return false;

    varResult.parray = pSafeArray;
    // Lock the SAFEARRAY
    hr = SafeArrayLock(pSafeArray);
    if (FAILED(hr))
        return false;

    pData = (double*)pSafeArray->pvData;
    if (!pData)
        return false;

    for (size_t i = 0; i < darrSource.size(); i++)
    {
        *pData = darrSource[i];
        pData++;
    }
    // Unock the SAFEARRAY
    hr = SafeArrayUnlock(pSafeArray);
    if (FAILED(hr))
        return false;

    return true;
}

bool CreateDoubleArray(std::vector<double>& darrResult, _variant_t& varSource)
{
    double* pData;
    HRESULT hr;
    SAFEARRAY * pSafeArray;

    // Check the type of data it contains
    if (varSource.vt != (VT_ARRAY | VT_R8))
        return false;
    // Get the SAFEARRAY pointer
    pSafeArray = varSource.parray;
    if (!pSafeArray)
        return false;
    // Lock the SAFEARRAY
    hr = SafeArrayLock(pSafeArray);
    if (FAILED(hr))
        return false;
    // Get the pointer to the data
    pData = (double*)pSafeArray->pvData;
    if (!pData)
        return false;
    // Get the data
    for (unsigned long i = 0; i < pSafeArray->rgsabound->cElements; i++)
    {
        darrResult.push_back(*pData);
        pData++;
    }
    // Unock the SAFEARRAY
    hr = SafeArrayUnlock(pSafeArray);
    if (FAILED(hr))
        return false;
    return true;
}

bool CreateStringArray(std::vector<BSTR>& strarrResult, _variant_t& varSource)
{
    BSTR* pData;
    HRESULT hr;
    SAFEARRAY * pSafeArray;

    // Check the type of data it contains
    if (varSource.vt != (VT_ARRAY | VT_BSTR))
        return false;
    // Get the SAFEARRAY pointer
    pSafeArray = varSource.parray;
    if (!pSafeArray)
        return false;
    if (SafeArrayGetDim(pSafeArray) != 1)
        return false;
    // Lock the SAFEARRAY
    hr = SafeArrayLock(pSafeArray);
    if (FAILED(hr))
        return false;
    // Get the pointer to the data
    pData = (BSTR*)pSafeArray->pvData;
    if (!pData)
        return false;
    // Get the data
    for (unsigned long i = 0; i < pSafeArray->rgsabound->cElements; i++)
    {
        strarrResult.push_back(::SysAllocString(*pData));
        pData++;
    }
    // Unock the SAFEARRAY
    hr = SafeArrayUnlock(pSafeArray);
    if (FAILED(hr))
        return false;
    return true;
}

void CreateTPPManager(daeCapeCreatableObject* pObject, ICapeThermoPropertyPackageManagerPtr& manager, ICapeIdentificationPtr& identification)
{
    CLSID clsid;
    HRESULT hr;
    IUnknownPtr pUnknown;

    hr = CLSIDFromString(pObject->m_strCreationString, &clsid);
    if (FAILED(hr))
        return;

    hr = ::CoCreateInstance(clsid, NULL, CLSCTX_ALL, IID_IUnknown, (void**)&pUnknown.GetInterfacePtr());
    if (FAILED(hr))
        return;
    pUnknown->AddRef();

    hr = pUnknown.QueryInterface<CO_COM::ICapeThermoPropertyPackageManager>(__uuidof(CO_COM::ICapeThermoPropertyPackageManager), &manager.GetInterfacePtr());
    if (FAILED(hr))
        return;

    hr = pUnknown.QueryInterface<CO_COM::ICapeIdentification>(__uuidof(CO_COM::ICapeIdentification), &identification.GetInterfacePtr());
    if (FAILED(hr))
        return;

    pUnknown->Release();
}

void CreateMaterialContext(IDispatchPtr& dispatchPackage, ICapeThermoMaterialContextPtr& materialContext)
{
    HRESULT hr;

    hr = dispatchPackage.QueryInterface<CO_COM::ICapeThermoMaterialContext>(__uuidof(CO_COM::ICapeThermoMaterialContext), &materialContext.GetInterfacePtr());
    if (FAILED(hr))
        std::cout << "ICapeThermoMaterialContext" << std::endl;
}

void CreatePropertyRoutine(IDispatchPtr& dispatchPackage, ICapeThermoPropertyRoutinePtr& propertyRoutine)
{
    HRESULT hr;

    hr = dispatchPackage.QueryInterface<CO_COM::ICapeThermoPropertyRoutine>(__uuidof(CO_COM::ICapeThermoPropertyRoutine), &propertyRoutine.GetInterfacePtr());
    if (FAILED(hr))
        std::cout << "ICapeThermoPropertyRoutine" << std::endl;
}

void CreateCompounds(IDispatchPtr& dispatchPackage, ICapeThermoCompoundsPtr& compounds)
{
    HRESULT hr;

    hr = dispatchPackage.QueryInterface<CO_COM::ICapeThermoCompounds>(__uuidof(CO_COM::ICapeThermoCompounds), &compounds.GetInterfacePtr());
    if (FAILED(hr))
        std::cout << "ICapeThermoCompounds" << std::endl;
}

void ProcessCapeOpenErrorCode(HRESULT hr, IDispatchPtr package)
{
    if (hr == ECapeUnknownHR)
    {
        std::wcout << "ECapeUnknownHR" << std::endl;
    }
    else if (hr == ECapeDataHR)
    {
        std::cout << "ECapeDataHR" << std::endl;
    }
    else if (hr == ECapeLicenceErrorHR)
    {
        std::cout << "ECapeLicenceErrorHR" << std::endl;
    }
    else if (hr == ECapeBadCOParameterHR)
    {
        std::cout << "ECapeBadCOParameterHR" << std::endl;
    }
    else if (hr == ECapeBadArgumentHR)
    {
        std::cout << "ECapeBadArgumentHR" << std::endl;
    }
    else if (hr == ECapeInvalidArgumentHR)
    {
        std::cout << "ECapeInvalidArgumentHR" << std::endl;
    }
    else if (hr == ECapeOutOfBoundsHR)
    {
        std::cout << "ECapeOutOfBoundsHR" << std::endl;
    }
    else if (hr == ECapeImplementationHR)
    {
        std::cout << "ECapeImplementationHR" << std::endl;
    }
    else if (hr == ECapeNoImplHR)
    {
        std::cout << "ECapeNoImplHR" << std::endl;
    }
    else if (hr == ECapeLimitedImplHR)
    {
        std::cout << "ECapeLimitedImplHR" << std::endl;
    }
    else if (hr == ECapeComputationHR)
    {
        std::cout << "ECapeComputationHR" << std::endl;
    }
    else if (hr == ECapeOutOfResourcesHR)
    {
        std::cout << "ECapeOutOfResourcesHR" << std::endl;
    }
    else if (hr == ECapeNoMemoryHR)
    {
        std::cout << "ECapeNoMemoryHR" << std::endl;
    }
    else if (hr == ECapeTimeOutHR)
    {
        std::cout << "ECapeTimeOutHR" << std::endl;
    }
    else if (hr == ECapeFailedInitialisationHR)
    {
        std::cout << "ECapeFailedInitialisationHR" << std::endl;
    }
    else if (hr == ECapeSolvingErrorHR)
    {
        std::cout << "ECapeSolvingErrorHR" << std::endl;
    }
    else if (hr == ECapeBadInvOrderHR)
    {
        std::cout << "ECapeBadInvOrderHR" << std::endl;
    }
    else if (hr == ECapeInvalidOperationHR)
    {
        std::cout << "ECapeInvalidOperationHR" << std::endl;
    }
    else if (hr == ECapePersistenceHR)
    {
        std::cout << "ECapePersistenceHR" << std::endl;
    }
    else if (hr == ECapeIllegalAccessHR)
    {
        std::cout << "ECapeIllegalAccessHR" << std::endl;
    }
    else if (hr == ECapePersistenceNotFoundHR)
    {
        std::cout << "ECapePersistenceNotFoundHR" << std::endl;
    }
    else if (hr == ECapePersistenceSystemErrorHR)
    {
        std::cout << "ECapePersistenceSystemErrorHR" << std::endl;
    }
    else if (hr == ECapePersistenceOverflowHR)// Bound
    {
        std::cout << "ECapePersistenceOverflowHR" << std::endl;
    }
    else if (hr == ECapeOutsideSolverScopeHR)
    {
        std::cout << "ECapeOutsideSolverScopeHR" << std::endl;
    }
    else if (hr == ECapeHessianInfoNotAvailableHR)
    {
        std::cout << "ECapeHessianInfoNotAvailableHR" << std::endl;
    }
    else if (hr == ECapeThrmPropertyNotAvailableHR)
    {
        std::cout << "ECapeThrmPropertyNotAvailableHR" << std::endl;
    }

    ECapeUserPtr error;
    HRESULT hr2 = package.QueryInterface<CO_COM::ECapeUser>(__uuidof(CO_COM::ECapeUser), &error.GetInterfacePtr());
    if (FAILED(hr2))
        return;
    std::wcout << error->Getoperation() << std::endl;
    std::wcout << error->Getdescription() << std::endl;
    std::wcout << error->Getscope() << std::endl;
    std::wcout << error->GetinterfaceName() << std::endl;
    std::wcout.flush();
    error->Release();
}
