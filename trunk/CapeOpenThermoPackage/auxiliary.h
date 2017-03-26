#pragma once
#include "stdafx.h"
#include <comcat.h>
#include <atlcom.h>
#include <comutil.h>
#include <iostream>
#include <vector>
#include <map>
#include "cape_open_package.h"

using namespace ATL;

using dae::tpp::daeeThermoPackagePropertyType;
using dae::tpp::daeeThermoPackagePhase;
using dae::tpp::daeeThermoPackageBasis;
using dae::tpp::daeThermoPhysicalPropertyPackage_t;
using dae::tpp::eMole;
using dae::tpp::eMass;
using dae::tpp::eUndefinedBasis;
using dae::tpp::eVapor;
using dae::tpp::eLiquid;
using dae::tpp::eSolid;

#import "CAPE-OPENv1-1-0.tlb" rename_namespace("CO_COM")
using namespace CO_COM;

typedef std::map<CComBSTR, _variant_t>                          ComBSTR_Variant_PropertyMap;
typedef std::map<CComBSTR, ComBSTR_Variant_PropertyMap>         ComBSTR_ComBSTR_Variant_PropertyMap;
typedef std::map<CComBSTR, ComBSTR_ComBSTR_Variant_PropertyMap> ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap;

CComObject<daeCapeThermoMaterial>* daeCreateThermoMaterial(const std::vector<BSTR>*                               compoundIDs = NULL,
                                                           const std::vector<BSTR>*                               compoundCASNumbers = NULL,
                                                           const std::map<std::string, daeeThermoPackagePhase>*   phases = NULL,
                                                           const ComBSTR_ComBSTR_Variant_PropertyMap*             overallProperties = NULL,
                                                           const ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap*     singlePhaseProperties = NULL,
                                                           const ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap*     twoPhaseProperties = NULL);

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

void get_com_error_info(_com_error& e, std::wstringstream& ss);
void get_cape_user_info(ECapeUserPtr capeUser, std::wstringstream& ss);

bool CreateVariantArray(std::vector<VARIANT>& varrResult, _variant_t& varSource);
bool CreateDoubleArray(std::vector<double>& darrResult, _variant_t& varSource);
bool CreateStringArray(std::vector<BSTR>& strarrResult, _variant_t& varSource);
bool CreateSafeArray(std::vector<double>& darrSource, _variant_t& varResult);
bool CreateSafeArray(std::vector<BSTR>& strarrSource, _variant_t& varResult);

void GetCLSIDsForCategory(GUID catID, std::vector<daeCapeCreatableObject>& objects);
HRESULT CreateTPPManager(daeCapeCreatableObject* pObject, ICapeThermoPropertyPackageManagerPtr& manager, ICapeIdentificationPtr& identification);
HRESULT CreateMaterialContext(IDispatchPtr& dispatchPackage, ICapeThermoMaterialContextPtr& materialContext);
HRESULT CreatePropertyRoutine(IDispatchPtr& dispatchPackage, ICapeThermoPropertyRoutinePtr& propertyRoutine);
HRESULT CreateErrorInterfaces(IDispatchPtr& dispatchPackage, ECapeUserPtr& capeUser);

_bstr_t phase_to_bstr(daeeThermoPackagePhase phase);
_bstr_t basis_to_bstr(daeeThermoPackageBasis basis);
//_bstr_t property_to_bstr(daeeThermoPhysicalProperty property);
double double_from_variant(_variant_t& vals_v);
std::string cstring_from_variant(_variant_t& vals_v);
_bstr_t bstring_from_variant(_variant_t& vals_v);
_variant_t create_array_from_double(double val);
_variant_t create_array_from_string(_bstr_t& val);
void print_string_array(BSTR name, _variant_t& vals_v);
void print_double_array(BSTR name, _variant_t& vals_v);
void print_thermo_manager_info(ICapeThermoPropertyPackageManagerPtr manager, ICapeIdentificationPtr identification);

#define GET_BSTR(bstr) (bstr.length() > 0 ? bstr.GetBSTR() : L"N/A")
#define DAE_THROW_EXCEPTION(ss, com_e, capeUser_e) \
    { \
        get_com_error_info(com_e, ss); \
        get_cape_user_info(capeUser_e, ss); \
        material->printAllProperties(L"Available properties:", ss); \
        _bstr_t what(ss.str().c_str()); \
        throw std::runtime_error(_com_util::ConvertBSTRToString(what)); \
    }
#define DAE_THROW_EXCEPTION2(ss) \
    { \
        material->printAllProperties(L"Available properties:", ss); \
        _bstr_t what(ss.str().c_str()); \
        throw std::runtime_error(_com_util::ConvertBSTRToString(what)); \
    }

void get_cape_user_info(ECapeUserPtr capeUser, std::wstringstream& ss)
{
    ss << L"CapeOpen error-code:     " << capeUser->Getcode()                    << std::endl;
    ss << L"CapeOpen description:    " << GET_BSTR(capeUser->Getdescription())   << std::endl;
    ss << L"CapeOpen interface name: " << GET_BSTR(capeUser->GetinterfaceName()) << std::endl;
    ss << L"CapeOpen scope:          " << GET_BSTR(capeUser->Getscope())         << std::endl;
    ss << L"CapeOpen operation:      " << GET_BSTR(capeUser->Getoperation())     << std::endl;
    ss << L"CapeOpen more info:      " << GET_BSTR(capeUser->GetmoreInfo())      << std::endl;
}

void get_com_error_info(_com_error& e, std::wstringstream& ss)
{
    ss << L"COM error-code:    " << e.Error()                           << std::endl;
    ss << L"COM error message: " << GET_BSTR(_bstr_t(e.ErrorMessage())) << std::endl;
    ss << L"COM description:   " << GET_BSTR(e.Description())           << std::endl;
}

void print_thermo_manager_info(ICapeThermoPropertyPackageManagerPtr manager, ICapeIdentificationPtr identification)
{
    std::vector<BSTR> strarrPackages;
    _variant_t pplist = manager->GetPropertyPackageList();
    CreateStringArray(strarrPackages, pplist);
    
    std::wcout << "TPP Manager: " << identification->GetComponentName() << std::endl;
    std::wcout << "    Description: " << identification->GetComponentDescription() << std::endl;
    std::wcout << "    Packages:    ";
    for (size_t i = 0; i < strarrPackages.size(); i++)
    {
        BSTR foundPackageName = strarrPackages[i];
        std::wcout << (i == 0 ? "" : ", ") << "'" << foundPackageName << "'";
    }
    std::wcout << std::endl;
}

void print_string_array(BSTR name, _variant_t& vals_v)
{
    std::vector<BSTR> strarrResult;
    CreateStringArray(strarrResult, vals_v);
    std::wcout << name << " = [";
    for (size_t i = 0; i < strarrResult.size(); i++)
        std::wcout << (i == 0 ? "" : ", ") << "'" << strarrResult[i] << "'";
    std::wcout << "]" << std::endl;
}

void print_double_array(BSTR name, _variant_t& vals_v)
{
    std::vector<double> strarrResult;
    CreateDoubleArray(strarrResult, vals_v);
    std::wcout << name << " = [";
    for (size_t i = 0; i < strarrResult.size(); i++)
        std::wcout << (i == 0 ? "" : ", ") << strarrResult[i];
    std::wcout << "]" << std::endl;
}

double double_from_variant(_variant_t& vals_v)
{
    std::vector<double> results;
    bool res = CreateDoubleArray(results, vals_v);
    if (!res || results.size() != 1)
    {
        throw std::runtime_error("Cannot create double from variant");
    }

    return results[0];
}

std::string cstring_from_variant(_variant_t& vals_v)
{
    std::vector<BSTR> results;
    bool res = CreateStringArray(results, vals_v);
    if (!res || results.size() != 1)
    {
        throw std::runtime_error("Cannot create cstring from variant");
    }

    _bstr_t bstr_res(results[0]);
    return std::string((LPCSTR)bstr_res);
}

_bstr_t bstring_from_variant(_variant_t& vals_v)
{
    std::vector<BSTR> results;
    bool res = CreateStringArray(results, vals_v);
    if (!res || results.size() != 1)
    {
        throw std::runtime_error("Cannot create bstr from variant");
    }

    return _bstr_t(results[0]);
}

_variant_t create_array_from_double(double val)
{
    _variant_t result_v;
    std::vector<double> strarr(1);
    strarr[0] = val;
    bool res = CreateSafeArray(strarr, result_v);
    if (!res)
    {
        std::cout << "Cannot create variant array from double" << std::endl;
        throw std::runtime_error("Cannot create variant array from double");
    }
    return result_v;
}

_variant_t create_array_from_string(_bstr_t& val)
{
    _variant_t result_v;
    std::vector<BSTR> strarr(1);
    strarr[0] = val.GetBSTR();
    bool res = CreateSafeArray(strarr, result_v);
    if (!res)
    {
        std::cout << "Cannot create variant array from string" << std::endl;
        throw std::runtime_error("Cannot create variant array from string");
    }
    return result_v;
}

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
        return _bstr_t("Unknown");
}

_bstr_t basis_to_bstr(daeeThermoPackageBasis basis)
{
    using namespace dae::tpp;
    if (basis == eMole)
        return _bstr_t("mole");
    else if (basis == eMass)
        return _bstr_t("mass");
    else if (basis == eUndefinedBasis)
        return _bstr_t("undefined");
    else
        return _bstr_t("undefined");
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

bool CreateVariantArray(std::vector<VARIANT>& varrResult, _variant_t& varSource)
{
    VARIANT* pData;
    HRESULT hr;
    SAFEARRAY * pSafeArray;

    // Check the type of data it contains
    if (varSource.vt != (VT_ARRAY | VT_VARIANT))
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
    pData = (VARIANT*)pSafeArray->pvData;
    if (!pData)
        return false;
    // Get the data
    for (unsigned long i = 0; i < pSafeArray->rgsabound->cElements; i++)
    {
        varrResult.push_back(*pData);
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

HRESULT CreateTPPManager(daeCapeCreatableObject* pObject, ICapeThermoPropertyPackageManagerPtr& manager, ICapeIdentificationPtr& identification)
{
    CLSID clsid;
    HRESULT hr;
    IUnknownPtr pUnknown;

    hr = CLSIDFromString(pObject->m_strCreationString, &clsid);
    if (FAILED(hr))
        return hr;

    hr = ::CoCreateInstance(clsid, NULL, CLSCTX_ALL, IID_IUnknown, (void**)&pUnknown.GetInterfacePtr());
    if (FAILED(hr))
        return hr;
    pUnknown->AddRef();

    hr = pUnknown.QueryInterface<CO_COM::ICapeThermoPropertyPackageManager>(__uuidof(CO_COM::ICapeThermoPropertyPackageManager), &manager.GetInterfacePtr());
    if (FAILED(hr))
        return hr;

    hr = pUnknown.QueryInterface<CO_COM::ICapeIdentification>(__uuidof(CO_COM::ICapeIdentification), &identification.GetInterfacePtr());
    if (FAILED(hr))
        return hr;

    pUnknown->Release();

    return S_OK;
}

HRESULT CreateMaterialContext(IDispatchPtr& dispatchPackage, ICapeThermoMaterialContextPtr& materialContext)
{
    HRESULT hr = dispatchPackage.QueryInterface<CO_COM::ICapeThermoMaterialContext>(__uuidof(CO_COM::ICapeThermoMaterialContext), &materialContext.GetInterfacePtr());

    return hr;
}

HRESULT CreateErrorInterfaces(IDispatchPtr& dispatchPackage, ECapeUserPtr& capeUser)
{
    HRESULT hr = dispatchPackage.QueryInterface<CO_COM::ECapeUser>(__uuidof(CO_COM::ECapeUser), &capeUser.GetInterfacePtr());

    return hr;
}

HRESULT CreatePropertyRoutine(IDispatchPtr& dispatchPackage, ICapeThermoPropertyRoutinePtr& propertyRoutine)
{
    HRESULT hr = dispatchPackage.QueryInterface<CO_COM::ICapeThermoPropertyRoutine>(__uuidof(CO_COM::ICapeThermoPropertyRoutine), &propertyRoutine.GetInterfacePtr());

    return hr;
}

HRESULT CreateCompounds(IDispatchPtr& dispatchPackage, ICapeThermoCompoundsPtr& compounds)
{
    HRESULT hr = dispatchPackage.QueryInterface<CO_COM::ICapeThermoCompounds>(__uuidof(CO_COM::ICapeThermoCompounds), &compounds.GetInterfacePtr());

    return hr;
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

/*
_bstr_t property_to_bstr(daeeThermoPhysicalProperty property)
{
    using namespace dae::tpp;

    switch (property)
    {
    case avogadroConstant:
        return _bstr_t("avogadroConstant");
    case boltzmannConstant:
        return _bstr_t("boltzmannConstant");
    case idealGasStateReferencePressure:
        return _bstr_t("idealGasStateReferencePressure");
    case molarGasConstant:
        return _bstr_t("molarGasConstant");
    case speedOfLightInVacuum:
        return _bstr_t("speedOfLightInVacuum");
    case standardAccelerationOfGravity:
        return _bstr_t("standardAccelerationOfGravity");
    case casRegistryNumber:
        return _bstr_t("casRegistryNumber");
    case chemicalFormula:
        return _bstr_t("chemicalFormula");
    case iupacName:
        return _bstr_t("iupacName");
    case SMILESformula:
        return _bstr_t("SMILESformula");
    case acentricFactor:
        return _bstr_t("acentricFactor");
    case associationParameter:
        return _bstr_t("associationParameter");
    case bornRadius:
        return _bstr_t("bornRadius");
    case charge:
        return _bstr_t("charge");
    case criticalCompressibilityFactor:
        return _bstr_t("criticalCompressibilityFactor");
    case criticalDensity:
        return _bstr_t("criticalDensity");
    case criticalPressure:
        return _bstr_t("criticalPressure");
    case criticalTemperature:
        return _bstr_t("criticalTemperature");
    case criticalVolume:
        return _bstr_t("criticalVolume");
    case diffusionVolume:
        return _bstr_t("diffusionVolume");
    case dipoleMoment:
        return _bstr_t("dipoleMoment");
    case energyLennardJones:
        return _bstr_t("energyLennardJones");
    case gyrationRadius:
        return _bstr_t("gyrationRadius");
    case heatOfFusionAtNormalFreezingPoint:
        return _bstr_t("heatOfFusionAtNormalFreezingPoint");
    case heatOfVaporizationAtNormalBoilingPoint:
        return _bstr_t("heatOfVaporizationAtNormalBoilingPoint");
    case idealGasEnthalpyOfFormationAt25C:
        return _bstr_t("idealGasEnthalpyOfFormationAt25C");
    case idealGasGibbsFreeEnergyOfFormationAt25C:
        return _bstr_t("idealGasGibbsFreeEnergyOfFormationAt25C");
    case liquidDensityAt25C:
        return _bstr_t("liquidDensityAt25C");
    case liquidVolumeAt25C:
        return _bstr_t("liquidVolumeAt25C");
    case lengthLennardJones:
        return _bstr_t("lengthLennardJones");
    case pureMolecularWeight: // was molecularWeight but clashes with the scalar single phase property
        return _bstr_t("molecularWeight");
    case normalBoilingPoint:
        return _bstr_t("normalBoilingPoint");
    case normalFreezingPoint:
        return _bstr_t("normalFreezingPoint");
    case parachor:
        return _bstr_t("parachor");
    case standardEntropyGas:
        return _bstr_t("standardEntropyGas");
    case standardEntropyLiquid:
        return _bstr_t("standardEntropyLiquid");
    case standardEntropySolid:
        return _bstr_t("standardEntropySolid");
    case standardEnthalpyAqueousDilution:
        return _bstr_t("standardEnthalpyAqueousDilution");
    case standardFormationEnthalpyGas:
        return _bstr_t("standardFormationEnthalpyGas");
    case standardFormationEnthalpyLiquid:
        return _bstr_t("standardFormationEnthalpyLiquid");
    case standardFormationEnthalpySolid:
        return _bstr_t("standardFormationEnthalpySolid");
    case standardFormationGibbsEnergyGas:
        return _bstr_t("standardFormationGibbsEnergyGas");
    case standardFormationGibbsEnergyLiquid:
        return _bstr_t("standardFormationGibbsEnergyLiquid");
    case standardFormationGibbsEnergySolid:
        return _bstr_t("standardFormationGibbsEnergySolid");
    case standardGibbsAqueousDilution:
        return _bstr_t("standardGibbsAqueousDilution");
    case triplePointPressure:
        return _bstr_t("triplePointPressure");
    case triplePointTemperature:
        return _bstr_t("triplePointTemperature");
    case vanderwaalsArea:
        return _bstr_t("vanderwaalsArea");
    case vanderwaalsVolume:
        return _bstr_t("vanderwaalsVolume");

        // daeePureCompoundTDProperty
    case cpAqueousInfiniteDilution:
        return _bstr_t("cpAqueousInfiniteDilution");
    case dielectricConstant:
        return _bstr_t("dielectricConstant");
    case expansivity:
        return _bstr_t("expansivity");
    case fugacityCoefficientOfVapor:
        return _bstr_t("fugacityCoefficientOfVapor");
    case glassTransitionPressure:
        return _bstr_t("glassTransitionPressure");
    case heatCapacityOfLiquid:
        return _bstr_t("heatCapacityOfLiquid");
    case heatCapacityOfSolid:
        return _bstr_t("heatCapacityOfSolid");
    case heatOfFusion:
        return _bstr_t("heatOfFusion");
    case heatOfSublimation:
        return _bstr_t("heatOfSublimation");
    case heatOfSolidSolidPhaseTransition:
        return _bstr_t("heatOfSolidSolidPhaseTransition");
    case heatOfVaporization:
        return _bstr_t("heatOfVaporization");
    case idealGasEnthalpy:
        return _bstr_t("idealGasEnthalpy");
    case idealGasEntropy:
        return _bstr_t("idealGasEntropy");
    case idealGasHeatCapacity:
        return _bstr_t("idealGasHeatCapacity");
    case meltingPressure:
        return _bstr_t("meltingPressure");
    case selfDiffusionCoefficientGas:
        return _bstr_t("selfDiffusionCoefficientGas");
    case selfDiffusionCoefficientLiquid:
        return _bstr_t("selfDiffusionCoefficientLiquid");
    case solidSolidPhaseTransitionPressure:
        return _bstr_t("solidSolidPhaseTransitionPressure");
    case sublimationPressure:
        return _bstr_t("sublimationPressure");
    case surfaceTensionSatLiquid:
        return _bstr_t("surfaceTensionSatLiquid");
    case thermalConductivityOfLiquid:
        return _bstr_t("thermalConductivityOfLiquid");
    case thermalConductivityOfSolid:
        return _bstr_t("thermalConductivityOfSolid");
    case thermalConductivityOfVapor:
        return _bstr_t("thermalConductivityOfVapor");
    case vaporPressure:
        return _bstr_t("vaporPressure");
    case virialCoefficient:
        return _bstr_t("virialCoefficient");
    case viscosityOfLiquid:
        return _bstr_t("viscosityOfLiquid");
    case viscosityOfVapor:
        return _bstr_t("viscosityOfVapor");
    case volumeChangeUponMelting:
        return _bstr_t("volumeChangeUponMelting");
    case volumeChangeUponSolidSolidPhaseTransition:
        return _bstr_t("volumeChangeUponSolidSolidPhaseTransition");
    case volumeChangeUponSublimation:
        return _bstr_t("volumeChangeUponSublimation");
    case volumeChangeUponVaporization:
        return _bstr_t("volumeChangeUponVaporization");
    case volumeOfLiquid:
        return _bstr_t("volumeOfLiquid");

        // PureCompoundPDProperty
    case boilingPointTemperature:
        return _bstr_t("boilingPointTemperature");
    case glassTransitionTemperature:
        return _bstr_t("glassTransitionTemperature");
    case meltingTemperature:
        return _bstr_t("meltingTemperature");
    case solidSolidPhaseTransitionTemperature:
        return _bstr_t("solidSolidPhaseTransitionTemperature");

        // SinglePhaseScalarProperties
    case activity:
        return _bstr_t("activity");
    case activityCoefficient:
        return _bstr_t("activityCoefficient");
    case compressibility:
        return _bstr_t("compressibility");
    case compressibilityFactor:
        return _bstr_t("compressibilityFactor");
    case density:
        return _bstr_t("density");
    case dissociationConstant:
        return _bstr_t("dissociationConstant");
    case enthalpy:
        return _bstr_t("enthalpy");
    case enthalpyF:
        return _bstr_t("enthalpyF");
    case enthalpyNF:
        return _bstr_t("enthalpyNF");
    case entropy:
        return _bstr_t("entropy");
    case entropyF:
        return _bstr_t("entropyF");
    case entropyNF:
        return _bstr_t("entropyNF");
    case excessEnthalpy:
        return _bstr_t("excessEnthalpy");
    case excessEntropy:
        return _bstr_t("excessEntropy");
    case excessGibbsEnergy:
        return _bstr_t("excessGibbsEnergy");
    case excessHelmholtzEnergy:
        return _bstr_t("excessHelmholtzEnergy");
    case excessInternalEnergy:
        return _bstr_t("excessInternalEnergy");
    case excessVolume:
        return _bstr_t("excessVolume");
    case flow:
        return _bstr_t("flow");
    case fraction:
        return _bstr_t("fraction");
    case fugacity:
        return _bstr_t("fugacity");
    case fugacityCoefficient:
        return _bstr_t("fugacityCoefficient");
    case gibbsEnergy:
        return _bstr_t("gibbsEnergy");
    case heatCapacityCp:
        return _bstr_t("heatCapacityCp");
    case heatCapacityCv:
        return _bstr_t("heatCapacityCv");
    case helmholtzEnergy:
        return _bstr_t("helmholtzEnergy");
    case internalEnergy:
        return _bstr_t("internalEnergy");
    case jouleThomsonCoefficient:
        return _bstr_t("jouleThomsonCoefficient");
    case logFugacity:
        return _bstr_t("logFugacity");
    case logFugacityCoefficient:
        return _bstr_t("logFugacityCoefficient");
    case meanActivityCoefficient:
        return _bstr_t("meanActivityCoefficient");
    case molecularWeight:
        return _bstr_t("molecularWeight");
    case osmoticCoefficient:
        return _bstr_t("osmoticCoefficient");
    case pH:
        return _bstr_t("pH");
    case pOH:
        return _bstr_t("pOH");
    case phaseFraction:
        return _bstr_t("phaseFraction");
    case pressure:
        return _bstr_t("pressure");
    case speedOfSound:
        return _bstr_t("speedOfSound");
    case temperature:
        return _bstr_t("temperature");
    case thermalConductivity:
        return _bstr_t("thermalConductivity");
    case totalFlow:
        return _bstr_t("totalFlow");
    case viscosity:
        return _bstr_t("viscosity");
    case volume:
        return _bstr_t("volume");

        // SinglePhaseVectorProperties
    case diffusionCoefficient:
        return _bstr_t("diffusionCoefficient");

        // TwoPhaseScalarProperties
    case kvalue:
        return _bstr_t("kvalue");
    case logKvalue:
        return _bstr_t("logKvalue");
    case surfaceTension:
        return _bstr_t("surfaceTension");
    default:
        return _bstr_t("unknown-property");
    }
}
*/
