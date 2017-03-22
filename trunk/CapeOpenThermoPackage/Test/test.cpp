// CapeOpenTPP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <comcat.h>
#include <atlcom.h>
#include <iostream>
#include <vector>
#include <map>
#include "../cape_open_package.h"

int main()
{
    ::CoInitialize(NULL);

    std::vector<std::string> arrNames(4);
    arrNames[0] = "Hydrogen";
    arrNames[1] = "Carbon monoxide";
    arrNames[2] = "Methane";
    arrNames[3] = "Carbon dioxide";

    dae::tpp::daeThermoPhysicalPropertyPackage_t* package = daeCreateCapeOpenPropertyPackage();
    package->LoadPackage("ChemSep Property Package Manager", "SMROG", arrNames);

    std::vector<double> results;
    std::vector<double> fraction(4);
    double pressure = 1e5;
    double temperature = 300;
    fraction[0] = 0.7557;
    fraction[1] = 0.0400;
    fraction[2] = 0.0350;
    fraction[3] = 0.1693;

    try
    {
        double density = package->SinglePhaseScalarProperty(dae::tpp::density, 
                                                            pressure, temperature, fraction, 
                                                            dae::tpp::eVapor, 
                                                            dae::tpp::eMole);
        
        std::wcout << "SMROG mixture density is: " << density << std::endl;
    }
    catch (std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
    }


    daeDeleteCapeOpenPropertyPackage(package);
    ::CoUninitialize();
}

/*
#import "C:\Program Files\Common Files\CAPE-OPEN\CAPE-OPENv1-1-0.tlb" rename_namespace("CO_COM")
using namespace CO_COM;

__declspec(dllimport) ICapeThermoMaterial* daeCreateThermoMaterial(const std::map<ATL::CComBSTR, _variant_t>* overallProperties = NULL,
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

void GetCLSIDsForCategory(GUID catID, std::vector<daeCapeCreatableObject>& objects)
{
	HRESULT hr;
	CLSID clsid;
	BSTR bstrClassName;
	BSTR bstrCLSID;
	ICatInformation *pCatInfo = NULL;
	//Create an instance of standard Component Category Manager
	hr = CoCreateInstance(CLSID_StdComponentCategoriesMgr, NULL, CLSCTX_INPROC_SERVER, IID_ICatInformation, (void **)&pCatInfo);
    if(FAILED(hr))
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
		// USERCLASSTYPE_FULL     The full type name of the class.
		// USERCLASSTYPE_SHORT    A short name (maximum of 15 characters) that is used for popup menus and the Links dialog box.
		// USERCLASSTYPE_APPNAME  The name of the application servicing the class and is used in the Result text in dialog boxes.
		OleRegGetUserType(clsid, USERCLASSTYPE_FULL, &bstrClassName);
		StringFromCLSID(clsid, &bstrCLSID);

		daeCapeCreatableObject object;
		object.m_strCreationString  = bstrCLSID;
		object.m_strDescription	    = L"";
		object.m_strName            = bstrClassName;
		object.m_strVersion         = L"";
		object.m_strCopyright       = L"";
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

int main()
{
	::CoInitialize(NULL);

    HRESULT hr;
    std::vector<daeCapeCreatableObject> objects;
    GUID catID = CapeThermoPropertyPackageManager;
    
    GetCLSIDsForCategory(catID, objects);
    for (size_t i = 0; i < objects.size(); i++)
    {
        daeCapeCreatableObject& object = objects[i];

        ICapeThermoPropertyPackageManagerPtr manager;
        ICapeIdentificationPtr identification;
        CreateTPPManager(&object, manager, identification);

        std::vector<BSTR> strarrPackages;
        _variant_t pplist = manager->GetPropertyPackageList();

        CreateStringArray(strarrPackages, pplist);

        std::wcout << "TPP Manager: " << identification->GetComponentName() << std::endl;
        std::wcout << "    Description: " << identification->GetComponentDescription() << std::endl;
        std::wcout << "    Packages:    ";
        for (size_t i = 0; i < strarrPackages.size(); i++)
        {
            BSTR packageName = strarrPackages[i];
            std::wcout << (i == 0 ? "" : ", ") << packageName;
        }
        std::wcout << std::endl;
    }

    daeCapeCreatableObject& object = objects[3];
    ICapeThermoPropertyPackageManagerPtr manager;
    ICapeIdentificationPtr identification;
    CreateTPPManager(&object, manager, identification);
    std::wcout << "ChemSep Manager: " << identification->GetComponentName() << std::endl;

    //BSTR packageName = L"nC6nC8nC10nC12";
    BSTR packageName = L"SMROG";
    IDispatchPtr package = manager->GetPropertyPackage(packageName);
    if (package == NULL)
    {
        std::cout << "package == NULL" << std::endl;
        return -1;
    }

    ICapeThermoCompoundsPtr compounds;
    CreateCompounds(package, compounds);
    if (compounds == NULL)
    {
        std::cout << "compounds == NULL" << std::endl;
        return -1;
    }
    std::cout << "GetNumCompounds = " << compounds->GetNumCompounds() << std::endl;
    _variant_t compIds, formulae, names, boilTemps, molwts, casnos;
    hr = compounds->GetCompoundList(&compIds, &formulae, &names, &boilTemps, &molwts, &casnos);
    if (FAILED(hr))
    {
        std::cout << "GetCompoundList" << std::endl;
        return -1;
    }
    compounds.Release();

    ICapeThermoPropertyRoutinePtr propertyRoutine;
    CreatePropertyRoutine(package, propertyRoutine);
    if (propertyRoutine == NULL)
    {
        std::cout << "propertyRoutine == NULL" << std::endl;
        return -1;
    }

    ICapeThermoMaterialPtr material = daeCreateThermoMaterial();
    if(material == NULL)
    {
        std::cout << "material == NULL" << std::endl;
        return -1;
    }

    ICapeThermoMaterialContextPtr materialContext;
    CreateMaterialContext(package, materialContext);
    if (materialContext == NULL)
    {
        std::cout << "materialContext == NULL" << std::endl;
        return -1;
    }

    //IDispatchPtr material = co_material->CreateMaterial();

    hr = materialContext->SetMaterial(material);
    if (FAILED(hr))
    {
        std::cout << "SetMaterial" << std::endl;
        return -1;
    }

    _variant_t pressure_v, temperature_v, fraction_v;
    std::vector<double> pressure(1), temperature(1), fraction(4);
    pressure[0] = 1e5;
    temperature[0] = 300;
    fraction[0] = 0.7557;
    fraction[1] = 0.0400;
    fraction[2] = 0.0350;
    fraction[3] = 0.1693;

    CreateSafeArray(pressure, pressure_v);
    CreateSafeArray(temperature, temperature_v);
    CreateSafeArray(fraction, fraction_v);

    hr = material->SetSinglePhaseProp(L"pressure", L"Vapor", L"Mole", pressure_v);
    if (FAILED(hr))
    {
        std::cout << "pressure" << std::endl;
        return -1;
    }

    hr = material->SetSinglePhaseProp(L"temperature", L"Vapor", L"Mole", temperature_v);
    if (FAILED(hr))
    {
        std::cout << "temperature" << std::endl;
        return -1;
    }

    hr = material->SetSinglePhaseProp(L"fraction", L"Vapor", L"Mole", fraction_v);
    if (FAILED(hr))
    {
        std::cout << "fraction" << std::endl;
        return -1;
    }

    _variant_t properties_v;
    std::vector<BSTR> strarrProperties(1);
    strarrProperties[0] = L"density";
    CreateSafeArray(strarrProperties, properties_v);

    try
    {
        _bstr_t property_bstr("density");
        _bstr_t phase_bstr("Vapor");
        bool valid = propertyRoutine->CheckSinglePhasePropSpec(property_bstr, phase_bstr);
        std::wcout << "CheckSinglePhasePropSpec valid " << valid << std::endl;

        hr = propertyRoutine->raw_CalcSinglePhaseProp(properties_v, _bstr_t("Vapor"));
    }
    catch (_com_error e)
    {
        std::wcout << "raw_CalcSinglePhaseProp failed" << std::endl;
        std::wcout << e.Error() << std::endl;
        std::wcout << e.ErrorMessage() << std::endl;
        std::wcout.flush();
        return -1;
    }

    if (FAILED(hr))
    {
        std::cout << "raw_CalcSinglePhaseProp" << std::endl;
        ProcessCapeOpenErrorCode(hr, propertyRoutine);
        return -1;
    }

    _variant_t density_v;
    hr = material->raw_GetSinglePhaseProp(_bstr_t("density"), _bstr_t("Vapor"), _bstr_t("Mole"), &density_v);
    if (FAILED(hr))
    {
        std::cout << "raw_GetSinglePhaseProp failed" << std::endl;
        return -1;
    }

    std::vector<double> darrDensities;
    CreateDoubleArray(darrDensities, density_v);
    double density = darrDensities[0];
    std::wcout << "SMROG mixture density is: " << density << std::endl;

    manager.Release();
    identification.Release();
    package.Release();
    propertyRoutine.Release();
    material.Release();
    materialContext.Release();
    
    ::CoUninitialize();
    
    return 0;
}
*/
