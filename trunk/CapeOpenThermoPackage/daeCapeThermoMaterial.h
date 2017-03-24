// daeCapeThermoMaterial.h : Declaration of the daeCapeThermoMaterial

#pragma once
#include "resource.h"       // main symbols
#include <map>
#include <iostream>
#include <vector>

#include "DAEToolsCapeOpen_i.h"
#include "auxiliary.h"


#if defined(_WIN32_WCE) && !defined(_CE_DCOM) && !defined(_CE_ALLOW_SINGLE_THREADED_OBJECTS_IN_MTA)
#error "Single-threaded COM objects are not properly supported on Windows CE platform, such as the Windows Mobile platforms that do not include full DCOM support. Define _CE_ALLOW_SINGLE_THREADED_OBJECTS_IN_MTA to force ATL to support creating single-thread COM object's and allow use of it's single-threaded COM object implementations. The threading model in your rgs file was set to 'Free' as that is the only threading model supported in non DCOM Windows CE platforms."
#endif


using namespace ATL;

class DAE_CAPE_OPEN_API daeCapeThermoPropertyRoutine : public dae::tpp::daeThermoPhysicalPropertyPackage_t
{
public:
    daeCapeThermoPropertyRoutine()
    {
    }
     
    void LoadPackage(const std::string& strPackageManager, 
                     const std::string& strPackageName, 
                     const std::vector<std::string>& strarrCompounds)
    {
        m_strarrCompounds = strarrCompounds;

        HRESULT hr;
        std::vector<daeCapeCreatableObject> objects;
        daeCapeCreatableObject* object = NULL;
        GUID catID = CapeThermoPropertyPackageManager;
        _bstr_t bstrPackageManager(strPackageManager.c_str());
        _bstr_t bstrPackageName(strPackageName.c_str());

        // 1. Get all thermo managers that support CapeThermoPropertyPackageManager category.
        GetCLSIDsForCategory(catID, objects);

        // 2. Iterate over found thermo package managers and try to find one with the specified name.
        for (size_t i = 0; i < objects.size(); i++)
        {
            daeCapeCreatableObject& availableObject = objects[i];

            ICapeThermoPropertyPackageManagerPtr manager;
            ICapeIdentificationPtr identification;

            // Instantiate manager.
            CreateTPPManager(&availableObject, manager, identification);

            // Get the list of available packages.
            std::vector<BSTR> strarrPackages;
            _variant_t pplist = manager->GetPropertyPackageList();
            CreateStringArray(strarrPackages, pplist);

            // Print info about the manager and its packages (optional).
            print_thermo_manager_info(manager, identification);

            // Try to find a manager with the specified name and the specified package
            _bstr_t bstrFoundManager = identification->GetComponentName();
            if (bstrFoundManager == bstrPackageManager)
            {
                for (size_t i = 0; i < strarrPackages.size(); i++)
                {
                    _bstr_t foundPackageName = strarrPackages[i];
                    if (bstrPackageName == foundPackageName)
                    {
                        object = &availableObject;
                    }
                }
            }
        }
        // 2.1 The requested thermo package manager and package could not be found - raise an exception.
        if(!object)
        {
            std::cout << "Cannot find thermo physical property package manager: " << strPackageManager << std::endl;
            throw std::runtime_error("Cannot find thermo physical property package manager: " + strPackageManager);
        }

        // 3. The thermo manager was found. Now instantiate it and store its 
        //    ICapeThermoPropertyPackageManager and ICapeIdentification interfaces 
        CreateTPPManager(object, manager, identification);
        if (manager == NULL)
        {
            std::cout << "manager == NULL" << std::endl;
            throw std::runtime_error("manager == NULL");
        }
        std::wcout << "The thermo package manager: '" << identification->GetComponentName() << "' successfully created." << std::endl;

        // 4. Create thermo package with the specified name
        _bstr_t packageName(strPackageName.c_str());
        package = manager->GetPropertyPackage(packageName);
        if (package == NULL)
        {
            std::cout << "package == NULL" << std::endl;
            throw std::runtime_error("package == NULL");
        }

        // 5. Get the ICapeThermoCompounds interface from the package.
        //    Check whether the thermo package supports our compounds.
        CreateCompounds(package, compounds);
        if (compounds == NULL)
        {
            std::cout << "compounds == NULL" << std::endl;
            throw std::runtime_error("compounds == NULL");
        }
        if(strarrCompounds.size() > compounds->GetNumCompounds())
            throw std::runtime_error("The number of compounds in the thermo package is lower than requested");
        for (size_t i = 0; i < m_strarrCompounds.size(); i++)
        {

        }

        _variant_t compIds, formulae, names, boilTemps, molwts, casnos;
        hr = compounds->GetCompoundList(&compIds, &formulae, &names, &boilTemps, &molwts, &casnos);
        if (FAILED(hr))
        {
            std::cout << "GetCompoundList" << std::endl;
            throw std::runtime_error("GetCompoundList failed");
        }

        // 6. Get the ICapeThermoPropertyRoutine interface from the package.
        CreatePropertyRoutine(package, propertyRoutine);
        if (propertyRoutine == NULL)
        {
            std::cout << "propertyRoutine == NULL" << std::endl;
            throw std::runtime_error("propertyRoutine == NULL");
        }

        // 7. Create our ICapeThermoMaterial implementation (to be sent to the package's ICapeMaterialContext).
        std::vector<BSTR> bstrCompounds;
        size_t nc = strarrCompounds.size();
        if (nc > 0)
        {
            bstrCompounds.resize(nc);
            for (size_t i = 0; i < nc; i++)
                bstrCompounds[i] = _bstr_t(strarrCompounds[i].c_str()).Detach();
        }

        material = daeCreateThermoMaterial(&bstrCompounds, NULL, NULL);
        if (material == NULL)
        {
            std::cout << "material == NULL" << std::endl;
            throw std::runtime_error("material == NULL");
        }

        // 8. Get the ICapeMaterialContext interface from the package and set its material. 
        CreateMaterialContext(package, materialContext);
        if (materialContext == NULL)
        {
            std::cout << "materialContext == NULL" << std::endl;
            throw std::runtime_error("materialContext == NULL");
        }

        hr = materialContext->SetMaterial(material);
        if (FAILED(hr))
        {
            std::cout << "SetMaterial" << std::endl;
            throw std::runtime_error("SetMaterial failed");
        }
    }

    virtual ~daeCapeThermoPropertyRoutine()
    {
        manager.Release();
        identification.Release();
        package.Release();
        compounds.Release();
        propertyRoutine.Release();
        material.Release();
        materialContext.Release();
    }

    HRESULT Set_SinglePhase_PTx(double P, double T, const std::vector<double>& x, _bstr_t& phase_bstr)
    {
        HRESULT hr;
        _variant_t pressure_v, temperature_v, fraction_v;

        _bstr_t basis_bstr = L"undefined"; // should be lower case

        std::vector<double> pressure(1), temperature(1), fraction;
        pressure[0] = P;
        temperature[0] = T;
        fraction = x;

        CreateSafeArray(pressure, pressure_v);
        CreateSafeArray(temperature, temperature_v);
        CreateSafeArray(fraction, fraction_v);

        hr = material->SetSinglePhaseProp(L"pressure", phase_bstr, basis_bstr, pressure_v);
        if (FAILED(hr))
        {
            std::cout << "pressure" << std::endl;
            return hr;
        }

        hr = material->SetSinglePhaseProp(L"temperature", phase_bstr, basis_bstr, temperature_v);
        if (FAILED(hr))
        {
            std::cout << "temperature" << std::endl;
            return hr;
        }

        hr = material->SetSinglePhaseProp(L"fraction", phase_bstr, basis_bstr, fraction_v);
        if (FAILED(hr))
        {
            std::cout << "fraction" << std::endl;
            return hr;
        }

        return S_OK;
    }

    /*std::string PureCompoundConstantStringProperty(daeeThermoPhysicalProperty property, const std::string& compound)
    {
        _bstr_t property_bstr = property_to_bstr(property);
        _bstr_t compound_bstr = compound.c_str();

        _variant_t properties_v = create_array_from_string(property_bstr);
        _variant_t compids_v    = create_array_from_string(compound_bstr);

        _variant_t cprops_v = compounds->GetCompoundConstant(properties_v, compids_v);

        return cstring_from_variant(cprops_v);
    }*/

    double PureCompoundConstantProperty(daeeThermoPhysicalProperty property, const std::string& compound)
    {
        _bstr_t property_bstr = property_to_bstr(property);
        _bstr_t compound_bstr = compound.c_str();

        _variant_t properties_v = create_array_from_string(property_bstr);
        _variant_t compids_v    = create_array_from_string(compound_bstr);

        _variant_t cprops_v     = compounds->GetCompoundConstant(properties_v, compids_v);
         
        return double_from_variant(cprops_v);
    }

    double PureCompoundTDProperty(daeeThermoPhysicalProperty property, double T, const std::string& compound)
    {
        HRESULT hr;

        _bstr_t property_bstr = property_to_bstr(property);
        _bstr_t compound_bstr = compound.c_str();

        _variant_t properties_v = create_array_from_string(property_bstr);
        _variant_t compids_v = create_array_from_string(compound_bstr);

        _variant_t results_v;
        hr = compounds->GetTDependentProperty(properties_v, T, compids_v, &results_v);
        if (FAILED(hr))
        {
            throw std::runtime_error("Cannot get PureCompoundTDProperty");
        }

        return double_from_variant(results_v);
    }

    double PureCompoundPDProperty(daeeThermoPhysicalProperty property, double P, const std::string& compound)
    {
        HRESULT hr;

        _bstr_t property_bstr = property_to_bstr(property);
        _bstr_t compound_bstr = compound.c_str();

        _variant_t properties_v = create_array_from_string(property_bstr);
        _variant_t compids_v = create_array_from_string(compound_bstr);

        _variant_t results_v;
        hr = compounds->GetPDependentProperty(properties_v, P, compids_v, &results_v);
        if (FAILED(hr))
        {
            throw std::runtime_error("Cannot get PureCompoundPDProperty");
        }

        return double_from_variant(results_v);
    }

    double SinglePhaseScalarProperty(daeeThermoPhysicalProperty property,
                                     double P,
                                     double T,
                                     const std::vector<double>& x,
                                     daeeThermoPackagePhase phase,
                                     daeeThermoPackageBasis basis = eMole)
    {
        HRESULT hr;

        _bstr_t property_bstr = property_to_bstr(property);
        _bstr_t phase_bstr    = phase_to_bstr(phase);
        _bstr_t basis_bstr    = basis_to_bstr(basis);

        hr = Set_SinglePhase_PTx(P, T, x, phase_bstr);
        if (FAILED(hr))
        {
            std::cout << "CalcSinglePhaseProp failed: cannot set T, P and x, hr = " << hr << std::endl;
            throw std::runtime_error("CalcSinglePhaseProp failed: cannot set T, P and x");
        }

        _variant_t results_v;
        _variant_t properties_v = create_array_from_string(property_bstr);

        bool valid = propertyRoutine->CheckSinglePhasePropSpec(property_bstr, phase_bstr);
        if (!valid)
        {
            std::string msg = "CheckSinglePhasePropSpec returned false; the property: ";
            msg += (LPCSTR)property_bstr;
            msg += " cannot be calculated for the phase: ";
            msg += (LPCSTR)phase_bstr;
            throw std::runtime_error(msg);
        }

        try
        {
            hr = propertyRoutine->raw_CalcSinglePhaseProp(properties_v, phase_bstr);
            if (FAILED(hr))
            {
                std::cout << "raw_CalcSinglePhaseProp failed: hr = " << hr << std::endl;
                std::string msg = "CalcSinglePhaseProp failed to calculate the property: ";
                msg += (LPCSTR)property_bstr;
                msg += " for the phase: ";
                msg += (LPCSTR)phase_bstr;
                throw std::runtime_error(msg);
            }
        }
        catch (_com_error e)
        {
            std::string msg ="CalcSinglePhaseProp failed\n Error: ";
            msg += (LPCSTR)_bstr_t(e.ErrorMessage());
            throw std::runtime_error(msg);
        }

        try
        {
            hr = material->GetSinglePhaseProp(property_bstr, phase_bstr, basis_bstr, &results_v);
        }
        catch (_com_error e)
        {
            std::string msg = "GetSinglePhaseProp failed\n Error: ";
            msg += (LPCSTR)_bstr_t(e.ErrorMessage());
            throw std::runtime_error(msg);
        }

        return double_from_variant(results_v);
    }

    void SinglePhaseVectorProperty(daeeThermoPhysicalProperty property,
                                   double P,
                                   double T,
                                   const std::vector<double>& x,
                                   daeeThermoPackagePhase phase,
                                   std::vector<double>& results,
                                   daeeThermoPackageBasis basis = eMole)
    {
    }

    double TwoPhaseScalarProperty(daeeThermoPhysicalProperty property,
                                  double P,
                                  double T,
                                  const std::vector<double>& x,
                                  daeeThermoPackageBasis basis = eMole)
    {
        return 0;
    }

    void TwoPhaseVectorProperty(daeeThermoPhysicalProperty property,
                                double P,
                                double T,
                                const std::vector<double>& x,
                                std::vector<double>& results,
                                daeeThermoPackageBasis basis = eMole)
    {

    }

public:
    std::vector<std::string>             m_strarrCompounds;
    ICapeThermoPropertyPackageManagerPtr manager;
    ICapeIdentificationPtr               identification;
    IDispatchPtr                         package;
    ICapeThermoCompoundsPtr              compounds;
    ICapeThermoPropertyRoutinePtr        propertyRoutine;
    ICapeThermoMaterialPtr               material;
    ICapeThermoMaterialContextPtr        materialContext;
};



// daeCapeThermoMaterial
class ATL_NO_VTABLE daeCapeThermoMaterial :
	public CComObjectRootEx<CComSingleThreadModel>,
	public CComCoClass<daeCapeThermoMaterial, &CLSID_daeCapeThermoMaterial>,
    public IDispatchImpl<ICapeThermoMaterial, &__uuidof(ICapeThermoMaterial), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>,
    public IDispatchImpl<ICapeThermoCompounds, &__uuidof(ICapeThermoCompounds), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>,
    public IDispatchImpl<IdaeCapeThermoMaterial, &__uuidof(IdaeCapeThermoMaterial), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>
    //public IDispatchImpl<ICapeThermoPhases, &__uuidof(ICapeThermoPhases), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>
{
public:
	daeCapeThermoMaterial()
	{
	}

DECLARE_REGISTRY_RESOURCEID(IDR_DAECAPETHERMOMATERIAL)

DECLARE_NOT_AGGREGATABLE(daeCapeThermoMaterial)

BEGIN_COM_MAP(daeCapeThermoMaterial)
	COM_INTERFACE_ENTRY(ICapeThermoMaterial)
    COM_INTERFACE_ENTRY(ICapeThermoCompounds)
    COM_INTERFACE_ENTRY(IdaeCapeThermoMaterial)
END_COM_MAP()

	DECLARE_PROTECT_FINAL_CONSTRUCT()

	HRESULT FinalConstruct()
	{
		return S_OK;
	}

	void FinalRelease()
	{
	}

public:
    // ICapeThermoMaterial
    ComBSTR_ComBSTR_Variant_PropertyMap         m_overallProperties;
    ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap m_singlePhaseProperties;
    ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap m_twoPhaseProperties;
    std::vector<BSTR>                           m_strarrCompounds;

    virtual HRESULT __stdcall raw_ClearAllProps()
    {
        //std::wcout << "raw_ClearAllProps " << std::endl;
        m_overallProperties.clear();
        m_singlePhaseProperties.clear();
        m_twoPhaseProperties.clear();
        return S_OK;
    }

    virtual HRESULT __stdcall raw_CopyFromMaterial(/*[in]*/ IDispatch** source)
    {
        //std::wcout << "raw_CopyFromMaterial " << std::endl;

        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_CreateMaterial(/*[out,retval]*/ IDispatch** materialObject)
    {
        //std::wcout << "raw_CreateMaterial " << std::endl;

        // Should I first create daeCapeThermoMaterial object and then QueryInterface for its IDispatch pointer?
        // The code below works for in process com objects.
        *materialObject = daeCreateThermoMaterial(&m_strarrCompounds, &m_overallProperties, &m_singlePhaseProperties, &m_twoPhaseProperties);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetOverallProp(/*[in]*/     BSTR property,
        /*[in]*/     BSTR basis,
        /*[in,out]*/ VARIANT* results)
    {
        //std::wcout << "raw_GetOverallProp " << std::endl;
        
        printOverallPoperties(L"Available overall properties: ");

        CComBSTR basis_c(basis);
        basis_c.ToLower();

        if (m_overallProperties.find(basis_c) == m_overallProperties.end())
        {
            std::wcout << "GetOverallProp cannot find basis " << (LPWSTR)basis_c << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& bstr_variant_map = m_overallProperties[basis_c];

        if (bstr_variant_map.find(CComBSTR(property)) == bstr_variant_map.end())
        {
            std::cout << "GetOverallProp cannot find overall property " << property << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }        
        _variant_t& val = bstr_variant_map[CComBSTR(property)];
        VariantCopy(results, &val);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetOverallTPFraction(/*[in,out]*/ double* temperature,
        /*[in,out]*/ double* pressure,
        /*[in,out]*/ VARIANT* composition)
    {
        //std::wcout << "raw_GetOverallTPFraction " << std::endl;
     
        CComBSTR basis_c = L"undefined"; // should be lower case

        if (m_overallProperties.find(basis_c) == m_overallProperties.end())
        {
            std::wcout << "GetOverallTPFraction cannot find basis " << (LPWSTR)basis_c << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& bstr_variant_map = m_overallProperties[basis_c];

        if (bstr_variant_map.find(CComBSTR(L"pressure")) == bstr_variant_map.end())
        {
            std::cout << "GetOverallProp cannot find overall pressure" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        if (bstr_variant_map.find(CComBSTR(L"temperature")) == bstr_variant_map.end())
        {
            std::cout << "GetOverallProp cannot find overall temperature" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        if (bstr_variant_map.find(CComBSTR(L"fraction")) == bstr_variant_map.end())
        {
            std::cout << "GetOverallProp cannot find overall fraction" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }

        _variant_t& temperature_v = bstr_variant_map[CComBSTR(L"temperature")];
        _variant_t& pressure_v    = bstr_variant_map[CComBSTR(L"pressure")];
        _variant_t& fraction_v    = bstr_variant_map[CComBSTR(L"fraction")];

        *temperature = double_from_variant(temperature_v);
        *pressure    = double_from_variant(pressure_v);
        VariantCopy(composition, &fraction_v);

        std::wcout << "Get overall T, P, fraction:" << std::endl;
        std::wcout << "pressure    = " << *pressure << std::endl;
        std::wcout << "temperature = " << *temperature << std::endl;
        print_double_array(L"fraction", fraction_v);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetPresentPhases(/*[in,out]*/ VARIANT* phaseLabels,
        /*[in,out]*/ VARIANT* phaseStatus)
    {
        //std::wcout << "raw_GetPresentPhases " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetSinglePhaseProp(/*[in]*/     BSTR     property,
        /*[in]*/     BSTR     phaseLabel,
        /*[in]*/     BSTR     basis,
        /*[in,out]*/ VARIANT* results)
    {
        //std::wcout << "raw_GetSinglePhaseProp " << std::endl;

        printSinglePhasePoperties(L"GetSinglePhaseProp (available single phase properties):");
        //printTwoPhasePoperties(L"GetSinglePhaseProp (available two phase properties):");
        //printOverallPoperties(L"GetSinglePhaseProp (available overall properties):");
        //std::wcout.flush();

        CComBSTR basis_c(basis);
        basis_c.ToLower();

        if (m_singlePhaseProperties.find(CComBSTR(phaseLabel)) == m_singlePhaseProperties.end())
        {
            std::wcout << "GetSinglePhaseProp phase " << phaseLabel << " not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_property_map = m_singlePhaseProperties[CComBSTR(phaseLabel)];

        if (bstr_bstr_property_map.find(basis_c) == bstr_bstr_property_map.end())
        {
            std::wcout << "GetSinglePhaseProp basis " << (LPWSTR)basis_c << " not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& bstr_property_map = bstr_bstr_property_map[basis_c];

        if (bstr_property_map.find(CComBSTR(property)) == bstr_property_map.end())
        {
            std::wcout << "GetSinglePhaseProp property " << property << " not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        _variant_t& val = bstr_property_map[CComBSTR(property)];
        VariantInit(results);
        VariantCopy(results, &val);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetTPFraction(/*[in]*/     BSTR     phaseLabel,
        /*[in,out]*/ double*  temperature,
        /*[in,out]*/ double*  pressure,
        /*[in,out]*/ VARIANT* composition)
    {
        //std::wcout << "raw_GetTPFraction " << std::endl;

        BSTR basis = L"undefined"; // should be lower case

        if (m_singlePhaseProperties.find(CComBSTR(phaseLabel)) == m_singlePhaseProperties.end())
        {
            std::wcout << "GetTPFraction phase " << phaseLabel << " not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_property_map = m_singlePhaseProperties[CComBSTR(phaseLabel)];

        if (bstr_bstr_property_map.find(CComBSTR(basis)) == bstr_bstr_property_map.end())
        {
            std::wcout << "GetTPFraction basis " << basis << " not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& bstr_property_map = bstr_bstr_property_map[CComBSTR(basis)];

        if (bstr_property_map.find(CComBSTR(L"temperature")) == bstr_property_map.end())
        {
            std::wcout << "GetTPFraction - temperature not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        if (bstr_property_map.find(CComBSTR(L"pressure")) == bstr_property_map.end())
        {
            std::wcout << "GetTPFraction - pressure not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        if (bstr_property_map.find(CComBSTR(L"fraction")) == bstr_property_map.end())
        {
            std::wcout << "GetTPFraction - fraction not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }

        _variant_t& temperature_v = bstr_property_map[CComBSTR(L"temperature")];
        _variant_t& pressure_v    = bstr_property_map[CComBSTR(L"pressure")];
        _variant_t& fraction_v    = bstr_property_map[CComBSTR(L"fraction")];

        *temperature = double_from_variant(temperature_v);
        *pressure    = double_from_variant(pressure_v);
        VariantCopy(composition, &fraction_v);

        //std::wcout << "Get T, P, fraction for the phase " << phaseLabel << ":" << std::endl;
        //std::wcout << "pressure    = " << *pressure << std::endl;
        //std::wcout << "temperature = " << *temperature << std::endl;
        //print_double_array(L"fraction", fraction_v);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetTwoPhaseProp(/*[in]*/     BSTR     property,
        /*[in]*/     VARIANT  phaseLabels,
        /*[in]*/     BSTR     basis,
        /*[in,out]*/ VARIANT* results)
    {
        //std::wcout << "raw_GetTwoPhaseProp " << std::endl;

        printTwoPhasePoperties(L"GetTwoPhaseProp (available properties):");

        CComBSTR basis_c(basis);
        basis_c.ToLower();

        std::vector<BSTR> strarrPhases;
        bool res = CreateStringArray(strarrPhases, _variant_t(phaseLabels));
        if(!res)
        {
            std::wcout << "Cannot get phase labels" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }

        _bstr_t bstr_phaseLabel;
        for (size_t i = 0; i < strarrPhases.size(); i++)
            bstr_phaseLabel += _bstr_t(i == 0 ? "" : "-") + _bstr_t(strarrPhases[i]);
        
        CComBSTR phaseLabel = bstr_phaseLabel.GetBSTR();

        if (m_twoPhaseProperties.find(phaseLabel) == m_twoPhaseProperties.end())
        {
            std::wcout << "GetTwoPhaseProp phase " << (LPWSTR)phaseLabel << " not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_property_map = m_twoPhaseProperties[CComBSTR(phaseLabel)];

        if (bstr_bstr_property_map.find(basis_c) == bstr_bstr_property_map.end())
        {
            std::wcout << "GetTwoPhaseProp basis " << (LPWSTR)basis_c << " not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& bstr_property_map = bstr_bstr_property_map[basis_c];

        if (bstr_property_map.find(CComBSTR(property)) == bstr_property_map.end())
        {
            std::wcout << "GetTwoPhaseProp property " << property << " not found" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }

        _variant_t& val_v = bstr_property_map[CComBSTR(property)];
        std::vector<double> darrResults;
        bool res2 = CreateDoubleArray(darrResults, val_v);

        _variant_t results_v;
        CreateSafeArray(darrResults, results_v);
        VariantCopy(results, &results_v);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_SetOverallProp(/*[in]*/ BSTR    property,
        /*[in]*/ BSTR    basis,
        /*[in]*/ VARIANT values)
    {
        //std::wcout << "raw_SetOverallProp " << property << ", " << basis << std::endl;

        CComBSTR basis_c(basis);
        basis_c.ToLower();
        CComBSTR property_c(property);

        m_overallProperties[basis_c][property_c] = _variant_t(values);
        //printOverallPoperties(L"Available overall properties: ");

        return S_OK;
    }

    virtual HRESULT __stdcall raw_SetPresentPhases(/*[in]*/ VARIANT phaseLabels,
        /*[in]*/ VARIANT phaseStatus)
    {
        //std::wcout << "raw_SetPresentPhases " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_SetSinglePhaseProp(/*[in]*/ BSTR    property,
        /*[in]*/ BSTR    phaseLabel,
        /*[in]*/ BSTR    basis,
        /*[in]*/ VARIANT values)
    {
        //std::wcout << "raw_SetSinglePhaseProp " << property << ", " << phaseLabel << ", " << basis << std::endl;

        CComBSTR phase_c(phaseLabel);
        CComBSTR basis_c(basis);
        basis_c.ToLower();
        CComBSTR property_c(property);

        m_singlePhaseProperties[phase_c][basis_c][property_c] = _variant_t(values);
        //printSinglePhasePoperties(L"Available single phase properties: ");

        return S_OK;
    }

    virtual HRESULT __stdcall raw_SetTwoPhaseProp(/*[in]*/ BSTR    property,
        /*[in]*/ VARIANT phaseLabels,
        /*[in]*/ BSTR    basis,
        /*[in]*/ VARIANT values)
    {
        //std::wcout << "raw_SetTwoPhaseProp " << property << ", " << basis << std::endl;

        std::vector<BSTR> strarrPhases;
        bool res = CreateStringArray(strarrPhases, _variant_t(phaseLabels));
        if (!res)
        {
            std::wcout << "Cannot get phase labels" << std::endl;
            return ECapeThrmPropertyNotAvailableHR;
        }

        _bstr_t bstr_phaseLabel;
        for (size_t i = 0; i < strarrPhases.size(); i++)
            bstr_phaseLabel += _bstr_t(i == 0 ? "" : "-") + _bstr_t(strarrPhases[i]);

        CComBSTR phaseLabel = bstr_phaseLabel.GetBSTR();
        CComBSTR basis_c(basis); 
        basis_c.ToLower();

        m_twoPhaseProperties[phaseLabel][basis_c][CComBSTR(property)] = _variant_t(values);
        //printTwoPhasePoperties(L"Available tw phase properties: ");

        return S_OK;
    }

    void printSinglePhasePoperties(BSTR heading)
    {
        std::wcout << heading << std::endl;
        for (ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap::iterator fit = m_singlePhaseProperties.begin(); fit != m_singlePhaseProperties.end(); ++fit)
        {
            CComBSTR                             phase                 = fit->first;
            ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_variant_map = fit->second;

            std::wcout << "  phase " << (LPWSTR)phase << ":" << std::endl;
            for (ComBSTR_ComBSTR_Variant_PropertyMap::iterator bit = bstr_bstr_variant_map.begin(); bit != bstr_bstr_variant_map.end(); ++bit)
            {
                CComBSTR                     basis            = bit->first;
                ComBSTR_Variant_PropertyMap& bstr_variant_map = bit->second;

                std::wcout << "    basis " << (LPWSTR)basis << ":" << std::endl;
                for (ComBSTR_Variant_PropertyMap::iterator it = bstr_variant_map.begin(); it != bstr_variant_map.end(); ++it)
                {
                    std::wcout << "      ";
                    std::vector<double> darrResult;
                    if (it->second.vt == (VT_ARRAY | VT_R8))
                    {
                        CreateDoubleArray(darrResult, it->second);
                        std::wcout << it->first.m_str << " = [";
                        for (size_t i = 0; i < darrResult.size(); i++)
                            std::wcout << (i == 0 ? "" : ", ") << darrResult[i];
                        std::wcout << "]" << std::endl;
                    }
                    else if (it->second.vt == VT_BSTR)
                    {
                        std::wcout << it->first.m_str << " = " << it->second.bstrVal << std::endl;
                    }
                    else
                    {
                        std::wcout << it->first.m_str << " = " << " unknown" << std::endl;
                    }
                }
            }
        }
    }

    void printTwoPhasePoperties(BSTR heading)
    {
        std::wcout << heading << std::endl;
        for (ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap::iterator fit = m_twoPhaseProperties.begin(); fit != m_twoPhaseProperties.end(); ++fit)
        {
            CComBSTR                             phase = fit->first;
            ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_variant_map = fit->second;

            std::wcout << "  phase " << (LPWSTR)phase << ":" << std::endl;
            for (ComBSTR_ComBSTR_Variant_PropertyMap::iterator bit = bstr_bstr_variant_map.begin(); bit != bstr_bstr_variant_map.end(); ++bit)
            {
                CComBSTR                     basis = bit->first;
                ComBSTR_Variant_PropertyMap& bstr_variant_map = bit->second;

                std::wcout << "    basis " << (LPWSTR)basis << ":" << std::endl;
                for (ComBSTR_Variant_PropertyMap::iterator it = bstr_variant_map.begin(); it != bstr_variant_map.end(); ++it)
                {
                    std::wcout << "      ";
                    std::vector<double> darrResult;
                    if (it->second.vt == (VT_ARRAY | VT_R8))
                    {
                        CreateDoubleArray(darrResult, it->second);
                        std::wcout << it->first.m_str << " = [";
                        for (size_t i = 0; i < darrResult.size(); i++)
                            std::wcout << (i == 0 ? "" : ", ") << darrResult[i];
                        std::wcout << "]" << std::endl;
                    }
                    else if (it->second.vt == VT_BSTR)
                    {
                        std::wcout << it->first.m_str << " = " << it->second.bstrVal << std::endl;
                    }
                    else
                    {
                        std::wcout << it->first.m_str << " = " << " unknown" << std::endl;
                    }
                }
            }
        }
    }

    void printOverallPoperties(BSTR heading)
    {
        std::wcout << heading << std::endl;
        for (ComBSTR_ComBSTR_Variant_PropertyMap::iterator fit = m_overallProperties.begin(); fit != m_overallProperties.end(); ++fit)
        {
            CComBSTR                     basis = fit->first;
            ComBSTR_Variant_PropertyMap& bstr_variant_map = fit->second;

            std::wcout << "  basis " << (LPWSTR)basis << ":" << std::endl;
            for (ComBSTR_Variant_PropertyMap::iterator it = bstr_variant_map.begin(); it != bstr_variant_map.end(); ++it)
            {
                std::wcout << "    ";
                std::vector<double> darrResult;
                if (it->second.vt == (VT_ARRAY | VT_R8))
                {
                    CreateDoubleArray(darrResult, it->second);
                    std::wcout << it->first.m_str << " = [";
                    for (size_t i = 0; i < darrResult.size(); i++)
                        std::wcout << (i == 0 ? "" : ", ") << darrResult[i];
                    std::wcout << "]" << std::endl;
                }
                else if (it->second.vt == VT_BSTR)
                {
                    std::wcout << it->first.m_str << " = " << it->second.bstrVal << std::endl;
                }
                else
                {
                    std::wcout << it->first.m_str << " = " << " unknown" << std::endl;
                }
            }
        }
    }

    //virtual HRESULT __stdcall raw_GetNumPhases(
    //    /*[out,retval]*/ long * num)
    //{

    //}

    //virtual HRESULT __stdcall raw_GetPhaseInfo(
    //    /*[in]*/ BSTR phaseLabel,
    //    /*[in]*/ BSTR phaseAttribute,
    //    /*[out,retval]*/ VARIANT * value)
    //{

    //}

    //virtual HRESULT __stdcall raw_GetPhaseList(
    //    /*[in,out]*/ VARIANT * phaseLabels,
    //    /*[in,out]*/ VARIANT * stateOfAggregation,
    //    /*[in,out]*/ VARIANT * keyCompoundId)
    //{

    //}

    
    virtual HRESULT __stdcall raw_GetCompoundConstant(
        /*[in]*/ VARIANT props,
        /*[in]*/ VARIANT compIds,
        /*[out,retval]*/ VARIANT * propVals)
    {
        //std::wcout << "raw_GetCompoundConstant " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetCompoundList(
        /*[in,out]*/ VARIANT * compIds,
        /*[in,out]*/ VARIANT * formulae,
        /*[in,out]*/ VARIANT * names,
        /*[in,out]*/ VARIANT * boilTemps,
        /*[in,out]*/ VARIANT * molwts,
        /*[in,out]*/ VARIANT * casnos)
    {
        //std::wcout << "raw_GetCompoundList " << std::endl;

        _variant_t names_v, compids_v;

        CreateSafeArray(m_strarrCompounds, names_v);
        CreateSafeArray(m_strarrCompounds, compids_v);

        //print_string_array(L"compound_names", names_v);
        print_string_array(L"Available compound ids in the material object: ", compids_v);

        VariantCopy(compIds, &compids_v.GetVARIANT());
        VariantCopy(names, &names_v.GetVARIANT());

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetConstPropList(
        /*[out,retval]*/ VARIANT * props)
    {
        std::wcout << "raw_GetConstPropList " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetNumCompounds(
        /*[out,retval]*/ long * num)
    {
        //std::wcout << "raw_GetNumCompounds " << std::endl;
        *num = m_strarrCompounds.size();
        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetPDependentProperty(
        /*[in]*/ VARIANT props,
        /*[in]*/ double pressure,
        /*[in]*/ VARIANT compIds,
        /*[in,out]*/ VARIANT * propVals)
    {
        //std::wcout << "raw_GetPDependentProperty " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetPDependentPropList(
        /*[out,retval]*/ VARIANT * props)
    {
        //std::wcout << "raw_GetPDependentPropList " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetTDependentProperty(
        /*[in]*/ VARIANT props,
        /*[in]*/ double temperature,
        /*[in]*/ VARIANT compIds,
        /*[in,out]*/ VARIANT * propVals)
    {
        //std::wcout << "raw_GetTDependentProperty " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetTDependentPropList(
        /*[out,retval]*/ VARIANT * props)
    {
        //std::wcout << "raw_GetTDependentPropList " << std::endl;
        return E_NOTIMPL;
    }

};

OBJECT_ENTRY_AUTO(__uuidof(daeCapeThermoMaterial), daeCapeThermoMaterial)
