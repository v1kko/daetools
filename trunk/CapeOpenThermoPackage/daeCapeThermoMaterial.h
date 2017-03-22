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

        GetCLSIDsForCategory(catID, objects);
        for (size_t i = 0; i < objects.size(); i++)
        {
            daeCapeCreatableObject& availableObject = objects[i];

            ICapeThermoPropertyPackageManagerPtr manager;
            ICapeIdentificationPtr identification;
            CreateTPPManager(&availableObject, manager, identification);

            std::vector<BSTR> strarrPackages;
            _variant_t pplist = manager->GetPropertyPackageList();

            CreateStringArray(strarrPackages, pplist);

            std::wcout << "TPP Manager: " << identification->GetComponentName() << std::endl;
            std::wcout << "    Description: " << identification->GetComponentDescription() << std::endl;
            std::wcout << "    Packages:    ";
            for (size_t i = 0; i < strarrPackages.size(); i++)
            {
                BSTR foundPackageName = strarrPackages[i];
                std::wcout << (i == 0 ? "" : ", ") << foundPackageName;
            }
            std::wcout << std::endl;

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
        if(!object)
        {
            std::cout << "Cannot find thermo physical property package manager: " << strPackageManager << std::endl;
            throw std::runtime_error("Cannot find thermo physical property package manager: " + strPackageManager);
        }

        CreateTPPManager(object, manager, identification);
        if (manager == NULL)
        {
            std::cout << "manager == NULL" << std::endl;
            throw std::runtime_error("manager == NULL");
        }
        std::wcout << "Package Manager: " << identification->GetComponentName() << " created." << std::endl;

        _bstr_t packageName(strPackageName.c_str());
        package = manager->GetPropertyPackage(packageName);
        if (package == NULL)
        {
            std::cout << "package == NULL" << std::endl;
            throw std::runtime_error("package == NULL");
        }

        CreateCompounds(package, compounds);
        if (compounds == NULL)
        {
            std::cout << "compounds == NULL" << std::endl;
            throw std::runtime_error("compounds == NULL");
        }
        if(strarrCompounds.size() > compounds->GetNumCompounds())
            throw std::runtime_error("");
        std::cout << "GetNumCompounds = " << strarrCompounds.size() << std::endl;

        _variant_t compIds, formulae, names, boilTemps, molwts, casnos;
        hr = compounds->GetCompoundList(&compIds, &formulae, &names, &boilTemps, &molwts, &casnos);
        if (FAILED(hr))
        {
            std::cout << "GetCompoundList" << std::endl;
            throw std::runtime_error("GetCompoundList failed");
        }

        CreatePropertyRoutine(package, propertyRoutine);
        if (propertyRoutine == NULL)
        {
            std::cout << "propertyRoutine == NULL" << std::endl;
            throw std::runtime_error("propertyRoutine == NULL");
        }

        std::vector<BSTR> bstrCompounds;
        size_t nc = strarrCompounds.size();
        if (nc > 0)
        {
            bstrCompounds.resize(nc);
            for (size_t i = 0; i < nc; i++)
                bstrCompounds[i] = _bstr_t(strarrCompounds[i].c_str()).GetBSTR();
        }

        material = daeCreateThermoMaterial(&bstrCompounds, NULL, NULL);
        if (material == NULL)
        {
            std::cout << "material == NULL" << std::endl;
            throw std::runtime_error("material == NULL");
        }

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

    void Set_SinglePhase_PTx(double P, double T, const std::vector<double>& x, _bstr_t& phase_bstr, _bstr_t& basis_bstr)
    {
        HRESULT hr;
        _variant_t pressure_v, temperature_v, fraction_v;

        std::vector<double> pressure(1), temperature(1), fraction;
        pressure[0] = P;
        temperature[0] = T;
        fraction = x;

        CreateSafeArray(pressure, pressure_v);
        CreateSafeArray(temperature, temperature_v);
        CreateSafeArray(fraction, fraction_v);

        hr = material->SetSinglePhaseProp(L"pressure", phase_bstr.GetBSTR(), basis_bstr.GetBSTR(), pressure_v);
        if (FAILED(hr))
        {
            std::cout << "pressure" << std::endl;
            throw std::runtime_error("");
        }

        hr = material->SetSinglePhaseProp(L"temperature", phase_bstr.GetBSTR(), basis_bstr.GetBSTR(), temperature_v);
        if (FAILED(hr))
        {
            std::cout << "temperature" << std::endl;
            throw std::runtime_error("");
        }

        hr = material->SetSinglePhaseProp(L"fraction", phase_bstr.GetBSTR(), basis_bstr.GetBSTR(), fraction_v);
        if (FAILED(hr))
        {
            std::cout << "fraction" << std::endl;
            throw std::runtime_error("");
        }
    }

    double PureCompoundConstantProperty(daeeThermoPhysicalProperty property,
        daeeThermoPackageBasis basis = eMole)
    {
        return 0;
    }

    double PureCompoundTDProperty(daeeThermoPhysicalProperty property,
        double T,
        daeeThermoPackageBasis basis = eMole)
    {
        //HRESULT hr;

        //_bstr_t property_bstr = property_to_bstr(property);

        //_variant_t results_v;
        //hr = compounds->GetTDependentProperty(property_bstr.GetBSTR(), T, compids, &results_v);
        //if (FAILED(hr))
        //{
        //    std::cout << "raw_GetSinglePhaseProp failed" << std::endl;
        //    throw std::runtime_error("");
        //}

        //std::vector<double> results;
        //CreateDoubleArray(results, results_v);
        //if (results.size() != 1)
        //{
        //    std::cout << "Invalid number of results" << std::endl;
        //    throw std::runtime_error("Invalid number of results");
        //}

        //return results[0];

        return 0;
    }

    double PureCompoundPDProperty(daeeThermoPhysicalProperty property,
        double P,
        daeeThermoPackageBasis basis = eMole)
    {
        return 0;
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

        Set_SinglePhase_PTx(P, T, x, phase_bstr, basis_bstr);

        _variant_t properties_v;
        std::vector<BSTR> strarrProperties(1);
        strarrProperties[0] = property_bstr.GetBSTR();
        CreateSafeArray(strarrProperties, properties_v);

        try
        {
            bool valid = propertyRoutine->CheckSinglePhasePropSpec(property_bstr.GetBSTR(), basis_bstr.GetBSTR());
            std::wcout << "CheckSinglePhasePropSpec valid " << valid << std::endl;

            hr = propertyRoutine->raw_CalcSinglePhaseProp(properties_v, phase_bstr.GetBSTR());
        }
        catch (_com_error e)
        {
            std::wcout << "raw_CalcSinglePhaseProp failed" << std::endl;
            std::wcout << e.Error() << std::endl;
            std::wcout << e.ErrorMessage() << std::endl;
            std::wcout.flush();
            throw std::runtime_error("");
        }

        if (FAILED(hr))
        {
            std::cout << "raw_CalcSinglePhaseProp" << std::endl;
            ProcessCapeOpenErrorCode(hr, propertyRoutine);
            throw std::runtime_error("");
        }

        _variant_t results_v;
        hr = material->raw_GetSinglePhaseProp(property_bstr.GetBSTR(), phase_bstr.GetBSTR(), basis_bstr.GetBSTR(), &results_v);
        if (FAILED(hr))
        {
            std::cout << "raw_GetSinglePhaseProp failed" << std::endl;
            throw std::runtime_error("");
        }

        std::vector<double> results;
        CreateDoubleArray(results, results_v);
        if (results.size() != 1)
        {
            std::cout << "Invalid number of results" << std::endl;
            throw std::runtime_error("Invalid number of results");
        }

        return results[0];
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

__declspec(dllexport) dae::tpp::daeThermoPhysicalPropertyPackage_t* daeCreateCapeOpenPropertyPackage()
{
    return new daeCapeThermoPropertyRoutine;
}

__declspec(dllexport) void daeDeleteCapeOpenPropertyPackage(dae::tpp::daeThermoPhysicalPropertyPackage_t* package)
{
    daeCapeThermoPropertyRoutine* co_package = dynamic_cast<daeCapeThermoPropertyRoutine*>(package);
    if (package)
        delete package;
    package = NULL;
}

// daeCapeThermoMaterial
class ATL_NO_VTABLE daeCapeThermoMaterial :
	public CComObjectRootEx<CComSingleThreadModel>,
	public CComCoClass<daeCapeThermoMaterial, &CLSID_daeCapeThermoMaterial>,
    public IDispatchImpl<ICapeThermoMaterial, &__uuidof(ICapeThermoMaterial), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>,
    public IDispatchImpl<ICapeThermoCompounds, &__uuidof(ICapeThermoCompounds), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>
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
    std::map<CComBSTR, _variant_t> overallProperties;
    std::map<CComBSTR, _variant_t> singleProperties;
    std::vector<BSTR>              m_strarrCompounds;

    virtual HRESULT __stdcall raw_ClearAllProps()
    {
        std::wcout << "raw_ClearAllProps " << std::endl;
        overallProperties.clear();
        singleProperties.clear();
        return S_OK;
    }

    virtual HRESULT __stdcall raw_CopyFromMaterial(/*[in]*/ IDispatch** source)
    {
        std::wcout << "raw_CopyFromMaterial " << std::endl;

        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_CreateMaterial(/*[out,retval]*/ IDispatch** materialObject)
    {
        std::wcout << "raw_CreateMaterial " << std::endl;
        *materialObject = daeCreateThermoMaterial(&m_strarrCompounds, &overallProperties, &singleProperties);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetOverallProp(/*[in]*/     BSTR property,
        /*[in]*/     BSTR basis,
        /*[in,out]*/ VARIANT* results)
    {
        std::wcout << "raw_GetOverallProp " << std::endl;
        if (overallProperties.find(CComBSTR(property)) == overallProperties.end())
        {
            std::cout << "raw_GetOverallProp " << property << std::endl;
            return E_FAIL;
        }

        _variant_t& val = overallProperties[CComBSTR(property)];
        VariantInit(results);
        VariantCopy(results, &val);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetOverallTPFraction(/*[in,out]*/ double* temperature,
        /*[in,out]*/ double* pressure,
        /*[in,out]*/ VARIANT* composition)
    {
        std::wcout << "raw_GetOverallTPFraction " << std::endl;
        std::vector<double> darrResult;
        CreateDoubleArray(darrResult, overallProperties[CComBSTR(L"temperature")]);
        *temperature = darrResult[0];

        darrResult.clear();
        CreateDoubleArray(darrResult, overallProperties[CComBSTR(L"temperature")]);
        *pressure = darrResult[0];

        VariantInit(composition);
        VariantCopy(composition, &overallProperties[CComBSTR(L"fraction")]);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetPresentPhases(/*[in,out]*/ VARIANT* phaseLabels,
        /*[in,out]*/ VARIANT* phaseStatus)
    {
        std::wcout << "raw_GetPresentPhases " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetSinglePhaseProp(/*[in]*/     BSTR     property,
        /*[in]*/     BSTR     phaseLabel,
        /*[in]*/     BSTR     basis,
        /*[in,out]*/ VARIANT* results)
    {
        std::wcout << "raw_GetSinglePhaseProp " << std::endl;
        for (std::map<CComBSTR, _variant_t>::iterator it = singleProperties.begin(); it != singleProperties.end(); ++it)
        {
            std::vector<double> darrResult;
            if (it->second.vt == (VT_ARRAY | VT_R8))
            {
                CreateDoubleArray(darrResult, it->second);
                std::wcout << it->first.m_str << " = [";
                for (size_t i = 0; i < darrResult.size(); i++)
                    std::wcout << (i == 0 ? "" : ", ") << darrResult[0];
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

        if (singleProperties.find(CComBSTR(property)) == singleProperties.end())
        {
            std::wcout << "raw_GetSinglePhaseProp " << property << " not found" << std::endl;
            return E_FAIL;
        }

        _variant_t& val = singleProperties[CComBSTR(property)];
        VariantInit(results);
        VariantCopy(results, &val);
        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetTPFraction(/*[in]*/     BSTR     phaseLabel,
        /*[in,out]*/ double*  temperature,
        /*[in,out]*/ double*  pressure,
        /*[in,out]*/ VARIANT* composition)
    {
        std::wcout << "raw_GetTPFraction " << std::endl;
        std::wcout << "phase = " << phaseLabel << std::endl;
        std::vector<double> darrResult;
        CreateDoubleArray(darrResult, singleProperties[CComBSTR(L"temperature")]);
        *temperature = darrResult[0];
        std::wcout << "temperature = " << (*temperature) << std::endl;

        darrResult.clear();
        CreateDoubleArray(darrResult, singleProperties[CComBSTR(L"pressure")]);
        *pressure = darrResult[0];
        std::wcout << "pressure = " << (*pressure) << std::endl;

        _variant_t comp = singleProperties[CComBSTR(L"fraction")];
        VariantInit(composition);
        VariantCopy(composition, &comp);

        darrResult.clear();
        CreateDoubleArray(darrResult, comp);
        std::wcout << "fraction = [";
        for (size_t i = 0; i < darrResult.size(); i++)
            std::wcout << (i == 0 ? "" : ", ") << darrResult[i];
        std::wcout << "]" << std::endl;

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetTwoPhaseProp(/*[in]*/     BSTR     property,
        /*[in]*/     VARIANT  phaseLabels,
        /*[in]*/     BSTR     basis,
        /*[in,out]*/ VARIANT* results)
    {
        std::wcout << "raw_GetTwoPhaseProp " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_SetOverallProp(/*[in]*/ BSTR    property,
        /*[in]*/ BSTR    basis,
        /*[in]*/ VARIANT values)
    {
        std::wcout << "raw_SetOverallProp " << std::endl;
        overallProperties[property] = _variant_t(values);
        std::wcout << "Set property " << property << std::endl;
        std::wcout << "overallProperties size " << overallProperties.size() << std::endl;

        for (std::map<CComBSTR, _variant_t>::iterator it = overallProperties.begin(); it != overallProperties.end(); ++it)
        {
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
        return S_OK;
    }

    virtual HRESULT __stdcall raw_SetPresentPhases(/*[in]*/ VARIANT phaseLabels,
        /*[in]*/ VARIANT phaseStatus)
    {
        std::wcout << "raw_SetPresentPhases " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_SetSinglePhaseProp(/*[in]*/ BSTR    property,
        /*[in]*/ BSTR    phaseLabel,
        /*[in]*/ BSTR    basis,
        /*[in]*/ VARIANT values)
    {
        std::wcout << "raw_SetSinglePhaseProp " << std::endl;
        singleProperties[CComBSTR(property)] = _variant_t(values);
        std::wcout << "Set single property " << property << std::endl;
        std::wcout << "singleProperties size " << singleProperties.size() << std::endl;

        for (std::map<CComBSTR, _variant_t>::iterator it = singleProperties.begin(); it != singleProperties.end(); ++it)
        {
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
        return S_OK;
    }

    virtual HRESULT __stdcall raw_SetTwoPhaseProp(/*[in]*/ BSTR    property,
        /*[in]*/ VARIANT phaseLabels,
        /*[in]*/ BSTR    basis,
        /*[in]*/ VARIANT values)
    {
        std::wcout << "raw_SetTwoPhaseProp " << std::endl;
        return E_NOTIMPL;
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
        std::wcout << "raw_SetTwoPhaseProp " << std::endl;
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
        std::wcout << "raw_GetCompoundList " << std::endl;

        _variant_t names_v, compids_v;

        VariantClear(compIds);
        VariantClear(names);

        //arrNames[0] = L"Hexane";  // L"Hydrogen";
        //arrNames[1] = L"Octane";  // L"Carbon monoxide";
        //arrNames[2] = L"Decane";  // L"Methane";
        //arrNames[3] = L"Dodecane";// L"Carbon dioxide";
        //std::vector<BSTR> arrNames(4);
        //arrNames[0] = L"Hydrogen";
        //arrNames[1] = L"Carbon monoxide";
        //arrNames[2] = L"Methane";
        //arrNames[3] = L"Carbon dioxide";
        //std::vector<BSTR> arrNames(2);
        //arrNames[0] = L"Water";
        //arrNames[1] = L"Ethanol";
        CreateSafeArray(m_strarrCompounds, names_v);

        std::vector<BSTR> strarrResult;
        CreateStringArray(strarrResult, names_v);
        std::wcout << "names = [";
        for (size_t i = 0; i < strarrResult.size(); i++)
            std::wcout << (i == 0 ? "" : ", ") << strarrResult[i];
        std::wcout << "]" << std::endl;

        CreateSafeArray(m_strarrCompounds, compids_v);

        strarrResult.clear();
        CreateStringArray(strarrResult, compids_v);
        std::wcout << "compIds = [";
        for (size_t i = 0; i < strarrResult.size(); i++)
            std::wcout << (i == 0 ? "" : ", ") << strarrResult[i];
        std::wcout << "]" << std::endl;

        VariantCopy(compIds, &compids_v.GetVARIANT());
        VariantCopy(names, &names_v.GetVARIANT());

        //_variant_t casnos_v;
        //std::vector<BSTR> arrCasNos(4);
        //arrCasNos[0] = L"1333 - 74 - 0";
        //arrCasNos[1] = L"630 - 08 - 0";
        //arrCasNos[2] = L"74 - 82 - 8";
        //arrCasNos[3] = L"124 - 38 - 9";
        //CreateSafeArray(arrCasNos, casnos_v);
        //casnos_v.Detach();

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
        std::wcout << "raw_GetNumCompounds " << std::endl;
        *num = m_strarrCompounds.size();
        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetPDependentProperty(
        /*[in]*/ VARIANT props,
        /*[in]*/ double pressure,
        /*[in]*/ VARIANT compIds,
        /*[in,out]*/ VARIANT * propVals)
    {
        std::wcout << "raw_GetPDependentProperty " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetPDependentPropList(
        /*[out,retval]*/ VARIANT * props)
    {
        std::wcout << "raw_GetPDependentPropList " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetTDependentProperty(
        /*[in]*/ VARIANT props,
        /*[in]*/ double temperature,
        /*[in]*/ VARIANT compIds,
        /*[in,out]*/ VARIANT * propVals)
    {
        std::wcout << "raw_GetTDependentProperty " << std::endl;
        return E_NOTIMPL;
    }

    virtual HRESULT __stdcall raw_GetTDependentPropList(
        /*[out,retval]*/ VARIANT * props)
    {
        std::wcout << "raw_GetTDependentPropList " << std::endl;
        return E_NOTIMPL;
    }

};

OBJECT_ENTRY_AUTO(__uuidof(daeCapeThermoMaterial), daeCapeThermoMaterial)
