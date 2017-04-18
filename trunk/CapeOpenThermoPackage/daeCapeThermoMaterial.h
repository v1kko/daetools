// daeCapeThermoMaterial.h : Declaration of the daeCapeThermoMaterial

#pragma once
#include "resource.h"       // main symbols
#include <map>
#include <iostream>
#include <vector>
#include <sstream>
#include "DAEToolsCapeOpen_i.h"
#include "auxiliary.h"

#if defined(_WIN32_WCE) && !defined(_CE_DCOM) && !defined(_CE_ALLOW_SINGLE_THREADED_OBJECTS_IN_MTA)
#error "Single-threaded COM objects are not properly supported on Windows CE platform, such as the Windows Mobile platforms that do not include full DCOM support. Define _CE_ALLOW_SINGLE_THREADED_OBJECTS_IN_MTA to force ATL to support creating single-thread COM object's and allow use of it's single-threaded COM object implementations. The threading model in your rgs file was set to 'Free' as that is the only threading model supported in non DCOM Windows CE platforms."
#endif

using namespace ATL;

// daeCapeThermoMaterial
class ATL_NO_VTABLE daeCapeThermoMaterial :
	public CComObjectRootEx<CComSingleThreadModel>,
	public CComCoClass<daeCapeThermoMaterial, &CLSID_daeCapeThermoMaterial>,
    public IDispatchImpl<ICapeThermoMaterial, &__uuidof(ICapeThermoMaterial), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>,
    public IDispatchImpl<ICapeThermoCompounds, &__uuidof(ICapeThermoCompounds), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>,
    public IDispatchImpl<ICapeThermoPhases, &__uuidof(ICapeThermoPhases), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>,
    public IDispatchImpl<IdaeCapeThermoMaterial, &__uuidof(IdaeCapeThermoMaterial), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>,
    public IDispatchImpl<ECapeRoot, &__uuidof(ECapeRoot), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>,
    public IDispatchImpl<ECapeUser, &__uuidof(ECapeUser), &LIBID_DAEToolsCapeOpenLib, /*wMajor =*/ 1, /*wMinor =*/ 0>
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
    COM_INTERFACE_ENTRY(ICapeThermoPhases)
    COM_INTERFACE_ENTRY2(IDispatch, ICapeThermoMaterial)
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
    ComBSTR_ComBSTR_Variant_PropertyMap             m_overallProperties;
    ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap     m_singlePhaseProperties;
    ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap     m_twoPhaseProperties;
    std::vector<BSTR>                               m_strarrCompoundIDs;
    std::vector<BSTR>                               m_strarrCompoundCASNumbers;
    std::map<std::string, daeeThermoPackagePhase>   m_mapAvailablePhases;
    std::map<std::string, eCapePhaseStatus>         m_mapPhasesStatus;

    _bstr_t m_name;             /*!< the name of this object */
    HRESULT m_hr;               /*!< the error code */
    _bstr_t m_errDescription;   /*!< the description of the last error, also used as the error name */
    _bstr_t m_errInterfaceName; /*!< the interface of the last error, e.g. ICapeUnit */
    _bstr_t m_errScope;         /*!< the scope of the last error, e.g. Validate */
    _bstr_t m_errOperation;     /*!< expected operation in ECapeBadInvOrder */
    _bstr_t m_errMoreInfo;      /*!< more info */


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

        return S_OK;
    }

    virtual HRESULT __stdcall raw_CreateMaterial(/*[out,retval]*/ IDispatch** materialObject)
    {
        //std::wcout << "raw_CreateMaterial " << std::endl;

        // Double check this.
        // Create a local object using the CComObject<daeCapeThermoMaterial>::CreateInstance function.
        // Nota bene:
        //   Here, we copied not only all settings but all properties, too. Is that correct?
        //   The function CopyFromMaterial should be used to copy properties from current material.
        CComObject<daeCapeThermoMaterial>* newMaterial = daeCreateThermoMaterial(&m_strarrCompoundIDs,
                                                                                 &m_strarrCompoundCASNumbers,
                                                                                 &m_mapAvailablePhases,
                                                                                 &m_mapPhasesStatus,
                                                                                 &m_overallProperties,
                                                                                 &m_singlePhaseProperties,
                                                                                 &m_twoPhaseProperties);

        if (!newMaterial)
        {
            std::wstringstream ss;
            ss << "Cannot create the the new material object" << std::endl;
            SetCapeError(ECapeUnknownHR, ss.str().c_str(), "CreateMaterial");
            return ECapeUnknownHR;
        }

        // Call QueryInterface to get the ICapeThermoMaterial interface stored in the materialObject pointer sent.
        HRESULT hr = newMaterial->QueryInterface(__uuidof(ICapeThermoMaterial), (void**)materialObject);
        if (FAILED(hr))
        {
            std::wstringstream ss;
            ss << "Cannot obtain ICapeThermoMaterial from the material object" << std::endl;
            SetCapeError(hr, ss.str().c_str(), "CreateMaterial");
            return ECapeUnknownHR;
        }

        // The caller of CreateMaterial() function now has a reference to ICapeThermoMaterial in the materialObject
        // so we can release our newMaterial object. The object will be deleted when the materialObject is released.
        newMaterial->Release();
        newMaterial = NULL;

        return hr;
    }

    virtual HRESULT __stdcall raw_GetOverallProp(/*[in]*/     BSTR property,
                                                 /*[in]*/     BSTR basis,
                                                 /*[in,out]*/ VARIANT* results)
    {
        //std::wcout << "raw_GetOverallProp " << std::endl;        
        //printOverallProperties(L"Available overall properties: ");

        // Internally only lower-case basis is used (i.e. some TPP use 'Mole' and others use 'mole').
        // Therefore, use the lower-case representation to avoid the problems above.
        CComBSTR basis_c(basis);
        basis_c.ToLower();

        if (m_overallProperties.find(basis_c) == m_overallProperties.end())
        {
            std::wstringstream ss;
            ss << "Cannot find basis " << (LPWSTR)basis_c << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetOverallProp");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& bstr_variant_map = m_overallProperties[basis_c];

        if (bstr_variant_map.find(CComBSTR(property)) == bstr_variant_map.end())
        {
            std::wstringstream ss;
            std::cout << "Cannot find property " << property << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetOverallProp");
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
     
        // Nota bene:
        //   P and T are basis independent, therefore use "undefined"
        //   x here represents mole fractions
        // Both should be lowercase!
        BSTR undefined_basis_bstr = basis_to_bstr(eUndefinedBasis).Detach();
        BSTR mole_basis_bstr      = basis_to_bstr(eMole).Detach();

        if (m_overallProperties.find(CComBSTR(undefined_basis_bstr)) == m_overallProperties.end())
        {
            std::wstringstream ss;
            ss << "Cannot find basis " << undefined_basis_bstr << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetOverallTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& undefined_bstr_variant_map = m_overallProperties[undefined_basis_bstr];

        if (m_overallProperties.find(CComBSTR(mole_basis_bstr)) == m_overallProperties.end())
        {
            std::wstringstream ss;
            ss << "Cannot find basis " << mole_basis_bstr << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetOverallTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& mole_bstr_variant_map = m_overallProperties[mole_basis_bstr];

        if (undefined_bstr_variant_map.find(CComBSTR(L"pressure")) == undefined_bstr_variant_map.end())
        {
            std::wstringstream ss;
            ss << "Cannot find overall pressure" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetOverallTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }
        if (undefined_bstr_variant_map.find(CComBSTR(L"temperature")) == undefined_bstr_variant_map.end())
        {
            std::wstringstream ss;
            ss << "Cannot find overall temperature" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetOverallTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }
        if (mole_bstr_variant_map.find(CComBSTR(L"fraction")) == mole_bstr_variant_map.end())
        {
            std::wstringstream ss;
            ss << "Cannot find overall fraction" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetOverallTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }

        _variant_t& temperature_v = undefined_bstr_variant_map[CComBSTR(L"temperature")];
        _variant_t& pressure_v    = undefined_bstr_variant_map[CComBSTR(L"pressure")];
        _variant_t& fraction_v    = mole_bstr_variant_map[CComBSTR(L"fraction")];

        *temperature = double_from_variant(temperature_v);
        *pressure    = double_from_variant(pressure_v);
        VariantCopy(composition, &fraction_v);

        //std::wcout << "Get overall T, P, fraction:" << std::endl;
        //std::wcout << "pressure    = " << *pressure << std::endl;
        //std::wcout << "temperature = " << *temperature << std::endl;
        //print_double_array(L"fraction", fraction_v);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetPresentPhases(/*[in,out]*/ VARIANT* phaseLabels,
                                                   /*[in,out]*/ VARIANT* phaseStatus)
    {
        //std::wcout << "raw_GetPresentPhases " << std::endl;

        _variant_t phaseLabels_v, phaseStatus_v;
        std::vector<BSTR> barrPhaseLabels;
        std::vector<eCapePhaseStatus> earrPhaseStatuses;
        for (std::map<std::string, eCapePhaseStatus>::iterator it = m_mapPhasesStatus.begin(); it != m_mapPhasesStatus.end(); it++)
        {
            barrPhaseLabels.push_back(_bstr_t(it->first.c_str()).Detach());
            earrPhaseStatuses.push_back(it->second);
        }

        CreateSafeArray(barrPhaseLabels, phaseLabels_v);
        CreateSafeArray(earrPhaseStatuses, phaseStatus_v);

        VariantCopy(phaseLabels, &phaseLabels_v);
        VariantCopy(phaseStatus, &phaseStatus_v);

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetSinglePhaseProp(/*[in]*/     BSTR     property,
                                                     /*[in]*/     BSTR     phaseLabel,
                                                     /*[in]*/     BSTR     basis,
                                                     /*[in,out]*/ VARIANT* results)
    {
        //printSinglePhaseProperties(L"GetSinglePhaseProp (available single phase properties):");
        //printTwoPhaseProperties(L"GetSinglePhaseProp (available two phase properties):");
        //printOverallProperties(L"GetSinglePhaseProp (available overall properties):");

        CComBSTR basis_c;        
        if (SysStringLen(basis) == 0)
            basis_c = basis_to_bstr(eUndefinedBasis).Detach();
        else
            basis_c = basis;
        basis_c.ToLower();

        //std::wcout << "raw_GetSinglePhaseProp get '" << property << "', phase '" << phaseLabel << "', basis '" << (LPWSTR)basis_c << "'" << std::endl;

        if (m_singlePhaseProperties.find(CComBSTR(phaseLabel)) == m_singlePhaseProperties.end())
        {
            std::wstringstream ss;
            ss << "Phase " << phaseLabel << " not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetSinglePhaseProp");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_property_map = m_singlePhaseProperties[CComBSTR(phaseLabel)];

        if (bstr_bstr_property_map.find(basis_c) == bstr_bstr_property_map.end())
        {
            std::wstringstream ss;
            ss << "Basis " << (LPWSTR)basis_c << " not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetSinglePhaseProp");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& bstr_property_map = bstr_bstr_property_map[basis_c];

        if (bstr_property_map.find(CComBSTR(property)) == bstr_property_map.end())
        {
            std::wstringstream ss;
            ss << "Property " << property << " not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetSinglePhaseProp");
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

        // Nota bene:
        //   P and T are basis independent, therefore use "undefined"
        //   x here represents mole fractions
        // Both should be lowercase!
        BSTR undefined_basis = basis_to_bstr(eUndefinedBasis).Detach();
        BSTR mole_basis      = basis_to_bstr(eMole).Detach();
        
        if (m_singlePhaseProperties.find(CComBSTR(phaseLabel)) == m_singlePhaseProperties.end())
        {
            std::wstringstream ss;
            ss << "Phase " << phaseLabel << " not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_property_map = m_singlePhaseProperties[CComBSTR(phaseLabel)];

        if (bstr_bstr_property_map.find(CComBSTR(undefined_basis)) == bstr_bstr_property_map.end())
        {
            std::wstringstream ss;
            ss << "Basis " << undefined_basis << " not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& undefined_bstr_property_map = bstr_bstr_property_map[CComBSTR(undefined_basis)];

        if (bstr_bstr_property_map.find(CComBSTR(mole_basis)) == bstr_bstr_property_map.end())
        {
            std::wstringstream ss;
            ss << "Basis " << mole_basis << " not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& mole_bstr_property_map = bstr_bstr_property_map[CComBSTR(mole_basis)];

        if (undefined_bstr_property_map.find(CComBSTR(L"temperature")) == undefined_bstr_property_map.end())
        {
            std::wstringstream ss;
            ss << "Property temperature not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }
        if (undefined_bstr_property_map.find(CComBSTR(L"pressure")) == undefined_bstr_property_map.end())
        {
            std::wstringstream ss;
            ss << "Property pressure not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }
        if (mole_bstr_property_map.find(CComBSTR(L"fraction")) == mole_bstr_property_map.end())
        {
            std::wstringstream ss;
            ss << "Property fraction not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTPFraction");
            return ECapeThrmPropertyNotAvailableHR;
        }

        _variant_t& temperature_v = undefined_bstr_property_map[CComBSTR(L"temperature")];
        _variant_t& pressure_v    = undefined_bstr_property_map[CComBSTR(L"pressure")];
        _variant_t& fraction_v    = mole_bstr_property_map[CComBSTR(L"fraction")];
        
        *temperature = double_from_variant(temperature_v);
        *pressure = double_from_variant(pressure_v);
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
        //printTwoPhaseProperties(L"GetTwoPhaseProp (available properties):");

        CComBSTR basis_c(basis);
        basis_c.ToLower();

        std::vector<BSTR> strarrPhases;
        bool res = CreateStringArray(strarrPhases, _variant_t(phaseLabels));
        if(!res)
        {
            std::wstringstream ss;
            ss << "Cannot create phase labels" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTwoPhaseProp");
            return ECapeThrmPropertyNotAvailableHR;
        }

        _bstr_t bstr_phaseLabel;
        for (size_t i = 0; i < strarrPhases.size(); i++)
            bstr_phaseLabel += _bstr_t(i == 0 ? "" : "-") + _bstr_t(strarrPhases[i]);
        
        CComBSTR phaseLabel = bstr_phaseLabel.GetBSTR();

        if (m_twoPhaseProperties.find(phaseLabel) == m_twoPhaseProperties.end())
        {
            std::wstringstream ss;
            ss << "Phase " << (LPWSTR)phaseLabel << " not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTwoPhaseProp");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_property_map = m_twoPhaseProperties[CComBSTR(phaseLabel)];

        if (bstr_bstr_property_map.find(basis_c) == bstr_bstr_property_map.end())
        {
            std::wstringstream ss;
            ss << "Basis " << (LPWSTR)basis_c << " not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTwoPhaseProp");
            return ECapeThrmPropertyNotAvailableHR;
        }
        ComBSTR_Variant_PropertyMap& bstr_property_map = bstr_bstr_property_map[basis_c];

        if (bstr_property_map.find(CComBSTR(property)) == bstr_property_map.end())
        {
            std::wstringstream ss;
            ss << "Property " << property << " not found" << std::endl;
            SetCapeError(ECapeThrmPropertyNotAvailableHR, ss.str().c_str(), "GetTwoPhaseProp");
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
        //printOverallProperties(L"Available overall properties: ");

        return S_OK;
    }

    virtual HRESULT __stdcall raw_SetPresentPhases(/*[in]*/ VARIANT phaseLabels,
                                                   /*[in]*/ VARIANT phaseStatus)
    {
        //std::wcout << "raw_SetPresentPhases " << std::endl;

        std::vector<BSTR> barrPhaseLabels;
        std::vector<eCapePhaseStatus> earrPhaseStatuses;
        CreateStringArray(barrPhaseLabels, _variant_t(phaseLabels));
        CreateEnumArray(earrPhaseStatuses, _variant_t(phaseStatus));

        m_mapPhasesStatus.clear();
        for (size_t i = 0; i < barrPhaseLabels.size(); i++)
        {
            std::string label = (LPCSTR)bstr_t(barrPhaseLabels[i]);
            m_mapPhasesStatus[label] = earrPhaseStatuses[i];
        }

        return S_OK;
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

        if (!basis)
            basis_c = basis_to_bstr(eUndefinedBasis).Detach();

        m_singlePhaseProperties[phase_c][basis_c][property_c] = _variant_t(values);
        //printSinglePhaseProperties(L"Available single phase properties: ");

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
            std::wstringstream ss;
            ss << "Cannot create phase labels" << std::endl;
            SetCapeError(ECapeUnknownHR, ss.str().c_str(), "SetTwoPhaseProp");
            return ECapeUnknownHR;
        }

        _bstr_t bstr_phaseLabel;
        for (size_t i = 0; i < strarrPhases.size(); i++)
            bstr_phaseLabel += _bstr_t(i == 0 ? "" : "-") + _bstr_t(strarrPhases[i]);

        CComBSTR phaseLabel = bstr_phaseLabel.GetBSTR();
        CComBSTR basis_c(basis); 
        basis_c.ToLower();
        CComBSTR property_c(property);

        if (!basis)
            basis_c = basis_to_bstr(eUndefinedBasis).Detach();

        m_twoPhaseProperties[phaseLabel][basis_c][property_c] = _variant_t(values);
        //printTwoPhaseProperties(L"Available tw phase properties: ");

        return S_OK;
    }

    void printSinglePhaseProperties(BSTR heading, std::wostream& _stdout_ = std::wcout)
    {
        _stdout_ << heading << std::endl;
        for (ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap::iterator fit = m_singlePhaseProperties.begin(); fit != m_singlePhaseProperties.end(); ++fit)
        {
            CComBSTR                             phase                 = fit->first;
            ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_variant_map = fit->second;

            _stdout_ << "    phase '" << (LPWSTR)phase << "':" << std::endl;
            for (ComBSTR_ComBSTR_Variant_PropertyMap::iterator bit = bstr_bstr_variant_map.begin(); bit != bstr_bstr_variant_map.end(); ++bit)
            {
                CComBSTR                     basis            = bit->first;
                ComBSTR_Variant_PropertyMap& bstr_variant_map = bit->second;

                _stdout_ << "      basis '" << (basis ? (LPWSTR)basis : L"") << "':" << std::endl;
                for (ComBSTR_Variant_PropertyMap::iterator it = bstr_variant_map.begin(); it != bstr_variant_map.end(); ++it)
                {
                    _stdout_ << "        ";
                    std::vector<double> darrResult;
                    if (it->second.vt == (VT_ARRAY | VT_R8))
                    {
                        CreateDoubleArray(darrResult, it->second);
                        _stdout_ << it->first.m_str << " = [";
                        for (size_t i = 0; i < darrResult.size(); i++)
                            _stdout_ << (i == 0 ? "" : ", ") << darrResult[i];
                        _stdout_ << "]" << std::endl;
                    }
                    else if (it->second.vt == VT_BSTR)
                    {
                        _stdout_ << it->first.m_str << " = " << it->second.bstrVal << std::endl;
                    }
                    else
                    {
                        _stdout_ << it->first.m_str << " = " << " unknown" << std::endl;
                    }
                }
            }
        }
    }

    void printTwoPhaseProperties(BSTR heading, std::wostream& _stdout_ = std::wcout)
    {
        _stdout_ << heading << std::endl;
        for (ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap::iterator fit = m_twoPhaseProperties.begin(); fit != m_twoPhaseProperties.end(); ++fit)
        {
            CComBSTR                             phase = fit->first;
            ComBSTR_ComBSTR_Variant_PropertyMap& bstr_bstr_variant_map = fit->second;

            _stdout_ << "    phase '" << (LPWSTR)phase << "':" << std::endl;
            for (ComBSTR_ComBSTR_Variant_PropertyMap::iterator bit = bstr_bstr_variant_map.begin(); bit != bstr_bstr_variant_map.end(); ++bit)
            {
                CComBSTR                     basis = bit->first;
                ComBSTR_Variant_PropertyMap& bstr_variant_map = bit->second;

                _stdout_ << "      basis '" << (LPWSTR)basis << "':" << std::endl;
                for (ComBSTR_Variant_PropertyMap::iterator it = bstr_variant_map.begin(); it != bstr_variant_map.end(); ++it)
                {
                    _stdout_ << "        ";
                    std::vector<double> darrResult;
                    if (it->second.vt == (VT_ARRAY | VT_R8))
                    {
                        CreateDoubleArray(darrResult, it->second);
                        _stdout_ << it->first.m_str << " = [";
                        for (size_t i = 0; i < darrResult.size(); i++)
                            _stdout_ << (i == 0 ? "" : ", ") << darrResult[i];
                        _stdout_ << "]" << std::endl;
                    }
                    else if (it->second.vt == VT_BSTR)
                    {
                        _stdout_ << it->first.m_str << " = " << it->second.bstrVal << std::endl;
                    }
                    else
                    {
                        _stdout_ << it->first.m_str << " = " << " unknown" << std::endl;
                    }
                }
            }
        }
    }

    void printOverallProperties(BSTR heading, std::wostream& _stdout_ = std::wcout)
    {
        _stdout_ << heading << std::endl;
        for (ComBSTR_ComBSTR_Variant_PropertyMap::iterator fit = m_overallProperties.begin(); fit != m_overallProperties.end(); ++fit)
        {
            CComBSTR                     basis = fit->first;
            ComBSTR_Variant_PropertyMap& bstr_variant_map = fit->second;

            _stdout_ << "    basis '" << (LPWSTR)basis << "':" << std::endl;
            for (ComBSTR_Variant_PropertyMap::iterator it = bstr_variant_map.begin(); it != bstr_variant_map.end(); ++it)
            {
                _stdout_ << "      ";
                std::vector<double> darrResult;
                if (it->second.vt == (VT_ARRAY | VT_R8))
                {
                    CreateDoubleArray(darrResult, it->second);
                    _stdout_ << it->first.m_str << " = [";
                    for (size_t i = 0; i < darrResult.size(); i++)
                        _stdout_ << (i == 0 ? "" : ", ") << darrResult[i];
                    _stdout_ << "]" << std::endl;
                }
                else if (it->second.vt == VT_BSTR)
                {
                    _stdout_ << it->first.m_str << " = " << it->second.bstrVal << std::endl;
                }
                else
                {
                    _stdout_ << it->first.m_str << " = " << " unknown" << std::endl;
                }
            }
        }
    }

    void printAllProperties(BSTR heading, std::wostream& _stdout_ = std::wcout)
    {
        _stdout_ << heading << std::endl;
        printSinglePhaseProperties(L"- singlePhaseProperties:", _stdout_);
        printTwoPhaseProperties(L"- twoPhaseProperties:", _stdout_);
        printOverallProperties(L"- overallProperties:", _stdout_);
    }

    //ICapeThermoPhases functions
    virtual HRESULT __stdcall raw_GetNumPhases(/*[out,retval]*/ long * num)
    {
        //std::wcout << "raw_GetNumPhases " << std::endl;
        *num = m_mapAvailablePhases.size();
        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetPhaseInfo(/*[in]*/         BSTR      phaseLabel,
                                               /*[in]*/         BSTR      phaseAttribute,
                                               /*[out,retval]*/ VARIANT * value)
    {
        //std::wcout << "raw_GetPhaseInfo " << std::endl;
        return ECapeNoImplHR;
    }

    virtual HRESULT __stdcall raw_GetPhaseList(/*[in,out]*/ VARIANT * phaseLabels,
                                               /*[in,out]*/ VARIANT * stateOfAggregation,
                                               /*[in,out]*/ VARIANT * keyCompoundId)
    {
        _variant_t phaseLabels_v, stateOfAggregation_v, keyCompoundId_v;
        std::vector<BSTR> bstrarrPhaseLabels, bstrarrStateOfAggregation;
        for (std::map<std::string, daeeThermoPackagePhase>::iterator it = m_mapAvailablePhases.begin(); it != m_mapAvailablePhases.end(); it++)
        {
            bstrarrPhaseLabels.push_back(_bstr_t(it->first.c_str()).Detach());
            bstrarrStateOfAggregation.push_back(phase_to_bstr(it->second).Detach());
        }

        CreateSafeArray(bstrarrPhaseLabels, phaseLabels_v);
        CreateSafeArray(bstrarrStateOfAggregation, stateOfAggregation_v);
        keyCompoundId_v = L"UNDEFINED";

        //print_string_array(L"compound_names", names_v);
        print_string_array(L"Available phase labels in the material object: ", phaseLabels_v);
        print_string_array(L"Available phase states of aggregation in the material object: ", stateOfAggregation_v);

        VariantCopy(phaseLabels, &phaseLabels_v.GetVARIANT());
        VariantCopy(stateOfAggregation, &phaseLabels_v.GetVARIANT());
        VariantCopy(keyCompoundId, &keyCompoundId_v.GetVARIANT());

        return S_OK;
    }

    // ICapeThermoCompounds functions
    virtual HRESULT __stdcall raw_GetCompoundConstant(/*[in]*/ VARIANT props,
                                                      /*[in]*/ VARIANT compIds,
                                                      /*[out,retval]*/ VARIANT * propVals)
    {
        //std::wcout << "raw_GetCompoundConstant " << std::endl;
        return ECapeNoImplHR;
    }

    virtual HRESULT __stdcall raw_GetCompoundList(/*[in,out]*/ VARIANT * compIds,
                                                  /*[in,out]*/ VARIANT * formulae,
                                                  /*[in,out]*/ VARIANT * names,
                                                  /*[in,out]*/ VARIANT * boilTemps,
                                                  /*[in,out]*/ VARIANT * molwts,
                                                  /*[in,out]*/ VARIANT * casnos)
    {
        //std::wcout << "raw_GetCompoundList " << std::endl;

        _variant_t names_v, compids_v, compcasnos_v;

        CreateSafeArray(m_strarrCompoundIDs,        names_v);
        CreateSafeArray(m_strarrCompoundIDs,        compids_v);
        CreateSafeArray(m_strarrCompoundCASNumbers, compcasnos_v);

        //print_string_array(L"compound_names", names_v);
        //print_string_array(L"Available compound ids in the material object: ", compids_v);
        //print_string_array(L"Available compound CAS numbers in the material object: ", compcasnos_v);

        VariantCopy(names,   &names_v.GetVARIANT());
        VariantCopy(compIds, &compids_v.GetVARIANT());
        VariantCopy(casnos,  &compcasnos_v.GetVARIANT());

        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetConstPropList(/*[out,retval]*/ VARIANT * props)
    {
        //std::wcout << "raw_GetConstPropList " << std::endl;
        return ECapeNoImplHR;
    }

    virtual HRESULT __stdcall raw_GetNumCompounds(/*[out,retval]*/ long * num)
    {
        //std::wcout << "raw_GetNumCompounds " << std::endl;
        *num = m_strarrCompoundIDs.size();
        return S_OK;
    }

    virtual HRESULT __stdcall raw_GetPDependentProperty(/*[in]*/ VARIANT props,
                                                        /*[in]*/ double pressure,
                                                        /*[in]*/ VARIANT compIds,
                                                        /*[in,out]*/ VARIANT * propVals)
    {
        //std::wcout << "raw_GetPDependentProperty " << std::endl;
        return ECapeNoImplHR;
    }

    virtual HRESULT __stdcall raw_GetPDependentPropList(/*[out,retval]*/ VARIANT * props)
    {
        //std::wcout << "raw_GetPDependentPropList " << std::endl;
        return ECapeNoImplHR;
    }

    virtual HRESULT __stdcall raw_GetTDependentProperty(/*[in]*/ VARIANT props,
                                                        /*[in]*/ double temperature,
                                                        /*[in]*/ VARIANT compIds,
                                                        /*[in,out]*/ VARIANT * propVals)
    {
        //std::wcout << "raw_GetTDependentProperty " << std::endl;
        return ECapeNoImplHR;
    }

    virtual HRESULT __stdcall raw_GetTDependentPropList(/*[out,retval]*/ VARIANT * props)
    {
        //std::wcout << "raw_GetTDependentPropList " << std::endl;
        return ECapeNoImplHR;
    }


    void SetCapeError(HRESULT hr, 
                      _bstr_t errDescription,
                      _bstr_t errScope,
                      _bstr_t errInterfaceName = L"ICapeThermoMaterial",
                      _bstr_t errOperation = L"N/A",
                      _bstr_t errMoreInfo = L"N/A")
    {
        m_hr                = hr;
        m_errDescription    = errDescription;
        m_errInterfaceName  = errInterfaceName;
        m_errScope          = errScope;    
        m_errOperation      = errOperation;
        m_errMoreInfo       = errMoreInfo;

        std::wstringstream ss;
        ss << std::endl << "DAE Tools ICapeThermoMaterial implementation." << std::endl;
        printSinglePhaseProperties(L"Available single phase properties:", ss);
        printTwoPhaseProperties(L"Available two phase properties:", ss);
        printOverallProperties(L"Available overall properties:", ss);
        m_errDescription += ss.str().c_str();
    }

    // ECapeRoot part
    virtual HRESULT __stdcall get_name(/*[out,retval]*/ BSTR * name)
    {
        *name = m_name.copy();
        return S_OK;
    }

    // ECapeUser part
    virtual HRESULT __stdcall get_code(/*[out,retval]*/ long * code)
    {
        *code = m_hr;
        return S_OK;
    }

    virtual HRESULT __stdcall get_description(/*[out,retval]*/ BSTR * description)
    {
        *description = m_errDescription.copy();
        return S_OK;
    }

    virtual HRESULT __stdcall get_scope(/*[out,retval]*/ BSTR * scope)
    {
        *scope = m_errScope.copy();
        return S_OK;
    }

    virtual HRESULT __stdcall get_interfaceName(/*[out,retval]*/ BSTR * interfaceName)
    {
        *interfaceName = m_errInterfaceName.copy();
        return S_OK;
    }

    virtual HRESULT __stdcall get_operation(/*[out,retval]*/ BSTR * operation)
    {
        *operation = m_errOperation.copy();
        return S_OK;
    }

    virtual HRESULT __stdcall get_moreInfo(/*[out,retval]*/ BSTR * moreInfo)
    {
        *moreInfo = m_errDescription.copy();
        return S_OK;
    }

};

OBJECT_ENTRY_AUTO(__uuidof(daeCapeThermoMaterial), daeCapeThermoMaterial)
