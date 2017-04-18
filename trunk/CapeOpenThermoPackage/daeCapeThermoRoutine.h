#pragma once
#include "resource.h"       // main symbols
#include <map>
#include <iostream>
#include <vector>
#include <sstream>
#include "daeCapeThermoMaterial.h"

using namespace ATL;

class DAE_CAPE_OPEN_API daeCapeThermoPropertyRoutine : public dae::tpp::daeThermoPhysicalPropertyPackage_t
{
public:
    daeCapeThermoPropertyRoutine()
    {
        material = NULL;
    }

    void LoadPackage(const std::string& strPackageManager,
                     const std::string& strPackageName,
                     const std::vector<std::string>& strarrCompoundIDs,
                     const std::vector<std::string>& strarrCompoundCASNumbers,
                     const std::map<std::string, daeeThermoPackagePhase>& mapPhases,
                     daeeThermoPackageBasis defaultBasis = eMole,
                     const std::map<std::string, std::string>& mapOptions = std::map<std::string, std::string>())
    {
        if (strarrCompoundIDs.size() == 0)
        {
            std::stringstream ss;
            ss << "Invalid number of compounds IDs (0) specified in the LoadPackage function"<< std::endl;
            throw std::runtime_error(ss.str().c_str());
        }
        if (strarrCompoundIDs.size() != 0 && 
            strarrCompoundCASNumbers.size() != 0 && 
            strarrCompoundCASNumbers.size() != strarrCompoundIDs.size())
        {
            std::stringstream ss;
            ss << "Invalid number of compounds IDs(" << strarrCompoundIDs.size() << ") and CAS numbers (" << strarrCompoundCASNumbers.size() 
               << ") specified in the LoadPackage function" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }
        if (mapPhases.size() == 0)
        {
            std::stringstream ss;
            ss << "Invalid number of available phases (" << mapPhases.size() << ") specified in the LoadPackage function" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }

        m_strarrCompoundIDs        = strarrCompoundIDs;
        m_strarrCompoundCASNumbers = strarrCompoundCASNumbers;
        m_mapPhases                = mapPhases;
        m_defaultBasis             = defaultBasis;

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

            // Instantiate manager. Continue if some error occurs.
            try
            {
                CreateTPPManager(&availableObject, manager, identification);
            }
            catch (_com_error& e)
            {
                continue;
            }

            // Get the list of available packages.
            std::vector<BSTR> strarrPackages;
            _variant_t pplist = manager->GetPropertyPackageList();
            CreateStringArray(strarrPackages, pplist);

            // Print info about the manager and its packages (optional).
            //print_thermo_manager_info(manager, identification);

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
        if (!object)
        {
            std::stringstream ss;
            ss << "Cannot find thermo physical property package manager: " << strPackageManager << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }
        try
        {
            // 3. The thermo manager was found. Now instantiate it and store its 
            //    ICapeThermoPropertyPackageManager and ICapeIdentification interfaces 
            hr = CreateTPPManager(object, manager, identification);
        }
        catch (_com_error& e)
        {
            std::stringstream ss;
            ss << "Instantiation of ICapeThermoPropertyPackageManager: '" << strPackageManager << "' failed with error-code = " << e.Error() << std::endl;
            ss << (LPCSTR)_bstr_t(e.ErrorMessage()) << std::endl;
            ss << (LPCSTR)e.Description() << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }
        if (FAILED(hr))
        {
            std::stringstream ss;
            ss << "Cannot instantiate thermo physical property package manager: '" << strPackageManager << "', error-code = " << hr << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }

        // 4. Create thermo package with the specified name
        _bstr_t packageName(strPackageName.c_str());
        package = manager->GetPropertyPackage(packageName);
        if (package == NULL)
        {
            std::stringstream ss;
            ss << "Cannot create thermo physical property package: '" << strPackageName << "'" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }

        // 5. Get the ICapeThermoCompounds interface from the package.
        //    Check whether the thermo package supports our compounds.
        hr = CreateCompounds(package, compounds);
        if (FAILED(hr))
        {
            std::stringstream ss;
            ss << "Cannot obtain ICapeThermoCompounds interface from the package: '" << strPackageName << "'" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }
        if (m_strarrCompoundIDs.size() > compounds->GetNumCompounds())
            throw std::runtime_error("The number of compounds in the thermo package is lower than requested");

        //_variant_t compIds, formulae, names, boilTemps, molwts, casnos;
        //hr = compounds->GetCompoundList(&compIds, &formulae, &names, &boilTemps, &molwts, &casnos);
        //if (FAILED(hr))
        //{
        //    std::stringstream ss;
        //    ss << "GetCompoundList failed" << std::endl;
        //    throw std::runtime_error(ss.str().c_str());
        //}

        // Get the ECapeUser error interface from the package
        hr = CreateErrorInterfaces(package, errorPackageCapeUser);
        if (FAILED(hr))
        {
            std::stringstream ss;
            ss << "Cannot obtain error interfaces from the package: '" << strPackageName << "'" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }

        // 6. Get the ICapeThermoPropertyRoutine interface from the package.
        hr = CreatePropertyRoutine(package, propertyRoutine);
        if (FAILED(hr))
        {
            std::stringstream ss;
            ss << "Cannot obtain ICapeThermoPropertyRoutine interface from the package: '" << strPackageName << "'" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }

        // 7. Create our ICapeThermoMaterial implementation (to be sent to the package's ICapeMaterialContext).
        std::vector<BSTR> bstrCompoundIDs, bstrCompoundCASNumbers;
        for (size_t i = 0; i < m_strarrCompoundIDs.size(); i++)
            bstrCompoundIDs.push_back( _bstr_t(m_strarrCompoundIDs[i].c_str()).Detach() );
        for (size_t i = 0; i < strarrCompoundCASNumbers.size(); i++)
            bstrCompoundCASNumbers.push_back(_bstr_t(strarrCompoundCASNumbers[i].c_str()).Detach());

        material = daeCreateThermoMaterial(&bstrCompoundIDs, &bstrCompoundCASNumbers, &m_mapPhases, NULL, NULL, NULL, NULL);
        if (!material)
        {
            std::stringstream ss;
            ss << "Cannot create new material" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }

        // Get the ECapeUser error interface from the material
        hr = CreateErrorInterfaces(package, errorMaterialCapeUser);
        if (FAILED(hr))
        {
            std::stringstream ss;
            ss << "Cannot obtain error interfaces from the package: '" << strPackageName << "'" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }

        hr = material->QueryInterface(__uuidof(ICapeThermoMaterial), (void**)&dispatchMaterial.GetInterfacePtr());
        if (FAILED(hr))
        {
            std::stringstream ss;
            ss << "Cannot obtain ICapeThermoMaterial interface from the material" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }

        // 8. Get the ICapeMaterialContext interface from the package and set its material. 
        hr = CreateMaterialContext(package, materialContext);
        if (FAILED(hr))
        {
            std::stringstream ss;
            ss << "Cannot obtain ICapeMaterialContext interface from the material" << std::endl;
            throw std::runtime_error(ss.str().c_str());
        }

        try
        {
            hr = materialContext->SetMaterial(dispatchMaterial.GetInterfacePtr());
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "ICapeThermoMaterialContext::SetMaterial failed" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        }
   }

    virtual ~daeCapeThermoPropertyRoutine()
    {
        manager.Release();
        identification.Release();
        package.Release();
        compounds.Release();
        propertyRoutine.Release();
        if(material)
            material->Release();
        dispatchMaterial.Release();
        materialContext.Release();
        errorMaterialCapeUser.Release();
        errorPackageCapeUser.Release();
    }

    std::string GetTPPName()
    {
        return std::string("CapeOpen TPP");
    }

    HRESULT Set_SinglePhase_PTx(double P, double T, const std::vector<double>& x, _bstr_t& phase_bstr)
    {
        HRESULT hr;
        _variant_t pressure_v, temperature_v, fraction_v;

        // Nota bene:
        //   P and T are basis independent, therefore use "undefined"
        //   x here represents mole fractions
        // Both should be lowercase!
        _bstr_t undefined_basis_bstr = basis_to_bstr(eUndefinedBasis);
        _bstr_t mole_basis_bstr      = basis_to_bstr(eMole);

        std::vector<double> pressure(1), temperature(1), fraction;
        pressure[0] = P;
        temperature[0] = T;
        fraction = x;

        CreateSafeArray(pressure, pressure_v);
        CreateSafeArray(temperature, temperature_v);
        CreateSafeArray(fraction, fraction_v);

        // Nota bene:
        //   It would be more efficient if I set these data directly to my material object
        //   rather than indirectly through SetSinglePhaseProp function calls.
        // These functions will throw an exception if some error occurs.
        // No need to catch it here - the calling routine will do it.
        hr = material->SetSinglePhaseProp(L"pressure", phase_bstr, undefined_basis_bstr, pressure_v);
        if (FAILED(hr))
            return hr;

        hr = material->SetSinglePhaseProp(L"temperature", phase_bstr, undefined_basis_bstr, temperature_v);
        if (FAILED(hr))
            return hr;

        hr = material->SetSinglePhaseProp(L"fraction", phase_bstr, mole_basis_bstr, fraction_v);
        if (FAILED(hr))
            return hr;

        return S_OK;
    }

    // ICapeThermoCompounds interface
    double GetCompoundConstant(const std::string& property, const std::string& compound)
    {
        _bstr_t property_bstr = property.c_str();
        _bstr_t compound_bstr = compound.c_str();

        _variant_t properties_v = create_array_from_string(property_bstr);
        _variant_t compids_v = create_array_from_string(compound_bstr);

        _variant_t cprops_v = compounds->GetCompoundConstant(properties_v, compids_v);
        // Returned value type is VT_ARRAY | VT_VARIANT: it is an array of variants
        // and its size should be 1 since we requested a single property.
        std::vector<VARIANT> varrResult;
        bool res = CreateVariantArray(varrResult, cprops_v);
        if(!res || varrResult.size() != 1)
            throw std::runtime_error("Invalid return type from GetCompoundConstant");
        variant_t value_t = varrResult[0];
        return (double)value_t;
    }

    double GetTDependentProperty(const std::string& property, double T, const std::string& compound)
    {
        HRESULT hr;

        _bstr_t property_bstr = property.c_str();
        _bstr_t compound_bstr = compound.c_str();

        _variant_t properties_v = create_array_from_string(property_bstr);
        _variant_t compids_v    = create_array_from_string(compound_bstr);
        _variant_t results_v;

        try
        {
            hr = compounds->GetTDependentProperty(properties_v, T, compids_v, &results_v);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "GetTDependentProperty failed to calculate the property '" << (LPWSTR)property_bstr
               << "' requested for the compound '" << (LPWSTR)compound_bstr << "' (or returned empty results)" << std::endl;

            _variant_t property_list_v;
            std::vector<BSTR> arrPropertyList;
            hr = compounds->raw_GetTDependentPropList(&property_list_v);
            CreateStringArray(arrPropertyList, property_list_v);
            ss << "Available properties from the thermo package:" << std::endl;
            ss << "[";
            for (size_t i = 0; i < arrPropertyList.size(); i++)
                ss << (i == 0 ? "" : ", ") << arrPropertyList[i];
            ss << "]" << std::endl;

            DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        }

        if (results_v.vt == VT_EMPTY)
        {
            std::wstringstream ss;
            ss << "GetTDependentProperty returned empty results for the property '" << (LPWSTR)property_bstr
                << "' requested for the compound '" << (LPWSTR)compound_bstr << "'" << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }

        return double_from_variant(results_v);
    }

    double GetPDependentProperty(const std::string& property, double P, const std::string& compound)
    {
        HRESULT hr;

        _bstr_t property_bstr = property.c_str();
        _bstr_t compound_bstr = compound.c_str();

        _variant_t properties_v = create_array_from_string(property_bstr);
        _variant_t compids_v    = create_array_from_string(compound_bstr);
        _variant_t results_v;

        try
        {
            hr = compounds->GetPDependentProperty(properties_v, P, compids_v, &results_v);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "GetPDependentProperty failed to calculate the property '" << (LPWSTR)property_bstr 
               << "' requested for the compound '" << (LPWSTR)compound_bstr << "'" << std::endl;

            _variant_t property_list_v;
            std::vector<BSTR> arrPropertyList;
            hr = compounds->raw_GetPDependentPropList(&property_list_v);
            CreateStringArray(arrPropertyList, property_list_v);
            ss << "Available properties from the thermo package:" << std::endl;
            ss << "[";
            for (size_t i = 0; i < arrPropertyList.size(); i++)
                ss << (i == 0 ? "" : ", ") << arrPropertyList[i];
            ss << "]" << std::endl;

            DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        }

        if (results_v.vt == VT_EMPTY)
        {
            std::wstringstream ss;
            ss << "GetPDependentProperty returned empty results for the property '" << (LPWSTR)property_bstr
                << "' requested for the compound '" << (LPWSTR)compound_bstr << "'" << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }

        return double_from_variant(results_v);
    }

    // ICapeThermoPropertyRoutine interface
    double CalcSinglePhaseScalarProperty(const std::string& property,
                                         double P,
                                         double T,
                                         const std::vector<double>& x,
                                         const std::string& phase,
                                         daeeThermoPackageBasis basis = eMole)
    {
        HRESULT hr;

        _bstr_t property_bstr = property.c_str();
        _bstr_t phase_bstr = phase.c_str();
        _bstr_t basis_bstr = basis_to_bstr(basis);

        if(m_mapPhases.find(phase) == m_mapPhases.end())
        {
            std::wstringstream ss;
            ss << "CalcSinglePhaseScalarProperty: the property '" << (LPWSTR)property_bstr << "' requested for the phase '" << (LPWSTR)phase_bstr 
               << "' not found in the material object" << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }

        try
        {
            // Nota bene:
            //   It would be more efficient if I set these data directly to my material object
            //   rather than indirectly through SetSinglePhaseProp function calls.
            hr = Set_SinglePhase_PTx(P, T, x, phase_bstr);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcSinglePhaseScalarProperty: Cannot set T, P or x" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorMaterialCapeUser);
        }

        _variant_t results_v;
        _variant_t properties_v = create_array_from_string(property_bstr);

        //try
        //{
        //    VARIANT_BOOL valid = propertyRoutine->CheckSinglePhasePropSpec(property_bstr, phase_bstr);
        //    if (!valid)
        //        _com_issue_errorex(ECapeUnknownHR, propertyRoutine, __uuidof(propertyRoutine));
        //}
        //catch (_com_error& e)
        //{
        //    std::wstringstream ss;
        //    ss << "CalcSinglePhaseScalarProperty: Checking the property specification returned FALSE" << std::endl;
        //    ss << "The property '" << (LPWSTR)property_bstr << "' cannot be calculated for the phase '" << (LPWSTR)phase_bstr << "'" << std::endl;
        //    DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        //}

        try
        {
            hr = propertyRoutine->CalcSinglePhaseProp(properties_v, phase_bstr);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcSinglePhaseScalarProperty: Calculation of the property failed" << std::endl;
            ss << "The property '" << (LPWSTR)property_bstr << "' cannot be calculated for the phase: " << (LPWSTR)phase_bstr << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        }

        try
        {
            // Nota bene:
            //   It would be more efficient if I get these data directly to my material object
            //   rather than indirectly through SetSinglePhaseProp function calls.
            hr = material->GetSinglePhaseProp(property_bstr, phase_bstr, basis_bstr, &results_v);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcSinglePhaseScalarProperty failed to obtain the property '" << (LPWSTR)property_bstr 
               << "' from the material for the phase '" << (LPWSTR)phase_bstr << "' and basis '" << (LPWSTR)basis_bstr << "'" << std::endl;
            ss << "Perhaps it was calculated for different basis (eMole, eMass, eUndefinedBasis)?" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorMaterialCapeUser);
        }

        return double_from_variant(results_v);
    }

    void CalcSinglePhaseVectorProperty(const std::string& property,
                                       double P,
                                       double T,
                                       const std::vector<double>& x,
                                       const std::string& phase,
                                       std::vector<double>& results,
                                       daeeThermoPackageBasis basis = eMole)
    {
        HRESULT hr;

        _bstr_t property_bstr = property.c_str();
        _bstr_t phase_bstr = phase.c_str();
        _bstr_t basis_bstr = basis_to_bstr(basis);

        if (m_mapPhases.find(phase) == m_mapPhases.end())
        {
            std::wstringstream ss;
            ss << "CalcSinglePhaseVectorProperty '" << (LPWSTR)property_bstr << "' requested for the phase '" << (LPWSTR)phase_bstr
                << "' not found in the material object" << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }

        try
        {
            // Nota bene:
            //   It would be more efficient if I set these data directly to my material object
            //   rather than indirectly through SetSinglePhaseProp function calls.
            hr = Set_SinglePhase_PTx(P, T, x, phase_bstr);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcSinglePhaseVectorProperty: Cannot set T, P or x" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorMaterialCapeUser);
        }

        _variant_t results_v;
        _variant_t properties_v = create_array_from_string(property_bstr);

        //try
        //{
        //    VARIANT_BOOL valid = propertyRoutine->CheckSinglePhasePropSpec(property_bstr, phase_bstr);
        //    if (!valid)
        //        _com_issue_errorex(ECapeUnknownHR, propertyRoutine, __uuidof(propertyRoutine));
        //}
        //catch (_com_error& e)
        //{
        //    std::wstringstream ss;
        //    ss << "CalcSinglePhaseVectorProperty: Checking the property specification returned FALSE" << std::endl;
        //    ss << "The property '" << (LPWSTR)property_bstr << "' cannot be calculated for the phase '" << (LPWSTR)phase_bstr << "'" << std::endl;
        //    DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        //}

        try
        {
            hr = propertyRoutine->CalcSinglePhaseProp(properties_v, phase_bstr);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcSinglePhaseVectorProperty: Calculation of the property failed" << std::endl;
            ss << "The property '" << (LPWSTR)property_bstr << "' cannot be calculated for the phase: " << (LPWSTR)phase_bstr << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        }

        try
        {
            // Nota bene:
            //   It would be more efficient if I get these data directly to my material object
            //   rather than indirectly through SetSinglePhaseProp function calls.
            hr = material->GetSinglePhaseProp(property_bstr, phase_bstr, basis_bstr, &results_v);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcSinglePhaseVectorProperty failed to obtain the property '" << (LPWSTR)property_bstr
                << "' from the material for the phase '" << (LPWSTR)phase_bstr << "' and basis '" << (LPWSTR)basis_bstr << "'" << std::endl;
            ss << "Perhaps it was calculated for different basis (eMole, eMass, eUndefinedBasis)?" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorMaterialCapeUser);
        }

        results.clear();
        bool res = CreateDoubleArray(results, results_v);
        if(!res)
        {
            std::wstringstream ss;
            ss << "CalcSinglePhaseVectorProperty: Cannot get an array from the results for the property '" << (LPWSTR)property_bstr << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }
    }

    double CalcTwoPhaseScalarProperty(const std::string& property,
                                      double P1, double T1, const std::vector<double>& x1,
                                      const std::string& phase1,
                                      double P2, double T2, const std::vector<double>& x2,
                                      const std::string& phase2,
                                      daeeThermoPackageBasis basis = eMole)
    {
        HRESULT hr;

        _bstr_t property_bstr = property.c_str();
        _bstr_t phase1_bstr = phase1.c_str();
        _bstr_t phase2_bstr = phase2.c_str();
        _bstr_t basis_bstr = basis_to_bstr(basis);

        _variant_t phases_v;
        std::vector<BSTR> phases(2);
        phases[0] = phase1_bstr.GetBSTR();
        phases[1] = phase2_bstr.GetBSTR();
        CreateSafeArray(phases, phases_v);

        if(m_mapPhases.find(phase1) == m_mapPhases.end())
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseScalarProperty '" << (LPWSTR)property_bstr << "' requested for the phase '" << (LPWSTR)phase1_bstr
               << "' not found in the material object" << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }
        if (m_mapPhases.find(phase2) == m_mapPhases.end())
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseScalarProperty '" << (LPWSTR)property_bstr << "' requested for the phase '" << (LPWSTR)phase2_bstr
                << "' not found in the material object" << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }

        try
        {
            // Nota bene:
            //   It would be more efficient if I set these data directly to my material object
            //   rather than indirectly through SetTwoPhaseProp function calls.
            hr = Set_SinglePhase_PTx(P1, T1, x1, phase1_bstr);
            hr = Set_SinglePhase_PTx(P2, T2, x2, phase2_bstr);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseScalarProperty: Cannot set T, P or x" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorMaterialCapeUser);
        }

        _variant_t results_v;
        _variant_t properties_v = create_array_from_string(property_bstr);

        //try
        //{
        //    VARIANT_BOOL valid = propertyRoutine->CheckTwoPhasePropSpec(property_bstr, phases_v);
        //    if (!valid)
        //        _com_issue_errorex(ECapeUnknownHR, propertyRoutine, __uuidof(propertyRoutine));
        //}
        //catch (_com_error& e)
        //{
        //    std::wstringstream ss;
        //    ss << "CalcTwoPhaseScalarProperty: Checking the property specification returned FALSE" << std::endl;
        //    ss << "The property '" << (LPWSTR)property_bstr << "' cannot be calculated for the phases '" 
        //       << (LPWSTR)phase1_bstr << "-" << (LPWSTR)phase2_bstr << "'" << std::endl;
        //    DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        //}

        try
        {
            hr = propertyRoutine->CalcTwoPhaseProp(properties_v, phases_v);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseScalarProperty: Calculation of the property failed" << std::endl;
            ss << "The property '" << (LPWSTR)property_bstr << "' cannot be calculated for the phases '" 
               << (LPWSTR)phase1_bstr << "-" << (LPWSTR)phase2_bstr << "'" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        }

        try
        {
            // Nota bene:
            //   It would be more efficient if I get these data directly to my material object
            //   rather than indirectly through SetSinglePhaseProp function calls.
            hr = material->GetTwoPhaseProp(property_bstr, phases_v, basis_bstr, &results_v);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseScalarProperty failed to obtain the property '" << (LPWSTR)property_bstr 
               << "' from the material for the phases '" << (LPWSTR)phase1_bstr << "-" << (LPWSTR)phase2_bstr << "'"
               << "' and basis '" << (LPWSTR)basis_bstr << "'" << std::endl;
            ss << "Perhaps it was calculated for different basis (eMole, eMass, eUndefinedBasis)?" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorMaterialCapeUser);
        }

        return double_from_variant(results_v);
    }

    void CalcTwoPhaseVectorProperty(const std::string& property,
                                    double P1, double T1, const std::vector<double>& x1,
                                    const std::string& phase1,
                                    double P2, double T2, const std::vector<double>& x2,
                                    const std::string& phase2,
                                    std::vector<double>& results,
                                    daeeThermoPackageBasis basis = eMole)
    {
        HRESULT hr;

        _bstr_t property_bstr = property.c_str();
        _bstr_t phase1_bstr = phase1.c_str();
        _bstr_t phase2_bstr = phase2.c_str();
        _bstr_t basis_bstr = basis_to_bstr(basis);

        _variant_t phases_v;
        std::vector<BSTR> phases(2);
        phases[0] = phase1_bstr.GetBSTR();
        phases[1] = phase2_bstr.GetBSTR();
        CreateSafeArray(phases, phases_v);

        if (m_mapPhases.find(phase1) == m_mapPhases.end())
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseVectorProperty '" << (LPWSTR)property_bstr << "' requested for the phase '" << (LPWSTR)phase1_bstr
                << "' not found in the material object" << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }
        if (m_mapPhases.find(phase2) == m_mapPhases.end())
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseVectorProperty '" << (LPWSTR)property_bstr << "' requested for the phase '" << (LPWSTR)phase2_bstr
                << "' not found in the material object" << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }

        try
        {
            // Nota bene:
            //   It would be more efficient if I set these data directly to my material object
            //   rather than indirectly through SetTwoPhaseProp function calls.
            hr = Set_SinglePhase_PTx(P1, T1, x1, phase1_bstr);
            hr = Set_SinglePhase_PTx(P2, T2, x2, phase2_bstr);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseVectorProperty: Cannot set T, P or x" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorMaterialCapeUser);
        }

        _variant_t results_v;
        _variant_t properties_v = create_array_from_string(property_bstr);

        //try
        //{
        //    VARIANT_BOOL valid = propertyRoutine->CheckTwoPhasePropSpec(property_bstr, phases_v);
        //    if (!valid)
        //        _com_issue_errorex(ECapeUnknownHR, propertyRoutine, __uuidof(propertyRoutine));
        //}
        //catch (_com_error& e)
        //{
        //    std::wstringstream ss;
        //    ss << "TwoPhaseVectorProperty: Checking the property specification returned FALSE" << std::endl;
        //    ss << "The property '" << (LPWSTR)property_bstr << "' cannot be calculated for the phases '"
        //        << (LPWSTR)phase1_bstr << "-" << (LPWSTR)phase2_bstr << "'" << std::endl;
        //    DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        //}

        try
        {
            hr = propertyRoutine->CalcTwoPhaseProp(properties_v, phases_v);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseVectorProperty: Calculation of the property failed" << std::endl;
            ss << "The property '" << (LPWSTR)property_bstr << "' cannot be calculated for the phases '"
                << (LPWSTR)phase1_bstr << "-" << (LPWSTR)phase2_bstr << "'" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorPackageCapeUser);
        }

        try
        {
            // Nota bene:
            //   It would be more efficient if I get these data directly to my material object
            //   rather than indirectly through SetSinglePhaseProp function calls.
            hr = material->GetTwoPhaseProp(property_bstr, phases_v, basis_bstr, &results_v);
        }
        catch (_com_error& e)
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseVectorProperty failed to obtain the property '" << (LPWSTR)property_bstr
                << "' from the material for the phases '" << (LPWSTR)phase1_bstr << "-" << (LPWSTR)phase2_bstr << "'"
                << "' and basis '" << (LPWSTR)basis_bstr << "'" << std::endl;
            ss << "Perhaps it was calculated for different basis (eMole, eMass, eUndefinedBasis)?" << std::endl;
            DAE_THROW_EXCEPTION(ss, e, errorMaterialCapeUser);
        }

        results.clear();
        bool res = CreateDoubleArray(results, results_v);
        if (!res)
        {
            std::wstringstream ss;
            ss << "CalcTwoPhaseVectorProperty: Cannot get an array from the results for the property '" << (LPWSTR)property_bstr << std::endl;
            DAE_THROW_EXCEPTION2(ss);
        }
    }

public:
    std::vector<std::string>                        m_strarrCompoundIDs;
    std::vector<std::string>                        m_strarrCompoundCASNumbers;
    std::map<std::string, daeeThermoPackagePhase>   m_mapPhases;
    daeeThermoPackageBasis                          m_defaultBasis;

    ICapeThermoPropertyPackageManagerPtr manager;
    ICapeIdentificationPtr               identification;
    IDispatchPtr                         package;
    ICapeThermoCompoundsPtr              compounds;
    ICapeThermoPropertyRoutinePtr        propertyRoutine;
    CComObject<daeCapeThermoMaterial>*   material;
    ICapeThermoMaterialPtr               dispatchMaterial;
    ICapeThermoMaterialContextPtr        materialContext;
    ECapeUserPtr                         errorPackageCapeUser;
    ECapeUserPtr                         errorMaterialCapeUser;
};

