// daeCapeThermoMaterial.cpp : Implementation of CdaeCapeThermoMaterial

#include "stdafx.h"
#include "daeCapeThermoMaterial.h"
#include "daeCapeThermoRoutine.h"

CComObject<daeCapeThermoMaterial>*  daeCreateThermoMaterial(const std::vector<BSTR>*                              compoundIDs,
                                                            const std::vector<BSTR>*                              compoundCASNumbers,
                                                            const std::map<std::string, daeeThermoPackagePhase>*  phases,
                                                            const std::map<std::string, eCapePhaseStatus>*        phasesStatus,
                                                            const ComBSTR_ComBSTR_Variant_PropertyMap*            overallProperties,
                                                            const ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap*    singlePhaseProperties,
                                                            const ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap*    twoPhaseProperties)
{
    CComObject<daeCapeThermoMaterial>* material;
    HRESULT hr = CComObject<daeCapeThermoMaterial>::CreateInstance(&material);
    ATLASSERT(SUCCEEDED(hr));
    // Increment reference count immediately
    material->AddRef();

    if (compoundIDs)
        material->m_strarrCompoundIDs = *compoundIDs;
    if (compoundCASNumbers)
        material->m_strarrCompoundCASNumbers = *compoundCASNumbers;
    if (phases && !phasesStatus) // create statuses from the list of phases
    {
        material->m_mapAvailablePhases = *phases;
        for (std::map<std::string, daeeThermoPackagePhase>::iterator it = material->m_mapAvailablePhases.begin(); it != material->m_mapAvailablePhases.end(); it++)
            material->m_mapPhasesStatus[it->first] = CO_COM::CAPE_UNKNOWNPHASESTATUS;
    }
    else if (phasesStatus) // copy the received arguments
    {
        material->m_mapPhasesStatus = *phasesStatus;
    }
    if(overallProperties)
        material->m_overallProperties = *overallProperties;
    if (singlePhaseProperties)
        material->m_singlePhaseProperties = *singlePhaseProperties;
    if (twoPhaseProperties)
        material->m_twoPhaseProperties = *twoPhaseProperties;

    return material;
}

__declspec(dllexport) dae::tpp::daeThermoPhysicalPropertyPackage_t* daeCreateCapeOpenPropertyPackage()
{
    return new daeCapeThermoPropertyRoutine;
}

/*
__declspec(dllexport) void daeDeleteCapeOpenPropertyPackage(dae::tpp::daeThermoPhysicalPropertyPackage_t* package)
{
    daeCapeThermoPropertyRoutine* co_package = dynamic_cast<daeCapeThermoPropertyRoutine*>(package);
    if (package)
        delete package;
    package = NULL;
}
*/

// CdaeCapeThermoMaterial

