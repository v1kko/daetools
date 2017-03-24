// daeCapeThermoMaterial.cpp : Implementation of CdaeCapeThermoMaterial

#include "stdafx.h"
#include "daeCapeThermoMaterial.h"

ICapeThermoMaterial* daeCreateThermoMaterial(const std::vector<BSTR>*                           compounds,
                                             const ComBSTR_ComBSTR_Variant_PropertyMap*         overallProperties,
                                             const ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap* singlePhaseProperties,
                                             const ComBSTR_ComBSTR_ComBSTR_Variant_PropertyMap* twoPhaseProperties)
{
    CComObject<daeCapeThermoMaterial>* material;
    HRESULT hr = CComObject<daeCapeThermoMaterial>::CreateInstance(&material);
    ATLASSERT(SUCCEEDED(hr));
    // Increment reference count immediately
    material->AddRef();

    if (compounds)
        material->m_strarrCompounds = *compounds;
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

__declspec(dllexport) void daeDeleteCapeOpenPropertyPackage(dae::tpp::daeThermoPhysicalPropertyPackage_t* package)
{
    daeCapeThermoPropertyRoutine* co_package = dynamic_cast<daeCapeThermoPropertyRoutine*>(package);
    if (package)
        delete package;
    package = NULL;
}


// CdaeCapeThermoMaterial

