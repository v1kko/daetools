// daeCapeThermoMaterial.cpp : Implementation of CdaeCapeThermoMaterial

#include "stdafx.h"
#include "daeCapeThermoMaterial.h"

ICapeThermoMaterial* daeCreateThermoMaterial(const std::vector<BSTR>*              compounds,
    const std::map<CComBSTR, _variant_t>* overallProperties,
    const std::map<CComBSTR, _variant_t>* singleProperties)
{
    CComObject<daeCapeThermoMaterial>* material;
    HRESULT hr = CComObject<daeCapeThermoMaterial>::CreateInstance(&material);
    ATLASSERT(SUCCEEDED(hr));
    // Increment reference count immediately
    material->AddRef();

    if (compounds)
    {
        size_t N = compounds->size();
        material->m_strarrCompounds.resize(N);
        for (size_t i = 0; i < N; i++)
            material->m_strarrCompounds[i] = ::SysAllocString((*compounds)[i]);
    }
    if(overallProperties)
        material->overallProperties = *overallProperties;
    if (singleProperties)
        material->singleProperties = *singleProperties;

    return material;
}

// CdaeCapeThermoMaterial

