#include "stdafx.h"
#include "coreimpl.h"

namespace dae
{
namespace core
{
daeCapeOpenThermoPhysicalPropertyPackage::daeCapeOpenThermoPhysicalPropertyPackage(const string& strName, daeModel* pModel, const string& strDescription)
{
    m_package = NULL;
    m_pModel = pModel;
    SetName(strName);
    m_strDescription = strDescription;
}

daeCapeOpenThermoPhysicalPropertyPackage::~daeCapeOpenThermoPhysicalPropertyPackage(void)
{

}

void daeCapeOpenThermoPhysicalPropertyPackage::LoadPackage(const std::string& strPackageManager,
                                                           const std::string& strPackageName,
                                                           const std::vector<std::string>& strarrCompounds)
{

}

adouble daeCapeOpenThermoPhysicalPropertyPackage::SinglePhaseScalarProperty(const std::string& property,
                                                                            const std::string& phase,
                                                                            adouble P, adouble T, adouble_array& X,
                                                                            const std::string& basis)
{
    return adouble();
}

adouble_array daeCapeOpenThermoPhysicalPropertyPackage::SinglePhaseVectorProperty(const std::string& property,
                                                                                  const std::string& phase,
                                                                                  adouble P, adouble T, adouble_array& X,
                                                                                  const std::string& basis)
{
    return adouble_array();
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::TwoPhaseProperty(const std::string& property,
                                                                   adouble P, adouble T, adouble_array& X,
                                                                   const std::string& basis)
{

    return adouble();
}

}
}

