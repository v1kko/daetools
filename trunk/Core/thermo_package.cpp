#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "nodes_array.h"

// Support for Cape-Open thermo physical property packages exist only in windows (it is COM technology)
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#include "../CapeOpenThermoPackage/cape_open_package.h"
#include <objbase.h>
#endif

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

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    ::CoInitialize(NULL);
#endif
}

daeCapeOpenThermoPhysicalPropertyPackage::~daeCapeOpenThermoPhysicalPropertyPackage(void)
{
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    daeDeleteCapeOpenPropertyPackage(m_package);
    ::CoUninitialize();
#endif
}

void daeCapeOpenThermoPhysicalPropertyPackage::LoadPackage(const std::string& strPackageManager,
                                                           const std::string& strPackageName,
                                                           const std::vector<std::string>& strarrCompounds)
{
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    m_package = daeCreateCapeOpenPropertyPackage();
    if(!m_package)
        daeDeclareAndThrowException(exInvalidPointer);
    m_package->LoadPackage(strPackageManager, strPackageName, strarrCompounds);
#endif
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::PureCompoundConstantProperty(daeeThermoPhysicalProperty property,
                                                                               const std::string& compound)
{
    adouble tmp;
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::ePureCompoundConstantProperty,
                                                                       adNodePtr(),
                                                                       adNodePtr(),
                                                                       adNodeArrayPtr(),
                                                                       property,
                                                                       dae::tpp::etppPhaseUnknown,
                                                                       dae::tpp::eUndefinedBasis,
                                                                       compound,
                                                                       m_package));
#endif

    return tmp;
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::PureCompoundTDProperty(daeeThermoPhysicalProperty property,
                                                                         const adouble& T,
                                                                         const std::string& compound)
{
    adouble tmp;
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::ePureCompoundTDProperty,
                                                                       adNodePtr(),
                                                                       T.node,
                                                                       adNodeArrayPtr(),
                                                                       property,
                                                                       dae::tpp::etppPhaseUnknown,
                                                                       dae::tpp::eUndefinedBasis,
                                                                       compound,
                                                                       m_package));
#endif

    return tmp;
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::PureCompoundPDProperty(daeeThermoPhysicalProperty property,
                                                                         const adouble& P,
                                                                         const std::string& compound)
{
    adouble tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::ePureCompoundPDProperty,
                                                                       P.node,
                                                                       adNodePtr(),
                                                                       adNodeArrayPtr(),
                                                                       property,
                                                                       dae::tpp::etppPhaseUnknown,
                                                                       dae::tpp::eUndefinedBasis,
                                                                       compound,
                                                                       m_package));
#endif

    return tmp;
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::SinglePhaseScalarProperty(daeeThermoPhysicalProperty property,
                                                                            const adouble& P,
                                                                            const adouble& T,
                                                                            const adouble_array& x,
                                                                            daeeThermoPackagePhase phase,
                                                                            daeeThermoPackageBasis basis)
{
    adouble tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::eSinglePhaseScalarProperty,
                                                                       P.node,
                                                                       T.node,
                                                                       x.node,
                                                                       property,
                                                                       phase,
                                                                       basis,
                                                                       std::string(""),
                                                                       m_package));
#endif

    return tmp;
}

adouble_array daeCapeOpenThermoPhysicalPropertyPackage::SinglePhaseVectorProperty(daeeThermoPhysicalProperty property,
                                                                                  const adouble& P,
                                                                                  const adouble& T,
                                                                                  const adouble_array& x,
                                                                                  daeeThermoPackagePhase phase,
                                                                                  daeeThermoPackageBasis basis)
{
    adouble_array tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    tmp.node = adNodeArrayPtr(new adThermoPhysicalPropertyPackageArrayNode(dae::tpp::eSinglePhaseVectorProperty,
                                                                           P.node,
                                                                           T.node,
                                                                           x.node,
                                                                           property,
                                                                           phase,
                                                                           basis,
                                                                           m_package));
#endif

    return tmp;
}

adouble daeCapeOpenThermoPhysicalPropertyPackage::TwoPhaseScalarProperty(daeeThermoPhysicalProperty property,
                                                                         const adouble& P,
                                                                         const adouble& T,
                                                                         const adouble_array& x,
                                                                         daeeThermoPackageBasis basis)

{
    adouble tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    tmp.node = adNodePtr(new adThermoPhysicalPropertyPackageScalarNode(dae::tpp::eTwoPhaseScalarProperty,
                                                                       P.node,
                                                                       T.node,
                                                                       x.node,
                                                                       property,
                                                                       dae::tpp::etppPhaseUnknown,
                                                                       basis,
                                                                       std::string(""),
                                                                       m_package));
#endif

    return tmp;
}

adouble_array daeCapeOpenThermoPhysicalPropertyPackage::TwoPhaseVectorProperty(daeeThermoPhysicalProperty property,
                                                                               const adouble& P,
                                                                               const adouble& T,
                                                                               const adouble_array& x,
                                                                               daeeThermoPackageBasis basis)
{
    adouble_array tmp;

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    tmp.setGatherInfo(true);
    tmp.node = adNodeArrayPtr(new adThermoPhysicalPropertyPackageArrayNode(dae::tpp::eTwoPhaseVectorProperty,
                                                                           P.node,
                                                                           T.node,
                                                                           x.node,
                                                                           property,
                                                                           dae::tpp::etppPhaseUnknown,
                                                                           basis,
                                                                           m_package));
#endif

    return tmp;
}

unit daeCapeOpenThermoPhysicalPropertyPackage::GetUnits(daeeThermoPhysicalProperty property)
{
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#endif
    return unit();
}

}
}

