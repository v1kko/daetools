/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2015
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_COOL_PROP_THERMO_H
#define DAE_COOL_PROP_THERMO_H

#include "../Core/thermo_package.h"
using namespace dae::tpp;
#include "CoolProp.h"
#include "AbstractState.h"
#include "crossplatform_shared_ptr.h"
using namespace CoolProp;

class daeCoolPropThermoPhysicalPropertyPackage : public daeThermoPhysicalPropertyPackage_t
{
public:
    daeCoolPropThermoPhysicalPropertyPackage();
    virtual ~daeCoolPropThermoPhysicalPropertyPackage();

public:
    virtual void LoadPackage(const std::string& strPackageManager,
                             const std::string& strPackageName,
                             const std::vector<std::string>& strarrCompoundIDs,
                             const std::vector<std::string>& strarrCompoundCASNumbers,
                             const std::map<std::string,daeeThermoPackagePhase>& mapAvailablePhases,
                             daeeThermoPackageBasis defaultBasis = eMole,
                             const std::map<std::string,std::string>& mapOptions = std::map<std::string,std::string>());

    virtual std::string GetTPPName();

    // ICapeThermoCompounds interface
    virtual double GetCompoundConstant(const std::string& property, const std::string& compound);
    virtual double GetTDependentProperty(const std::string& property, double T, const std::string& compound);
    virtual double GetPDependentProperty(const std::string& property, double P, const std::string& compound);

    // ICapeThermoPropertyRoutine interface
    virtual double CalcSinglePhaseScalarProperty(const std::string& property,
                                                 double P,
                                                 double T,
                                                 const std::vector<double>& x,
                                                 const std::string& phase,
                                                 daeeThermoPackageBasis basis = eMole);

    virtual void CalcSinglePhaseVectorProperty(const std::string& property,
                                               double P,
                                               double T,
                                               const std::vector<double>& x,
                                               const std::string& phase,
                                               std::vector<double>& results,
                                               daeeThermoPackageBasis basis = eMole);

    virtual double CalcTwoPhaseScalarProperty(const std::string& property,
                                              double P1,
                                              double T1,
                                              const std::vector<double>& x1,
                                              const std::string& phase1,
                                              double P2,
                                              double T2,
                                              const std::vector<double>& x2,
                                              const std::string& phase2,
                                              daeeThermoPackageBasis basis = eMole);

    virtual void CalcTwoPhaseVectorProperty(const std::string& property,
                                            double P1,
                                            double T1,
                                            const std::vector<double>& x1,
                                            const std::string& phase1,
                                            double P2,
                                            double T2,
                                            const std::vector<double>& x2,
                                            const std::string& phase2,
                                            std::vector<double>& results,
                                            daeeThermoPackageBasis basis = eMole);

protected:
    double GetScalarProperty(const std::string& capeOpenProperty, daeeThermoPackageBasis eBasis);
    void   GetVectorProperty(const std::string& capeOpenProperty, daeeThermoPackageBasis eBasis, std::vector<double>& results);

protected:
    shared_ptr<AbstractState>                       m_mixture;
    std::vector<std::string>                        m_strarrCompoundIDs;
    std::vector<std::string>                        m_strarrCompoundCASNumbers;
    std::map<std::string, daeeThermoPackagePhase>   m_mapAvailablePhases;
    daeeThermoPackageBasis                          m_defaultBasis;
    std::string                                     m_defaultBackend;
    std::string                                     m_defaultReferenceState;
};

#endif
