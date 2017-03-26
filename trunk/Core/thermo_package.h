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
#ifndef DAE_THERMO_PACKAGE_H
#define DAE_THERMO_PACKAGE_H

#include <string>
#include <vector>

namespace dae
{
namespace tpp
{
enum daeeThermoPackagePropertyType
{
    ePureCompoundConstantProperty,
    ePureCompoundTDProperty,
    ePureCompoundPDProperty,
    eSinglePhaseScalarProperty,
    eSinglePhaseVectorProperty,
    eTwoPhaseScalarProperty,
    eTwoPhaseVectorProperty
};

enum daeeThermoPackageBasis
{
    eMole = 0,
    eMass,
    eUndefinedBasis
};

// Phase State Of Aggregation
enum daeeThermoPackagePhase
{
    etppPhaseUnknown = 0,
    eVapor,
    eLiquid,
    eSolid
};

/*********************************************************************************************
    daeThermoPhysicalPropertyPackage_t
**********************************************************************************************/
class daeThermoPhysicalPropertyPackage_t
{
public:
    virtual ~daeThermoPhysicalPropertyPackage_t() {}

public:
    virtual void LoadPackage(const std::string& strPackageManager,
                             const std::string& strPackageName,
                             const std::vector<std::string>& strarrCompoundIDs,
                             const std::vector<std::string>& strarrCompoundCASNumbers,
                             const std::map<std::string,daeeThermoPackagePhase>& mapAvailablePhases,
                             daeeThermoPackageBasis defaultBasis = eMole) = 0;

    virtual double PureCompoundConstantProperty(const std::string& property, const std::string& compound) = 0;

    virtual double PureCompoundTDProperty(const std::string& property, double T, const std::string& compound) = 0;

    virtual double PureCompoundPDProperty(const std::string& property, double P, const std::string& compound) = 0;

    virtual double SinglePhaseScalarProperty(const std::string& property,
                                             double P,
                                             double T,
                                             const std::vector<double>& x,
                                             const std::string& phase,
                                             daeeThermoPackageBasis basis = eMole) = 0;

    virtual void SinglePhaseVectorProperty(const std::string& property,
                                           double P,
                                           double T,
                                           const std::vector<double>& x,
                                           const std::string& phase,
                                           std::vector<double>& results,
                                           daeeThermoPackageBasis basis = eMole) = 0;

    virtual double TwoPhaseScalarProperty(const std::string& property,
                                          double P1,
                                          double T1,
                                          const std::vector<double>& x1,
                                          const std::string& phase1,
                                          double P2,
                                          double T2,
                                          const std::vector<double>& x2,
                                          const std::string& phase2,
                                          daeeThermoPackageBasis basis = eMole) = 0;

    virtual void TwoPhaseVectorProperty(const std::string& property,
                                        double P1,
                                        double T1,
                                        const std::vector<double>& x1,
                                        const std::string& phase1,
                                        double P2,
                                        double T2,
                                        const std::vector<double>& x2,
                                        const std::string& phase2,
                                        std::vector<double>& results,
                                        daeeThermoPackageBasis basis = eMole) = 0;
};

}
}

#endif
