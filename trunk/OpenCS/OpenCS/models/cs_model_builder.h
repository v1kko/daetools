/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_MODEL_BUILDER_H
#define CS_MODEL_BUILDER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "../cs_machine.h"
#include "../cs_model.h"
#include "cs_nodes.h"
#include "cs_number.h"
#include "cs_partitioners.h"

namespace cs
{
typedef std::shared_ptr<csModel_t> csModelPtr;

class OPENCS_MODELS_API csModelBuilder_t
{
public:
    csModelBuilder_t();
    virtual ~csModelBuilder_t();

    void Initialize_ODE_System(uint32_t           noVariables,
                               uint32_t           noDofs,
                               real_t             defaultVariableValue          = 0.0,
                               real_t             defaultAbsoluteTolerance      = 1e-5,
                               const std::string& defaultVariableName           = "x");

    void Initialize_DAE_System(uint32_t           noVariables,
                               uint32_t           noDofs,
                               real_t             defaultVariableValue          = 0.0,
                               real_t             defaultVariableTimeDerivative = 0.0,
                               real_t             defaultAbsoluteTolerance      = 1e-5,
                               const std::string& defaultVariableName           = "x");

    const csNumber_t&               GetTime() const;
    const std::vector<csNumber_t>&  GetDegreesOfFreedom() const;
    const std::vector<csNumber_t>&  GetVariables() const;
    const std::vector<csNumber_t>&  GetTimeDerivatives() const;

    void SetModelEquations(const std::vector<csNumber_t>& equations);
    void SetVariableValues(const std::vector<real_t>& values);
    void SetVariableTimeDerivatives(const std::vector<real_t>& timeDerivatives);
    void SetDegreeOfFreedomValues(const std::vector<real_t>& dofs);
    void SetVariableNames(const std::vector<std::string>& names);
    void SetVariableTypes(const std::vector<int32_t>& types);
    void SetAbsoluteToleances(const std::vector<real_t>& absTolerances);

    void EvaluateEquations(real_t               currentTime,
                           std::vector<real_t>& equations);
    void EvaluateDerivatives(real_t                 currentTime,
                             real_t                 timeStep,
                             std::vector<uint32_t>& IA,
                             std::vector<uint32_t>& JA,
                             std::vector<real_t>&   A,
                             bool                   generateIncidenceMatrix = false);

    void GetSparsityPattern(std::vector< std::map<uint32_t,uint32_t> >& incidenceMatrix);
    void GetSparsityPattern(std::vector<uint32_t>& IA, std::vector<uint32_t>& JA);

    std::vector<csModelPtr> PartitionSystem(uint32_t                                    Npe,
                                            csGraphPartitioner_t*                       graphPartitioner,
                                            const std::vector<std::string>&             balancingConstraints  = std::vector<std::string>(),
                                            bool                                        logPartitionResults   = false,
                                            const std::map<csUnaryFunctions,uint32_t>&  unaryOperationsFlops  = std::map<csUnaryFunctions,uint32_t>(),
                                            const std::map<csBinaryFunctions,uint32_t>& binaryOperationsFlops = std::map<csBinaryFunctions,uint32_t>());

    void ExportModels(const std::vector<csModelPtr>& models, const std::string& outputDirectory, const std::string& simulationOptionsJSON);

protected:
    void Initialize(uint32_t           noVariables,
                    uint32_t           noDofs,
                    real_t             defaultVariableValue          = 0.0,
                    real_t             defaultVariableTimeDerivative = 0.0,
                    real_t             defaultAbsoluteTolerance      = 1e-5,
                    const std::string& defaultVariableName           = "x");

public:
    bool                     isODESystem;
    /* Internal data (dofs, variables, time derivatives, etc.). */
    uint32_t                 Ndofs;
    uint32_t                 Nvariables;
    csNumber_t               time_ptr;
    std::vector<csNumber_t>  dofs_ptrs;
    std::vector<csNumber_t>  variables_ptrs;
    std::vector<csNumber_t>  timeDerivatives_ptrs;

    /* Equations (set by the user).*/
    std::vector<csNumber_t>  equationNodes_ptrs;

    /* Model structure. */
    std::vector<real_t>      dofValues;
    std::vector<real_t>      variableValues;
    std::vector<real_t>      variableDerivatives;
    std::vector<real_t>      sensitivityValues;
    std::vector<real_t>      sensitivityDerivatives;
    std::vector<std::string> variableNames;
    std::vector<int32_t>     variableTypes;
    std::vector<real_t>      absoluteTolerances;
};

}

#endif
