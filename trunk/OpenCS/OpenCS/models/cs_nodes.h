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
#ifndef CS_NODES_H
#define CS_NODES_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <exception>
#include "../cs_machine.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#ifdef OpenCS_Models_EXPORTS
#define OPENCS_MODELS_API __declspec(dllexport)
#else
#define OPENCS_MODELS_API __declspec(dllimport)
#endif
#else
#define OPENCS_MODELS_API
#endif

namespace cs
{
const int32_t csAlgebraicVariable	 = 0;
const int32_t csDifferentialVariable = 1;

class OPENCS_MODELS_API csNodeEvaluationContext_t
{
public:
    csNodeEvaluationContext_t()
    {
        currentTime     = 0.0;
        inverseTimeStep = 0.0;
        jacobianIndex   = -1;
        sensitivityIndex= -1;
        dofs            = NULL;
        values          = NULL;
        timeDerivatives = NULL;
        svalues         = NULL;
        sdvalues        = NULL;
    }

    real_t   currentTime;
    real_t   inverseTimeStep;
    uint32_t jacobianIndex;
    uint32_t sensitivityIndex;
    real_t*  dofs;
    real_t*  values;
    real_t*  timeDerivatives;
    real_t*  svalues;
    real_t*  sdvalues;
};

class OPENCS_MODELS_API csNode_t
{
public:
    virtual ~csNode_t(){}

    virtual std::string ToLatex() const = 0;
    virtual adouble_t   Evaluate(const csNodeEvaluationContext_t& EC) const = 0;
    virtual void        CollectVariableTypes(std::vector<int32_t>& variableTypes) const = 0;
    virtual void        CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const = 0;

    /* Static functions. */
    static void     CreateComputeStack(csNode_t* node, std::vector<csComputeStackItem_t>& computeStack);
    static uint32_t GetComputeStackSize(csNode_t* node);
    static uint32_t GetComputeStackFlops(csNode_t*                                   adnode,
                                         const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                         const std::map<csBinaryFunctions,uint32_t>& binaryOps);
};
typedef std::shared_ptr<csNode_t> csNodePtr;

/* Constants */
class OPENCS_MODELS_API csConstantNode : public csNode_t
{
public:
    csConstantNode(real_t value_);

    std::string ToLatex() const;
    adouble_t   Evaluate(const csNodeEvaluationContext_t& EC) const;
    void        CollectVariableTypes(std::vector<int32_t>& variableTypes) const;
    void        CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const;

    real_t value;
};

/* Current simulation time */
class OPENCS_MODELS_API csTimeNode : public csNode_t
{
public:
    csTimeNode();

    std::string ToLatex() const;
    adouble_t   Evaluate(const csNodeEvaluationContext_t& EC) const;
    void        CollectVariableTypes(std::vector<int32_t>& variableTypes) const;
    void        CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const;
};

/* Degree of freedom */
class OPENCS_MODELS_API csDegreeOfFreedomNode : public csNode_t
{
public:
    csDegreeOfFreedomNode(uint32_t overallIndex_, uint32_t dofIndex_);

    std::string ToLatex() const;
    adouble_t   Evaluate(const csNodeEvaluationContext_t& EC) const;
    void        CollectVariableTypes(std::vector<int32_t>& variableTypes) const;
    void        CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const;

    uint32_t overallIndex;
    uint32_t dofIndex;
};

/* Variable value */
class OPENCS_MODELS_API csVariableNode : public csNode_t
{
public:
    csVariableNode(uint32_t overallIndex_, uint32_t blockIndex_);

    std::string ToLatex() const;
    adouble_t   Evaluate(const csNodeEvaluationContext_t& EC) const;
    void        CollectVariableTypes(std::vector<int32_t>& variableTypes) const;
    void        CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const;

    uint32_t overallIndex;
    uint32_t blockIndex;
};

/* Variable time derivative */
class OPENCS_MODELS_API csTimeDerivativeNode : public csNode_t
{
public:
    csTimeDerivativeNode(uint32_t overallIndex_, uint32_t blockIndex_);

    std::string ToLatex() const;
    adouble_t   Evaluate(const csNodeEvaluationContext_t& EC) const;
    void        CollectVariableTypes(std::vector<int32_t>& variableTypes) const;
    void        CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const;

    uint32_t overallIndex;
    uint32_t blockIndex;
};

/* Unary operators and functions: -, sqrt, log, log10, exp, sin, cos, ... */
class OPENCS_MODELS_API csUnaryNode : public csNode_t
{
public:
    csUnaryNode(csUnaryFunctions function_, csNodePtr operand_);

    std::string ToLatex() const;
    adouble_t   Evaluate(const csNodeEvaluationContext_t& EC) const;
    void        CollectVariableTypes(std::vector<int32_t>& variableTypes) const;
    void        CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const;

    csUnaryFunctions function;
    csNodePtr        operand;
};

/* Binary operators and functions: +, -, *, /, **, pow, atan2, min, max */
class OPENCS_MODELS_API csBinaryNode : public csNode_t
{
public:
    csBinaryNode(csBinaryFunctions function_, csNodePtr leftOperand_, csNodePtr rightOperand_);

    std::string ToLatex() const;
    adouble_t   Evaluate(const csNodeEvaluationContext_t& EC) const;
    void        CollectVariableTypes(std::vector<int32_t>& variableTypes) const;
    void        CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const;

    csBinaryFunctions function;
    csNodePtr         leftOperand;
    csNodePtr         rightOperand;
};

}

#endif
