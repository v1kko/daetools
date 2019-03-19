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
#include "cs_nodes.h"
#include "cs_model_builder.h"

namespace cs
{
/* Constants */
csConstantNode::csConstantNode(real_t value_)
{
    value = value_;
}

adouble_t csConstantNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    return adouble_t_(value, 0.0);
}

void csConstantNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
}

void csConstantNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
}

std::string csConstantNode::ToLatex(const csModelBuilder_t* mb) const
{
    char buffer[256];
    std::snprintf(buffer, 256, "%.15f", value);
    std::string svalue = buffer;

    /* Right trim '0' and '.' (if present after removal of zeros). */
    svalue.erase(svalue.find_last_not_of('0')+1);
    svalue.erase(svalue.find_last_not_of('.')+1);

    svalue = "{" + svalue + "}";
    return svalue;
}

/* Time */
csTimeNode::csTimeNode()
{
}

adouble_t csTimeNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    return adouble_t_(EC.currentTime, 0.0);
}

void csTimeNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
}

void csTimeNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
}

std::string csTimeNode::ToLatex(const csModelBuilder_t* mb) const
{
    return "time";
}

/* Variable values */
csDegreeOfFreedomNode::csDegreeOfFreedomNode(uint32_t overallIndex_, uint32_t dofIndex_)
{
    overallIndex = overallIndex_;
    dofIndex     = dofIndex_;
}

adouble_t csDegreeOfFreedomNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    if(!EC.dofs)
        csThrowException("DOF values not set in csNodeEvaluationContext_t");

    return adouble_t_(EC.dofs[dofIndex], 0.0);
}

void csDegreeOfFreedomNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
}

void csDegreeOfFreedomNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
}

std::string csDegreeOfFreedomNode::ToLatex(const csModelBuilder_t* mb) const
{
    if(mb)
    {
        return mb->dofNames[dofIndex];
    }
    else
    {
        char buffer[256];
        std::snprintf(buffer, 256, "y[%d]", dofIndex);
        return std::string(buffer);
    }
}


csVariableNode::csVariableNode(uint32_t overallIndex_, uint32_t blockIndex_)
{
    overallIndex = overallIndex_;
    blockIndex   = blockIndex_;
}

adouble_t csVariableNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    if(!EC.values)
        csThrowException("Variable values not set in csNodeEvaluationContext_t");

    if(EC.jacobianIndex == overallIndex)
        return adouble_t_(EC.values[blockIndex], 1.0);
    else
        return adouble_t_(EC.values[blockIndex], 0.0);
}

void csVariableNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
    variableIndexes[blockIndex] = overallIndex;
}

void csVariableNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
}

std::string csVariableNode::ToLatex(const csModelBuilder_t* mb) const
{
    if(mb)
    {
        return mb->variableNames[blockIndex];
    }
    else
    {
        char buffer[256];
        std::snprintf(buffer, 256, "x[%d]", blockIndex);
        return std::string(buffer);
    }
}

/* Variable time derivatives */
csTimeDerivativeNode::csTimeDerivativeNode(uint32_t overallIndex_, uint32_t blockIndex_)
{
    overallIndex = overallIndex_;
    blockIndex   = blockIndex_;
}

adouble_t csTimeDerivativeNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    if(!EC.timeDerivatives)
        csThrowException("Derivative values not set in csNodeEvaluationContext_t");

    if(EC.jacobianIndex == overallIndex)
        return adouble_t_(EC.timeDerivatives[blockIndex], EC.inverseTimeStep);
    else
        return adouble_t_(EC.timeDerivatives[blockIndex], 0.0);
}

void csTimeDerivativeNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
    variableIndexes[blockIndex] = overallIndex;
}

void csTimeDerivativeNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
    variableTypes[blockIndex] = csDifferentialVariable;
}

std::string csTimeDerivativeNode::ToLatex(const csModelBuilder_t* mb) const
{
    if(mb)
    {
        return "d" + mb->variableNames[blockIndex] + "/dt";
    }
    else
    {
        char buffer[256];
        std::snprintf(buffer, 256, "x_t[%d]", blockIndex);
        return std::string(buffer);
    }
}

/* Unary operators and functions: -, sqrt, log, log10, exp, sin, cos, ... */
csUnaryNode::csUnaryNode(csUnaryFunctions function_, csNodePtr operand_)
{
    if(!operand_)
        csThrowException("Invalid csUnaryNode operand");

    function = function_;
    operand  = operand_;
}

adouble_t csUnaryNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    if(!operand)
        csThrowException("Operand not set in csUnaryNode");

    adouble_t res;
    adouble_t operand_val = operand->Evaluate(EC);

    switch(function)
    {
    case eSign:
        res = _sign_(operand_val);
        break;
    case eSin:
        res = _sin_(operand_val);
        break;
    case eCos:
        res = _cos_(operand_val);
        break;
    case eTan:
        res = _tan_(operand_val);
        break;
    case eArcSin:
        res = _asin_(operand_val);
        break;
    case eArcCos:
        res = _acos_(operand_val);
        break;
    case eArcTan:
        res = _atan_(operand_val);
        break;
    case eSqrt:
        res = _sqrt_(operand_val);
        break;
    case eExp:
        res = _exp_(operand_val);
        break;
    case eLn:
        res = _log_(operand_val);
        break;
    case eLog:
        res = _log10_(operand_val);
        break;
    case eAbs:
        res = _abs_(operand_val);
        break;
    case eCeil:
        res = _ceil_(operand_val);
        break;
    case eFloor:
        res = _floor_(operand_val);
        break;
    case eSinh:
        res = _sinh_(operand_val);
        break;
    case eCosh:
        res = _cosh_(operand_val);
        break;
    case eTanh:
        res = _tanh_(operand_val);
        break;
    case eArcSinh:
        res = _asinh_(operand_val);
        break;
    case eArcCosh:
        res = _acosh_(operand_val);
        break;
    case eArcTanh:
        res = _atanh_(operand_val);
        break;
    case eErf:
        res = _erf_(operand_val);
        break;
    default:
        csThrowException("exNotImplemented");
    }
    return res;
}

void csUnaryNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
    if(!operand)
        csThrowException("Operand not set in csUnaryNode");

    operand->CollectVariableIndexes(variableIndexes);
}

void csUnaryNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
    if(!operand)
        csThrowException("Operand not set in csUnaryNode");

    operand->CollectVariableTypes(variableTypes);
}

std::string csUnaryNode::ToLatex(const csModelBuilder_t* mb) const
{
    if(!operand)
        csThrowException("Operand not set in csUnaryNode");

    std::string res;
    std::string s_operand = operand->ToLatex(mb);

    switch(function)
    {
    case eSign:
        res += "\\left( - ";
        res += s_operand;
        res += " \\right)";
        break;
    case eSin:
        res += "\\sin";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eCos:
        res += "\\cos";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eTan:
        res += "\\tan";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eArcSin:
        res += "\\arcsin";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eArcCos:
        res += "\\arccos";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eArcTan:
        res += "\\arctan";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eSqrt:
        res += "\\sqrt";
        res += " { ";
        res += s_operand;
        res += " }";
        break;
    case eExp:
        res += "\\exp";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eLn:
        res += "\\ln";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eLog:
        res += "\\log";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eAbs:
        res += "{ ";
        res += "\\left| ";
        res += s_operand;
        res += " \\right| ";
        res += "}";
        break;
    case eCeil:
        res += "ceil";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eFloor:
        res += "floor";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eSinh:
        res += "\\sinh";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eCosh:
        res += "\\cosh";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eTanh:
        res += "\\tanh";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eArcSinh:
        res += "asinh";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eArcCosh:
        res += "acosh";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eArcTanh:
        res += "atanh";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    case eErf:
        res += "erf";
        res += " \\left( ";
        res += s_operand;
        res += " \\right)";
        break;
    default:
        csThrowException("exNotImplemented");
    }
    return res;
}

/* Binary operators and functions: +, -, *, /, **, pow, atan2, min, max */
csBinaryNode::csBinaryNode(csBinaryFunctions function_, csNodePtr leftOperand_, csNodePtr rightOperand_)
{
    if(!leftOperand_)
        csThrowException("Invalid left csBinaryNode operand");
    if(!rightOperand_)
        csThrowException("Invald right csBinaryNode operand");

    function     = function_;
    leftOperand  = leftOperand_;
    rightOperand = rightOperand_;
}

adouble_t csBinaryNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    if(!leftOperand)
        csThrowException("Left operand not set in csBinaryNode");
    if(!rightOperand)
        csThrowException("Right operand not set in csBinaryNode");

    adouble_t res;
    adouble_t l_operand = leftOperand->Evaluate(EC);
    adouble_t r_operand = rightOperand->Evaluate(EC);

    switch(function)
    {
    case ePlus:
        res = _plus_(l_operand, r_operand);
        break;
    case eMinus:
        res = _minus_(l_operand, r_operand);
        break;
    case eMulti:
        res = _multi_(l_operand, r_operand);
        break;
    case eDivide:
        res = _divide_(l_operand, r_operand);
        break;
    case ePower:
        res = _pow_(l_operand, r_operand);
        break;
    case eMin:
        res = _min_(l_operand, r_operand);
        break;
    case eMax:
        res = _max_(l_operand, r_operand);
        break;
    case eArcTan2:
        res = _atan2_(l_operand, r_operand);
        break;
    default:
        csThrowException("exNotImplemented");
    }
    return res;
}

void csBinaryNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
    if(!leftOperand)
        csThrowException("Left operand not set in csBinaryNode");
    if(!rightOperand)
        csThrowException("Right operand not set in csBinaryNode");

    leftOperand->CollectVariableIndexes(variableIndexes);
    rightOperand->CollectVariableIndexes(variableIndexes);
}

void csBinaryNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
    if(!leftOperand)
        csThrowException("Left operand not set in csBinaryNode");
    if(!rightOperand)
        csThrowException("Right operand not set in csBinaryNode");

    leftOperand->CollectVariableTypes(variableTypes);
    rightOperand->CollectVariableTypes(variableTypes);
}

std::string csBinaryNode::ToLatex(const csModelBuilder_t* mb) const
{
    if(!leftOperand)
        csThrowException("Left operand not set in csBinaryNode");
    if(!rightOperand)
        csThrowException("Right operand not set in csBinaryNode");

    std::string res;
    std::string l_operand = leftOperand->ToLatex(mb);
    std::string r_operand = rightOperand->ToLatex(mb);

    res = "\\left("; // Start
    switch(function)
    {
    case ePlus:
        res += l_operand;
        res += " + ";
        res += r_operand;
        break;
    case eMinus:
        res += l_operand;
        res += " - ";
        res += r_operand;
        break;
    case eMulti:
        res += l_operand;
        res += " \\cdot ";
        res += r_operand;
        break;
    case eDivide:
        res += "\\frac{";
        res += l_operand;
        res += "}{";
        res += r_operand;
        res += "}";
        break;
    case ePower:
        res += "pow \\left(";
        res += l_operand;
        res += ", ";
        res += r_operand;
        res += "\\right)";
        break;
    case eMin:
        res += "min \\left(";
        res += l_operand;
        res += ", ";
        res += r_operand;
        res += "\\right)";
        break;
    case eMax:
        res += "max \\left(";
        res += l_operand;
        res += ", ";
        res += r_operand;
        res += "\\right)";
        break;
    case eArcTan2:
        res += "atan2 \\left(";
        res += l_operand;
        res += ", ";
        res += r_operand;
        res += "\\right)";
        break;
    default:
        csThrowException("exNotImplemented");
    }
    res += "\\right)"; // End

    return res;
}

/* CreateComputeStack functions. */
uint32_t csConstantNode::CreateComputeStack(std::vector<csComputeStackItem_t>& computeStack)
{
    csComputeStackItem_t item;
    memset(&item, 0, sizeof(csComputeStackItem_t));
    item.opCode         = eOP_Constant;
    item.resultLocation = eOP_Result_to_value;
    item.data.value     = value;
    computeStack.push_back(item);

    return 1; // single item added
}

uint32_t csTimeNode::CreateComputeStack(std::vector<csComputeStackItem_t>& computeStack)
{
    csComputeStackItem_t item;
    memset(&item, 0, sizeof(csComputeStackItem_t));
    item.opCode         = eOP_Time;
    item.resultLocation = eOP_Result_to_value;
    computeStack.push_back(item);

    return 1; // single item added
}

uint32_t csDegreeOfFreedomNode::CreateComputeStack(std::vector<csComputeStackItem_t>& computeStack)
{
    csComputeStackItem_t item;
    memset(&item, 0, sizeof(csComputeStackItem_t));
    item.opCode                        = eOP_DegreeOfFreedom;
    item.resultLocation                = eOP_Result_to_value;
    item.data.dof_indexes.overallIndex = overallIndex;
    item.data.dof_indexes.dofIndex     = dofIndex;
    computeStack.push_back(item);

    return 1; // single item added
}

uint32_t csVariableNode::CreateComputeStack(std::vector<csComputeStackItem_t>& computeStack)
{
    csComputeStackItem_t item;
    memset(&item, 0, sizeof(csComputeStackItem_t));
    item.opCode                    = eOP_Variable;
    item.resultLocation            = eOP_Result_to_value;
    item.data.indexes.overallIndex = overallIndex;
    item.data.indexes.blockIndex   = blockIndex;
    computeStack.push_back(item);

    return 1; // single item added
}

uint32_t csTimeDerivativeNode::CreateComputeStack(std::vector<csComputeStackItem_t>& computeStack)
{
    csComputeStackItem_t item;
    memset(&item, 0, sizeof(csComputeStackItem_t));
    item.opCode                    = eOP_TimeDerivative;
    item.resultLocation            = eOP_Result_to_value;
    item.data.indexes.overallIndex = overallIndex;
    item.data.indexes.blockIndex   = blockIndex;
    computeStack.push_back(item);

    return 1; // single item added
}

uint32_t csUnaryNode::CreateComputeStack(std::vector<csComputeStackItem_t>& computeStack)
{
    uint32_t noItems = 0;

    // 1. Process operand:
    noItems += operand->CreateComputeStack(computeStack);

    // 2. Add the unary operation:
    csComputeStackItem_t item;
    memset(&item, 0, sizeof(csComputeStackItem_t));
    item.opCode         = eOP_Unary;
    item.resultLocation = eOP_Result_to_value;
    item.function       = function;
    computeStack.push_back(item);
    noItems += 1;

    return noItems;
}

uint32_t csBinaryNode::CreateComputeStack(std::vector<csComputeStackItem_t>& computeStack)
{
    uint32_t noItems = 0;

    // 1. Process left operand:
    // Add ops to produce left value.
    uint32_t noItemsLeft = leftOperand->CreateComputeStack(computeStack);
    if(noItemsLeft == 0)
        csThrowException("exInvalidCall");
    csComputeStackItem_t& litem = computeStack.back();
    litem.resultLocation = eOP_Result_to_lvalue;
    noItems += noItemsLeft;

    // 2. Process right operand:
    // Add ops to produce right value.
    uint32_t noItemsRight = rightOperand->CreateComputeStack(computeStack);
    if(noItemsRight == 0)
        csThrowException("exInvalidCall");
    csComputeStackItem_t& ritem = computeStack.back();
    ritem.resultLocation = eOP_Result_to_rvalue;
    noItems += noItemsRight;

    // 3. Add the binary operation:
    csComputeStackItem_t item;
    memset(&item, 0, sizeof(csComputeStackItem_t));
    item.opCode         = eOP_Binary;
    item.resultLocation = eOP_Result_to_value;
    item.function       = function;
    computeStack.push_back(item);
    noItems += 1;

    return noItems;
}

/* Static function. */
uint32_t csNode_t::CreateComputeStack(csNode_t* adnode, std::vector<csComputeStackItem_t>& computeStack)
{
    if(!adnode)
        csThrowException("Invalid node");

    // Get the index of the first item (to set the size member).
    uint32_t start = computeStack.size();

    // Create Compute Stack items.
    uint32_t noItemsCreated = adnode->CreateComputeStack(computeStack);

    // "size" member of the first item contains the stack size.
    uint32_t stackSize = computeStack.size() - start;
    if(stackSize == 0)
        csThrowException("The size of the compute stack is zero");
    computeStack[start].size = stackSize;

    return noItemsCreated;
}

/* GetComputeStackSize functions. */
uint32_t csConstantNode::GetComputeStackSize()
{
    return 1;
}

uint32_t csTimeNode::GetComputeStackSize()
{
    return 1;
}

uint32_t csVariableNode::GetComputeStackSize()
{
    return 1;
}

uint32_t csDegreeOfFreedomNode::GetComputeStackSize()
{
    return 1;
}

uint32_t csTimeDerivativeNode::GetComputeStackSize()
{
    return 1;
}

uint32_t csUnaryNode::GetComputeStackSize()
{
    // 1. Process the operand
    uint32_t noItems = operand->GetComputeStackSize();

    // 2. Add the unary operation
    noItems++;

    return noItems;
}

uint32_t csBinaryNode::GetComputeStackSize()
{
    // 1. Process left operand
    uint32_t noItems = leftOperand->GetComputeStackSize();

    // 2. Process right operand
    noItems += rightOperand->GetComputeStackSize();

    // 3. Add the binary operation
    noItems++;

    return noItems;
}

/* GetComputeStackFlops functions. */
uint32_t csConstantNode::GetComputeStackFlops(const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                              const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    return 0;
}

uint32_t csTimeNode::GetComputeStackFlops(const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                          const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    return 0;
}

uint32_t csDegreeOfFreedomNode::GetComputeStackFlops(const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                                     const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    return 0;
}

uint32_t csVariableNode::GetComputeStackFlops(const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                              const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    return 0;
}

uint32_t csTimeDerivativeNode::GetComputeStackFlops(const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                                    const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    return 0;
}

uint32_t csUnaryNode::GetComputeStackFlops(const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                           const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    // 1. Process the operand
    uint32_t flops = operand->GetComputeStackFlops(unaryOps, binaryOps);

    // 2. Add the unary operation
    std::map<csUnaryFunctions,uint32_t>::const_iterator it = unaryOps.find(function);
    if(it != unaryOps.end())
        flops += it->second;
    else
        flops += 1;

    return flops;
}

uint32_t csBinaryNode::GetComputeStackFlops(const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                            const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    // 1. Process left operand
    uint32_t flops = leftOperand->GetComputeStackFlops(unaryOps, binaryOps);

    // 2. Process right operand
    flops += rightOperand->GetComputeStackFlops(unaryOps, binaryOps);

    // 3. Add the binary operation
    std::map<csBinaryFunctions,uint32_t>::const_iterator it = binaryOps.find(function);
    if(it != binaryOps.end())
        flops += it->second;
    else
        flops += 1;

    return flops;
}

/* Static functions. */
/*
void csNode_t::CreateComputeStack(csNode_t* adnode, std::vector<csComputeStackItem_t>& computeStack)
{
    if(!adnode)
        csThrowException("exInvalidCall");

    uint32_t start = computeStack.size();
    if( dynamic_cast<csConstantNode*>(adnode) )
    {
        csConstantNode* cnode = dynamic_cast<csConstantNode*>(adnode);

        csComputeStackItem_t item;
        memset(&item, 0, sizeof(csComputeStackItem_t));
        item.opCode         = eOP_Constant;
        item.resultLocation = eOP_Result_to_value;
        item.data.value     = cnode->value;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<csTimeNode*>(adnode) )
    {
        csComputeStackItem_t item;
        memset(&item, 0, sizeof(csComputeStackItem_t));
        item.opCode         = eOP_Time;
        item.resultLocation = eOP_Result_to_value;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<csDegreeOfFreedomNode*>(adnode) )
    {
        csDegreeOfFreedomNode* vnode = dynamic_cast<csDegreeOfFreedomNode*>(adnode);

        csComputeStackItem_t item;
        memset(&item, 0, sizeof(csComputeStackItem_t));
        item.resultLocation = eOP_Result_to_value;

        item.opCode                        = eOP_DegreeOfFreedom;
        item.data.dof_indexes.overallIndex = vnode->overallIndex;
        item.data.dof_indexes.dofIndex     = vnode->dofIndex;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<csVariableNode*>(adnode) )
    {
        csVariableNode* vnode = dynamic_cast<csVariableNode*>(adnode);
        csComputeStackItem_t item;
        memset(&item, 0, sizeof(csComputeStackItem_t));
        item.resultLocation = eOP_Result_to_value;

        item.opCode                    = eOP_Variable;
        item.data.indexes.overallIndex = vnode->overallIndex;
        item.data.indexes.blockIndex   = vnode->blockIndex;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<csTimeDerivativeNode*>(adnode) )
    {
        csTimeDerivativeNode* tdnode = dynamic_cast<csTimeDerivativeNode*>(adnode);

        csComputeStackItem_t item;
        memset(&item, 0, sizeof(csComputeStackItem_t));
        item.opCode         = eOP_TimeDerivative;
        item.resultLocation = eOP_Result_to_value;
        item.data.indexes.overallIndex = tdnode->overallIndex;
        item.data.indexes.blockIndex   = tdnode->blockIndex;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<csUnaryNode*>(adnode) )
    {
        csUnaryNode* unode = dynamic_cast<csUnaryNode*>(adnode);

        csNode_t::CreateComputeStack(unode->operand.get(), computeStack);

        csComputeStackItem_t item;
        memset(&item, 0, sizeof(csComputeStackItem_t));
        item.opCode         = eOP_Unary;
        item.resultLocation = eOP_Result_to_value;
        item.function       = unode->function;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<csBinaryNode*>(adnode) )
    {
        csBinaryNode* bnode = dynamic_cast<csBinaryNode*>(adnode);

        // 1. Process left node:
        // Add ops to produce left value.
        if(!bnode->leftOperand)
            csThrowException("exInvalidCall");
        csNode_t::CreateComputeStack(bnode->leftOperand.get(), computeStack);
        if(computeStack.size() == 0)
            csThrowException("exInvalidCall");
        csComputeStackItem_t& litem = computeStack.back();
        litem.resultLocation = eOP_Result_to_lvalue;

        // 2. Process right node:
        // Add ops to produce right value.
        csNode_t::CreateComputeStack(bnode->rightOperand.get(), computeStack);
        if(computeStack.size() == 0)
            csThrowException("exInvalidCall");
        csComputeStackItem_t& ritem = computeStack.back();
        ritem.resultLocation = eOP_Result_to_rvalue;

        // 3. Add the binary operation
        csComputeStackItem_t item;
        memset(&item, 0, sizeof(csComputeStackItem_t));
        item.opCode         = eOP_Binary;
        item.resultLocation = eOP_Result_to_value;
        item.function       = bnode->function;
        computeStack.push_back(item);
    }
    else
    {
        csThrowException("Invalid node");
    }

    // "size" member of the first item contains the stack size.
    uint32_t stackSize = computeStack.size() - start;
    if(stackSize == 0)
        csThrowException("The size of the compute stack is zero");
    computeStack[start].size = stackSize;
}

uint32_t csNode_t::GetComputeStackSize(csNode_t* adnode)
{
    if(!adnode)
        csThrowException("exInvalidPointer");

    uint32_t noItems = 0;
    if( dynamic_cast<csConstantNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<csTimeNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<csVariableNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<csDegreeOfFreedomNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<csTimeDerivativeNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<csUnaryNode*>(adnode) )
    {
        csUnaryNode* unode = dynamic_cast<csUnaryNode*>(adnode);

        // 1. Process the operand
        noItems += csNode_t::GetComputeStackSize(unode->operand.get());

        // 2. Add the unary operation
        noItems++;
    }
    else if( dynamic_cast<csBinaryNode*>(adnode) )
    {
        csBinaryNode* bnode = dynamic_cast<csBinaryNode*>(adnode);

        // 1. Process left operand
        noItems += csNode_t::GetComputeStackSize(bnode->leftOperand.get());

        // 2. Process right operand
        noItems += csNode_t::GetComputeStackSize(bnode->rightOperand.get());

        // 3. Add the binary operation
        noItems++;
    }
    else
    {
        csThrowException("Invalid runtime node");
    }

    return noItems;
}

uint32_t csNode_t::GetComputeStackFlops(csNode_t*                                   adnode,
                                        const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                        const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    uint32_t flops = 0;

    if( dynamic_cast<csUnaryNode*>(adnode) )
    {
        csUnaryNode* unode = dynamic_cast<csUnaryNode*>(adnode);

        // 1. Process the operand
        flops += csNode_t::GetComputeStackFlops(unode->operand.get(), unaryOps, binaryOps);

        // 2. Add the unary operation
        std::map<csUnaryFunctions,uint32_t>::const_iterator it = unaryOps.find( unode->function );
        if(it != unaryOps.end())
            flops += it->second;
        else
            flops += 1;
    }
    else if( dynamic_cast<csBinaryNode*>(adnode) )
    {
        csBinaryNode* bnode = dynamic_cast<csBinaryNode*>(adnode);

        // 1. Process left operand
        flops += csNode_t::GetComputeStackFlops(bnode->leftOperand.get(), unaryOps, binaryOps);

        // 2. Process right operand
        flops += csNode_t::GetComputeStackFlops(bnode->rightOperand.get(), unaryOps, binaryOps);

        // 3. Add the binary operation
        std::map<csBinaryFunctions,uint32_t>::const_iterator it = binaryOps.find( bnode->function );
        if(it != binaryOps.end())
            flops += it->second;
        else
            flops += 1;
    }

    return flops;
}
*/

}
