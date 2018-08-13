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

std::string csConstantNode::ToLatex() const
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

std::string csTimeNode::ToLatex() const
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
        throw std::runtime_error("DOF values not set in csNodeEvaluationContext_t");

    return adouble_t_(EC.dofs[dofIndex], 0.0);
}

void csDegreeOfFreedomNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
}

void csDegreeOfFreedomNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
}

std::string csDegreeOfFreedomNode::ToLatex() const
{
    char buffer[256];
    std::snprintf(buffer, 256, "y[%d]", dofIndex);
    return std::string(buffer);
}


csVariableNode::csVariableNode(uint32_t overallIndex_, uint32_t blockIndex_)
{
    overallIndex = overallIndex_;
    blockIndex   = blockIndex_;
}

adouble_t csVariableNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    if(!EC.values)
        throw std::runtime_error("Variable values not set in csNodeEvaluationContext_t");

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

std::string csVariableNode::ToLatex() const
{
    char buffer[256];
    std::snprintf(buffer, 256, "x[%d]", blockIndex);
    return std::string(buffer);
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
        throw std::runtime_error("Derivative values not set in csNodeEvaluationContext_t");

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

std::string csTimeDerivativeNode::ToLatex() const
{
    char buffer[256];
    std::snprintf(buffer, 256, "x_t[%d]", blockIndex);
    return std::string(buffer);
}

/* Unary operators and functions: -, sqrt, log, log10, exp, sin, cos, ... */
csUnaryNode::csUnaryNode(csUnaryFunctions function_, csNodePtr operand_)
{
    if(!operand_)
        throw std::runtime_error("Invalid csUnaryNode operand");

    function = function_;
    operand  = operand_;
}

adouble_t csUnaryNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    if(!operand)
        throw std::runtime_error("Operand not set in csUnaryNode");

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
        throw std::runtime_error("exNotImplemented");
    }
    return res;
}

void csUnaryNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
    if(!operand)
        throw std::runtime_error("Operand not set in csUnaryNode");

    operand->CollectVariableIndexes(variableIndexes);
}

void csUnaryNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
    if(!operand)
        throw std::runtime_error("Operand not set in csUnaryNode");

    operand->CollectVariableTypes(variableTypes);
}

std::string csUnaryNode::ToLatex() const
{
    if(!operand)
        throw std::runtime_error("Operand not set in csUnaryNode");

    std::string res;
    std::string s_operand = operand->ToLatex();

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
        throw std::runtime_error("exNotImplemented");
    }
    return res;
}

/* Binary operators and functions: +, -, *, /, **, pow, atan2, min, max */
csBinaryNode::csBinaryNode(csBinaryFunctions function_, csNodePtr leftOperand_, csNodePtr rightOperand_)
{
    if(!leftOperand_)
        throw std::runtime_error("Invalid left csBinaryNode operand");
    if(!rightOperand_)
        throw std::runtime_error("Invald right csBinaryNode operand");

    function     = function_;
    leftOperand  = leftOperand_;
    rightOperand = rightOperand_;
}

adouble_t csBinaryNode::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    if(!leftOperand)
        throw std::runtime_error("Left operand not set in csBinaryNode");
    if(!rightOperand)
        throw std::runtime_error("Right operand not set in csBinaryNode");

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
        throw std::runtime_error("exNotImplemented");
    }
    return res;
}

void csBinaryNode::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
    if(!leftOperand)
        throw std::runtime_error("Left operand not set in csBinaryNode");
    if(!rightOperand)
        throw std::runtime_error("Right operand not set in csBinaryNode");

    leftOperand->CollectVariableIndexes(variableIndexes);
    rightOperand->CollectVariableIndexes(variableIndexes);
}

void csBinaryNode::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
    if(!leftOperand)
        throw std::runtime_error("Left operand not set in csBinaryNode");
    if(!rightOperand)
        throw std::runtime_error("Right operand not set in csBinaryNode");

    leftOperand->CollectVariableTypes(variableTypes);
    rightOperand->CollectVariableTypes(variableTypes);
}

std::string csBinaryNode::ToLatex() const
{
    if(!leftOperand)
        throw std::runtime_error("Left operand not set in csBinaryNode");
    if(!rightOperand)
        throw std::runtime_error("Right operand not set in csBinaryNode");

    std::string res;
    std::string l_operand = leftOperand->ToLatex();
    std::string r_operand = rightOperand->ToLatex();

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
        throw std::runtime_error("exNotImplemented");
    }
    res += "\\right)"; // End

    return res;
}

/* Static functions. */
void csNode_t::CreateComputeStack(csNode_t* adnode, std::vector<csComputeStackItem_t>& computeStack)
{
    if(!adnode)
        throw std::runtime_error("exInvalidCall");

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
            throw std::runtime_error("exInvalidCall");
        csNode_t::CreateComputeStack(bnode->leftOperand.get(), computeStack);
        if(computeStack.size() == 0)
            throw std::runtime_error("exInvalidCall");
        csComputeStackItem_t& litem = computeStack.back();
        litem.resultLocation = eOP_Result_to_lvalue;

        // 2. Process right node:
        // Add ops to produce right value.
        csNode_t::CreateComputeStack(bnode->rightOperand.get(), computeStack);
        if(computeStack.size() == 0)
            throw std::runtime_error("exInvalidCall");
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
        throw std::runtime_error("Invalid node");
    }

    // "size" member of the first item contains the stack size.
    uint32_t stackSize = computeStack.size() - start;
    if(stackSize == 0)
        throw std::runtime_error("The size of the compute stack is zero");
    computeStack[start].size = stackSize;
}

uint32_t csNode_t::GetComputeStackSize(csNode_t* adnode)
{
    if(!adnode)
        throw std::runtime_error("exInvalidPointer");

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
        throw std::runtime_error("Invalid runtime node");
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


}
