/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2017
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "coreimpl.h"
#include "nodes.h"
#include "../IDAS_DAESolver/dae_array_matrix.h"
#include <stack>

namespace dae
{
namespace core
{
using namespace computestack;

/***********************************************************************************
   Create compute stack functions
***********************************************************************************/
inline void processFCVP(const size_t overallIndex,
                        const daeFloatCoefficientVariableProduct& fcvp,
                        std::vector<adComputeStackItem_t>& computeStack,
                        daeBlock* block,
                        daeDataProxy_t* dataProxy,
                        const std::map<size_t, size_t>& mapAssignedVarsIndexes)
{
    // Add compute items for coefficient * Variable[blockIndex] expressions.

    // 1. Add coefficient as left item.
    adComputeStackItem_t coeffitem;
    memset(&coeffitem, 0, sizeof(adComputeStackItem_t));
    coeffitem.opCode         = eOP_Constant;
    coeffitem.resultLocation = eOP_Result_to_lvalue;
    coeffitem.data.value = fcvp.coefficient;
    computeStack.push_back(coeffitem);

    // 2. Add variable value as right item.
    adComputeStackItem_t varitem;
    memset(&varitem, 0, sizeof(adComputeStackItem_t));
    varitem.resultLocation = eOP_Result_to_rvalue;
#ifdef ComputeStackDebug
    varitem.variableName = fcvp.variable->GetName();
#endif
    if(dataProxy->GetVariableType(overallIndex) == cnAssigned)
    {
        std::map<size_t, size_t>::const_iterator it = mapAssignedVarsIndexes.find(overallIndex);
        if(it == mapAssignedVarsIndexes.cend())
            daeDeclareAndThrowException(exInvalidCall);

        size_t dofIndex = it->second;
        varitem.opCode                        = eOP_DegreeOfFreedom;
        varitem.data.dof_indexes.overallIndex = overallIndex;
        varitem.data.dof_indexes.dofIndex     = dofIndex;
    }
    else
    {
        varitem.opCode                    = eOP_Variable;
        varitem.data.indexes.overallIndex = overallIndex;
        varitem.data.indexes.blockIndex   = block->FindVariableBlockIndex(overallIndex);
    }
    computeStack.push_back(varitem);

    // 3. Add the * binary operation and store it in the right value.
    adComputeStackItem_t item;
    memset(&item, 0, sizeof(adComputeStackItem_t));
    item.opCode         = eOP_Binary;
    item.function       = eMulti;
    item.resultLocation = eOP_Result_to_rvalue;
    computeStack.push_back(item);
}

void adNode::CreateComputeStack(adNode* adnode, std::vector<adComputeStackItem_t>& computeStack, daeBlock_t* pBlock, real_t scaling)
{
    if(!adnode)
        daeDeclareAndThrowException(exInvalidPointer);

    daeBlock* block           = dynamic_cast<daeBlock*>(pBlock);
    daeDataProxy_t* dataProxy = block->m_pDataProxy;
    const std::map<size_t, size_t>& mapAssignedVarsIndexes = dataProxy->GetAssignedVarsIndexes();

    uint32_t start = computeStack.size();
    if( dynamic_cast<adConstantNode*>(adnode) )
    {
        adConstantNode* cnode = dynamic_cast<adConstantNode*>(adnode);

        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_Constant;
        item.resultLocation = eOP_Result_to_value;
        item.data.value     = cnode->m_quantity.getValue();
        computeStack.push_back(item);
    }
    else if( dynamic_cast<adRuntimeParameterNode*>(adnode) )
    {
        adRuntimeParameterNode* pnode = dynamic_cast<adRuntimeParameterNode*>(adnode);

        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_Constant;
        item.resultLocation = eOP_Result_to_value;
        item.data.value     = *pnode->m_pdValue;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<adDomainIndexNode*>(adnode) )
    {
        adDomainIndexNode* dinode = dynamic_cast<adDomainIndexNode*>(adnode);

        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_Constant;
        item.resultLocation = eOP_Result_to_value;
        item.data.value     = *dinode->m_pdPointValue;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<adTimeNode*>(adnode) )
    {
        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_Time;
        item.resultLocation = eOP_Result_to_value;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<adInverseTimeStepNode*>(adnode) )
    {
        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_InverseTimeStep;
        item.resultLocation = eOP_Result_to_value;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<adRuntimeVariableNode*>(adnode) )
    {
        /* Important:
         *   vnode->m_bIsAssigned and vnode->m_nBlockIndex might not be initialised at the moment of creation
         *   of the compute stack (if it is created after SolveInitial then they are available).
         *   Therefore, use the information from the block.
         */
        adRuntimeVariableNode* vnode = dynamic_cast<adRuntimeVariableNode*>(adnode);

        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.resultLocation = eOP_Result_to_value;
#ifdef ComputeStackDebug
        item.variableName = vnode->m_pVariable->GetName();
#endif
        if(dataProxy->GetVariableType(vnode->m_nOverallIndex) == cnAssigned)
        {
            std::map<size_t, size_t>::const_iterator it = mapAssignedVarsIndexes.find(vnode->m_nOverallIndex);
            if(it == mapAssignedVarsIndexes.cend())
                daeDeclareAndThrowException(exInvalidCall);

            size_t dofIndex = it->second;
            item.opCode                        = eOP_DegreeOfFreedom;
            item.data.dof_indexes.overallIndex = vnode->m_nOverallIndex;
            item.data.dof_indexes.dofIndex     = dofIndex;
        }
        else
        {
            item.opCode                    = eOP_Variable;
            item.data.indexes.overallIndex = vnode->m_nOverallIndex;
            item.data.indexes.blockIndex   = block->FindVariableBlockIndex(vnode->m_nOverallIndex); // blockIndex has not been initialised yet in vnode
        }
        computeStack.push_back(item);
    }
    else if( dynamic_cast<adRuntimeTimeDerivativeNode*>(adnode) )
    {
        adRuntimeTimeDerivativeNode* tdnode = dynamic_cast<adRuntimeTimeDerivativeNode*>(adnode);

        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_TimeDerivative;
        item.resultLocation = eOP_Result_to_value;
#ifdef ComputeStackDebug
        item.variableName = tdnode->m_pVariable->GetName();
#endif
        item.data.indexes.overallIndex = tdnode->m_nOverallIndex;
        item.data.indexes.blockIndex   = block->FindVariableBlockIndex(tdnode->m_nOverallIndex); // blockIndex has not been initialised yet in tdnode
        computeStack.push_back(item);
    }
    else if( dynamic_cast<adFloatCoefficientVariableSumNode*>(adnode) )
    {
        adFloatCoefficientVariableSumNode* fcvsnode = dynamic_cast<adFloatCoefficientVariableSumNode*>(adnode);

        // Base
        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_Constant;
        item.resultLocation = eOP_Result_to_lvalue;
        item.data.value     = fcvsnode->m_base;
        computeStack.push_back(item);

        std::map<size_t, daeFloatCoefficientVariableProduct>::const_iterator it;
        for(it = fcvsnode->m_sum.begin(); it != fcvsnode->m_sum.end(); it++)
        {
            size_t overallIndex = it->first;
            const daeFloatCoefficientVariableProduct& fcvp = it->second;

            // Push the result of coeff*Var[bi] to rvalue
            processFCVP(overallIndex, fcvp, computeStack, block, dataProxy, mapAssignedVarsIndexes);

            // Add the + binary operation
            adComputeStackItem_t plusitem;
            memset(&plusitem, 0, sizeof(adComputeStackItem_t));
            plusitem.opCode   = eOP_Binary;
            plusitem.function = ePlus;
            // Always place the result into the lvalue.
            // It will be corrected for the last item.
            plusitem.resultLocation = eOP_Result_to_lvalue;
            computeStack.push_back(plusitem);
        }

        // The last item must always place the result to the values stack.
        if(computeStack.size() == 0)
            daeDeclareAndThrowException(exInvalidCall);
        adComputeStackItem_t& last_item = computeStack.back();
        last_item.resultLocation = eOP_Result_to_value;
    }
    else if( dynamic_cast<adUnaryNode*>(adnode) )
    {
        adUnaryNode* unode = dynamic_cast<adUnaryNode*>(adnode);

        adNode::CreateComputeStack(unode->node.get(), computeStack, block);

        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_Unary;
        item.resultLocation = eOP_Result_to_value;
        item.function       = unode->eFunction;
        computeStack.push_back(item);
    }
    else if( dynamic_cast<adBinaryNode*>(adnode) )
    {
        adBinaryNode* bnode = dynamic_cast<adBinaryNode*>(adnode);

        // 1. Process left node:
        // Add ops to produce left value.
        if(!bnode->left)
            daeDeclareAndThrowException(exInvalidPointer);
        adNode::CreateComputeStack(bnode->left.get(), computeStack, block);
        if(computeStack.size() == 0)
            daeDeclareAndThrowException(exInvalidCall);
        adComputeStackItem_t& litem = computeStack.back();
        litem.resultLocation = eOP_Result_to_lvalue;

        // 2. Process right node:
        // Add ops to produce right value.
        adNode::CreateComputeStack(bnode->right.get(), computeStack, block);
        if(computeStack.size() == 0)
            daeDeclareAndThrowException(exInvalidCall);
        adComputeStackItem_t& ritem = computeStack.back();
        ritem.resultLocation = eOP_Result_to_rvalue;

        // 3. Add the binary operation
        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_Binary;
        item.resultLocation = eOP_Result_to_value;
        item.function       = bnode->eFunction;
        computeStack.push_back(item);
    }
    else
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid runtime node";
        throw e;
    }

    // Will be added only for the top level call if scaling is not 1.0.
    if(scaling != 1.0)
    {
        adComputeStackItem_t item;
        memset(&item, 0, sizeof(adComputeStackItem_t));
        item.opCode         = eOP_Unary;
        item.resultLocation = eOP_Result_to_value;
        item.function       = eScaling;
        item.data.value     = scaling;
        computeStack.push_back(item);
    }

    // "size" member of the first item contains the stack size.
    uint32_t stackSize = computeStack.size() - start;
    if(stackSize == 0)
    {
        daeDeclareException(exInvalidCall);
        e << "The size of the compute stack is zero";
        throw e;
    }
    computeStack[start].size = stackSize;
}

uint32_t adNode::GetComputeStackSize(adNode* adnode)
{
    if(!adnode)
        daeDeclareAndThrowException(exInvalidPointer);

    uint32_t noItems = 0;
    if( dynamic_cast<adConstantNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<adRuntimeParameterNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<adDomainIndexNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<adTimeNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<adInverseTimeStepNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<adRuntimeVariableNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<adRuntimeTimeDerivativeNode*>(adnode) )
    {
        noItems++;
    }
    else if( dynamic_cast<adFloatCoefficientVariableSumNode*>(adnode) )
    {
        adFloatCoefficientVariableSumNode* fcvsnode = dynamic_cast<adFloatCoefficientVariableSumNode*>(adnode);

        // Base
        noItems++;

        std::map<size_t, daeFloatCoefficientVariableProduct>::const_iterator it;
        for(it = fcvsnode->m_sum.begin(); it != fcvsnode->m_sum.end(); it++)
        {
            noItems += 4;
        }
    }
    else if( dynamic_cast<adUnaryNode*>(adnode) )
    {
        adUnaryNode* unode = dynamic_cast<adUnaryNode*>(adnode);

        noItems += adNode::GetComputeStackSize(unode->node.get());

        noItems++;
    }
    else if( dynamic_cast<adBinaryNode*>(adnode) )
    {
        adBinaryNode* bnode = dynamic_cast<adBinaryNode*>(adnode);

        // 1. Process left node:
        noItems += adNode::GetComputeStackSize(bnode->left.get());

        // 2. Process right node:
        noItems += adNode::GetComputeStackSize(bnode->right.get());

        // 3. Add the binary operation
        noItems++;
    }
    else
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid runtime node";
        throw e;
    }

    return noItems;
}

/***********************************************************************************
   Estimate compute stack value/lvalue/rvalue sizes
***********************************************************************************/
void adNode::EstimateComputeStackSizes(const std::vector<adComputeStackItem_t>& computeStack, size_t start, size_t end,
                                      int& max_valueSize, int& max_lvalueSize, int& max_rvalueSize)
{
    int  valueSize = 0;
    int lvalueSize = 0;
    int rvalueSize = 0;

    max_valueSize  = 0;
    max_lvalueSize = 0;
    max_rvalueSize = 0;

    for(size_t i = start; i < end; i++)
    {
        const adComputeStackItem_t& item = computeStack[i];

        if(item.opCode == eOP_Constant)
        {
        }
        else if(item.opCode == eOP_Time)
        {
        }
        else if(item.opCode == eOP_InverseTimeStep)
        {
        }
        else if(item.opCode == eOP_Variable)
        {
        }
        else if(item.opCode == eOP_DegreeOfFreedom)
        {
        }
        else if(item.opCode == eOP_TimeDerivative)
        {
        }
        else if(item.opCode == eOP_Unary)
        {
            valueSize--;
        }
        else if(item.opCode == eOP_Binary)
        {
            lvalueSize--;
            rvalueSize--;
        }
        else
        {
            throw std::runtime_error("Invalid op code");
        }

        // At the end push the result into the requested stack.
        if(item.resultLocation == eOP_Result_to_value)
            valueSize++;
        else if(item.resultLocation == eOP_Result_to_lvalue)
            lvalueSize++;
        else if(item.resultLocation == eOP_Result_to_rvalue)
            rvalueSize++;
        else
            throw std::runtime_error("Invalid resultLocation code");

        max_valueSize  = std::max(max_valueSize,  valueSize);
        max_lvalueSize = std::max(max_lvalueSize, lvalueSize);
        max_rvalueSize = std::max(max_rvalueSize, rvalueSize);
    }
    valueSize--;

    if(valueSize != 0)
        throw std::runtime_error("Length of the value list is not zero");
    if(lvalueSize != 0)
        throw std::runtime_error("Length of the lvalue list is not zero");
    if(rvalueSize != 0)
        throw std::runtime_error("Length of the rvalue list is not zero");
}

}
}
