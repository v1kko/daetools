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
using namespace cs;
#include <typeinfo>

namespace daetools
{
namespace core
{
csNode_wrapper::csNode_wrapper(daeEquationExecutionInfo* eei, daeBlock* pblock)
{
    eeinfo = eei;
    block  = pblock;
    if(!eeinfo)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!block)
        daeDeclareAndThrowException(exInvalidPointer);
}

csNode_wrapper::~csNode_wrapper()
{

}

std::string csNode_wrapper::ToLatex(const cs::csModelBuilder_t* mb) const
{
    adNode* adnode = eeinfo->GetEquationEvaluationNodeRawPtr();
    if(!adnode)
        daeDeclareAndThrowException(exInvalidPointer);

    daeDataProxy_t*	pDataProxy = block->GetDataProxy();
    daeModel*       pModel     = dynamic_cast<daeModel*>(pDataProxy->GetTopLevelModel());

    daeNodeSaveAsContext c(pModel);
    return adnode->SaveAsLatex(&c);
}

adouble_t csNode_wrapper::Evaluate(const csNodeEvaluationContext_t& EC) const
{
    daeDeclareAndThrowException(exInvalidCall);
    return adouble_t();
}

static void collectVariableTypes(adNode* adnode, std::vector<int32_t>& variableTypes, std::map<size_t, size_t>& blockIndexes)
{
    if(!adnode)
        daeDeclareAndThrowException(exInvalidPointer);

    adUnaryNode* unode = NULL;
    adBinaryNode* bnode = NULL;
    adRuntimeTimeDerivativeNode* tdnode = NULL;
    std::map<size_t, size_t>::const_iterator cit;
    std::map<size_t, size_t>::const_iterator cit_end = blockIndexes.end();

    // No need to check all classes (only differential variables are set).
    if( tdnode = dynamic_cast<adRuntimeTimeDerivativeNode*>(adnode) )
    {
        cit = blockIndexes.find(tdnode->m_nOverallIndex);
        if(cit == cit_end) // It must be found - it is not a degree of freedom
            daeDeclareAndThrowException(exInvalidCall);

        variableTypes[cit->second] = cs::csDifferentialVariable;
    }
    else if( unode = dynamic_cast<adUnaryNode*>(adnode) )
    {
        collectVariableTypes(unode->node.get(), variableTypes, blockIndexes);
    }
    else if( bnode = dynamic_cast<adBinaryNode*>(adnode) )
    {
        collectVariableTypes(bnode->left.get(), variableTypes, blockIndexes);
        collectVariableTypes(bnode->right.get(), variableTypes, blockIndexes);
    }
}

void csNode_wrapper::CollectVariableTypes(std::vector<int32_t>& variableTypes) const
{
    adNode* adnode = eeinfo->GetEquationEvaluationNodeRawPtr();
    if(!adnode)
        daeDeclareAndThrowException(exInvalidPointer);

    // If evaluation mode is ComputeStack then adNode classes are not initialised
    // and blockIndexes and isAssigned are not set.
    // In that case, blockIndexes must be obtained from the block mapVariableIndexes.
    std::map<size_t, size_t>& mapIndexes = block->m_mapVariableIndexes; // map{overallIndex : blockIndex}

    collectVariableTypes(adnode, variableTypes, mapIndexes);
}

void csNode_wrapper::CollectVariableIndexes(std::map<uint32_t,uint32_t>& variableIndexes) const
{
    adNode* adnode = eeinfo->GetEquationEvaluationNodeRawPtr();
    if(!adnode)
        daeDeclareAndThrowException(exInvalidPointer);

    // No need to do anything, just copy the indexes from the eeinfo.
    // Important: mind the reversed order map<bi, oi> (not like in daetools: map<oi, bi>).
    for(std::map<size_t, size_t>::const_iterator cit = eeinfo->m_mapIndexes.begin(); cit != eeinfo->m_mapIndexes.end(); cit++)
        variableIndexes[cit->second] = cit->first;

//    printf("collected indexes:\n");
//    for(std::map<uint32_t, uint32_t>::const_iterator cit = variableIndexes.begin(); cit != variableIndexes.end(); cit++)
//        printf("<%lu, %lu> ", cit->first, cit->second);
//    printf("\n");
}

uint32_t csNode_wrapper::CreateComputeStack(std::vector<csComputeStackItem_t>& computeStack)
{
    adNode* adnode = eeinfo->GetEquationEvaluationNodeRawPtr();
    if(!adnode)
        daeDeclareAndThrowException(exInvalidPointer);

    uint32_t sizeBefore = computeStack.size();
    adNode::CreateComputeStack(adnode, computeStack, block, eeinfo->m_dScaling);
    uint32_t sizeAfter = computeStack.size();

    return uint32_t(sizeAfter - sizeBefore);
}

static uint32_t getComputeStackSize(adNode* adnode)
{
    if(!adnode)
        daeDeclareAndThrowException(exInvalidPointer);

    adUnaryNode* unode = NULL;
    adBinaryNode* bnode = NULL;
    adFloatCoefficientVariableSumNode* fcvsnode = NULL;

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
    else if( fcvsnode = dynamic_cast<adFloatCoefficientVariableSumNode*>(adnode) )
    {
        // Base
        noItems++;

        std::map<size_t, daeFloatCoefficientVariableProduct>::const_iterator it;
        for(it = fcvsnode->m_sum.begin(); it != fcvsnode->m_sum.end(); it++)
        {
            noItems += 4;
        }
    }
    else if( unode = dynamic_cast<adUnaryNode*>(adnode) )
    {
        // 1. Process the operand:
        noItems += getComputeStackSize(unode->node.get());

        // 2. Add the unary operation
        noItems++;
    }
    else if( bnode = dynamic_cast<adBinaryNode*>(adnode) )
    {
        // 1. Process the left operand:
        noItems += getComputeStackSize(bnode->left.get());

        // 2. Process the right operand:
        noItems += getComputeStackSize(bnode->right.get());

        // 3. Add the binary operation
        noItems++;
    }
    else if( dynamic_cast<adScalarExternalFunctionNode*>(adnode) )
    {
        daeDeclareException(exInvalidCall);
        e << "External functions are not supported by OpenCS framework";
        throw e;
    }
    else if( dynamic_cast<adThermoPhysicalPropertyPackageScalarNode*>(adnode) )
    {
        daeDeclareException(exInvalidCall);
        e << "Thermo-physical property packages are not supported by OpenCS framework";
        throw e;
    }
    else
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid runtime node";
        throw e;
    }

    return noItems;
}

uint32_t csNode_wrapper::GetComputeStackSize()
{
    adNode* adnode = eeinfo->GetEquationEvaluationNodeRawPtr();
    return getComputeStackSize(adnode);
}

static uint32_t getComputeStackFlops(adNode* adnode,
                                     const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                     const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    if(!adnode)
        daeDeclareAndThrowException(exInvalidPointer);

    adUnaryNode* unode = NULL;
    adBinaryNode* bnode = NULL;
    adFloatCoefficientVariableSumNode* fcvsnode = NULL;

    size_t flops = 0;
    // No need to check all classes (for the rest it is equal to zero).
    if( fcvsnode = dynamic_cast<adFloatCoefficientVariableSumNode*>(adnode) )
    {
        // One addition and one multiplication per item
        std::map<csBinaryFunctions,uint32_t>::const_iterator it;

        std::map<size_t, daeFloatCoefficientVariableProduct>::const_iterator it_fcvp;
        for(it_fcvp = fcvsnode->m_sum.begin(); it_fcvp != fcvsnode->m_sum.end(); it_fcvp++)
        {
            it = binaryOps.find(cs::ePlus);
            if(it != binaryOps.end())
                flops += it->second;
            else
                flops += 1;

            it = binaryOps.find(cs::eMulti);
            if(it != binaryOps.end())
                flops += it->second;
            else
                flops += 1;
        }
    }
    else if( unode = dynamic_cast<adUnaryNode*>(adnode) )
    {
        // 1. Process the operand
        flops += getComputeStackFlops(unode->node.get(), unaryOps, binaryOps);

        // 2. Add the unary operation
        std::map<csUnaryFunctions,uint32_t>::const_iterator it = unaryOps.find( (csUnaryFunctions)unode->eFunction );
        if(it != unaryOps.end())
            flops += it->second;
        else
            flops += 1;
    }
    else if( bnode = dynamic_cast<adBinaryNode*>(adnode) )
    {
        // 1. Process left operand
        flops += getComputeStackFlops(bnode->left.get(), unaryOps, binaryOps);

        // 2. Process right operand
        flops += getComputeStackFlops(bnode->right.get(), unaryOps, binaryOps);

        // 3. Add the binary operation
        std::map<csBinaryFunctions,uint32_t>::const_iterator it = binaryOps.find( (csBinaryFunctions)bnode->eFunction );
        if(it != binaryOps.end())
            flops += it->second;
        else
            flops += 1;
    }

    return flops;
}

uint32_t csNode_wrapper::GetComputeStackFlops(const std::map<csUnaryFunctions,uint32_t>&  unaryOps,
                                              const std::map<csBinaryFunctions,uint32_t>& binaryOps)
{
    adNode* adnode = eeinfo->GetEquationEvaluationNodeRawPtr();
    return getComputeStackFlops(adnode, unaryOps, binaryOps);
}


}
}
