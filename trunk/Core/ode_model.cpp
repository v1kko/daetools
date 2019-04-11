#include "coreimpl.h"
#include "nodes.h"
#include "../variable_types.h"

namespace dae
{
namespace core
{

///*********************************************************
//    daeODEModel
//*********************************************************/
//daeODEModel::daeODEModel(std::string strName, daeModel* pModel, std::string strDescription):
//      daeModel(strName, pModel, strDescription)
//{
//}

///******************************************************************
//    daeFiniteElementEquation
//*******************************************************************/
//daeODEEquation::daeODEEquation()
//{
//}

//daeODEEquation::~daeODEEquation()
//{
//}

//void daeODEEquation::CreateEquationExecutionInfos(daeModel* pModel, std::vector<daeEquationExecutionInfo*>& ptrarrEqnExecutionInfosCreated, bool bAddToTheModel)
//{
//    // May be declined if requested in the daetools.cfg config file.
//    adNodeImpl::SetMemoryPool(eRuntimeNodesPool);

//    daeEquation::CreateEquationExecutionInfos(pModel, ptrarrEqnExecutionInfosCreated, bAddToTheModel);
//}

//void daeODEEquation::Open(io::xmlTag_t* pTag)
//{
//    daeEquation::Open(pTag);
//}

//void daeODEEquation::Save(io::xmlTag_t* pTag) const
//{
//    daeEquation::Save(pTag);
//}

//void daeODEEquation::OpenRuntime(io::xmlTag_t* pTag)
//{
//}

//void daeODEEquation::SaveRuntime(io::xmlTag_t* pTag) const
//{
//    daeEquation::SaveRuntime(pTag);
//}

//bool daeODEEquation::CheckObject(std::vector<string>& strarrErrors) const
//{
//    return true;
//}

//void daeODEEquation::SetODEVariable(adouble odeVariable)
//{
//    // ODE Variable must be a variable specified by its adSetupVariableNode node.
//    if(!odeVariable.node || !dynamic_cast<adSetupVariableNode*>(odeVariable.node.get()))
//    {
//        daeDeclareException(exInvalidCall);
//        e << "Invalid ODE variable specified for equation " << GetCanonicalName()
//          << " (must be a variable not NULL nor an expression)";
//        throw e;
//    }

//    m_pODEVariableNode = odeVariable.node;
//}

//adouble daeODEEquation::GetODEVariable(void) const
//{
//    adouble ad;
//    ad.setGatherInfo(true);
//    ad.node = m_pODEVariableNode;
//    return ad;
//}

}
}
