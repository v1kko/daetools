#include "coreimpl.h"
#include "nodes.h"
#include "../variable_types.h"

namespace dae 
{
namespace core 
{

/*********************************************************
    daeFiniteElementModel
*********************************************************/
daeFiniteElementModel::daeFiniteElementModel(std::string strName, daeModel* pModel, std::string strDescription, daeFiniteElementObject_t* fe):
      daeModel(strName, pModel, strDescription),
      m_fe(fe),
      m_omega("&Omega;", this, unit(), "Omega")
{
    daeVariable* pVariable;

    if(!m_fe)
        daeDeclareAndThrowException(exInvalidPointer);

    m_fe->SetModel(this);

    // Initialize daetools wrapper matrices and arrays that will be used by adFEMatrixItem/VectorItem nodes
    m_Aij.reset(m_fe->Asystem());
    m_Mij.reset(m_fe->Msystem());
    m_Fi.reset(m_fe->Fload());

    daeFiniteElementObjectInfo feObjectInfo = m_fe->GetObjectInfo();

    // Initialize domains and parameters
    std::vector<daePoint> coords(feObjectInfo.m_nTotalNumberDOFs);
    m_omega.CreateUnstructuredGrid(coords);

    std::cout << "feObjectInfo.m_nTotalNumberDOFs = " << feObjectInfo.m_nTotalNumberDOFs << std::endl;

    for(size_t i = 0; i < feObjectInfo.m_VariableInfos.size(); i++)
    {
        const daeFiniteElementVariableInfo& feVarInfo = feObjectInfo.m_VariableInfos[i];

        std::cout << "  m_strName        = " << feVarInfo.m_strName        << std::endl;
        std::cout << "  m_strDescription = " << feVarInfo.m_strDescription << std::endl;
        std::cout << "  m_nMultiplicity  = " << feVarInfo.m_nMultiplicity  << std::endl;
        std::cout << "  m_nNumberOfDOFs  = " << feVarInfo.m_nNumberOfDOFs  << std::endl;

        daeDomain* omega_i = new daeDomain("&Omega;_"+toString(i), this, unit(), "FE sub domain " + toString(i));
        omega_i->CreateArray(feVarInfo.m_nNumberOfDOFs);
        m_ptrarrFESubDomains.push_back(omega_i);

        pVariable = new daeVariable(feVarInfo.m_strName, variable_types::no_t, this, feVarInfo.m_strDescription, omega_i);
        m_ptrarrFEVariables.push_back(pVariable);
   }
}

void daeFiniteElementModel::DeclareEquations(void)
{
    daeModel::DeclareEquations();
}

void daeFiniteElementModel::DeclareEquationsForWeakForm(void)
{
    if(!m_fe)
        daeDeclareAndThrowException(exInvalidPointer);

    m_fe->AssembleSystem();

    daeFiniteElementObjectInfo feObjectInfo = m_fe->GetObjectInfo();

    size_t startRow = 0;
    size_t endRow   = feObjectInfo.m_nTotalNumberDOFs;

    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
    daeFiniteElementEquation* pEquation = new daeFiniteElementEquation(*this, m_ptrarrFEVariables, startRow, endRow);

    if(!pCurrentState)
    {
        string strEqName = "dealIIFESystem_" + toString<size_t>(m_ptrarrEquations.size());
        pEquation->SetName(strEqName);
        AddEquation(pEquation);
    }
    else
    {
        string strEqName = "dealIIFESystem_" + toString<size_t>(pCurrentState->m_ptrarrEquations.size());
        pEquation->SetName(strEqName);
        pCurrentState->AddEquation(pEquation);
    }

    pEquation->SetDescription("");
    pEquation->SetScaling(1.0);
    pEquation->daeEquation::DistributeOnDomain(m_omega, eClosedClosed);
    pEquation->SetResidual(Constant(0.0));


    // Add surface integral equations
    int counter = 0;
    const std::map< unsigned int, std::vector< std::pair<adouble,adouble> > >* mapSI = m_fe->SurfaceIntegrals();
    std::map< unsigned int, std::vector< std::pair<adouble,adouble> > >::const_iterator cit = mapSI->begin();
    for(; cit != mapSI->end(); cit++)
    {
        const std::vector< std::pair<adouble,adouble> >& arrPairsVaribleIntegral = cit->second;

        for(size_t i = 0; i < arrPairsVaribleIntegral.size(); i++)
        {
            const std::pair<adouble,adouble>& p = arrPairsVaribleIntegral[i];
            const adouble& ad_variable = p.first;
            const adouble& ad_integral = p.second;

            if(!ad_variable.node || !dynamic_cast<adSetupVariableNode*>(ad_variable.node.get()))
            {
                daeDeclareException(exInvalidCall);
                e << "The variable to store the result of the surface integral is not specified (must be a single variable)";
                throw e;
            }

            adSetupVariableNode* setupVarNode = dynamic_cast<adSetupVariableNode*>(ad_variable.node.get());
            daeVariable* variable = setupVarNode->m_pVariable;
            if(!variable)
                daeDeclareAndThrowException(exInvalidPointer);

            daeEquation* pEq = new daeEquation();
            string strEqName = "FESurfaceIntegral_" + variable->GetName();
            pEq->SetName(strEqName);
            //pEq->SetBuildJacobianExpressions(true);

            if(!pCurrentState)
            {
                AddEquation(pEq);
            }
            else
            {
                pCurrentState->AddEquation(pEq);
            }

            pEq->SetDescription("");
            pEq->SetScaling(1.0);

            //daeNodeSaveAsContext c(this);
            //adSetupVariableNode* psvn = dynamic_cast<adSetupVariableNode*>(ad_variable.node.get());
            //printf("ad_variable = %s (%s)\n", ad_variable.node->SaveAsLatex(&c).c_str(), (psvn ? psvn->GetObjectClassName().c_str() : "nullptr"));
            //printf("ad_integral = %s\n", ad_integral.node->SaveAsLatex(&c).c_str());

            pEq->SetResidual(ad_variable - ad_integral);

            counter++;
        }
    }

    // Add volume integral equations
    counter = 0;
    const std::vector< std::pair<adouble,adouble> >* vecVI = m_fe->VolumeIntegrals();

    for(size_t i = 0; i < vecVI->size(); i++)
    {
        const std::pair<adouble,adouble>& p = (*vecVI)[i];
        const adouble& ad_variable = p.first;
        const adouble& ad_integral = p.second;

        if(!ad_variable.node || !dynamic_cast<adSetupVariableNode*>(ad_variable.node.get()))
        {
            daeDeclareException(exInvalidCall);
            e << "The variable to store the result of the volume integral is not specified (must be a single variable)";
            throw e;
        }

        adSetupVariableNode* setupVarNode = dynamic_cast<adSetupVariableNode*>(ad_variable.node.get());
        daeVariable* variable = setupVarNode->m_pVariable;
        if(!variable)
            daeDeclareAndThrowException(exInvalidPointer);

        daeEquation* pEq = new daeEquation();
        string strEqName = "FEVolumeIntegral_" + variable->GetName();
        pEq->SetName(strEqName);
        //pEq->SetBuildJacobianExpressions(true);

        if(!pCurrentState)
        {
            AddEquation(pEq);
        }
        else
        {
            pCurrentState->AddEquation(pEq);
        }

        pEq->SetDescription("");
        pEq->SetScaling(1.0);

        //daeNodeSaveAsContext c(this);
        //adSetupVariableNode* psvn = dynamic_cast<adSetupVariableNode*>(ad_variable.node.get());
        //printf("ad_variable = %s (%s)\n", ad_variable.node->SaveAsLatex(&c).c_str(), (psvn ? psvn->GetObjectClassName().c_str() : "nullptr"));
        //printf("ad_integral = %s\n", ad_integral.node->SaveAsLatex(&c).c_str());

        pEq->SetResidual(ad_variable - ad_integral);

        counter++;
    }
}

void daeFiniteElementModel::UpdateEquations(const daeExecutionContext* pExecutionContext)
{
    daeModel::UpdateEquations(pExecutionContext);

    // Here we call ReAssembleSystem() to update the system matrices and rhs if the flag NeedsReAssembling is set.
    if(m_fe->NeedsReAssembling())
    {
        std::cout << "daeFiniteElementModel: re-assembling the system" << std::endl;
        m_fe->ReAssembleSystem();
    }
}

/******************************************************************
    daeFiniteElementEquation
*******************************************************************/
daeFiniteElementEquation::daeFiniteElementEquation(const daeFiniteElementModel& feModel, const std::vector<daeVariable*>& arrVariables, size_t startRow, size_t endRow):
    m_FEModel(feModel),
    m_ptrarrVariables(arrVariables),
    m_startRow(startRow),
    m_endRow(endRow)
{
}

daeFiniteElementEquation::~daeFiniteElementEquation()
{
}

struct CompareKeys
{
    template <typename Pair>
    bool operator() (Pair const &lhs, Pair const &rhs) const
    {
        return lhs.first == rhs.first;
    }
};

inline adouble create_adouble(adNode* n)
{
    return adouble(0.0, 0.0, true, n);
}

void daeFiniteElementEquation::CreateEquationExecutionInfos(daeModel* pModel, std::vector<daeEquationExecutionInfo*>& ptrarrEqnExecutionInfosCreated, bool bAddToTheModel)
{
    daeVariable* variable;
    size_t indexes[1], counter;
    size_t nNoDomains, nIndex, column, internalVariableIndex;
    adouble a_Aij, a_Mij, a_Fi;
    daeFiniteElementObject_t* fe ;
    daeEquationExecutionInfo* pEquationExecutionInfo;
    std::vector<unsigned int> narrRowIndices;
    std::vector< std::pair<size_t, daeVariable*> > arrGlobalDOFToVariablePoint;

    if(!pModel)
        daeDeclareAndThrowException(exInvalidPointer);

    nNoDomains = m_ptrarrDistributedEquationDomainInfos.size();
    if(nNoDomains != 1)
        daeDeclareAndThrowException(exInvalidCall);
    ptrarrEqnExecutionInfosCreated.clear();

    /***************************************************************************************************/
    // Try to predict requirements and reserve the memory for all EquationExecutionInfos (could save a lot of memory)
    // AddEquationExecutionInfo() does not use dae_push_back() !!!
    size_t NoEqns = this->GetNumberOfEquations();
    if(bAddToTheModel)
    {
        pModel->m_ptrarrEquationExecutionInfos.reserve( pModel->m_ptrarrEquationExecutionInfos.size() + NoEqns );
    }
    else
    {
        ptrarrEqnExecutionInfosCreated.reserve(NoEqns);
        this->m_ptrarrEquationExecutionInfos.reserve(NoEqns);
    }
    /***************************************************************************************************/

    // At the moment we do not this...
    daeExecutionContext EC;
    EC.m_pDataProxy					= pModel->m_pDataProxy.get();
    EC.m_pEquationExecutionInfo		= NULL;
    EC.m_eEquationCalculationMode	= eGatherInfo;

    boost::shared_ptr<daeDataProxy_t> pDataProxy = pModel->GetDataProxy();
    daeModel* pTopLevelModel = dynamic_cast<daeModel*>(pDataProxy->GetTopLevelModel());
    pTopLevelModel->PropagateGlobalExecutionContext(&EC);

    bool bPrintInfo = pModel->m_pDataProxy->PrintInfo();
    if(bPrintInfo)
        std::cout << "FEEquation start = " << m_startRow << " end = " << m_endRow << std::endl;

    fe = m_FEModel.m_fe;
    if(!fe)
        daeDeclareAndThrowException(exInvalidPointer);

    counter = 0;
    arrGlobalDOFToVariablePoint.resize(NoEqns);
    for(size_t i = 0; i < m_ptrarrVariables.size(); i++)
    {
        variable = m_ptrarrVariables[i];
        for(size_t j = 0; j < variable->GetNumberOfPoints(); j++)
        {
            arrGlobalDOFToVariablePoint[counter] = std::pair<size_t, daeVariable*>(j, variable);
            counter++;
            if(bPrintInfo)
                std::cout << (boost::format("%d : (%d : %s)") % counter % j % variable->GetName()).str() << std::endl;
        }
    }
    if(counter != m_FEModel.m_Aij->GetNrows())
        daeDeclareAndThrowException(exInvalidCall);

    counter = 0;
    for(size_t row = m_startRow; row < m_endRow; row++)
    {
        pEquationExecutionInfo = new daeEquationExecutionInfo(this);
        pEquationExecutionInfo->m_dScaling = this->m_dScaling;

        pEquationExecutionInfo->m_narrDomainIndexes.resize(1);
        pEquationExecutionInfo->m_narrDomainIndexes[0] = counter++;

        // Set EEI for the forthcoming calls to adNode::Evaluate()
        EC.m_pEquationExecutionInfo = pEquationExecutionInfo;

        // Reset equation's contributions
        a_Aij = 0;
        a_Mij = 0;

        // RHS

        // If existing, evaluate Setup adNode from the matrix into a Runtime adNode
        //a_Fi = create_adouble(new adFEVectorItemNode("f", *m_FEModel.m_Fi, row, unit()));
        adouble adFi_item = m_FEModel.m_Fi->GetItem(row);
        if(adFi_item.node)
            a_Fi = adFi_item.node->Evaluate(&EC);
        else
            a_Fi = create_adouble(new adConstantNode(adFi_item.getValue()));

        narrRowIndices.clear();
        fe->RowIndices(row, narrRowIndices);
        for(size_t i = 0; i < narrRowIndices.size(); i++)
        {
            // This is a global DOF index (from 0 to Ndofs)
            column = narrRowIndices[i];

            // internalVariableIndex is a local index (within a variable) that matches variable's global DOF index
            internalVariableIndex = arrGlobalDOFToVariablePoint[column].first;
            variable              = arrGlobalDOFToVariablePoint[column].second;

            // Set it to be the variable's local index (we need it to create adoubles with runtime nodes)
            indexes[0] = internalVariableIndex;

            //if(!a_Aij.node)
            //    a_Aij =         create_adouble(new adFEMatrixItemNode("A", *m_FEModel.m_Aij, row, column, unit())) * variable->Create_adouble(indexes, 1);
            //else
            //    a_Aij = a_Aij + create_adouble(new adFEMatrixItemNode("A", *m_FEModel.m_Aij, row, column, unit())) * variable->Create_adouble(indexes, 1);

            adouble adAij_item = m_FEModel.m_Aij->GetItem(row, column);
            if(adAij_item.node)
            {
                adouble ad = adAij_item.node->Evaluate(&EC) * variable->Create_adouble(indexes, 1);
                if(!a_Aij.node)
                    a_Aij = ad;
                else
                    a_Aij = a_Aij + ad;
            }
            else
            {
                if(adAij_item.getValue() != 0.0)
                {
                    adouble ad = create_adouble(new adConstantNode(adAij_item.getValue())) * variable->Create_adouble(indexes, 1);
                    if(!a_Aij.node)
                        a_Aij = ad;
                    else
                        a_Aij = a_Aij + ad;
                }
            }

            /* ACHTUNG, ACHTUNG!!
               The mass matrix M is not going to change - wherever we have an item in M equal to zero it is going to stay zero
               (meaning that the FiniteElement object cannot suddenly sneak in differential variables into the system AFTER initialization).
               Therefore, skip an item if we encounter a zero.
            */
            adouble adMij_item = m_FEModel.m_Mij->GetItem(row, column);
            if(adMij_item.node || adMij_item.getValue() != 0.0)
            {
                //if(!a_Mij.node)
                //    a_Mij =         create_adouble(new adFEMatrixItemNode("M", *m_FEModel.m_Mij, row, column, unit())) * variable->Calculate_dt(indexes, 1);
                //else
                //    a_Mij = a_Mij + create_adouble(new adFEMatrixItemNode("M", *m_FEModel.m_Mij, row, column, unit())) * variable->Calculate_dt(indexes, 1);

                if(adMij_item.node)
                {

                    adouble ad = adMij_item.node->Evaluate(&EC) * variable->Calculate_dt(indexes, 1);
                    if(!a_Mij.node)
                        a_Mij = ad;
                    else
                        a_Mij = a_Mij + ad;
                }
                else
                {
                    if(adMij_item.getValue() != 0.0)
                    {
                        adouble ad = create_adouble(new adConstantNode(adMij_item.getValue())) * variable->Calculate_dt(indexes, 1);
                        if(!a_Mij.node)
                            a_Mij = ad;
                        else
                            a_Mij = a_Mij + ad;
                    }
                }

                nIndex = variable->GetOverallIndex() + internalVariableIndex;
                m_pModel->m_pDataProxy->SetVariableTypeGathered(nIndex, cnDifferential);
            }
        }

        pEquationExecutionInfo->m_EquationEvaluationNode = (a_Mij + a_Aij - a_Fi).node;

        /* ACHTUNG, ACHTUNG!!
           We already have m_EquationEvaluationNode and a call to GatherInfo() seems unnecesary.
           This way we avoided creation of setup nodes first and then evaluating them into the runtime ones
           during the GatherInfo() function. Are there some side-effects of this vodoo-mojo?
           Update: there are side-effects!
              mapIndexes is not populated with indexes during the GatherInfo call.
              We have to build the map manually since the runtime node has not been formed by evaluation of the setup node.
              Note: Here we DO NOT add fixed variables to the map.

        pEquationExecutionInfo->GatherInfo(EC, pModel);
        */

        pEquationExecutionInfo->m_mapIndexes.clear();
        pEquationExecutionInfo->m_EquationEvaluationNode->AddVariableIndexToArray(pEquationExecutionInfo->m_mapIndexes, false);

        if(bAddToTheModel)
            pModel->AddEquationExecutionInfo(pEquationExecutionInfo);
        else
            ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);

        // This vector is redundant - all EquationExecutionInfos exist in models and states too.
        // However, daeEquation owns the pointers.
        this->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
    }

    pTopLevelModel->PropagateGlobalExecutionContext(NULL);
}

void daeFiniteElementEquation::Open(io::xmlTag_t* pTag)
{
    string strName;

    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);

    daeObject::Open(pTag);
}

void daeFiniteElementEquation::Save(io::xmlTag_t* pTag) const
{
    string strName, strValue;

    daeObject::Save(pTag);

    strName = "EquationType";
    SaveEnum(pTag, strName, GetEquationType());

    //strName = "Expression";
    //adNode::SaveNode(pTag, strName, m_pResidualNode.get());

    strName = "Residual";
    io::xmlTag_t* pChildTag = pTag->AddTag(strName);
    if(!pChildTag)
        daeDeclareAndThrowException(exXMLIOError);
    strValue = "$${ \\left [ \\mathbf{M_{ij}} \\right ] \\left \\{ \\frac{\\partial x_j}{\\partial t} \\right \\} } + { \\left [ \\mathbf{A_{ij}} \\right ] \\left \\{ {x_j} \\right \\} } = \\left \\{ f_i\\right \\}$$";
    pChildTag->SetValue(strValue);

    strName = "DistributedDomainInfos";
    pTag->SaveObjectArray(strName, m_ptrarrDistributedEquationDomainInfos);
}

void daeFiniteElementEquation::OpenRuntime(io::xmlTag_t* pTag)
{
}

void daeFiniteElementEquation::SaveRuntime(io::xmlTag_t* pTag) const
{
    string strName, strValue;

    daeObject::SaveRuntime(pTag);

//	strName = "EquationDefinitionMode";
//	SaveEnum(pTag, strName, m_eEquationType);

//	strName = "EquationEvaluationMode";
//	SaveEnum(pTag, strName, m_eEquationEvaluationMode);

    strName = "Residual";
    io::xmlTag_t* pChildTag = pTag->AddTag(strName);
    if(!pChildTag)
        daeDeclareAndThrowException(exXMLIOError);
    strValue = "$${ \\left [ \\mathbf{M_{ij}} \\right ] \\left \\{ \\frac{\\partial x_j}{\\partial t} \\right \\} } + { \\left [ \\mathbf{A_{ij}} \\right ] \\left \\{ {x_j} \\right \\} } = \\left \\{ f_i\\right \\}$$";
    pChildTag->SetValue(strValue);

//	strName = "DistributedDomainInfos";
//	pTag->SaveRuntimeObjectArray(strName, m_ptrarrDistributedEquationDomainInfos);

    strName = "EquationExecutionInfos";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrEquationExecutionInfos);
}

bool daeFiniteElementEquation::CheckObject(std::vector<string>& strarrErrors) const
{
    string strError;
    // Check base class
    if(!daeEquation::CheckObject(strarrErrors))
        return false;
    // Check residual node
    if(m_ptrarrDistributedEquationDomainInfos.size() != 1)
    {
        strError = "FiniteElement equation [" + GetCanonicalName() + "] must be distributed on exactly one unstructured grid domain";
        strarrErrors.push_back(strError);
        return false;
    }
    daeDomain* pDomain = m_ptrarrDistributedEquationDomainInfos[0]->m_pDomain;
    if(!pDomain)
    {
        strError = "Invalid domain in FiniteElement equation [" + GetCanonicalName() + "]";
        strarrErrors.push_back(strError);
        return false;
    }
    if(pDomain->GetType() != eUnstructuredGrid)
    {
        strError = "FiniteElement equation [" + GetCanonicalName() + "] must be distributed on an unstructured grid domain";
        strarrErrors.push_back(strError);
        return false;
    }
    return true;
}

daeDEDI* daeFiniteElementEquation::DistributeOnDomain(daeDomain& rDomain, daeeDomainBounds eDomainBounds, const string& strName)
{
    daeDeclareException(exInvalidCall);
    e << "daeFiniteElementEquation equations cannot be distributed using DistributeOnDomain functions "
      << "(it is done during the daeFiniteElementModel::CreateFiniteElementEquation call)";
    throw e;
    return NULL;
}

daeDEDI* daeFiniteElementEquation::DistributeOnDomain(daeDomain& rDomain, const std::vector<size_t>& narrDomainIndexes, const string& strName)
{
    daeDeclareException(exInvalidCall);
    e << "daeFiniteElementEquation equations cannot be distributed using DistributeOnDomain functions "
      << "(it is done during the daeFiniteElementModel::CreateFiniteElementEquation call)";
    throw e;
    return NULL;
}

daeDEDI* daeFiniteElementEquation::DistributeOnDomain(daeDomain& rDomain, const size_t* pnarrDomainIndexes, size_t n, const string& strName)
{
    daeDeclareException(exInvalidCall);
    e << "daeFiniteElementEquation equations cannot be distributed using DistributeOnDomain functions "
      << "(it is done during the daeFiniteElementModel::CreateFiniteElementEquation call)";
    throw e;
    return NULL;
}

}
}
