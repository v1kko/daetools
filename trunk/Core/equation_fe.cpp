#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"

namespace dae 
{
namespace core 
{
/******************************************************************
    daeFiniteElementEquationExecutionInfo
*******************************************************************/
daeFiniteElementEquationExecutionInfo::daeFiniteElementEquationExecutionInfo(daeEquation* pEquation)
                                     : daeEquationExecutionInfo(pEquation)
{
}

daeFiniteElementEquationExecutionInfo::~daeFiniteElementEquationExecutionInfo()
{
}

void daeFiniteElementEquationExecutionInfo::GatherInfo(daeExecutionContext& EC, daeModel* pModel)
{
}

void daeFiniteElementEquationExecutionInfo::SetEvaluationNode(adouble a)
{
    if(!a.node)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid node argument in daeFiniteElementEquationExecutionInfo::SetEvaluationNode function";
        throw e;
    }
    m_EquationEvaluationNode = a.node;
}

/******************************************************************
    daeFiniteElementEquation
*******************************************************************/
daeFiniteElementEquation::daeFiniteElementEquation(daeFiniteElementModel* fe)
{
    m_pFEModel = fe;
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

void daeFiniteElementEquation::Update()
{
    if(!m_pFEModel)
        daeDeclareException(exInvalidPointer);

    m_pFEModel->AssembleEquation(this);

    std::map<size_t, size_t> mapIndexes;
    daeEquationExecutionInfo* pEquationExecutionInfo;
    // We have to check the index map since the sparsity pattern may have been changed
    // during the AssembleEquation update of residual nodes which is not allowed.
    // ACHTUNG, ACHTUNG!! Here we DO NOT add fixed variables to the map
    for(size_t i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
    {
        pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
        mapIndexes.clear();
        pEquationExecutionInfo->m_EquationEvaluationNode->AddVariableIndexToArray(mapIndexes, false);

        if(mapIndexes.size() != pEquationExecutionInfo->m_mapIndexes.size() ||
           !std::equal(mapIndexes.begin(), mapIndexes.end(), pEquationExecutionInfo->m_mapIndexes.begin(), CompareKeys()))
        {
            daeDeclareException(exInvalidCall);
            e << "Sparsity patter has been changed in the last update of the finite element equation " << GetCanonicalName()
              << " in the row " << i << "; initial indexes: " << toString(pEquationExecutionInfo->m_mapIndexes) << " - new indexes: " << toString(mapIndexes);
            throw e;
        }
    }
}

/*
void daeFiniteElementEquation::UpdateEquation(daeSparseMatrix<real_t>& K, daeSparseMatrix<real_t>& Kdt, std::vector<real_t>& F)
{
    if(K.GetNrows() != F.size())
        daeDeclareException(exInvalidCall);

    bool bAddIndexes;
    size_t nOverallIndex, N, j, row, col;
    std::vector<size_t> rowIndexes;
    std::vector<size_t> narrDomainIndexes;
    daeEquationExecutionInfo* pEEI;
    adouble residual;

    std::cout << "0" << std::endl;
    N = K.GetNrows();

    narrDomainIndexes.resize(1);

    m_pResidualNode = adNodePtr(new adConstantNode(0));

    if(m_ptrarrEquationExecutionInfos.empty())
        m_ptrarrEquationExecutionInfos.resize(N, NULL);

    for(row = 0; row < N; row++)
    {
        std::cout << "1" << std::endl;
        if(m_ptrarrEquationExecutionInfos[row] == NULL)
        {
            bAddIndexes = true;
            pEEI = new daeFiniteElementEquationExecutionInfo(this);
            pEEI->m_dScaling = m_dScaling;
            pEEI->m_narrDomainIndexes.reserve(1);
            pEEI->m_narrDomainIndexes.push_back(row);
            m_ptrarrEquationExecutionInfos[row] = pEEI;
            std::cout << "21" << std::endl;
        }
        else
        {
            bAddIndexes = false;
            pEEI = m_ptrarrEquationExecutionInfos[row];
            std::cout << "22" << std::endl;
        }

        residual = 0;

        // 1. Add K[row,*] items
        K.RowNonzeros(row, rowIndexes);
        std::cout << "K(" << row << ") = " << toString(rowIndexes) << std::endl;
        for(j = 0; j < rowIndexes.size(); j++)
        {
            col = rowIndexes[j];
            std::cout << "col = " << col << std::endl;
            nOverallIndex = m_pVariable->GetOverallIndex() + col;
            std::cout << "nOverallIndex = " << nOverallIndex << std::endl;
            narrDomainIndexes[0] = col;

            adouble aij(0.0, 0.0, true, new adRuntimeVariableNode(m_pVariable, nOverallIndex, narrDomainIndexes));
            std::cout << "aij = " << aij << std::endl;
            residual = residual + aij * K.GetItem(row, col);
            std::cout << "residual = " << residual << std::endl;

            // Only add it once (if the bAddIndexes is true)
            if(bAddIndexes)
                pEEI->AddVariableInEquation(nOverallIndex);
            std::cout << "AddVariableInEquation = " << nOverallIndex << std::endl;
        }

        // 2. Add Kdt[row,*] items

//        if(K.GetNrows() > 0)
//        {
//            Kdt.RowNonzeros(row, rowIndexes);
//            std::cout << "Kdt(" << row << ") = " << toString(rowIndexes) << std::endl;
//            for(j = 0; j < rowIndexes.size(); j++)
//            {
//                col = rowIndexes[j];
//                nOverallIndex = m_pVariable->GetOverallIndex() + col;
//                narrDomainIndexes[0] = col;

//                adouble aij(0.0, 0.0, true, new adRuntimeTimeDerivativeNode(m_pVariable, nOverallIndex, 1, narrDomainIndexes));
//                residual = residual + aij * Kdt.GetItem(row, col);

//                // Only add it once (if the bAddIndexes is true)
//                if(bAddIndexes)
//                    pEEI->AddVariableInEquation(nOverallIndex);
//            }
//        }

        // 3. Add F[i] item
        std::cout << "F(" << row << ") = " << F[row] << std::endl;
        residual = residual - F[row];

        // Finally set the residual node
        pEEI->m_EquationEvaluationNode = residual.node;
    }
}
*/

void daeFiniteElementEquation::Initialize()
{
    daeEquationExecutionInfo* pEEI;

    if(m_ptrarrDistributedEquationDomainInfos.size() != 1)
        daeDeclareException(exInvalidCall);

    daeDomain* pDomain = m_ptrarrDistributedEquationDomainInfos[0]->m_pDomain;
    if(!pDomain)
        daeDeclareException(exInvalidPointer);

    size_t N = pDomain->GetNumberOfPoints();

    m_ptrarrEquationExecutionInfos.EmptyAndFreeMemory();
    m_ptrarrEquationExecutionInfos.reserve(N);

    m_pResidualNode = adNodePtr(new adConstantNode(0));

    for(size_t i = 0; i < N; i++)
    {
        pEEI = new daeFiniteElementEquationExecutionInfo(this);
        pEEI->m_dScaling = m_dScaling;
        pEEI->m_narrDomainIndexes.reserve(1);
        pEEI->m_narrDomainIndexes.push_back(i);
        pEEI->m_EquationEvaluationNode = adNodePtr(new adConstantNode(0));
        m_ptrarrEquationExecutionInfos.push_back(pEEI);
    }
}

void daeFiniteElementEquation::CreateEquationExecutionInfos(daeModel* pModel, vector<daeEquationExecutionInfo*>& ptrarrEqnExecutionInfosCreated, bool bAddToTheModel)
{
    size_t d1;
    size_t nNoDomains;
    daeEquationExecutionInfo* pEquationExecutionInfo;

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

    for(d1 = 0; d1 < m_ptrarrEquationExecutionInfos.size(); d1++)
    {
        pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[d1];

        // We have to build the map from scratch since the runtime node has not been formed by evaluation of the setup node.
        // ACHTUNG, ACHTUNG!! Here we DO NOT add fixed variables to the map
        pEquationExecutionInfo->m_mapIndexes.clear();
        pEquationExecutionInfo->m_EquationEvaluationNode->AddVariableIndexToArray(pEquationExecutionInfo->m_mapIndexes, false);

        if(bAddToTheModel)
            pModel->AddEquationExecutionInfo(pEquationExecutionInfo);
        else
            ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
    }
}

bool daeFiniteElementEquation::CheckObject(vector<string>& strarrErrors) const
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

/******************************************************************
    daeFiniteElementModel
*******************************************************************/
daeFiniteElementModel::daeFiniteElementModel(void)
{

}

daeFiniteElementModel::daeFiniteElementModel(string strName, daeModel* pModel, string strDescription)
                     : daeModel(strName, pModel, strDescription)
{

}

daeFiniteElementModel::~daeFiniteElementModel(void)
{

}

void daeFiniteElementModel::Open(io::xmlTag_t* pTag)
{
    daeModel::Open(pTag);
}

void daeFiniteElementModel::Save(io::xmlTag_t* pTag) const
{
    daeModel::Save(pTag);
}

void daeFiniteElementModel::OpenRuntime(io::xmlTag_t* pTag)
{
    daeModel::OpenRuntime(pTag);
}

void daeFiniteElementModel::SaveRuntime(io::xmlTag_t* pTag) const
{
    daeModel::SaveRuntime(pTag);
}

bool daeFiniteElementModel::CheckObject(std::vector<string>& strarrErrors) const
{
    return daeModel::CheckObject(strarrErrors);
}

void daeFiniteElementModel::GetVariableRuntimeNodes(daeVariable& variable, std::vector<adouble>& arrRuntimeNodes)
{
    if(variable.Domains().size() != 1)
    {
        daeDeclareException(exInvalidCall);
        e << "Runtime nodes can only be returned for variables distributed on a single unstructured grid domain (variable " << variable.GetCanonicalName() << ")";
        throw e;
    }

    daeDomain* pDomain = variable.Domains()[0];
    if(pDomain->GetType() != eUnstructuredGrid)
    {
        daeDeclareException(exInvalidCall);
        e << "Runtime nodes can only be returned for variables distributed on a single unstructured grid domain (variable " << variable.GetCanonicalName() << ")";
        throw e;
    }

    size_t nOverallIndex;
    std::vector<size_t> narrDomainIndexes;
    size_t N = pDomain->GetNumberOfPoints();
    arrRuntimeNodes.clear();
    arrRuntimeNodes.resize(N);
    narrDomainIndexes.resize(1);
    for(size_t i = 0; i < N; i++)
    {
        nOverallIndex        = variable.GetOverallIndex() + i;
        narrDomainIndexes[0] = i;

        // We must ensure that the gatherInfo flag is always true
        arrRuntimeNodes[i] = adouble(0.0, 0.0, true, new adRuntimeVariableNode(&variable, nOverallIndex, narrDomainIndexes));
    }
}

void daeFiniteElementModel::GetTimeDerivativeRuntimeNodes(daeVariable& variable, std::vector<adouble>& arrRuntimeNodes)
{
    if(variable.Domains().size() != 1)
    {
        daeDeclareException(exInvalidCall);
        e << "Runtime nodes can only be returned for variables distributed on a single unstructured grid domain (variable " << variable.GetCanonicalName() << ")";
        throw e;
    }

    daeDomain* pDomain = variable.Domains()[0];
    if(pDomain->GetType() != eUnstructuredGrid)
    {
        daeDeclareException(exInvalidCall);
        e << "Runtime nodes can only be returned for variables distributed on a single unstructured grid domain (variable " << variable.GetCanonicalName() << ")";
        throw e;
    }

    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidCall);

    size_t nOverallIndex;
    std::vector<size_t> narrDomainIndexes;
    size_t N = pDomain->GetNumberOfPoints();
    arrRuntimeNodes.clear();
    arrRuntimeNodes.resize(N);
    narrDomainIndexes.resize(1);
    for(size_t i = 0; i < N; i++)
    {
        nOverallIndex        = variable.GetOverallIndex() + i;
        narrDomainIndexes[0] = i;

        // We must ensure that the gatherInfo flag is always true
        arrRuntimeNodes[i] = adouble(0.0, 0.0, true, new adRuntimeTimeDerivativeNode(&variable, nOverallIndex, 1, narrDomainIndexes));

        // ACHTUNG,ACHTUNG!! The VariableTypeGathered flag must be set
        m_pDataProxy->SetVariableTypeGathered(nOverallIndex, cnDifferential);
    }
}

daeFiniteElementEquation* daeFiniteElementModel::CreateFiniteElementEquation(const string& strName, daeDomain* pDomain, string strDescription, real_t dScaling)
{
    string strEqName;

    if(!pDomain)
        daeDeclareAndThrowException(exInvalidPointer);

    if(pDomain->GetType() != eUnstructuredGrid)
    {
        daeDeclareException(exInvalidCall);
        e << "FiniteElement functions can only be distributed on unstructured grid domain (domain " << pDomain->GetCanonicalName()
          << " is " << dae::io::g_EnumTypesCollection->esmap_daeDomainType.GetString(pDomain->GetType()) << ") ";
        throw e;
    }

    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());

    daeFiniteElementEquation* pEquation = new daeFiniteElementEquation(this);

    if(!pCurrentState)
    {
        strEqName = (strName.empty() ? "Equation_" + toString<size_t>(m_ptrarrEquations.size()) : strName);
        pEquation->SetName(strEqName);
        AddEquation(pEquation);
    }
    else
    {
        strEqName = (strName.empty() ? "Equation_" + toString<size_t>(pCurrentState->m_ptrarrEquations.size()) : strName);
        pEquation->SetName(strEqName);
        pCurrentState->AddEquation(pEquation);
    }

    pEquation->SetDescription(strDescription);
    pEquation->SetScaling(dScaling);
    pEquation->daeEquation::DistributeOnDomain(*pDomain, eClosedClosed);
    pEquation->Initialize();

    return pEquation;
}

void daeFiniteElementModel::AssembleEquation(daeFiniteElementEquation* /*pEquation*/)
{
    daeDeclareAndThrowException(exNotImplemented);
}

}
}
