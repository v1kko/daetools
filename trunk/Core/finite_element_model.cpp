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
daeFiniteElementModel::daeFiniteElementModel(std::string strName, daeModel* pModel, std::string strDescription, daeFiniteElementObject* fe):
      daeModel(strName, pModel, strDescription),
      m_fe(fe),
      //m_dimension("dimension", this,               unit(), "Number of spatial dimensions"),
      m_omega    ("&Omega;",   this,                 unit(), "Omega domain"),
      m_T        ("T",         variable_types::no_t, this,   "Temperature", &m_omega)
{
    if(!m_fe)
        daeDeclareAndThrowException(exInvalidPointer);

    // Initialize daetools wrapper matrices and arrays that will be used by adFEMatrixItem/VectorItem nodes
    matK.reset  (m_fe->SystemMatrix());
    matKdt.reset(m_fe->SystemMatrix_dt());
    vecf.reset  (m_fe->SystemRHS());

    // Initialize domains and parameters
    size_t n = matK->GetNrows();
    std::vector<daePoint> coords(n);
    m_omega.CreateUnstructuredGrid(coords);
    std::cout << "n = " << n << std::endl;
    //m_dimension.CreateArray(dim);
}

void daeFiniteElementModel::DeclareEquations(void)
{
    daeModel::DeclareEquations();

    if(!m_fe)
        daeDeclareAndThrowException(exInvalidPointer);

    m_fe->AssembleSystem();
    daeFiniteElementEquation* eq = CreateFiniteElementEquation("FE", &m_omega, "", 1.0);
    eq->SetResidual(Constant(0.0));
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
    daeFiniteElementEquation* pEquation = new daeFiniteElementEquation(*this, m_T, 0, m_omega.GetNumberOfPoints());

    if(!pCurrentState)
    {
        strEqName = (strName.empty() ? "FiniteEquation_" + toString<size_t>(m_ptrarrEquations.size()) : strName);
        pEquation->SetName(strEqName);
        AddEquation(pEquation);
    }
    else
    {
        strEqName = (strName.empty() ? "FiniteEquation_" + toString<size_t>(pCurrentState->m_ptrarrEquations.size()) : strName);
        pEquation->SetName(strEqName);
        pCurrentState->AddEquation(pEquation);
    }

    pEquation->SetDescription(strDescription);
    pEquation->SetScaling(dScaling);
    pEquation->daeEquation::DistributeOnDomain(*pDomain, eClosedClosed);
    return pEquation;
}

/******************************************************************
    daeFiniteElementEquation
*******************************************************************/
daeFiniteElementEquation::daeFiniteElementEquation(const daeFiniteElementModel& feModel, const daeVariable& variable, size_t startRow, size_t endRow):
    m_FEModel(feModel),
    m_Variable(variable),
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
    size_t nNoDomains, nIndex;
    daeEquationExecutionInfo* pEquationExecutionInfo;
    adouble a_K, a_Kdt, a_f;

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
    std::cout << "FEEquation NoEqns = " << NoEqns << std::endl;
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

    size_t indexes[1];
    std::cout << "FEEquation Nrows = " << m_endRow - m_startRow << std::endl;
    daeFiniteElementObject* fe = m_FEModel.m_fe;
    if(!fe)
        daeDeclareAndThrowException(exInvalidPointer);

    for(size_t row = m_startRow; row < m_endRow; row++)
    {
        pEquationExecutionInfo = new daeEquationExecutionInfo(this);
        pEquationExecutionInfo->m_dScaling = this->m_dScaling;

        pEquationExecutionInfo->m_narrDomainIndexes.resize(1);
        pEquationExecutionInfo->m_narrDomainIndexes[0] = row;

        // Reset equation's contributions
        a_K   = 0;
        a_Kdt = 0;

        // RHS
        a_f = create_adouble(new adFEVectorItemNode("f", *m_FEModel.vecf, row, unit()));

        // begin_row() and end_row() return pointers created with the new operator
        boost::scoped_ptr<daeSparseMatrixRowIterator> iter(fe->RowIterator(row));
        for(iter->first(); !iter->isDone(); iter->next())
        {
            const size_t col = iter->currentItem();
            indexes[0] = col;

            if(!a_K.node)
                a_K =       create_adouble(new adFEMatrixItemNode("K", *m_FEModel.matK, row, col, unit())) * m_Variable.Create_adouble(indexes, 1);
            else
                a_K = a_K + create_adouble(new adFEMatrixItemNode("K", *m_FEModel.matK, row, col, unit())) * m_Variable.Create_adouble(indexes, 1);

            /* ACHTUNG, ACHTUNG!!
               The matrix Kdt is not going to change - wherever we have a matrix_dt item equal to zero it is going to stay zero
               (meaning that the FiniteElement object cannot suddenly sneak in differential variables to the system AFTER creation).
               Therefore, skip an item if we encounter a zero.
            */
            if(m_FEModel.matKdt->GetItem(row, col) != 0)
            {
                if(!a_Kdt.node)
                    a_Kdt =         create_adouble(new adFEMatrixItemNode("", *m_FEModel.matKdt, row, col, unit())) * m_Variable.Calculate_dt(indexes, 1);
                else
                    a_Kdt = a_Kdt + create_adouble(new adFEMatrixItemNode("", *m_FEModel.matKdt, row, col, unit())) * m_Variable.Calculate_dt(indexes, 1);

                nIndex = m_Variable.GetOverallIndex() + col;
                m_pModel->m_pDataProxy->SetVariableTypeGathered(nIndex, cnDifferential);
            }
        }

        pEquationExecutionInfo->m_EquationEvaluationNode = (a_Kdt + a_K - a_f).node;

        /* ACHTUNG, ACHTUNG!!
           We already set m_EquationEvaluationNode and the call to GatherInfo() seems unnecesary.
           This way we avoided creation of setup nodes first and then evaluating them into the runtime ones
           during the GatherInfo() function. Are there some side-effects of this vodoo-mojo?
           Update: there are side-effects
              mapIndexes is not populated with the variable indexes during the GatherInfo call!
              We have to build the map from scratch since the runtime node has not been formed by evaluation of the setup node.
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
