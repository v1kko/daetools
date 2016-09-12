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
      m_omega("&Omega;", this, unit(), "Omega")
{
    daeVariable* pVariable;

    if(!m_fe)
        daeDeclareAndThrowException(exInvalidPointer);

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

        /*
        if(feVarInfo.m_nMultiplicity == 1)
        {
            pVariable = new daeVariable(feVarInfo.m_strName, variable_types::no_t, this, feVarInfo.m_strDescription, &m_omega_c);
            m_ptrarrFEVariables.push_back(pVariable);
        }
        else
        {
            daeDomain* pd = new daeDomain("FED_" + toString(m_ptrarrFEDomains.size()), this, unit(), "FE Domain " + toString(m_ptrarrFEDomains.size()));
            pd->CreateArray(feVarInfo.m_nMultiplicity);
            m_ptrarrFEDomains.push_back(pd);

            pVariable = new daeVariable(feVarInfo.m_strName, variable_types::no_t, this, feVarInfo.m_strDescription, pd, &m_omega_c);
            m_ptrarrFEVariables.push_back(pVariable);
        }
        */
   }
}

void daeFiniteElementModel::DeclareEquations(void)
{
    daeModel::DeclareEquations();

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

struct feVariableInfo
{

};

void daeFiniteElementEquation::CreateEquationExecutionInfos(daeModel* pModel, std::vector<daeEquationExecutionInfo*>& ptrarrEqnExecutionInfosCreated, bool bAddToTheModel)
{
    daeVariable* variable;
    size_t indexes[1], counter;
    size_t nNoDomains, nIndex, column, internalVariableIndex;
    adouble a_Aij, a_Mij, a_Fi;
    daeFiniteElementObject* fe ;
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

        // Reset equation's contributions
        a_Aij = 0;
        a_Mij = 0;

        // RHS
        a_Fi = create_adouble(new adFEVectorItemNode("f", *m_FEModel.m_Fi, row, unit()));

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

            if(!a_Aij.node)
                a_Aij =         create_adouble(new adFEMatrixItemNode("A", *m_FEModel.m_Aij, row, column, unit())) * variable->Create_adouble(indexes, 1);
            else
                a_Aij = a_Aij + create_adouble(new adFEMatrixItemNode("A", *m_FEModel.m_Aij, row, column, unit())) * variable->Create_adouble(indexes, 1);

            /* ACHTUNG, ACHTUNG!!
               The mass matrix M is not going to change - wherever we have an item in M equal to zero it is going to stay zero
               (meaning that the FiniteElement object cannot suddenly sneak in differential variables into the system AFTER initialization).
               Therefore, skip an item if we encounter a zero.
            */
            if(m_FEModel.m_Mij->GetItem(row, column).node ||
               m_FEModel.m_Mij->GetItem(row, column).getValue() != 0)
            {
                if(!a_Mij.node)
                    a_Mij =         create_adouble(new adFEMatrixItemNode("M", *m_FEModel.m_Mij, row, column, unit())) * variable->Calculate_dt(indexes, 1);
                else
                    a_Mij = a_Mij + create_adouble(new adFEMatrixItemNode("M", *m_FEModel.m_Mij, row, column, unit())) * variable->Calculate_dt(indexes, 1);

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
