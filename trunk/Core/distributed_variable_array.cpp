#include "stdafx.h"
#include "coreimpl.h"
#include "nodes_array.h"
using namespace boost;

namespace dae
{
namespace core
{

daeExecutionContext::daeExecutionContext()
{
    m_pDataProxy										= NULL;
    m_pBlock											= NULL;
    m_dInverseTimeStep									= 0;
    m_pEquationExecutionInfo							= NULL;
    m_eEquationCalculationMode							= eECMUnknown;
    m_nCurrentVariableIndexForJacobianEvaluation		= ULONG_MAX;
    m_nCurrentParameterIndexForSensitivityEvaluation	= ULONG_MAX;
    m_nIndexInTheArrayOfCurrentParameterForSensitivityEvaluation = ULONG_MAX;
}

size_t daeVariable::GetNumberOfPoints() const
{
    vector<daeDomain*>::size_type i;
    daeDomain* pDomain;
    size_t nTotalNumberOfVariables = 1;
    for(i = 0; i < m_ptrDomains.size(); i++)
    {
        pDomain = m_ptrDomains[i];
        if(!pDomain)
            daeDeclareAndThrowException(exInvalidPointer);
        if(pDomain->m_nNumberOfPoints == 0)
        {
            daeDeclareException(exInvalidCall);
            e << "Number of points in domain [" << pDomain->GetCanonicalName() << "] in variable [" << GetCanonicalName() << "] is zero; did you forget to initialize it?";
            throw e;
        }
        nTotalNumberOfVariables *= pDomain->m_nNumberOfPoints;
    }
    return nTotalNumberOfVariables;
}

size_t daeVariable::CalculateIndex(const std::vector<size_t>& narrDomainIndexes) const
{
    size_t* indexes;
    size_t i, N, index;

    N = narrDomainIndexes.size();
    if(N == 0)
    {
        index = CalculateIndex(NULL, 0);
    }
    else
    {
        index = CalculateIndex(&narrDomainIndexes[0], N);
    }
    return index;
}

void daeVariable::InitializeBlockIndexes(const std::map<size_t, size_t>& mapOverallIndex_BlockIndex)
{
    std::map<size_t, size_t>::const_iterator iter;
    size_t noPoints = GetNumberOfPoints();
    size_t startOI = m_nOverallIndex;
    size_t endOI   = m_nOverallIndex + noPoints;

    m_narrBlockIndexes.resize(noPoints);
    for(size_t i = 0, oi = startOI; oi < endOI; i++, oi++)
    {
        iter = mapOverallIndex_BlockIndex.find(oi);
        if(iter != mapOverallIndex_BlockIndex.end()) // if found
            m_narrBlockIndexes[i] = iter->second;
        else
            m_narrBlockIndexes[i] = ULONG_MAX;
    }
}

size_t daeVariable::GetOverallIndex(void) const
{
    return m_nOverallIndex;
}

const std::vector<size_t>& daeVariable::GetBlockIndexes() const
{
    return m_narrBlockIndexes;
}


int daeVariable::GetType() const
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);

    boost::shared_ptr<daeDataProxy_t> pDataProxy = m_pModel->GetDataProxy();
    if(!pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    if(GetNumberOfPoints() == 1)
    {
        return pDataProxy->GetVariableType(GetOverallIndex());
    }
    else
    {
        size_t nStart = GetOverallIndex();
        size_t nEnd   = GetOverallIndex() + GetNumberOfPoints();

        bool foundAlgebraic     = false;
        bool foundDifferrential = false;
        bool foundAssigned      = false;
        for(size_t k = nStart; k < nEnd; k++)
        {
            int varType = pDataProxy->GetVariableType(k);

            if(varType == cnAlgebraic)
                foundAlgebraic = true;
            else if(varType == cnAssigned)
                foundAssigned = true;
            else if(varType == cnDifferential)
                foundDifferrential = true;
            else
                daeDeclareAndThrowException(exInvalidCall);
        }

        if(foundAlgebraic && !foundDifferrential && !foundAssigned)
            return cnAlgebraic;
        else if(!foundAlgebraic && foundDifferrential && !foundAssigned)
            return cnDifferential;
        else if(!foundAlgebraic && !foundDifferrential && foundAssigned)
            return cnAssigned;

        else if(foundAlgebraic && foundDifferrential && !foundAssigned)
            return cnSomePointsDifferential;
        else if(foundAlgebraic && !foundDifferrential && foundAssigned)
            return cnSomePointsAssigned;
        else
            return cnMixedAlgebraicAssignedDifferential;
    }
}

size_t daeVariable::CalculateIndex(const size_t* indexes, const size_t N) const
{
    size_t		i, j, nIndex, temp;
    daeDomain	*pDomain;

    if(m_ptrDomains.size() != N)
    {
        daeDeclareException(exInvalidCall);
        e << "Illegal number of domains (" << N << ") in variable " << GetCanonicalName() << " (must be " << m_ptrDomains.size() << ")";
        throw e;
    }

// Check the pointers and the bounds first
    for(i = 0; i < N; i++)
    {
        pDomain = m_ptrDomains[i];
        if(!pDomain)
            daeDeclareAndThrowException(exInvalidPointer);
        if(indexes[i] >= pDomain->GetNumberOfPoints())
        {
            daeDeclareException(exOutOfBounds);
            e << "Invalid index in variable " << GetCanonicalName() << "; index = " << indexes[i]
              << " while number of points in domain " << pDomain->GetCanonicalName() << " is " << pDomain->GetNumberOfPoints();
            throw e;
        }
    }

// Calculate the index
    nIndex = 0;
    for(i = 0; i < N; i++)
    {
        temp = indexes[i];
        for(j = i+1; j < N; j++)
            temp *= m_ptrDomains[j]->GetNumberOfPoints();
        nIndex += temp;
    }

    return nIndex;
}

void daeVariable::Fill_adouble_array(vector<adouble>& arrValues, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const
{
    if(currentN == N) // create and add adouble to the vector
    {
        dae_push_back(arrValues, Create_adouble(indexes, N));
    }
    else // continue iterating
    {
        const daeArrayRange& r = ranges[currentN];

        vector<size_t> narrPoints;
    // If the size is 1 it is going to call Fill_adouble_array() only once
    // Otherwise, narrPoints.size() times
        r.GetPoints(narrPoints);
        for(size_t i = 0; i < narrPoints.size(); i++)
        {
            indexes[currentN] = narrPoints[i]; // Wasn't it bug below??? It should be narrPoints[i]!!
            Fill_adouble_array(arrValues, ranges, indexes, N, currentN + 1);
        }

//		if(r.m_eType == eRangeConstantIndex)
//		{
//			indexes[currentN] = r.m_nIndex;
//			Fill_adouble_array(arrValues, ranges, indexes, N, currentN + 1);
//		}
//		else if(r.m_eType == eRangeDomainIterator)
//		{
//			indexes[currentN] = r.m_pDEDI->GetCurrentIndex();
//			Fill_adouble_array(arrValues, ranges, indexes, N, currentN + 1);
//		}
//		else if(r.m_eType == eRange)
//		{
//			vector<size_t> narrPoints;
//			r.m_Range.GetPoints(narrPoints);
//			for(size_t i = 0; i < narrPoints.size(); i++)
//			{
//				indexes[currentN] = i; // BUG !!!!!!!! Shouldn't it be narrPoints[i]???
//				Fill_adouble_array(arrValues, ranges, indexes, N, currentN + 1);
//			}
//		}
    }
}

void daeVariable::Fill_dt_array(vector<adouble>& arrValues, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const
{
    if(currentN == N) // create and add adouble to the vector
    {
        dae_push_back(arrValues, Calculate_dt(indexes, N));
    }
    else // continue iterating
    {
        const daeArrayRange& r = ranges[currentN];

        vector<size_t> narrPoints;
    // If the size is 1 it is going to call Fill_dt_array() only once
    // Otherwise, narrPoints.size() times
        r.GetPoints(narrPoints);
        for(size_t i = 0; i < narrPoints.size(); i++)
        {
            indexes[currentN] = narrPoints[i]; // Wasn't it bug below??? It should be narrPoints[i] !!
            Fill_dt_array(arrValues, ranges, indexes, N, currentN + 1);
        }

//		if(r.m_eType == eRangeConstantIndex)
//		{
//			indexes[currentN] = r.m_nIndex;
//			Fill_dt_array(arrValues, ranges, indexes, N, currentN + 1);
//		}
//		else if(r.m_eType == eRangeDomainIterator)
//		{
//			indexes[currentN] = r.m_pDEDI->GetCurrentIndex();
//			Fill_dt_array(arrValues, ranges, indexes, N, currentN + 1);
//		}
//		else if(r.m_eType == eRange)
//		{
//			vector<size_t> narrPoints;
//			r.m_Range.GetPoints(narrPoints);
//			for(size_t i = 0; i < narrPoints.size(); i++)
//			{
//				indexes[currentN] = i; // BUG !!!!!!!! Shouldn't it be narrPoints[i]???
//				Fill_dt_array(arrValues, ranges, indexes, N, currentN + 1);
//			}
//		}
    }
}

void daeVariable::Fill_partial_array(vector<adouble>& arrValues,
                                     size_t nOrder,
                                     const daeDomain_t& rDomain,
                                     const daeArrayRange* ranges, size_t* indexes,
                                     const size_t N, size_t currentN,
                                     daeeDiscretizationMethod  eDiscretizationMethod,
                                     const std::map<std::string, std::string>& mapDiscretizationOptions) const
{
    if(currentN == N) // create and add adouble to the vector
    {
        dae_push_back(arrValues, partial(nOrder, rDomain, indexes, N, eDiscretizationMethod, mapDiscretizationOptions));
    }
    else // continue iterating
    {
        const daeArrayRange& r = ranges[currentN];

        vector<size_t> narrPoints;
    // If the size is 1 it is going to call Fill_partial_array() only once
    // Otherwise, narrPoints.size() times
        r.GetPoints(narrPoints);
        for(size_t i = 0; i < narrPoints.size(); i++)
        {
            indexes[currentN] = narrPoints[i]; // Wasn't it bug below??? It should be narrPoints[i] !!
            Fill_partial_array(arrValues, nOrder, rDomain, ranges, indexes, N, currentN + 1, eDiscretizationMethod, mapDiscretizationOptions);
        }

//		if(r.m_eType == eRangeConstantIndex)
//		{
//			indexes[currentN] = r.m_nIndex;
//			Fill_partial_array(arrValues, nOrder, rDomain, ranges, indexes, N, currentN + 1);
//		}
//		else if(r.m_eType == eRangeDomainIterator)
//		{
//			indexes[currentN] = r.m_pDEDI->GetCurrentIndex();
//			Fill_partial_array(arrValues, nOrder, rDomain, ranges, indexes, N, currentN + 1);
//		}
//		else if(r.m_eType == eRange)
//		{
//			vector<size_t> narrPoints;
//			r.m_Range.GetPoints(narrPoints);
//			for(size_t i = 0; i < narrPoints.size(); i++)
//			{
//				indexes[currentN] = i; // BUG !!!!!!!! Shouln't it be narrPoints[i]???
//				Fill_partial_array(arrValues, nOrder, rDomain, ranges, indexes, N, currentN + 1);
//			}
//		}
    }
}

adouble_array daeVariable::Create_adouble_array(const daeArrayRange* ranges, const size_t N) const
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

// First create all adoubles (according to the ranges sent)
// The result is array of values, and if GetGatherMode flag is set
// also the adNode* in each adouble
    adouble_array varArray;
    size_t* indexes = new size_t[N];
    Fill_adouble_array(varArray.m_arrValues, ranges, indexes, N, 0);
    delete[] indexes;

// Now I should create adNodeArray* node
// I rely here on adNode* in each of adouble in varArray.m_arrValues
// created above
//	if(m_pModel->m_pDataProxy->GetGatherInfo())
//	{
//		adRuntimeVariableNodeArray* node = new adRuntimeVariableNodeArray();
//		varArray.node = adNodeArrayPtr(node);
//		varArray.setGatherInfo(true);
//		node->m_pVariable = const_cast<daeVariable*>(this);
//
//		size_t size = varArray.m_arrValues.size();
//		if(size == 0)
//			daeDeclareAndThrowException(exInvalidCall);
//
//		node->m_ptrarrVariableNodes.resize(size);
//		for(size_t i = 0; i < size; i++)
//			node->m_ptrarrVariableNodes[i] = varArray.m_arrValues[i].node;
//		node->m_arrRanges.resize(N);
//		for(size_t i = 0; i < N; i++)
//			node->m_arrRanges[i] = ranges[i];
//	}

    return varArray;
}

adouble_array daeVariable::Calculate_dt_array(const daeArrayRange* ranges, const size_t N) const
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    adouble_array varArray;
    size_t* indexes = new size_t[N];
    Fill_dt_array(varArray.m_arrValues, ranges, indexes, N, 0);
    delete[] indexes;

//	if(m_pModel->m_pDataProxy->GetGatherInfo())
//	{
//		adRuntimeTimeDerivativeNodeArray* node = new adRuntimeTimeDerivativeNodeArray();
//		varArray.node = adNodeArrayPtr(node);
//		varArray.setGatherInfo(true);
//		node->m_pVariable = const_cast<daeVariable*>(this);
//		node->m_nOrder   = 1;
//
//		size_t size = varArray.m_arrValues.size();
//		if(size == 0)
//			daeDeclareAndThrowException(exInvalidCall);
//
//		node->m_ptrarrTimeDerivativeNodes.resize(size);
//		for(size_t i = 0; i < size; i++)
//			node->m_ptrarrTimeDerivativeNodes[i] = varArray.m_arrValues[i].node;
//		node->m_arrRanges.resize(N);
//		for(size_t i = 0; i < N; i++)
//			node->m_arrRanges[i] = ranges[i];
//	}

    return varArray;
}

adouble_array daeVariable::partial_array(const size_t nOrder,
                                         const daeDomain_t& rDomain,
                                         const daeArrayRange* ranges,
                                         const size_t N,
                                         const daeeDiscretizationMethod  eDiscretizationMethod,
                                         const std::map<std::string, std::string>& mapDiscretizationOptions) const
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    adouble_array varArray;
    size_t* indexes = new size_t[N];
    Fill_partial_array(varArray.m_arrValues, nOrder, rDomain, ranges, indexes, N, 0, eDiscretizationMethod, mapDiscretizationOptions);
    delete[] indexes;

//	if(m_pModel->m_pDataProxy->GetGatherInfo())
//	{
//		adRuntimePartialDerivativeNodeArray* node = new adRuntimePartialDerivativeNodeArray();
//		varArray.node = adNodeArrayPtr(node);
//		varArray.setGatherInfo(true);
//		node->m_pVariable = const_cast<daeVariable*>(this);
//		const daeDomain* pDomain = dynamic_cast<const daeDomain*>(&rDomain);
//		node->m_pDomain = const_cast<daeDomain*>(pDomain);
//		node->m_nOrder = nOrder;
//
//		size_t size = varArray.m_arrValues.size();
//		if(size == 0)
//			daeDeclareAndThrowException(exInvalidCall);
//
//		node->m_ptrarrPartialDerivativeNodes.resize(size);
//		for(size_t i = 0; i < size; i++)
//			node->m_ptrarrPartialDerivativeNodes[i] = varArray.m_arrValues[i].node;
//		node->m_arrRanges.resize(N);
//		for(size_t i = 0; i < N; i++)
//			node->m_arrRanges[i] = ranges[i];
//	}

    return varArray;
}

adouble_array daeVariable::CreateSetupVariableArray(const daeArrayRange* ranges, const size_t N) const
{
    adouble_array varArray;

    if(m_ptrDomains.size() != N)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid variable array call for [" << GetCanonicalName() << "], number of domains is " << m_ptrDomains.size() << " - but only " << N << " is given";
        throw e;
    }
    if(!m_pModel)
    {
        daeDeclareException(exInvalidPointer);
        e << "Invalid parent model in variable [" << GetCanonicalName() << "]";
        throw e;
    }
// Check if domains in indexes correspond to domains here
    for(size_t i = 0; i < N; i++)
    {
        if(ranges[i].m_eType == eRangeDomainIndex)
        {
            if(ranges[i].m_domainIndex.m_eType == eDomainIterator ||
               ranges[i].m_domainIndex.m_eType == eIncrementedDomainIterator)
            {
                if(m_ptrDomains[i] != ranges[i].m_domainIndex.m_pDEDI->m_pDomain)
                {
                    // If it is not the same domain check the number of points
                    // It is acceptable to create a domain iterator on a domain 'x' and iterate over
                    // some other variable which is distributed over another domain but with
                    // the same number of points!
                    if(m_ptrDomains[i]->GetNumberOfPoints() != ranges[i].m_domainIndex.m_pDEDI->m_pDomain->GetNumberOfPoints())
                    {
                        if(m_pModel->m_pDataProxy)
                        {
                            string f = "Warning: You should not call the array() function with the domain iterator created on the domain [%s]; "
                                       " use the domain [%s] instead "
                                       "(or a domain with the same number of points) as %d. index argument in variable [%s]";
                            string msg = (boost::format(f) % ranges[i].m_domainIndex.m_pDEDI->m_pDomain->GetCanonicalName() %
                                                             m_ptrDomains[i]->GetCanonicalName() %
                                                             (i+1) %
                                                             GetCanonicalName()).str();
                            m_pModel->m_pDataProxy->LogMessage(msg, 0);
                        }
                    }
                }
            }
        }
        else if(ranges[i].m_eType == eRange)
        {
            if(m_ptrDomains[i] != ranges[i].m_Range.m_pDomain)
            {
                daeDeclareException(exInvalidCall);
                e << "You cannot create daeArrayRange with the domain [" << ranges[i].m_Range.m_pDomain->GetCanonicalName()
                  << "]; you must use the domain [" << m_ptrDomains[i]->GetCanonicalName() << "] as " << i+1 << ". range argument "
                  << "in variable [" << GetCanonicalName() << "] in function array()";
                throw e;
            }
        }
    }

    adSetupVariableNodeArray* node = new adSetupVariableNodeArray();
    varArray.node = adNodeArrayPtr(node);
    varArray.setGatherInfo(true);

    node->m_pVariable = const_cast<daeVariable*>(this);
    node->m_arrRanges.resize(N);
    for(size_t i = 0; i < N; i++)
        node->m_arrRanges[i] = ranges[i];

    return varArray;
}

adouble_array daeVariable::CreateSetupTimeDerivativeArray(const daeArrayRange* ranges, const size_t N) const
{
    adouble_array varArray;

    if(m_ptrDomains.size() != N)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid time derivative array call for [" << GetCanonicalName() << "], number of domains is " << m_ptrDomains.size() << " - but only " << N << " is given";
        throw e;
    }
    if(!m_pModel)
    {
        daeDeclareException(exInvalidPointer);
        e << "Invalid parent model in variable [" << GetCanonicalName() << "]";
        throw e;
    }
// Check if domains in indexes correspond to domains here
    for(size_t i = 0; i < N; i++)
    {
        if(ranges[i].m_eType == eRangeDomainIndex)
        {
            if(ranges[i].m_domainIndex.m_eType == eDomainIterator ||
               ranges[i].m_domainIndex.m_eType == eIncrementedDomainIterator)
            {
                if(m_ptrDomains[i] != ranges[i].m_domainIndex.m_pDEDI->m_pDomain)
                {
                    // If it is not the same domain check the number of points
                    // It is acceptable to create a domain iterator on a domain 'x' and iterate over
                    // some other variable which is distributed over another domain but with
                    // the same number of points!
                    if(m_ptrDomains[i]->GetNumberOfPoints() != ranges[i].m_domainIndex.m_pDEDI->m_pDomain->GetNumberOfPoints())
                    {
                        if(m_pModel->m_pDataProxy)
                        {
                            string f = "Warning: You should not call the dt_array() function with the domain iterator created on the domain [%s]; "
                                       " use the domain [%s] instead "
                                       "(or a domain with the same number of points) as %d. index argument in variable [%s]";
                            string msg = (boost::format(f) % ranges[i].m_domainIndex.m_pDEDI->m_pDomain->GetCanonicalName() %
                                                             m_ptrDomains[i]->GetCanonicalName() %
                                                             (i+1) %
                                                             GetCanonicalName()).str();
                            m_pModel->m_pDataProxy->LogMessage(msg, 0);
                        }
                    }
                }
            }
        }
        else if(ranges[i].m_eType == eRange)
        {
            if(m_ptrDomains[i] != ranges[i].m_Range.m_pDomain)
            {
                daeDeclareException(exInvalidCall);
                e << "You cannot create daeArrayRange with the domain [" << ranges[i].m_Range.m_pDomain->GetCanonicalName()
                  << "]; you must use the domain [" << m_ptrDomains[i]->GetCanonicalName() << "] as " << i+1 << ". range argument "
                  << "in variable [" << GetCanonicalName() << "] in function dt_array()";
                throw e;
            }
        }
    }

    adSetupTimeDerivativeNodeArray* node = new adSetupTimeDerivativeNodeArray();
    varArray.node = adNodeArrayPtr(node);
    varArray.setGatherInfo(true);

    node->m_nOrder = 1;
    node->m_pVariable = const_cast<daeVariable*>(this);
    node->m_arrRanges.resize(N);
    for(size_t i = 0; i < N; i++)
        node->m_arrRanges[i] = ranges[i];

    return varArray;
}

adouble_array daeVariable::CreateSetupPartialDerivativeArray(const size_t nOrder,
                                                             const daeDomain_t& rDomain,
                                                             const daeArrayRange* ranges,
                                                             const size_t N,
                                                             const daeeDiscretizationMethod  eDiscretizationMethod,
                                                             const std::map<std::string, std::string>& mapDiscretizationOptions) const
{
    adouble_array varArray;

    if(m_ptrDomains.size() != N)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid partial derivative array call for [" << GetCanonicalName() << "], number of domains is " << m_ptrDomains.size() << " - but only " << N << " is given";
        throw e;
    }
    if(!m_pModel)
    {
        daeDeclareException(exInvalidPointer);
        e << "Invalid parent model in variable [" << GetCanonicalName() << "]";
        throw e;
    }
// Check if domains in indexes correspond to domains here
    for(size_t i = 0; i < N; i++)
    {
        if(ranges[i].m_eType == eRangeDomainIndex)
        {
            if(ranges[i].m_domainIndex.m_eType == eDomainIterator ||
               ranges[i].m_domainIndex.m_eType == eIncrementedDomainIterator)
            {
                if(m_ptrDomains[i] != ranges[i].m_domainIndex.m_pDEDI->m_pDomain)
                {
                    // If it is not the same domain check the number of points
                    // It is acceptable to create a domain iterator on a domain 'x' and iterate over
                    // some other variable which is distributed over another domain but with
                    // the same number of points!
                    if(m_ptrDomains[i]->GetNumberOfPoints() != ranges[i].m_domainIndex.m_pDEDI->m_pDomain->GetNumberOfPoints())
                    {
                        if(m_pModel->m_pDataProxy)
                        {
                            string f = "Warning: You should not call the d_array() or d2_array() functions with the domain iterator created on the domain [%s]; "
                                       " use the domain [%s] instead "
                                       "(or a domain with the same number of points) as %d. index argument in variable [%s]";
                            string msg = (boost::format(f) % ranges[i].m_domainIndex.m_pDEDI->m_pDomain->GetCanonicalName() %
                                                             m_ptrDomains[i]->GetCanonicalName() %
                                                             (i+1) %
                                                             GetCanonicalName()).str();
                            m_pModel->m_pDataProxy->LogMessage(msg, 0);
                        }
                    }
                }
            }
        }
        else if(ranges[i].m_eType == eRange)
        {
            if(m_ptrDomains[i] != ranges[i].m_Range.m_pDomain)
            {
                daeDeclareException(exInvalidCall);
                e << "You cannot create daeArrayRange with the domain [" << ranges[i].m_Range.m_pDomain->GetCanonicalName()
                  << "]; you must use the domain [" << m_ptrDomains[i]->GetCanonicalName() << "] as " << i+1 << ". range argument "
                  << "in variable [" << GetCanonicalName() << "] in function d/d2_array()";
                throw e;
            }
        }
    }

    adSetupPartialDerivativeNodeArray* node = new adSetupPartialDerivativeNodeArray();
    varArray.node = adNodeArrayPtr(node);
    varArray.setGatherInfo(true);

    node->m_nOrder = nOrder;
    node->m_pVariable = const_cast<daeVariable*>(this);
    const daeDomain* pDomain = dynamic_cast<const daeDomain*>(&rDomain);
    node->m_pDomain = const_cast<daeDomain*>(pDomain);
    node->m_arrRanges.resize(N);
    for(size_t i = 0; i < N; i++)
        node->m_arrRanges[i] = ranges[i];
    node->m_eDiscretizationMethod = eDiscretizationMethod;
    node->m_mapDiscretizationOptions = mapDiscretizationOptions;

    return varArray;
}

adouble daeVariable::Create_adouble(const size_t* indexes, const size_t N) const
{
    adouble tmp;
    size_t nIndex, nIndexWithinVariable;
    daeExecutionContext* pExecutionContext;

    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    nIndexWithinVariable = CalculateIndex(indexes, N);
    nIndex = m_nOverallIndex + nIndexWithinVariable;

/**************************************************************************************
  New code; just we need is to add variable indexes to the equation
**************************************************************************************/
    if(m_pModel->m_pDataProxy->GetVariableType(nIndex) != cnAssigned)
    {
        if(m_pModel->m_pExecutionContextForGatherInfo)
        {
            pExecutionContext = m_pModel->m_pExecutionContextForGatherInfo;
            if(!pExecutionContext)
                daeDeclareAndThrowException(exInvalidPointer);

            daeEquationExecutionInfo* pEquationExecutionInfo = pExecutionContext->m_pEquationExecutionInfo;
        // StateTransitions CreateRuntimeNode call ends up here
        // In that case we dont have pEquationExecutionInfo nor we need it
            if(pEquationExecutionInfo && pExecutionContext->m_eEquationCalculationMode == eGatherInfo)
            {
                pEquationExecutionInfo->AddVariableInEquation(nIndex);
            }
        }
    }

/**************************************************************************************
  Old code; all this seems non necessary (we dont need values in this function)
  Just we need is to add variable indexes to the equation
***************************************************************************************
    if(m_pModel->m_pDataProxy->GetVariableType(nIndex) == cnAssigned)
    {
        tmp.setValue(GetValueAt(nIndex));
        tmp.setDerivative(0); // Since it is assigned value
    }
    else
    {
        // If it is NULL then we have already initialized
        if(m_pModel->m_pExecutionContextForGatherInfo)
        {
            pExecutionContext = m_pModel->m_pExecutionContextForGatherInfo;
        }
        else
        {
            pExecutionContext = m_pModel->m_pDataProxy->GetExecutionContext(nIndex);
        }
        if(!pExecutionContext)
        {
            daeDeclareException(exInvalidPointer);
            e << "Cannot find ExecutionContext for variable: " << GetCanonicalName() << ", index: " << nIndex;
            throw e;
        }

        if(pExecutionContext->m_eEquationCalculationMode == eCalculate)
        {
            tmp.setValue(GetValueAt(nIndex));
            tmp.setDerivative(0); // No need for it
        }
        else if(pExecutionContext->m_eEquationCalculationMode == eCalculateJacobian)
        {
            tmp.setValue(GetValueAt(nIndex));
            tmp.setDerivative(GetADValueAt(nIndex));
        }
        else if(pExecutionContext->m_eEquationCalculationMode == eGatherInfo)
        {
        // Here we dont need any calculation
            //tmp.setValue(0);
            //tmp.setDerivative(0);

            daeEquationExecutionInfo* pEquationExecutionInfo = pExecutionContext->m_pEquationExecutionInfo;
            if(!pEquationExecutionInfo)
                daeDeclareAndThrowException(exInvalidPointer);
            pEquationExecutionInfo->AddVariableInEquation(nIndex);
        }
        else if(pExecutionContext->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs)
        {
        // Here we dont need any calculation
            //tmp.setValue(0);
            //tmp.setDerivative(0);
        }
        else if(pExecutionContext->m_eEquationCalculationMode == eCalculateSensitivities)
        {
            daeDeclareAndThrowException(exInvalidCall)
        }
        else if(pExecutionContext->m_eEquationCalculationMode == eCalculateGradients)
        {
            daeDeclareAndThrowException(exInvalidCall)
        }
        else
        {
            daeDeclareException(exMiscellanous);
            e << "Unknown function evaluation mode";
            throw e;
        }
    }
*/

    if(m_pModel->m_pDataProxy->GetGatherInfo())
    {
        std::map<size_t, adouble>::const_iterator it = m_pModel->m_pDataProxy->m_mapRuntimeVariableNodes.find(nIndex);
        if(it != m_pModel->m_pDataProxy->m_mapRuntimeVariableNodes.end()) // if found
        {
            //std::cout << "Found runtime variable " << GetName() << "[" << nIndex << "] in the map" << std::endl;
            return it->second;
        }

        adRuntimeVariableNode* node = new adRuntimeVariableNode();
        node->m_pVariable     = const_cast<daeVariable*>(this);
        node->m_nOverallIndex = nIndex;
        if(N > 0)
        {
            node->m_narrDomains.resize(N);
            for(size_t i = 0; i < N; i++)
                node->m_narrDomains[i] = indexes[i];
        }
        tmp.node = adNodePtr(node);
        tmp.setGatherInfo(true);

        // Add it to the map for it has not been added yet
        //daeVariable* self = const_cast<daeVariable*>(this);
        m_pModel->m_pDataProxy->m_mapRuntimeVariableNodes[nIndex] = tmp;
        //std::cout << "Added runtime variable " << GetName() << "[" << nIndex << "] to the map" << std::endl;
    }

    return tmp;
}

adouble daeVariable::CreateSetupVariable(const daeDomainIndex* indexes, const size_t N) const
{
    if(m_ptrDomains.size() != N)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid get value call for [" << GetCanonicalName() << "], number of domains is " << m_ptrDomains.size() << " - but only " << N << " is given";
        throw e;
    }
    if(!m_pModel)
    {
        daeDeclareException(exInvalidPointer);
        e << "Invalid parent model in variable [" << GetCanonicalName() << "]";
        throw e;
    }
// Check if domains in indexes correspond to domains here
    for(size_t i = 0; i < N; i++)
    {
        if(indexes[i].m_eType == eDomainIterator ||
           indexes[i].m_eType == eIncrementedDomainIterator)
        {
            if(m_ptrDomains[i] != indexes[i].m_pDEDI->m_pDomain)
            {
                // If it is not the same domain check the number of points
                // It is acceptable to create a domain iterator on a domain 'x' and iterate over
                // some other variable which is distributed over another domain but with
                // the same number of points!
                if(m_ptrDomains[i]->GetNumberOfPoints() != indexes[i].m_pDEDI->m_pDomain->GetNumberOfPoints())
                {
                    if(m_pModel->m_pDataProxy)
                    {
                        string f = "Warning: You should not call the operator() function with the domain iterator created on the domain [%s]; "
                                   " use the domain [%s] instead "
                                   "(or a domain with the same number of points) as %d. index argument in variable [%s]";
                        string msg = (boost::format(f) % indexes[i].m_pDEDI->m_pDomain->GetCanonicalName() %
                                                         m_ptrDomains[i]->GetCanonicalName() %
                                                         (i+1) %
                                                         GetCanonicalName()).str();
                        m_pModel->m_pDataProxy->LogMessage(msg, 0);
                    }
                }
            }
        }
    }

    // If all daeDomainIndexes are integers store the adNodePtr in the map
    std::vector<size_t> int_indexes;
    if(N > 0)
        int_indexes.reserve(N);
    for(size_t i = 0; i < N; i++)
    {
        if(indexes[i].m_eType == eConstantIndex)
            int_indexes.push_back(indexes[i].m_nIndex);
        else
            break;
    }

    size_t nOverallIndex = -1;
    if(int_indexes.size() == N) // all indexes are constant indexes
    {
        nOverallIndex = m_nOverallIndex + CalculateIndex(int_indexes);
        std::map<size_t, adouble>::const_iterator it = m_pModel->m_pDataProxy->m_mapSetupVariableNodes.find(nOverallIndex);
        if(it != m_pModel->m_pDataProxy->m_mapSetupVariableNodes.end()) // if found
        {
            //std::cout << "Found variable " << GetName() << "(" << int_indexes[0] << ") in the map" << std::endl;
            return it->second;
        }
    }

    // Not found in the map: create a new one
    adouble tmp;
    adSetupVariableNode* node = new adSetupVariableNode();
    node->m_pVariable = const_cast<daeVariable*>(this);

    if(N > 0)
    {
        node->m_arrDomains.resize(N);
        for(size_t i = 0; i < N; i++)
            node->m_arrDomains[i] = indexes[i];
    }
    tmp.node = adNodePtr(node);
    tmp.setGatherInfo(true);

    // If all indexes are constant indexes and overall index found add it to the map
    if(int_indexes.size() == N && nOverallIndex != -1)
    {
        //std::cout << "Added variable " << GetName() << "(" << int_indexes[0] << ") to the map" << std::endl;
        daeVariable* self = const_cast<daeVariable*>(this);
        m_pModel->m_pDataProxy->m_mapSetupVariableNodes[nOverallIndex] = tmp;
    }

    return tmp;
}

adouble daeVariable::Calculate_dt(const size_t* indexes, const size_t N) const
{
    adouble tmp;
    size_t nIndex;
    daeExecutionContext* pExecutionContext;

    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

    nIndex = m_nOverallIndex + CalculateIndex(indexes, N);

    if(m_pModel->m_pDataProxy->GetVariableType(nIndex) == cnAssigned)
    {
        daeDeclareException(exInvalidCall);
        e << "Differential variable [" << GetCanonicalName() << "] cannot be fixed";
        throw e;
    }

/*********************************************************************************************
  New code; just I need is to set variable type and to add variable indexes to the equation
**********************************************************************************************/
    if(m_pModel->m_pExecutionContextForGatherInfo)
    {
        pExecutionContext = m_pModel->m_pExecutionContextForGatherInfo;
        if(!pExecutionContext)
            daeDeclareAndThrowException(exInvalidPointer);

        m_pModel->m_pDataProxy->SetVariableTypeGathered(nIndex, cnDifferential);

        daeEquationExecutionInfo* pEquationExecutionInfo = pExecutionContext->m_pEquationExecutionInfo;
    // StateTransitions::CreateRuntimeNode call may end up here
    // In that case we dont have pEquationExecutionInfo nor we need it
        if(pEquationExecutionInfo)
        {
            if(pExecutionContext->m_eEquationCalculationMode == eGatherInfo)
            {
                pEquationExecutionInfo->AddVariableInEquation(nIndex);
            }
            else
            {
            // If the mode is eGatherInfo I dont have to check whether variable is differential
            // since I dont know the variable types yet! Otherwise I do the check.
                if(m_pModel->m_pDataProxy->GetVariableTypeGathered(nIndex) != cnDifferential)
                {
                    daeDeclareException(exInvalidCall);
                    e << "Cannot get time derivative for non differential variable [" << GetCanonicalName();
                    throw e;
                }
            }
        }
    }

/**************************************************************************************
  Old code; all this seems non necessary (we dont need values in this function)
***************************************************************************************
    if(m_pModel->m_pExecutionContextForGatherInfo)
        pExecutionContext = m_pModel->m_pExecutionContextForGatherInfo;
    else
        pExecutionContext = m_pModel->m_pDataProxy->GetExecutionContext(nIndex);
    if(!pExecutionContext)
        daeDeclareAndThrowException(exInvalidPointer);

    // If the mode is eGatherInfo I dont have to check whether variable is differential
    // since I dont know the variable types yet !!
    if(pExecutionContext->m_eEquationCalculationMode != eGatherInfo)
    {
        if(m_pModel->m_pDataProxy->GetVariableTypeGathered(nIndex) != cnDifferential)
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot get time derivative for non differential variable [" << GetCanonicalName();
            throw e;
        }
    }

    if(pExecutionContext->m_eEquationCalculationMode == eGatherInfo)
    {
        daeEquationExecutionInfo* pEquationExecutionInfo = pExecutionContext->m_pEquationExecutionInfo;
        if(!pEquationExecutionInfo)
            daeDeclareAndThrowException(exInvalidPointer);

        m_pModel->m_pDataProxy->SetVariableTypeGathered(nIndex, cnDifferential);
        pEquationExecutionInfo->AddVariableInEquation(nIndex);
    // Here we dont need any calculation
        //tmp.setValue(0);
        //tmp.setDerivative(0);
    }
    else if(pExecutionContext->m_eEquationCalculationMode == eCalculateJacobian)
    {
        tmp.setValue(*m_pModel->m_pDataProxy->GetTimeDerivative(nIndex));
        if(pExecutionContext->m_nCurrentVariableIndexForJacobianEvaluation == nIndex)
            tmp.setDerivative(pExecutionContext->m_dInverseTimeStep);
        else
            tmp.setDerivative(0);
    }
    else if(pExecutionContext->m_eEquationCalculationMode == eCalculate)
    {
        tmp.setValue(*m_pModel->m_pDataProxy->GetTimeDerivative(nIndex));
        tmp.setDerivative(0); // No need for it
    }
    else if(pExecutionContext->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs)
    {
    // Here we dont need any calculation
        //tmp.setValue(0);
        //tmp.setDerivative(0);
    }
    else if(pExecutionContext->m_eEquationCalculationMode == eCalculateSensitivities)
    {
        daeDeclareAndThrowException(exInvalidCall)
    }
    else if(pExecutionContext->m_eEquationCalculationMode == eCalculateGradients)
    {
        daeDeclareAndThrowException(exInvalidCall)
    }
    else
    {
        // Unknown state
        daeDeclareException(exInvalidCall);
        e << "Unknown function evaluation mode for variable [" << GetCanonicalName();
        throw e;
    }
*/

    if(m_pModel->m_pDataProxy->GetGatherInfo())
    {
        std::map<size_t, adouble>::const_iterator it = m_pModel->m_pDataProxy->m_mapRuntimeTimeDerivativeNodes.find(nIndex);
        if(it != m_pModel->m_pDataProxy->m_mapRuntimeTimeDerivativeNodes.end()) // if found
            return it->second;

        adRuntimeTimeDerivativeNode* node = new adRuntimeTimeDerivativeNode();
        node->m_pVariable = const_cast<daeVariable*>(this);
        node->m_nOverallIndex = nIndex;
        //node->m_nOrder = 1;
        if(N > 0)
        {
            node->m_narrDomains.resize(N);
            for(size_t i = 0; i < N; i++)
                node->m_narrDomains[i] = indexes[i];
        }
        tmp.node = adNodePtr(node);
        tmp.setGatherInfo(true);

        // Add it to the map for it has not been added yet
        //daeVariable* self = const_cast<daeVariable*>(this);
        m_pModel->m_pDataProxy->m_mapRuntimeTimeDerivativeNodes[nIndex] = tmp;
    }
    return tmp;
}

adouble daeVariable::CreateSetupTimeDerivative(const daeDomainIndex* indexes, const size_t N) const
{
    if(m_ptrDomains.size() != N)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid time derivative call for [" << GetCanonicalName() << "], number of domains is " << m_ptrDomains.size() << " - but only " << N << " is given";
        throw e;
    }
    if(!m_pModel)
    {
        daeDeclareException(exInvalidPointer);
        e << "Invalid parent model in variable [" << GetCanonicalName() << "]";
        throw e;
    }
// Check if domains in indexes correspond to domains here
    for(size_t i = 0; i < N; i++)
    {
        if(indexes[i].m_eType == eDomainIterator ||
           indexes[i].m_eType == eIncrementedDomainIterator)
        {
            if(m_ptrDomains[i] != indexes[i].m_pDEDI->m_pDomain)
            {
                // If it is not the same domain check the number of points
                // It is acceptable to create a domain iterator on a domain 'x' and iterate over
                // some other variable which is distributed over another domain but with
                // the same number of points!
                if(m_ptrDomains[i]->GetNumberOfPoints() != indexes[i].m_pDEDI->m_pDomain->GetNumberOfPoints())
                {
                    if(m_pModel->m_pDataProxy)
                    {
                        string f = "Warning: You should not call the function dt() with the domain iterator created on the domain [%s]; "
                                   " use the domain [%s] instead "
                                   "(or a domain with the same number of points) as %d. index argument in variable [%s]";
                        string msg = (boost::format(f) % indexes[i].m_pDEDI->m_pDomain->GetCanonicalName() %
                                                         m_ptrDomains[i]->GetCanonicalName() %
                                                         (i+1) %
                                                         GetCanonicalName()).str();
                        m_pModel->m_pDataProxy->LogMessage(msg, 0);
                    }
                }
            }
        }
    }

    // If all daeDomainIndexes are integers store the adNodePtr in the map
    std::vector<size_t> int_indexes;
    if(N > 0)
        int_indexes.reserve(N);
    for(size_t i = 0; i < N; i++)
    {
        if(indexes[i].m_eType == eConstantIndex)
            int_indexes.push_back(indexes[i].m_nIndex);
        else
            break;
    }

    size_t nOverallIndex = -1;
    if(int_indexes.size() == N) // all indexes are constant indexes
    {
        nOverallIndex = m_nOverallIndex + CalculateIndex(int_indexes);
        std::map<size_t, adouble>::const_iterator it = m_pModel->m_pDataProxy->m_mapSetupTimeDerivativeNodes.find(nOverallIndex);
        if(it != m_pModel->m_pDataProxy->m_mapSetupTimeDerivativeNodes.end()) // if found
            return it->second;
    }

    // Not found in the map: create a new one
    adouble tmp;
    adSetupTimeDerivativeNode* node = new adSetupTimeDerivativeNode();
    node->m_pVariable = const_cast<daeVariable*>(this);
    //node->m_nOrder = 1;

    if(N > 0)
    {
        node->m_arrDomains.resize(N);
        for(size_t i = 0; i < N; i++)
            node->m_arrDomains[i] = indexes[i];
    }
    tmp.node = adNodePtr(node);
    tmp.setGatherInfo(true);

    // If all indexes are constant indexes and overall index found add it to the map
    if(int_indexes.size() == N && nOverallIndex != -1)
    {
        daeVariable* self = const_cast<daeVariable*>(this);
        m_pModel->m_pDataProxy->m_mapSetupTimeDerivativeNodes[nOverallIndex] = tmp;
    }

    return tmp;
}

adouble daeVariable::partial(const size_t nOrder,
                             const daeDomain_t& D,
                             const size_t* indexes,
                             const size_t N,
                             const daeeDiscretizationMethod  eDiscretizationMethod,
                             const std::map<std::string, std::string>& mapDiscretizationOptions) const
{
    adouble tmp;
    vector<adouble> V;
    size_t i, nNoPoints, nFixedDomain, nIndex;

    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pModel->m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);
    if(m_ptrDomains.size() == 0)
    {
        daeDeclareException(exInvalidCall);
        e << "Cannot get partial derivative for non distributed variable [" << GetCanonicalName();
        throw e;
    }
    if(N != m_ptrDomains.size())
    {
        daeDeclareException(exInvalidCall);
        e << "Illegal number of domains, variable " << GetCanonicalName();
        throw e;
    }
    if(D.GetType() != eStructuredGrid)
    {
        daeDeclareException(exInvalidCall);
        e << "Partial derivatives can be get only for structured grids, domain [" << D.GetCanonicalName() << "], variable [" << GetCanonicalName() << "]";
        throw e;
    }

// Find which domain is fixed
    nFixedDomain = ULONG_MAX;
    for(i = 0; i < m_ptrDomains.size(); i++)
    {
        if(m_ptrDomains[i] == &D)
            nFixedDomain = i;
    }
    if(nFixedDomain == ULONG_MAX)
    {
        daeDeclareException(exInvalidCall);
        e << "Illegal domain for partial derivative, variable [" << GetCanonicalName() << "]";
        throw e;
    }

// Find index of the point for which we are trying to find derivative
    nIndex    = indexes[nFixedDomain];
    nNoPoints = D.GetNumberOfPoints();

    const daeDomain* pDomain = dynamic_cast<const daeDomain*>(&D);
    if(!pDomain)
        daeDeclareAndThrowException(exInvalidPointer);

    daePartialDerivativeVariable pdv(nOrder, *this, *pDomain, nFixedDomain, N, indexes, eDiscretizationMethod, mapDiscretizationOptions);
    tmp = pdv.CreateSetupNode();

//	if(m_pModel->m_pDataProxy->GetGatherInfo())
//	{
//		adRuntimePartialDerivativeNode* node = new adRuntimePartialDerivativeNode();
//		node->pardevnode = tmp.node;
//		node->m_nOverallIndex = nIndex;
//		const daeDomain* pDomain = dynamic_cast<const daeDomain*>(&D);
//		node->m_pDomain = const_cast<daeDomain*>(pDomain);
//		node->m_pVariable = const_cast<daeVariable*>(this);
//		node->m_nOrder = nOrder;
//		if(N > 0)
//		{
//			node->m_narrDomains.resize(N);
//			for(size_t i = 0; i < N; i++)
//				node->m_narrDomains[i] = indexes[i];
//		}
//		tmp.node = adNodePtr(node);
//		tmp.setGatherInfo(true);
//	}

    return tmp;
}

adouble daeVariable::CreateSetupPartialDerivative(const size_t nOrder,
                                                  const daeDomain_t& D,
                                                  const daeDomainIndex* indexes,
                                                  const size_t N,
                                                  const daeeDiscretizationMethod  eDiscretizationMethod,
                                                  const std::map<std::string, std::string>& mapDiscretizationOptions) const
{
    if(m_ptrDomains.size() != N)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid partial derivative call for [" << GetCanonicalName() << "], number of domains is " << m_ptrDomains.size() << " - but only " << N << " is given";
        throw e;
    }
    if(!m_pModel)
    {
        daeDeclareException(exInvalidPointer);
        e << "Invalid parent model in variable [" << GetCanonicalName() << "]";
        throw e;
    }
// Check if domains in indexes correspond to domains here
    for(size_t i = 0; i < N; i++)
    {
        if(indexes[i].m_eType == eDomainIterator ||
           indexes[i].m_eType == eIncrementedDomainIterator)
        {
            if(m_ptrDomains[i] != indexes[i].m_pDEDI->m_pDomain)
            {
                // If it is not the same domain check the number of points
                // It is acceptable to create a domain iterator on a domain 'x' and iterate over
                // some other variable which is distributed over another domain but with
                // the same number of points!
                if(m_ptrDomains[i]->GetNumberOfPoints() != indexes[i].m_pDEDI->m_pDomain->GetNumberOfPoints())
                {
                    if(m_pModel->m_pDataProxy)
                    {
                        string f = "Warning: You should not call the functions d() or d2() with the domain iterator created on the domain [%s]; "
                                   " use the domain [%s] instead "
                                   "(or a domain with the same number of points) as %d. index argument in variable [%s]";
                        string msg = (boost::format(f) % indexes[i].m_pDEDI->m_pDomain->GetCanonicalName() %
                                                         m_ptrDomains[i]->GetCanonicalName() %
                                                         (i+1) %
                                                         GetCanonicalName()).str();
                        m_pModel->m_pDataProxy->LogMessage(msg, 0);
                    }
                }
            }
        }
    }

    adouble tmp;
    adSetupPartialDerivativeNode* node = new adSetupPartialDerivativeNode();
    const daeDomain* pDomain = dynamic_cast<const daeDomain*>(&D);
    node->m_pDomain = const_cast<daeDomain*>(pDomain);
    node->m_pVariable = const_cast<daeVariable*>(this);
    node->m_nOrder = nOrder;
    node->m_eDiscretizationMethod = eDiscretizationMethod;
    node->m_mapDiscretizationOptions = mapDiscretizationOptions;

    if(N > 0)
    {
        node->m_arrDomains.resize(N);
        for(size_t i = 0; i < N; i++)
            node->m_arrDomains[i] = indexes[i];
    }
    tmp.node = adNodePtr(node);
    tmp.setGatherInfo(true);
    return tmp;
}


}
}
