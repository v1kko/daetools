#include "stdafx.h"
#include "coreimpl.h"
#include <boost/format.hpp>

namespace dae 
{
namespace core 
{
// Discretization functions
adouble pd_BFD(daePartialDerivativeVariable& pdv);
adouble pd_FFD(daePartialDerivativeVariable& pdv);
adouble pd_CFD(daePartialDerivativeVariable& pdv);
adouble pd_upwindCCFV(daePartialDerivativeVariable& pdv);

// daePartialDerivativeVariable class
daePartialDerivativeVariable::daePartialDerivativeVariable(const size_t			                     nOrder,
                                                           const daeVariable&	                     rVariable,
                                                           const daeDomain&		                     rDomain,
                                                           const size_t			                     nDomainIndex,
                                                           const size_t                              nNoIndexes,
                                                           const size_t*                             pIndexes,
                                                           const daeeDiscretizationMethod            eDiscretizationMethod,
                                                           const std::map<std::string, std::string>& mapDiscretizationOptions):
										m_nOrder(nOrder),
										m_rVariable(rVariable),
										m_rDomain(rDomain),
										m_nDomainIndex(nDomainIndex),
                                        m_nNoIndexes(nNoIndexes),
                                        m_eDiscretizationMethod(eDiscretizationMethod),
                                        m_mapDiscretizationOptions(mapDiscretizationOptions)
{
	m_pIndexes = new size_t[nNoIndexes];
	for(size_t i = 0; i < nNoIndexes; i++)
		m_pIndexes[i] = pIndexes[i];
}

daePartialDerivativeVariable::~daePartialDerivativeVariable(void)
{
	if(m_pIndexes)
		delete[] m_pIndexes;
}

size_t daePartialDerivativeVariable::GetPoint(void) const
{
	return m_pIndexes[m_nDomainIndex];
}

size_t daePartialDerivativeVariable::GetOrder(void) const
{
    return m_nOrder;
}

const daeDomain& daePartialDerivativeVariable::GetDomain(void) const
{
	return m_rDomain;
}

const daeVariable& daePartialDerivativeVariable::GetVariable(void) const
{
	return m_rVariable;
}

adouble daePartialDerivativeVariable::operator[](size_t nIndex)
{
	adouble ad;
	size_t nOldIndex = m_pIndexes[m_nDomainIndex];
	m_pIndexes[m_nDomainIndex] = nIndex;
	ad = m_rVariable.Create_adouble(m_pIndexes, m_nNoIndexes);
	m_pIndexes[m_nDomainIndex] = nOldIndex;
	return ad;
}

adouble daePartialDerivativeVariable::operator()(size_t nIndex)
{
	adouble ad;
	size_t nOldIndex = m_pIndexes[m_nDomainIndex];
	m_pIndexes[m_nDomainIndex] = nIndex;
	ad = m_rVariable.Create_adouble(m_pIndexes, m_nNoIndexes);
	m_pIndexes[m_nDomainIndex] = nOldIndex;
	return ad;
}

adouble daePartialDerivativeVariable::CreateSetupNode(void)
{
    if(m_rDomain.GetType() != eStructuredGrid)
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot calculate partial derivative per domain [%s]: domain is not a structured grid";
        e << (boost::format(msg) % m_rDomain.GetCanonicalName()).str();
        throw e;
    }

    if(m_eDiscretizationMethod == eBFDM)
        return pd_BFD(*this);
    else if(m_eDiscretizationMethod == eFFDM)
        return pd_FFD(*this);
    else if(m_eDiscretizationMethod == eCFDM)
        return pd_CFD(*this);
    else if(m_eDiscretizationMethod == eUpwindCCFV)
        return pd_upwindCCFV(*this);
    else
        daeDeclareAndThrowException(exInvalidCall);
}

// Discretization functions implementation
adouble pd_upwindCCFV(daePartialDerivativeVariable& /*pdv*/)
{
    daeDeclareAndThrowException(exNotImplemented);
    return adouble();
}

adouble pd_BFD(daePartialDerivativeVariable& /*pdv*/)
{
    daeDeclareAndThrowException(exNotImplemented);
    return adouble();
}

adouble pd_FFD(daePartialDerivativeVariable& /*pdv*/)
{
    daeDeclareAndThrowException(exNotImplemented);
    return adouble();
}

adouble pd_CFD(daePartialDerivativeVariable& pdv)
{
    adouble pardev;
// Index which we are calculating partial derivative for
    const size_t n = pdv.GetPoint();
// Domain which we are calculating partial derivative for
    const daeDomain& d = pdv.GetDomain();
// Number of points in the domain we are calculating partial derivative for
    const size_t N = d.GetNumberOfPoints();
// Discretization order should be in the DiscretizationOptions map, otherwise a default value will be used
    size_t nDiscretizationOrder = 2;
    std::map<std::string, std::string>::const_iterator citer = pdv.m_mapDiscretizationOptions.find("DiscretizationOrder");
    // If DiscretizationOrder found in options use it, otherwise use the default value of 2
    if(citer != pdv.m_mapDiscretizationOptions.end())
        nDiscretizationOrder = fromString<size_t>(citer->second);

    switch(nDiscretizationOrder)
    {
    case 2:
        if(N < 3)
        {
            daeDeclareException(exInvalidCall);
            boost::format bf("Cannot evaluate partial derivative per domain [%s]: "
                             "the number of points cannot be lower than 3");
            e << (bf % d.GetCanonicalName()).str();
            throw e;
        }

        if(pdv.GetOrder() == 1)
        {
            if(n == 0) // LEFT BOUND
            {
            //	dV(0)/dD = (-3V[0] + 4V[1] - V[2]) / (D[2] - D[0])
                pardev = (-3*pdv[0] + 4*pdv[1] - pdv[2]) / (d[2] - d[0]);
            }
            else if(n == N-1) // RIGHT BOUND
            {
            //	dV(n)/dD = (3V[n] - 4V[n-1] + V[n-2]) / (D[n] - D[n-2])
                pardev = (3*pdv[n] - 4*pdv[n-1] + pdv[n-2]) / (d[n] - d[n-2]);
            }
            else // INTERIOR POINTs
            {
            //	dV(i)/dD = (V[i+1] - V[i-1]) / (D[i+1] - D[i-1])
                pardev = (pdv[n+1] - pdv[n-1]) / (d[n+1] - d[n-1]);
            }
        }
        else if(pdv.GetOrder() == 2)
        {
            if(n == 0) // LEFT BOUND
            {
            //	dV(0)/dD = (V[0] - 2V[1] + V[2]) / ((D[2] - D[1]) * (D[1] - D[0]))
                pardev = (pdv[0] - 2*pdv[1] + pdv[2]) / ( (d[2] - d[1]) * (d[1] - d[0]) );
            }
            else if(n == N-1) // RIGHT BOUND
            {
            //	dV(n)/dD = (V[n] - 2V[n-1] + V[n-2]) / ((D[n] - D[n-1]) * (D[n-1] - D[n-2]))
                pardev = (pdv[n] - 2*pdv[n-1] + pdv[n-2]) / ((d[n] - d[n-1]) * (d[n-1] - d[n-2]));
            }
            else // INTERIOR POINTs
            {
            //	d2V(i)/dD2 = (V[i+1] - 2V[i] + V[i-1]) / ((D[i+1] - D[i]) * (D[i] - D[i-1]))
                pardev = (pdv[n+1] - 2*pdv[n] + pdv[n-1]) / ((d[n+1] - d[n]) * (d[n] - d[n-1]));
            }
        }
        else
        {
            daeDeclareException(exNotImplemented);
            boost::format bf("Cannot evaluate partial derivatives of orders higher than 2 (order %d requested)");
            e << (bf % pdv.GetOrder()).str();
            throw e;
        }
        break;

    case 4:
        {
            daeDeclareException(exNotImplemented);
            boost::format bf("Cannot evaluate a partial derivative using Center Finite Difference method of order %d "
                             "(not implemented at the moment)");
            e << (bf % nDiscretizationOrder).str();
            throw e;
        }
        break;

    case 6:
        {
            daeDeclareException(exNotImplemented);
            boost::format bf("Cannot evaluate a partial derivative using Center Finite Difference method of order %d "
                             "(not implemented at the moment)");
            e << (bf % nDiscretizationOrder).str();
            throw e;
        }
        break;

    default:
        {
            daeDeclareException(exNotImplemented);
            boost::format bf("Cannot evaluate a partial derivative using Center Finite Difference method of order %d "
                             "(only orders 2, 4 and 6 are supported)");
            e << (bf % nDiscretizationOrder).str();
            throw e;
        }
    }

    return pardev;
}



}
}
