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

adouble pd_BFD(daePartialDerivativeVariable& pdv)
{
    adouble pardev;
// Index for which we are calculating partial derivative
    const size_t n = pdv.GetPoint();
// Domain for which we are calculating partial derivative
    const daeDomain& d = pdv.GetDomain();
// Number of points in the domain for which we are calculating partial derivative
    const size_t N = d.GetNumberOfPoints();
// Discretization order should be in the DiscretizationOptions map, otherwise a default value will be used
    size_t nDiscretizationOrder = 1;
    std::map<std::string, std::string>::const_iterator citer = pdv.m_mapDiscretizationOptions.find("DiscretizationOrder");
    // If DiscretizationOrder found in options use it, otherwise use the default value of 1
    if(citer != pdv.m_mapDiscretizationOptions.end())
        nDiscretizationOrder = fromString<size_t>(citer->second);

    if(pdv.GetOrder() > 2)
    {
        daeDeclareException(exNotImplemented);
        boost::format bf("Cannot evaluate partial derivatives of orders higher than 2 (order %d requested)");
        e << (bf % pdv.GetOrder()).str();
        throw e;
    }
    if(N < 3)
    {
        daeDeclareException(exInvalidCall);
        boost::format bf("Cannot evaluate partial derivative per domain [%s]: "
                         "the number of points cannot be lower than 3");
        e << (bf % d.GetCanonicalName()).str();
        throw e;
    }

    switch(nDiscretizationOrder)
    {
    case 1:
        if(pdv.GetOrder() == 1) // 1st derivative
        {
            if(n == 0) // LEFT BOUND
            {
            //	dV(0)/dD = (V[1] - V[0]) / (D[1] - D[0])
            // Nota Bene:
            //   In case of the Neumann bc: dy/dx = 0, the expression below will produce y[1] = y[0] so this should be ok
                pardev = (pdv[1] - pdv[0]) / (d[1] - d[0]);
            }
            else // INTERIOR POINTs + RIGHT BOUND
            {
            //	dV(i)/dD = (V[i] - V[i-1]) / (D[i] - D[i-1])
                pardev = (pdv[n] - pdv[n-1]) / (d[n] - d[n-1]);
            }
        }
        else if(pdv.GetOrder() == 2) // 2nd derivative
        {
            if(n == 0) // LEFT BOUND  ->  DOUBLE-CHECK!!!
            {
            // Nota Bene:
            //   Identical to the CFD formula for interior points.
                pardev = (pdv[2] - 2*pdv[1] + pdv[0]) / ((d[2] - d[1])*(d[1] - d[0]));
            }
            else if(n == 1) // ONE AFTER THE LEFT BOUND ->   DOUBLE-CHECK!!!
            {
            // Nota Bene:
            //   Identical to the CFD formula for interior points:
            //   dV[1]/dD = (V[2] - 2V[1] + V[0]) / ((D[2] - D[1]) * (D[1] - D[0]))
                pardev = (pdv[2] - 2*pdv[1] + pdv[0]) / ((d[2] - d[1])*(d[1] - d[0]));
            }
            else // INTERIOR POINTs INCLUDING THE RIGHT BOUND
            {
            // d2V(i)/dD2 = (V[i-2] - 2V[i-1] + V[i]) / ((D[i-1] - D[i-2]) * (D[i] - D[i-1]))
                pardev = (pdv[n-2] - 2*pdv[n-1] + pdv[n]) / ((d[n-1] - d[n-2]) * (d[n] - d[n-1]));
            }
        }
        else
        {
            // Already handled, but just in case we end up here somehow
            daeDeclareAndThrowException(exNotImplemented);
        }

        break;

    default:
        {
            daeDeclareException(exNotImplemented);
            boost::format bf("Cannot evaluate a partial derivative using the Backward Finite Difference method of order %d "
                             "(only the discretization order 1 is supported)");
            e << (bf % nDiscretizationOrder).str();
            throw e;
        }
    }

    return pardev;
}

adouble pd_FFD(daePartialDerivativeVariable& pdv)
{
    adouble pardev;
// Index for which we are calculating partial derivative
    const size_t n = pdv.GetPoint();
// Domain for which we are calculating partial derivative
    const daeDomain& d = pdv.GetDomain();
// Number of points in the domain for which we are calculating partial derivative
    const size_t N = d.GetNumberOfPoints();
// Discretization order should be in the DiscretizationOptions map, otherwise a default value will be used
    size_t nDiscretizationOrder = 1;
    std::map<std::string, std::string>::const_iterator citer = pdv.m_mapDiscretizationOptions.find("DiscretizationOrder");
    // If DiscretizationOrder found in options use it, otherwise use the default value of 1
    if(citer != pdv.m_mapDiscretizationOptions.end())
        nDiscretizationOrder = fromString<size_t>(citer->second);

    if(pdv.GetOrder() > 2)
    {
        daeDeclareException(exNotImplemented);
        boost::format bf("Cannot evaluate partial derivatives of orders higher than 2 (order %d requested)");
        e << (bf % pdv.GetOrder()).str();
        throw e;
    }

    switch(nDiscretizationOrder)
    {
    case 1:
        if(pdv.GetOrder() == 1) // 1st derivative
        {
            if(N < 3)
            {
                daeDeclareException(exInvalidCall);
                boost::format bf("Cannot evaluate partial derivative per domain [%s]: "
                                 "the number of points cannot be lower than 3");
                e << (bf % d.GetCanonicalName()).str();
                throw e;
            }

            if(n == N-1) // RIGHT BOUND
            {
            //	dV(n-1)/dD = (V[n-1] - V[n-2]) / (D[n-1] - D[n-2])
            // Nota Bene:
            //   In case of the Neumann bc: dy/dx = 0, the expression below will produce y[n-1] = y[n-2], so this should be ok
                pardev = (pdv[n-1] - pdv[n-2]) / (d[n-1] - d[n-2]);
            }
            else // INTERIOR POINTs INCLUDING THE LEFT BOUND
            {
            //	dV(i)/dD = (V[i+1] - V[i]) / (D[i+1] - D[i])
                pardev = (pdv[n+1] - pdv[n]) / (d[n+1] - d[n]);
            }
        }
        else if(pdv.GetOrder() == 2) // 2nd derivative
        {
            if(N < 4)
            {
                daeDeclareException(exInvalidCall);
                boost::format bf("Cannot evaluate partial derivative per domain [%s]: "
                                 "the number of points cannot be lower than 4");
                e << (bf % d.GetCanonicalName()).str();
                throw e;
            }

            if(n == N-1) // RIGHT BOUND
            {
            // Nota Bene:
            //   This is a CFD formula for the right bound.
            //	dV(n)/dD = (V[n] - 2V[n-1] + V[n-2]) / ((D[n] - D[n-1]) * (D[n-1] - D[n-2]))
                pardev = (pdv[n] - 2*pdv[n-1] + pdv[n-2]) / ((d[n] - d[n-1]) * (d[n-1] - d[n-2]));
            }
            else if(n == N-2) // ONE BEFORE THE RIGHT BOUND
            {
            // Nota Bene:
            //   This is a CFD formula for interior points.
            //   This formula is obtained by calculating a derivative of the FFD 1st order derivative:
            //      dV[n]/dD = (V[n+1]-V[n])/dx
            //      d2V[n]/dD2 = d[V[n+1]-V[n])/dx] = [(V[n+1]-V[n])/dx - (V[n]-V[n-1])/dx] / dx
            //                                          BFD formula        BFD formula
            //                 = (V[n+1] - 2V[n] + V[n-1]) / ((D[n+1] - D[n]) * (D[n] - D[n-1]))
                pardev = (pdv[n+1] - 2*pdv[n] + pdv[n-1]) / ((d[n+1] - d[n]) * (d[n] - d[n-1]));
            }
            else // INTERIOR POINTs INCLUDING THE LEFT BOUND
            {
            //	d2V(i)/dD2 = (V[i+2] - 2V[i+1] + V[i]) / ((D[i+2] - D[i+1]) * (D[i+1] - D[i]))
                pardev = (pdv[n+2] - 2*pdv[n+1] + pdv[n]) / ((d[n+2] - d[n+1]) * (d[n+1] - d[n]));
            }
        }
        else
        {
            // Already handled, but just in case we end up here somehow
            daeDeclareAndThrowException(exNotImplemented);
        }

        break;

    default:
        {
            daeDeclareException(exNotImplemented);
            boost::format bf("Cannot evaluate a partial derivative using the Forward Finite Difference method of order %d "
                             "(only the discretization order 1 is supported)");
            e << (bf % nDiscretizationOrder).str();
            throw e;
        }
    }

    return pardev;
}

adouble pd_CFD(daePartialDerivativeVariable& pdv)
{
    adouble pardev;
// Index for which we are calculating partial derivative
    const size_t n = pdv.GetPoint();
// Domain for which we are calculating partial derivative
    const daeDomain& d = pdv.GetDomain();
// Number of points in the domain for which we are calculating partial derivative
    const size_t N = d.GetNumberOfPoints();
// Discretization order should be in the DiscretizationOptions map, otherwise a default value will be used
    size_t nDiscretizationOrder = 2;
    std::map<std::string, std::string>::const_iterator citer = pdv.m_mapDiscretizationOptions.find("DiscretizationOrder");
    // If DiscretizationOrder found in options use it, otherwise use the default value of 2
    if(citer != pdv.m_mapDiscretizationOptions.end())
        nDiscretizationOrder = fromString<size_t>(citer->second);

    if(pdv.GetOrder() > 2)
    {
        daeDeclareException(exNotImplemented);
        boost::format bf("Cannot evaluate partial derivatives of orders higher than 2 (order %d requested)");
        e << (bf % pdv.GetOrder()).str();
        throw e;
    }

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
            // Already handled, but just in case we end up here somehow
            daeDeclareAndThrowException(exNotImplemented);
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

    default:
        {
            daeDeclareException(exNotImplemented);
            boost::format bf("Cannot evaluate a partial derivative using Center Finite Difference method of order %d "
                             "(only the discretization orders 2 and 4 are supported)");
            e << (bf % nDiscretizationOrder).str();
            throw e;
        }
    }

    return pardev;
}



}
}
