#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{
daePartialDerivativeVariable::daePartialDerivativeVariable(const size_t			nOrder,
														   const daeVariable&	rVariable,
														   const daeDomain&		rDomain,
														   const size_t			nDomainIndex,
														   const size_t			nNoIndexes,
														   const size_t*		pIndexes):
										m_nOrder(nOrder),
										m_rVariable(rVariable),
										m_rDomain(rDomain),
										m_nDomainIndex(nDomainIndex),
										m_nNoIndexes(nNoIndexes)
{
	m_pIndexes = new size_t[nNoIndexes];
	for(size_t i = 0; i < nNoIndexes; i++)
		m_pIndexes[i] = pIndexes[i];
}

daePartialDerivativeVariable::~daePartialDerivativeVariable(void)
{
	if(m_pIndexes)
		delete m_pIndexes;
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






}
}
