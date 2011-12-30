#include "stdafx.h"
#include "coreimpl.h"
#include "adouble.h"
#include <typeinfo>
#include "nodes_array.h"
using namespace boost;

namespace dae 
{
namespace core 
{
#define adCheckArrays(left, right)	if((left).GetSize() == 0 || (right).GetSize() == 0)   \
									{ \
										daeDeclareAndThrowException(exInvalidCall);   \
									} \
									if((left).GetSize() > 1 && (right).GetSize() > 1) \
									{ \
										if((left).GetSize() != (right).GetSize()) \
											daeDeclareAndThrowException(exInvalidCall); \
									}

/*********************************************************************************************
	adouble_array
**********************************************************************************************/
adouble_array::adouble_array()
{
	m_bGatherInfo = false;
}

adouble_array::adouble_array(const adouble_array& a)
{
	m_bGatherInfo = false;
	if(a.getGatherInfo())
	{
		m_bGatherInfo = true;
		node  = CLONE_NODE_ARRAY(a.node);  
	}
	m_arrValues = a.m_arrValues;
}

adouble_array::~adouble_array()
{
}
	
size_t adouble_array::GetSize(void) const
{
	return m_arrValues.size();
}
	
void adouble_array::Resize(size_t n)
{
	m_arrValues.clear();
	m_arrValues.resize(n);
}

void adouble_array::operator =(const adouble_array& a) 
{
	if(this->m_bGatherInfo || a.getGatherInfo())
	{
		this->m_bGatherInfo = true;
		this->node = CLONE_NODE_ARRAY(a.node);  
	}
	m_arrValues = a.m_arrValues;
}

adouble& adouble_array::operator[](size_t nIndex)
{
// This enables operations on an array which has only one element,
// for instance single variables, parameters and constants
// They always return its single value (for arbitrary index)
	if(m_arrValues.size() == 1) 
	{
		return m_arrValues[0];
	}
	else
	{
		if(nIndex >= m_arrValues.size())
			daeDeclareAndThrowException(exInvalidCall);
		
		return m_arrValues[nIndex];
	}
}

const adouble& adouble_array::operator[](size_t nIndex) const
{
// This enables operations on an array which has only one element,
// for instance single variables, parameters and constants
// They always return its single value (for arbitrary index)
	if(m_arrValues.size() == 1) 
	{
		return m_arrValues[0];
	}
	else
	{
		if(nIndex >= m_arrValues.size())
			daeDeclareAndThrowException(exInvalidCall);
		
		return m_arrValues[nIndex];
	}
}

adouble adouble_array::GetItem(size_t nIndex)
{
// This enables operations on an array which has only one element,
// for instance single variables, parameters and constants
// They always return its single value (for arbitrary index)
	if(m_arrValues.size() == 1) 
	{
		return m_arrValues[0];
	}
	else
	{
		if(nIndex >= m_arrValues.size())
			daeDeclareAndThrowException(exInvalidCall);
		
		return m_arrValues[nIndex];
	}
}

bool adouble_array::getGatherInfo(void) const
{
	return m_bGatherInfo;
}

void adouble_array::setGatherInfo(bool bGatherInfo)
{
	m_bGatherInfo = bGatherInfo;
}

const adouble_array adouble_array::operator -(void) const
{
	adouble_array tmp;
	if(getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eSign, 
			                                                    CLONE_NODE_ARRAY(node) ));
	    return tmp;
	}
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = -(*this)[i];
	return tmp;
}

const adouble_array adouble_array::operator +(const adouble_array& a) const
{
	adouble_array tmp;
	if(getGatherInfo() || a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(ePlus, 
			                                                     CLONE_NODE_ARRAY(node), 
									                             CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}

	adCheckArrays(*this, a)
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] + a[i];
	return tmp;
}

const adouble_array adouble_array::operator +(const real_t v) const
{
	adouble_array tmp;
	if(getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(ePlus, 
														         CLONE_NODE_ARRAY(node), 
														         shared_ptr<adNodeArray>(new adConstantNodeArray(v, UNITS(node))) ));
	    return tmp;
	}
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] + v;
	return tmp;
}

const adouble_array adouble_array::operator +(const adouble& a) const
{
	adouble_array tmp;
	if(getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(ePlus, 
														         CLONE_NODE_ARRAY(node), 
														         shared_ptr<adNodeArray>(new adSingleNodeArray(CLONE_NODE(a.node, a.getValue())) ) ));
	    return tmp;
	}
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] + a;
	return tmp;
}

const adouble_array operator +(const adouble& a, const adouble_array& arr)
{
	adouble_array tmp;
	if(arr.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(ePlus, 
														         shared_ptr<adNodeArray>(new adSingleNodeArray(CLONE_NODE(a.node, a.getValue())) ),
														         CLONE_NODE_ARRAY(arr.node)
																 ));
	    return tmp;
	}
	
	size_t n = arr.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = a + arr[i];
	return tmp;
}

const adouble_array operator +(const real_t v, const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(ePlus, 
														         shared_ptr<adNodeArray>(new adConstantNodeArray(v, UNITS(a.node))),
															     CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = v + a[i];
	return tmp;
}

const adouble_array adouble_array::operator -(const adouble_array& a) const
{
	adouble_array tmp;
	if(getGatherInfo() || a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMinus, 
			                                                     CLONE_NODE_ARRAY(node), 
									                             CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}

	adCheckArrays(*this, a)
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] - a[i];
	return tmp;
}

const adouble_array adouble_array::operator -(const real_t v) const
{
	adouble_array tmp;
	if(getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMinus, 
														         CLONE_NODE_ARRAY(node), 
														         shared_ptr<adNodeArray>(new adConstantNodeArray(v, UNITS(node))) ));
	    return tmp;
	}
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] - v;
	return tmp;
}

const adouble_array adouble_array::operator -(const adouble& a) const
{
	adouble_array tmp;
	if(getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMinus, 
														         CLONE_NODE_ARRAY(node), 
														         shared_ptr<adNodeArray>(new adSingleNodeArray(CLONE_NODE(a.node, a.getValue())) ) ));
	    return tmp;
	}
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] - a;
	return tmp;
}

const adouble_array operator -(const adouble& a, const adouble_array& arr)
{
	adouble_array tmp;
	if(arr.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMinus, 
														         shared_ptr<adNodeArray>(new adSingleNodeArray(CLONE_NODE(a.node, a.getValue())) ),
														         CLONE_NODE_ARRAY(arr.node)
																 ));
	    return tmp;
	}
	
	size_t n = arr.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = a - arr[i];
	return tmp;
}

const adouble_array operator -(const real_t v, const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMinus, 
														         shared_ptr<adNodeArray>(new adConstantNodeArray(v, UNITS(a.node))),
															     CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = v - a[i];
	return tmp;
}
											
const adouble_array adouble_array::operator *(const adouble_array& a) const
{
	adouble_array tmp;
	if(getGatherInfo() || a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMulti, 
			                                                     CLONE_NODE_ARRAY(node), 
									                             CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}

	adCheckArrays(*this, a)
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] * a[i];
	return tmp;
}

const adouble_array adouble_array::operator *(const real_t v) const
{
	adouble_array tmp;
	if(getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMulti, 
														         CLONE_NODE_ARRAY(node), 
														         shared_ptr<adNodeArray>(new adConstantNodeArray(v, UNITS(node))) ));
	    return tmp;
	}
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] * v;
	return tmp;
}

const adouble_array adouble_array::operator *(const adouble& a) const
{
	adouble_array tmp;
	if(getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMulti, 
														         CLONE_NODE_ARRAY(node), 
														         shared_ptr<adNodeArray>(new adSingleNodeArray(CLONE_NODE(a.node, a.getValue())) ) ));
	    return tmp;
	}
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] * a;
	return tmp;
}

const adouble_array operator *(const adouble& a, const adouble_array& arr)
{
	adouble_array tmp;
	if(arr.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMulti, 
														         shared_ptr<adNodeArray>(new adSingleNodeArray(CLONE_NODE(a.node, a.getValue())) ),
														         CLONE_NODE_ARRAY(arr.node)
																 ));
	    return tmp;
	}
	
	size_t n = arr.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = a * arr[i];
	return tmp;
}

const adouble_array operator *(const real_t v, const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eMulti, 
														         shared_ptr<adNodeArray>(new adConstantNodeArray(v, UNITS(a.node))),
															     CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = v * a[i];
	return tmp;
}

const adouble_array adouble_array::operator /(const adouble_array& a) const
{
	adouble_array tmp;
	if(getGatherInfo() || a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eDivide, 
			                                                     CLONE_NODE_ARRAY(node), 
									                             CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}

	adCheckArrays(*this, a);
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] / a[i];
	return tmp;
}

const adouble_array adouble_array::operator /(const real_t v) const
{
	adouble_array tmp;
	if(getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eDivide, 
														         CLONE_NODE_ARRAY(node), 
														         shared_ptr<adNodeArray>(new adConstantNodeArray(v, UNITS(node))) ));
	    return tmp;
	}
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] / v;
	return tmp;
}

const adouble_array adouble_array::operator /(const adouble& a) const
{
	adouble_array tmp;
	if(getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eDivide, 
														         CLONE_NODE_ARRAY(node), 
														         shared_ptr<adNodeArray>(new adSingleNodeArray(CLONE_NODE(a.node, a.getValue())) ) ));
	    return tmp;
	}
	
	size_t n = GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = (*this)[i] / a;
	return tmp;
}

const adouble_array operator /(const adouble& a, const adouble_array& arr)
{
	adouble_array tmp;
	if(arr.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eDivide, 
														         shared_ptr<adNodeArray>(new adSingleNodeArray(CLONE_NODE(a.node, a.getValue())) ),
														         CLONE_NODE_ARRAY(arr.node)
																 ));
	    return tmp;
	}
	
	size_t n = arr.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = a / arr[i];
	return tmp;
}

const adouble_array operator /(const real_t v, const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adBinaryNodeArray(eDivide, 
														         shared_ptr<adNodeArray>(new adConstantNodeArray(v, UNITS(a.node))),
															     CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = v / a[i];
	return tmp;
}

/*********************************************************************************************
	dt, sum, product, integral, min, max, average
**********************************************************************************************/
const adouble daeModel::dt(const adouble& a) const
{
	adouble tmp;
	tmp.setGatherInfo(true);
	tmp.node = shared_ptr<adNode>(new adSetupExpressionDerivativeNode(const_cast<daeModel*>(this),
																	  CLONE_NODE(a.node, a.getValue()) ));
	return tmp;
}

const adouble daeModel::d(const adouble& a, daeDomain& domain) const
{
	adouble tmp;
	tmp.setGatherInfo(true);
	tmp.node = shared_ptr<adNode>(new adSetupExpressionPartialDerivativeNode(const_cast<daeModel*>(this),
																			 &domain,
																			 CLONE_NODE(a.node, a.getValue()) ));
	return tmp;
}

// Called only during declaring of equations !!!
const adouble daeModel::average(const adouble_array& a) const
{
	adouble tmp;

	tmp.setGatherInfo(true);
	tmp.node = shared_ptr<adNode>(new adSetupSpecialFunctionNode(eAverage, 
																 const_cast<daeModel*>(this),
																 CLONE_NODE_ARRAY(a.node) ));
	return tmp;
}

const adouble daeModel::__average__(const adouble_array& a) const
{
	adouble tmp;

	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>(new adRuntimeSpecialFunctionNode(eAverage,
																	   const_cast<daeModel*>(this),
																	   CLONE_NODE_ARRAY(a.node) ));
		return tmp;
	}

	tmp = __sum__(a) / a.m_arrValues.size();
	return tmp;
}

const adouble daeModel::sum(const adouble_array& a) const
{
	adouble tmp;
	tmp.setGatherInfo(true);
	tmp.node = shared_ptr<adNode>( new adSetupSpecialFunctionNode(eSum, 
																  const_cast<daeModel*>(this),
																  CLONE_NODE_ARRAY(a.node) ) );
	return tmp;
}

const adouble daeModel::__sum__(const adouble_array& a) const
{
	adouble tmp;

	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>(new adRuntimeSpecialFunctionNode(eSum,
																	   const_cast<daeModel*>(this),
																	   CLONE_NODE_ARRAY(a.node) ));
		return tmp;
	}

	tmp = a[0];
	for(size_t i = 1; i < a.GetSize(); i++)
		tmp = tmp + a[i];

	return tmp;
}

const adouble daeModel::product(const adouble_array& a) const
{
	adouble tmp;

	tmp.setGatherInfo(true);
	tmp.node = shared_ptr<adNode>(new adSetupSpecialFunctionNode(eProduct,
																 const_cast<daeModel*>(this),
																 CLONE_NODE_ARRAY(a.node) ));
	return tmp;
}

const adouble daeModel::__product__(const adouble_array& a) const
{
	adouble tmp;

	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>(new adRuntimeSpecialFunctionNode(eProduct,
																	   const_cast<daeModel*>(this),
																	   CLONE_NODE_ARRAY(a.node) ));
		return tmp;
	}

	tmp = a[0];
	for(size_t i = 1; i < a.GetSize(); i++)
		tmp = tmp * a[i];
	return tmp;
}

const adouble daeModel::integral(const adouble_array& a) const
{
	adouble tmp;
	size_t i, nCount;
	daeArrayRange range;
	vector<daeArrayRange> arrRanges;
	daeDomain* pDomain = NULL;

	adNodeArrayImpl* n = dynamic_cast<adNodeArrayImpl*>(a.node.get());
	if(!n)
		daeDeclareAndThrowException(exInvalidPointer);
	n->GetArrayRanges(arrRanges);
	
	nCount = 0;
	for(i = 0; i < arrRanges.size(); i++)
		if(arrRanges[i].m_eType == eRange)
			nCount++;

	if(nCount != 1)
	{
		daeDeclareException(exInvalidCall);
		e << "At the moment, it is possible to calculate one dimensional integrals only, in model [" << GetCanonicalName() << "]";
		throw e;
	}
	
	for(i = 0; i < arrRanges.size(); i++)
	{
		if(arrRanges[i].m_eType == eRange)
		{
			range = arrRanges[i];
			pDomain = range.m_Range.m_pDomain;
			break;
		}
	}

	tmp.setGatherInfo(true);
	adSetupIntegralNode* node = new adSetupIntegralNode(eSingleIntegral,
														const_cast<daeModel*>(this),
														CLONE_NODE_ARRAY(a.node),
														pDomain,
														range);
	tmp.node = shared_ptr<adNode>(node);
	return tmp;
}

const adouble daeModel::__integral__(const adouble_array& a, daeDomain* pDomain, const vector<size_t>& narrPoints) const
{
	adouble tmp;
	size_t i, i1, i2;
	
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		adRuntimeIntegralNode* node = new adRuntimeIntegralNode(eSingleIntegral,
																const_cast<daeModel*>(this),
																CLONE_NODE_ARRAY(a.node),
																pDomain,
																narrPoints);
		tmp.node = shared_ptr<adNode>(node);
		return tmp;
	}
	
	for(i = 0; i < narrPoints.size() - 1; i++)
	{
		i1 = narrPoints[i];
		i2 = narrPoints[i+1];
	
	// In daeDomain operator[] I always return node, not the value
	// Therefore I cannot call here operator[] but function GetPoint
		tmp = tmp + (a[i1] + a[i2]) * ( pDomain->GetPoint(i2) - pDomain->GetPoint(i1) ) / 2;

	// Old code:
	//	tmp = tmp + (a[i1] + a[i2]) * ( (*pDomain)[i2] - (*pDomain)[i1] ) / 2;
	}

	return tmp;
}

const adouble daeModel::min(const adouble_array& a) const
{
	adouble tmp;

	tmp.setGatherInfo(true);
	tmp.node = shared_ptr<adNode>(new adSetupSpecialFunctionNode(eMinInArray, 
																 const_cast<daeModel*>(this),
																 CLONE_NODE_ARRAY(a.node) ));
	return tmp;
}

const adouble daeModel::__min__(const adouble_array& a) const
{
	adouble tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>(new adRuntimeSpecialFunctionNode(eMinInArray,
																	   const_cast<daeModel*>(this),
																	   CLONE_NODE_ARRAY(a.node) ));
		return tmp;
	}
		
//	tmp = a[0];
//	for(size_t i = 1; i < a.GetSize(); i++)
//		if(a[i].getValue() < tmp.getValue())
//			tmp = a[i];
//	return tmp;

	tmp = a[0];
	for(size_t i = 1; i < a.GetSize(); i++)
		tmp = dae::core::__min__(tmp, a[i]);		
	return tmp;
}

const adouble daeModel::max(const adouble_array& a) const
{
	adouble tmp;

	tmp.setGatherInfo(true);
	tmp.node = shared_ptr<adNode>(new adSetupSpecialFunctionNode(eMaxInArray, 
																 const_cast<daeModel*>(this),
																 CLONE_NODE_ARRAY(a.node) ));
	return tmp;
}

const adouble daeModel::__max__(const adouble_array& a) const
{
	adouble tmp;

	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>(new adRuntimeSpecialFunctionNode(eMaxInArray,
																	   const_cast<daeModel*>(this),
																	   CLONE_NODE_ARRAY(a.node) ));
		return tmp;
	}
	
//	tmp = a[0];
//	for(size_t i = 1; i < a.GetSize(); i++)
//		if(a[i].getValue() > tmp.getValue())
//			tmp = a[i];
//	return tmp;

	tmp = a[0];
	for(size_t i = 1; i < a.GetSize(); i++)
		tmp = dae::core::__max__(tmp, a[i]);
	return tmp;
}

const adouble daeModel::time(void) const
{
	adouble tmp;
	tmp.setGatherInfo(true);
	tmp.node = shared_ptr<adNode>(new adTimeNode());
	return tmp;
}

const adouble daeModel::constant(const quantity& q) const
{
	adouble tmp;
	tmp.setGatherInfo(true);
	tmp.node = shared_ptr<adNode>(new adConstantNode(q));
	return tmp;
}

const adouble_array exp(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eExp, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = exp(a[i]);
	return tmp;
}

const adouble_array sqrt(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eSqrt, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = sqrt(a[i]);
	return tmp;
}

const adouble_array log(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eLn, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = log(a[i]);
	return tmp;
}

const adouble_array log10(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eLog, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = log10(a[i]);
	return tmp;
}

const adouble_array abs(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eAbs, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = abs(a[i]);
	return tmp;
}

const adouble_array floor(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eFloor, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = floor(a[i]);
	return tmp;
}

const adouble_array ceil(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eCeil, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = ceil(a[i]);
	return tmp;
}

const adouble_array sin(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eSin, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = sin(a[i]);
	return tmp;
}

const adouble_array cos(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eCos, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = cos(a[i]);
	return tmp;
}

const adouble_array tan(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eTan, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = tan(a[i]);
	return tmp;
}

const adouble_array asin(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eArcSin, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = asin(a[i]);
	return tmp;
}

const adouble_array acos(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eArcCos, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = acos(a[i]);
	return tmp;
}

const adouble_array atan(const adouble_array& a)
{
	adouble_array tmp;
	if(a.getGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>(new adUnaryNodeArray(eArcTan, 
			                                                    CLONE_NODE_ARRAY(a.node) ));
	    return tmp;
	}
	
	size_t n = a.GetSize();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = atan(a[i]);
	return tmp;
}


}
}
