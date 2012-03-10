#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{
/******************************************************************************
	daeArrayRange
*******************************************************************************/
daeArrayRange::daeArrayRange(void) 
		: m_eType(eRaTUnknown)
{
}

daeArrayRange::daeArrayRange(daeDomainIndex domainIndex) 
		: m_eType(eRangeDomainIndex), m_domainIndex(domainIndex)
{
}

daeArrayRange::daeArrayRange(daeIndexRange range) 
		: m_eType(eRange), m_Range(range)
{
}

size_t daeArrayRange::GetNoPoints(void) const
{
	if(m_eType == eRangeDomainIndex)
	{
		return 1;
	}
	else if(m_eType == eRange)
	{
		std::vector<size_t> narrPoints;
		m_Range.GetPoints(narrPoints);
		return narrPoints.size();
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	return 0;
}

void daeArrayRange::GetPoints(vector<size_t>& narrPoints) const
{
	narrPoints.clear();
	
	if(m_eType == eRangeDomainIndex)
	{
		narrPoints.push_back(m_domainIndex.GetCurrentIndex());
	}
	else if(m_eType == eRange)
	{
		m_Range.GetPoints(narrPoints);
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}

string daeArrayRange::GetRangeAsString(void) const
{
	if(m_eType == eRangeDomainIndex)
	{
		return m_domainIndex.GetIndexAsString();
	}
	else if(m_eType == eRange)
	{
		return m_Range.ToString();
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	return string("?");
}

void daeArrayRange::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void daeArrayRange::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Type";
	SaveEnum(pTag, strName, m_eType);

	if(m_eType == eRangeDomainIndex)
	{
		strName = "DomainIndex";
		pTag->SaveObject(strName, &m_domainIndex);
	}
	else if(m_eType == eRange)
	{
		strName = "Range";
		pTag->SaveObject(strName, &m_Range);
	}
	else
	{
		daeDeclareAndThrowException(exXMLIOError)
	}
}
	
/******************************************************************************
	daeIndexRange
*******************************************************************************/
daeIndexRange::daeIndexRange(void)
{
	m_eType		  = eIRTUnknown;
	m_pDomain     = NULL;
	m_iStartIndex = 0;
	m_iEndIndex   = 0;
	m_iStride     = 0;
}

daeIndexRange::daeIndexRange(daeDomain* pDomain)
{
	if(!pDomain)
		daeDeclareAndThrowException(exInvalidCall);
	
	m_eType		  = eAllPointsInDomain;
	m_pDomain     = pDomain;
	m_iStartIndex = 0;
	m_iEndIndex   = 0;
	m_iStride     = 0;
}

daeIndexRange::daeIndexRange(daeDomain* pDomain, const vector<size_t>& narrCustomPoints)
{
	if(!pDomain)
		daeDeclareAndThrowException(exInvalidCall);
	if(narrCustomPoints.size() == 0)
	{
		daeDeclareException(exInvalidCall);
		e << "daeIndexRange list of indexes is empty";
		throw e;
	}

	m_eType				= eCustomRange;
	m_pDomain			= pDomain;
	m_narrCustomPoints	= narrCustomPoints;
	m_iStartIndex		= 0;
	m_iEndIndex			= 0;
	m_iStride			= 0;
}

daeIndexRange::daeIndexRange(daeDomain* pDomain, 
							 int iStartIndex, 
							 int iEndIndex, 
							 int iStride)
{
	if(!pDomain)
		daeDeclareAndThrowException(exInvalidCall);
	
	m_eType		  = eRangeOfIndexes;
	m_pDomain     = pDomain;
	m_iStartIndex = iStartIndex;
	m_iEndIndex   = iEndIndex;
	m_iStride     = iStride;
	
	if(m_iStartIndex == 0 && m_iEndIndex == 0)
		m_iEndIndex = -1;
}

void daeIndexRange::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void daeIndexRange::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Type";
	SaveEnum(pTag, strName, m_eType);

	strName = "Domain";
	pTag->SaveObjectRef(strName, m_pDomain);

	if(m_eType == eAllPointsInDomain)
	{
	}
	else if(m_eType == eRangeOfIndexes)
	{
		strName = "StartIndex";
		pTag->Save(strName, m_iStartIndex);

		strName = "EndIndex";
		pTag->Save(strName, m_iEndIndex);

		strName = "Stride";
		pTag->Save(strName, m_iStride);
	}
	else if(m_eType == eCustomRange)
	{
		strName = "Indexes";
		pTag->SaveArray(strName, m_narrCustomPoints);
	}
	else
	{
		daeDeclareAndThrowException(exXMLIOError)
	}
}

size_t daeIndexRange::GetNoPoints(void) const
{
	vector<size_t> narrPoints;
	GetPoints(narrPoints);
	return narrPoints.size();
}

void daeIndexRange::GetPoints(vector<size_t>& narrCustomPoints) const
{
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);

	narrCustomPoints.clear();
	if(m_eType == eAllPointsInDomain)
	{
		size_t n = m_pDomain->GetNumberOfPoints();
		narrCustomPoints.resize(n);
		for(size_t i = 0; i < n; i++)
			narrCustomPoints[i] = i;
	}
	else if(m_eType == eRangeOfIndexes)
	{
		if(m_iStartIndex < 0 || m_iStartIndex >= (int)m_pDomain->GetNumberOfPoints())
			daeDeclareAndThrowException(exOutOfBounds);
		if(m_iEndIndex >= (int)m_pDomain->GetNumberOfPoints())
			daeDeclareAndThrowException(exOutOfBounds);
		
		int iEnd = (m_iEndIndex == -1 ? m_pDomain->GetNumberOfPoints() : m_iEndIndex);

		narrCustomPoints.clear();
		for(int i = m_iStartIndex; i < iEnd; i += m_iStride)
			narrCustomPoints.push_back((size_t)i);
		
		if(narrCustomPoints.empty())
		{
			daeDeclareException(exInvalidCall);
			e << "daeIndexRange slice size is zero";
			throw e;
		}
	}
	else if(m_eType == eCustomRange)
	{
		if(m_narrCustomPoints.size() == 0)
			daeDeclareAndThrowException(exInvalidCall);

		narrCustomPoints = m_narrCustomPoints;
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
}

string daeIndexRange::ToString(void) const
{
	if(m_eType == eAllPointsInDomain)
	{
		return string("*"); 
	}
	else if(m_eType == eRangeOfIndexes)
	{
		return (toString<int>(m_iStartIndex) + 
			   string(":") + 
			   toString<int>(m_iEndIndex) + 
			   string(";") + 
			   toString<int>(m_iStride));
	}
	else if(m_eType == eCustomRange)
	{
		string strItems;
		strItems = "[";
		for(size_t i = 0; i < m_narrCustomPoints.size(); i++)
		{
			if(i != 0)
				strItems += ", ";
			strItems += toString<size_t>(m_narrCustomPoints[i]);
		}
		strItems += "]";
		return strItems;
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
}


/******************************************************************
	daeDomainIndex
*******************************************************************/
daeDomainIndex::daeDomainIndex(void) 
		: m_eType(eDITUnknown), m_nIndex(ULONG_MAX), m_pDEDI(NULL), m_iIncrement(0)
{
}

daeDomainIndex::daeDomainIndex(size_t nIndex) 
		: m_eType(eConstantIndex), m_nIndex(nIndex), m_pDEDI(NULL), m_iIncrement(0)
{
}

daeDomainIndex::daeDomainIndex(daeDistributedEquationDomainInfo* pDEDI) 
		: m_eType(eDomainIterator), m_nIndex(ULONG_MAX), m_pDEDI(pDEDI), m_iIncrement(0)
{
}

daeDomainIndex::daeDomainIndex(daeDistributedEquationDomainInfo* pDEDI, int iIncrement)
		: m_eType(eIncrementedDomainIterator), m_nIndex(ULONG_MAX), m_pDEDI(pDEDI), m_iIncrement(iIncrement)
{
}

size_t daeDomainIndex::GetCurrentIndex(void) const
{
	if(m_eType == eConstantIndex)
	{
		if(m_nIndex == ULONG_MAX)
			daeDeclareAndThrowException(exInvalidCall);
		return m_nIndex;
	}
	else if(m_eType == eDomainIterator)
	{
		if(!m_pDEDI)
			daeDeclareAndThrowException(exInvalidPointer);
		return m_pDEDI->GetCurrentIndex();
	}
	else if(m_eType == eIncrementedDomainIterator)
	{
		if(!m_pDEDI)
			daeDeclareAndThrowException(exInvalidPointer);
		return m_pDEDI->GetCurrentIndex() + m_iIncrement;
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	
	return ULONG_MAX;
}

string daeDomainIndex::GetIndexAsString(void) const
{
	if(m_eType == eConstantIndex)
	{
		return toString<size_t>(m_nIndex);
	}
	else if(m_eType == eDomainIterator)
	{
		if(!m_pDEDI)
			daeDeclareAndThrowException(exInvalidPointer);
		return m_pDEDI->GetName();
	}
	else if(m_eType == eIncrementedDomainIterator)
	{
		if(!m_pDEDI)
			daeDeclareAndThrowException(exInvalidPointer);
		return (m_pDEDI->GetName() + (m_iIncrement >= 0 ? "+" : "") + toString<int>(m_iIncrement));
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	
	return string("?");
}

void daeDomainIndex::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)

	string strName;

	strName = "Type";
	OpenEnum(pTag, strName, m_eType);

	if(m_eType == eConstantIndex)
	{
		strName = "Index";
		pTag->Open(strName, m_nIndex);
	}
	else if(m_eType == eDomainIterator)
	{
		//strName = "DEDI";
		//m_pDEDI = pTag->OpenObjectRef(strName);
	}
	else if(m_eType == eIncrementedDomainIterator)
	{
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented)
	}
}

void daeDomainIndex::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Type";
	SaveEnum(pTag, strName, m_eType);

	if(m_eType == eConstantIndex)
	{
		strName = "Index";
		pTag->Save(strName, m_nIndex);
	}
	else if(m_eType == eDomainIterator)
	{
		strName = "DEDI";
		pTag->SaveObjectRef(strName, m_pDEDI);
	}
	else if(m_eType == eIncrementedDomainIterator)
	{
		strName = "DEDI";
		pTag->SaveObjectRef(strName, m_pDEDI);
		
		strName = "Increment";
		pTag->Save(strName, m_iIncrement);
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented)
	}
}


}
}
