#include "stdafx.h"
#include "coreimpl.h"
#include "nodes_array.h"
#include "units_io.h"

namespace dae 
{
namespace core 
{
/******************************************************************
	daeParameter
*******************************************************************/
daeParameter::daeParameter(void)
{
	m_bReportingOn = false;
	m_pModel       = NULL;
	m_pParentPort  = NULL;
}
	
daeParameter::daeParameter(string strName, const unit& units, daeModel* pModel, string strDescription, 
						   daeDomain* d1, daeDomain* d2, daeDomain* d3, daeDomain* d4, daeDomain* d5, daeDomain* d6, daeDomain* d7, daeDomain* d8)
{
	m_bReportingOn = false;
	m_Unit         = units;
	m_pModel       = pModel;
	m_pParentPort  = NULL;

	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	pModel->AddParameter(*this, strName, units, strDescription);
	
	m_ptrDomains = dae::makeVector<daeDomain*>(d1, d2, d3, d4, d5, d6, d7, d8);
}
	
daeParameter::daeParameter(string strName, const unit& units, daePort* pPort, string strDescription, 
						   daeDomain* d1, daeDomain* d2, daeDomain* d3, daeDomain* d4, daeDomain* d5, daeDomain* d6, daeDomain* d7, daeDomain* d8)
{
	m_bReportingOn = false;
	m_Unit         = units;
	m_pModel       = NULL;
	m_pParentPort  = pPort;

	if(!pPort)
		daeDeclareAndThrowException(exInvalidPointer);
	pPort->AddParameter(*this, strName, units, strDescription);
	
	m_ptrDomains = dae::makeVector<daeDomain*>(d1, d2, d3, d4, d5, d6, d7, d8);
}

daeParameter::~daeParameter(void)
{
}

void daeParameter::Clone(const daeParameter& rObject)
{
	m_Unit           = rObject.m_Unit;
	m_darrValues	 = rObject.m_darrValues;
	FindDomains(rObject.m_ptrDomains, m_ptrDomains, m_pModel);
}

void daeParameter::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);	
	
	m_ptrDomains.clear();
	m_darrValues.clear();

	daeObject::Open(pTag);

//	strName = "Units";
//	OpenEnum(pTag, strName, m_Unit);

	strName = "DomainRefs";
	daeFindDomainByID del(m_pModel);
	pTag->OpenObjectRefArray(strName, m_ptrDomains, &del);
}

void daeParameter::Save(io::xmlTag_t* pTag) const
{
	string strName, strValue;

	daeObject::Save(pTag);

	strName = "Units";
	units::Save(pTag, strName, m_Unit);
	
	strName = "MathMLUnits";
	io::xmlTag_t* pChildTag = pTag->AddTag(strName);

	strName = "math";
	io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);

	strName = "xmlns";
	strValue = "http://www.w3.org/1998/Math/MathML";
	pMathMLTag->AddAttribute(strName, strValue);
	
	units::SaveAsPresentationMathML(pMathMLTag, m_Unit);

	strName = "DomainRefs";
	pTag->SaveObjectRefArray(strName, m_ptrDomains);
}

void daeParameter::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport, strDomains;
	boost::format fmtFile(strExport);

	if(c.m_bExportDefinition)
	{
		if(eLanguage == ePYDAE)
		{
		}
		else if(eLanguage == eCDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "daeParameter %1%;\n";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName();
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}		
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			if(!m_ptrDomains.empty())
				strDomains = ", [" + toString_StrippedRelativeNames<daeDomain*, daeModel*>(m_ptrDomains, m_pModel, "self.") + "]";
			
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.%1% = daeParameter(\"%2%\", %3%, self, \"%4%\"%5%)\n";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% m_strShortName 
					% m_Unit.toString() 
					% m_strDescription
					% strDomains;
		}
		else if(eLanguage == eCDAE)
		{
			if(!m_ptrDomains.empty())
				strDomains = ", " + toString_StrippedRelativeNames<daeDomain*, daeModel*>(m_ptrDomains, m_pModel, "&");

			strExport = ",\n" + c.CalculateIndent(c.m_nPythonIndentLevel) + "%1%(\"%2%\", %3%, this, \"%4%\"%5%)";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% m_strShortName 
					% m_Unit.toString() 
					% m_strDescription
					% strDomains;
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += fmtFile.str();
}

void daeParameter::OpenRuntime(io::xmlTag_t* pTag)
{
//	string strName;

//	if(!m_pModel)
//		daeDeclareAndThrowException(exInvalidPointer);	
	
//	m_ptrDomains.clear();
//	m_darrValues.clear();

//	daeObject::OpenRuntime(pTag);

//	strName = "Type";
//	OpenEnum(pTag, strName, m_eParameterType);

//	strName = "DomainRefs";
//	daeFindDomainByID del(m_pModel);
//	pTag->OpenObjectRefArray(strName, m_ptrDomains, &del);

//	strName = "Values";
//	pTag->OpenArray(strName, m_darrValues);
}

void daeParameter::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName, strValue;

	daeObject::SaveRuntime(pTag);

	strName = "Units";
	units::Save(pTag, strName, m_Unit);
	
	strName = "MathMLUnits";
	io::xmlTag_t* pChildTag = pTag->AddTag(strName);

	strName = "math";
	io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);

	strName = "xmlns";
	strValue = "http://www.w3.org/1998/Math/MathML";
	pMathMLTag->AddAttribute(strName, strValue);
	
	units::SaveAsPresentationMathML(pMathMLTag, m_Unit);

	strName = "DomainRefs";
	pTag->SaveObjectRefArray(strName, m_ptrDomains);

	strName = "Values";
	pTag->SaveArray(strName, m_darrValues);
}

string daeParameter::GetCanonicalName(void) const
{
	if(m_pParentPort)
		return m_pParentPort->GetCanonicalName() + '.' + m_strShortName;
	else
		return daeObject::GetCanonicalName();
}

bool daeParameter::GetReportingOn(void) const
{
	return m_bReportingOn;
}

void daeParameter::SetReportingOn(bool bOn)
{
	m_bReportingOn = bOn;
}

void daeParameter::Initialize(void)
{
	vector<daeDomain*>::size_type i;
	daeDomain* pDomain;

	m_darrValues.clear();
	size_t nTotalNumberOfPoints = 1;
	for(i = 0; i < m_ptrDomains.size(); i++)
	{
		pDomain = m_ptrDomains[i];
		if(!pDomain)
			daeDeclareAndThrowException(exInvalidPointer);
		if(pDomain->GetNumberOfPoints() == 0)
		{
			daeDeclareException(exInvalidCall);
			e << "Number of points in domain [" << pDomain->GetCanonicalName() << "] in parameter [" << GetCanonicalName() << "] must not be zero; did you forget to initialize it?";
			throw e;
		}
		nTotalNumberOfPoints *= pDomain->GetNumberOfPoints();
	}
	m_darrValues.resize(nTotalNumberOfPoints);

// Create stock runtime nodes (adRuntimeParameterNode)
	//m_ptrarrRuntimeNodes.clear();
	//size_t ND = m_ptrDomains.size();
	//size_t* indexes = new size_t[ND];
	//SetIndexes(indexes, ND, 0);
}

//adouble daeParameter::PreCreateRuntimeNodes()
//{
//	size_t nTotalNumberOfPoints = m_darrValues.size();
	
//	m_ptrarrRuntimeNodes.resize(nTotalNumberOfPoints);
//	for(size_t i = 0; i < nTotalNumberOfPoints; i++)
//		m_ptrarrRuntimeNodes[i] = adNodePtr(new adRuntimeParameterNode(m_darrValues[i]));
//}

adouble daeParameter::Create_adouble(const size_t* indexes, const size_t N) const
{
	adouble tmp;
	size_t nIndex;

	if(!indexes)
		nIndex = 0;
	else
		nIndex = CalculateIndex(indexes, N);

	tmp.setValue(m_darrValues[nIndex]);
	tmp.setDerivative(0);

	if(m_pModel->m_pDataProxy->GetGatherInfo())
	{
		adRuntimeParameterNode* node = new adRuntimeParameterNode();
		node->m_pParameter = const_cast<daeParameter*>(this);
		node->m_dValue = m_darrValues[nIndex];
		if(N > 0)
		{
			node->m_narrDomains.resize(N);
			for(size_t i = 0; i < N; i++)
				node->m_narrDomains[i] = indexes[i];
		}
		tmp.node = adNodePtr(node);
		tmp.setGatherInfo(true);
	}
	return tmp;
}

adouble daeParameter::CreateSetupParameter(const daeDomainIndex* indexes, const size_t N) const
{
	adouble tmp;
	adSetupParameterNode* node = new adSetupParameterNode();
	node->m_pParameter = const_cast<daeParameter*>(this);

// Check if domains in indexes correspond to domains here
	for(size_t i = 0; i < N; i++)
	{
		if(indexes[i].m_eType == eDomainIterator ||
		   indexes[i].m_eType == eIncrementedDomainIterator)
		{
			if(m_ptrDomains[i] != indexes[i].m_pDEDI->m_pDomain)
			{
				daeDeclareException(exInvalidCall);
				e << "You cannot create daeDomainIndex with the domain [" << indexes[i].m_pDEDI->m_pDomain->GetCanonicalName() 
				  << "]; you must use domain [" << m_ptrDomains[i]->GetCanonicalName() << "] as " << i+1 << ". index argument "
				  << "in parameter [" << GetCanonicalName() << "] in operator()";
				throw e;
			}
		}
	}
	
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

void daeParameter::Fill_adouble_array(vector<adouble>& arrValues, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const
{
	if(currentN == N) // create and add adouble to the vector
	{
		//arrValues.push_back(Create_adouble(indexes, N));
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

adouble_array daeParameter::Create_adouble_array(const daeArrayRange* ranges, const size_t N) const
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
//		adRuntimeParameterNodeArray* node = new adRuntimeParameterNodeArray();
//		varArray.node = adNodeArrayPtr(node);
//		varArray.setGatherInfo(true);
//		node->m_pParameter = const_cast<daeParameter*>(this);
//	
//		size_t size = varArray.m_arrValues.size();
//		if(size == 0)
//			daeDeclareAndThrowException(exInvalidCall); 
//		
//		node->m_ptrarrParameterNodes.resize(size);
//		for(size_t i = 0; i < size; i++)
//			node->m_ptrarrParameterNodes[i] = varArray.m_arrValues[i].node;
//		node->m_arrRanges.resize(N);
//		for(size_t i = 0; i < N; i++)
//			node->m_arrRanges[i] = ranges[i];
//	}
	return varArray;
}

adouble_array daeParameter::CreateSetupParameterArray(const daeArrayRange* ranges, const size_t N) const
{
	adouble_array varArray;

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
					daeDeclareException(exInvalidCall);
					e << "You cannot create daeArrayRange with the domain [" << ranges[i].m_domainIndex.m_pDEDI->m_pDomain->GetCanonicalName() 
					  << "]; you must use the domain [" << m_ptrDomains[i]->GetCanonicalName() << "] as " << i+1 << ". range argument "
					  << "in parameter [" << GetCanonicalName() << "] in function array()";
					throw e;
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
				  << "in parameter [" << GetCanonicalName() << "] in function array()";
				throw e;
			}
		}
	}

	adSetupParameterNodeArray* node = new adSetupParameterNodeArray();
	varArray.node = adNodeArrayPtr(node);
	varArray.setGatherInfo(true);

	node->m_pParameter = const_cast<daeParameter*>(this);
	node->m_arrRanges.resize(N);
	for(size_t i = 0; i < N; i++)
		node->m_arrRanges[i] = ranges[i];

	return varArray;
}

daeDomain* daeParameter::GetDomain(size_t nIndex) const
{
	if(nIndex >= m_ptrDomains.size())
	{
		daeDeclareException(exInvalidCall);
		e << "Invalid domain index [" << nIndex << "] in parameter [" << GetCanonicalName() << "]";
		throw e;
	}
	
	return m_ptrDomains[nIndex];
}

unit daeParameter::GetUnits(void) const
{
	return m_Unit;
}

void daeParameter::SetUnits(const unit& units)
{
	m_Unit = units;
}

void daeParameter::GetDomains(vector<daeDomain_t*>& ptrarrDomains)
{
	ptrarrDomains.clear();
	for(size_t i = 0; i < m_ptrDomains.size(); i++)
		ptrarrDomains.push_back(m_ptrDomains[i]);
}

void daeParameter::DistributeOnDomain(daeDomain& rDomain)
{
	dae_push_back(m_ptrDomains, &rDomain);
}

real_t* daeParameter::GetValuePointer(void)
{
	return &m_darrValues[0];
}

size_t daeParameter::GetNumberOfPoints(void) const
{
	if(m_darrValues.empty())
	{	
		daeDeclareException(exInvalidCall); 
		e << "Number of points in the parameter [" << GetCanonicalName() << "] must not be zero; did you forget to initialize it?";
		throw e;
	}
	return m_darrValues.size();
}

size_t daeParameter::CalculateIndex(const size_t* indexes, const size_t N) const
{
	size_t		i, j, nIndex, temp;
	daeDomain	*pDomain;

	if(m_ptrDomains.size() != N)
	{	
		daeDeclareException(exInvalidCall);
		e << "Illegal number of domains, parameter " << GetCanonicalName();
		throw e;
	}

// Check the pointers and the bounds first
	for(i = 0; i < N; i++)
	{
		pDomain = m_ptrDomains[i];
		if(!pDomain)
			daeDeclareAndThrowException(exInvalidPointer);
		if(indexes[i] >= pDomain->GetNumberOfPoints())
			daeDeclareAndThrowException(exOutOfBounds);
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

void daeParameter::GetDomainsIndexesMap(std::map<size_t, std::vector<size_t> >& mapDomainsIndexes, size_t nIndexBase) const
{
    std::vector<size_t> narrDomainIndexes;
    size_t d1, d2, d3, d4, d5, d6, d7, d8;
	daeDomain *pDomain1, *pDomain2, *pDomain3, 
		      *pDomain4, *pDomain5, *pDomain6,
			  *pDomain7, *pDomain8;

    size_t nNoDomains    = m_ptrDomains.size();
    size_t nIndexCounter = 0;
    
    if(nNoDomains == 0)
    {
        narrDomainIndexes.clear();
        mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
    }
    else if(nNoDomains == 1)
    {
        pDomain1 = m_ptrDomains[0];
        if(!pDomain1)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(1);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 2)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        if(!pDomain1 || !pDomain2)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(2);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 3)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        if(!pDomain1 || !pDomain2 || !pDomain3)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(3);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 4)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(4);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 5)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        pDomain5 = m_ptrDomains[4];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4 || !pDomain5)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(5);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        for(d5 = 0; d5 < pDomain5->m_nNumberOfPoints; d5++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            narrDomainIndexes[4] = nIndexBase + d5;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 6)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        pDomain5 = m_ptrDomains[4];
        pDomain6 = m_ptrDomains[5];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4 || 
           !pDomain5 || !pDomain6)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(6);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        for(d5 = 0; d5 < pDomain5->m_nNumberOfPoints; d5++)
        for(d6 = 0; d6 < pDomain6->m_nNumberOfPoints; d6++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            narrDomainIndexes[4] = nIndexBase + d5;
            narrDomainIndexes[5] = nIndexBase + d6;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 7)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        pDomain5 = m_ptrDomains[4];
        pDomain6 = m_ptrDomains[5];
        pDomain7 = m_ptrDomains[6];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4 || 
           !pDomain5 || !pDomain6 || !pDomain7)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(7);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        for(d5 = 0; d5 < pDomain5->m_nNumberOfPoints; d5++)
        for(d6 = 0; d6 < pDomain6->m_nNumberOfPoints; d6++)
        for(d7 = 0; d7 < pDomain7->m_nNumberOfPoints; d7++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            narrDomainIndexes[4] = nIndexBase + d5;
            narrDomainIndexes[5] = nIndexBase + d6;
            narrDomainIndexes[6] = nIndexBase + d7;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else if(nNoDomains == 8)
    {
        pDomain1 = m_ptrDomains[0];
        pDomain2 = m_ptrDomains[1];
        pDomain3 = m_ptrDomains[2];
        pDomain4 = m_ptrDomains[3];
        pDomain5 = m_ptrDomains[4];
        pDomain6 = m_ptrDomains[5];
        pDomain7 = m_ptrDomains[6];
        pDomain8 = m_ptrDomains[7];
        if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4 || 
           !pDomain5 || !pDomain6 || !pDomain7 || !pDomain8)
            daeDeclareAndThrowException(exInvalidPointer);

        narrDomainIndexes.resize(8);
        for(d1 = 0; d1 < pDomain1->m_nNumberOfPoints; d1++)
        for(d2 = 0; d2 < pDomain2->m_nNumberOfPoints; d2++)
        for(d3 = 0; d3 < pDomain3->m_nNumberOfPoints; d3++)
        for(d4 = 0; d4 < pDomain4->m_nNumberOfPoints; d4++)
        for(d5 = 0; d5 < pDomain5->m_nNumberOfPoints; d5++)
        for(d6 = 0; d6 < pDomain6->m_nNumberOfPoints; d6++)
        for(d7 = 0; d7 < pDomain7->m_nNumberOfPoints; d7++)
        for(d8 = 0; d8 < pDomain8->m_nNumberOfPoints; d8++)
        {
            narrDomainIndexes[0] = nIndexBase + d1;
            narrDomainIndexes[1] = nIndexBase + d2;
            narrDomainIndexes[2] = nIndexBase + d3;
            narrDomainIndexes[3] = nIndexBase + d4;
            narrDomainIndexes[4] = nIndexBase + d5;
            narrDomainIndexes[5] = nIndexBase + d6;
            narrDomainIndexes[6] = nIndexBase + d7;
            narrDomainIndexes[7] = nIndexBase + d8;
            mapDomainsIndexes[nIndexCounter] = narrDomainIndexes;
            nIndexCounter++;
        }
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }
}

void daeParameter::SetValues(real_t values)
{
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	m_darrValues.assign(m_darrValues.size(), values);
}

void daeParameter::SetValue(real_t value)
{
	if(m_ptrDomains.size() != 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 0; it should be " << m_ptrDomains.size();
		throw e;
	}
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	m_darrValues[0] = value;
}

void daeParameter::SetValue(size_t nDomain1, real_t value)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 1; it should be " << m_ptrDomains.size();
		throw e;
	}
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	size_t indexes[1] = {nDomain1};
	m_darrValues[CalculateIndex(indexes, 1)] = value;
}

void daeParameter::SetValue(size_t nDomain1, size_t nDomain2, real_t value)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 2; it should be " << m_ptrDomains.size();
		throw e;
	}
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	size_t indexes[2] = {nDomain1, nDomain2};
	m_darrValues[CalculateIndex(indexes, 2)] = value;
}

void daeParameter::SetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, real_t value)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 3; it should be " << m_ptrDomains.size();
		throw e;
	}
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	size_t indexes[3] = {nDomain1, nDomain2, nDomain3};
	m_darrValues[CalculateIndex(indexes, 3)] = value;
}

void daeParameter::SetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, real_t value)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 4; it should be " << m_ptrDomains.size();
		throw e;
	}
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	size_t indexes[4] = {nDomain1, nDomain2, nDomain3, nDomain4};
	m_darrValues[CalculateIndex(indexes, 4)] = value;
}

void daeParameter::SetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, real_t value)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 5; it should be " << m_ptrDomains.size();
		throw e;
	}
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	size_t indexes[5] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5};
	m_darrValues[CalculateIndex(indexes, 5)] = value;
}

void daeParameter::SetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, real_t value)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 6; it should be " << m_ptrDomains.size();
		throw e;
	}
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	size_t indexes[6] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6};
	m_darrValues[CalculateIndex(indexes, 6)] = value;
}

void daeParameter::SetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7, real_t value)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 7; it should be " << m_ptrDomains.size();
		throw e;
	}
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	size_t indexes[7] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6, nDomain7};
	m_darrValues[CalculateIndex(indexes, 7)] = value;
}

void daeParameter::SetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7, size_t nDomain8, real_t value)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 8; it should be " << m_ptrDomains.size();
		throw e;
	}
// If not previously initialized, do it now
	if(m_darrValues.size() == 0)
		Initialize();

	size_t indexes[8] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6, nDomain7, nDomain8};
	m_darrValues[CalculateIndex(indexes, 8)] = value;
}

real_t daeParameter::GetValue(void)
{
	if(m_ptrDomains.size() != 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 0; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << GetCanonicalName() << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	return m_darrValues[0];
}

real_t daeParameter::GetValue(size_t nDomain1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 1; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << GetCanonicalName() << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[1] = {nDomain1};
	return m_darrValues[CalculateIndex(indexes, 1)];
}

real_t daeParameter::GetValue(size_t nDomain1, size_t nDomain2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 2; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << GetCanonicalName() << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[2] = {nDomain1, nDomain2};
	return m_darrValues[CalculateIndex(indexes, 2)];
}

real_t daeParameter::GetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 3; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << GetCanonicalName() << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[3] = {nDomain1, nDomain2, nDomain3};
	return m_darrValues[CalculateIndex(indexes, 3)];
}

real_t daeParameter::GetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 4; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << GetCanonicalName() << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[4] = {nDomain1, nDomain2, nDomain3, nDomain4};
	return m_darrValues[CalculateIndex(indexes, 4)];
}

real_t daeParameter::GetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 5; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << GetCanonicalName() << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[5] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5};
	return m_darrValues[CalculateIndex(indexes, 5)];
}

real_t daeParameter::GetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 6; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << GetCanonicalName() << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[6] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6};
	return m_darrValues[CalculateIndex(indexes, 6)];
}

real_t daeParameter::GetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 7; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << GetCanonicalName() << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[7] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6, nDomain7};
	return m_darrValues[CalculateIndex(indexes, 7)];
}

real_t daeParameter::GetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7, size_t nDomain8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << GetCanonicalName() << "]" << "Number of domains is 8; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << GetCanonicalName() << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[8] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6, nDomain7, nDomain8};
	return m_darrValues[CalculateIndex(indexes, 8)];
}


void daeParameter::SetValues(const quantity& q)
{
	real_t values = q.scaleTo(m_Unit).getValue();
	SetValues(values);
}

void daeParameter::SetValue(const quantity& q)
{
	real_t value = q.scaleTo(m_Unit).getValue();
	SetValue(value);
}

void daeParameter::SetValue(size_t nD1, const quantity& q)
{
	real_t value = q.scaleTo(m_Unit).getValue();
	SetValue(nD1, value);
}

void daeParameter::SetValue(size_t nD1, size_t nD2, const quantity& q)
{
	real_t value = q.scaleTo(m_Unit).getValue();
	SetValue(nD1, nD2, value);
}

void daeParameter::SetValue(size_t nD1, size_t nD2, size_t nD3, const quantity& q)
{
	real_t value = q.scaleTo(m_Unit).getValue();
	SetValue(nD1, nD2, nD3, value);
}

void daeParameter::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& q)
{
	real_t value = q.scaleTo(m_Unit).getValue();
	SetValue(nD1, nD2, nD3, nD4, value);
}

void daeParameter::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& q)
{
	real_t value = q.scaleTo(m_Unit).getValue();
	SetValue(nD1, nD2, nD3, nD4, nD5, value);
}

void daeParameter::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& q)
{
	real_t value = q.scaleTo(m_Unit).getValue();
	SetValue(nD1, nD2, nD3, nD4, nD5, nD6, value);
}

void daeParameter::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& q)
{
	real_t value = q.scaleTo(m_Unit).getValue();
	SetValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, value);
}

void daeParameter::SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& q)
{
	real_t value = q.scaleTo(m_Unit).getValue();
	SetValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8, value);
}

quantity daeParameter::GetQuantity(void)
{
	real_t value = GetValue();
	return quantity(value, m_Unit);
}

quantity daeParameter::GetQuantity(size_t nD1)
{
	real_t value = GetValue(nD1);
	return quantity(value, m_Unit);
}

quantity daeParameter::GetQuantity(size_t nD1, size_t nD2)
{
	real_t value = GetValue(nD1, nD2);
	return quantity(value, m_Unit);
}

quantity daeParameter::GetQuantity(size_t nD1, size_t nD2, size_t nD3)
{
	real_t value = GetValue(nD1, nD2, nD3);
	return quantity(value, m_Unit);
}

quantity daeParameter::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4);
	return quantity(value, m_Unit);
}

quantity daeParameter::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4, nD5);
	return quantity(value, m_Unit);
}

quantity daeParameter::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4, nD5, nD6);
	return quantity(value, m_Unit);
}

quantity daeParameter::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7);
	return quantity(value, m_Unit);
}

quantity daeParameter::GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8)
{
	real_t value = GetValue(nD1, nD2, nD3, nD4, nD5, nD6, nD7, nD8);
	return quantity(value, m_Unit);
}



bool daeParameter::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check parameter type	
//	if(m_eParameterType == ePTUnknown)
//	{
//		strError = "Invalid parameter type in parameter [" + GetCanonicalName() + "]";
//		strarrErrors.push_back(strError);
//		bCheck = false;
//	}
	
// Check value array	
	if(m_darrValues.size() == 0)
	{
		strError = "Parameter values have not been set for parameter [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check domains	
	if(m_ptrDomains.size() == 0)
	{
		if(m_darrValues.size() != 1)
		{
			strError = "Invalid number of values (should be 1) in parameter [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
	}
	else
	{
		daeDomain* pDomain;	
		for(size_t i = 0; i < m_ptrDomains.size(); i++)
		{
			pDomain = m_ptrDomains[i];
			if(!pDomain)
			{
				strError = "Invalid domain in parameter [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
			
			if(!pDomain->CheckObject(strarrErrors))
				bCheck = false;
		}
	}

	return bCheck;
}


}
}
