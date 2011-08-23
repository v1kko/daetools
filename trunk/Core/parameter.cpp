#include "stdafx.h"
#include "coreimpl.h"
#include "nodes_array.h"

namespace dae 
{
namespace core 
{
/******************************************************************
	daeParameter
*******************************************************************/
daeParameter::daeParameter(void)
{
	m_eParameterType	= ePTUnknown;
	m_pModel			= NULL;
}
	
daeParameter::daeParameter(string strName, daeeParameterType paramType, daeModel* pModel, string strDescription, 
						   daeDomain* d1, daeDomain* d2, daeDomain* d3, daeDomain* d4, daeDomain* d5, daeDomain* d6, daeDomain* d7, daeDomain* d8)
{
	m_eParameterType	= ePTUnknown;
	m_pModel			= NULL;

	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	pModel->AddParameter(*this, strName, paramType, strDescription);
	
	m_ptrDomains = dae::makeVector<daeDomain*>(d1, d2, d3, d4, d5, d6, d7, d8);
}
	
daeParameter::daeParameter(string strName, daeeParameterType paramType, daePort* pPort, string strDescription, 
						   daeDomain* d1, daeDomain* d2, daeDomain* d3, daeDomain* d4, daeDomain* d5, daeDomain* d6, daeDomain* d7, daeDomain* d8)
{
	m_eParameterType	= ePTUnknown;
	m_pModel			= NULL;

	if(!pPort)
		daeDeclareAndThrowException(exInvalidPointer);
	pPort->AddParameter(*this, strName, paramType, strDescription);
	
	m_ptrDomains = dae::makeVector<daeDomain*>(d1, d2, d3, d4, d5, d6, d7, d8);
}

daeParameter::~daeParameter(void)
{
}

void daeParameter::Clone(const daeParameter& rObject)
{
	m_eParameterType = rObject.m_eParameterType;
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

	strName = "Type";
	OpenEnum(pTag, strName, m_eParameterType);

	strName = "DomainRefs";
	daeFindDomainByID del(m_pModel);
	pTag->OpenObjectRefArray(strName, m_ptrDomains, &del);
}

void daeParameter::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "Type";
	SaveEnum(pTag, strName, m_eParameterType);

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
					% dae::io::g_EnumTypesCollection->esmap_daeeParameterType.GetString(m_eParameterType) 
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
					% dae::io::g_EnumTypesCollection->esmap_daeeParameterType.GetString(m_eParameterType) 
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
	string strName;

	daeObject::SaveRuntime(pTag);

	strName = "Type";
	SaveEnum(pTag, strName, m_eParameterType);

	strName = "DomainRefs";
	pTag->SaveObjectRefArray(strName, m_ptrDomains);

	strName = "Values";
	pTag->SaveArray(strName, m_darrValues);
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
			e << "Number of points in domain [" << pDomain->GetCanonicalName() << "] in parameter [" << m_strCanonicalName << "] must not be zero; did you forget to initialize it?";
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
		tmp.node = boost::shared_ptr<adNode>(node);
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
				  << "in parameter [" << m_strCanonicalName << "] in operator()";
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
	tmp.node = boost::shared_ptr<adNode>(node);
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
	if(m_pModel->m_pDataProxy->GetGatherInfo())
	{
		adRuntimeParameterNodeArray* node = new adRuntimeParameterNodeArray();
		varArray.node = boost::shared_ptr<adNodeArray>(node);
		varArray.setGatherInfo(true);
		node->m_pParameter = const_cast<daeParameter*>(this);
		
		size_t size = varArray.m_arrValues.size();
		if(size == 0)
			daeDeclareAndThrowException(exInvalidCall); 
		
		node->m_ptrarrParameterNodes.resize(size);
		for(size_t i = 0; i < size; i++)
			node->m_ptrarrParameterNodes[i] = varArray.m_arrValues[i].node;
		node->m_arrRanges.resize(N);
		for(size_t i = 0; i < N; i++)
			node->m_arrRanges[i] = ranges[i];
	}
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
					  << "in parameter [" << m_strCanonicalName << "] in function array()";
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
				  << "in parameter [" << m_strCanonicalName << "] in function array()";
				throw e;
			}
		}
	}

	adSetupParameterNodeArray* node = new adSetupParameterNodeArray();
	varArray.node = boost::shared_ptr<adNodeArray>(node);
	varArray.setGatherInfo(true);

	node->m_pParameter = const_cast<daeParameter*>(this);
	node->m_arrRanges.resize(N);
	for(size_t i = 0; i < N; i++)
		node->m_arrRanges[i] = ranges[i];

	return varArray;
}

daeeParameterType daeParameter::GetParameterType(void) const
{
	return m_eParameterType;
}

void daeParameter::SetParameterType(daeeParameterType eParameterType)
{
	m_eParameterType = eParameterType;
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

size_t daeParameter::CalculateIndex(const size_t* indexes, const size_t N) const
{
	size_t		i, j, nIndex, temp;
	daeDomain	*pDomain;

	if(m_ptrDomains.size() != N)
	{	
		daeDeclareException(exInvalidCall);
		e << "Illegal number of domains, parameter " << m_strCanonicalName;
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

void daeParameter::SetValue(real_t value)
{
	if(m_ptrDomains.size() != 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 0; it should be " << m_ptrDomains.size();
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 1; it should be " << m_ptrDomains.size();
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 2; it should be " << m_ptrDomains.size();
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 3; it should be " << m_ptrDomains.size();
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 4; it should be " << m_ptrDomains.size();
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 5; it should be " << m_ptrDomains.size();
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 6; it should be " << m_ptrDomains.size();
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 7; it should be " << m_ptrDomains.size();
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 8; it should be " << m_ptrDomains.size();
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 0; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << m_strCanonicalName << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	return m_darrValues[0];
}

real_t daeParameter::GetValue(size_t nDomain1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 1; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << m_strCanonicalName << "] is zero; did you forget to initialize the domains that it is distributed on?";
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 2; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << m_strCanonicalName << "] is zero; did you forget to initialize the domains that it is distributed on?";
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 3; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << m_strCanonicalName << "] is zero; did you forget to initialize the domains that it is distributed on?";
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 4; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << m_strCanonicalName << "] is zero; did you forget to initialize the domains that it is distributed on?";
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 5; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << m_strCanonicalName << "] is zero; did you forget to initialize the domains that it is distributed on?";
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 6; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << m_strCanonicalName << "] is zero; did you forget to initialize the domains that it is distributed on?";
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
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 7; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << m_strCanonicalName << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[7] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6, nDomain7};
	return m_darrValues[CalculateIndex(indexes, 7)];
}

real_t daeParameter:: GetValue(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7, size_t nDomain8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid set parameter value call for [" << m_strCanonicalName << "]" << "Number of domains is 8; it should be " << m_ptrDomains.size();
		throw e;
	}
	if(m_darrValues.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in parameter [" << m_strCanonicalName << "] is zero; did you forget to initialize the domains that it is distributed on?";
		throw e;
	}

	size_t indexes[8] = {nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6, nDomain7, nDomain8};
	return m_darrValues[CalculateIndex(indexes, 8)];
}

bool daeParameter::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check parameter type	
	if(m_eParameterType == ePTUnknown)
	{
		strError = "Invalid parameter type in parameter [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
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
