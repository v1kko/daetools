#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{
/******************************************************************
	daePortArray
*******************************************************************/
daePortArray::daePortArray(int n) : N(n)
{
	m_ePortType               = eUnknownPort;
	_currentVariablesIndex    = ULONG_MAX;
	m_nVariablesStartingIndex = ULONG_MAX;
}

daePortArray::~daePortArray(void)
{
}

size_t daePortArray::GetDimensions(void) const
{
	return N;
}

void daePortArray::DistributeOnDomain(daeDomain& rDomain)
{
	if(!(&rDomain))
		daeDeclareAndThrowException(exInvalidPointer);
	m_ptrarrDomains.push_back(&rDomain);
}

void daePortArray::Create()
{
	daeDomain* pDomain;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(m_ptrarrDomains.size() != (size_t)N)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Number of domains is " << m_ptrarrDomains.size() << "; it should be " << N;
		throw e;
	}

	if(m_ePortType == eUnknownPort)
	{
		daeDeclareException(exInvalidCall); 
		e << "Invalid port type";
		throw e;
	}

	for(size_t it = 0; it < m_ptrarrDomains.size(); it++)
	{
		pDomain = m_ptrarrDomains[it];
		if(!pDomain)
			daeDeclareAndThrowException(exInvalidPointer); 
		if(pDomain->GetNumberOfPoints() == 0)
		{	
			daeDeclareException(exInvalidCall); 
			e << "Number of points in domain [" << pDomain->GetCanonicalName() << "] is 0; did you forget to initialize it?";
			throw e;
		}
	}
}

size_t daePortArray::GetVariablesStartingIndex(void) const
{
	return m_nVariablesStartingIndex;
}

void daePortArray::SetVariablesStartingIndex(size_t nVariablesStartingIndex)
{
	m_nVariablesStartingIndex = nVariablesStartingIndex;
}

daePort_t* daePortArray::GetPort(vector<size_t>& narrIndexes)
{
	if(narrIndexes.size() == 1)
		return GetPort(narrIndexes[0]);
	else if(narrIndexes.size() == 2)
		return GetPort(narrIndexes[0], narrIndexes[1]);
	else if(narrIndexes.size() == 3)
		return GetPort(narrIndexes[0], narrIndexes[1], narrIndexes[2]);
	else if(narrIndexes.size() == 4)
		return GetPort(narrIndexes[0], narrIndexes[1], narrIndexes[2], narrIndexes[3]);
	else
		daeDeclareAndThrowException(exNotImplemented);

	return NULL;
}

void daePortArray::Open(io::xmlTag_t* pTag)
{
	string strName;

	daeObject::Open(pTag);

	strName = "Domains";
	//io::daeFindDomainByID fd(m_pModel);
	//io::daeOpenObjectRefArray<daeDomain, unsigned long>(pTag, strName, m_ptrarrDomains, &fd);
}

void daePortArray::Save(io::xmlTag_t* pTag) const
{
	string strName;
	vector<unsigned long> arrIDs;

	daeObject::Save(pTag);

	//strName = "Domains";
	//for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
	//	arrIDs.push_back(m_ptrarrDomains[i]->GetID());
	//io::daeSaveObjectRefArray<unsigned long>(pTag, arrIDs, strName);
}

void daePortArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{	
}

void daePortArray::DetectVariableTypesForExport(std::vector<const daeVariableType*>& ptrarrVariableTypes) const
{
}

void daePortArray::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daePortArray::SaveRuntime(io::xmlTag_t* pTag) const
{
	daeObject::SaveRuntime(pTag);
}

void daePortArray::GetDomains(vector<daeDomain_t*>& ptrarrDomains)
{
	ptrarrDomains.clear();
	for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
		ptrarrDomains.push_back(m_ptrarrDomains[i]);
}

daePort_t* daePortArray::GetPort(size_t /*n1*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

daePort_t* daePortArray::GetPort(size_t /*n1*/, size_t /*n2*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

daePort_t* daePortArray::GetPort(size_t /*n1*/, size_t /*n2*/, size_t /*n3*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

daePort_t* daePortArray::GetPort(size_t /*n1*/, size_t /*n2*/, size_t /*n3*/, size_t /*n4*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

daePort_t* daePortArray::GetPort(size_t /*n1*/, size_t /*n2*/, size_t /*n3*/, size_t /*n4*/, size_t /*n5*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

bool daePortArray::CheckObject(vector<string>& strarrErrors) const
{
	string strError;
	bool bCheck = true;
	daeDomain* pDomain;

// Check base class
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;
	
// Check port type
	if(m_ePortType == eUnknownPort)
	{
		strError = "Invalid port type in port array: [" + GetCanonicalName() + "]";
		bCheck = false;
	}
	
// Check number of domains
	if(m_ptrarrDomains.size() != (size_t)N)
	{	
		strError = "Invalid number of domains in port array: [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check each domain
	for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
	{
		pDomain = m_ptrarrDomains[i];
		if(!pDomain)
		{
			strError = "Invalid domain in port array: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pDomain->CheckObject(strarrErrors))
			bCheck = false;
	}
	return bCheck;
}

/******************************************************************
	daeModelArray
*******************************************************************/
daeModelArray::daeModelArray(int n) : N(n)
{
	_currentVariablesIndex    = ULONG_MAX;
	m_nVariablesStartingIndex = ULONG_MAX;
}

daeModelArray::~daeModelArray(void)
{
}

size_t daeModelArray::GetDimensions() const
{
	return N;
}

void daeModelArray::DistributeOnDomain(daeDomain& rDomain)
{
	if(!(&rDomain))
		daeDeclareAndThrowException(exInvalidPointer); 
	m_ptrarrDomains.push_back(&rDomain);
}

void daeModelArray::GetDomains(vector<daeDomain_t*>& ptrarrDomains)
{
	ptrarrDomains.clear();
	for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
		ptrarrDomains.push_back(m_ptrarrDomains[i]);
}

size_t daeModelArray::GetVariablesStartingIndex(void) const
{
	return m_nVariablesStartingIndex;
}

void daeModelArray::SetVariablesStartingIndex(size_t nVariablesStartingIndex)
{
	m_nVariablesStartingIndex = nVariablesStartingIndex;
}

daeModel_t* daeModelArray::GetModel(vector<size_t>& narrIndexes)
{
	if(narrIndexes.size() == 1)
		return GetModel(narrIndexes[0]);
	else if(narrIndexes.size() == 2)
		return GetModel(narrIndexes[0], narrIndexes[1]);
	else if(narrIndexes.size() == 3)
		return GetModel(narrIndexes[0], narrIndexes[1], narrIndexes[2]);
	else if(narrIndexes.size() == 4)
		return GetModel(narrIndexes[0], narrIndexes[1], narrIndexes[2], narrIndexes[3]);
	else
		daeDeclareAndThrowException(exNotImplemented); 

	return NULL;
}

daeModel_t* daeModelArray::GetModel(size_t /*n1*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

daeModel_t* daeModelArray::GetModel(size_t /*n1*/, size_t /*n2*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

daeModel_t* daeModelArray::GetModel(size_t /*n1*/, size_t /*n2*/, size_t /*n3*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

daeModel_t* daeModelArray::GetModel(size_t /*n1*/, size_t /*n2*/, size_t /*n3*/, size_t /*n4*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

daeModel_t* daeModelArray::GetModel(size_t /*n1*/, size_t /*n2*/, size_t /*n3*/, size_t /*n4*/, size_t /*n5*/)
{
	daeDeclareAndThrowException(exNotImplemented); 
	return NULL;
}

void daeModelArray::Open(io::xmlTag_t* pTag)
{
	string strName;

	daeObject::Open(pTag);

	//strName = "Domains";
	//io::daeFindDomainByID fd(m_pModel);
	//io::daeOpenObjectRefArray<daeDomain, unsigned long>(pTag, strName, m_ptrarrDomains, &fd);
}

void daeModelArray::Save(io::xmlTag_t* pTag) const
{
	string strName;
	vector<unsigned long> arrIDs;

	daeObject::Save(pTag);

	//strName = "Domains";
	//for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
	//	arrIDs.push_back(m_ptrarrDomains[i]->GetID());
	//io::daeSaveObjectRefArray<unsigned long>(pTag, arrIDs, strName);
}

void daeModelArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
}

void daeModelArray::DetectVariableTypesForExport(std::vector<const daeVariableType*>& ptrarrVariableTypes) const
{
}

void daeModelArray::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daeModelArray::SaveRuntime(io::xmlTag_t* pTag) const
{
	daeObject::SaveRuntime(pTag);
}

void daeModelArray::Create()
{
	daeDomain* pDomain;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	if(m_ptrarrDomains.size() != (size_t)N)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Number of domains is " << m_ptrarrDomains.size() << "; it should be " << N;
		throw e;
	}

	for(size_t it = 0; it < m_ptrarrDomains.size(); it++)
	{
		pDomain = m_ptrarrDomains[it];
		if(!pDomain)
			daeDeclareAndThrowException(exInvalidPointer); 
		if(pDomain->GetNumberOfPoints() == 0)
		{	
			daeDeclareException(exInvalidCall); 
			e << "Number of points in domain [" << pDomain->GetCanonicalName() << "] is 0; did you forget to initialize it?";
			throw e;
		}
	}
}

bool daeModelArray::CheckObject(vector<string>& strarrErrors) const
{
	string strError;
	bool bCheck = true;
	daeDomain* pDomain;

// Check base class
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check number of domains
	if(m_ptrarrDomains.size() != (size_t)N)
	{	
		strError = "Invalid number of domains in model array: [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check each domain
	for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
	{
		pDomain = m_ptrarrDomains[i];
		if(!pDomain)
		{
			strError = "Invalid domain in model array: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pDomain->CheckObject(strarrErrors))
			bCheck = false;
	}
	return bCheck;
}
	
	
}
}
