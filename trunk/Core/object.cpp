#include "stdafx.h"
#include "coreimpl.h"
#include <typeinfo>
#include "xmlfunctions.h"

namespace dae 
{
namespace core 
{
/********************************************************************
	daeObject
*********************************************************************/
daeObject::daeObject(void)
{
	m_pModel = NULL;
	m_strLibraryName = "DAE.Core";
}

daeObject::~daeObject(void)
{
}

void daeObject::Clone(const daeObject& rObject)
{
	m_pModel           = NULL;
//	m_strCanonicalName = rObject.m_strCanonicalName;
	m_strShortName     = rObject.m_strShortName;
	m_strDescription   = rObject.m_strDescription;
}

void daeObject::Open(io::xmlTag_t* pTag)
{
	string strName, strValue;

	daeSerializable::Open(pTag);

	strName = "Name";
	pTag->Open(strName, m_strShortName);
//	m_strCanonicalName = m_strShortName;
	
	strName = "Description";
	pTag->Open(strName, m_strDescription);
}

void daeObject::Save(io::xmlTag_t* pTag) const
{
	string strName, strValue;

	daeSerializable::Save(pTag);

	strName = "Name";
	pTag->Save(strName, m_strShortName);

	strName = "Description";
	pTag->Save(strName, m_strDescription);
	
	SaveNameAsMathML(pTag, string("MathMLName"));
}

void daeObject::SaveNameAsMathML(io::xmlTag_t* pTag, string strMathMLTag) const
{
	string strName, strValue;

	io::xmlTag_t* pChildTag = pTag->AddTag(strMathMLTag);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	strName = "math";
	io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);
	if(!pMathMLTag)
		daeDeclareAndThrowException(exXMLIOError);

	strName = "xmlns";
	strValue = "http://www.w3.org/1998/Math/MathML";
	pMathMLTag->AddAttribute(strName, strValue);

	strName = "mrow";
	io::xmlTag_t* mrow = pMathMLTag->AddTag(strName);
	if(!mrow)
		daeDeclareAndThrowException(exXMLIOError);
	
	xml::xmlPresentationCreator::WrapIdentifier(mrow, m_strShortName);
}

void daeObject::SaveRelativeNameAsMathML(io::xmlTag_t* pTag, string strMathMLTag, const daeObject* pParent) const
{
	string strName, strValue, strRelName;

	io::xmlTag_t* pChildTag = pTag->AddTag(strMathMLTag);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	strName = "math";
	io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);
	if(!pMathMLTag)
		daeDeclareAndThrowException(exXMLIOError);

	strName = "xmlns";
	strValue = "http://www.w3.org/1998/Math/MathML";
	pMathMLTag->AddAttribute(strName, strValue);

	strName = "mrow";
	io::xmlTag_t* mrow = pMathMLTag->AddTag(strName);
	if(!mrow)
		daeDeclareAndThrowException(exXMLIOError);
	
	if(pParent)
		strRelName = daeGetRelativeName(pParent, this);
	else
		strRelName = GetNameRelativeToParentModel();
	
	xml::xmlPresentationCreator::WrapIdentifier(mrow, strRelName);
}

void daeObject::OpenRuntime(io::xmlTag_t* /*pTag*/)
{
}

void daeObject::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName, strValue;

	daeSerializable::Save(pTag);

	strName = "Name";
	pTag->Save(strName, m_strShortName);

	strName = "Description";
	pTag->Save(strName, m_strDescription);
	
	SaveNameAsMathML(pTag, string("MathMLName"));
}

string daeObject::GetCanonicalName(void) const
{
	if(m_pModel)
		return m_pModel->GetCanonicalName() + '.' + m_strShortName;
	else
		return m_strShortName;
}

//void daeObject::SetCanonicalName(const string& strCanonicalName)
//{
//	if(strCanonicalName.empty())
//	{	
//		daeDeclareException(exInvalidCall);
//		e << "The name cannot be empty";
//		throw e;
//	}
//	m_strCanonicalName = strCanonicalName;
//}

string daeObject::GetDescription(void) const
{
	return m_strDescription;
}

void daeObject::SetDescription(const string& strDescription)
{
	m_strDescription = strDescription;
}

string daeObject::GetStrippedName(void) const
{
	string strStrippedName = m_strShortName;
	dae::RemoveAllNonAlphaNumericCharacters(strStrippedName);
	return strStrippedName;
}

string daeObject::GetStrippedNameRelativeToParentModel(void) const
{
	string strStrippedName = GetNameRelativeToParentModel();
	dae::RemoveAllNonAlphaNumericCharacters(strStrippedName);
	return strStrippedName;
}

string daeObject::GetNameRelativeToParentModel(void) const
{
	if(!m_pModel)
	{
		return GetCanonicalName();
	}
	else
	{		
		return daeGetRelativeName(m_pModel->GetCanonicalName(), GetCanonicalName());
	}
}

string daeGetRelativeName(const daeObject* parent, const daeObject* child)
{
	string strParent = (parent ? parent->GetCanonicalName() : string("")); 
	string strChild  = (child ? child->GetCanonicalName() : string(""));
	return dae::core::daeGetRelativeName(strParent, strChild);
}

string daeGetRelativeName(const string& strParent, const string& strChild)
{
	string::size_type iFounded = strChild.find(strParent);
	if(iFounded == string::npos || iFounded != 0)
		return strChild;

	size_t n = strParent.size();
	string strName = strChild.substr(n, strChild.size());
	LTrim(strName, ' ');
	LTrim(strName, '.');
	return strName;
}

string daeGetStrippedRelativeName(const daeObject* parent, const daeObject* child)
{
	string strStrippedName = dae::core::daeGetRelativeName(parent, child);
	dae::RemoveAllNonAlphaNumericCharacters(strStrippedName);
	return strStrippedName;
}

string daeObject::GetName(void) const
{
	return m_strShortName;
}
	
void daeObject::SetName(const string& strName)
{
//	if(m_strCanonicalName.empty() || m_strCanonicalName == "")
//	{
//		m_strCanonicalName = strName;
//	}
//	else
//	{
//		std::vector<std::string> strarrNames = ParseString(m_strCanonicalName, '.');
//		size_t n = strarrNames.size();
//		if(n == 1)
//		{
//			m_strCanonicalName = strName;
//		}
//		else
//		{
//			strarrNames.pop_back();
//			strarrNames.push_back(strName);
//			m_strCanonicalName.clear();
//			for(size_t i = 0; i < n; i++)
//			{
//				if(i == 0)
//				{
//					m_strCanonicalName = strarrNames[i];
//				}
//				else
//				{
//					m_strCanonicalName += ".";
//					m_strCanonicalName += strarrNames[i];
//				}
//			}
//		}
//	}
	
	m_strShortName = strName;
}

daeModel_t* daeObject::GetModel(void) const
{
	return m_pModel;
}
	
void daeObject::SetModel(daeModel* pModel)
{
	m_pModel = pModel;
}

void daeObject::LogMessage(const string& strMessage, size_t nSeverity) const
{
	if(m_pModel && m_pModel->m_pDataProxy)
		m_pModel->m_pDataProxy->LogMessage(strMessage, nSeverity);
}

bool daeObject::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

// Check model pointer
	if(!m_pModel)
	{
		strError = "Invalid parent model in object [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check name
	if(m_strShortName.empty())
	{	
		strError = "Object name is empty in object [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
	for(size_t i = 0; i < m_strShortName.size(); i++)
	{
		if(i == 0)
		{
			if((!::isalpha(m_strShortName[0])) && (m_strShortName[0] != '_') && (m_strShortName[0] != '&') && (m_strShortName[0] != ';'))
			{
				strError = "The first letter of the object name must be _ or alphanumeric character in object [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
		}
		else
		{ 
			if(!(::isalnum(m_strShortName[i]) || m_strShortName[i] == '_'
											  || m_strShortName[i] == ',' 
											  || m_strShortName[i] == '&' 
											  || m_strShortName[i] == ';' 
											  || m_strShortName[i] == '(' 
											  || m_strShortName[i] == ')' ))
			{	
				strError = "Object name contains invalid characters in object [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
		}
	}

// Check cannonical name
//	if(m_strCanonicalName.empty())
//	{	
//		strError = "Object cannonical name is empty in object [" + GetCanonicalName() + "]";
//		strarrErrors.push_back(strError);
//		bCheck = false;
//	}
	
	return bCheck;
}

void daeObject::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	
}


}
}
