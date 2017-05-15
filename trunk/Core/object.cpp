#include "stdafx.h"
#include "coreimpl.h"
#include <typeinfo>
#include <map>
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
	m_strShortName     = rObject.m_strShortName;
	m_strDescription   = rObject.m_strDescription;
}

void daeObject::Open(io::xmlTag_t* pTag)
{
	string strName, strValue;

	daeSerializable::Open(pTag);

	strName = "Name";
	pTag->Open(strName, m_strShortName);
	
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

string daeObject::GetCanonicalNameAndPrepend(const std::string& prependToName) const
{
    if(m_pModel)
        return m_pModel->GetCanonicalName() + '.' + prependToName + m_strShortName;
    else
        return prependToName + m_strShortName;
}

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

string daeGetStrippedName(const string& strName)
{
    string strStrippedName = strName;
    dae::RemoveAllNonAlphaNumericCharacters(strStrippedName);
    return strStrippedName;
}


string daeGetStrippedRelativeName(const daeObject* parent, const daeObject* child)
{
	string strStrippedName = dae::core::daeGetRelativeName(parent, child);
	dae::RemoveAllNonAlphaNumericCharacters(strStrippedName);
	return strStrippedName;
}

bool daeIsValidObjectName(const string& strName)
{
/*
    Rules:
     - The object name must not be empty.
     - The first letter of the object name must be a letter or '&' character.
     - The rest can be a combination of alphanumeric and '_,&;()' characters.
*/
    if(strName.empty())
        return false;
	
    int semicolon = 0;
    int ampersend = 0;
	for(size_t i = 0; i < strName.size(); i++)
	{
        if(strName[i] == '&')
            ampersend++;
        else if(strName[i] == ';')
            semicolon++;
		
        if(i == 0)
		{
			if((!::isalpha(strName[0])) && (strName[0] != '&') && (strName[0] != ';'))
                return false;
        }
		else
		{ 
			if(!(::isalnum(strName[i]) || strName[i] == '_'
                                       || strName[i] == ',' 
                                       || strName[i] == '&' 
                                       || strName[i] == ';' 
                                       || strName[i] == '(' 
                                       || strName[i] == ')' ))
                return false;
		}
	}  
    
    // There may be only equal number of '&' and ';'
    if(semicolon != ampersend)
        return false;
    else if(semicolon == 0 && ampersend > 0)
        return false;
    else if(semicolon > 0 && ampersend == 0)
        return false;
    
    // Check for something like ';&' or ';name&'
    size_t current = 0;
    size_t amperFound = strName.find('&', current);
    size_t semiFound  = strName.find(';', current);
    while(semiFound != std::string::npos && amperFound != std::string::npos)
    {
        if(amperFound == semiFound+1) // ';&'
            return false;
        else if(amperFound+1 == semiFound) // '&;'
            return false;
        else if(amperFound > semiFound) // ';name&'
            return false;
        
        current = std::max(amperFound, semiFound) + 1;
        amperFound = strName.find('&', current);
        semiFound  = strName.find(';', current);
    }
    
    return true;
}

string daeObject::GetName(void) const
{
	return m_strShortName;
}
	
void daeObject::SetName(const string& strName)
{
    if(!daeIsValidObjectName(strName))
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot set the name of the object [%s]: invalid characters found. \n";
        e << (boost::format(msg) % strName).str();
        e <<  "The naming convention is: \n";
        e << " - the object name must not be empty \n";
        e << " - the name can contain HTML codes for Greek letters such as '&alpha;' etc \n";
        e << " - the first character must be a letter or '&' character \n";
        e << " - the rest can be a combination of the alphanumeric and '_,&;()' characters";
		throw e;
    }
    
    m_strShortName = strName;
}

daeModel_t* daeObject::GetModel(void) const
{
	return m_pModel;
}
	
void daeObject::SetModel(daeModel* pModel)
{
    if(!pModel)
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot set the parent model of the object [%s]: the model is a NULL pointer/reference";
        e << (boost::format(msg) % GetCanonicalName()).str();
		throw e;
    }

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
    if(!daeIsValidObjectName(m_strShortName))
	{	
		strError = "Invalid name of the object [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
	return bCheck;
}

void daeObject::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	
}


}
}
