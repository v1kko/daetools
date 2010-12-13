#include "stdafx.h"
#include "xmlfile.h"
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include "helpers.h"

using namespace std;

namespace dae 
{
namespace xml
{

std::string xmlTag::m_strLevel = "  ";

xmlTag::xmlTag()
{
	m_pParentTag = NULL;
} 

xmlTag& xmlTag::operator=(const xmlTag & rTag)
{
	size_t i;
	xmlTag *pTag, *pNewTag;
	xmlAttribute *pAttribute, *pNewAttribute;

	EmptyMaps();

	for(i = 0; i < rTag.m_mapTags.size(); i++)
	{
		pTag = rTag.m_mapTags[i];
		if(pTag)
		{
			pNewTag = new xmlTag(*pTag);
			pNewTag->m_pParentTag = this;
			m_mapTags.push_back(pNewTag);
		}
	}

	for(i = 0; i < rTag.m_mapAttributes.size(); i++)
	{
		pAttribute = rTag.m_mapAttributes[i];
		if(pAttribute)
		{
			pNewAttribute = new xmlAttribute(*pAttribute);
			pNewAttribute->m_pParentTag = this;
			m_mapAttributes.push_back(pNewAttribute);
		}
	}

	m_strValue		= rTag.m_strValue;
	m_strTagName	= rTag.m_strTagName;
	m_pParentTag	= rTag.m_pParentTag;

	return *this;
} 

xmlTag::xmlTag(const xmlTag & rTag)
{
	size_t i;
	xmlTag *pTag, *pNewTag;
	xmlAttribute *pAttribute, *pNewAttribute;

	for(i = 0; i < rTag.m_mapTags.size(); i++)
	{
		pTag = rTag.m_mapTags[i];
		if(pTag)
		{
			pNewTag = new xmlTag(*pTag);
			pNewTag->m_pParentTag = this;
			m_mapTags.push_back(pNewTag);
		}
	}

	for(i = 0; i < rTag.m_mapAttributes.size(); i++)
	{
		pAttribute = rTag.m_mapAttributes[i];
		if(pAttribute)
		{
			pNewAttribute = new xmlAttribute(*pAttribute);
			pNewAttribute->m_pParentTag = this;
			m_mapAttributes.push_back(pNewAttribute);
		}
	}

	m_strValue		= rTag.m_strValue;
	m_strTagName	= rTag.m_strTagName;
	m_pParentTag	= rTag.m_pParentTag;
} 

xmlTag::~xmlTag()
{
	EmptyMaps();
} 

xmlTag* xmlTag::Clone()
{
	return new xmlTag(*this);
}

void xmlTag::EmptyMaps()
{
	xmlTag* pTag;
	xmlAttribute* pAttribute;
	size_t i;
	for(i = 0; i < m_mapTags.size(); i++)
	{
		pTag = m_mapTags[i];
		if(pTag)
		{
			delete pTag;
			pTag = NULL;
		}
	}
	m_mapTags.erase(m_mapTags.begin(), m_mapTags.end());
	for(i = 0; i < m_mapAttributes.size(); i++)
	{
		pAttribute = m_mapAttributes[i];
		if(pAttribute)
		{
			delete pAttribute;
			pAttribute = NULL;
		}
	}
	m_mapAttributes.erase(m_mapAttributes.begin(), m_mapAttributes.end());
}

void xmlTag::AddTag(xmlTag * pTag)
{
	pTag->m_pParentTag = this;
	m_mapTags.push_back(pTag);
} 

xmlTag_t* xmlTag::AddTag(const std::string& strName)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	AddTag(pTag);
	return pTag;
}

xmlTag_t* xmlTag::AddTag(const std::string& strName, std::string Value)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	pTag->SetValue(Value);
	AddTag(pTag);
	return pTag;
}

xmlTag_t* xmlTag::AddTag(const std::string& strName, float Value)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	pTag->SetValue(Value);
	AddTag(pTag);
	return pTag;
}

xmlTag_t* xmlTag::AddTag(const std::string& strName, double Value)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	pTag->SetValue(Value);
	AddTag(pTag);
	return pTag;
}

xmlTag_t* xmlTag::AddTag(const std::string& strName, int Value)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	pTag->SetValue(Value);
	AddTag(pTag);
	return pTag;
}

xmlTag_t* xmlTag::AddTag(const std::string& strName, long Value)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	pTag->SetValue(Value);
	AddTag(pTag);
	return pTag;
}

xmlTag_t* xmlTag::AddTag(const std::string& strName, unsigned long Value)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	pTag->SetValue(Value);
	AddTag(pTag);
	return pTag;
}

xmlTag_t* xmlTag::AddTag(const std::string& strName, unsigned int Value)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	pTag->SetValue(Value);
	AddTag(pTag);
	return pTag;
}

xmlTag_t* xmlTag::AddTag(const std::string& strName, bool Value)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	pTag->SetValue(Value);
	AddTag(pTag);
	return pTag;
}

xmlTag_t* xmlTag::AddTag(const std::string& strName, char Value)
{
	xmlTag* pTag = new xmlTag;
	pTag->SetName(strName);
	pTag->SetValue(Value);
	AddTag(pTag);
	return pTag;
}

void xmlTag::InsertTag(xmlTag* pTag, size_t iPosition)
{
	pTag->m_pParentTag = this;

	if(iPosition < m_mapTags.size())
		m_mapTags.insert(m_mapTags.begin() + iPosition, pTag);
	else
		m_mapTags.push_back(pTag);
}

bool xmlTag::RemoveTag(xmlTag* pTagToRemove)
{
	xmlTag* pTag;
	for(size_t i = 0; i < m_mapTags.size(); i++)
	{
		pTag = m_mapTags[i];	
		if(pTag && pTag == pTagToRemove)
		{
			m_mapTags.erase(m_mapTags.begin() + i);
			return true;
		}
	}
	return false;
}

xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName)
{
	xmlAttribute* pAtrribute = new xmlAttribute;
	pAtrribute->m_strName  = strName;
	pAtrribute->m_pParentTag = this;
	m_mapAttributes.push_back(pAtrribute);
	return pAtrribute;
}
xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName, std::string Value)
{
	xmlAttribute_t* pAtrribute = AddAttribute(strName);
	pAtrribute->SetValue(Value);
	return pAtrribute;
}
xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName, float Value)
{
	xmlAttribute_t* pAtrribute = AddAttribute(strName);
	pAtrribute->SetValue(Value);
	return pAtrribute;
}
xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName, double Value)
{
	xmlAttribute_t* pAtrribute = AddAttribute(strName);
	pAtrribute->SetValue(Value);
	return pAtrribute;
}
xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName, int Value)
{
	xmlAttribute_t* pAtrribute = AddAttribute(strName);
	pAtrribute->SetValue(Value);
	return pAtrribute;
}
xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName, long Value)
{
	xmlAttribute_t* pAtrribute = AddAttribute(strName);
	pAtrribute->SetValue(Value);
	return pAtrribute;
}
xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName, unsigned long Value)
{
	xmlAttribute_t* pAtrribute = AddAttribute(strName);
	pAtrribute->SetValue(Value);
	return pAtrribute;
}
xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName, unsigned int Value)
{
	xmlAttribute_t* pAtrribute = AddAttribute(strName);
	pAtrribute->SetValue(Value);
	return pAtrribute;
}
xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName, bool Value)
{
	xmlAttribute_t* pAtrribute = AddAttribute(strName);
	pAtrribute->SetValue(Value);
	return pAtrribute;
}
xmlAttribute_t* xmlTag::AddAttribute(const std::string& strName, char Value)
{
	xmlAttribute_t* pAtrribute = AddAttribute(strName);
	pAtrribute->SetValue(Value);
	return pAtrribute;
}

bool xmlTag::RemoveAttribute(const std::string& strAttributeName)
{
	xmlAttribute* pAtrribute;
	for(size_t i = 0; i < m_mapAttributes.size(); i++)
	{
		pAtrribute = m_mapAttributes[i];	
		if(pAtrribute && pAtrribute->m_strName == strAttributeName)
		{
			m_mapAttributes.erase(m_mapAttributes.begin() + i);
			return true;
		}
	}
	return false;
}

xmlAttribute_t* xmlTag::FindAttribute(const std::string& strName) const
{
	for(size_t i = 0; i < m_mapAttributes.size(); i++)
	{	
		xmlAttribute* pAttribute = m_mapAttributes[i];
		if(pAttribute->m_strName == strName)
			return pAttribute;
	}
	return NULL;
}

bool xmlTag::SetAttribute(const std::string & strAttributeName, const std::string& strValue)
{
	for(size_t i = 0; i < m_mapAttributes.size(); i++)
		if(m_mapAttributes[i]->m_strName == strAttributeName)
		{
			m_mapAttributes[i]->m_strValue = strValue;
			return true;
		}

	return false;
}

std::string xmlTag::GetAttribute(const std::string & strAttributeName) const
{
	for(size_t i = 0; i < m_mapAttributes.size(); i++)
		if(m_mapAttributes[i]->m_strName == strAttributeName)
			return m_mapAttributes[i]->m_strValue;

	return "";
} 

xmlAttribute* xmlTag::GetAttributePtr(const std::string & strAttributeName)
{
	for(size_t i = 0; i < m_mapAttributes.size(); i++)
		if(m_mapAttributes[i] && m_mapAttributes[i]->m_strName == strAttributeName)
			return m_mapAttributes[i];

	return NULL;
}

std::ostream & xmlTag::Save(size_t iLevel, std::ostream & inStream) const
{
	for(size_t j = 0; j < iLevel; j++)
		inStream << m_strLevel;

	inStream << "<" 
			 << PostProcessString(m_strTagName);
	
	WriteAttributes(inStream);

	if (m_strValue.size() || m_mapTags.size())
	{
		inStream << ">";
		
		if (m_mapTags.size())
		{
			inStream << std::endl;

			for(size_t i = 0; i < m_mapTags.size(); i++)
				m_mapTags[i]->Save(iLevel+1, inStream);
			
			for(size_t k = 0; k < iLevel; k++)
				inStream << m_strLevel;

			inStream << PostProcessString(m_strValue) << "</" << PostProcessString(m_strTagName) << ">" ;
		}
		else
		{
			inStream << PostProcessString(m_strValue) << "</" << PostProcessString(m_strTagName) << ">" ;
		}
	}
	else
	{
		inStream << "/>";
	}
	inStream << std::endl;
	
	return inStream;
} 

xmlTag_t* xmlTag::FindTag(const std::string& strTagName) const
{
	for(size_t i = 0; i < m_mapTags.size(); i++)
		if(m_mapTags[i]->m_strTagName == strTagName)
			return m_mapTags[i];

	return NULL;
}

bool xmlTag::ContainTags() const
{
	return (m_mapTags.size() > 0 ? true : false);
}

bool xmlTag::ContainAttributes() const
{
	return (m_mapAttributes.size() > 0 ? true : false);
}

std::string xmlTag::GetPath(char cSeparator) const
{
	std::string strPath;
	xmlTag* pParent = NULL;
	xmlTag* pCurrentTag = (xmlTag*)this;
	strPath = m_strTagName;
	for(;;)
	{
		pParent = pCurrentTag->m_pParentTag;
		if(pParent == NULL)
			break;
		strPath = pParent->m_strTagName + cSeparator + strPath;
		pCurrentTag = pParent;
	}
	return strPath;
}

std::vector<xmlTag_t*> xmlTag::FindMultipleTag(const std::string& strTagName) const
{
	std::vector<xmlTag_t*> arrTag;	
	std::string str;	
	for(size_t i = 0; i < m_mapTags.size(); i++)
		if(m_mapTags[i]->m_strTagName == strTagName)
			arrTag.push_back(m_mapTags[i]);

	return arrTag;
}

xmlReadStream & xmlTag::Parse(const std::string & inStartTag, xmlReadStream & inStream)
{
	ParseAttributes (inStartTag);
	
	if (inStartTag.find("/>") == inStartTag.length()-2)
	{
		// tag is closed
		return inStream;
	}
	
	string aString;
	for (;;) 
	{
		inStream.readString(aString);
		if (aString.length() == 0)
		{
			// empty string indicates end of file
			break;
		}
		
		if (aString.find("<?") == 0)
		{
			// it is an XML description field
			// ignore it
			continue;
		}
		if (aString.find("<!") == 0)
		{
			// it is an XML description field
			// ignore it
			continue;
		}
		if (aString.find("</") == 0)
		{
			// it is the end of the tag
			// exit
			break;
		}
		if (aString.find("<") == 0)
		{
			// it is a nested tag
			xmlTag * aNestedTag = new xmlTag;
			aNestedTag->Parse (aString, inStream);
			AddTag(aNestedTag);
			continue;
		}
		// it is a value
		m_strValue = PreProcessString(aString);
	} 

	return inStream;
}

xmlTag & xmlTag::ParseAttributes(const std::string & inString)
{
	bool bIsAttributeNameCompleted = false;

	size_t index = inString.find_first_of(" />");
	if (index == string::npos)
	{
		std::string strError;
		strError = "Invalid start tag [";
		strError += inString;
		strError += "]\n in [";
		strError += GetPath('/');
		strError += "]";
		throw new xmlException(strError);
	}

	m_strTagName = PreProcessString(inString.substr(1, index-1));
	
	if(m_strTagName.length() == 0)
	{
		std::string strError;
		strError = "Invalid element name [";
		strError += inString;
		strError += "]\n in [";
		strError += GetPath('/');
		strError += "]";
		throw new xmlException(strError);
	}
	
	if (!isalpha(m_strTagName[0]))
	{
		std::string strError;
		strError = "Invalid element name [";
		strError += inString;
		strError += "]\n in [";
		strError += GetPath('/');
		strError += "]";
		throw new xmlException(strError);
	}
	
	size_t endIndex = inString.find_last_not_of (" />");
	
	string anAttributes = inString.substr(index+1, endIndex-index);
	string aName;
	string aValue;
	
	
	for(size_t i = 0; i < anAttributes.length(); i++)
	{
		char c = anAttributes[i];
		switch (c)
		{
		case '=':
			if (aName.length() == 0)
			{
				std::string strError;
				strError = "Invalid attribute name [";
				strError += inString;
				strError += "]\n in [";
				strError += GetPath('/');
				strError += "]";
				throw new xmlException(strError);
			}
	/*********   29. jul 2003.    **************/
			if(aName.length() > 0)
				if(aValue.length() > 0)
					aValue += c;
			
			break;
	/******************************************/
		case ' ': //Blanko je dozvoljeno samo u vrednosti parametra
			if(aName.length())
				if(aValue.length() > 1)
					aValue += c;
				break;
		case '\"':
		case '\'':
			if(aValue.length() && aValue[0] == c)// Zatvarajuci navodnik, tj. zavrsio sam uzimanje i imena i vrednosti atributa
			{
				// it is closing value
				if(aValue[aValue.length()-1] == ' ')
					aValue = aValue.substr(1, aValue.length()-2);
				else
					aValue = aValue.substr(1);

				if(FindAttribute(aName))
				{
					std::string strError;
					strError = "Double attribute found [";
					strError += aName;
					strError += "]\n in [";
					strError += GetPath('/');
					strError += "]";
					throw new xmlException(strError);
				}
		/*********   29. jul 2003.   (Mada u principu nepotrebno, jer se = u imenu taga preskace) **************/
				if(aName.find('=') != std::string::npos)
				{
					std::string strError;
					strError = "Invalid tag name found [";
					strError += aName;
					strError += "]\n in [";
					strError += GetPath('/');
					strError += "]";
					throw new xmlException(strError);
				}
		/***********************************************************************/

				AddAttribute(PreProcessString(aName), PreProcessString(aValue));
				aName = "";
				aValue = "";
				bIsAttributeNameCompleted = false;
			}
			else// Otvarajuci navodnik, tj. pocinjem sa uzimanjem vrednosti atributa
			{
				bIsAttributeNameCompleted = true;
				aValue += c;
			}
			break;
		default:
			if (aValue.length())
				aValue += c;
			else
				aName += c;
		}
	} 
	return *this;
} 


string xmlTag::PostProcessString(const std::string & inString) const
{
	string aResult;
/* 16. februar 2008.  
     Dodao sam podrsku za vrednosti oblika: &...;
	 Korisno u MathML, npr &PartialD; itd
*/
	size_t nLength = inString.length();
	if(nLength > 0 && inString[0] == '&' && inString[nLength-1] == ';')
	{
		aResult = inString;
		return aResult;
	}

	for(size_t i = 0; i < nLength; i++)
	{
		switch (inString[i])
		{
		case '&':
			aResult += "&amp;";
			break;
		case '<':
			aResult += "&lt;";
			break;
		case '>':
			aResult += "&gt;";
			break;
		case '\'':
			aResult += "&apos;";
			break;
		case '\"':
			aResult += "&quot;";
			break;
			
		default:
			aResult += inString[i];
		}
	} 
	return aResult;
}

string xmlTag::PreProcessString(const std::string & aString) const
{
	string inString (aString);
	
	for(size_t start = inString.find('&'); start != string::npos; start = inString.find('&', start+1) )
	{
		size_t end = inString.find(';', start);
		if (end == string::npos)
		{
			std::string strError;
			strError = "Invalid entity refferrence [";
			strError += inString;
			strError += "] in [";
			strError += GetPath('/');
			strError += "]";
			throw new xmlException(strError);
		}
		
		string aRef = inString.substr(start, end-start);
		string aBegin = inString.substr(0, start);
		if (aRef == "&lt")
			aBegin += '<'; 
		else if (aRef == "&gt")
			aBegin += '>'; 
		else if (aRef == "&amp")
			aBegin += '&'; 
		else if (aRef == "&apos")
			aBegin += '\''; 
		else if (aRef == "&quot")
			aBegin += '\"'; 
		else
		{
			std::string strError;
			strError = "Invalid entity refferrence [";
			strError += aRef;
			strError += "] in [";
			strError += GetPath('/');
			strError += "]";
			throw new xmlException(strError);
		}

		if (end+1 < inString.length())
			aBegin += inString.substr(end+1);

		inString = aBegin;
	} 
	return inString;
}


void xmlTag::GetValue(std::string& Value) const
{
	Value = m_strValue;
} 
void xmlTag::GetValue(int& Value) const
{
	Value = fromString<int>(m_strValue);
} 
void xmlTag::GetValue(long& Value) const
{
	Value = fromString<long>(m_strValue);
} 
void xmlTag::GetValue(unsigned long& Value) const
{
	Value = fromString<unsigned long>(m_strValue);
} 
void xmlTag::GetValue(unsigned int& Value) const
{
    Value = fromString<unsigned int>(m_strValue);
} 
void xmlTag::GetValue(float& Value) const
{
	Value = fromString<float>(m_strValue);
} 
void xmlTag::GetValue(double& Value) const
{
	Value = fromString<double>(m_strValue);
} 
void xmlTag::GetValue(bool& Value) const
{
	if(m_strValue == "true" || m_strValue == "True" || m_strValue == "1" || m_strValue == "yes")
		Value = true;
	else
		Value = false;
} 
void xmlTag::GetValue(char& Value) const
{
	if(m_strValue.length() > 0)
		Value = m_strValue[0]; 
}
//void xmlTag::GetValue(unsigned char** Value, size_t& nSize) const
//{
//	StringToHex(m_strValue, Value, nSize); 
//}

void xmlTag::SetValue(std::string Value)
{
	m_strValue = Value;
} 
void xmlTag::SetValue(int Value)
{
	m_strValue = toString<int>(Value);
} 
void xmlTag::SetValue(long Value)
{
	m_strValue = toString<long>(Value);
} 
void xmlTag::SetValue(unsigned long Value)
{
	m_strValue = toString<unsigned long>(Value);
} 
void xmlTag::SetValue(unsigned int Value)
{
    m_strValue = toString<unsigned int>(Value);
} 
void xmlTag::SetValue(float Value)
{
	if( (float)((int)Value) == Value )
	{
		m_strValue = toString<int>(Value);
	}
	else
	{
		m_strValue = toStringFormatted<float>(Value, -1, 7, false);
		LTrim(m_strValue, ' ');
		RTrim(m_strValue, '0');
	}
} 
void xmlTag::SetValue(double Value)
{
	if( (double)((int)Value) == Value )
	{
		m_strValue = toString<int>(Value);
	}
	else
	{
		m_strValue = toStringFormatted<double>(Value, -1, 16, false);
		LTrim(m_strValue, ' ');
		RTrim(m_strValue, '0');
	}
} 
void xmlTag::SetValue(bool Value)
{
	if(Value)
		m_strValue = "True";
	else
		m_strValue = "False";
} 
void xmlTag::SetValue(char Value)
{
	m_strValue = Value; 
}


std::string xmlTag::GetName() const
{
	return m_strTagName;
} 

void xmlTag::SetName(const std::string& strName)
{
	m_strTagName = strName;
} 

xmlTagArray& xmlTag::GetTagArray()
{
	return m_mapTags;
} 

xmlAttributeArray& xmlTag::GetAttributeArray()
{
	return m_mapAttributes;
} 

void xmlTag::WriteAttributes( std::ostream & inStream ) const
{
	for(size_t i = 0; i < m_mapAttributes.size(); i++)
	{
		inStream << " " 
				 << PostProcessString(m_mapAttributes[i]->m_strName)
				 << "=\""
				 << PostProcessString(m_mapAttributes[i]->m_strValue)
				 << "\"";
	}
}

xmlTag* xmlTag::GetParentTag() const
{
	return m_pParentTag;
}

void xmlTag::SetParentTag(xmlTag* pTag)
{
	m_pParentTag = pTag;
}

bool xmlTag::GetChildTagValue(const std::string& strTagName, std::string& strTagValue) const
{
	xmlTag_t* pChildTag;
	pChildTag = FindTag(strTagName);
	if(!pChildTag)
		return false;
	pChildTag->GetValue(strTagValue);
	return true;
}
bool xmlTag::GetChildTagValue(const std::string& strTagName, int& iTagValue) const
{
	xmlTag_t* pChildTag;
	pChildTag = FindTag(strTagName);
	if(!pChildTag)
		return false;
	pChildTag->GetValue(iTagValue);
	return true;
}
bool xmlTag::GetChildTagValue(const std::string& strTagName, long& lTagValue) const
{
	xmlTag_t* pChildTag;
	pChildTag = FindTag(strTagName);
	if(!pChildTag)
		return false;
	pChildTag->GetValue(lTagValue);
	return true;
}
bool xmlTag::GetChildTagValue(const std::string& strTagName, unsigned long& lTagValue) const
{
	xmlTag_t* pChildTag;
	pChildTag = FindTag(strTagName);
	if(!pChildTag)
		return false;
	pChildTag->GetValue(lTagValue);
	return true;
}
bool xmlTag::GetChildTagValue(const std::string& strTagName, unsigned int& lTagValue) const
{
	xmlTag_t* pChildTag;
	pChildTag = FindTag(strTagName);
	if(!pChildTag)
		return false;
	pChildTag->GetValue(lTagValue);
	return true;
}
bool xmlTag::GetChildTagValue(const std::string& strTagName, float& fTagValue) const
{
	xmlTag_t* pChildTag;
	pChildTag = FindTag(strTagName);
	if(!pChildTag)
		return false;
	pChildTag->GetValue(fTagValue);
	return true;
}
bool xmlTag::GetChildTagValue(const std::string& strTagName, double& dTagValue) const
{
	xmlTag_t* pChildTag;
	pChildTag = FindTag(strTagName);
	if(!pChildTag)
		return false;
	pChildTag->GetValue(dTagValue);
	return true;
}
bool xmlTag::GetChildTagValue(const std::string& strTagName, bool& bTagValue) const
{
	xmlTag_t* pChildTag;
	pChildTag = FindTag(strTagName);
	if(!pChildTag)
		return false;
	pChildTag->GetValue(bTagValue);
	return true;
}
bool xmlTag::GetChildTagValue(const std::string& strTagName, char& cTagValue) const
{
	xmlTag_t* pChildTag;
	pChildTag = FindTag(strTagName);
	if(!pChildTag)
		return false;
	pChildTag->GetValue(cTagValue);
	return true;
}

//bool xmlTag::GetChildTagValue(const std::string& strTagName, unsigned char** Value, size_t& nSize) const
//{
//	xmlTag_t* pChildTag;
//	pChildTag = FindTag(strTagName);
//	if(!pChildTag)
//		return false;
//	pChildTag->GetValue(Value, nSize);
//	return true;
//}

bool xmlTag::GetMultipleChildTagValue(const std::string& strTagName, std::vector<std::string>& Array) const
{
	size_t i;
	xmlTag_t* pChildTag;
	std::vector<xmlTag_t*> tagArray;
	std::string Value;

	Array.erase(Array.begin(), Array.end());
	tagArray = FindMultipleTag(strTagName);
	for(i = 0; i < tagArray.size(); i++)
	{
		pChildTag = tagArray[i];
		if(!pChildTag)
			return false;
		pChildTag->GetValue(Value);
		Array.push_back(Value);
	}
	return i == 0 ? false : true;
}
bool xmlTag::GetMultipleChildTagValue(const std::string& strTagName, std::vector<int>& Array) const
{
	size_t i;
	xmlTag_t* pChildTag;
	std::vector<xmlTag_t*> tagArray;
	int Value;

	Array.erase(Array.begin(), Array.end());
	tagArray = FindMultipleTag(strTagName);
	for(i = 0; i < tagArray.size(); i++)
	{
		pChildTag = tagArray[i];
		if(!pChildTag)
			return false;
		pChildTag->GetValue(Value);
		Array.push_back(Value);
	}
	return i == 0 ? false : true;
}
bool xmlTag::GetMultipleChildTagValue(const std::string& strTagName, std::vector<long>& Array) const
{
	size_t i;
	xmlTag_t* pChildTag;
	std::vector<xmlTag_t*> tagArray;
	long Value;

	Array.erase(Array.begin(), Array.end());
	tagArray = FindMultipleTag(strTagName);
	for(i = 0; i < tagArray.size(); i++)
	{
		pChildTag = tagArray[i];
		if(!pChildTag)
			return false;
		pChildTag->GetValue(Value);
		Array.push_back(Value);
	}
	return i == 0 ? false : true;
}
bool xmlTag::GetMultipleChildTagValue(const std::string& strTagName, std::vector<unsigned int>& Array) const
{
	size_t i;
	xmlTag_t* pChildTag;
	std::vector<xmlTag_t*> tagArray;
    unsigned int Value;

	Array.erase(Array.begin(), Array.end());
	tagArray = FindMultipleTag(strTagName);
	for(i = 0; i < tagArray.size(); i++)
	{
		pChildTag = tagArray[i];
		if(!pChildTag)
			return false;
		pChildTag->GetValue(Value);
		Array.push_back(Value);
	}
	return i == 0 ? false : true;
}
bool xmlTag::GetMultipleChildTagValue(const std::string& strTagName, std::vector<unsigned long>& Array) const
{
    size_t i;
    xmlTag_t* pChildTag;
    std::vector<xmlTag_t*> tagArray;
    unsigned long Value;

    Array.erase(Array.begin(), Array.end());
    tagArray = FindMultipleTag(strTagName);
    for(i = 0; i < tagArray.size(); i++)
    {
        pChildTag = tagArray[i];
        if(!pChildTag)
            return false;
        pChildTag->GetValue(Value);
        Array.push_back(Value);
    }
    return i == 0 ? false : true;
}
bool xmlTag::GetMultipleChildTagValue(const std::string& strTagName, std::vector<double>& Array) const
{
	size_t i;
	xmlTag_t* pChildTag;
	std::vector<xmlTag_t*> tagArray;
	double Value;

	Array.erase(Array.begin(), Array.end());
	tagArray = FindMultipleTag(strTagName);
	for(i = 0; i < tagArray.size(); i++)
	{
		pChildTag = tagArray[i];
		if(!pChildTag)
			return false;
		pChildTag->GetValue(Value);
		Array.push_back(Value);
	}
	return i == 0 ? false : true;
}
bool xmlTag::GetMultipleChildTagValue(const std::string& strTagName, std::vector<float>& Array) const
{
	size_t i;
	xmlTag_t* pChildTag;
	std::vector<xmlTag_t*> tagArray;
	float Value;

	Array.erase(Array.begin(), Array.end());
	tagArray = FindMultipleTag(strTagName);
	for(i = 0; i < tagArray.size(); i++)
	{
		pChildTag = tagArray[i];
		if(!pChildTag)
			return false;
		pChildTag->GetValue(Value);
		Array.push_back(Value);
	}
	return i == 0 ? false : true;
}
bool xmlTag::GetMultipleChildTagValue(const std::string& strTagName, std::vector<bool>& Array) const
{
	size_t i;
	xmlTag_t* pChildTag;
	std::vector<xmlTag_t*> tagArray;
	bool Value;

	Array.erase(Array.begin(), Array.end());
	tagArray = FindMultipleTag(strTagName);
	for(i = 0; i < tagArray.size(); i++)
	{
		pChildTag = tagArray[i];
		if(!pChildTag)
			return false;
		pChildTag->GetValue(Value);
		Array.push_back(Value);
	}
	return i == 0 ? false : true;
}
bool xmlTag::GetMultipleChildTagValue(const std::string& strTagName, std::vector<char>& Array) const
{
	size_t i;
	xmlTag_t* pChildTag;
	std::vector<xmlTag_t*> tagArray;
	char Value;

	Array.erase(Array.begin(), Array.end());
	tagArray = FindMultipleTag(strTagName);
	for(i = 0; i < tagArray.size(); i++)
	{
		pChildTag = tagArray[i];
		if(!pChildTag)
			return false;
		pChildTag->GetValue(Value);
		Array.push_back(Value);
	}
	return i == 0 ? false : true;
}







void xmlAttribute::GetValue(std::string& Value) const
{
	Value = m_strValue;
} 
void xmlAttribute::GetValue(int& Value) const
{
	Value = fromString<int>(m_strValue);
} 
void xmlAttribute::GetValue(long& Value) const
{
	Value = fromString<long>(m_strValue);
} 
void xmlAttribute::GetValue(unsigned long& Value) const
{
	Value = fromString<unsigned long>(m_strValue);
} 
void xmlAttribute::GetValue(unsigned int& Value) const
{
    Value = fromString<unsigned int>(m_strValue);
} 
void xmlAttribute::GetValue(float& Value) const
{
	Value = fromString<float>(m_strValue);
} 
void xmlAttribute::GetValue(double& Value) const
{
	Value = fromString<double>(m_strValue);
} 
void xmlAttribute::GetValue(bool& Value) const
{
	if(m_strValue == "true" || m_strValue == "True" || m_strValue == "1" || m_strValue == "yes")
		Value = true;
	else
		Value = false;
} 
void xmlAttribute::GetValue(char& Value) const
{
	if(m_strValue.length() > 0)
		Value = m_strValue[0]; 
}
//void xmlAttribute::GetValue(unsigned char** Value, size_t& nSize) const
//{
//	StringToHex(m_strValue, Value, nSize); 
//}

void xmlAttribute::SetValue(std::string Value)
{
	m_strValue = Value;
} 
void xmlAttribute::SetValue(int Value)
{
	m_strValue = toString<int>(Value);
} 
void xmlAttribute::SetValue(long Value)
{
	m_strValue = toString<long>(Value);
} 
void xmlAttribute::SetValue(unsigned long Value)
{
	m_strValue = toString<unsigned long>(Value);
} 
void xmlAttribute::SetValue(unsigned int Value)
{
    m_strValue = toString<unsigned int>(Value);
} 
void xmlAttribute::SetValue(float Value)
{
	m_strValue = toStringFormatted<float>(Value, -1, 20, false);
} 
void xmlAttribute::SetValue(double Value)
{
	m_strValue = toStringFormatted<double>(Value, -1, 30, false);
} 
void xmlAttribute::SetValue(bool Value)
{
	if(Value)
		m_strValue = "True";
	else
		m_strValue = "False";
} 
void xmlAttribute::SetValue(char Value)
{
	m_strValue = Value; 
}
//void xmlAttribute::SetValue(unsigned char* Value, size_t nSize)
//{
//	HexToString(Value, nSize, m_strValue);
//}

std::string xmlAttribute::GetName() const
{
	return m_strName;
} 

void xmlAttribute::SetName(const std::string& strName)
{
	m_strName = strName;
} 


}
}

