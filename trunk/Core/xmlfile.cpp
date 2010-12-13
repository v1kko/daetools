#include "stdafx.h"
#include "xmlfile.h"
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

#include "helpers.h"

namespace dae 
{
namespace xml
{

xmlFile::xmlFile ()
{
	m_eDocumentType		= eXML;
	m_strEncoding		= "iso-8859-1";
	m_pRootTag			= new xmlTag;
	m_strFileName		= "";
	m_strXMLVersion		= "1.0";
	m_strXSLTFileName	= "";
} 

xmlFile::~xmlFile ()
{
	if(m_pRootTag)
		delete m_pRootTag;
} 

bool xmlFile::Save(std::string strFileName)
{
	if(strFileName.empty())
		return false;
	m_strFileName = strFileName;
	ofstream aStream (m_strFileName.c_str());
	if(!aStream.is_open())
		return false;

	aStream << FormatHeaders();

	m_pRootTag->Save(0, aStream);
	return true;
} 

bool xmlFile::Save(const xmlTag* pTag, std::string strFileName)
{
	if(!pTag)
		return false;
	if(strFileName.empty())
		return false;

	ofstream aStream (strFileName.c_str());
	if(!aStream.is_open())
		return false;

	aStream << FormatHeaders();

	pTag->Save(0, aStream);
	return true;
}

bool xmlFile::Open(std::string strFileName)
{
	m_strFileName = strFileName;
	ifstream aStream (m_strFileName.c_str());
	if(!aStream.is_open())
		return false;

	xmlReadStream anXMLStream (aStream);
	Parse(anXMLStream);
	if(!ParseHeaders())
		return false;

	return true;
} 

void xmlFile::Parse(xmlReadStream& inStream)
{
	string aString;
	for(;;) 
	{
		inStream.readString(aString);
		if (aString.length() == 0)
		{
			// empty string indicates end of file
			break;
		}		
		if(aString.find("<?xml ") == 0 || aString.find("<?XML ") == 0)
		{
			// it is an XML description field
			m_strXMLInfoString = aString;
			continue;
		}
		if(aString.find("<?xml-stylesheet ") == 0 || aString.find("<?XML-STYLESHEET ") == 0)
		{
			// it is an XSLT stylesheet field
			m_strXSLTInfoString = aString;
			continue;
		}
		if(aString.find("<!") == 0)
		{
			// it is an XML description field
			m_strDescription = aString;
			continue;
		}
		if(aString.find("<") == 0)
		{
			m_pRootTag->SetParentTag(NULL);
			m_pRootTag->Parse(aString, inStream);
			break;
		}		
	} 
} 

bool xmlFile::ParseHeaders()
{
	size_t iFound, iSep1, iSep2;

// Parse version and encoding
	iFound = m_strXMLInfoString.find("<?xml version=", 0);
	iSep1  = m_strXMLInfoString.find("\"", iFound+1);
	iSep2  = m_strXMLInfoString.find("\"", iSep1+1);
	if(iFound == std::string::npos)
		return false;
	if(iSep1 <= iFound)
		return false;
	if(iSep2 <= iFound)
		return false;
	if(iSep2 <= iSep1)
		return false;
	m_strXMLVersion = m_strXMLInfoString.substr(iSep1+1, iSep2-iSep1-1);

	iFound = m_strXMLInfoString.find("encoding=", iSep2);
	iSep1  = m_strXMLInfoString.find("\"", iFound+1);
	iSep2  = m_strXMLInfoString.find("\"", iSep1+1);
	if(iFound == std::string::npos)
		return false;
	if(iSep1 <= iFound)
		return false;
	if(iSep2 <= iFound)
		return false;
	if(iSep2 <= iSep1)
		return false;
	m_strEncoding = m_strXMLInfoString.substr(iSep1+1, iSep2-iSep1-1);

// Parse xslt file
	iFound = m_strXSLTInfoString.find("<?xml-stylesheet type=\"text/xsl\" href=", 0);
	if(iFound == std::string::npos)// XSLT file is not mandatory
		return true;
	iFound = m_strXSLTInfoString.find(" href=", 0);
	iSep1  = m_strXSLTInfoString.find("\"", iFound+1);
	iSep2  = m_strXSLTInfoString.find("\"", iSep1+1);
	if(iFound == string::npos)
		return false;
	if(iSep1 <= iFound)
		return false;
	if(iSep2 <= iFound)
		return false;
	if(iSep2 <= iSep1)
		return false;
	m_strXSLTFileName = m_strXSLTInfoString.substr(iSep1+1, iSep2-iSep1-1);

	return true;
}


std::string xmlFile::FormatHeaders()
{
	std::string strEncoding, strResult;

	if(m_eDocumentType == eXML)
	{
		strResult += "<?xml "; 
		strResult += "version=\"";
		strResult += m_strXMLVersion; 
		strResult += "\" encoding=\""; 
		strResult += m_strEncoding; 
		strResult += "\"?>\n"; 

		if(!m_strXSLTFileName.empty())
		{
			strResult += "<?xml-stylesheet type=\"text/xsl\" href=\""; 
			strResult += m_strXSLTFileName; 
			strResult += "\"?>\n";			
		};
	}
	else if(m_eDocumentType == eMathML)
	{
		strResult += "<?xml "; 
		strResult += "version=\"";
		strResult += m_strXMLVersion; 
		strResult += "\"?>\n"; 

		strResult += "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.1 plus MathML 2.0//EN\" \n";
		strResult += "       \"http://www.w3.org/TR/MathML2/dtd/xhtml-math11-f.dtd\"[ \n";
		strResult += "        <!ENTITY mathml \"http://www.w3.org/1998/Math/MathML\" > \n";
		strResult += "]> \n";

		if(!m_strXSLTFileName.empty())
		{
			strResult += "<?xml-stylesheet type=\"text/xsl\" href=\""; 
			strResult += m_strXSLTFileName; 
			strResult += "\"?>\n";			
		};
	}
	else
	{
		// Do nothing since it is the html file and it needs no headers
	}

	return strResult;
}

xmlTag* xmlFile::GetRootTag()
{
	return m_pRootTag;
}

std::string xmlFile::GetFileName()
{
	return m_strFileName;
}

std::string xmlFile::GetXSLTFileName()
{
	return m_strXSLTFileName;
}

void xmlFile::SetXSLTFileName(std::string strXSLTfilename)
{
	m_strXSLTFileName = strXSLTfilename;
}

std::string xmlFile::GetXMLVersion()
{
	return m_strXMLVersion;
}

void xmlFile::SetXMLVersion(std::string strXMLVersion)
{
	m_strXMLVersion = strXMLVersion;
}

xmleDocumentType xmlFile::GetDocumentType()
{
	return m_eDocumentType;
}

void xmlFile::SetDocumentType(xmleDocumentType eType)
{
	m_eDocumentType = eType;
}

std::string xmlFile::GetEncoding()
{
	return m_strEncoding;
}

void xmlFile::SetEncoding(std::string strEncoding)
{
	m_strEncoding = strEncoding;
}


}
}
