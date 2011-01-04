/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef XML_FILE
#define XML_FILE

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#ifdef DAEDLL
#ifdef MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllexport)
#else // MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllimport)
#endif // MODEL_EXPORTS
#else // DAEDLL
#define DAE_CORE_API
#endif // DAEDLL

#else // WIN32
#define DAE_CORE_API 
#endif // WIN32

#include <string>
#include <vector>
#include "io.h"
using namespace dae::io;

namespace dae 
{
class daeClassFactoryManager_t;

namespace xml
{
enum xmleDocumentType
{
	eXML,
	eHTML,
	eMathML
};

class xmlTag;
class  xmlReadStream
{
public:

   xmlReadStream (std::istream & inStream);
   ~xmlReadStream ();

   // identical functions
   // faster
   void readString(std::string & inString);
   // more convienient
   std::string readString();

protected:

   bool isStarter (const char c) const;
   bool isTerminator (const char c) const;
   bool isSeparator (const char c) const;

   void validateString(std::string & inString);

private:

   // disable copy constructor and assignment operator
   xmlReadStream ( const xmlReadStream & );
   xmlReadStream & operator = ( const xmlReadStream & );

   std::string theBuffer;
   std::istream & theStream;

   static std::string theSeparators;
   static char theTerminators;
   static char theStarters;

}; 

class DAE_CORE_API xmlFile
{
public:

	xmlFile();
	~xmlFile();

	bool Save(std::string strFileName);
	bool Save(const xmlTag* pTag, std::string strFileName);
	bool Open(std::string strFileName);

	xmlTag*			 GetRootTag();
	std::string		 GetFileName();
	std::string		 GetXSLTFileName();
	void			 SetXSLTFileName(std::string strXSLTfilename);
	std::string		 GetXMLVersion();
	void			 SetXMLVersion(std::string strXMLVersion);
	xmleDocumentType GetDocumentType();
	void			 SetDocumentType(xmleDocumentType eType);
	std::string   	 GetEncoding();
	void			 SetEncoding(std::string strEncoding);

protected:
	void        Parse(xmlReadStream& inStream);
	std::string FormatHeaders();
	bool		ParseHeaders();

protected:
	xmlTag*			 m_pRootTag;
	std::string		 m_strXMLVersion;
	std::string		 m_strXSLTFileName;
	std::string		 m_strFileName;
	xmleDocumentType m_eDocumentType;
	std::string		 m_strEncoding;

	std::string      m_strXMLInfoString;
	std::string      m_strXSLTInfoString;
	std::string      m_strDescription;
}; 


class DAE_CORE_API xmlException : public std::exception
{
public:
	xmlException(const char * lpszDescription);
	xmlException(std::string& strDescription);
        virtual ~xmlException() throw();

public:
        virtual const char* what() const throw();

public:
	std::string m_strDescription;
};


class DAE_CORE_API xmlAttribute : public io::xmlAttribute_t
{
public:
	xmlAttribute()
	{	
		m_pParentTag = NULL;
	}
	xmlAttribute(const xmlAttribute& rAttribute)
	{
		m_strValue		= rAttribute.m_strValue;
		m_strName		= rAttribute.m_strName;
		m_pParentTag	= rAttribute.m_pParentTag;
	};
	xmlAttribute* Clone()
	{
		return new xmlAttribute(*this);
	};

public:
	virtual std::string	GetName() const;
	virtual void		SetName(const std::string& strName);

	virtual void GetValue(std::string& Value) const;
	virtual void GetValue(int& Value) const;
	virtual void GetValue(long& Value) const;
	virtual void GetValue(unsigned long& Value) const;
    virtual void GetValue(unsigned int& Value) const;
	virtual void GetValue(float& Value) const;
	virtual void GetValue(double& Value) const;
	virtual void GetValue(bool& Value) const;
	virtual void GetValue(char& Value) const;

	virtual void SetValue(std::string Value);
	virtual void SetValue(int Value);
	virtual void SetValue(long Value);
	virtual void SetValue(unsigned long Value);
    virtual void SetValue(unsigned int Value);
	virtual void SetValue(float Value);
	virtual void SetValue(double Value);
	virtual void SetValue(bool Value);
	virtual void SetValue(char Value);

public:
	std::string m_strName;
	std::string m_strValue;
	xmlTag*		m_pParentTag;
};

typedef std::vector<xmlTag*>		xmlTagArray;
typedef std::vector<xmlAttribute*>	xmlAttributeArray;

class DAE_CORE_API xmlTag : public io::xmlTag_t
{
public:
	xmlTag();
	xmlTag(const xmlTag & rTag);
	xmlTag* Clone();
	virtual ~xmlTag();
	xmlTag& operator=(const xmlTag & rTag);
	void EmptyMaps();

public:
	std::string	GetName() const;
	void		SetName(const std::string& strName);

	void GetValue(std::string& Value) const;
	void GetValue(int& Value) const;
	void GetValue(long& Value) const;
	void GetValue(unsigned long& Value) const;
    void GetValue(unsigned int& Value) const;
	void GetValue(float& Value) const;
	void GetValue(double& Value) const;
	void GetValue(bool& Value) const;
	void GetValue(char& Value) const;
//	void GetValue(unsigned char** Value, size_t& nSize) const;

	void SetValue(std::string Value);
	void SetValue(int Value);
	void SetValue(long Value);
	void SetValue(unsigned long Value);
    void SetValue(unsigned int Value);
	void SetValue(float Value);
	void SetValue(double Value);
	void SetValue(bool Value);
	void SetValue(char Value);
//	void SetValue(unsigned char* Value, size_t nSize);

	xmlTag_t* AddTag(const std::string& strName);
	xmlTag_t* AddTag(const std::string& strName, std::string Value);
	xmlTag_t* AddTag(const std::string& strName, float Value);
	xmlTag_t* AddTag(const std::string& strName, double Value);
	xmlTag_t* AddTag(const std::string& strName, int Value);
	xmlTag_t* AddTag(const std::string& strName, long Value);
	xmlTag_t* AddTag(const std::string& strName, unsigned long Value);
    xmlTag_t* AddTag(const std::string& strName, unsigned int Value);
	xmlTag_t* AddTag(const std::string& strName, bool Value);
	xmlTag_t* AddTag(const std::string& strName, char Value);
//	xmlTag_t* AddTag(const std::string& strName, unsigned char* Value, size_t nSize);

	xmlAttribute_t* AddAttribute(const std::string& strName);
	xmlAttribute_t* AddAttribute(const std::string& strName, std::string Value);
	xmlAttribute_t* AddAttribute(const std::string& strName, float Value);
	xmlAttribute_t* AddAttribute(const std::string& strName, double Value);
	xmlAttribute_t* AddAttribute(const std::string& strName, int Value);
	xmlAttribute_t* AddAttribute(const std::string& strName, long Value);
	xmlAttribute_t* AddAttribute(const std::string& strName, unsigned long Value);
    xmlAttribute_t* AddAttribute(const std::string& strName, unsigned int Value);
	xmlAttribute_t* AddAttribute(const std::string& strName, bool Value);
	xmlAttribute_t* AddAttribute(const std::string& strName, char Value);

	xmlTag_t*				FindTag(const std::string& strName) const;
	std::vector<xmlTag_t*>	FindMultipleTag(const std::string& strName) const;
	xmlAttribute_t*			FindAttribute(const std::string& strName) const;


	bool GetChildTagValue(const std::string& strTagName, std::string& strTagValue) const;
	bool GetChildTagValue(const std::string& strTagName, int& iTagValue) const;
	bool GetChildTagValue(const std::string& strTagName, long& lTagValue) const;
    bool GetChildTagValue(const std::string& strTagName, unsigned int& nTagValue) const;
	bool GetChildTagValue(const std::string& strTagName, float& fTagValue) const;
	bool GetChildTagValue(const std::string& strTagName, double& dTagValue) const;
	bool GetChildTagValue(const std::string& strTagName, bool& bTagValue) const;
	bool GetChildTagValue(const std::string& strTagName, char& cTagValue) const;
	bool GetChildTagValue(const std::string& strTagName, unsigned long& lTagValue) const;
//	bool GetChildTagValue(const std::string& strTagName, unsigned char** Value, size_t& nSize) const;

	bool GetMultipleChildTagValue(const std::string& strTagName, std::vector<std::string>& Array) const;
	bool GetMultipleChildTagValue(const std::string& strTagName, std::vector<int>& Array) const;
	bool GetMultipleChildTagValue(const std::string& strTagName, std::vector<long>& Array) const;
    bool GetMultipleChildTagValue(const std::string& strTagName, std::vector<unsigned long>& Array) const;
    bool GetMultipleChildTagValue(const std::string& strTagName, std::vector<unsigned int>& Array) const;
    bool GetMultipleChildTagValue(const std::string& strTagName, std::vector<double>& Array) const;
	bool GetMultipleChildTagValue(const std::string& strTagName, std::vector<float>& Array) const;
	bool GetMultipleChildTagValue(const std::string& strTagName, std::vector<bool>& Array) const;
	bool GetMultipleChildTagValue(const std::string& strTagName, std::vector<char>& Array) const;

	bool ContainTags() const;
	bool ContainAttributes() const;

	std::string GetPath(char cSeparator = '\\') const;

	bool		  SetAttribute(const std::string& strAttributeName, const std::string& strValue);
	std::string   GetAttribute(const std::string& strAttributeName) const;
	xmlAttribute* GetAttributePtr(const std::string & strAttributeName);

	xmlTagArray&		GetTagArray();
	xmlAttributeArray&	GetAttributeArray();

	void AddTag(xmlTag* pNewTag);

	void InsertTag(xmlTag* pTag, size_t iPosition);
	bool RemoveTag(xmlTag* pTagToRemove);
	bool RemoveAttribute(const std::string& strAttributeName);

	std::ostream&  Save(size_t iLevel, std::ostream & inStream) const;
	xmlReadStream& Parse(const std::string & inStart, xmlReadStream & inStream);

	xmlTag* GetParentTag() const;
	void    SetParentTag(xmlTag* pTag);

protected:
	static std::string m_strLevel;
	
protected:
	xmlTag&		 ParseAttributes(const std::string & inStart);
	virtual void WriteAttributes(std::ostream & inStream) const;

	// replace special characters with escape seq
	std::string PostProcessString(const std::string & inAttribute) const;
	// replace escape seq with special characters 
	std::string PreProcessString(const std::string & inAttribute) const;

	xmlTagArray			m_mapTags;
	xmlAttributeArray	m_mapAttributes;
	std::string			m_strValue;
	std::string			m_strTagName;
	xmlTag*				m_pParentTag;

}; 

DAE_CORE_API bool    GetValue(xml::xmlTag* pTag, std::string& strChildTagName,std::string& strValue);
DAE_CORE_API bool    GetMultipleValue(xml::xmlTag* pTag, std::string& strChildTagName, std::vector<std::string>& strarrValues);
DAE_CORE_API bool    InsertMultipleTag(xml::xmlTag* pTag, std::string& strTagName, std::vector<std::string>& strarrValues);
DAE_CORE_API xmlTag* InsertTag(xml::xmlTag* pTag, std::string& strTagName, std::string& strValue);


/********************************************************************
	Open/Save objects that:
	 - need to be created with new
	 - need to be created with wxCreateDynamicObject
*********************************************************************/
template<class ObjectType, class ContainerType>
void OpenObjects(xmlTag* pTag, const daeClassFactoryManager_t* pCFManager, std::vector<ContainerType*>& ptrarrObjects, const string& strParentTagName, const string& strChildTagName)
{
	ObjectType* pObject;
	xmlTag* pChildTag, *pTagParent;
	std::vector<xmlTag*> arrTags;

	if(!pTag)
	{
		string strError = "Invalid parent tag while opening " + strParentTagName + " tag";
		throw new xmlException(strError);
	}

	pTagParent = pTag->FindTag(strParentTagName);
	if(!pTagParent)
	{
		string strError = "Error opening " + strParentTagName + " tag";
		throw new xmlException(strError);
	}

	arrTags = pTagParent->FindMultipleTag(strChildTagName);
	for(size_t i = 0; i < arrTags.size(); i++)
	{
		pChildTag = arrTags[i];
		if(!pChildTag)
		{
			string strError = "Error opening " + strChildTagName + " tag in " + strParentTagName + " tag";
			throw new xmlException(strError);
		}

		pObject = new ObjectType();
		if(!pObject)
		{
			string strError = "Cannot create object ";
			throw new xmlException(strError);
		}
		ptrarrObjects.push_back(pObject);
		pObject->Open(pChildTag, pCFManager);
	}

}

//template<class ContainerType>
//void OpenCreateObjects(xmlTag* pTag, std::vector<ContainerType*>& ptrarrObjects, const string& strParentTagName, const string& strChildTagName)
//{
//	string strName, strClass;
//	wxObject* pwxObject;
//	ContainerType* pObject;
//	xmlTag* pChildTag, *pTagParent;
//	std::vector<xmlTag*> arrTags;
//
//	if(!pTag)
//	{
//		string strError = "Invalid parent tag while opening " + strParentTagName + " tag";
//		throw new xmlException(strError);
//	}
//
//	pTagParent = pTag->FindTag(strParentTagName);
//	if(!pTagParent)
//	{
//		string strError = "Error opening " + strParentTagName + " tag";
//		throw new xmlException(strError);
//	}
//
//	arrTags = pTagParent->FindMultipleTag(strChildTagName);
//	for(size_t i = 0; i < arrTags.size(); i++)
//	{
//		pChildTag = arrTags[i];
//		if(!pChildTag)
//		{
//			string strError = "Error opening " + strChildTagName + " tag in " + strParentTagName + " tag";
//			throw new xmlException(strError);
//		}
//
//		strName = "Class";
//		if(!pChildTag->GetChildTagValue(strName, strClass))
//		{
//			string strError = "Cannot find Class tag in " + strChildTagName + " tag";
//			throw new xmlException(strError);
//		}
//		if(strClass.empty())
//		{
//			string strError = "Class tag cannot be an empty value";
//			throw new xmlException(strError);
//		}
//
//		pwxObject = wxCreateDynamicObject(strClass.c_str());
//		if(!pwxObject)
//		{
//			string strError = "Cannot create object " + strClass;
//			throw new xmlException(strError);
//		}
//
//		pObject = dynamic_cast<ContainerType*>(pwxObject);
//		if(!pObject)
//		{
//			string strError = "Cannot convert wxObject* pointer to " + strClass + "* pointer";
//			throw new xmlException(strError);
//		}
//
//		ptrarrObjects.push_back(pObject);
//		pObject->Open(pChildTag);
//	}
//}

template<class ObjectType, class ContainerType>
void SaveObjects(xmlTag* pTag, const std::vector<ContainerType*>& ptrarrObjects, const string& strParentTagName, const string& strChildTagName)
{
	ObjectType* pObject;
	ContainerType* pContainerObject;
	xmlTag* pChildTag, *pTagParent;

	if(!pTag)
	{
		string strError = "Invalid parent tag while saving " + strParentTagName + " tag";
		throw new xmlException(strError);
	}

	pTagParent = pTag->AddTag(strParentTagName);
	for(size_t i = 0; i < ptrarrObjects.size(); i++)
	{
		pContainerObject = ptrarrObjects[i];
		if(!pContainerObject)
		{
			string strError = "Invalid drawing object in " + strParentTagName + " tag";
			throw new xmlException(strError);
		}

		pObject = dynamic_cast<ObjectType*>(pContainerObject);
		if(pObject)
		{
			pChildTag = pTagParent->AddTag(strChildTagName);
			pObject->Save(pChildTag);
		}
	}
}

/********************************************************************
	Open/Save array of values that don't need to be created with new
	          like std::vector<int>, std::vector<double>, ...
*********************************************************************/
template<class ObjectType>
void OpenArray(xmlTag* pTag, std::vector<ObjectType>& arrValues, const string& strTagName)
{
	ObjectType Value;
	xmlTag* pChildTag, *pTagParent;
	std::vector<xmlTag*> arrTags;
	string strName = "Item";

	if(!pTag)
	{
		string strError = "Invalid parent tag while opening " + strTagName + " tag";
		throw new xmlException(strError);
	}

	pTagParent = pTag->FindTag(strTagName);
	if(!pTagParent)
	{
		string strError = "Error opening " + strTagName + " tag";
		throw new xmlException(strError);
	}

	arrTags = pTagParent->FindMultipleTag(strName);
	for(size_t i = 0; i < arrTags.size(); i++)
	{
		pChildTag = arrTags[i];
		if(!pChildTag)
		{
			string strError = "Error opening " + strTagName + " tag";
			throw new xmlException(strError);
		}

		pChildTag->GetValue(Value);
		arrValues.push_back(Value);
	}
}

template<class ObjectType>
void SaveArray(xmlTag* pTag, const std::vector<ObjectType>& arrValues, const string& strTagName)
{
	if(!pTag)
	{
		string strError = "Invalid parent tag while saving " + strTagName + " tag";
		throw new xmlException(strError);
	}

	string strName = "Item";
	xmlTag* pTagParent = pTag->AddTag(strTagName);
	for(size_t i = 0; i < arrValues.size(); i++)
		pTagParent->AddTag(strName, arrValues[i]);
}

/********************************************************************
	Open/Save object that:
	 - already exists
	 - needs to be created with new
	 - needs to be created with CreateDynamicObject
*********************************************************************/
template<class ObjectType>
void OpenObject(xmlTag* pTag, const daeClassFactoryManager_t* pCFManager, const string& strTagName, ObjectType* pObject)
{
	xmlTag* pChildTag;

	if(!pTag)
	{
		string strError = "Invalid parent tag while saving " + strTagName + " tag";
		throw new xmlException(strError);
	}
	if(!pObject)
	{
		string strError = "Invalid object for " + strTagName;
		throw new xmlException(strError);
	}

	pChildTag = pTag->FindTag(strTagName);
	if(!pChildTag)
	{
		string strError = "Error opening " + strTagName + " tag";
		throw new xmlException(strError);
	}
	pObject->Open(pChildTag, pCFManager);
}

template<class ObjectType>
ObjectType* OpenNewObject(xmlTag* pTag, const daeClassFactoryManager_t* pCFManager, const string& strTagName)
{
	ObjectType* pObject;
	xmlTag* pChildTag;

	if(!pTag)
	{
		string strError = "Invalid parent tag while saving " + strTagName + " tag";
		throw new xmlException(strError);
	}

	pChildTag = pTag->FindTag(strTagName);
	if(!pChildTag)
	{
		string strError = "Error opening " + strTagName + " tag";
		throw new xmlException(strError);
	}

	pObject = new ObjectType();
	if(!pObject)
	{
		string strError = "Cannot allocate object for " + strTagName;
		throw new xmlException(strError);
	}

	pObject->Open(pChildTag, pCFManager);
	return pObject;
}

//template<class ObjectType>
//ObjectType* OpenCreateObject(xmlTag* pTag, const string& strTagName)
//{
//	string strName, strClass;
//	wxObject* pwxObject;
//	ObjectType* pObject;
//	xmlTag* pChildTag;
//
//	if(!pTag)
//	{
//		string strError = "Invalid parent tag while saving " + strTagName + " tag";
//		throw new xmlException(strError);
//	}
//
//	pChildTag = pTag->FindTag(strTagName);
//	if(!pChildTag)
//	{
//		string strError = "Error opening " + strTagName + " tag";
//		throw new xmlException(strError);
//	}
//
//	strName = "Class";
//	if(!pChildTag->GetChildTagValue(strName, strClass))
//	{
//		string strError = "Cannot find Class tag in " + strTagName + " tag";
//		throw new xmlException(strError);
//	}
//	if(strClass.empty())
//	{
//		string strError = "Class tag cannot be an empty value";
//		throw new xmlException(strError);
//	}
//
//	pwxObject = wxCreateDynamicObject(strClass.c_str());
//	if(!pwxObject)
//	{
//		string strError = "Cannot create object " + strClass;
//		throw new xmlException(strError);
//	}
//
//	pObject = dynamic_cast<ObjectType*>(pwxObject);
//	if(!pObject)
//	{
//		string strError = "Cannot convert wxObject* pointer to " + strClass + "* pointer";
//		throw new xmlException(strError);
//	}
//
//	pObject->Open(pChildTag);
//	return pObject;
//}

template<class ObjectType>
xmlTag* SaveObject(xmlTag* pTag, const string& strTagName, const ObjectType* pObject)
{
	if(!pTag)
	{
		string strError = "Invalid parent tag while saving " + strTagName + " tag";
		throw new xmlException(strError);
	}
	if(!pObject)
	{
		string strError = "Invalid object in " + strTagName;
		throw new xmlException(strError);
	}

	xmlTag* pChildTag = pTag->AddTag(strTagName);
	pObject->Save(pChildTag);
	return pChildTag;
}

/********************************************************************
	Open/Save simple data like int, double, string, ...
*********************************************************************/
template<class ObjectType>
void OpenSimple(xmlTag* pTag, const string& strTagName, ObjectType& Value)
{
	if(!pTag)
	{
		string strError = "Invalid parent tag while saving " + strTagName + " tag";
		throw new xmlException(strError);
	}
	if(!pTag->GetChildTagValue(strTagName, Value))
	{
		string strError = "Error opening " + strTagName + " tag";
		throw new xmlException(strError);
	}
}

template<class ObjectType>
void SaveSimple(xmlTag* pTag, const string& strTagName, const ObjectType& Value)
{
	if(!pTag)
	{
		string strError = "Invalid parent tag while saving " + strTagName + " tag";
		throw new xmlException(strError);
	}
	pTag->AddTag(strTagName, Value);
}

/********************************************************************
	Open/Save enum data
*********************************************************************/
template<class EnumType>
void OpenEnumerated(xmlTag* pTag, const string& strTagName, EnumType& Value)
{
	int iValue;

	if(!pTag)
	{
		string strError = "Invalid parent tag while saving " + strTagName + " tag";
		throw new xmlException(strError);
	}
	if(!pTag->GetChildTagValue(strTagName, iValue))
	{
		string strError = "Error opening " + strTagName + " tag";
		throw new xmlException(strError);
	}
	Value = (EnumType)iValue;

}

template<class EnumType>
void SaveEnumerated(xmlTag* pTag, const string& strTagName, const EnumType& Value)
{
	if(!pTag)
	{
		string strError = "Invalid parent tag while saving " + strTagName + " tag";
		throw new xmlException(strError);
	}
	pTag->AddTag(strTagName, (int)Value);
}



}
}

#endif
