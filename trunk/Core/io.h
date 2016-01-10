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
#ifndef DAE_IO_H
#define DAE_IO_H

#include <string>
#include <vector>
#include <iomanip>
#include "definitions.h"
#include <boost/smart_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>

namespace dae
{
namespace io
{
/********************************************************************
	daeOnOpenObjectDelegate_t
*********************************************************************/
template<class TYPE>
class daeOnOpenObjectDelegate_t
{
public:
	virtual ~daeOnOpenObjectDelegate_t(void){}

public:
	virtual void BeforeOpenObject(TYPE* pObject) = 0;
	virtual void AfterOpenObject(TYPE* pObject)  = 0;
};

/********************************************************************
	daeOnOpenObjectArrayDelegate_t
*********************************************************************/
template<class TYPE>
class daeOnOpenObjectArrayDelegate_t : public daeOnOpenObjectDelegate_t<TYPE>
{
public:
	virtual void AfterAllObjectsOpened(void)  = 0;
};

/********************************************************************
	daeOnOpenRefDelegate_t
*********************************************************************/
template<class TYPE>
class daeOnOpenRefDelegate_t
{
public:
	virtual ~daeOnOpenRefDelegate_t(void){}

public:
	virtual TYPE* FindObjectByID(size_t id) = 0;
};


/********************************************************************
	xmlAttribute_t
*********************************************************************/
class xmlAttribute_t
{
public:
	virtual ~xmlAttribute_t(void){}

public:
	virtual std::string	GetName() const						= 0;
	virtual void		SetName(const std::string& strName) = 0;

	virtual void GetValue(std::string& Value) const		= 0;
	virtual void GetValue(int& Value) const				= 0;
	virtual void GetValue(long& Value) const			= 0;
	virtual void GetValue(unsigned long long& Value) const	= 0;
    virtual void GetValue(unsigned long& Value) const   = 0;
    virtual void GetValue(unsigned int& Value) const	= 0;
	virtual void GetValue(float& Value) const			= 0;
	virtual void GetValue(double& Value) const			= 0;
	virtual void GetValue(bool& Value) const			= 0;
	virtual void GetValue(char& Value) const			= 0;

	virtual void SetValue(std::string Value)	= 0;
	virtual void SetValue(int Value)			= 0;
    virtual void SetValue(long Value)           = 0;
	virtual void SetValue(unsigned long long Value)	= 0;
    virtual void SetValue(unsigned long Value)  = 0;
    virtual void SetValue(unsigned int Value)	= 0;
	virtual void SetValue(float Value)			= 0;
	virtual void SetValue(double Value)			= 0;
	virtual void SetValue(bool Value)			= 0;
	virtual void SetValue(char Value)			= 0;
};

/********************************************************************
	xmlTag_t
*********************************************************************/
class xmlTag_t
{
public:
	virtual ~xmlTag_t(void){}

public:
	virtual std::string	GetName() const						= 0;
	virtual void		SetName(const std::string& strName) = 0;

	virtual void GetValue(std::string& Value) const		= 0;
	virtual void GetValue(int& Value) const				= 0;
	virtual void GetValue(long& Value) const			= 0;
	virtual void GetValue(unsigned long long& Value) const	= 0;
    virtual void GetValue(unsigned long& Value) const   = 0;
    virtual void GetValue(unsigned int& Value) const	= 0;
	virtual void GetValue(float& Value) const			= 0;
	virtual void GetValue(double& Value) const			= 0;
	virtual void GetValue(bool& Value) const			= 0;
	virtual void GetValue(char& Value) const			= 0;

	virtual void SetValue(std::string Value)	= 0;
	virtual void SetValue(int Value)			= 0;
	virtual void SetValue(long Value)			= 0;
    virtual void SetValue(unsigned long long Value)  = 0;
	virtual void SetValue(unsigned long Value)	= 0;
    virtual void SetValue(unsigned int Value)	= 0;
	virtual void SetValue(float Value)			= 0;
	virtual void SetValue(double Value)			= 0;
	virtual void SetValue(bool Value)			= 0;
	virtual void SetValue(char Value)			= 0;

	virtual xmlTag_t* AddTag(const std::string& strName)						= 0;
	virtual xmlTag_t* AddTag(const std::string& strName, std::string Value)		= 0;
	virtual xmlTag_t* AddTag(const std::string& strName, float Value)			= 0;
	virtual xmlTag_t* AddTag(const std::string& strName, double Value)			= 0;
	virtual xmlTag_t* AddTag(const std::string& strName, int Value)				= 0;
	virtual xmlTag_t* AddTag(const std::string& strName, long Value)			= 0;
    virtual xmlTag_t* AddTag(const std::string& strName, unsigned long long Value)   = 0;
	virtual xmlTag_t* AddTag(const std::string& strName, unsigned long Value)	= 0;
    virtual xmlTag_t* AddTag(const std::string& strName, unsigned int Value)	= 0;
	virtual xmlTag_t* AddTag(const std::string& strName, bool Value)			= 0;
	virtual xmlTag_t* AddTag(const std::string& strName, char Value)			= 0;

	virtual xmlAttribute_t* AddAttribute(const std::string& strName)						= 0;
	virtual xmlAttribute_t* AddAttribute(const std::string& strName, std::string Value)		= 0;
	virtual xmlAttribute_t* AddAttribute(const std::string& strName, float Value)			= 0;
	virtual xmlAttribute_t* AddAttribute(const std::string& strName, double Value)			= 0;
	virtual xmlAttribute_t* AddAttribute(const std::string& strName, int Value)				= 0;
	virtual xmlAttribute_t* AddAttribute(const std::string& strName, long Value)			= 0;
	virtual xmlAttribute_t* AddAttribute(const std::string& strName, unsigned long long Value)	= 0;
    virtual xmlAttribute_t* AddAttribute(const std::string& strName, unsigned long Value)   = 0;
    virtual xmlAttribute_t* AddAttribute(const std::string& strName, unsigned int Value)	= 0;
	virtual xmlAttribute_t* AddAttribute(const std::string& strName, bool Value)			= 0;
	virtual xmlAttribute_t* AddAttribute(const std::string& strName, char Value)			= 0;

	virtual xmlTag_t*				FindTag(const std::string& strName) const			= 0;
	virtual std::vector<xmlTag_t*>	FindMultipleTag(const std::string& strName) const	= 0;
	virtual xmlAttribute_t*			FindAttribute(const std::string& strName) const		= 0;

/********************************************************************
	daeOpen
*********************************************************************/
	template<class TYPE>
	void Open(const std::string& strName, TYPE& Value)
	{
		xmlTag_t* pChildTag = this->FindTag(strName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		pChildTag->GetValue(Value);
	}

/******************************************************************
	daeSave
*******************************************************************/
	template<class TYPE>
	void Save(const std::string& strName, const TYPE& Value)
	{
		this->AddTag(strName, Value);
	}

/******************************************************************
	OpenArray
*******************************************************************/
	template<class TYPE>
	void OpenArray(const string& strArrayName, std::vector<TYPE>& arrObjects, string strItemName = string("Item"))
	{
		TYPE item;
		xmlTag_t* pItemTag;

		xmlTag_t* pChildTag = this->FindTag(strArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		std::vector<xmlTag_t*> arrTags = pChildTag->FindMultipleTag(strItemName);
		for(size_t i = 0; i < arrTags.size(); i++)
		{
			pItemTag = arrTags[i];
			if(!pItemTag)
				daeDeclareAndThrowException(exXMLIOError);

			pItemTag->GetValue(item);
			arrObjects.push_back(item);
		}
	}

	template<class TYPE>
	void OpenArray(const string& strArrayName, TYPE** pValues, size_t& N, string strItemName = string("Item"))
	{
		TYPE item;
		xmlTag_t* pItemTag;

		xmlTag_t* pChildTag = this->FindTag(strArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		std::vector<xmlTag_t*> arrTags = pChildTag->FindMultipleTag(strItemName);
		N = arrTags.size();
		if(N == 0)
			daeDeclareAndThrowException(exXMLIOError);

		*pValues = new TYPE[N];

		for(size_t i = 0; i < arrTags.size(); i++)
		{
			pItemTag = arrTags[i];

			pItemTag->GetValue(item);
			(*pValues)[i] = item;
		}
	}

/******************************************************************
	SaveArray
*******************************************************************/
	template<class TYPE>
	void SaveArray(const string& strArrayName, const std::vector<TYPE>& arrObjects, string strItemName = string("Item"))
	{
		xmlTag_t* pChildTag = this->AddTag(strArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		for(size_t i = 0; i < arrObjects.size(); i++)
		{
			pChildTag->Save<TYPE>(strItemName, arrObjects[i]);
		}
	}

	template<class TYPE>
	void SaveArray(const string& strArrayName, const TYPE* pValues, const size_t N, string strItemName = string("Item"))
	{
		xmlTag_t* pChildTag = this->AddTag(strArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		for(size_t i = 0; i < N; i++)
		{
			pChildTag->Save<TYPE>(strItemName, pValues[i]);
		}
	}

/******************************************************************
	OpenMap
*******************************************************************/
	template<class TYPE1, class TYPE2>
	void OpenMap(const string& strArrayName, std::map<TYPE1, TYPE2>& mapObjects, string strItemName = string("Item"))
	{
		TYPE1 first;
		TYPE2 second;
		xmlTag_t* pItemTag;

		xmlTag_t* pChildTag = this->FindTag(strArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		string strFirst  = "first";
		string strSecond = "second";
		std::vector<xmlTag_t*> arrTags = pChildTag->FindMultipleTag(strItemName);
		for(size_t i = 0; i < arrTags.size(); i++)
		{
			pItemTag = arrTags[i];
			if(!pItemTag)
				daeDeclareAndThrowException(exXMLIOError);

			pItemTag->Open<TYPE1>(strFirst, first);
			pItemTag->Open<TYPE2>(strSecond, second);
			mapObjects.insert(std::pair<TYPE1, TYPE2>(first, second));
		}
	}

/******************************************************************
	SaveMap
*******************************************************************/
    template<typename TYPE1, typename TYPE2>
	void SaveMap(const string& strArrayName, const std::map<TYPE1, TYPE2>& mapObjects, string strItemName = string("Item"))
	{
        typedef typename std::map<TYPE1, TYPE2>::const_iterator ci;
        xmlTag_t* pItemTag;
        ci it;

		xmlTag_t* pChildTag = this->AddTag(strArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		string strFirst  = "first";
		string strSecond = "second";
		for(it = mapObjects.begin(); it != mapObjects.end(); it++)
		{
			pItemTag = pChildTag->AddTag(strItemName);

			pItemTag->Save<TYPE1>(strFirst,  (*it).first);
			pItemTag->Save<TYPE2>(strSecond, (*it).second);
		}
	}

/********************************************************************
	daeOpenEnum
*********************************************************************/
	template<class ENUM>
	void OpenEnum(const std::string& strEnumName, ENUM& eValue)
	{
		int iValue;
		xmlTag_t* pChildTag = this->FindTag(strEnumName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		pChildTag->GetValue(iValue);
		eValue = static_cast<ENUM>(iValue);
	}

/******************************************************************
	daeSaveEnum
*******************************************************************/
	template<class ENUM>
	void SaveEnum(const std::string& strEnumName, const ENUM& eValue)
	{
		this->AddTag(strEnumName, (int)eValue);
	}

/******************************************************************
	OpenObject
*******************************************************************/
	template<class TYPE, class OOD>
	TYPE* OpenObject(const std::string& strObjectName, daeOnOpenObjectDelegate_t<OOD>* ood = NULL)
	{
		xmlTag_t* pChildTag = this->FindTag(strObjectName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		return pChildTag->OpenObject<TYPE, OOD>(ood);
	}

	template<class TYPE, class OOD>
	TYPE* OpenObject(daeOnOpenObjectDelegate_t<OOD>* ood)
	{
		string strNULLValue;
		this->GetValue(strNULLValue);
		if(strNULLValue == string("NULL"))
			return NULL;

		TYPE* pObject = new TYPE;

		if(ood)
			ood->BeforeOpenObject(pObject);
		pObject->Open(this);
		if(ood)
			ood->AfterOpenObject(pObject);

		return pObject;
	}

	template<class TYPE, class OOD>
	void OpenExistingObject(const std::string& strObjectName, TYPE* pObject, daeOnOpenObjectDelegate_t<OOD>* ood = NULL)
	{
		if(!pObject)
			daeDeclareAndThrowException(exXMLIOError);

		xmlTag_t* pChildTag = this->FindTag(strObjectName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		return pChildTag->OpenExistingObject<TYPE, OOD>(ood, pObject);
	}

	template<class TYPE, class OOD>
	void OpenExistingObject(daeOnOpenObjectDelegate_t<OOD>* ood, TYPE* pObject)
	{
		string strNULLValue;
		this->GetValue(strNULLValue);
		if(strNULLValue == string("NULL"))
			return;

		if(ood)
			ood->BeforeOpenObject(pObject);
		pObject->Open(this);
		if(ood)
			ood->AfterOpenObject(pObject);
	}

/******************************************************************
	daeSaveObject
*******************************************************************/
	template<class TYPE>
	void SaveObject(const string& strObjectName, const TYPE* pObject)
	{
		xmlTag_t* pChildTag = this->AddTag(strObjectName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		if(pObject)
			pObject->Save(pChildTag);
		else
			pChildTag->SetValue(string("NULL"));
	}

// Runtime object
	template<class TYPE>
	void SaveRuntimeObject(const string& strObjectName, const TYPE* pObject)
	{
		xmlTag_t* pChildTag = this->AddTag(strObjectName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		if(pObject)
			pObject->SaveRuntime(pChildTag);
		else
			pChildTag->SetValue(string("NULL"));
	}
	
/******************************************************************
	OpenObjectArray
*******************************************************************/
	template<class TYPE, class OOAD>
	void OpenObjectArray(const string& strObjectArrayName, std::vector<TYPE*>& ptrarrObjects, daeOnOpenObjectArrayDelegate_t<OOAD>* ooad = NULL, string strObjectName = string("Object"))
	{
		TYPE* pObject;
		xmlTag_t* pObjectTag;

		xmlTag_t* pChildTag = this->FindTag(strObjectArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		std::vector<xmlTag_t*> arrTags = pChildTag->FindMultipleTag(strObjectName);

		for(size_t i = 0; i < arrTags.size(); i++)
		{
			pObjectTag = arrTags[i];
			if(!pObjectTag)
				daeDeclareAndThrowException(exXMLIOError);

			pObject = pObjectTag->OpenObject<TYPE, OOAD>(ooad);
			if(!pObject)
				daeDeclareAndThrowException(exXMLIOError);

			ptrarrObjects.push_back(pObject);
		}
		if(ooad)
			ooad->AfterAllObjectsOpened();
	}

/******************************************************************
	SaveObjectArray
*******************************************************************/
	template<class TYPE>
	void SaveObjectArray(const string& strObjectArrayName, const std::vector<TYPE*>& ptrarrObjects, string strObjectName = string("Object"))
	{
		xmlTag_t* pChildTag = this->AddTag(strObjectArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		for(size_t i = 0; i < ptrarrObjects.size(); i++)
		{
			pChildTag->SaveObject<TYPE>(strObjectName, ptrarrObjects[i]);
		}
	}

	template<class TYPE>
	void SaveObjectArray(const string& strObjectArrayName, const std::vector< boost::shared_ptr<TYPE> >& ptrarrObjects, string strObjectName = string("Object"))
	{
		xmlTag_t* pChildTag = this->AddTag(strObjectArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		for(size_t i = 0; i < ptrarrObjects.size(); i++)
		{
			pChildTag->SaveObject<TYPE>(strObjectName, ptrarrObjects[i].get());
		}
	}

	template<class TYPE>
	void SaveObjectArray(const string& strObjectArrayName, const std::vector<TYPE>& ptrarrObjects, string strObjectName = string("Object"))
	{
		xmlTag_t* pChildTag = this->AddTag(strObjectArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		for(size_t i = 0; i < ptrarrObjects.size(); i++)
		{
			pChildTag->SaveObject<TYPE>(strObjectName, &ptrarrObjects[i]);
		}
	}
	
// Runtime objects array
	template<class TYPE>
	void SaveRuntimeObjectArray(const string& strObjectArrayName, const std::vector<TYPE*>& ptrarrObjects, string strObjectName = string("Object"))
	{
		xmlTag_t* pChildTag = this->AddTag(strObjectArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		for(size_t i = 0; i < ptrarrObjects.size(); i++)
		{
			pChildTag->SaveRuntimeObject<TYPE>(strObjectName, ptrarrObjects[i]);
		}
	}
	
/*************************************************************************
	OpenObjectRef
**************************************************************************/
	template<class TYPE>
	TYPE* OpenObjectRef(const string& strObjectRefName, daeOnOpenRefDelegate_t<TYPE>* oord)
	{
		xmlTag_t* pChildTag = this->FindTag(strObjectRefName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);
		if(!oord)
			daeDeclareAndThrowException(exXMLIOError);

		return pChildTag->OpenObjectRef<TYPE>(oord);
	}

	template<class TYPE>
	TYPE* OpenObjectRef(daeOnOpenRefDelegate_t<TYPE>* oord)
	{
		unsigned long id = 0;
		string strName = "ID";
		xmlAttribute_t* pAttrID = this->FindAttribute(strName);
		if(!pAttrID)
		{
			string strNULLValue;
			this->GetValue(strNULLValue);
			if(strNULLValue == string("NULL"))
				return NULL;
			else
				daeDeclareAndThrowException(exXMLIOError);
		}
		
		pAttrID->GetValue(id);

		if(oord)
			return oord->FindObjectByID(id);

		return NULL;
	}

/*************************************************************************
	daeSaveObjectRef
    pParent is used if object reference is requested to an object which is not 
    a direct parent of the pObject
**************************************************************************/
	template<class TYPE>
	void SaveObjectRef(const string& strObjectRefName, const TYPE* pObject)
	{
		xmlTag_t* pChildTag;

		if(pObject)
		{
			string strName = "ID";
			string strRelName = pObject->GetNameRelativeToParentModel();
			
			pChildTag = this->AddTag(strObjectRefName, strRelName);
			pChildTag->AddAttribute(strName, pObject->GetID());
			
			pObject->SaveRelativeNameAsMathML(pChildTag, string("ObjectRefMathML"));
		}
		else
		{
			pChildTag = this->AddTag(strObjectRefName, string("NULL"));
		}
	}

	template<class TYPE, class PARENT_TYPE>
	void SaveObjectRef(const string& strObjectRefName, const TYPE* pObject, const PARENT_TYPE* pParent)
	{
		xmlTag_t* pChildTag;

		if(pObject)
		{
			string strName = "ID";
			string strRelName = daeGetRelativeName(pParent, pObject);
			
			pChildTag = this->AddTag(strObjectRefName, strRelName);
			pChildTag->AddAttribute(strName, pObject->GetID());
			
			pObject->SaveRelativeNameAsMathML(pChildTag, string("ObjectRefMathML"), pParent);
		}
		else
		{
			pChildTag = this->AddTag(strObjectRefName, string("NULL"));
		}
	}

/******************************************************************
	OpenRefArray
*******************************************************************/
	template<class TYPE>
	void OpenObjectRefArray(const string& strObjectRefArrayName, std::vector<TYPE*>& Objects, daeOnOpenRefDelegate_t<TYPE>* oord, string strRefName = string("ObjectRef"))
	{
		TYPE* pObject;
		xmlTag_t* pRefTag;

		xmlTag_t* pChildTag = this->FindTag(strObjectRefArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		std::vector<xmlTag_t*> arrTags = pChildTag->FindMultipleTag(strRefName);
		for(size_t i = 0; i < arrTags.size(); i++)
		{
			pRefTag = arrTags[i];
			if(!pRefTag)
				daeDeclareAndThrowException(exXMLIOError);

			pObject = OpenObjectRef<TYPE>(oord);
			if(!pObject)
				daeDeclareAndThrowException(exXMLIOError);

			Objects.push_back(pObject);
		}
	}

/******************************************************************
	SaveRefArray
*******************************************************************/
	template<class TYPE>
	void SaveObjectRefArray(const string& strObjectRefArrayName, const std::vector<TYPE*>& arrObjects, string strRefName = string("ObjectRef"))
	{
		xmlTag_t* pChildTag = this->AddTag(strObjectRefArrayName);
		if(!pChildTag)
			daeDeclareAndThrowException(exXMLIOError);

		for(size_t i = 0; i < arrObjects.size(); i++)
		{
			pChildTag->SaveObjectRef<TYPE>(strRefName, arrObjects[i]);
		}
	}
};

/********************************************************************
	daeSerializable_t
*********************************************************************/
class daeSerializable_t
{
public:
	virtual ~daeSerializable_t(void){}

public:
	virtual size_t	GetID() const				= 0;
	virtual string	GetVersion() const			= 0;
	virtual string	GetLibrary() const			= 0;
	virtual string	GetObjectClassName() const	= 0;

	virtual void	Open(xmlTag_t* pTag)		= 0;
	virtual void	Save(xmlTag_t* pTag) const	= 0;
};


}
}

#endif
