/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2015
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_IO_FUNCTIONS_H
#define DAE_IO_FUNCTIONS_H

#include "class_factory.h"
#include <typeinfo.h>

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
template<class TYPE, class REFERENCE_TYPE>
class daeOnOpenRefDelegate_t
{
public:
	virtual ~daeOnOpenRefDelegate_t(void){}

public:
	virtual TYPE* FindObjectByReference(REFERENCE_TYPE& ref) = 0;
};

/********************************************************************
	daeOpenPOD
*********************************************************************/
template<class TYPE>
void daeOpenPOD(const xmlTag* m_pCurrentTag, const string& strTagName, TYPE& Value)
{
	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	if(!m_pCurrentTag->GetChildTagValue(strTagName, Value))
	{
		daeDeclareException(exXMLIOError);
		e << string("Cannot find tag ") + strTagName + string(" in tag ") + m_pCurrentTag->GetName();
		throw e;
	}
}

/********************************************************************
	daeSavePOD
*********************************************************************/
template<class TYPE>
void daeSavePOD(xmlTag* m_pCurrentTag, const string& strTagName, const TYPE& Value)
{
	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	m_pCurrentTag->AddTag(strTagName, Value);
}

/********************************************************************
	daeOpenEnum
*********************************************************************/
template<class ENUM>
void daeOpenEnum(const xmlTag* m_pCurrentTag, const string& strTagName, ENUM& Value)
{
	int iValue;

	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	if(!m_pCurrentTag->GetChildTagValue(strTagName, iValue))
	{
		daeDeclareException(exXMLIOError);
		e << string("Cannot find tag ") + strTagName + string(" in tag ") + m_pCurrentTag->GetName();
		throw e;
	}
	Value = (ENUM)iValue;
}

/******************************************************************
	daeSaveEnum
*******************************************************************/
template<class ENUM>
void daeSaveEnum(xmlTag* m_pCurrentTag, const string& strTagName, const ENUM& Value)
{
	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 
	m_pCurrentTag->AddTag(strTagName, (int)Value);
}

/******************************************************************
	OpenPODArray
*******************************************************************/
template<class TYPE>
void OpenPODArray(const xmlTag* m_pCurrentTag, const string& strParentTagName, vector<TYPE>& arrValues)
{
	TYPE Value;
	xmlTag* pChildTag, *m_pCurrentTagParent;
	vector<xmlTag*> arrTags;

	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	pTagParent = m_pCurrentTag->FindTag(strParentTagName);
	if(!pTagParent)
	{
		daeDeclareException(exXMLIOError);
		e << string("Cannot find tag ") + strParentTagName + string(" in tag ") + m_pCurrentTag->GetName();
		throw e;
	}

	string strChildTagName = "Item";
	arrTags = pTagParent->FindMultipleTag(strChildTagName);
	for(size_t i = 0; i < arrTags.size(); i++)
	{
		pChildTag = arrTags[i];
		if(!pChildTag)
			daeDeclareAndThrowException(exInvalidPointer); 

		pChildTag->GetValue(Value);
		arrValues.push_back(Value);
	}
}

/******************************************************************
	SavePODArray
*******************************************************************/
template<class TYPE>
void SavePODArray(xmlTag* m_pCurrentTag, const string& strParentTagName, const vector<TYPE>& arrValues)
{
	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	xmlTag* pTagParent = m_pCurrentTag->AddTag(strParentTagName);
	string strChildTagName = "Item";

	for(size_t i = 0; i < arrValues.size(); i++)
		pTagParent->AddTag(strChildTagName, arrValues[i]);
}

/******************************************************************
	daeOpenObject Opens the object from the given tag
*******************************************************************/
template<class TYPE, class DELEGATE>
TYPE* daeOpenObject(const xmlTag* m_pCurrentTag, daeOnOpenObjectDelegate_t<DELEGATE>* ood, const daeClassFactoryManager_t* pCFManager)
{
	TYPE* pObject = NULL;
	string strName, strClass, strVersion;

	strName = "Class";
	io::daeOpenPOD<string>(m_pCurrentTag, strName, strClass);

	strName = "Version";
	io::daeOpenPOD<string>(m_pCurrentTag, strName, strVersion);

//------------------------------------------------------------------------------>
//------------------------------------------------------------------------------>
	if(pCFManager)
	{
		if(dynamic_cast<daeModel*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateModel(strClass, strVersion));
		else if(dynamic_cast<daeDomain*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateDomain(strClass, strVersion));
		else if(dynamic_cast<daeParameter*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateParameter(strClass, strVersion));
		else if(dynamic_cast<daeVariable*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateVariable(strClass, strVersion));
		else if(dynamic_cast<daeEquation*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateEquation(strClass, strVersion));
		else if(dynamic_cast<daeSTN*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateSTN(strClass, strVersion));
		else if(dynamic_cast<daePort*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreatePort(strClass, strVersion));
		else if(dynamic_cast<daeState*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateState(strClass, strVersion));
		else if(dynamic_cast<daeStateTransition*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateStateTransition(strClass, strVersion));
		else if(dynamic_cast<daePortConnection*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreatePortConnection(strClass, strVersion));

		else if(dynamic_cast<daeSimulation_t*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateSimulation(strClass, strVersion));
		else if(dynamic_cast<daeOptimization_t*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateOptimization(strClass, strVersion));

		else if(dynamic_cast<daeDataReceiver_t*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateDataReceiver(strClass, strVersion));
		else if(dynamic_cast<daeDataReporter_t*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateDataReporter(strClass, strVersion));

		else if(dynamic_cast<daeLASolver_t*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateLASolver(strClass, strVersion));
		else if(dynamic_cast<daeNLASolver_t*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateNLASolver(strClass, strVersion));
		else if(dynamic_cast<daeDAESolver_t*>(pObject))
			pObject = dynamic_cast<TYPE*>(pCFManager->CreateDAESolver(strClass, strVersion));

		else
		{
			daeDeclareException(exXMLIOError);
			e << string("Unsupported ") + typeid(TYPE).name();
			throw e;
		}
	}
	else
	{
		pObject = new TYPE;
	}
//------------------------------------------------------------------------------>
//------------------------------------------------------------------------------>

	if(!pObject)
	{
		daeDeclareException(exXMLIOError);
		e << string("Cannot instantiate object ") + typeid(TYPE).name();
		throw e;
	}

	if(ood)
		ood->BeforeOpenObject(pObject);
	pObject->Open(m_pCurrentTag, pCFManager);
	if(ood)
		ood->AfterOpenObject(pObject);

	return pObject;
}

/*************************************************************************
	daeOpenObject Looks for the tag in the given parent tag and opens it
**************************************************************************/
template<class TYPE, class DELEGATE>
TYPE* daeOpenObject(const xmlTag* pParentTag, const string& strTagName, daeOnOpenObjectDelegate_t<DELEGATE>* ood, const daeClassFactoryManager_t* pCFManager)
{
	xmlTag* pChildTag;

	if(!pParentTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	pChildTag = pParentTag->FindTag(strTagName);
	if(!pChildTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	return daeOpenObject<TYPE, DELEGATE>(pChildTag, ood, pCFManager);
}

/*************************************************************************
	daeSaveObject
**************************************************************************/
template<class TYPE>
xmlTag* daeSaveObject(xmlTag* m_pCurrentTag, const string& strTagName, const TYPE* pObject)
{
	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!pObject)
		daeDeclareAndThrowException(exInvalidPointer); 

	xmlTag* pChildTag = m_pCurrentTag->AddTag(strTagName);
	pObject->Save(pChildTag);
	return pChildTag;
}

/******************************************************************
	daeOpenObjectArray
*******************************************************************/
template<class TYPE, class DELEGATE>
void daeOpenObjectArray(const xmlTag* m_pCurrentTag, const string& strArrayTagName, vector<TYPE*>& ptrarrObjects, daeOnOpenObjectArrayDelegate_t<DELEGATE>* ood, const daeClassFactoryManager_t* pCFManager)
{
	TYPE* pObject;
	xmlTag* pTagParent;
	xmlTag* pChildTag;
	vector<xmlTag*> arrTags;

	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!ood)
		daeDeclareAndThrowException(exInvalidPointer); 

	pTagParent = m_pCurrentTag->FindTag(strArrayTagName);
	if(!pTagParent)
		daeDeclareAndThrowException(exInvalidPointer); 

	string strItemTagName = "Item";
	arrTags = pTagParent->FindMultipleTag(strItemTagName);
	for(size_t i = 0; i < arrTags.size(); i++)
	{
		pChildTag = arrTags[i];
		if(!pChildTag)
			daeDeclareAndThrowException(exInvalidPointer); 

		pObject = daeOpenObject<TYPE, DELEGATE>(pChildTag, ood, pCFManager);
		if(!pObject)
			daeDeclareAndThrowException(exInvalidPointer); 

		ptrarrObjects.push_back(pObject);
	}
	if(ood)
		ood->AfterAllObjectsOpened();	
}

/******************************************************************
	daeSaveObjectArray
*******************************************************************/
template<class TYPE>
void daeSaveObjectArray(xmlTag* m_pCurrentTag, const vector<TYPE*>& ptrarrObjects, const string& strParentTagName)
{
	TYPE* pObject;
	xmlTag *pTagParent;

	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	string strChildTagName = "Item";
	pTagParent = m_pCurrentTag->AddTag(strParentTagName);
	for(size_t i = 0; i < ptrarrObjects.size(); i++)
	{
		pObject = ptrarrObjects[i];
		if(!pObject)
			daeDeclareAndThrowException(exInvalidPointer); 

		daeSaveObject<TYPE>(pTagParent, strChildTagName, pObject);
	}
}

/*************************************************************************
	daeSaveObjectRef
**************************************************************************/
template<class REFERENCE_TYPE>
xmlTag* daeSaveObjectRef(xmlTag* m_pCurrentTag, const string& strTagName, const REFERENCE_TYPE& ref)
{
	string strName;

	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	xmlTag* pChildTag = m_pCurrentTag->AddTag(strTagName);

	strName = "ID";
	daeSavePOD<REFERENCE_TYPE>(pChildTag, strName, ref);

	//strName = "CanonicalName";
	//daeSavePOD<string>(pChildTag, strName, pObject->GetCanonicalName());

	return pChildTag;
}

/******************************************************************
	daeSaveObjectRefArray
*******************************************************************/
template<class REFERENCE_TYPE>
void daeSaveObjectRefArray(xmlTag* m_pCurrentTag, const vector<REFERENCE_TYPE>& arrReferences, const string& strParentTagName)
{
	REFERENCE_TYPE ref;
	xmlTag* pTagParent;

	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 

	string strChildTagName = "Item";

	pTagParent = m_pCurrentTag->AddTag(strParentTagName);
	for(size_t i = 0; i < arrReferences.size(); i++)
	{
		ref = arrReferences[i];
		daeSaveObjectRef<REFERENCE_TYPE>(pTagParent, strChildTagName, ref);
	}
}

/*************************************************************************
	daeOpenObjectRef
**************************************************************************/
template<class TYPE, class REFERENCE_TYPE>
TYPE* daeOpenObjectRef(const xmlTag* m_pCurrentTag, daeOnOpenRefDelegate_t<TYPE, REFERENCE_TYPE>* oord)
{
	REFERENCE_TYPE ref;
	string strName;

	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!oord)
		daeDeclareAndThrowException(exInvalidPointer); 

	strName = "ID";
	daeOpenPOD<REFERENCE_TYPE>(m_pCurrentTag, strName, ref);
	return oord->FindObjectByReference(ref);
}

/******************************************************************
	daeOpenObjectRefArray
*******************************************************************/
template<class TYPE, class REFERENCE_TYPE>
void daeOpenObjectRefArray(const xmlTag* m_pCurrentTag, const string& strArrayTagName, vector<TYPE*>& ptrarrObjects, daeOnOpenRefDelegate_t<TYPE, REFERENCE_TYPE>* oord)
{
	TYPE* pObject;
	xmlTag* pTagParent;
	xmlTag* pChildTag;
	vector<xmlTag*> arrTags;

	if(!m_pCurrentTag)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!oord)
		daeDeclareAndThrowException(exInvalidPointer); 

	pTagParent = m_pCurrentTag->FindTag(strArrayTagName);
	if(!pTagParent)
		daeDeclareAndThrowException(exInvalidPointer); 

	string strItemTagName = "Item";
	arrTags = pTagParent->FindMultipleTag(strItemTagName);
	for(size_t i = 0; i < arrTags.size(); i++)
	{
		pChildTag = arrTags[i];
		if(!pChildTag)
			daeDeclareAndThrowException(exInvalidPointer); 

		pObject = daeOpenObjectRef<TYPE, REFERENCE_TYPE>(pChildTag, oord);
		if(!pObject)
			daeDeclareAndThrowException(exInvalidPointer); 

		ptrarrObjects.push_back(pObject);
	}
}


}
}

#endif
