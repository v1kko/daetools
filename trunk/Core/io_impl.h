#ifndef DAE_IO_IMPL_H
#define DAE_IO_IMPL_H

#include "io.h"
#include "definitions.h"
#include "core.h"

namespace dae
{

#define daeDeclareDynamicClass(Class) public: std::string GetObjectClassName(void) const {return std::string(#Class);}

namespace io
{
/********************************************************************
	daeSerializable
*********************************************************************/
class daeSerializable : public daeSerializable_t
{
public:
	daeSerializable(void);
	virtual ~daeSerializable(void);

public:
	size_t		GetID() const;
	void		SetID(unsigned long nID);
	string		GetVersion() const;
	string		GetLibrary() const;
	void		Open(io::xmlTag_t* pTag);
	void		Save(io::xmlTag_t* pTag) const;

protected:
	size_t m_nID;
	std::string m_strLibraryName;
};

/******************************************************************
	SaveObjectRuntime
*******************************************************************/
template<class TYPE>
void SaveObjectRuntime(xmlTag_t* pTag, const std::string& strObjectName, const TYPE* pObject)
{
	xmlTag_t* pChildTag = pTag->AddTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);
	pObject->SaveRuntime(pChildTag);
}


/******************************************************************
	OpenObjectRuntime
*******************************************************************/
template<class TYPE, class OOD>
TYPE* OpenObjectRuntime(xmlTag_t* pTag, const std::string& strObjectName, daeOnOpenObjectDelegate_t<OOD>* ood = NULL)
{
	xmlTag_t* pChildTag = pTag->FindTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	TYPE* pObject = new TYPE;
	if(ood)
		ood->BeforeOpenObject(pObject);
	pObject->OpenRuntime(pChildTag);
	if(ood)
		ood->AfterOpenObject(pObject);

	return pObject;
}

/******************************************************************
	SaveObjectArrayRuntime
*******************************************************************/
template<class TYPE>
void SaveObjectArrayRuntime(xmlTag_t* pTag, const std::string& strObjectArrayName, const std::vector<TYPE*>& ptrarrObjects, std::string strObjectName = std::string("Object"))
{
	xmlTag_t* pObjectTag;
	xmlTag_t* pChildTag = pTag->AddTag(strObjectArrayName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	for(size_t i = 0; i < ptrarrObjects.size(); i++)
	{
		pObjectTag = pChildTag->AddTag(strObjectName);
		ptrarrObjects[i]->SaveRuntime(pObjectTag);
	}
}

/******************************************************************
	OpenObjectArrayRuntime
*******************************************************************/
template<class TYPE, class OOAD>
void OpenObjectArrayRuntime(xmlTag_t* pTag, const std::string& strObjectArrayName, std::vector<TYPE*>& ptrarrObjects, daeOnOpenObjectArrayDelegate_t<OOAD>* ooad = NULL, std::string strObjectName = std::string("Object"))
{
	TYPE* pObject;
	xmlTag_t* pObjectTag;

	xmlTag_t* pChildTag = pTag->FindTag(strObjectArrayName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	std::vector<xmlTag_t*> arrTags = pChildTag->FindMultipleTag(strObjectName);
	for(size_t i = 0; i < arrTags.size(); i++)
	{
		pObjectTag = arrTags[i];
		if(!pObjectTag)
			daeDeclareAndThrowException(exXMLIOError);

		pObject = new TYPE;
		if(ooad)
			ooad->BeforeOpenObject(pObject);
		pObject->OpenRuntime(pObjectTag);
		if(ooad)
			ooad->AfterOpenObject(pObject);

		ptrarrObjects.push_back(pObject);
	}
	if(ooad)
		ooad->AfterAllObjectsOpened();
}

/******************************************************************
	enum helper functions
*******************************************************************/
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeDomainType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeParameterType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeDomainBounds& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeDiscretizationMethod& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeePortType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeFunctionType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeUnaryFunctions& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeBinaryFunctions& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeLogicalUnaryOperator& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeLogicalBinaryOperator& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeConditionType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeSTNType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeDomainIndexType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeEquationDefinitionMode& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeEquationEvaluationMode& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeSpecialUnaryFunctions& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeIntegralFunctions& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeRangeType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeIndexRangeType& eValue);

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeDomainType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeParameterType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeDomainBounds eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeDiscretizationMethod eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeePortType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeFunctionType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeUnaryFunctions eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeBinaryFunctions eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeLogicalUnaryOperator eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeLogicalBinaryOperator eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeConditionType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeSTNType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeDomainIndexType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeEquationDefinitionMode eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeEquationEvaluationMode eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeSpecialUnaryFunctions eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeIntegralFunctions eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeRangeType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeIndexRangeType eValue);


}
}

#endif
