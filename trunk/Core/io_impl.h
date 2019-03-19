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
#ifndef DAE_IO_IMPL_H
#define DAE_IO_IMPL_H

#include "io.h"
#include "definitions.h"
#include "core.h"

namespace dae
{

#define daeDeclareDynamicClass(Class) typedef Class this_type; \
                                      public: std::string GetObjectClassName(void) const {return std::string(#Class);}

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
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeEquationType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeSpecialUnaryFunctions& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeIntegralFunctions& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeRangeType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeIndexRangeType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeActionType& eValue);
void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, core::daeeVariableValueConstraint& eValue);

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
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeEquationType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeSpecialUnaryFunctions eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeIntegralFunctions eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeRangeType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeIndexRangeType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeActionType eValue);
void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const core::daeeVariableValueConstraint eValue);

/*********************************************************************************************
    daeEnumStringMap
**********************************************************************************************/
template<typename ENUM>
class daeEnumStringMap
{
public:
  typedef map<string, ENUM> map_string_enum;
  typedef map<ENUM, string> map_enum_string;
  typedef typename map<string, ENUM>::const_iterator iter_string_enum;
  typedef typename map<ENUM, string>::const_iterator iter_enum_string;
  typedef std::pair<string, ENUM> pair_string_enum;
  typedef std::pair<ENUM, string> pair_enum_string;

public:
    void Add(ENUM key, string value)
    {
        pair_enum_string pes(key, value);
        pair_string_enum pse(value, key);

        m_mapStringEnum.insert(pse);
        m_mapEnumString.insert(pes);
    }

    ENUM GetEnum(const string& value) const
    {
        iter_string_enum iter = m_mapStringEnum.find(value);
        if(iter != m_mapStringEnum.end())
            return iter->second;
        else
            return (ENUM)-1;
    }

    string GetString(ENUM value) const
    {
        iter_enum_string iter = m_mapEnumString.find(value);
        if(iter != m_mapEnumString.end())
            return iter->second;
        else
            return "Unknown";
    }

protected:
    map_enum_string m_mapEnumString;
    map_string_enum m_mapStringEnum;
};

#define daeesmapAdd1(ESMAP, E1)								ESMAP.Add(E1, string(#E1));

#define daeesmapAdd2(ESMAP, E1, E2)							ESMAP.Add(E1, string(#E1)); \
                                                            ESMAP.Add(E2, string(#E2));

#define daeesmapAdd3(ESMAP, E1, E2, E3)						ESMAP.Add(E1, string(#E1)); \
                                                            ESMAP.Add(E2, string(#E2)); \
                                                            ESMAP.Add(E3, string(#E3));

#define daeesmapAdd4(ESMAP, E1, E2, E3, E4)					ESMAP.Add(E1, #E1); \
                                                            ESMAP.Add(E2, #E2); \
                                                            ESMAP.Add(E3, #E3); \
                                                            ESMAP.Add(E4, #E4);

#define daeesmapAdd5(ESMAP, E1, E2, E3, E4, E5)				ESMAP.Add(E1, #E1); \
                                                            ESMAP.Add(E2, #E2); \
                                                            ESMAP.Add(E3, #E3); \
                                                            ESMAP.Add(E4, #E4); \
                                                            ESMAP.Add(E5, #E5);

#define daeesmapAdd6(ESMAP, E1, E2, E3, E4, E5, E6)			ESMAP.Add(E1, #E1); \
                                                            ESMAP.Add(E2, #E2); \
                                                            ESMAP.Add(E3, #E3); \
                                                            ESMAP.Add(E4, #E4); \
                                                            ESMAP.Add(E5, #E5); \
                                                            ESMAP.Add(E6, #E6);

#define daeesmapAdd7(ESMAP, E1, E2, E3, E4, E5, E6, E7)		ESMAP.Add(E1, #E1); \
                                                            ESMAP.Add(E2, #E2); \
                                                            ESMAP.Add(E3, #E3); \
                                                            ESMAP.Add(E4, #E4); \
                                                            ESMAP.Add(E5, #E5); \
                                                            ESMAP.Add(E6, #E6); \
                                                            ESMAP.Add(E7, #E7);

#define daeesmapAdd8(ESMAP, E1, E2, E3, E4, E5, E6, E7, E8)	ESMAP.Add(E1, #E1); \
                                                            ESMAP.Add(E2, #E2); \
                                                            ESMAP.Add(E3, #E3); \
                                                            ESMAP.Add(E4, #E4); \
                                                            ESMAP.Add(E5, #E5); \
                                                            ESMAP.Add(E6, #E6); \
                                                            ESMAP.Add(E7, #E7); \
                                                            ESMAP.Add(E8, #E8);


class daeEnumTypesCollection
{
public:
    daeEnumTypesCollection(void)
    {
        using namespace dae::core;
        daeesmapAdd3(esmap_daeDomainType,               eArray, eStructuredGrid, eUnstructuredGrid);
        daeesmapAdd3(esmap_daeeParameterType,           eReal, eInteger, eBool);
        daeesmapAdd3(esmap_daeePortType,                eUnknownPort, eInletPort, eOutletPort);
        daeesmapAdd3(esmap_daeeOptimizationVariableType,eIntegerVariable, eBinaryVariable, eContinuousVariable);
        daeesmapAdd2(esmap_daeeConstraintType,          eInequalityConstraint, eEqualityConstraint);
        daeesmapAdd2(esmap_daeeModelLanguage,           eCDAE, ePYDAE);
        daeesmapAdd8(esmap_daeeDomainBounds,            eOpenOpen, eOpenClosed, eClosedOpen,
                                                        eClosedClosed, eLowerBound, eUpperBound,
                                                        eFunctor, eCustomBound);
    }

    daeEnumStringMap<core::daeeDomainType>                  esmap_daeDomainType;
    daeEnumStringMap<core::daeeParameterType>               esmap_daeeParameterType;
    daeEnumStringMap<core::daeePortType>                    esmap_daeePortType;
    daeEnumStringMap<core::daeeOptimizationVariableType>	esmap_daeeOptimizationVariableType;
    daeEnumStringMap<core::daeeConstraintType>              esmap_daeeConstraintType;
    daeEnumStringMap<core::daeeModelLanguage>               esmap_daeeModelLanguage;
    daeEnumStringMap<core::daeeDomainBounds>                esmap_daeeDomainBounds;
};

static boost::shared_ptr<daeEnumTypesCollection> g_EnumTypesCollection = boost::shared_ptr<daeEnumTypesCollection>(new daeEnumTypesCollection());

}
}

#endif
