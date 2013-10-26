#include "stdafx.h"
#include "io_impl.h"
using namespace dae::core;

namespace dae
{
namespace io
{
/********************************************************************
	daeSerializable
*********************************************************************/
daeSerializable::daeSerializable(void)
{
	m_nID = (size_t)this;
}

daeSerializable::~daeSerializable(void)
{
}

size_t daeSerializable::GetID() const
{
	return m_nID;
}

void daeSerializable::SetID(unsigned long nID)
{
	m_nID = nID;
}

string daeSerializable::GetLibrary() const
{
	return m_strLibraryName;
}

string daeSerializable::GetVersion() const
{
	return daeVersion();
}

void daeSerializable::Open(io::xmlTag_t* pTag)
{
	string strName;
	xmlAttribute_t* pAttr;

	strName = "ID";
	pAttr = pTag->FindAttribute(strName);
	if(!pAttr)
		daeDeclareAndThrowException(exXMLIOError);
	pAttr->GetValue(m_nID);
	
	strName = "Library";
	pAttr = pTag->FindAttribute(strName);
	if(!pAttr)
		daeDeclareAndThrowException(exXMLIOError);
	pAttr->GetValue(m_strLibraryName);
}

void daeSerializable::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "ID";
	pTag->AddAttribute(strName, m_nID);

	strName = "Class";
	pTag->AddAttribute(strName, GetObjectClassName());

	strName = "Library";
	pTag->AddAttribute(strName, m_strLibraryName);

	strName = "Version";
	pTag->AddAttribute(strName, daeVersion(true));
}

/******************************************************************
	enum helper functions
*******************************************************************/
const string strUnknown = "Unknown";

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeDomainType eValue)
{
	if(eValue == eArray)
		pTag->AddTag(strEnumName, string("eArray"));
    else if(eValue == eStructuredGrid)
        pTag->AddTag(strEnumName, string("eStructuredGrid"));
    else if(eValue == eUnstructuredGrid)
        pTag->AddTag(strEnumName, string("eUnstructuredGrid"));
    else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeParameterType eValue)
{
	if(eValue == eReal)
		pTag->AddTag(strEnumName, string("eReal"));
	else if(eValue == eInteger)
		pTag->AddTag(strEnumName, string("eInteger"));
	else if(eValue == eBool)
		pTag->AddTag(strEnumName, string("eBool"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeDomainBounds eValue)
{
	if(eValue == eOpenOpen)
		pTag->AddTag(strEnumName, string("eOpenOpen"));
	else if(eValue == eOpenClosed)
		pTag->AddTag(strEnumName, string("eOpenClosed"));
	else if(eValue == eClosedOpen)
		pTag->AddTag(strEnumName, string("eClosedOpen"));
	else if(eValue == eClosedClosed)
		pTag->AddTag(strEnumName, string("eClosedClosed"));
	else if(eValue == eLowerBound)
		pTag->AddTag(strEnumName, string("eLowerBound"));
	else if(eValue == eUpperBound)
		pTag->AddTag(strEnumName, string("eUpperBound"));
	else if(eValue == eFunctor)
		pTag->AddTag(strEnumName, string("eFunctor"));
	else if(eValue == eCustomBound)
		pTag->AddTag(strEnumName, string("eCustomBound"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeDiscretizationMethod eValue)
{
	if(eValue == eCFDM)
		pTag->AddTag(strEnumName, string("eCFDM"));
	else if(eValue == eFFDM)
		pTag->AddTag(strEnumName, string("eFFDM"));
	else if(eValue == eBFDM)
		pTag->AddTag(strEnumName, string("eBFDM"));
	else if(eValue == eCustomDM)
		pTag->AddTag(strEnumName, string("eCustomDM"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeePortType eValue)
{
	if(eValue == eInletPort)
		pTag->AddTag(strEnumName, string("eInletPort"));
	else if(eValue == eOutletPort)
		pTag->AddTag(strEnumName, string("eOutletPort"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeFunctionType eValue)
{
	if(eValue == eUnary)
		pTag->AddTag(strEnumName, string("eUnary"));
	else if(eValue == eBinary)
		pTag->AddTag(strEnumName, string("eBinary"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeUnaryFunctions eValue)
{
	if(eValue == eSign)
		pTag->AddTag(strEnumName, string("eSign"));
	else if(eValue == eSqrt)
		pTag->AddTag(strEnumName, string("eSqrt"));
	else if(eValue == eExp)
		pTag->AddTag(strEnumName, string("eExp"));
	else if(eValue == eLog)
		pTag->AddTag(strEnumName, string("eLog"));
	else if(eValue == eLn)
		pTag->AddTag(strEnumName, string("eLn"));
	else if(eValue == eAbs)
		pTag->AddTag(strEnumName, string("eAbs"));
	else if(eValue == eSin)
		pTag->AddTag(strEnumName, string("eSin"));
	else if(eValue == eCos)
		pTag->AddTag(strEnumName, string("eCos"));
	else if(eValue == eTan)
		pTag->AddTag(strEnumName, string("eTan"));
	else if(eValue == eArcSin)
		pTag->AddTag(strEnumName, string("eArcSin"));
	else if(eValue == eArcCos)
		pTag->AddTag(strEnumName, string("eArcCos"));
	else if(eValue == eArcTan)
		pTag->AddTag(strEnumName, string("eArcTan"));
	else if(eValue == eCeil)
		pTag->AddTag(strEnumName, string("eCeil"));
	else if(eValue == eFloor)
		pTag->AddTag(strEnumName, string("eFloor"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeBinaryFunctions eValue)
{
	if(eValue == ePlus)
		pTag->AddTag(strEnumName, string("ePlus"));
	else if(eValue == eMinus)
		pTag->AddTag(strEnumName, string("eMinus"));
	else if(eValue == eMulti)
		pTag->AddTag(strEnumName, string("eMulti"));
	else if(eValue == eDivide)
		pTag->AddTag(strEnumName, string("eDivide"));
	else if(eValue == ePower)
		pTag->AddTag(strEnumName, string("ePower"));
	else if(eValue == eMin)
		pTag->AddTag(strEnumName, string("eMin"));
	else if(eValue == eMax)
		pTag->AddTag(strEnumName, string("eMax"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeLogicalUnaryOperator eValue)
{
	if(eValue == eNot)
		pTag->AddTag(strEnumName, string("eNot"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeLogicalBinaryOperator eValue)
{
	if(eValue == eAnd)
		pTag->AddTag(strEnumName, string("eAnd"));
	else if(eValue == eOr)
		pTag->AddTag(strEnumName, string("eOr"));
	else
		pTag->AddTag(strEnumName, string("Unknown"));
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeConditionType eValue)
{
	if(eValue == eNotEQ)
		pTag->AddTag(strEnumName, string("eNotEQ"));
	else if(eValue == eEQ)
		pTag->AddTag(strEnumName, string("eEQ"));
	else if(eValue == eGT)
		pTag->AddTag(strEnumName, string("eGT"));
	else if(eValue == eGTEQ)
		pTag->AddTag(strEnumName, string("eGTEQ"));
	else if(eValue == eLT)
		pTag->AddTag(strEnumName, string("eLT"));
	else if(eValue == eLTEQ)
		pTag->AddTag(strEnumName, string("eLTEQ"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeSTNType eValue)
{
	if(eValue == eSTN)
		pTag->AddTag(strEnumName, string("eSTN"));
	else if(eValue == eIF)
		pTag->AddTag(strEnumName, string("eIF"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeDomainIndexType eValue)
{
	if(eValue == eConstantIndex)
		pTag->AddTag(strEnumName, string("eConstantIndex"));
	else if(eValue == eDomainIterator)
		pTag->AddTag(strEnumName, string("eDomainIterator"));
	else if(eValue == eIncrementedDomainIterator)
		pTag->AddTag(strEnumName, string("eIncrementedDomainIterator"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeEquationType eValue)
{
	if(eValue == eExplicitODE)
		pTag->AddTag(strEnumName, string("eExplicitODE"));
	else if(eValue == eImplicitODE)
		pTag->AddTag(strEnumName, string("eImplicitODE"));
    else if(eValue == eAlgebraic)
		pTag->AddTag(strEnumName, string("eAlgebraic"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeSpecialUnaryFunctions eValue)
{
	if(eValue == eSum)
		pTag->AddTag(strEnumName, string("eSum"));
	else if(eValue == eProduct)
		pTag->AddTag(strEnumName, string("eProduct"));
	else if(eValue == eMinInArray)
		pTag->AddTag(strEnumName, string("eMinInArray"));
	else if(eValue == eMaxInArray)
		pTag->AddTag(strEnumName, string("eMaxInArray"));
	else if(eValue == eAverage)
		pTag->AddTag(strEnumName, string("eAverage"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeIntegralFunctions eValue)
{
	if(eValue == eSingleIntegral)
		pTag->AddTag(strEnumName, string("eSingleIntegral"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeRangeType eValue)
{
	if(eValue == eRangeDomainIndex)
		pTag->AddTag(strEnumName, string("eRangeDomainIndex"));
	else if(eValue == eRange)
		pTag->AddTag(strEnumName, string("eRange"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeIndexRangeType eValue)
{
	if(eValue == eAllPointsInDomain)
		pTag->AddTag(strEnumName, string("eAllPointsInDomain"));
	else if(eValue == eRangeOfIndexes)
		pTag->AddTag(strEnumName, string("eRangeOfIndexes"));
	else if(eValue == eCustomRange)
		pTag->AddTag(strEnumName, string("eCustomRange"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}

void SaveEnum(xmlTag_t* pTag, const std::string& strEnumName, const daeeActionType eValue)
{
	if(eValue == eChangeState)
		pTag->AddTag(strEnumName, string("eChangeState"));
	else if(eValue == eSendEvent)
		pTag->AddTag(strEnumName, string("eSendEvent"));
	else if(eValue == eReAssignOrReInitializeVariable)
		pTag->AddTag(strEnumName, string("eReAssignOrReInitializeVariable"));
	else if(eValue == eUserDefinedAction)
		pTag->AddTag(strEnumName, string("eUserDefinedAction"));
	else
		pTag->AddTag(strEnumName, strUnknown);
}



void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeDomainType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eArray")
		eValue = eArray;
    else if(strValue == "eStructuredGrid")
        eValue = eStructuredGrid;
    else if(strValue == "eUnstructuredGrid")
        eValue = eUnstructuredGrid;
    else
		eValue = eDTUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeParameterType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eReal")
		eValue = eReal;
	else if(strValue == "eInteger")
		eValue = eInteger;
	else if(strValue == "eBool")
		eValue = eBool;
	else
		eValue = ePTUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeDomainBounds& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eOpenOpen")
		eValue = eOpenOpen;
	else if(strValue == "eOpenClosed")
		eValue = eOpenClosed;
	else if(strValue == "eClosedOpen")
		eValue = eClosedOpen;
	else if(strValue == "eClosedClosed")
		eValue = eClosedClosed;
	else if(strValue == "eFunctor")
		eValue = eFunctor;
	else if(strValue == "eLowerBound")
		eValue = eLowerBound;
	else if(strValue == "eUpperBound")
		eValue = eUpperBound;
	else if(strValue == "eCustomBound")
		eValue = eCustomBound;
	else
		eValue = eDBUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeDiscretizationMethod& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eCFDM")
		eValue = eCFDM;
	else if(strValue == "eFFDM")
		eValue = eFFDM;
	else if(strValue == "eBFDM")
		eValue = eBFDM;
	else if(strValue == "eCustomDM")
		eValue = eCustomDM;
	else
		eValue = eDMUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeePortType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eInletPort")
		eValue = eInletPort;
	else if(strValue == "eOutletPort")
		eValue = eOutletPort;
	else
		eValue = eUnknownPort;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeFunctionType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eUnary")
		eValue = eUnary;
	else if(strValue == "eBinary")
		eValue = eBinary;
	else
		eValue = eFTUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeUnaryFunctions& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eSign")
		eValue = eSign;
	else if(strValue == "eSqrt")
		eValue = eSqrt;
	else if(strValue == "eExp")
		eValue = eExp;
	else if(strValue == "eLog")
		eValue = eLog;
	else if(strValue == "eLn")
		eValue = eLn;
	else if(strValue == "eAbs")
		eValue = eAbs;
	else if(strValue == "eSin")
		eValue = eSin;
	else if(strValue == "eCos")
		eValue = eCos;
	else if(strValue == "eTan")
		eValue = eTan;
	else if(strValue == "eArcSin")
		eValue = eArcSin;
	else if(strValue == "eArcCos")
		eValue = eArcCos;
	else if(strValue == "eArcTan")
		eValue = eArcTan;
	else if(strValue == "eCeil")
		eValue = eCeil;
	else if(strValue == "eFloor")
		eValue = eFloor;
	else
		eValue = eUFUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeBinaryFunctions& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "ePlus")
		eValue = ePlus;
	else if(strValue == "eMinus")
		eValue = eMinus;
	else if(strValue == "eMulti")
		eValue = eMulti;
	else if(strValue == "eDivide")
		eValue = eDivide;
	else if(strValue == "ePower")
		eValue = ePower;
	else if(strValue == "eMin")
		eValue = eMin;
	else if(strValue == "eMax")
		eValue = eMax;
	else
		eValue = eBFUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeLogicalUnaryOperator& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eNot")
		eValue = eNot;
	else
		eValue = eUOUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeLogicalBinaryOperator& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eAnd")
		eValue = eAnd;
	else if(strValue == "eOr")
		eValue = eOr;
	else
		eValue = eBOUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeConditionType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eNotEQ")
		eValue = eNotEQ;
	else if(strValue == "eEQ")
		eValue = eEQ;
	else if(strValue == "eGT")
		eValue = eGT;
	else if(strValue == "eGTEQ")
		eValue = eGTEQ;
	else if(strValue == "eLT")
		eValue = eLT;
	else if(strValue == "eLTEQ")
		eValue = eLTEQ;
	else
		eValue = eCTUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeSTNType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eSTN")
		eValue = eSTN;
	else if(strValue == "eIF")
		eValue = eIF;
	else
		eValue = eSTNTUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeDomainIndexType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eConstantIndex")
		eValue = eConstantIndex;
	else if(strValue == "eDomainIterator")
		eValue = eDomainIterator;
	else if(strValue == "eIncrementedDomainIterator")
		eValue = eIncrementedDomainIterator;
	else
		eValue = eDITUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeEquationType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eExplicitODE")
		eValue = eExplicitODE;
	else if(strValue == "eImplicitODE")
		eValue = eImplicitODE;
    else if(strValue == "eAlgebraic")
		eValue = eAlgebraic;
	else
		eValue = eETUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeSpecialUnaryFunctions& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eSum")
		eValue = eSum;
	else if(strValue == "eProduct")
		eValue = eProduct;
	else if(strValue == "eMinInArray")
		eValue = eMinInArray;
	else if(strValue == "eMaxInArray")
		eValue = eMaxInArray;
	else if(strValue == "eAverage")
		eValue = eAverage;
	else
		eValue = eSUFUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeIntegralFunctions& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eSingleIntegral")
		eValue = eSingleIntegral;
	else
		eValue = eIFUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeRangeType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eRangeDomainIndex")
		eValue = eRangeDomainIndex;
	else if(strValue == "eRange")
		eValue = eRange;
	else
		eValue = eRaTUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeIndexRangeType& eValue)
{
	string strValue;
	pTag->Open(strEnumName, strValue);

	if(strValue == "eAllPointsInDomain")
		eValue = eAllPointsInDomain;
	else if(strValue == "eRangeOfIndexes")
		eValue = eRangeOfIndexes;
	else if(strValue == "eCustomRange")
		eValue = eCustomRange;
	else
		eValue = eIRTUnknown;
}

void OpenEnum(xmlTag_t* pTag, const std::string& strEnumName, daeeActionType& eValue)
{
	string strValue;	
	pTag->Open(strEnumName, strValue);

	if(strValue == "eChangeState")
		eValue = eChangeState;
	else if(strValue == "eSendEvent")
		eValue = eSendEvent;
	else if(strValue == "eReAssignOrReInitializeVariable")
		eValue = eReAssignOrReInitializeVariable;
	else if(strValue == "eUserDefinedAction")
		eValue = eUserDefinedAction;
	else
		eValue = eUnknownAction;
}



}
}
