#ifndef UNITS_IO_H
#define UNITS_IO_H

#include "helpers.h"
#include "core.h"
#include "io.h"
#include "../Units/units.h"

namespace units
{
using namespace daetools::io;
using namespace daetools::core;

inline void Open(xmlTag_t* pTag, const std::string& strName, unit& u)
{
}

inline void Save(xmlTag_t* pTag, const std::string& strName, const unit& u)
{
	xmlTag_t* pChildTag = pTag->AddTag(strName);

	for(std::map<std::string, double>::const_iterator iter = u.units.begin(); iter != u.units.end(); iter++)
		pChildTag->Save(iter->first, iter->second);
}

inline void Save(xmlTag_t* pTag, const string& strName, const std::vector<unit>& arrObjects, string strItemName = string("Item"))
{
	xmlTag_t* pChildTag = pTag->AddTag(strName);

	for(size_t i = 0; i < arrObjects.size(); i++)
	{
		Save(pChildTag, strItemName, arrObjects[i]);
	}
}

inline std::string Export(daeeModelLanguage eLanguage, daeModelExportContext& c, const unit& u)
{
	std::string strResult;
	
	if(eLanguage == eCDAE)
	{
		for(std::map<std::string, double>::const_iterator iter = u.units.begin(); iter != u.units.end(); iter++)
		{
			if(iter != u.units.begin())
				strResult += "*";
			
			strResult += (boost::format("(%1% ^ (%2%))") % (*iter).first % (*iter).second).str();
		}
	}
	else if(eLanguage == ePYDAE)
	{
		for(std::map<std::string, double>::const_iterator iter = u.units.begin(); iter != u.units.end(); iter++)
		{
			if(iter != u.units.begin())
				strResult += "*";
			
			strResult += (boost::format("(%1% ** (%2%))") % (*iter).first % (*iter).second).str();
		}	
	}
	else
	{
		daeDeclareAndThrowException(daetools::exNotImplemented);
	}
	
	return strResult;
}

inline void SaveUnit(xmlTag_t* pTag, std::string strUnit, double dExponent)
{
	xmlTag_t *msup, *temp;
	
	if(dExponent == 1)
	{
		temp = pTag->AddTag(string("mi"), strUnit);	
		temp->AddAttribute(string("mathvariant"), string("italic"));
	}
	else if(dExponent == 0)
	{
	}
	else
	{
		msup = pTag->AddTag(string("msup"), string(""));
		temp = msup->AddTag(string("mi"), strUnit);	
		temp->AddAttribute(string("mathvariant"), string("italic"));
		msup->AddTag(string("mn"), dExponent);	
	}
	
}

inline void SaveAsPresentationMathML(xmlTag_t* pTag, const unit& u)
{
	std::map<std::string, double>::const_iterator iter;
	
	xmlTag_t* mrow = pTag->AddTag(string("mrow"), string(""));
	for(iter = u.units.begin(); iter != u.units.end(); iter++)
	{
		if(iter->second >= 0)
			SaveUnit(mrow, iter->first, iter->second);
	}
	for(iter = u.units.begin(); iter != u.units.end(); iter++)
	{
		if(iter->second < 0)
			SaveUnit(mrow, iter->first, iter->second);
	}
}


inline void Open(xmlTag_t* pTag, const std::string& strName, quantity& q)
{
}

inline void Save(xmlTag_t* pTag, const std::string& strName, const quantity& q)
{
	xmlTag_t* pChildTag = pTag->AddTag(strName);

	Save(pChildTag, std::string("units"), q.getUnits());
	pChildTag->Save(std::string("value"), q.getValue());
}

inline void Save(xmlTag_t* pTag, const string& strName, const std::vector<quantity>& arrObjects, string strItemName = string("Item"))
{
	xmlTag_t* pChildTag = pTag->AddTag(strName);

	for(size_t i = 0; i < arrObjects.size(); i++)
	{
		Save(pChildTag, strItemName, arrObjects[i]);
	}
}

inline std::string Export(daeeModelLanguage eLanguage, daeModelExportContext& c, const quantity& q)
{
	if(eLanguage == eCDAE)
	{
        if(q.getUnits() == unit())
    		return (boost::format("Constant(%1%)") % q.getValue()).str();
        else
            return (boost::format("Constant(%1% * (%2%))") % q.getValue() % Export(eLanguage, c, q.getUnits())).str();
	}
	else if(eLanguage == ePYDAE)
	{
        if(q.getUnits() == unit())
    		return (boost::format("Constant(%1%)") % q.getValue()).str();
		else
            return (boost::format("Constant(%1% * (%2%))") % q.getValue() % Export(eLanguage, c, q.getUnits())).str();
	}
	else
	{
		daeDeclareAndThrowException(daetools::exNotImplemented);
		return std::string("");
	}
}

inline void SaveAsPresentationMathML(xmlTag_t* pTag, const quantity& q)
{
	xmlTag_t* mrow;

	mrow = pTag->AddTag(string("mrow"), string(""));
	mrow->AddTag(string("mn"), q.getValue());	
	SaveAsPresentationMathML(mrow, q.getUnits());	
}

inline void SaveAsPresentationMathML(xmlTag_t* pTag, const std::vector<quantity>& arrObjects)
{
	xmlTag_t* mrow;

	mrow = pTag->AddTag(string("mrow"), string(""));
	mrow->AddTag(string("mo"), string("["));	

	for(size_t i = 0; i < arrObjects.size(); i++)
	{
		if(i != 0)
			mrow->AddTag(string("mo"), string(","));	
			
		SaveAsPresentationMathML(mrow, arrObjects[i]);
	}
	
	mrow->AddTag(string("mo"), string("]"));	
}

}

#endif
