#ifndef DAE_MODEL_EXPORT_H
#define DAE_MODEL_EXPORT_H

#include "core.h"
#include <boost/format.hpp>

namespace dae 
{
namespace core 
{
class daeModel;
/********************************************************************
	daeModelExportContext
*********************************************************************/
class daeModelExportContext
{
public:
	std::string CalculateIndent(size_t nPythonIndentLevel)
	{
		std::string strIndent;
		daeConfig& cfg = daeConfig::GetConfig();
        std::string strPythonIndent = cfg.GetString("daetools.core.pythonIndent", "    ");
		
		for(size_t i = 0; i < nPythonIndentLevel; i++)
			strIndent += strPythonIndent;
		
		return strIndent;		
	}
	
public:
	size_t			m_nPythonIndentLevel;
	const daeModel*	m_pModel;
	bool			m_bExportDefinition;
};

/********************************************************************
	daeExportable_t
*********************************************************************/
class daeExportable_t
{
public:
	virtual ~daeExportable_t(void){}

public:
	virtual void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const = 0;
};

/********************************************************************
	Export functions
*********************************************************************/
template<class TYPE>
void ExportObjectArray(const std::vector<TYPE>& ptrarrObjects, std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c)
{
	TYPE pObject;
	
	for(size_t i = 0; i < ptrarrObjects.size(); i++)
	{
		pObject = ptrarrObjects[i];
		pObject->Export(strContent, eLanguage, c);
	}
}

template<class TYPE>
void CreateDefinitionObjectArray(const std::vector<TYPE>& ptrarrObjects, std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c)
{
	TYPE pObject;
	
	for(size_t i = 0; i < ptrarrObjects.size(); i++)
	{
		if(i != 0)
			strContent += "\n";
		pObject = ptrarrObjects[i];
		pObject->CreateDefinition(strContent, eLanguage, c);
	}
}

}
}

#endif
