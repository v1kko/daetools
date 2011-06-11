#include "stdafx.h"
#include "coreimpl.h"
#include <typeinfo> 

namespace dae 
{
namespace core 
{
daeModel::daeModel()
{
	_currentSTN							= NULL;
	m_pExecutionContextForGatherInfo	= NULL;
	m_nVariablesStartingIndex			= 0;
	m_nTotalNumberOfVariables			= 0;
	m_pCondition						= NULL;

// When used programmatically they dont own pointers !!!!!
	m_ptrarrDomains.SetOwnershipOnPointers(false);
	m_ptrarrPorts.SetOwnershipOnPointers(false);
	m_ptrarrVariables.SetOwnershipOnPointers(false);
	m_ptrarrParameters.SetOwnershipOnPointers(false);
	m_ptrarrModels.SetOwnershipOnPointers(false);
	m_ptrarrPortArrays.SetOwnershipOnPointers(false);
	m_ptrarrModelArrays.SetOwnershipOnPointers(false);
}

//daeModel::daeModel(string strName)
//{
//	_currentSTN							= NULL;
//	m_pExecutionContextForGatherInfo	= NULL;
//	m_nVariablesStartingIndex			= 0;
//	m_nTotalNumberOfVariables			= 0;
//	m_pCondition						= NULL;
//
//// When used programmatically they dont own pointers !!!!!
//	m_ptrarrDomains.SetOwnershipOnPointers(false);
//	m_ptrarrPorts.SetOwnershipOnPointers(false);
//	m_ptrarrVariables.SetOwnershipOnPointers(false);
//	m_ptrarrParameters.SetOwnershipOnPointers(false);
//	m_ptrarrModels.SetOwnershipOnPointers(false);
//	m_ptrarrPortArrays.SetOwnershipOnPointers(false);
//	m_ptrarrModelArrays.SetOwnershipOnPointers(false);
//	
//	SetName(strName);
//}

daeModel::daeModel(string strName, daeModel* pModel, string strDescription)
{
	_currentSTN							= NULL;
	m_pExecutionContextForGatherInfo	= NULL;
	m_nVariablesStartingIndex			= 0;
	m_nTotalNumberOfVariables			= 0;
	m_pCondition						= NULL;

// When used programmatically they dont own pointers !!!!!
	m_ptrarrDomains.SetOwnershipOnPointers(false);
	m_ptrarrPorts.SetOwnershipOnPointers(false);
	m_ptrarrVariables.SetOwnershipOnPointers(false);
	m_ptrarrParameters.SetOwnershipOnPointers(false);
	m_ptrarrModels.SetOwnershipOnPointers(false);
	m_ptrarrPortArrays.SetOwnershipOnPointers(false);
	m_ptrarrModelArrays.SetOwnershipOnPointers(false);
	
	SetName(strName);
	SetDescription(strDescription);
	if(pModel)
		pModel->AddModel(this);
}

daeModel::~daeModel()
{
	if(m_pCondition)
	{
		delete m_pCondition;
		m_pCondition = NULL;
	}
}

void daeModel::Open(io::xmlTag_t* pTag)
{
	string strName;

	m_ptrarrModels.EmptyAndFreeMemory();
	m_ptrarrEquations.EmptyAndFreeMemory();
	m_ptrarrSTNs.EmptyAndFreeMemory();
	m_ptrarrPortConnections.EmptyAndFreeMemory();
	m_ptrarrDomains.EmptyAndFreeMemory();
	m_ptrarrParameters.EmptyAndFreeMemory();
	m_ptrarrVariables.EmptyAndFreeMemory();
	m_ptrarrPorts.EmptyAndFreeMemory();
	m_ptrarrEquationExecutionInfos.EmptyAndFreeMemory();
	m_ptrarrPortArrays.EmptyAndFreeMemory();
	m_ptrarrModelArrays.EmptyAndFreeMemory();

	m_ptrarrModels.SetOwnershipOnPointers(true);
	m_ptrarrEquations.SetOwnershipOnPointers(true);
	m_ptrarrSTNs.SetOwnershipOnPointers(true);
	m_ptrarrPortConnections.SetOwnershipOnPointers(true);
	m_ptrarrDomains.SetOwnershipOnPointers(true);
	m_ptrarrParameters.SetOwnershipOnPointers(true);
	m_ptrarrVariables.SetOwnershipOnPointers(true);
	m_ptrarrPorts.SetOwnershipOnPointers(true);
	m_ptrarrEquationExecutionInfos.SetOwnershipOnPointers(true);
	m_ptrarrPortArrays.SetOwnershipOnPointers(true);
	m_ptrarrModelArrays.SetOwnershipOnPointers(true);
	
	m_pDataProxy.reset();
	m_nVariablesStartingIndex	= 0;
	m_ptrarrEquationExecutionInfos.EmptyAndFreeMemory();

	daeObject::Open(pTag);

	if(m_strShortName.empty())
		daeDeclareAndThrowException(exInvalidPointer); 

	daeSetModelAndCanonicalNameDelegate<daeObject> del(this, this);

// VARIABLE TYPES
	//strName = "VariableTypes";
	//pTag->OpenObjectArray(strName, m_ptrarrVariableTypes, &del);

// DOMAINS
	strName = "Domains";
	pTag->OpenObjectArray(strName, m_ptrarrDomains, &del);

// PARAMETERS
	strName = "Parameters";
	pTag->OpenObjectArray(strName, m_ptrarrParameters, &del);

// VARIABLES
	strName = "Variables";
	pTag->OpenObjectArray(strName, m_ptrarrVariables, &del);

// PORTS
	strName = "Ports";
	pTag->OpenObjectArray(strName, m_ptrarrPorts, &del);

// EQUATIONS
	strName = "Equations";
	pTag->OpenObjectArray(strName, m_ptrarrEquations, &del);

// STNs
	strName = "STNs";
	pTag->OpenObjectArray(strName, m_ptrarrSTNs, &del);

// PORTCONNECTIONs
	strName = "PortConnections";
	pTag->OpenObjectArray(strName, m_ptrarrPortConnections, &del);

// CHILD MODELS
	strName = "Units";
	pTag->OpenObjectArray(strName, m_ptrarrModels, &del);

// MODEL ARRAYS
	//strName = "UnitArrays";
	//pTag->OpenObjectArray(strName, m_ptrarrModelArrays, &del);

// PORT ARRAYS
	//strName = "PortArrays";
	//pTag->OpenObjectArray(strName, m_ptrarrPortArrays, &del);
}

void daeModel::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

// DOMAINS
	strName = "Domains";
	pTag->SaveObjectArray(strName, m_ptrarrDomains);

// PARAMETERS
	strName = "Parameters";
	pTag->SaveObjectArray(strName, m_ptrarrParameters);

// VARIABLES
	strName = "Variables";
	pTag->SaveObjectArray(strName, m_ptrarrVariables);

// PORTS
	strName = "Ports";
	pTag->SaveObjectArray(strName, m_ptrarrPorts);

// EQUATIONS
	strName = "Equations";
	pTag->SaveObjectArray(strName, m_ptrarrEquations);

// STNs
	strName = "STNs";
	pTag->SaveObjectArray(strName, m_ptrarrSTNs);

// PORTCONNECTIONs
	strName = "PortConnections";
	pTag->SaveObjectArray(strName, m_ptrarrPortConnections);

// CHILD MODELS
	strName = "Units";
	pTag->SaveObjectArray(strName, m_ptrarrModels);

// MODELARRAYS
	strName = "UnitArrays";
	pTag->SaveObjectArray(strName, m_ptrarrModelArrays);

// PORTARRAYS
	strName = "PortArrays";
	pTag->SaveObjectArray(strName, m_ptrarrPortArrays);
}

void daeModel::OpenRuntime(io::xmlTag_t* pTag)
{
/*
	string strName;

	daeObject::OpenRuntime(pTag);

	m_ptrarrModels.EmptyAndFreeMemory();
	m_ptrarrEquations.EmptyAndFreeMemory();
	m_ptrarrSTNs.EmptyAndFreeMemory();
	m_ptrarrPortConnections.EmptyAndFreeMemory();
	m_ptrarrDomains.EmptyAndFreeMemory();
	m_ptrarrParameters.EmptyAndFreeMemory();
	m_ptrarrVariables.EmptyAndFreeMemory();
	m_ptrarrPorts.EmptyAndFreeMemory();
	m_ptrarrEquationExecutionInfos.EmptyAndFreeMemory();
	m_ptrarrPortArrays.EmptyAndFreeMemory();
	m_ptrarrModelArrays.EmptyAndFreeMemory();

	m_ptrarrModels.SetOwnershipOnPointers(true);
	m_ptrarrEquations.SetOwnershipOnPointers(true);
	m_ptrarrSTNs.SetOwnershipOnPointers(true);
	m_ptrarrPortConnections.SetOwnershipOnPointers(true);
	m_ptrarrDomains.SetOwnershipOnPointers(true);
	m_ptrarrParameters.SetOwnershipOnPointers(true);
	m_ptrarrVariables.SetOwnershipOnPointers(true);
	m_ptrarrPorts.SetOwnershipOnPointers(true);
	m_ptrarrEquationExecutionInfos.SetOwnershipOnPointers(true);
	m_ptrarrPortArrays.SetOwnershipOnPointers(true);
	m_ptrarrModelArrays.SetOwnershipOnPointers(true);
	
	m_pDataProxy.reset();
	m_nVariablesStartingIndex	= 0;
	m_ptrarrEquationExecutionInfos.EmptyAndFreeMemory();

	if(m_strShortName.empty())
		daeDeclareAndThrowException(exInvalidPointer); 

	daeSetModelAndCanonicalNameDelegate<daeObject> del(this, this);

// VARIABLE TYPES
	//strName = "VariableTypes";
	//pTag->OpenObjectArray(strName, m_ptrarrVariableTypes, &del);

// DOMAINS
	strName = "Domains";
	OpenObjectArrayRuntime(pTag, strName, m_ptrarrDomains, &del);

// PARAMETERS
	strName = "Parameters";
	OpenObjectArrayRuntime(pTag, strName, m_ptrarrParameters, &del);

// VARIABLES
	strName = "Variables";
	OpenObjectArrayRuntime(pTag, strName, m_ptrarrVariables, &del);

// PORTS
	strName = "Ports";
	OpenObjectArrayRuntime(pTag, strName, m_ptrarrPorts, &del);

// EQUATIONS
	strName = "Equations";
	OpenObjectArrayRuntime(pTag, strName, m_ptrarrEquations, &del);

// STNs
	strName = "STNs";
	OpenObjectArrayRuntime(pTag, strName, m_ptrarrSTNs, &del);

// PORTCONNECTIONs
	strName = "PortConnections";
	OpenObjectArrayRuntime(pTag, strName, m_ptrarrPortConnections, &del);

// CHILD MODELS
	strName = "Models";
	OpenObjectArrayRuntime(pTag, strName, m_ptrarrModels, &del);

// MODEL ARRAYS
	//strName = "ModelArrays";
	//OpenObjectArrayRuntime(pTag, strName, m_ptrarrModelArrays, &del);

// PORT ARRAYS
	//strName = "PortArrays";
	//OpenObjectArrayRuntime(pTag, strName, m_ptrarrPortArrays, &del);

// DATA PROXY
	daeDataProxy_t* pDataProxy = pTag->OpenObject<daeDataProxy_t, daeDataProxy_t>(strName);
	m_pDataProxy.reset(pDataProxy);

// TOTAL NUMBER OF VARIABLES
	strName = "TotalNumberOfVariables";
	pTag->Open(strName, m_nTotalNumberOfVariables);

// EQUATION EXECUTION INFOS
	strName = "EquationExecutionInfos";
	pTag->OpenObjectArray<daeEquationExecutionInfo, daeEquationExecutionInfo>(strName, m_ptrarrEquationExecutionInfos);
*/
}

void daeModel::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::SaveRuntime(pTag);

// DOMAINS
	strName = "Domains";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrDomains);

// PARAMETERS
	strName = "Parameters";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrParameters);

// VARIABLES
	strName = "Variables";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrVariables);

// PORTS
	strName = "Ports";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrPorts);

// EQUATIONS
	strName = "Equations";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrEquations);

// STNs
	strName = "STNs";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrSTNs);

// PORTCONNECTIONs
	strName = "PortConnections";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrPortConnections);

// CHILD MODELS
	strName = "Units";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrModels);

// MODELARRAYS
	strName = "UnitArrays";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrModelArrays);

// PORTARRAYS
	strName = "PortArrays";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrPortArrays);

/*
// DATA PROXY
	pTag->SaveObject(strName, m_pDataProxy.get());

// TOTAL NUMBER OF VARIABLES
	strName = "TotalNumberOfVariables";
	pTag->Save(strName, m_nTotalNumberOfVariables);

// EQUATION EXECUTION INFOS
	strName = "EquationExecutionInfos";
	pTag->SaveObjectArray(strName, m_ptrarrEquationExecutionInfos);
*/
}

// Objects can be models and ports
string daeModel::ExportObjects(std::vector<daeObject*>& ptrarrObjects, daeeModelLanguage eLanguage) const
{
	size_t i;
	daeModelExportContext c;
	boost::format fmtFile;
	daeModel* pModel;
	daePort* pPort;
	std::vector<daeModel*> ptrarrModels;
	std::vector<daePort*>  ptrarrPorts;
	std::vector<const daeVariableType*> ptrarrVariableTypes;
	string strFile, strVariableTypes, strPorts, strModels;

	for(i = 0; i < ptrarrObjects.size(); i++)
	{
		pModel = dynamic_cast<daeModel*>(ptrarrObjects[i]);
		pPort  = dynamic_cast<daePort*>(ptrarrObjects[i]);
		
		if(pModel)
			ptrarrModels.push_back(pModel); 
		else if(pPort)
			ptrarrPorts.push_back(pPort); 
		else
			daeDeclareAndThrowException(exRuntimeCheck); 
	}

// Detect all variable types
	for(i = 0; i < ptrarrModels.size(); i++)
		ptrarrModels[i]->DetectVariableTypesForExport(ptrarrVariableTypes);
	for(i = 0; i < ptrarrPorts.size(); i++)
		ptrarrPorts[i]->DetectVariableTypesForExport(ptrarrVariableTypes);
	
	c.m_nPythonIndentLevel = 0;
	c.m_bExportDefinition  = true;

	//c.m_strClassName = strClassName;
	//c.m_bExportDefinition = false;

	if(eLanguage == ePYDAE)
	{
	/* Arguments:
	   1. Variable types
	   2. Ports
	   3. Models
	*/
		strFile  = 
		"\"\"\"********************************************************************************\n"
		"                 DAE Tools: pyDAE module, www.daetools.com\n"
		"                 Copyright (C) Dragan Nikolic, 2010\n"
		"***********************************************************************************\n"
		"DAE Tools is free software; you can redistribute it and/or modify it under the\n"
		"terms of the GNU General Public License version 3 as published by the Free Software\n"
		"Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT\n"
		"ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A\n"
		"PARTICULAR PURPOSE. See the GNU General Public License for more details.\n"
		"You should have received a copy of the GNU General Public License along with the\n"
		"DAE Tools software; if not, see <http://www.gnu.org/licenses/>.\n"
		"********************************************************************************\"\"\"\n"
		"\n"
		"import sys\n"		
		"from daetools.pyDAE import *\n"
		"from time import localtime, strftime\n"
		"from daeVariableTypes import *\n\n"
		"%1%\n"
		"%2%\n"
		"%3%\n";
		
		ExportObjectArray(ptrarrVariableTypes, strVariableTypes, eLanguage, c);
		CreateDefinitionObjectArray(ptrarrPorts, strPorts, eLanguage, c);
		CreateDefinitionObjectArray(ptrarrModels, strModels, eLanguage, c);
	}
	else if(eLanguage == eCDAE)
	{
		strFile  += 
		"/********************************************************************************\n"
		"                 DAE Tools: cDAE module, www.daetools.com\n"
		"                 Copyright (C) Dragan Nikolic, 2010\n"
		"*********************************************************************************\n"
		"DAE Tools is free software; you can redistribute it and/or modify it under the\n"
		"terms of the GNU General Public License version 3 as published by the Free Software\n"
		"Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT\n"
		"ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A\n"
		"PARTICULAR PURPOSE. See the GNU General Public License for more details.\n"
		"You should have received a copy of the GNU General Public License along with the\n"
		"DAE Tools software; if not, see <http://www.gnu.org/licenses/>.\n"
		"*********************************************************************************/\n"
		"\n"
		"#include \"variable_types.h\";\n\n"
		"%1%\n"
		"%2%\n"
		"%3%\n";
		
		ExportObjectArray(ptrarrVariableTypes, strVariableTypes, eLanguage, c);
		CreateDefinitionObjectArray(ptrarrPorts, strPorts, eLanguage, c);
		CreateDefinitionObjectArray(ptrarrModels, strModels, eLanguage, c);
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented); 
	}
	
	fmtFile.parse(strFile);
	fmtFile % strVariableTypes % strPorts % strModels;
	
	return fmtFile.str();
}

void daeModel::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport;
	boost::format fmtFile;

	if(c.m_bExportDefinition)
	{
		if(eLanguage == ePYDAE)
		{
		}
		else if(eLanguage == eCDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "%1% %2%;\n";
			fmtFile.parse(strExport);
			fmtFile % GetObjectClassName() % GetStrippedName();
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}		
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.%1% = %2%(\"%3%\", self, \"%4%\")\n";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% GetObjectClassName()
					% m_strShortName 
					% m_strDescription;
		}
		else if(eLanguage == eCDAE)
		{
			strExport = ",\n" + c.CalculateIndent(c.m_nPythonIndentLevel) + "%1%(\"%2%\", this, \"%3%\")";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% m_strShortName 
					% m_strDescription;
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += fmtFile.str();
}

void daeModel::CreateDefinition(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	boost::format fmtFile;
	string strComment, strFile, strCXXDeclaration, strConstructor, strDeclareEquations;
	
	c.m_pModel = this;
	
	if(eLanguage == ePYDAE)
	{
		strComment = "#"; 
		
	/* Arguments:
	   1. Class name
	   2. Variable types
	   3. Port classes
	   4. Constructor contents (domains, params, variables, ports, units ...)
	   5. Equations and STNs/IFs
	*/
		strFile  = 
		"class %1%(daeModel):\n"
		"    def __init__(self, Name, Parent = None, Description = \"\"):\n"
		"        daeModel.__init__(self, Name, Parent, Description)\n\n"
		"%2%"
		"\n"
		"    def DeclareEquations(self):\n"
		"%3%";
		
		c.m_bExportDefinition = false;
		c.m_nPythonIndentLevel = 2;
		
		//strConstructor += c.CalculateIndent(c.m_nPythonIndentLevel) + strComment + " Domains \n";
		ExportObjectArray(m_ptrarrDomains,         strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrParameters,      strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrVariables,       strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrPorts,           strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrModels,          strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrModelArrays,     strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrPortArrays,      strConstructor, eLanguage, c);
		
		ExportObjectArray(m_ptrarrEquations,       strDeclareEquations, eLanguage, c);
		ExportObjectArray(m_ptrarrSTNs,            strDeclareEquations, eLanguage, c);
		ExportObjectArray(m_ptrarrPortConnections, strDeclareEquations, eLanguage, c);
		
		fmtFile.parse(strFile);
		fmtFile % GetObjectClassName() % strConstructor % (strDeclareEquations.empty() ? (c.CalculateIndent(c.m_nPythonIndentLevel) + "pass\n") : strDeclareEquations);
	}
	else if(eLanguage == eCDAE)
	{
		strComment   = "//"; 
		
		strFile  = 
		"class %1% : public daeModel\n"
		"{\n"
		"daeDeclareDynamicClass(%1%)\n"
		"public:\n"
		"%2%\n"
		"    %1%(string strName, daeModel* pParent = NULL, string strDescription = \"\") \n"
		"      : daeModel(strName, pParent, strDescription)"
		"%3%\n"
		"    {\n"
		"    }\n"
		"\n"
		"    void DeclareEquations(void)\n"
		"    {\n"
		"        daeEquation* eq;\n\n"
		"%4%"
		"    }\n"
		"};\n";
		
		c.m_bExportDefinition = true;
		c.m_nPythonIndentLevel = 1;
		
		ExportObjectArray(m_ptrarrDomains,     strCXXDeclaration, eLanguage, c);
		ExportObjectArray(m_ptrarrParameters,  strCXXDeclaration, eLanguage, c);
		ExportObjectArray(m_ptrarrVariables,   strCXXDeclaration, eLanguage, c);
		ExportObjectArray(m_ptrarrPorts,       strCXXDeclaration, eLanguage, c);
		ExportObjectArray(m_ptrarrModels,      strCXXDeclaration, eLanguage, c);
		ExportObjectArray(m_ptrarrModelArrays, strCXXDeclaration, eLanguage, c);
		ExportObjectArray(m_ptrarrPortArrays,  strCXXDeclaration, eLanguage, c);


		c.m_bExportDefinition = false;
		c.m_nPythonIndentLevel = 2;

		ExportObjectArray(m_ptrarrDomains,     strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrParameters,  strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrVariables,   strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrPorts,       strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrModels,      strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrModelArrays, strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrPortArrays,  strConstructor, eLanguage, c);

		c.m_bExportDefinition = false;
		c.m_nPythonIndentLevel = 2;
		
		ExportObjectArray(m_ptrarrEquations,       strDeclareEquations, eLanguage, c);
		ExportObjectArray(m_ptrarrSTNs,            strDeclareEquations, eLanguage, c);
		ExportObjectArray(m_ptrarrPortConnections, strDeclareEquations, eLanguage, c);
		
		fmtFile.parse(strFile);
		fmtFile % GetObjectClassName() % strCXXDeclaration % strConstructor % strDeclareEquations;
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented); 
	}
	
	strContent += fmtFile.str();
}

void daeModel::DetectVariableTypesForExport(std::vector<const daeVariableType*>& ptrarrVariableTypes) const
{
	size_t i, j;
	bool bFound;

	for(i = 0; i < m_ptrarrVariables.size(); i++)
	{
		bFound = false;
		for(j = 0; j < ptrarrVariableTypes.size(); j++)
		{
			if(ptrarrVariableTypes[j]->GetName() == m_ptrarrVariables[i]->m_VariableType.GetName())
			{
				bFound = true;
				break;
			}
		}
		if(!bFound)
			ptrarrVariableTypes.push_back(&m_ptrarrVariables[i]->m_VariableType);
	}

	for(i = 0; i < m_ptrarrPorts.size(); i++)
		m_ptrarrPorts[i]->DetectVariableTypesForExport(ptrarrVariableTypes);

	for(i = 0; i < m_ptrarrModels.size(); i++)
		m_ptrarrModels[i]->DetectVariableTypesForExport(ptrarrVariableTypes);

	for(i = 0; i < m_ptrarrPortArrays.size(); i++)
		m_ptrarrPortArrays[i]->DetectVariableTypesForExport(ptrarrVariableTypes);

	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
		m_ptrarrModelArrays[i]->DetectVariableTypesForExport(ptrarrVariableTypes);
}

//#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
//#include <***.h>
//#else
//#include <dlfcn.h>
//#endif

boost::shared_ptr<daeExternalObject_t> daeModel::LoadExternalObject(const string& strPath)
{
	boost::shared_ptr<daeExternalObject_t> extobj;

//	void* lib_handle;
//	pfnGetExternalObject pfn;
//	
//	lib_handle = dlopen(strPath.c_str(), RTLD_LAZY);
//	if(!lib_handle) 
//		daeDeclareAndThrowException(exInvalidCall);
//	
//	pfn = dlsym(lib_handle, "GetExternalObject");
//	if ((error = dlerror()) != NULL)  
//	{
//		fprintf(stderr, "%s\n", error);
//		exit(1);
//	}
//	
//	(*fn)(&x);
//	printf("Valx=%d\n",x);
//	
//	dlclose(lib_handle);
	
	return extobj;
}

void daeModel::AddDomain(daeDomain* pDomain)
{
	std::string strName = pDomain->GetName();
	if(strName.empty())
	{
		daeDeclareException(exInvalidCall);
		e << "Domain name cannot be empty";
		throw e;
	}
	if(CheckName(m_ptrarrDomains, strName))
	{
		daeDeclareException(exInvalidCall); 
		e << "Domain [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
		throw e;
	}

    SetModelAndCanonicalName(pDomain);
    m_ptrarrDomains.push_back(pDomain);
}

void daeModel::AddVariable(daeVariable* pVariable)
{
	std::string strName = pVariable->GetName();
	if(strName.empty())
	{
		daeDeclareException(exInvalidCall);
		e << "Variable name cannot be empty";
		throw e;
	}
	if(CheckName(m_ptrarrVariables, strName))
	{
		daeDeclareException(exInvalidCall); 
		e << "Variable [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
		throw e;
	}

    SetModelAndCanonicalName(pVariable);
    m_ptrarrVariables.push_back(pVariable);
}

void daeModel::AddParameter(daeParameter* pParameter)
{
	std::string strName = pParameter->GetName();
	if(strName.empty())
	{
		daeDeclareException(exInvalidCall);
		e << "Parameter name cannot be empty";
		throw e;
	}
	if(CheckName(m_ptrarrParameters, strName))
	{
		daeDeclareException(exInvalidCall); 
		e << "Parameter [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
		throw e;
	}
	
    SetModelAndCanonicalName(pParameter);
    m_ptrarrParameters.push_back(pParameter);
}

void daeModel::AddModel(daeModel* pModel)
{
	std::string strName = pModel->GetName();
	if(strName.empty())
	{
		daeDeclareException(exInvalidCall);
		e << "Model name cannot be empty";
		throw e;
	}
	if(CheckName(m_ptrarrModels, strName))
	{
		daeDeclareException(exInvalidCall); 
		e << "Child Model [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
		throw e;
	}

    SetModelAndCanonicalName(pModel);
    m_ptrarrModels.push_back(pModel);
}

void daeModel::AddPort(daePort* pPort)
{
	std::string strName = pPort->GetName();
	if(strName.empty())
	{
		daeDeclareException(exInvalidCall);
		e << "Port name cannot be empty";
		throw e;
	}
	if(CheckName(m_ptrarrPorts, strName))
	{
		daeDeclareException(exInvalidCall); 
		e << "Port [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
		throw e;
	}

    SetModelAndCanonicalName(pPort);
    m_ptrarrPorts.push_back(pPort);
}

void daeModel::AddPortConnection(daePortConnection* pPortConnection)
{
    SetModelAndCanonicalName(pPortConnection);
    m_ptrarrPortConnections.push_back(pPortConnection);
}

void daeModel::AddPortArray(daePortArray* pPortArray)
{
	std::string strName = pPortArray->GetName();
	if(strName.empty())
	{
		daeDeclareException(exInvalidCall); 
		e << "PortArray name cannot be empty";
		throw e;
	}
	if(CheckName(m_ptrarrPortArrays, strName))
	{
		daeDeclareException(exInvalidCall); 
		e << "PortArray [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
		throw e;
	}

    SetModelAndCanonicalName(pPortArray);
    m_ptrarrPortArrays.push_back(pPortArray);
}

void daeModel::AddModelArray(daeModelArray* pModelArray)
{
	std::string strName = pModelArray->GetName();
	if(strName.empty())
	{
		daeDeclareException(exInvalidCall); 
		e << "ModelArray name cannot be empty";
		throw e;
	}
	if(CheckName(m_ptrarrModelArrays, strName))
	{
		daeDeclareException(exInvalidCall); 
		e << "ModelArray [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
		throw e;
	}

    SetModelAndCanonicalName(pModelArray);
    m_ptrarrModelArrays.push_back(pModelArray);
}

void daeModel::AddEquation(daeEquation* pEquation)
{
    SetModelAndCanonicalName(pEquation);
    m_ptrarrEquations.push_back(pEquation);
}

void daeModel::AddDomain(daeDomain& rDomain, const string& strName, string strDescription)
{
	rDomain.SetName(strName);
	rDomain.SetDescription(strDescription);
	AddDomain(&rDomain);
}

void daeModel::AddVariable(daeVariable& rVariable, const string& strName, const daeVariableType& rVariableType, string strDescription)
{
	rVariable.SetName(strName);
	rVariable.SetDescription(strDescription);
	rVariable.SetVariableType(rVariableType);
	AddVariable(&rVariable);
}

void daeModel::AddParameter(daeParameter& rParameter, const string& strName, daeeParameterType eParameterType, string strDescription)
{
	rParameter.SetName(strName);
	rParameter.SetDescription(strDescription);
	rParameter.SetParameterType(eParameterType);
	AddParameter(&rParameter);
}

daeSTN* daeModel::AddSTN(const string& strName)
{
	if(strName.empty())
	{
		daeDeclareException(exInvalidCall);
		e << "STN name cannot be empty";
		throw e;
	}
	if(CheckName(m_ptrarrSTNs, strName))
	{
		daeDeclareException(exInvalidCall); 
		e << "STN [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
		throw e;
	}

	daeSTN* pSTN = new daeSTN;
	daeState* pParentState = GetStateFromStack();

	pSTN->SetName(strName);
	SetModelAndCanonicalName(pSTN);

	if(!pParentState)
	{
		pSTN->m_pParentState = NULL;
		m_ptrarrSTNs.push_back(pSTN);
	}
	else
	{
		pSTN->m_pParentState = pParentState;
		pParentState->AddNestedSTN(pSTN);
	}

	return pSTN;
}

daeIF* daeModel::AddIF(const string& strCondition)
{
	daeIF* pIF = new daeIF;
	daeState* pParentState = GetStateFromStack();

	string strName;

	if(!pParentState)
	{
		strName = "IF_" + toString<size_t>(m_ptrarrSTNs.size());
		pIF->m_pParentState = NULL;
		m_ptrarrSTNs.push_back(pIF);
	}
	else
	{
		strName = "IF_" + toString<size_t>(pParentState->GetNumberOfSTNs());
		pIF->m_pParentState = pParentState;
		pParentState->AddNestedSTN(pIF);
	}
	pIF->SetName(strName);
	SetModelAndCanonicalName(pIF);

	return pIF;
}

void daeModel::IF(const daeCondition& rCondition, real_t dEventTolerance)
{
	daeIF* _pIF = AddIF(string(""));
	string strStateName = "State" + toString<size_t>(_pIF->m_ptrarrStates.size());
	daeState* _pState = _pIF->AddState(strStateName);
	daeStateTransition* _pST = new daeStateTransition;
	_pST->CreateIF(string("Condition0"), _pState, rCondition, dEventTolerance);
}

void daeModel::ELSE_IF(const daeCondition& rCondition, real_t dEventTolerance)
{
// Current state must exist!!!
// Otherwise throw an exception
	daeState* pCurrentState = GetStateFromStack();
	if(!pCurrentState)
	{
		daeDeclareException(exInvalidCall); 
		e << "ELSE_IF can only be called after call to IF() or ELSE_IF()";
		throw e;
	}

// Parrent of the current state must be an IF block!!
// Otherwise throw an exception
	daeIF* _pIF = dynamic_cast<daeIF*>(pCurrentState->GetSTN());
	if(!_pIF)
	{
		daeDeclareException(exInvalidCall); 
		e << "Invalid IF block";
		throw e;
	}

// Add new state
	string strStateName = "State" + toString<size_t>(_pIF->m_ptrarrStates.size());
	daeState* _pState = _pIF->AddState(strStateName);
	daeStateTransition* _pST = new daeStateTransition;
	_pST->CreateIF(string("Condition0"), _pState, rCondition, dEventTolerance);
}

void daeModel::ELSE(void)
{
// Current state must exist!!!
// Otherwise throw an exception
	daeState* pCurrentState = GetStateFromStack();
	if(!pCurrentState)
	{
		daeDeclareException(exInvalidCall); 
		e << "ELSE can only be called after call to IF() or ELSE_IF()";
		throw e;
	}

// Parrent of the current state must be an IF block!!
// Otherwise throw an exception
	daeIF* _pIF = dynamic_cast<daeIF*>(pCurrentState->GetSTN());
	if(!_pIF)
	{
		daeDeclareException(exInvalidCall); 
		e << "Invalid IF block";
		throw e;
	}

// Create ELSE state
	_pIF->CreateElse();
}

void daeModel::END_IF(void)
{
// Current state must exist!!!
// Otherwise throw an exception
	daeState* pCurrentState = GetStateFromStack();
	if(!pCurrentState)
	{
		daeDeclareException(exInvalidCall); 
		e << "ELSE can only be called after call to IF() or ELSE_IF()";
		throw e;
	}

// Parrent of the current state must be an IF block!!
// Otherwise throw an exception
	daeIF* _pIF = dynamic_cast<daeIF*>(pCurrentState->GetSTN());
	if(!_pIF)
	{
		daeDeclareException(exInvalidCall); 
		e << "Invalid IF block";
		throw e;
	}

// Initialize the IF block
	_pIF->Initialize();
}

daeSTN* daeModel::STN(const string& strSTN)
{
	_currentSTN = AddSTN(strSTN);
	return _currentSTN;
}

daeState* daeModel::STATE(const string& strState)
{
// Current STN must exist!!!
// Otherwise throw an exception
	if(!_currentSTN)
	{
		daeDeclareException(exInvalidCall); 
		e << "STATE() can only be called after call to STN()";
		throw e;
	}

	daeState* pState = _currentSTN->AddState(strState);
	return pState;
}

void daeModel::END_STN(void)
{
// Current STN must exist!!!
// Otherwise throw an exception
	if(!_currentSTN)
	{
		daeDeclareException(exInvalidCall); 
		e << "END_STN() can only be called after call to STN()";
		throw e;
	}

	_currentSTN->Initialize();
	_currentSTN = NULL;
}

void daeModel::SWITCH_TO(const string& strState, const daeCondition& rCondition, real_t dEventTolerance)
{
// Current STN must exist!!!
// Otherwise throw an exception
	if(!_currentSTN)
	{
		daeDeclareException(exInvalidCall); 
		e << "SWITCH_TO() can only be called after call to STN()";
		throw e;
	}
	
// Current state must exist!!!
// Otherwise throw an exception
	daeState* pCurrentState = GetStateFromStack();
	if(!pCurrentState)
	{
		daeDeclareException(exInvalidCall); 
		e << "SWITCH_TO() can only be called after call to STATE()";
		throw e;
	}

	daeStateTransition* _pST = new daeStateTransition;
	_pST->CreateSTN(string("Condition"), pCurrentState, strState, rCondition, dEventTolerance);
}

void daeModel::AddPortArray(daePortArray& rPortArray, const string& strName, daeePortType ePortType, string strDescription)
{
	rPortArray.SetName(strName);
	rPortArray.SetDescription(strDescription);
	rPortArray.m_ePortType = ePortType;
	AddPortArray(&rPortArray);
}


void daeModel::AddModelArray(daeModelArray& rModelArray, const string& strName, string strDescription)
{
	rModelArray.SetName(strName);
	rModelArray.SetDescription(strDescription);
	AddModelArray(&rModelArray);
}

void daeModel::AddModel(daeModel& rModel, const string& strName, string strDescription)
{
	rModel.SetName(strName);
	rModel.SetDescription(strDescription);
	AddModel(&rModel);
}

void daeModel::AddPort(daePort& rPort, const string& strName, daeePortType ePortType, string strDescription)
{
	rPort.SetName(strName);
	rPort.SetDescription(strDescription);
	rPort.SetType(ePortType);
	AddPort(&rPort);
}
	
daeEquation* daeModel::CreateEquation(const string& strName, string strDescription)
{
	string strEqName;
	daeEquation* pEquation = new daeEquation();
	daeState* pParentState = GetStateFromStack();

	pEquation->SetDescription(strDescription);
	
	if(!pParentState)
	{
		strEqName = (strName.empty() ? "Equation_" + toString<size_t>(m_ptrarrEquations.size()) : strName);
		pEquation->SetName(strEqName);
		AddEquation(pEquation);
	}
	else
	{
		strEqName = (strName.empty() ? "Equation_" + toString<size_t>(pParentState->m_ptrarrEquations.size()) : strName);
		pEquation->SetName(strEqName);
		pParentState->AddEquation(pEquation);
	}
	return pEquation;
}

void daeModel::ConnectPorts(daePort* pPortFrom, daePort* pPortTo)
{
	daePortConnection* pPortConnection = new daePortConnection;
	string strName = pPortFrom->GetName() + "_" + pPortTo->GetName();
	pPortConnection->SetName(strName);
	pPortConnection->m_pPortFrom = pPortFrom;
	pPortConnection->m_pPortTo   = pPortTo;
	AddPortConnection(pPortConnection);
}

boost::shared_ptr<daeDataProxy_t> daeModel::GetDataProxy(void) const
{
	return m_pDataProxy;
}

daeeInitialConditionMode daeModel::GetInitialConditionMode(void) const
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	return m_pDataProxy->GetInitialConditionMode();
}

void daeModel::SetInitialConditionMode(daeeInitialConditionMode eMode)
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	m_pDataProxy->SetInitialConditionMode(eMode);
}

void daeModel::SetReportingOn(bool bOn)
{
	size_t i;
	daePort* pPort;
	daeModel* pModel;
	daePortArray* pPortArray;
	daeModelArray* pModelArray;
	daeVariable* pVariable;

// Set reporting in all variables
	for(i = 0; i < m_ptrarrVariables.size(); i++)
	{
		pVariable = m_ptrarrVariables[i];
		pVariable->SetReportingOn(bOn);
	}
	
// Set reporting in ports
	for(i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pPort = m_ptrarrPorts[i];
		pPort->SetReportingOn(bOn);
	}

// Set reporting in child models
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->SetReportingOn(bOn);
	}
	
// Set reporting in each portarray
	for(i = 0; i < m_ptrarrPortArrays.size(); i++)
	{
		pPortArray = m_ptrarrPortArrays[i];
		pPortArray->SetReportingOn(bOn);
	}
	
// Set reporting in each modelarray
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->SetReportingOn(bOn);
	}
}

//void daeModel::DeclareData()
//{
//	daeDeclareException(exNotImplemented);
//	e << "DeclareData() function MUST be implemented in daeModel derived classes, model [" << m_strCanonicalName << "]";
//	throw e;
//}

void daeModel::DeclareEquations()
{
	daeDeclareException(exNotImplemented);
	e << "DeclareEquations() function MUST be implemented in daeModel derived classes, model [" << m_strCanonicalName << "]";
	throw e;
}

void daeModel::DeclareEquationsBase()
{
	size_t i;
	daeModel* pModel;
	daeModelArray* pModelArray;

// Then, create equations for each child-model
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->DeclareEquations();
	}
	
// Finally, create equations for each modelarray
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->DeclareEquations();
	}
}

void daeModel::CreatePortConnectionEquations(void)
{
	size_t i;
	daePortConnection* pConnection;
	daeModel* pModel;
	daeModelArray* pModelArray;

// First, create port connection equations
	for(i = 0; i < m_ptrarrPortConnections.size(); i++)
	{
		pConnection = m_ptrarrPortConnections[i];
		if(!pConnection)
			daeDeclareException(exInvalidPointer); 

		pConnection->CreateEquations();
	}

// Then, create port connection equations for each child-model
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->CreatePortConnectionEquations();
	}
	
// Finally, create port connection equations for each modelarray
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->CreatePortConnectionEquations();
	}
}

void daeModel::CreateEquationExecutionInfo(daeEquation* pEquation, vector<daeEquationExecutionInfo*>& ptrarrEqnExecutionInfosCreated, bool bAddToTheModel)
{
	size_t d1, d2, d3, d4, d5, d6, d7, d8;
	size_t nNoDomains;
	daeEquationExecutionInfo* pEquationExecutionInfo;
	daeDistributedEquationDomainInfo *pDistrEqnDomainInfo1, *pDistrEqnDomainInfo2, *pDistrEqnDomainInfo3, 
		                             *pDistrEqnDomainInfo4, *pDistrEqnDomainInfo5, *pDistrEqnDomainInfo6,
									 *pDistrEqnDomainInfo7, *pDistrEqnDomainInfo8;

	ptrarrEqnExecutionInfosCreated.clear();

	nNoDomains = pEquation->m_ptrarrDistributedEquationDomainInfos.size();

	if(nNoDomains > 0)
	{
		// Here I have to create one EquationExecutionInfo for each point in each domain
		// where the equation is defined
		if(nNoDomains == 1)
		{
			pDistrEqnDomainInfo1 = pEquation->m_ptrarrDistributedEquationDomainInfos[0];
			if(!pDistrEqnDomainInfo1)
				daeDeclareAndThrowException(exInvalidPointer);

			for(d1 = 0; d1 < pDistrEqnDomainInfo1->m_narrDomainPoints.size(); d1++)
			{
				pEquationExecutionInfo = new daeEquationExecutionInfo;
				pEquationExecutionInfo->m_pEquation = pEquation;
				pEquationExecutionInfo->m_pModel    = this;

				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo1->m_pDomain);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo1->m_narrDomainPoints[d1]);
				if(bAddToTheModel)
					AddEquationExecutionInfo(pEquationExecutionInfo);
				else
					ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
				pEquationExecutionInfo->GatherInfo();
				
				// This vector is redundant - all EquationExecutionInfos already exist in models and states
				// However, it is useful when saving RuntimeReport
				pEquation->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
			}			
		}
		else if(nNoDomains == 2)
		{
			pDistrEqnDomainInfo1 = pEquation->m_ptrarrDistributedEquationDomainInfos[0];
			pDistrEqnDomainInfo2 = pEquation->m_ptrarrDistributedEquationDomainInfos[1];
			if(!pDistrEqnDomainInfo1 || !pDistrEqnDomainInfo2)
				daeDeclareAndThrowException(exInvalidPointer);

			for(d1 = 0; d1 < pDistrEqnDomainInfo1->m_narrDomainPoints.size(); d1++)
			for(d2 = 0; d2 < pDistrEqnDomainInfo2->m_narrDomainPoints.size(); d2++)
			{
				pEquationExecutionInfo = new daeEquationExecutionInfo;
				pEquationExecutionInfo->m_pEquation = pEquation;
				pEquationExecutionInfo->m_pModel    = this;

				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo1->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo2->m_pDomain);

				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo1->m_narrDomainPoints[d1]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo2->m_narrDomainPoints[d2]);

				if(bAddToTheModel)
					AddEquationExecutionInfo(pEquationExecutionInfo);
				else
					ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
				pEquationExecutionInfo->GatherInfo();
				
				// This vector is redundant - all EquationExecutionInfos already exist in models and states
				// However, it is useful when saving RuntimeReport
				pEquation->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
			}
		}
		else if(nNoDomains == 3)
		{
			pDistrEqnDomainInfo1 = pEquation->m_ptrarrDistributedEquationDomainInfos[0];
			pDistrEqnDomainInfo2 = pEquation->m_ptrarrDistributedEquationDomainInfos[1];
			pDistrEqnDomainInfo3 = pEquation->m_ptrarrDistributedEquationDomainInfos[2];
			if(!pDistrEqnDomainInfo1 || !pDistrEqnDomainInfo2 || !pDistrEqnDomainInfo3)
				daeDeclareAndThrowException(exInvalidPointer);

			for(d1 = 0; d1 < pDistrEqnDomainInfo1->m_narrDomainPoints.size(); d1++)
			for(d2 = 0; d2 < pDistrEqnDomainInfo2->m_narrDomainPoints.size(); d2++)
			for(d3 = 0; d3 < pDistrEqnDomainInfo3->m_narrDomainPoints.size(); d3++)
			{
				pEquationExecutionInfo = new daeEquationExecutionInfo;
				pEquationExecutionInfo->m_pEquation = pEquation;
				pEquationExecutionInfo->m_pModel    = this;

				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo1->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo2->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo3->m_pDomain);

				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo1->m_narrDomainPoints[d1]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo2->m_narrDomainPoints[d2]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo3->m_narrDomainPoints[d3]);

				if(bAddToTheModel)
					AddEquationExecutionInfo(pEquationExecutionInfo);
				else
					ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
				pEquationExecutionInfo->GatherInfo();
				
				// This vector is redundant - all EquationExecutionInfos already exist in models and states
				// However, it is useful when saving RuntimeReport
				pEquation->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
			}
		}
		else if(nNoDomains == 4)
		{
			pDistrEqnDomainInfo1 = pEquation->m_ptrarrDistributedEquationDomainInfos[0];
			pDistrEqnDomainInfo2 = pEquation->m_ptrarrDistributedEquationDomainInfos[1];
			pDistrEqnDomainInfo3 = pEquation->m_ptrarrDistributedEquationDomainInfos[2];
			pDistrEqnDomainInfo4 = pEquation->m_ptrarrDistributedEquationDomainInfos[3];
			if(!pDistrEqnDomainInfo1 || !pDistrEqnDomainInfo2 || !pDistrEqnDomainInfo3 || !pDistrEqnDomainInfo4)
				daeDeclareAndThrowException(exInvalidPointer);

			for(d1 = 0; d1 < pDistrEqnDomainInfo1->m_narrDomainPoints.size(); d1++)
			for(d2 = 0; d2 < pDistrEqnDomainInfo2->m_narrDomainPoints.size(); d2++)
			for(d3 = 0; d3 < pDistrEqnDomainInfo3->m_narrDomainPoints.size(); d3++)
			for(d4 = 0; d4 < pDistrEqnDomainInfo4->m_narrDomainPoints.size(); d4++)
			{
				pEquationExecutionInfo = new daeEquationExecutionInfo;
				pEquationExecutionInfo->m_pEquation = pEquation;
				pEquationExecutionInfo->m_pModel    = this;

				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo1->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo2->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo3->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo4->m_pDomain);

				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo1->m_narrDomainPoints[d1]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo2->m_narrDomainPoints[d2]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo3->m_narrDomainPoints[d3]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo4->m_narrDomainPoints[d4]);

				if(bAddToTheModel)
					AddEquationExecutionInfo(pEquationExecutionInfo);
				else
					ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
				pEquationExecutionInfo->GatherInfo();
				
				// This vector is redundant - all EquationExecutionInfos already exist in models and states
				// However, it is useful when saving RuntimeReport
				pEquation->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
			}
		}
		else if(nNoDomains == 5)
		{
			pDistrEqnDomainInfo1 = pEquation->m_ptrarrDistributedEquationDomainInfos[0];
			pDistrEqnDomainInfo2 = pEquation->m_ptrarrDistributedEquationDomainInfos[1];
			pDistrEqnDomainInfo3 = pEquation->m_ptrarrDistributedEquationDomainInfos[2];
			pDistrEqnDomainInfo4 = pEquation->m_ptrarrDistributedEquationDomainInfos[3];
			pDistrEqnDomainInfo5 = pEquation->m_ptrarrDistributedEquationDomainInfos[4];
			if(!pDistrEqnDomainInfo1 || !pDistrEqnDomainInfo2 || !pDistrEqnDomainInfo3 || !pDistrEqnDomainInfo4 || !pDistrEqnDomainInfo5)
				daeDeclareAndThrowException(exInvalidPointer);

			for(d1 = 0; d1 < pDistrEqnDomainInfo1->m_narrDomainPoints.size(); d1++)
			for(d2 = 0; d2 < pDistrEqnDomainInfo2->m_narrDomainPoints.size(); d2++)
			for(d3 = 0; d3 < pDistrEqnDomainInfo3->m_narrDomainPoints.size(); d3++)
			for(d4 = 0; d4 < pDistrEqnDomainInfo4->m_narrDomainPoints.size(); d4++)
			for(d5 = 0; d5 < pDistrEqnDomainInfo5->m_narrDomainPoints.size(); d5++)
			{
				pEquationExecutionInfo = new daeEquationExecutionInfo;
				pEquationExecutionInfo->m_pEquation = pEquation;
				pEquationExecutionInfo->m_pModel    = this;

				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo1->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo2->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo3->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo4->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo5->m_pDomain);

				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo1->m_narrDomainPoints[d1]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo2->m_narrDomainPoints[d2]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo3->m_narrDomainPoints[d3]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo4->m_narrDomainPoints[d4]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo5->m_narrDomainPoints[d5]);

				if(bAddToTheModel)
					AddEquationExecutionInfo(pEquationExecutionInfo);
				else
					ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
				pEquationExecutionInfo->GatherInfo();
				
				// This vector is redundant - all EquationExecutionInfos already exist in models and states
				// However, it is useful when saving RuntimeReport
				pEquation->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
			}
		}
		else if(nNoDomains == 6)
		{
			pDistrEqnDomainInfo1 = pEquation->m_ptrarrDistributedEquationDomainInfos[0];
			pDistrEqnDomainInfo2 = pEquation->m_ptrarrDistributedEquationDomainInfos[1];
			pDistrEqnDomainInfo3 = pEquation->m_ptrarrDistributedEquationDomainInfos[2];
			pDistrEqnDomainInfo4 = pEquation->m_ptrarrDistributedEquationDomainInfos[3];
			pDistrEqnDomainInfo5 = pEquation->m_ptrarrDistributedEquationDomainInfos[4];
			pDistrEqnDomainInfo6 = pEquation->m_ptrarrDistributedEquationDomainInfos[5];
			if(!pDistrEqnDomainInfo1 || !pDistrEqnDomainInfo2 || !pDistrEqnDomainInfo3 || !pDistrEqnDomainInfo4 || 
			   !pDistrEqnDomainInfo5 || !pDistrEqnDomainInfo6)
				daeDeclareAndThrowException(exInvalidPointer);

			for(d1 = 0; d1 < pDistrEqnDomainInfo1->m_narrDomainPoints.size(); d1++)
			for(d2 = 0; d2 < pDistrEqnDomainInfo2->m_narrDomainPoints.size(); d2++)
			for(d3 = 0; d3 < pDistrEqnDomainInfo3->m_narrDomainPoints.size(); d3++)
			for(d4 = 0; d4 < pDistrEqnDomainInfo4->m_narrDomainPoints.size(); d4++)
			for(d5 = 0; d5 < pDistrEqnDomainInfo5->m_narrDomainPoints.size(); d5++)
			for(d6 = 0; d6 < pDistrEqnDomainInfo6->m_narrDomainPoints.size(); d6++)
			{
				pEquationExecutionInfo = new daeEquationExecutionInfo;
				pEquationExecutionInfo->m_pEquation = pEquation;
				pEquationExecutionInfo->m_pModel    = this;

				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo1->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo2->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo3->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo4->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo5->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo6->m_pDomain);

				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo1->m_narrDomainPoints[d1]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo2->m_narrDomainPoints[d2]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo3->m_narrDomainPoints[d3]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo4->m_narrDomainPoints[d4]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo5->m_narrDomainPoints[d5]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo6->m_narrDomainPoints[d6]);

				if(bAddToTheModel)
					AddEquationExecutionInfo(pEquationExecutionInfo);
				else
					ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
				pEquationExecutionInfo->GatherInfo();
				
				// This vector is redundant - all EquationExecutionInfos already exist in models and states
				// However, it is useful when saving RuntimeReport
				pEquation->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
			}
		}
		else if(nNoDomains == 7)
		{
			pDistrEqnDomainInfo1 = pEquation->m_ptrarrDistributedEquationDomainInfos[0];
			pDistrEqnDomainInfo2 = pEquation->m_ptrarrDistributedEquationDomainInfos[1];
			pDistrEqnDomainInfo3 = pEquation->m_ptrarrDistributedEquationDomainInfos[2];
			pDistrEqnDomainInfo4 = pEquation->m_ptrarrDistributedEquationDomainInfos[3];
			pDistrEqnDomainInfo5 = pEquation->m_ptrarrDistributedEquationDomainInfos[4];
			pDistrEqnDomainInfo6 = pEquation->m_ptrarrDistributedEquationDomainInfos[5];
			pDistrEqnDomainInfo7 = pEquation->m_ptrarrDistributedEquationDomainInfos[6];
			if(!pDistrEqnDomainInfo1 || !pDistrEqnDomainInfo2 || !pDistrEqnDomainInfo3 || !pDistrEqnDomainInfo4 || 
			   !pDistrEqnDomainInfo5 || !pDistrEqnDomainInfo6 || !pDistrEqnDomainInfo7)
				daeDeclareAndThrowException(exInvalidPointer);

			for(d1 = 0; d1 < pDistrEqnDomainInfo1->m_narrDomainPoints.size(); d1++)
			for(d2 = 0; d2 < pDistrEqnDomainInfo2->m_narrDomainPoints.size(); d2++)
			for(d3 = 0; d3 < pDistrEqnDomainInfo3->m_narrDomainPoints.size(); d3++)
			for(d4 = 0; d4 < pDistrEqnDomainInfo4->m_narrDomainPoints.size(); d4++)
			for(d5 = 0; d5 < pDistrEqnDomainInfo5->m_narrDomainPoints.size(); d5++)
			for(d6 = 0; d6 < pDistrEqnDomainInfo6->m_narrDomainPoints.size(); d6++)
			for(d7 = 0; d7 < pDistrEqnDomainInfo7->m_narrDomainPoints.size(); d7++)
			{
				pEquationExecutionInfo = new daeEquationExecutionInfo;
				pEquationExecutionInfo->m_pEquation = pEquation;
				pEquationExecutionInfo->m_pModel    = this;

				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo1->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo2->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo3->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo4->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo5->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo6->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo7->m_pDomain);

				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo1->m_narrDomainPoints[d1]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo2->m_narrDomainPoints[d2]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo3->m_narrDomainPoints[d3]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo4->m_narrDomainPoints[d4]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo5->m_narrDomainPoints[d5]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo6->m_narrDomainPoints[d6]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo7->m_narrDomainPoints[d7]);

				if(bAddToTheModel)
					AddEquationExecutionInfo(pEquationExecutionInfo);
				else
					ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
				pEquationExecutionInfo->GatherInfo();
				
				// This vector is redundant - all EquationExecutionInfos already exist in models and states
				// However, it is useful when saving RuntimeReport
				pEquation->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
			}
		}
		else if(nNoDomains == 8)
		{
			pDistrEqnDomainInfo1 = pEquation->m_ptrarrDistributedEquationDomainInfos[0];
			pDistrEqnDomainInfo2 = pEquation->m_ptrarrDistributedEquationDomainInfos[1];
			pDistrEqnDomainInfo3 = pEquation->m_ptrarrDistributedEquationDomainInfos[2];
			pDistrEqnDomainInfo4 = pEquation->m_ptrarrDistributedEquationDomainInfos[3];
			pDistrEqnDomainInfo5 = pEquation->m_ptrarrDistributedEquationDomainInfos[4];
			pDistrEqnDomainInfo6 = pEquation->m_ptrarrDistributedEquationDomainInfos[5];
			pDistrEqnDomainInfo7 = pEquation->m_ptrarrDistributedEquationDomainInfos[6];
			pDistrEqnDomainInfo8 = pEquation->m_ptrarrDistributedEquationDomainInfos[7];
			if(!pDistrEqnDomainInfo1 || !pDistrEqnDomainInfo2 || !pDistrEqnDomainInfo3 || !pDistrEqnDomainInfo4 || 
			   !pDistrEqnDomainInfo5 || !pDistrEqnDomainInfo6 || !pDistrEqnDomainInfo7 || !pDistrEqnDomainInfo8)
				daeDeclareAndThrowException(exInvalidPointer);

			for(d1 = 0; d1 < pDistrEqnDomainInfo1->m_narrDomainPoints.size(); d1++)
			for(d2 = 0; d2 < pDistrEqnDomainInfo2->m_narrDomainPoints.size(); d2++)
			for(d3 = 0; d3 < pDistrEqnDomainInfo3->m_narrDomainPoints.size(); d3++)
			for(d4 = 0; d4 < pDistrEqnDomainInfo4->m_narrDomainPoints.size(); d4++)
			for(d5 = 0; d5 < pDistrEqnDomainInfo5->m_narrDomainPoints.size(); d5++)
			for(d6 = 0; d6 < pDistrEqnDomainInfo6->m_narrDomainPoints.size(); d6++)
			for(d7 = 0; d7 < pDistrEqnDomainInfo7->m_narrDomainPoints.size(); d7++)
			for(d8 = 0; d8 < pDistrEqnDomainInfo8->m_narrDomainPoints.size(); d8++)
			{
				pEquationExecutionInfo = new daeEquationExecutionInfo;
				pEquationExecutionInfo->m_pEquation = pEquation;
				pEquationExecutionInfo->m_pModel    = this;

				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo1->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo2->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo3->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo4->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo5->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo6->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo7->m_pDomain);
				pEquationExecutionInfo->m_ptrarrDomains.push_back(pDistrEqnDomainInfo8->m_pDomain);

				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo1->m_narrDomainPoints[d1]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo2->m_narrDomainPoints[d2]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo3->m_narrDomainPoints[d3]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo4->m_narrDomainPoints[d4]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo5->m_narrDomainPoints[d5]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo6->m_narrDomainPoints[d6]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo7->m_narrDomainPoints[d7]);
				pEquationExecutionInfo->m_narrDomainIndexes.push_back(pDistrEqnDomainInfo8->m_narrDomainPoints[d8]);

				if(bAddToTheModel)
					AddEquationExecutionInfo(pEquationExecutionInfo);
				else
					ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
				pEquationExecutionInfo->GatherInfo();
				
				// This vector is redundant - all EquationExecutionInfos already exist in models and states
				// However, it is useful when saving RuntimeReport
				pEquation->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
			}
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented);
		}
	}
	else
	{
		pEquationExecutionInfo = new daeEquationExecutionInfo;
		pEquationExecutionInfo->m_pEquation = pEquation;
		pEquationExecutionInfo->m_pModel = this;
		if(bAddToTheModel)
			AddEquationExecutionInfo(pEquationExecutionInfo);
		else
			ptrarrEqnExecutionInfosCreated.push_back(pEquationExecutionInfo);
		pEquationExecutionInfo->GatherInfo();

		// This vector is redundant - all EquationExecutionInfos already exist in models and states
		// However, it is useful when saving RuntimeReport
		pEquation->m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
	}
}

void daeModel::PropagateDataProxy(boost::shared_ptr<daeDataProxy_t> pDataProxy)
{
	size_t i;
	daeModel* pModel;
	daeModelArray* pModelArray;

	m_pDataProxy = pDataProxy;

	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		pModel->PropagateDataProxy(pDataProxy);
	}
	
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		if(!pModelArray)
			daeDeclareAndThrowException(exInvalidPointer);

		pModelArray->PropagateDataProxy(pDataProxy);
	}
}

void daeModel::PropagateGlobalExecutionContext(daeExecutionContext* pExecutionContext)
{
	size_t i;
	daeModel* pModel;
	daeModelArray* pModelArray;

	m_pExecutionContextForGatherInfo = pExecutionContext;

	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		pModel->PropagateGlobalExecutionContext(pExecutionContext);
	}
	
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		if(!pModelArray)
			daeDeclareAndThrowException(exInvalidPointer);

		pModelArray->PropagateGlobalExecutionContext(pExecutionContext);
	}
}

void daeModel::BuildUpSTNsAndEquations()
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	daeExecutionContext EC;
	EC.m_pDataProxy               = m_pDataProxy.get();
	EC.m_eEquationCalculationMode = eCreateFunctionsIFsSTNs;

	m_pDataProxy->SetGatherInfo(true);
	PropagateGlobalExecutionContext(&EC);
		m_ptrarrStackStates.clear();
	// Declare equations in this model	
		DeclareEquations();
	// Declare equations in child models and model arrays
		DeclareEquationsBase();
		m_ptrarrStackStates.clear();
	// Create indexes in DEDIs (they are not created in the moment of declaration!)
		InitializeDEDIs();
	// Create runtime condition nodes based on setup nodes
		InitializeSTNs();		
	m_pDataProxy->SetGatherInfo(false);
	PropagateGlobalExecutionContext(NULL);
}

void daeModel::InitializeDEDIs(void)
{
	size_t i, k;
	daeSTN* pSTN;
	daeModel* pModel;
	daePortConnection* pConnection;
	daeModelArray* pModelArray;
	daeEquation* pEquation;
	
// First, InitializeDEDIs for equations in this model
	for(i = 0; i < m_ptrarrEquations.size(); i++)
	{
		pEquation = m_ptrarrEquations[i];
		if(!pEquation)
			daeDeclareAndThrowException(exInvalidPointer);

		pEquation->InitializeDEDIs();
	}

// Then, InitializeDEDIs for port connection equations
	for(i = 0; i < m_ptrarrPortConnections.size(); i++)
	{
		pConnection = m_ptrarrPortConnections[i];
		if(!pConnection)
			daeDeclareException(exInvalidPointer); 

		for(k = 0; k < pConnection->m_ptrarrEquations.size(); k++)
		{
			pEquation = pConnection->m_ptrarrEquations[k];
			if(!pEquation)
				daeDeclareAndThrowException(exInvalidPointer);
	
			pEquation->InitializeDEDIs();
		}
	}

	// Then, InitializeDEDIs for equations in the STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer);

		pSTN->InitializeDEDIs();
	}

// Next, InitializeDEDIs for equations in each child-model
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		pModel->InitializeDEDIs();
	}
	
// Finally, InitializeDEDIs for equations in each modelarray
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		if(!pModelArray)
			daeDeclareAndThrowException(exInvalidPointer);
		pModelArray->InitializeDEDIs();
	}
}

void daeModel::BuildUpPortConnectionEquations()
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	daeExecutionContext EC;
	EC.m_pDataProxy               = m_pDataProxy.get();
	EC.m_eEquationCalculationMode = eCreateFunctionsIFsSTNs;

	m_pDataProxy->SetGatherInfo(true);
	PropagateGlobalExecutionContext(&EC);
		CreatePortConnectionEquations();
	m_pDataProxy->SetGatherInfo(false);
	PropagateGlobalExecutionContext(NULL);
}

void daeModel::InitializeParameters()
{
	size_t i;
	daePort* pPort;
	daeModel* pModel;
	daeModelArray* pModelArray;
	daePortArray* pPortArray;
	daeParameter* pParameter;

// First, initialize all parameters in the model
	for(i = 0; i < m_ptrarrParameters.size(); i++)
	{
		pParameter = m_ptrarrParameters[i];
		pParameter->Initialize();
	}

// Then, initialize all parameters in the contained ports
	for(i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pPort = m_ptrarrPorts[i];
		pPort->InitializeParameters();
	}

// Then, initialize all parameters in the child-models
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->InitializeParameters();
	}
	
// Next, initialize all parameters in each portarray
	for(i = 0; i < m_ptrarrPortArrays.size(); i++)
	{
		pPortArray = m_ptrarrPortArrays[i];
		pPortArray->InitializeParameters();
	}
	
// Finally, initialize all parameters in the modelarrays
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->InitializeParameters();
	}
}

void daeModel::InitializePortAndModelArrays()
{
	size_t i;
	daeModel* pModel;
	daeModelArray* pModelArray;
	daePortArray* pPortArray;

// First, initialize all port arrays in the model
	for(i = 0; i < m_ptrarrPortArrays.size(); i++)
	{
		pPortArray = m_ptrarrPortArrays[i];
		pPortArray->Create();
	}

// Then, initialize all model arrays in the model
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->Create();
	}

// Finally, initialize all arrays in the child-models
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->InitializePortAndModelArrays();
	}
}

void daeModel::InitializeVariables()
{
	size_t i;
	daePort* pPort;
	daeModel* pModel;
	daeVariable* pVariable;
	daeModelArray* pModelArray;
	daePortArray* pPortArray;

	m_nTotalNumberOfVariables = 0;
	_currentVariablesIndex = m_nVariablesStartingIndex;

// First, initialize all variables in the model
	for(i = 0; i < m_ptrarrVariables.size(); i++)
	{
		pVariable = m_ptrarrVariables[i];
		pVariable->m_nOverallIndex = _currentVariablesIndex;
		_currentVariablesIndex += pVariable->GetNumberOfPoints();
	}

// Then, initialize all variables in the contained ports
	for(i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pPort = m_ptrarrPorts[i];
		pPort->SetVariablesStartingIndex(_currentVariablesIndex);
		pPort->InitializeVariables();
		_currentVariablesIndex = pPort->_currentVariablesIndex;
	}

// Then, initialize all variables in the child-models
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->SetVariablesStartingIndex(_currentVariablesIndex);
		pModel->InitializeVariables();
		_currentVariablesIndex = pModel->_currentVariablesIndex;
	}

// Next, initialize all variables in the portarrays
	for(i = 0; i < m_ptrarrPortArrays.size(); i++)
	{
		pPortArray = m_ptrarrPortArrays[i];
		pPortArray->SetVariablesStartingIndex(_currentVariablesIndex);
		pPortArray->InitializeVariables();
		_currentVariablesIndex = pPortArray->_currentVariablesIndex;
	}
	
// Finally, initialize all variables in the modelarrays
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->SetVariablesStartingIndex(_currentVariablesIndex);
		pModelArray->InitializeVariables();
		_currentVariablesIndex = pModelArray->_currentVariablesIndex;
	}
	
	m_nTotalNumberOfVariables = _currentVariablesIndex - m_nVariablesStartingIndex;
}

void daeModel::InitializeSTNs(void)
{
	size_t i, k;
	daeSTN* pSTN;
	daeModel* pModel;
	daeModelArray* pModelArray;
	
	// Initialize STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer);

		pSTN->InitializeStateTransitions();
	}

// Next, initialize STNs in each child-model
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->InitializeSTNs();
	}
	
// Finally, initialize STNs in each modelarray
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->InitializeSTNs();
	}
}

void daeModel::InitializeEquations()
{
	size_t i, k;
	daeSTN* pSTN;
	daeModel* pModel;
	daePortConnection* pConnection;
	daeModelArray* pModelArray;
	daeEquation* pEquation;
	vector<daeEquationExecutionInfo*> ptrarrEqnExecutionInfosCreated;
	
// First, create EqnExecInfos info for equations in this model
	for(i = 0; i < m_ptrarrEquations.size(); i++)
	{
		pEquation = m_ptrarrEquations[i];
		if(!pEquation)
			daeDeclareAndThrowException(exInvalidPointer);

	// Create EqnExecInfos, call GatherInfo for each of them, and add them to the model
		CreateEquationExecutionInfo(pEquation, ptrarrEqnExecutionInfosCreated, true);
	}

// Then, create EqnExecInfos for port connection equations
	for(i = 0; i < m_ptrarrPortConnections.size(); i++)
	{
		pConnection = m_ptrarrPortConnections[i];
		if(!pConnection)
			daeDeclareException(exInvalidPointer); 

		for(k = 0; k < pConnection->m_ptrarrEquations.size(); k++)
		{
			pEquation = pConnection->m_ptrarrEquations[k];
			if(!pEquation)
				daeDeclareAndThrowException(exInvalidPointer);
	
		// Create EqnExecInfos, call GatherInfo for each of them, and add them to the model
			CreateEquationExecutionInfo(pEquation, ptrarrEqnExecutionInfosCreated, true);
		}
	}

	// Then, create EqnExecInfos for equations in the STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer);

		pSTN->CreateEquationExecutionInfo();
	}

// Next, create EqnExecInfos for equations in each child-model
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->InitializeEquations();
	}
	
// Finally, gather info for equations in each modelarray
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->InitializeEquations();
	}
}

void daeModel::CollectAllSTNs(vector<daeSTN*>& ptrarrSTNs) const
{
	size_t i;
	daeModel* pModel;
	daeModelArray* pModelArray;

// Fill array ptrarrSTNs with all STNs in the model
	ptrarrSTNs.insert(ptrarrSTNs.end(), m_ptrarrSTNs.begin(), m_ptrarrSTNs.end());

/////////////////////////////////////////////////////////////////////////////////////
// BUG!!!
// What about STNs nested within states???
// States should take care of them, I guess
/////////////////////////////////////////////////////////////////////////////////////

// Then, fill it with STNs in each child-model
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->CollectAllSTNs(ptrarrSTNs);
	}
	
// Finally, ill it with STNs in each modelarray
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->CollectAllSTNs(ptrarrSTNs);
	}
}

void daeModel::CollectEquationExecutionInfosFromSTNs(vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const
{
	size_t i;
	daeSTN* pSTN;
	daeModel* pModel;
	daeModelArray* pModelArray;

// Fill array ptrarrEquationExecutionInfo with all execution infos in STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer);

		pSTN->CollectEquationExecutionInfos(ptrarrEquationExecutionInfo);
	}

/////////////////////////////////////////////////////////////////////////////////////
// BUG!!!
// What about STNs nested within states???
// States should take care of them, I guess
/////////////////////////////////////////////////////////////////////////////////////

// Then, fill it with execution info in each child-model
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->CollectEquationExecutionInfosFromSTNs(ptrarrEquationExecutionInfo);
	}
	
// Finally, fill it with execution info in each modelarray
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->CollectEquationExecutionInfosFromSTNs(ptrarrEquationExecutionInfo);
	}
}

void daeModel::CollectEquationExecutionInfosFromModels(vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const
{
	size_t i;
	daeModel* pModel;
	daeModelArray* pModelArray;
	daeEquationExecutionInfo* pEquationExecutionInfo;

// Fill array ptrarrEquationExecutionInfo with all execution info in the model
	for(i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
	{
		pEquationExecutionInfo = m_ptrarrEquationExecutionInfos[i];
		if(!pEquationExecutionInfo)
			daeDeclareAndThrowException(exInvalidPointer);

		ptrarrEquationExecutionInfo.push_back(pEquationExecutionInfo);
	}
	
// Then, fill it with execution info in each child-model
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		pModel->CollectEquationExecutionInfosFromModels(ptrarrEquationExecutionInfo);
	}
	
// Finally, fill it with execution info in each modelarray
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->CollectEquationExecutionInfosFromModels(ptrarrEquationExecutionInfo);
	}
}

void daeModel::DoBlockDecomposition(bool bDoBlockDecomposition, vector<daeBlock_t*>& ptrarrBlocks)
{
	size_t i, k, m;
//	bool bOverlapped;
//	long j, Nleft;
	size_t nIndex, nEquationIndex;
//	daeSTN *pSTNin, *pSTNout;
//	daeState* pState;
	daeBlock* pBlock;
	daeBoolArray barrVars;
	pair<size_t, size_t> uintPair;
	vector<size_t> narrVariablesIndexesInEquation;
	vector<daeSTN*>	ptrarrSTNs;
	vector<daeVariable*> ptrarrAllVariables;
	map<size_t, size_t>::iterator iter;
	vector<daeEquationExecutionInfo*> ptrarrEEIfromModels, ptrarrEEIfromSTNs, ptrarrBlockEquationExecutionInfo;
	vector<string> strarrErrors;
	daeEquationExecutionInfo *pEquationExec;
	daeEquationExecutionInfo *pEqExec;
	daeEquation *pEquation;
	daeBoolArray* pbarrVariableFlags;
	vector<daeEquationExecutionInfo*> ptrarrAllEquationExecutionInfosInModel;

	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

/***********************************************************************************
	Populate vector with all existing equation execution infos
************************************************************************************/
	CollectEquationExecutionInfosFromModels(ptrarrEEIfromModels);

//	Ovo prikuplja samo EquationInfos iz aktivnog stanja!!! 
//	To vodi u gresku kada imam ukljucenu BlockDecomposition
	CollectEquationExecutionInfosFromSTNs(ptrarrEEIfromSTNs);

	ptrarrAllEquationExecutionInfosInModel.insert(ptrarrAllEquationExecutionInfosInModel.begin(), ptrarrEEIfromSTNs.begin(),   ptrarrEEIfromSTNs.end());
	ptrarrAllEquationExecutionInfosInModel.insert(ptrarrAllEquationExecutionInfosInModel.end(),   ptrarrEEIfromModels.begin(), ptrarrEEIfromModels.end());

//	size_t nVar = m_pDataProxy->GetTotalNumberOfVariables();
//	size_t nEq  = ptrarrAllEquationExecutionInfosInModel.size();

/***********************************************************************************
	Print the initial system before block decomposition
************************************************************************************/
/**/
//	pbarrVariableFlags = new daeBoolArray[nEq];
//	for(k = 0; k < nEq; k++)
//	{
//		pEquationExec = ptrarrAllEquationExecutionInfosInModel[k];
//		if(!pEquationExec)
//			daeDeclareAndThrowException(exInvalidPointer);
//		//pEquation = pEquationExec->m_pEquation;
//		//if(!pEquation)
//		//	daeDeclareAndThrowException(exInvalidPointer);
//
//		pbarrVariableFlags[k].resize(nVar, false);
//		for(i = 0; i < pEquationExec->m_narrDomainIndexes.size(); i++)
//		{
//			nIndex = pEquationExec->m_narrDomainIndexes[i];
//			pbarrVariableFlags[k][nIndex] = true;
//		}
//	}
//
//	cout << "Initial system before block decomposition:" << endl;
//	cout << "     Number of variables: " << nVar << endl;
//	cout << "     Number of equations: " << nEq << endl;
//	for(k = 0; k < nEq; k++)
//	{
//		for(m = 0; m < nVar; m++)
//			cout << (pbarrVariableFlags[k][m] ? "X" : "-") << " ";
//		cout << endl;
//	}
//	cout << endl << endl;
//	delete[] pbarrVariableFlags;
/**/

/***********************************************************************************
	Build-up the blocks
************************************************************************************/
	if(bDoBlockDecomposition)
	{
	}
	else // Without the blockdecomposition
	{
		size_t i, k;
		daeSTN* pSTN;
		size_t nNoEquations = ptrarrAllEquationExecutionInfosInModel.size();
		pBlock = new daeBlock;
		pBlock->SetName(string("Block N-1"));
		pBlock->SetDataProxy(m_pDataProxy.get());
		ptrarrBlocks.push_back(pBlock);

		for(i = 0; i < nNoEquations; i++)
		{
			pEquationExec = ptrarrAllEquationExecutionInfosInModel[i];
			if(!pEquationExec)
				daeDeclareAndThrowException(exInvalidPointer);

			pBlock->AddVariables(pEquationExec->m_mapIndexes);
		}

////////////////////////////////////////////////////////////////////////
// BUG!!!! 30.07.2009
// A sta sa STNovima iz child modela i modelarrays????
// 31.07.2009 I corrected the code and now I use ALL STNs
////////////////////////////////////////////////////////////////////////
		vector<daeSTN*> ptrarrAllSTNs;
		
		CollectAllSTNs(ptrarrAllSTNs);
		
		map<size_t, size_t> mapVariableIndexes;
		for(i = 0; i < ptrarrAllSTNs.size(); i++)
		{
			pSTN = ptrarrAllSTNs[i];
			if(!pSTN)
				daeDeclareAndThrowException(exInvalidPointer);

			pSTN->CollectVariableIndexes(mapVariableIndexes);
		}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
		
		map<size_t, size_t>::iterator iter, iterIndexInBlock;

		nEquationIndex = 0;
		for(k = 0; k < ptrarrEEIfromModels.size(); k++)
		{
			pEquationExec = ptrarrEEIfromModels[k];
			if(!pEquationExec)
				daeDeclareAndThrowException(exInvalidPointer);

			pEquationExec->m_nEquationIndexInBlock = nEquationIndex;
			pEquationExec->m_pBlock = pBlock;
			pBlock->AddEquationExecutionInfo(pEquationExec);
//----------------->
		// Here I have to associate overall variable indexes in equation to corresponding indexes in the block
		// m_mapIndexes<OverallIndex, BlockIndex>
			for(iter = pEquationExec->m_mapIndexes.begin(); iter != pEquationExec->m_mapIndexes.end(); iter++)
			{
			// Try to find OverallIndex in the map of BlockIndexes
				iterIndexInBlock = pBlock->m_mapVariableIndexes.find((*iter).first);
				if(iterIndexInBlock == pBlock->m_mapVariableIndexes.end())
				{
					daeDeclareException(exInvalidCall);
					e << "Cannot find overall variable index [" << toString<size_t>((*iter).first) << "] in equation " << pEquationExec->m_pEquation->m_strCanonicalName;
					throw e;
				}
				(*iter).second = (*iterIndexInBlock).second;
			}
//------------------->
			nEquationIndex++;
		}

		pBlock->m_ptrarrSTNs = ptrarrAllSTNs;
		for(i = 0; i < ptrarrAllSTNs.size(); i++)
		{
			pSTN = ptrarrAllSTNs[i];
			if(!pSTN)
				daeDeclareAndThrowException(exInvalidPointer);

			if(pSTN->m_ptrarrStates.size() == 0)
			{
				daeDeclareException(exInvalidCall);
				e << "Number of states is 0 in STN " << pSTN->m_strCanonicalName;
				throw e;
			}

			pSTN->SetIndexesWithinBlockToEquationExecutionInfos(pBlock, nEquationIndex);

		}

	// Initialize the block
		pBlock->Initialize();

	// Finaly, check the block
		if(!pBlock->CheckObject(strarrErrors))
		{
			daeDeclareException(exRuntimeCheck);
			for(vector<string>::iterator it = strarrErrors.begin(); it != strarrErrors.end(); it++)
				e << *it << string("\n");
			throw e;
		}
				
// Print the block
//		cout << "Results of block decomposition:" << endl;
//		barrVars.resize(nNoEquations, false);
//		for(iter = pBlock->m_mapVariableIndexes.begin(); iter != pBlock->m_mapVariableIndexes.end(); iter++)
//			barrVars[iter->first] = true;
//	
//		for(k = 0; k < nNoEquations; k++)
//			cout << (barrVars[k] ? "X" : "-") << " ";
//		cout << endl;
//	
//		for(k = 0; k < nNoEquations; k++)
//			barrVars[k] = false;

	}
	
/***********************************************************************************
	Print the results of block decomposition
************************************************************************************/
/*
	cout << "Results of block decomposition:" << endl;
	barrVars.resize(nVar, false);
	for(i = 0; i < m_ptrarrBlocks.size(); i++)
	{
		pBlock = m_ptrarrBlocks[i];
		for(iter = pBlock->m_mapVariableIndexes.begin(); iter != pBlock->m_mapVariableIndexes.end(); iter++)
			barrVars[iter->first] = true;

		for(k = 0; k < nVar; k++)
			cout << (barrVars[k] ? "X" : "-") << " ";
		cout << endl;

		for(k = 0; k < nVar; k++)
			barrVars[k] = false;
	}
*/
}

void daeModel::SetDefaultInitialGuesses(void)
{
	size_t i;
	daeModel* pModel;
	daeModelArray* pModelArray;
	daePortArray* pPortArray;
	daeVariable* pVariable;
	daeVariableType_t* pVariableType;

	size_t nNumberOfVariables = m_ptrarrVariables.size();
	for(size_t i = 0; i < nNumberOfVariables; i++)
	{
		pVariable = m_ptrarrVariables[i];
		if(!pVariable)
			daeDeclareAndThrowException(exInvalidPointer);
		pVariableType = pVariable->GetVariableType();
		if(!pVariableType)
			daeDeclareAndThrowException(exInvalidPointer); 
		pVariable->SetInitialGuesses(pVariableType->GetInitialGuess());
	}

	daePort* pPort;
	for(i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pPort = m_ptrarrPorts[i];
		if(!pPort)
			daeDeclareAndThrowException(exInvalidPointer); 
		
		nNumberOfVariables = pPort->m_ptrarrVariables.size();
		for(size_t k = 0; k < nNumberOfVariables; k++)
		{
			pVariable = pPort->m_ptrarrVariables[k];
			if(!pVariable)
				daeDeclareAndThrowException(exInvalidPointer); 
			pVariableType = pVariable->GetVariableType();
			if(!pVariableType)
				daeDeclareAndThrowException(exInvalidPointer); 
			pVariable->SetInitialGuesses(pVariableType->GetInitialGuess());
		}	
	}	

	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer); 
		pModel->SetDefaultInitialGuesses();
	}

	for(i = 0; i < m_ptrarrPortArrays.size(); i++)
	{
		pPortArray = m_ptrarrPortArrays[i];
		pPortArray->SetDefaultInitialGuesses();
	}
	
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->SetDefaultInitialGuesses();
	}
}

void daeModel::SetDefaultAbsoluteTolerances()
{
	size_t i;
	daeModel* pModel;
	daeVariable* pVariable;
	daeVariableType_t* pVariableType;
	daeModelArray* pModelArray;
	daePortArray* pPortArray;

	size_t nNumberOfVariables = m_ptrarrVariables.size();
	for(i = 0; i < nNumberOfVariables; i++)
	{
		pVariable = m_ptrarrVariables[i];
		if(!pVariable)
			daeDeclareAndThrowException(exInvalidPointer); 
		pVariableType = pVariable->GetVariableType();
		if(!pVariableType)
			daeDeclareAndThrowException(exInvalidPointer); 
		pVariable->SetAbsoluteTolerances(pVariableType->GetAbsoluteTolerance());
	}	

	daePort* pPort;
	for(i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pPort = m_ptrarrPorts[i];
		if(!pPort)
			daeDeclareAndThrowException(exInvalidPointer); 
		
		nNumberOfVariables = pPort->m_ptrarrVariables.size();
		for(size_t k = 0; k < nNumberOfVariables; k++)
		{
			pVariable = pPort->m_ptrarrVariables[k];
			if(!pVariable)
				daeDeclareAndThrowException(exInvalidPointer); 
			pVariableType = pVariable->GetVariableType();
			if(!pVariableType)
				daeDeclareAndThrowException(exInvalidPointer); 
			pVariable->SetAbsoluteTolerances(pVariableType->GetAbsoluteTolerance());
		}	
	}	

	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer); 
		pModel->SetDefaultAbsoluteTolerances();
	}
	
	for(i = 0; i < m_ptrarrPortArrays.size(); i++)
	{
		pPortArray = m_ptrarrPortArrays[i];
		pPortArray->SetDefaultAbsoluteTolerances();
	}
	
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		pModelArray->SetDefaultAbsoluteTolerances();
	}
}

void daeModel::SetGlobalConditionContext(void)
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_globalConditionContext.m_pDataProxy				= m_pDataProxy.get();
	m_globalConditionContext.m_eEquationCalculationMode	= eCreateFunctionsIFsSTNs;

	m_pDataProxy->SetGatherInfo(true);
	PropagateGlobalExecutionContext(&m_globalConditionContext);
}

void daeModel::UnsetGlobalConditionContext(void)
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_globalConditionContext.m_pDataProxy				= NULL;
	m_globalConditionContext.m_eEquationCalculationMode	= eECMUnknown;

	m_pDataProxy->SetGatherInfo(false);
	PropagateGlobalExecutionContext(NULL);
}

void daeModel::SetGlobalCondition(daeCondition condition)
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pExecutionContextForGatherInfo)
		daeDeclareAndThrowException(exInvalidPointer);
	
	if(m_pCondition)
		delete m_pCondition;
	m_pCondition = new daeCondition;

	*m_pCondition          = condition.m_pConditionNode->CreateRuntimeNode(m_pExecutionContextForGatherInfo);
	m_pCondition->m_pModel = this;
	m_pCondition->BuildExpressionsArray(m_pExecutionContextForGatherInfo);

	m_pDataProxy->SetGatherInfo(false);
	PropagateGlobalExecutionContext(NULL);
}

void daeModel::ResetGlobalCondition(void)
{
	if(m_pCondition)
		delete m_pCondition;
	m_pCondition = NULL;
}

daeCondition* daeModel::GetGlobalCondition() const
{
	return m_pCondition;	
}

size_t daeModel::GetNumberOfSTNs(void) const
{
	return m_ptrarrSTNs.size();
}

daeState* daeModel::GetStateFromStack()
{
	if(m_ptrarrStackStates.size() == 0)
		return NULL;
	return m_ptrarrStackStates[m_ptrarrStackStates.size() - 1];
}

void daeModel::PutStateToStack(daeState* pState)
{
	m_ptrarrStackStates.push_back(pState);
}

void daeModel::RemoveStateFromStack()
{
	if(!m_ptrarrStackStates.empty())
		m_ptrarrStackStates.pop_back();
}

daeDomain* daeModel::FindDomain(unsigned long nID) const
{
	size_t i;
	daeModel* pModel;
	daePort* pPort;
	daeDomain* pDomain;

// Look in local domains
	for(i = 0; i < m_ptrarrDomains.size(); i++)
	{
		pDomain = m_ptrarrDomains[i];
		if(pDomain && pDomain->m_nID == nID)
			return pDomain;
	}
// Look in local ports
	for(i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pPort = m_ptrarrPorts[i];
		pDomain = pPort->FindDomain(nID);
		if(pDomain)
			return pDomain;
	}
// Look in child models' domains
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		pDomain = pModel->FindDomain(nID);
		if(pDomain)
			return pDomain;
	}

	return NULL;
}

daeVariable* daeModel::FindVariable(unsigned long nID) const
{
	size_t i;
	daeModel* pModel;
	daePort* pPort;
	daeVariable* pVariable;

// Look in local variables
	for(i = 0; i < m_ptrarrVariables.size(); i++)
	{
		pVariable = m_ptrarrVariables[i];
		if(pVariable && pVariable->m_nID == nID)
			return pVariable;
	}
// Look in local ports
	for(i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pPort = m_ptrarrPorts[i];
		pVariable = pPort->FindVariable(nID);
		if(pVariable)
			return pVariable;
	}
// Look in child models' variables
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		pVariable = pModel->FindVariable(nID);
		if(pVariable)
			return pVariable;
	}

	return NULL;
}

daePort* daeModel::FindPort(unsigned long nID) const
{
	size_t i;
	daeModel* pModel;
	daePort* pPort;

	for(i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pPort = m_ptrarrPorts[i];
		if(pPort && pPort->m_nID == nID)
			return pPort;
	}
// Look in child models' ports
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		pPort = pModel->FindPort(nID);
		if(pPort)
			return pPort;
	}

	return NULL;
}

bool daeModel::FindObject(string& strCanonicalName, daeObjectType& ObjectType)
{
	bool bFound;
	string strTemp;
	vector<string> strarrHierarchy;

	strarrHierarchy = ParseString(strCanonicalName, '.');
	ObjectType.m_strFullName = strCanonicalName;

	if(strarrHierarchy.empty())
	{
// Invalid string was sent
		return false;
	}
	else if(strarrHierarchy.size() == 1) 
	{
// Its only a single name, lets find what kind of object it is
		bFound = ParseSingleToken(strarrHierarchy[0], ObjectType.m_strName, ObjectType.m_narrDomains);
		if(!bFound)
			return false;
		return DetectObject(ObjectType.m_strName, ObjectType.m_narrDomains, ObjectType.m_eObjectType, &ObjectType.m_pObject);
	}
	else
	{
// Its a name with the complex hierarchy so set the calling object to this pointer and call FindObject iterativelly...
		return FindObject(strarrHierarchy, ObjectType);
	}

	return true;
}

bool daeModel::FindObject(vector<string>& strarrHierarchy, daeObjectType& ObjectType)
{
	bool			bFound;
	daeeObjectType	eObjectType;
	string			strName;
	vector<size_t>	narrDomains;
	daeObject_t*	pObject;

	try
	{
		if(strarrHierarchy.empty()) 
			return false;

		bFound = ParseSingleToken(strarrHierarchy[0], strName, narrDomains);
		if(!bFound)
			return false;
		bFound = DetectObject(strName, narrDomains, eObjectType, &pObject);
		if(!bFound)
			return false;

	// If there is only one item left then fill the structure and return
		if(strarrHierarchy.size() == 1)
		{
	// What if I want to check whether certain port exists (for instance - then the last item is not var/param/domain!!!
	// If I call this function from daeModel then the last item can be something else beside a var/param/domain...
	// In that case I dont need this IF bellow
			//if(eObjectType != eObjectTypeParameter &&
			//   eObjectType != eObjectTypeDomain    &&
			//   eObjectType != eObjectTypeVariable)
			//	return false;

			ObjectType.m_eObjectType = eObjectType;
			ObjectType.m_strName	 = strName;
			ObjectType.m_narrDomains = narrDomains;
			ObjectType.m_pObject     = pObject;
			return true;
		}

	// There are more than one item, so the item I have just detected must be some container (model, port)
	// It cannot be model- or port-array because I have string like model(n1, n2, ..., nn).[...].modelarray(n1, n2, ..., nn)
	// Thus if I ask for a pointer to a model- or port-array it can be the last item but not in the middle of the string
		if(eObjectType != eObjectTypePort && eObjectType != eObjectTypeModel)
			return false;

	// Remove the processed item
		strarrHierarchy.erase(strarrHierarchy.begin());

	// Depending on a type of a container, continue searching
		daePort* pPort;
		daeModel* pModel;
		if(eObjectType == eObjectTypePort)
		{
			pPort = dynamic_cast<daePort*>(pObject);
			if(!pPort)
				return false;
			return pPort->FindObject(strarrHierarchy, ObjectType);
		}
		else if(eObjectType == eObjectTypeModel)
		{
			pModel = dynamic_cast<daeModel*>(pObject);
			if(!pModel)
				return false;
			return pModel->FindObject(strarrHierarchy, ObjectType);
		}
		else
		{
			return false;
		}
	}
	catch(std::exception& e)
	{
		string strException = e.what();
		return false;
	}

	return false;
}

bool daeModel::DetectObject(string& strShortName, vector<size_t>& narrDomains, daeeObjectType& eType, daeObject_t** ppObject)
{
	daeObject_t* pObject;

	try
	{
		if(m_strShortName == strShortName)
		{
			eType     = eObjectTypeModel;
			*ppObject = this;
			return true;
		}

		pObject = FindDomain(strShortName);
		if(pObject)
		{
			eType     = eObjectTypeDomain;
			*ppObject = pObject;
			return true;
		}

		pObject = FindParameter(strShortName);
		if(pObject)
		{
			eType     = eObjectTypeParameter;
			*ppObject = pObject;
			return true;
		}

		pObject = FindVariable(strShortName);
		if(pObject)
		{
			eType     = eObjectTypeVariable;
			*ppObject = pObject;
			return true;
		}

		pObject = FindPort(strShortName);
		if(pObject)
		{
			if(narrDomains.size() > 0)
			{	
				eType     = eObjectTypePortArray;
				*ppObject = pObject;
				return true;
			}
			eType     = eObjectTypePort;
			*ppObject = pObject;
			return true;
		}

		pObject = FindModel(strShortName);
		if(pObject)
		{
			if(narrDomains.size() > 0)
			{	
				eType     = eObjectTypeUnknown;
				*ppObject = NULL;
				return false;
			}
			eType     = eObjectTypeModel;
			*ppObject = pObject;
			return true;
		}

		pObject = FindPortArray(strShortName);
		if(pObject)
		{
			if(narrDomains.size() == 0)
			{	
				eType     = eObjectTypeUnknown;
				*ppObject = NULL;
				return false;
			}

			daePortArray* pPortArray = dynamic_cast<daePortArray*>(pObject);
			if(!pPortArray)
			{	
				eType     = eObjectTypeUnknown;
				*ppObject = NULL;
				return false;
			}
			if(narrDomains.size() != pPortArray->GetDimensions())
			{	
				eType     = eObjectTypeUnknown;
				*ppObject = NULL;
				return false;
			}

			daePort_t* pPort = pPortArray->GetPort(narrDomains);
			if(!pPort)
			{
				eType     = eObjectTypeUnknown;
				*ppObject = NULL;
				return false;
			}

			eType     = eObjectTypePort;
			*ppObject = pPort;
			return true;
		}

		pObject = FindModelArray(strShortName);
		if(pObject)
		{
			if(narrDomains.size() == 0)
			{	
				eType     = eObjectTypeModelArray;
				*ppObject = pObject;
				return true;
			}

			daeModelArray* pModelArray = dynamic_cast<daeModelArray*>(pObject);
			if(!pModelArray)
			{	
				eType     = eObjectTypeUnknown;
				*ppObject = NULL;
				return false;
			}
			if(narrDomains.size() != pModelArray->GetDimensions())
			{	
				eType     = eObjectTypeUnknown;
				*ppObject = NULL;
				return false;
			}

			daeModel_t* pModel = pModelArray->GetModel(narrDomains);
			if(!pModel)
			{
				eType     = eObjectTypeUnknown;
				*ppObject = NULL;
				return false;
			}

			eType     = eObjectTypeModel;
			*ppObject = pModel;
			return true;
		}
	}
	catch(std::exception& e)
	{
		string strException = e.what();
		eType     = eObjectTypeUnknown;
		*ppObject = NULL;
		return false;
	}

	eType     = eObjectTypeUnknown;
	*ppObject = NULL;
	return false;
}
	
void daeModel::InitializeStage1(void)
{
// Create domains, parameters, variables, ports etc
//	DeclareData();
//	DeclareDataBase();
}

void daeModel::InitializeStage2(void)
{
// Create model and port arrays (since I have initialized domains now)
// It also calls DeclareData() for each Model/Port in the arrays
	InitializePortAndModelArrays();

// Initialize variables' indexes
	InitializeVariables();
}

void daeModel::InitializeStage3(daeLog_t* pLog)
{
// Create DataProxy, initialize its arrays and propagate it to all child models
	if(m_nTotalNumberOfVariables == 0)
		daeDeclareAndThrowException(exInvalidCall);

	m_pDataProxy.reset(new daeDataProxy_t);
	m_pDataProxy->Initialize(this, pLog, m_nTotalNumberOfVariables);
	PropagateDataProxy(m_pDataProxy);

// Create equations
	BuildUpSTNsAndEquations();

// Now we have all elements created - its a good moment to check everything
	vector<string> strarrErrors;
	if(!CheckObject(strarrErrors))
	{
		daeDeclareException(exRuntimeCheck);
		for(vector<string>::iterator it = strarrErrors.begin(); it != strarrErrors.end(); it++)
			e << *it << "\n";
		throw e;
	}
	
// Now, after port connections have been checked, create port connection equations
	BuildUpPortConnectionEquations();

// Set default initial guesses and abs. tolerances
	SetDefaultInitialGuesses();
	SetDefaultAbsoluteTolerances();
}

void daeModel::InitializeStage4(void)
{
// Initialize equations
	m_pDataProxy->SetGatherInfo(true);
		InitializeEquations();
	m_pDataProxy->SetGatherInfo(false);
}

void daeModel::InitializeStage5(bool bDoBlockDecomposition, vector<daeBlock_t*>& ptrarrBlocks)
{
// Do block decomposition (if requested)
	DoBlockDecomposition(bDoBlockDecomposition, ptrarrBlocks);	
}

void daeModel::StoreInitializationValues(const std::string& strFileName) const
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_pDataProxy->Store(strFileName);
}

void daeModel::LoadInitializationValues(const std::string& strFileName) const
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_pDataProxy->Load(strFileName);
}

bool daeModel::IsModelDynamic() const
{
	if(!m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
		
	return m_pDataProxy->IsModelDynamic();		
}

//void daeModel::SetInitialConditions(real_t value)
//{
//	if(!m_pDataProxy)
//		daeDeclareAndThrowException(exInvalidPointer);
//
//	if(GetInitialConditionMode() != eQuasySteadyState)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "To set steady state conditions you should first set the InitialConditionMode to eQuasySteadyState";
//		throw e;
//	}
//
//	size_t n = m_pDataProxy->GetTotalNumberOfVariables();
//	for(size_t i = 0; i < n; i++)
//	{
//		if(m_pDataProxy->GetVariableTypeGathered(i) == cnDifferential)
//		{
//			m_pDataProxy->SetTimeDerivative(i, value);
//			m_pDataProxy->SetVariableType(i, cnDifferential);
//		}
//	}
//}

void daeModel::GetModelInfo(daeModelInfo& mi) const
{
	if(!m_pDataProxy)
		return;
		
	size_t i;
	
	mi.m_nNumberOfVariables             = GetTotalNumberOfVariables();
	mi.m_nNumberOfEquations             = GetTotalNumberOfEquations();
	mi.m_nNumberOfStateVariables	    = 0;
	mi.m_nNumberOfFixedVariables	    = 0;
	mi.m_nNumberOfDifferentialVariables	= 0;
	mi.m_nNumberOfInitialConditions	    = 0;
	
	for(i = 0; i < m_pDataProxy->GetTotalNumberOfVariables(); i++)
	{
		if(m_pDataProxy->GetVariableType(i) == cnNormal)
			mi.m_nNumberOfStateVariables++;
		else if(m_pDataProxy->GetVariableType(i) == cnDifferential)
			mi.m_nNumberOfInitialConditions++;
		else if(m_pDataProxy->GetVariableType(i) == cnFixed)
			mi.m_nNumberOfFixedVariables++;
	}
	
	for(i = 0; i < m_pDataProxy->GetTotalNumberOfVariables(); i++)
	{
		if(m_pDataProxy->GetVariableTypeGathered(i) == cnDifferential)
			mi.m_nNumberOfDifferentialVariables++;
	}
}

bool daeModel::CheckObject(vector<string>& strarrErrors) const
{
	size_t i;
	string strError;
	daeDomain* pDomain;
	daeParameter* pParameter;
	daeVariable* pVariable;
	daeEquation* pEquation;
	daePortConnection* pPortConnection;
	daeSTN* pSTN;
	daeModel* pModel;
	daeModelArray* pModelArray;
	daePort* pPort;
	daePortArray* pPortArray;

	bool bCheck = true;

// Check all domains
	for(i = 0; i < m_ptrarrDomains.size(); i++)
	{
		pDomain = m_ptrarrDomains[i];
		if(!pDomain)
		{
			strError = "Invalid domain in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pDomain->CheckObject(strarrErrors))
			bCheck = false;
	}

// Check all parameters
	for(i = 0; i < m_ptrarrParameters.size(); i++)
	{
		pParameter = m_ptrarrParameters[i];
		if(!pParameter)
		{
			strError = "Invalid parameter in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pParameter->CheckObject(strarrErrors))
			bCheck = false;
	}

// Check all variables
	for(i = 0; i < m_ptrarrVariables.size(); i++)
	{
		pVariable = m_ptrarrVariables[i];
		if(!pVariable)
		{
			strError = "Invalid variable in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pVariable->CheckObject(strarrErrors))
			bCheck = false;
	}

// Check all ports
	for(i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pPort = m_ptrarrPorts[i];
		if(!pPort)
		{
			strError = "Invalid port in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pPort->CheckObject(strarrErrors))
			bCheck = false;
	}

// Check all portarrays
	for(i = 0; i < m_ptrarrPortArrays.size(); i++)
	{
		pPortArray = m_ptrarrPortArrays[i];
		if(!pPortArray)
		{
			strError = "Invalid portarray in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pPortArray->CheckObject(strarrErrors))
			bCheck = false;
	}

// Check all equations
	for(i = 0; i < m_ptrarrEquations.size(); i++)
	{
		pEquation = m_ptrarrEquations[i];
		if(!pEquation)
		{
			strError = "Invalid equation in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pEquation->CheckObject(strarrErrors))
			bCheck = false;
	}
	
// Check all port connections
	for(i = 0; i < m_ptrarrPortConnections.size(); i++)
	{
		pPortConnection = m_ptrarrPortConnections[i];
		if(!pPortConnection)
		{
			strError = "Invalid port connection in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pPortConnection->CheckObject(strarrErrors))
			bCheck = false;
	}
	
// Check all STNs
	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
		{
			strError = "Invalid state transition network in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pSTN->CheckObject(strarrErrors))
			bCheck = false;
	}

// Check all child models
	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
		{
			strError = "Invalid child model in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pModel->CheckObject(strarrErrors))
			bCheck = false;
	}

// Check all modelarrays
	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		if(!pModelArray)
		{
			strError = "Invalid model array in the model: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pModelArray->CheckObject(strarrErrors))
			bCheck = false;
	}

	return bCheck;
}

size_t daeModel::GetTotalNumberOfVariables(void) const
{
	return m_nTotalNumberOfVariables;
}

size_t daeModel::GetTotalNumberOfEquations(void) const
{
	size_t i, nNoEqns;
	daeSTN* pSTN;
	daeEquation* pEquation;
	daePortConnection* pPortConnection;
	daeModel* pModel;
	daeModelArray* pModelArray;

	nNoEqns = 0;
	for(i = 0; i < m_ptrarrEquations.size(); i++)
	{
		pEquation = m_ptrarrEquations[i];
		if(!pEquation)
			daeDeclareAndThrowException(exInvalidPointer); 
		nNoEqns += pEquation->GetNumberOfEquations();
	}

	for(i = 0; i < m_ptrarrPortConnections.size(); i++)
	{
		pPortConnection = m_ptrarrPortConnections[i];
		if(!pPortConnection)
			daeDeclareAndThrowException(exInvalidPointer); 
		nNoEqns += pPortConnection->GetTotalNumberOfEquations();
	}

	for(i = 0; i < m_ptrarrSTNs.size(); i++)
	{
		pSTN = m_ptrarrSTNs[i];
		if(!pSTN)
			daeDeclareAndThrowException(exInvalidPointer); 
		nNoEqns += pSTN->GetNumberOfEquations();
	}

	for(i = 0; i < m_ptrarrModels.size(); i++)
	{
		pModel = m_ptrarrModels[i];
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer); 
		nNoEqns += pModel->GetTotalNumberOfEquations();
	}

	for(i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pModelArray = m_ptrarrModelArrays[i];
		if(!pModelArray)
			daeDeclareAndThrowException(exInvalidPointer); 
		nNoEqns += pModelArray->GetTotalNumberOfEquations();
	}

	return nNoEqns;
}

size_t daeModel::GetVariablesStartingIndex(void) const
{
	return m_nVariablesStartingIndex;
}

void daeModel::SetVariablesStartingIndex(size_t nVariablesStartingIndex)
{
	m_nVariablesStartingIndex = nVariablesStartingIndex;
}

void daeModel::SetModelAndCanonicalName(daeObject* pObject)
{
	if(!pObject)
		daeDeclareAndThrowException(exInvalidPointer);
	string strName;
	strName = m_strCanonicalName + "." + pObject->m_strShortName;
	pObject->m_strCanonicalName = strName;
	pObject->m_pModel = this;
}

void daeModel::GetSTNs(vector<daeSTN_t*>& ptrarrSTNs)
{
	ptrarrSTNs.clear();
	for(size_t i = 0; i < m_ptrarrSTNs.size(); i++)
		ptrarrSTNs.push_back(m_ptrarrSTNs[i]);
}

void daeModel::GetPorts(vector<daePort_t*>& ptrarrPorts)
{
	ptrarrPorts.clear();
	for(size_t i = 0; i < m_ptrarrPorts.size(); i++)
		ptrarrPorts.push_back(m_ptrarrPorts[i]);
}
	
void daeModel::GetEquations(vector<daeEquation_t*>& ptrarrEquations)
{
	ptrarrEquations.clear();
	for(size_t i = 0; i < m_ptrarrEquations.size(); i++)
		ptrarrEquations.push_back(m_ptrarrEquations[i]);
}

void daeModel::GetModels(vector<daeModel_t*>& ptrarrModels)
{
	ptrarrModels.clear();
	for(size_t i = 0; i < m_ptrarrModels.size(); i++)
		ptrarrModels.push_back(m_ptrarrModels[i]);
}
	
void daeModel::GetDomains(vector<daeDomain_t*>& ptrarrDomains)
{
	ptrarrDomains.clear();
	for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
		ptrarrDomains.push_back(m_ptrarrDomains[i]);
}
	
void daeModel::GetVariables(vector<daeVariable_t*>& ptrarrVariables)
{
	ptrarrVariables.clear();
	for(size_t i = 0; i < m_ptrarrVariables.size(); i++)
		ptrarrVariables.push_back(m_ptrarrVariables[i]);
}

void daeModel::GetPortConnections(vector<daePortConnection_t*>& ptrarrPortConnections)
{
	ptrarrPortConnections.clear();
	for(size_t i = 0; i < m_ptrarrPortConnections.size(); i++)
		ptrarrPortConnections.push_back(m_ptrarrPortConnections[i]);
}

void daeModel::GetParameters(vector<daeParameter_t*>& ptrarrParameters)
{
	ptrarrParameters.clear();
	for(size_t i = 0; i < m_ptrarrParameters.size(); i++)
		ptrarrParameters.push_back(m_ptrarrParameters[i]);
}

void daeModel::GetPortArrays(vector<daePortArray_t*>& ptrarrPortArrays)
{
	ptrarrPortArrays.clear();
	for(size_t i = 0; i < m_ptrarrPortArrays.size(); i++)
		ptrarrPortArrays.push_back(m_ptrarrPortArrays[i]);
}
	
void daeModel::GetModelArrays(vector<daeModelArray_t*>& ptrarrModelArrays)
{
	ptrarrModelArrays.clear();
	for(size_t i = 0; i < m_ptrarrModelArrays.size(); i++)
		ptrarrModelArrays.push_back(m_ptrarrModelArrays[i]);
}

void daeModel::GetEquationExecutionInfo(vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfos)
{
	ptrarrEquationExecutionInfos.clear();
	for(size_t i = 0; i < m_ptrarrEquationExecutionInfos.size(); i++)
		ptrarrEquationExecutionInfos.push_back(m_ptrarrEquationExecutionInfos[i]);
}

void daeModel::AddEquationExecutionInfo(daeEquationExecutionInfo* pEquationExecutionInfo)
{
	if(!pEquationExecutionInfo)
		daeDeclareAndThrowException(exInvalidPointer);

	m_ptrarrEquationExecutionInfos.push_back(pEquationExecutionInfo);
}

daeDomain_t* daeModel::FindDomain(string& strName)
{
	daeDomain* pObject;
	for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
	{
		pObject = m_ptrarrDomains[i];
		if(pObject->m_strShortName == strName)
			return pObject;
	}
	return NULL;
}

daeParameter_t* daeModel::FindParameter(string& strName)
{
	daeParameter* pObject;
	for(size_t i = 0; i < m_ptrarrParameters.size(); i++)
	{
		pObject = m_ptrarrParameters[i];
		if(pObject->m_strShortName == strName)
			return pObject;
	}
	return NULL;
}

daeVariable_t* daeModel::FindVariable(string& strName)
{
	daeVariable* pObject;
	for(size_t i = 0; i < m_ptrarrVariables.size(); i++)
	{
		pObject = m_ptrarrVariables[i];
		if(pObject->m_strShortName == strName)
			return pObject;
	}
	return NULL;
}

daePort_t* daeModel::FindPort(string& strName)
{
	daePort* pObject;
	for(size_t i = 0; i < m_ptrarrPorts.size(); i++)
	{
		pObject = m_ptrarrPorts[i];
		if(pObject->m_strShortName == strName)
			return pObject;
	}
	return NULL;
}

daeModel_t* daeModel::FindModel(string& strName)
{
	daeModel* pObject;
	for(size_t i = 0; i < m_ptrarrModels.size(); i++)
	{
		pObject = m_ptrarrModels[i];
		if(pObject->m_strShortName == strName)
			return pObject;
	}
	return NULL;
}

daePortArray_t* daeModel::FindPortArray(string& strName)
{
	daePortArray* pObject;
	for(size_t i = 0; i < m_ptrarrPortArrays.size(); i++)
	{
		pObject = m_ptrarrPortArrays[i];
		if(pObject->m_strShortName == strName)
			return pObject;
	}
	return NULL;
}

daeModelArray_t* daeModel::FindModelArray(string& strName)
{
	daeModelArray* pObject;
	for(size_t i = 0; i < m_ptrarrModelArrays.size(); i++)
	{
		pObject = m_ptrarrModelArrays[i];
		if(pObject->m_strShortName == strName)
			return pObject;
	}
	return NULL;
}

/*********************************************************************************************
	daeObjectType
**********************************************************************************************/
daeObjectType::daeObjectType()
{
	m_eObjectType = eObjectTypeUnknown;
	m_pObject	  = NULL;
}

daeObjectType::~daeObjectType(void)
{

}

}
}
