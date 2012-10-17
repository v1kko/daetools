#include "stdafx.h"
#include "coreimpl.h"
#include "event_handling.h"

namespace dae 
{
namespace core 
{
/******************************************************************
	daePort
*******************************************************************/
daePort::daePort()
{
	m_ePortType					= eUnknownPort;
	m_nVariablesStartingIndex	= ULONG_MAX;
	_currentVariablesIndex		= ULONG_MAX;
	m_ptrarrVariables.SetOwnershipOnPointers(false);
	m_ptrarrDomains.SetOwnershipOnPointers(false);
	m_ptrarrParameters.SetOwnershipOnPointers(false);
}
	
daePort::daePort(string strName, daeePortType portType, daeModel* parent, string strDescription)
{
	if(!parent)
		daeDeclareAndThrowException(exInvalidPointer);

	SetName(strName);
	m_ePortType					= portType;
	m_nVariablesStartingIndex	= ULONG_MAX;
	_currentVariablesIndex		= ULONG_MAX;
	m_ptrarrVariables.SetOwnershipOnPointers(false);
	m_ptrarrDomains.SetOwnershipOnPointers(false);
	m_ptrarrParameters.SetOwnershipOnPointers(false);

	parent->AddPort(*this, strName, portType, strDescription);
}

daePort::~daePort()
{
}

void daePort::Clone(const daePort& rObject)
{
	m_ePortType               = rObject.m_ePortType;
	m_nVariablesStartingIndex = ULONG_MAX;
	
	for(size_t i = 0; i < rObject.m_ptrarrDomains.size(); i++)
	{
		daeDomain* pDomain = new daeDomain(rObject.m_ptrarrDomains[i]->m_strShortName, 
										   this, 
										   rObject.m_ptrarrDomains[i]->m_strDescription);
		pDomain->Clone(*rObject.m_ptrarrDomains[i]);
	}
	
	for(size_t i = 0; i < rObject.m_ptrarrParameters.size(); i++)
	{
		daeParameter* pParameter = new daeParameter(rObject.m_ptrarrParameters[i]->m_strShortName, 
													rObject.m_ptrarrParameters[i]->m_Unit, 
													this, 
													rObject.m_ptrarrParameters[i]->m_strDescription);
		pParameter->Clone(*rObject.m_ptrarrParameters[i]);
	}
	
	for(size_t i = 0; i < rObject.m_ptrarrVariables.size(); i++)
	{
		daeVariable* pVariable = new daeVariable(rObject.m_ptrarrVariables[i]->m_strShortName, 
												 rObject.m_ptrarrVariables[i]->m_VariableType, 
												 this, 
												 rObject.m_ptrarrVariables[i]->m_strDescription);
		pVariable->Clone(*rObject.m_ptrarrVariables[i]);
	}
}

void daePort::CleanUpSetupData(void)
{
//	Variables are needed for data reporting
	clean_vector(m_ptrarrParameters);
}

void daePort::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeObject::Open(pTag);

	m_ptrarrDomains.EmptyAndFreeMemory();
	m_ptrarrParameters.EmptyAndFreeMemory();
	m_ptrarrVariables.EmptyAndFreeMemory();

	m_ptrarrDomains.SetOwnershipOnPointers(true);
	m_ptrarrParameters.SetOwnershipOnPointers(true);
	m_ptrarrVariables.SetOwnershipOnPointers(true);

	daeSetModelAndCanonicalNameDelegate<daeObject> del(this, m_pModel);

	strName = "PortType";
	OpenEnum(pTag, strName, m_ePortType);

	strName = "Domains";
	pTag->OpenObjectArray(strName, m_ptrarrDomains, &del);

	strName = "Parameters";
	pTag->OpenObjectArray(strName, m_ptrarrParameters, &del);

	strName = "Variables";
	pTag->OpenObjectArray(strName, m_ptrarrVariables, &del);
}

void daePort::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "PortType";
	SaveEnum(pTag, strName, m_ePortType);

	strName = "Domains";
	pTag->SaveObjectArray(strName, m_ptrarrDomains);

	strName = "Parameters";
	pTag->SaveObjectArray(strName, m_ptrarrParameters);

	strName = "Variables";
	pTag->SaveObjectArray(strName, m_ptrarrVariables);
}

void daePort::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
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

void daePort::CreateDefinition(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	boost::format fmtFile;
	string strComment, strFile, strCXXDeclaration, strConstructor;
	
	if(eLanguage == ePYDAE)
	{
		strComment = "#"; 
		
	/* Arguments:
	   1. Class name
	   4. Constructor contents (domains, params, variables)
	*/
		strFile  = 
		"class %1%(daePort):\n"
		"    def __init__(self, Name, PortType, Model, Description = \"\"):\n"
		"        daePort.__init__(self, Name, PortType, Model, Description)\n\n"
		"%2%\n";
		
		c.m_bExportDefinition = false;
		c.m_nPythonIndentLevel = 2;
		
		//strConstructor += c.CalculateIndent(c.m_nPythonIndentLevel) + strComment + " Domains \n";
		ExportObjectArray(m_ptrarrDomains,    strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrParameters, strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrVariables,  strConstructor, eLanguage, c);
	
		fmtFile.parse(strFile);
		fmtFile % GetObjectClassName() % strConstructor;
	}
	else if(eLanguage == eCDAE)
	{
		strComment   = "//"; 
		
		strFile  = 
		"class %1% : public daePort\n"
		"{\n"
		"daeDeclareDynamicClass(%1%)\n"
		"public:\n"
		"%2%\n"
		"    %1%(string strName, daeePortType portType, daeModel* parent, string strDescription = \"\")\n"
		"      : daePort(strName, portType, parent, strDescription)"
		"%3%\n"
		"    {\n"
		"    }\n"
		"};\n";
		
		c.m_bExportDefinition = true;
		c.m_nPythonIndentLevel = 1;
		
		ExportObjectArray(m_ptrarrDomains,    strCXXDeclaration, eLanguage, c);
		ExportObjectArray(m_ptrarrParameters, strCXXDeclaration, eLanguage, c);
		ExportObjectArray(m_ptrarrVariables,  strCXXDeclaration, eLanguage, c);

		c.m_bExportDefinition = false;
		c.m_nPythonIndentLevel = 2;

		ExportObjectArray(m_ptrarrDomains,    strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrParameters, strConstructor, eLanguage, c);
		ExportObjectArray(m_ptrarrVariables,  strConstructor, eLanguage, c);

		fmtFile.parse(strFile);
		fmtFile % GetObjectClassName() % strCXXDeclaration % strConstructor;
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented); 
	}
	
	strContent += fmtFile.str();
}

void daePort::DetectVariableTypesForExport(std::vector<const daeVariableType*>& ptrarrVariableTypes) const
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
}

void daePort::OpenRuntime(io::xmlTag_t* pTag)
{
//	string strName;

//	if(!m_pModel)
//		daeDeclareAndThrowException(exInvalidPointer);

//	daeObject::OpenRuntime(pTag);

//	m_ptrarrDomains.EmptyAndFreeMemory();
//	m_ptrarrParameters.EmptyAndFreeMemory();
//	m_ptrarrVariables.EmptyAndFreeMemory();

//	m_ptrarrDomains.SetOwnershipOnPointers(true);
//	m_ptrarrParameters.SetOwnershipOnPointers(true);
//	m_ptrarrVariables.SetOwnershipOnPointers(true);

//	daeSetModelAndCanonicalNameDelegate<daeObject> del(this, m_pModel);

//	strName = "PortType";
//	OpenEnum(pTag, strName, m_ePortType);

//	strName = "Domains";
//	OpenObjectArrayRuntime(pTag, strName, m_ptrarrDomains, &del);

//	strName = "Parameters";
//	OpenObjectArrayRuntime(pTag, strName, m_ptrarrParameters, &del);

//	strName = "Variables";
//	OpenObjectArrayRuntime(pTag, strName, m_ptrarrVariables, &del);
}

void daePort::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::SaveRuntime(pTag);

	strName = "PortType";
	SaveEnum(pTag, strName, m_ePortType);

	strName = "Domains";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrDomains);

	strName = "Parameters";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrParameters);

	strName = "Variables";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrVariables);
}

void daePort::InitializeVariables()
{
	size_t i;
	daeVariable* pVariable;

	_currentVariablesIndex = m_nVariablesStartingIndex;
	for(i = 0; i < m_ptrarrVariables.size(); i++)
	{
		pVariable = m_ptrarrVariables[i];
		pVariable->m_nOverallIndex = _currentVariablesIndex;
		_currentVariablesIndex += pVariable->GetNumberOfPoints();
	}
}

void daePort::InitializeParameters()
{
	size_t i;
	daeParameter* pParameter;

	for(i = 0; i < m_ptrarrParameters.size(); i++)
	{
		pParameter = m_ptrarrParameters[i];
		pParameter->Initialize();
	}
}

void daePort::SetReportingOn(bool bOn)
{
	size_t i;
	daeVariable* pVariable;
	daeParameter* pParameter;

// Set reporting on for all variables
	for(i = 0; i < m_ptrarrVariables.size(); i++)
	{
		pVariable = m_ptrarrVariables[i];
		pVariable->SetReportingOn(bOn);
	}
	
// Set reporting on for all parameters
	for(i = 0; i < m_ptrarrParameters.size(); i++)
	{
		pParameter = m_ptrarrParameters[i];
		pParameter->SetReportingOn(bOn);
	}
}

void daePort::AddDomain(daeDomain* pDomain)
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
		e << "Domain [" << strName << "] already exists in the port [" << GetCanonicalName() << "]";
		throw e;
	}

    SetModelAndCanonicalName(pDomain);
	dae_push_back(m_ptrarrDomains, pDomain);
}

void daePort::AddVariable(daeVariable* pVariable)
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
		e << "Variable [" << strName << "] already exists in the port [" << GetCanonicalName() << "]";
		throw e;
	}

    SetModelAndCanonicalName(pVariable);
	dae_push_back(m_ptrarrVariables, pVariable);
}

void daePort::AddParameter(daeParameter* pParameter)
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
		e << "Parameter [" << strName << "] already exists in the port [" << GetCanonicalName() << "]";
		throw e;
	}

    SetModelAndCanonicalName(pParameter);
	dae_push_back(m_ptrarrParameters, pParameter);
}

void daePort::AddDomain(daeDomain& rDomain, const string& strName, const unit& units, string strDescription)
{
	rDomain.SetUnits(units);
	rDomain.SetName(strName);
	rDomain.SetDescription(strDescription);
	AddDomain(&rDomain);
}

void daePort::AddVariable(daeVariable& rVariable, const string& strName, const daeVariableType& rVariableType, string strDescription)
{
	rVariable.SetName(strName);
	rVariable.SetDescription(strDescription);
	rVariable.SetVariableType(rVariableType);
	AddVariable(&rVariable);
}

void daePort::AddParameter(daeParameter& rParameter, const string& strName, const unit& units, string strDescription)
{
	rParameter.SetName(strName);
	rParameter.SetDescription(strDescription);
	rParameter.SetUnits(units);
	AddParameter(&rParameter);
}

size_t daePort::GetNumberOfVariables() const
{
	size_t i;
	daeVariable* pVariable;

	size_t nTotalNumberOfVariables = 0;
	for(i = 0; i < m_ptrarrVariables.size(); i++)
	{
		pVariable = m_ptrarrVariables[i];
		nTotalNumberOfVariables += pVariable->GetNumberOfPoints();
	}
	return nTotalNumberOfVariables;
}

bool daePort::FindObject(vector<string>& strarrHierarchy, daeObjectType& ObjectType)
{
	bool			bFound;
	daeeObjectType	eObjectType;
	string			strName;
	vector<size_t>	narrDomains;
	daeObject*		pObject;

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
	// If I call this function from daePort then the last item HAS to be var/param/domain
			if(eObjectType != eObjectTypeParameter &&
			   eObjectType != eObjectTypeDomain    &&
			   eObjectType != eObjectTypeVariable)
				return false;

			ObjectType.m_eObjectType = eObjectType;
			ObjectType.m_strName	 = strName;
			ObjectType.m_narrDomains = narrDomains;
			ObjectType.m_pObject     = pObject;
			return true;
		}

	// There are more than one item, so the item I have just detected must be some container (model, port, model- or port-array)
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

bool daePort::DetectObject(string& strShortName, vector<size_t>& /*narrDomains*/, daeeObjectType& eType, daeObject** ppObject)
{
	daeObject* pObject;

	try
	{
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

daeePortType daePort::GetType(void) const
{
	return m_ePortType;
}

void daePort::GetDomains(vector<daeDomain_t*>& ptrarrDomains)
{
	ptrarrDomains.clear();
	dae_set_vector(m_ptrarrDomains, ptrarrDomains);
}
	 
void daePort::GetVariables(vector<daeVariable_t*>& ptrarrVariables)
{
	ptrarrVariables.clear();
	dae_set_vector(m_ptrarrVariables, ptrarrVariables);
}
	
void daePort::GetParameters(vector<daeParameter_t*>& ptrarrParameters)
{
	ptrarrParameters.clear();
	dae_set_vector(m_ptrarrParameters, ptrarrParameters);
}

void daePort::SetType(daeePortType eType)
{
	m_ePortType = eType;
}

size_t daePort::GetVariablesStartingIndex() const
{
	return m_nVariablesStartingIndex;
}

void daePort::SetVariablesStartingIndex(size_t nVariablesStartingIndex)
{
	m_nVariablesStartingIndex = nVariablesStartingIndex;
}

daeObject_t* daePort::FindObjectFromRelativeName(string& strRelativeName)
{
// Parse string to get an array from the string 'object_1.object_2.[...].object_n'
	vector<string> strarrNames = ParseString(strRelativeName, '.');
	
	if(strarrNames.size() == 0)
	{
		daeDeclareAndThrowException(exInvalidCall);
	}
	else if(strarrNames.size() == 1)
	{
		return FindObject(strarrNames[0]);
	}
	else
	{
		return FindObjectFromRelativeName(strarrNames);
	}
	return NULL;	
}

daeObject_t* daePort::FindObjectFromRelativeName(vector<string>& strarrNames)
{
	if(strarrNames.size() == 1)
	{
		return FindObject(strarrNames[0]);
	}		
	else
	{
	// Get the first item and erase it from the vector
		vector<string>::iterator firstItem = strarrNames.begin();
		string strName = strarrNames[0];
		strarrNames.erase(firstItem);
		
	// Find the object with the name == strName
		daeObject_t* pObject = FindObject(strName);
		if(!pObject)
			daeDeclareAndThrowException(exInvalidPointer);
		
	// Call FindObjectFromRelativeName with the shortened the vector
		daePort_t*  pPort  = dynamic_cast<daePort_t*>(pObject);
		daeModel_t* pModel = dynamic_cast<daeModel_t*>(pObject);
		
		if(pModel)
			return pModel->FindObjectFromRelativeName(strarrNames);
		else if(pPort)
			return pPort->FindObjectFromRelativeName(strarrNames);
		else
			return NULL;
	}
}

daeObject_t* daePort::FindObject(string& strName)
{
	daeObject_t* pObject;
	
	pObject = FindParameter(strName);
	if(pObject)
		return pObject;
	
	pObject = FindDomain(strName);
	if(pObject)
		return pObject;
	
	pObject = FindVariable(strName);
	if(pObject)
		return pObject;

	return NULL;
}

daeDomain* daePort::FindDomain(unsigned long nID) const
{
	daeDomain* pDomain;
	for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
	{
		pDomain = m_ptrarrDomains[i];
		if(pDomain && pDomain->m_nID == nID)
			return pDomain;
	}
	return NULL;
}

daeVariable* daePort::FindVariable(unsigned long nID) const
{
	daeVariable* pVariable;
	for(size_t i = 0; i < m_ptrarrVariables.size(); i++)
	{
		pVariable = m_ptrarrVariables[i];
		if(pVariable && pVariable->m_nID == nID)
			return pVariable;
	}
	return NULL;
}

daeDomain* daePort::FindDomain(string& strName)
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

daeParameter* daePort::FindParameter(string& strName)
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

daeVariable* daePort::FindVariable(string& strName)
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

const std::vector<daeDomain*>& daePort::Domains() const
{
	return m_ptrarrDomains;
}

const std::vector<daeVariable*>& daePort::Variables() const
{
	return m_ptrarrVariables;
}

const std::vector<daeParameter*>& daePort::Parameters() const
{
	return m_ptrarrParameters;
}

void daePort::SetModelAndCanonicalName(daeObject* pObject)
{
	if(!pObject)
		daeDeclareAndThrowException(exInvalidPointer);
//	string strName;
//	strName = m_strCanonicalName + "." + pObject->m_strShortName;
//	pObject->m_strCanonicalName = strName;

	pObject->m_pModel = m_pModel;
}

bool daePort::CheckObject(vector<string>& strarrErrors) const
{
	size_t i;
	string strError;
	daeDomain* pDomain;
	daeParameter* pParameter;
	daeVariable* pVariable;

	bool bCheck = true;
	
	dae_capacity_check(m_ptrarrDomains);
	dae_capacity_check(m_ptrarrParameters);
	dae_capacity_check(m_ptrarrVariables);

// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check port type	
	if(m_ePortType == eUnknownPort)
	{
		strError = "Invalid port type in port [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check variable starting index	
	if(m_nVariablesStartingIndex == ULONG_MAX)
	{
		strError = "Invalid variables start index in port [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check all domains
	for(i = 0; i < m_ptrarrDomains.size(); i++)
	{
		pDomain = m_ptrarrDomains[i];
		if(!pDomain)
		{
			strError = "Invalid domain in port: [" + GetCanonicalName() + "]";
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
			strError = "Invalid parameter in port: [" + GetCanonicalName() + "]";
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
			strError = "Invalid variable in port: [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(!pVariable->CheckObject(strarrErrors))
			bCheck = false;
	}

	return bCheck;
}

/******************************************************************
	daePortConnection
*******************************************************************/
daePortConnection::daePortConnection()
{
	m_pPortFrom = NULL;
	m_pPortTo   = NULL;
}

daePortConnection::daePortConnection(daePort* pPortFrom, daePort* pPortTo)
{
	if(!pPortFrom)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!pPortTo)
		daeDeclareAndThrowException(exInvalidPointer); 
	
	m_pPortFrom = pPortFrom;
	m_pPortTo   = pPortTo;
}

daePortConnection::~daePortConnection()
{
}

void daePortConnection::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeObject::Open(pTag);

	daeFindPortByID del(m_pModel);

	strName = "PortFrom";
	m_pPortFrom = pTag->OpenObjectRef(strName, &del);
	if(!m_pPortFrom)
		daeDeclareAndThrowException(exXMLIOError); 

	strName = "PortTo";
	m_pPortTo = pTag->OpenObjectRef(strName, &del);
	if(!m_pPortTo)
		daeDeclareAndThrowException(exXMLIOError); 
}

void daePortConnection::Save(io::xmlTag_t* pTag) const
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pPortFrom)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pPortTo)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeObject::Save(pTag);

	strName = "PortFrom";
	pTag->SaveObjectRef(strName, m_pPortFrom, m_pModel);

	strName = "PortTo";
	pTag->SaveObjectRef(strName, m_pPortTo, m_pModel);
}

void daePortConnection::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport;
	boost::format fmtFile;

	if(c.m_bExportDefinition)
	{
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.ConnectPorts(self.%1%, self.%2%)\n";
			fmtFile.parse(strExport);
			fmtFile % daeGetStrippedRelativeName(m_pModel, m_pPortFrom) 
					% daeGetStrippedRelativeName(m_pModel, m_pPortTo);
		}
		else if(eLanguage == eCDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "ConnectPorts(&%1%, &%2%);\n";
			fmtFile.parse(strExport);
			fmtFile % daeGetStrippedRelativeName(m_pModel, m_pPortFrom) 
					% daeGetStrippedRelativeName(m_pModel, m_pPortTo);
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += fmtFile.str();
}

void daePortConnection::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daePortConnection::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pPortFrom)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pPortTo)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeObject::SaveRuntime(pTag);

	strName = "PortFrom";
	pTag->SaveObjectRef(strName, m_pPortFrom, m_pModel);

	strName = "PortTo";
	pTag->SaveObjectRef(strName, m_pPortTo, m_pModel);
}

daePort_t* daePortConnection::GetPortFrom(void) const
{
	return m_pPortFrom;
}
	
daePort_t* daePortConnection::GetPortTo(void) const
{
	return m_pPortTo;
}

size_t daePortConnection::GetTotalNumberOfEquations(void) const
{
	daeEquation* pEquation;

	size_t nNoEqns = 0;
	for(size_t i = 0; i < m_ptrarrEquations.size(); i++)
	{
		pEquation = m_ptrarrEquations[i];
		if(!pEquation)
			daeDeclareAndThrowException(exInvalidPointer); 
		nNoEqns += pEquation->GetNumberOfEquations();
	}
	return nNoEqns;
}

void daePortConnection::CreateEquations(void)
{
	string strName;
	size_t j, k, iNoVars, nNoDomains;
	daeDomain *pDomain;
	daeVariable *pVarFrom, *pVarTo;
	daePortEqualityEquation* pEquation;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	
// Check if there is equal number of variables in each port
	iNoVars = m_pPortFrom->m_ptrarrVariables.size();

	for(j = 0; j < iNoVars; j++)
	{
		pVarFrom = m_pPortFrom->m_ptrarrVariables[j];
		pVarTo   = m_pPortTo->m_ptrarrVariables[j];

		nNoDomains = pVarFrom->m_ptrDomains.size();

		pEquation = new daePortEqualityEquation();
		pEquation->Initialize(pVarFrom, pVarTo);
		SetModelAndCanonicalName(pEquation);
		//m_ptrarrEquations.push_back(pEquation);
		dae_push_back(m_ptrarrEquations, pEquation);

		strName  = pVarFrom->GetName();
		strName += "_";
		strName += pVarTo->GetName();
		pEquation->SetName(strName);

		daeDEDI* pDEDI;
		vector<daeDEDI*> arrDEDIs;
		for(k = 0; k < nNoDomains; k++)
		{
			pDomain = pVarFrom->m_ptrDomains[k];
			pDEDI = pEquation->DistributeOnDomain(*pDomain, eClosedClosed);
			arrDEDIs.push_back(pDEDI);
		}

		if(nNoDomains == 0)
			pEquation->SetResidual((*pVarFrom)() - (*pVarTo)());
		else if(nNoDomains == 1)
			pEquation->SetResidual((*pVarFrom)(arrDEDIs[0])                                                     
								 - (*pVarTo)  (arrDEDIs[0]));
		else if(nNoDomains == 2)
			pEquation->SetResidual((*pVarFrom)(arrDEDIs[0], arrDEDIs[1])                
								 - (*pVarTo)  (arrDEDIs[0], arrDEDIs[1]));
		else if(nNoDomains == 3)
			pEquation->SetResidual((*pVarFrom)(arrDEDIs[0], arrDEDIs[1], arrDEDIs[2])    
								 - (*pVarTo)  (arrDEDIs[0], arrDEDIs[1], arrDEDIs[2]));
		else if(nNoDomains == 4)
			pEquation->SetResidual((*pVarFrom)(arrDEDIs[0], arrDEDIs[1], arrDEDIs[2], arrDEDIs[3])              
								 - (*pVarTo)  (arrDEDIs[0], arrDEDIs[1], arrDEDIs[2], arrDEDIs[3]));
		else if(nNoDomains == 5)
			pEquation->SetResidual((*pVarFrom)(arrDEDIs[0], arrDEDIs[1], arrDEDIs[2], arrDEDIs[3], arrDEDIs[4])
								 - (*pVarTo)  (arrDEDIs[0], arrDEDIs[1], arrDEDIs[2], arrDEDIs[3], arrDEDIs[4]));
		else 
			daeDeclareAndThrowException(exNotImplemented); 
	}
}

void daePortConnection::GetEquations(vector<daeEquation*>& ptrarrEquations) const
{
	dae_set_vector(m_ptrarrEquations, ptrarrEquations);
}

void daePortConnection::SetModelAndCanonicalName(daeObject* pObject)
{
	if(!pObject)
		daeDeclareAndThrowException(exInvalidPointer);
//	string strName;
//	strName = m_strCanonicalName + "." + pObject->GetName();
//	pObject->SetCanonicalName(strName);

	pObject->m_pModel = m_pModel;
}

bool daePortConnection::CheckObject(vector<string>& strarrErrors) const
{
	string strError;
	daeDomain *pDomainFrom, *pDomainTo;
	daeVariable *pVarFrom, *pVarTo;
	size_t j, k, nNoDomains, iNoVarsFrom, iNoVarsTo;

	bool bCheck = true;
	
// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check left port pointer	
	if(!m_pPortFrom)
	{
		strError = "Invalid from-port in port connection [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}

// Check right port pointer	
	if(!m_pPortTo)
	{
		strError = "Invalid to-port in port connection [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}

// Check if there is equal number of variables in each port
	iNoVarsFrom = m_pPortFrom->m_ptrarrVariables.size();
	iNoVarsTo   = m_pPortTo->m_ptrarrVariables.size();
	if(iNoVarsFrom != iNoVarsTo)
	{
		strError = "Number of variables in port [ " + m_pPortFrom->GetCanonicalName() 
		  + "] and [" + m_pPortTo->GetCanonicalName() + "] must be equal";
		strarrErrors.push_back(strError);
		return false;
	}

	for(j = 0; j < iNoVarsFrom; j++)
	{
		pVarFrom = m_pPortFrom->m_ptrarrVariables[j];
		if(!pVarFrom)
		{
			strError = "Invalid left variable in port connection [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		
		pVarTo = m_pPortTo->m_ptrarrVariables[j];
		if(!pVarTo)
		{
			strError = "Invalid right variable in port connection [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}

	// Check if there is equal number of domains in each variable
		if(pVarFrom->m_ptrDomains.size() != pVarTo->m_ptrDomains.size())
		{
			strError = "Number of domains in variable [ " + pVarFrom->GetCanonicalName() +
			           "] and variable [" + pVarTo->GetCanonicalName() + "] is not equal" +
					   " in port connection [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}

		if(pVarFrom->m_ptrDomains.size() > 5)
		{
			strError = "Maximal number of domains must be 5 in variable [ " + pVarFrom->GetCanonicalName() +
					   " in port connection [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}
		if(pVarTo->m_ptrDomains.size() > 5)
		{
			strError = "Maximal number of domains must be 5 in variable [ " + pVarTo->GetCanonicalName() +
					   " in port connection [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
			continue;
		}

		nNoDomains = pVarFrom->m_ptrDomains.size();

	// Check if there is equal number of points in each domain
		for(k = 0; k < nNoDomains; k++)
		{
			pDomainFrom = pVarFrom->m_ptrDomains[k];
			if(!pDomainFrom)
			{
				strError = "Invalid domain in port connection [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
			
			pDomainTo = pVarTo->m_ptrDomains[k];
			if(!pDomainTo)
			{
				strError = "Invalid domain in port connection [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
			
			if(pDomainFrom->GetNumberOfPoints() != pDomainTo->GetNumberOfPoints())
			{
				strError = "Number of points in domains in variable [ " + pVarFrom->GetCanonicalName() +
				           "] and variable [" + pVarTo->GetCanonicalName() + "] is not equal" + 
	   				       " in port connection [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
			}
		}
	}
		
	return bCheck;
}

/******************************************************************
	daeEventPortConnection
*******************************************************************/
daeEventPortConnection::daeEventPortConnection() //: sender(&receiver)
{
	m_pPortFrom = NULL;
	m_pPortTo   = NULL;
}

daeEventPortConnection::daeEventPortConnection(daeEventPort* pPortFrom, daeEventPort* pPortTo) //: sender(&receiver)
{
	if(!pPortFrom)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!pPortTo)
		daeDeclareAndThrowException(exInvalidPointer); 
	
	m_pPortFrom = pPortFrom;
	m_pPortTo   = pPortTo;
	
	pPortTo->Attach(pPortFrom);
/*
	receiver.reset(new daeRemoteEventReceiver());
	sender.reset(new daeRemoteEventSender());
	pPortTo->Attach(sender.get());
	receiver->Attach(pPortFrom);
*/
}

daeEventPortConnection::~daeEventPortConnection()
{
}

void daeEventPortConnection::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeObject::Open(pTag);

	daeFindEventPortByID del(m_pModel);

	strName = "PortFrom";
	m_pPortFrom = pTag->OpenObjectRef(strName, &del);
	if(!m_pPortFrom)
		daeDeclareAndThrowException(exXMLIOError); 

	strName = "PortTo";
	m_pPortTo = pTag->OpenObjectRef(strName, &del);
	if(!m_pPortTo)
		daeDeclareAndThrowException(exXMLIOError); 
}

void daeEventPortConnection::Save(io::xmlTag_t* pTag) const
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pPortFrom)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pPortTo)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeObject::Save(pTag);

	strName = "PortFrom";
	pTag->SaveObjectRef(strName, m_pPortFrom, m_pModel);

	strName = "PortTo";
	pTag->SaveObjectRef(strName, m_pPortTo, m_pModel);
}

void daeEventPortConnection::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport;
	boost::format fmtFile;

	if(c.m_bExportDefinition)
	{
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.ConnectEventPorts(self.%1%, self.%2%)\n";
			fmtFile.parse(strExport);
			fmtFile % daeGetStrippedRelativeName(m_pModel, m_pPortFrom) 
					% daeGetStrippedRelativeName(m_pModel, m_pPortTo);
		}
		else if(eLanguage == eCDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "ConnectEventPorts(&%1%, &%2%);\n";
			fmtFile.parse(strExport);
			fmtFile % daeGetStrippedRelativeName(m_pModel, m_pPortFrom) 
					% daeGetStrippedRelativeName(m_pModel, m_pPortTo);
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += fmtFile.str();
}

void daeEventPortConnection::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daeEventPortConnection::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pPortFrom)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pPortTo)
		daeDeclareAndThrowException(exInvalidPointer);

	daeObject::SaveRuntime(pTag);

	strName = "PortFrom";
	pTag->SaveObjectRef(strName, m_pPortFrom, m_pModel);

	strName = "PortTo";
	pTag->SaveObjectRef(strName, m_pPortTo, m_pModel);
}

daeEventPort_t* daeEventPortConnection::GetPortFrom(void) const
{
	return m_pPortFrom;
}
	
daeEventPort_t* daeEventPortConnection::GetPortTo(void) const
{
	return m_pPortFrom;
}

void daeEventPortConnection::SetModelAndCanonicalName(daeObject* pObject)
{
	if(!pObject)
		daeDeclareAndThrowException(exInvalidPointer);
//	string strName;
//	strName = m_strCanonicalName + "." + pObject->GetName();
//	pObject->SetCanonicalName(strName);

	pObject->m_pModel = m_pModel;
}

bool daeEventPortConnection::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;
	
// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check left port pointer	
	if(!m_pPortFrom)
	{
		strError = "Invalid from-port in event port connection [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}

// Check right port pointer	
	if(!m_pPortTo)
	{
		strError = "Invalid to-port in event port connection [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}

	return bCheck;
}


}
}
