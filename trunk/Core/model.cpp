#include "stdafx.h"
#include "coreimpl.h"
#include <typeinfo>
#include "nodes.h"

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

namespace dae
{
namespace core
{
daeModel::daeModel()
{
    m_pExecutionContextForGatherInfo	= NULL;
    m_nVariablesStartingIndex			= 0;
    m_nTotalNumberOfVariables			= 0;

// When used programmatically they dont own pointers !!!!!
    m_ptrarrDomains.SetOwnershipOnPointers(false);
    m_ptrarrPorts.SetOwnershipOnPointers(false);
    m_ptrarrEventPorts.SetOwnershipOnPointers(false);
    m_ptrarrVariables.SetOwnershipOnPointers(false);
    m_ptrarrParameters.SetOwnershipOnPointers(false);
    m_ptrarrComponents.SetOwnershipOnPointers(false);
    m_ptrarrPortArrays.SetOwnershipOnPointers(false);
    m_ptrarrComponentArrays.SetOwnershipOnPointers(false);
    m_ptrarrExternalFunctions.SetOwnershipOnPointers(false);
}

daeModel::daeModel(string strName, daeModel* pModel, string strDescription)
{
    m_pExecutionContextForGatherInfo	= NULL;
    m_nVariablesStartingIndex			= 0;
    m_nTotalNumberOfVariables			= 0;

// When used programmatically they dont own pointers !!!!!
    m_ptrarrDomains.SetOwnershipOnPointers(false);
    m_ptrarrPorts.SetOwnershipOnPointers(false);
    m_ptrarrEventPorts.SetOwnershipOnPointers(false);
    m_ptrarrVariables.SetOwnershipOnPointers(false);
    m_ptrarrParameters.SetOwnershipOnPointers(false);
    m_ptrarrComponents.SetOwnershipOnPointers(false);
    m_ptrarrPortArrays.SetOwnershipOnPointers(false);
    m_ptrarrComponentArrays.SetOwnershipOnPointers(false);
    m_ptrarrExternalFunctions.SetOwnershipOnPointers(false);

    SetName(strName);
    SetDescription(strDescription);
    if(pModel)
        pModel->AddModel(this);
}

daeModel::~daeModel()
{
}

void daeModel::Clone(const daeModel& rObject)
{
    for(size_t i = 0; i < rObject.m_ptrarrDomains.size(); i++)
    {
        daeDomain* pDomain = new daeDomain(rObject.m_ptrarrDomains[i]->m_strShortName,
                                           this,
                                           rObject.m_ptrarrDomains[i]->m_strDescription);
        pDomain->Clone(*rObject.m_ptrarrDomains[i]);
    }

    for(size_t i = 0; i < rObject.m_ptrarrComponents.size(); i++)
    {
        daeModel* pModel = new daeModel(rObject.m_ptrarrComponents[i]->m_strShortName,
                                        this,
                                        rObject.m_ptrarrComponents[i]->m_strDescription);
        pModel->Clone(*rObject.m_ptrarrComponents[i]);
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

    for(size_t i = 0; i < rObject.m_ptrarrPorts.size(); i++)
    {
        daePort* pPort = new daePort(rObject.m_ptrarrPorts[i]->m_strShortName,
                                     rObject.m_ptrarrPorts[i]->GetType(),
                                     this,
                                     rObject.m_ptrarrPorts[i]->m_strDescription);
        pPort->Clone(*rObject.m_ptrarrPorts[i]);
    }

    for(size_t i = 0; i < rObject.m_ptrarrEventPorts.size(); i++)
    {
        daeEventPort* pEventPort = new daeEventPort(rObject.m_ptrarrEventPorts[i]->m_strShortName,
                                                    rObject.m_ptrarrEventPorts[i]->GetType(),
                                                    this,
                                                    rObject.m_ptrarrEventPorts[i]->m_strDescription);
        pEventPort->Clone(*rObject.m_ptrarrEventPorts[i]);
    }

    for(size_t i = 0; i < rObject.m_ptrarrEquations.size(); i++)
    {
        daeEquation* pEquation = CreateEquation(rObject.m_ptrarrEquations[i]->m_strShortName,
                                                rObject.m_ptrarrEquations[i]->m_strDescription,
                                                rObject.m_ptrarrEquations[i]->m_dScaling);
        pEquation->Clone(*rObject.m_ptrarrEquations[i]);
    }

    for(size_t i = 0; i < rObject.m_ptrarrSTNs.size(); i++)
    {
        daeSTN* pSTN = AddSTN(rObject.m_ptrarrSTNs[i]->m_strShortName);
        pSTN->Clone(*rObject.m_ptrarrSTNs[i]);
    }
}

void daeModel::UpdateEquations(const daeExecutionContext* pExecutionContext)
{
// Default implementation does nothing (just propagates call to child models and model arrays)
    size_t i;

// Then, create port connection equations for each child-model
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        daeModel* pModel = m_ptrarrComponents[i];
        pModel->UpdateEquations(pExecutionContext);
    }

// Finally, create port connection equations for each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        daeModelArray* pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->UpdateEquations(pExecutionContext);
    }
}

void daeModel::CleanUpSetupData()
{
/*
    Domains, Parameters, Variables, Ports, EventPorts, Models and ExternalFunctions
    are created and owned by the user; here we just hold references on these objects.
    Equations, PortConnections and EventPortConnections are internally owned by models.

    In general, we may not clean the following vectors:
     - Domains (some nodes hold pointers to their data-points)
     - Variables (needed for data reporting)
     - Models
     - Ports
     - STNS/IFs/States/StateTransitions
*/
    //std::cout << "daeModel::CleanUpSetupData" << std::endl;

    clean_vector(m_ptrarrDomains);
    clean_vector(m_ptrarrParameters);
    clean_vector(m_ptrarrEventPorts);
    clean_vector(m_ptrarrExternalFunctions);

    clean_vector(m_ptrarrEquations);
    clean_vector(m_ptrarrPortConnections);
    clean_vector(m_ptrarrEventPortConnections);

    for(size_t i = 0; i < m_ptrarrComponents.size(); i++)
        m_ptrarrComponents[i]->CleanUpSetupData();

    for(size_t i = 0; i < m_ptrarrComponentArrays.size(); i++)
        m_ptrarrComponentArrays[i]->CleanUpSetupData();

    for(size_t i = 0; i < m_ptrarrPorts.size(); i++)
        m_ptrarrPorts[i]->CleanUpSetupData();
    clean_vector(m_ptrarrPortArrays);
}

void daeModel::Open(io::xmlTag_t* pTag)
{
    string strName;

    m_ptrarrComponents.EmptyAndFreeMemory();
    m_ptrarrEquations.EmptyAndFreeMemory();
    m_ptrarrSTNs.EmptyAndFreeMemory();
    m_ptrarrPortConnections.EmptyAndFreeMemory();
    m_ptrarrEventPortConnections.EmptyAndFreeMemory();
    m_ptrarrDomains.EmptyAndFreeMemory();
    m_ptrarrParameters.EmptyAndFreeMemory();
    m_ptrarrVariables.EmptyAndFreeMemory();
    m_ptrarrPorts.EmptyAndFreeMemory();
    m_ptrarrEventPorts.EmptyAndFreeMemory();
    m_ptrarrOnEventActions.EmptyAndFreeMemory();
    m_ptrarrOnConditionActions.EmptyAndFreeMemory();
    m_ptrarrPortArrays.EmptyAndFreeMemory();
    m_ptrarrComponentArrays.EmptyAndFreeMemory();

    m_ptrarrComponents.SetOwnershipOnPointers(true);
    m_ptrarrEquations.SetOwnershipOnPointers(true);
    m_ptrarrSTNs.SetOwnershipOnPointers(true);
    m_ptrarrPortConnections.SetOwnershipOnPointers(true);
    m_ptrarrEventPortConnections.SetOwnershipOnPointers(true);
    m_ptrarrDomains.SetOwnershipOnPointers(true);
    m_ptrarrParameters.SetOwnershipOnPointers(true);
    m_ptrarrVariables.SetOwnershipOnPointers(true);
    m_ptrarrPorts.SetOwnershipOnPointers(true);
    m_ptrarrEventPorts.SetOwnershipOnPointers(true);
    m_ptrarrOnEventActions.SetOwnershipOnPointers(true);
    m_ptrarrOnConditionActions.SetOwnershipOnPointers(true);
    m_ptrarrPortArrays.SetOwnershipOnPointers(true);
    m_ptrarrComponentArrays.SetOwnershipOnPointers(true);

    m_pDataProxy.reset();
    m_nVariablesStartingIndex	= 0;
    m_ptrarrEquationExecutionInfos.clear();

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

// EVENT PORTS
    strName = "EventPorts";
    pTag->OpenObjectArray(strName, m_ptrarrEventPorts, &del);

// EQUATIONS
    strName = "Equations";
    pTag->OpenObjectArray(strName, m_ptrarrEquations, &del);

// STNs
    strName = "STNs";
    pTag->OpenObjectArray(strName, m_ptrarrSTNs, &del);

// PORT CONNECTIONs
    strName = "PortConnections";
    pTag->OpenObjectArray(strName, m_ptrarrPortConnections, &del);

// EVENT PORT CONNECTIONs
    strName = "EventPortConnections";
    pTag->OpenObjectArray(strName, m_ptrarrEventPortConnections, &del);

// CHILD MODELS
    strName = "Units";
    pTag->OpenObjectArray(strName, m_ptrarrComponents, &del);

// MODEL ARRAYS
    //strName = "UnitArrays";
    //pTag->OpenObjectArray(strName, m_ptrarrComponentArrays, &del);

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

// EVENT PORTS
    strName = "EventPorts";
    pTag->SaveObjectArray(strName, m_ptrarrEventPorts);

// EQUATIONS
    strName = "Equations";
    pTag->SaveObjectArray(strName, m_ptrarrEquations);

// ON EVENT ACTIONS
    strName = "OnEventActions";
    pTag->SaveObjectArray(strName, m_ptrarrOnEventActions);

// ON CONDITION ACTIONS
    strName = "OnConditionActions";
    pTag->SaveObjectArray(strName, m_ptrarrOnConditionActions);

// STNs
    strName = "STNs";
    pTag->SaveObjectArray(strName, m_ptrarrSTNs);

// PORTCONNECTIONs
    strName = "PortConnections";
    pTag->SaveObjectArray(strName, m_ptrarrPortConnections);

// EVENT PORT CONNECTIONs
    strName = "EventPortConnections";
    pTag->SaveObjectArray(strName, m_ptrarrEventPortConnections);

// CHILD MODELS
    strName = "Units";
    pTag->SaveObjectArray(strName, m_ptrarrComponents);

// MODELARRAYS
    strName = "UnitArrays";
    pTag->SaveObjectArray(strName, m_ptrarrComponentArrays);

// PORTARRAYS
    strName = "PortArrays";
    pTag->SaveObjectArray(strName, m_ptrarrPortArrays);
}

void daeModel::OpenRuntime(io::xmlTag_t* pTag)
{
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

// EVENT PORTS
    strName = "EventPorts";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrEventPorts);

// EQUATIONS
    strName = "Equations";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrEquations);

// ON EVENT ACTIONS
    strName = "OnEventActions";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrOnEventActions);

// ON CONDITION ACTIONS
    strName = "OnConditionActions";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrOnConditionActions);

// STNs
    strName = "STNs";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrSTNs);

// PORTCONNECTIONs
    strName = "PortConnections";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrPortConnections);

// EVENT PORT CONNECTIONs
    strName = "EventPortConnections";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrEventPortConnections);

// CHILD MODELS
    strName = "Units";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrComponents);

// MODELARRAYS
    strName = "UnitArrays";
    pTag->SaveRuntimeObjectArray(strName, m_ptrarrComponentArrays);

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
string daeModel::ExportObjects(std::vector<daeExportable_t*>& ptrarrObjects, daeeModelLanguage eLanguage) const
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
        "                 Copyright (C) Dragan Nikolic, 2015\n"
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
        "                 Copyright (C) Dragan Nikolic, 2015\n"
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
        "#include \"variable_types.h\"\n\n"
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
    string strFile, strCXXDeclaration, strConstructor, strDeclareEquations;

    c.m_pModel = this;

    if(eLanguage == ePYDAE)
    {
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
        ExportObjectArray(m_ptrarrEventPorts,      strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrComponents,      strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrComponentArrays, strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrPortArrays,      strConstructor, eLanguage, c);

        ExportObjectArray(m_ptrarrEquations,			strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrSTNs,					strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrPortConnections,		strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrEventPortConnections,	strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrOnEventActions,		strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrOnConditionActions,	strDeclareEquations, eLanguage, c);

        fmtFile.parse(strFile);
        fmtFile % GetObjectClassName() % strConstructor % (strDeclareEquations.empty() ? (c.CalculateIndent(c.m_nPythonIndentLevel) + "pass\n") : strDeclareEquations);
    }
    else if(eLanguage == eCDAE)
    {
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
        ExportObjectArray(m_ptrarrEventPorts,  strCXXDeclaration, eLanguage, c);
        ExportObjectArray(m_ptrarrComponents,      strCXXDeclaration, eLanguage, c);
        ExportObjectArray(m_ptrarrComponentArrays, strCXXDeclaration, eLanguage, c);
        ExportObjectArray(m_ptrarrPortArrays,  strCXXDeclaration, eLanguage, c);


        c.m_bExportDefinition = false;
        c.m_nPythonIndentLevel = 2;

        ExportObjectArray(m_ptrarrDomains,     strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrParameters,  strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrVariables,   strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrPorts,       strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrEventPorts,  strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrComponents,      strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrComponentArrays, strConstructor, eLanguage, c);
        ExportObjectArray(m_ptrarrPortArrays,  strConstructor, eLanguage, c);

        c.m_bExportDefinition = false;
        c.m_nPythonIndentLevel = 2;

        ExportObjectArray(m_ptrarrEquations,			strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrSTNs,					strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrPortConnections,		strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrEventPortConnections,	strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrOnEventActions,		strDeclareEquations, eLanguage, c);
        ExportObjectArray(m_ptrarrOnConditionActions,	strDeclareEquations, eLanguage, c);

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

    for(i = 0; i < m_ptrarrComponents.size(); i++)
        m_ptrarrComponents[i]->DetectVariableTypesForExport(ptrarrVariableTypes);

    for(i = 0; i < m_ptrarrPortArrays.size(); i++)
        m_ptrarrPortArrays[i]->DetectVariableTypesForExport(ptrarrVariableTypes);

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
        m_ptrarrComponentArrays[i]->DetectVariableTypesForExport(ptrarrVariableTypes);
}

/*
//#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
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
*/

template<class CLASS>
bool compareCanonicalNames(CLASS first, CLASS second)
{
    return (first->GetCanonicalName() < second->GetCanonicalName());
}

void daeModel::GetCoSimulationInterface(std::vector<daeParameter_t*>& ptrarrParameters,
                                        std::vector<daeVariable_t*>&  ptrarrInputs,
                                        std::vector<daeVariable_t*>&  ptrarrOutputs,
                                        std::vector<daeSTN_t*>&       ptrarrSTNs)
{
    // Collect all parameters and ports and initialize parameters/inputs/outputs arrays
    std::vector<daeVariable_t*> ptrarrVariables;
    std::map<std::string, daeParameter_t*> mapParameters;
    std::map<std::string, daeSTN_t*> mapSTNs;
    std::vector<daePort_t*> arrPorts;

    // All parameters
    CollectAllParameters(mapParameters);
    // All STNs
    CollectAllSTNs(mapSTNs);
    // Only ports in the top-level model
    GetPorts(arrPorts);

    for(std::map<std::string, daeParameter_t*>::iterator iter = mapParameters.begin(); iter != mapParameters.end(); iter++)
        ptrarrParameters.push_back(iter->second);

    for(std::map<std::string, daeSTN_t*>::iterator iter = mapSTNs.begin(); iter != mapSTNs.end(); iter++)
        ptrarrSTNs.push_back(iter->second);

    // Only ports from the top-level model (not internal models!)
    for(size_t i = 0; i < arrPorts.size(); i++)
    {
        daePort_t* port = arrPorts[i];

        ptrarrVariables.clear();
        port->GetVariables(ptrarrVariables);

        if(port->GetType() == eInletPort)
        {
            for(size_t i = 0; i < ptrarrVariables.size(); i++)
            {
                daeVariable_t* pVariable = ptrarrVariables[i];

            // Achtung, Achtung!!
            // To enable external tools to change values of input ports, port variables must be assigned (that is be a DOF).
            // Throw an exception if that is NOT the case!
                if(pVariable->GetType() != cnAssigned)
                {
                    daeDeclareException(exInvalidCall);
                    e << "Inlet port variables [" << pVariable->GetCanonicalName() << "] must have assigned values (must be DOFs)";
                    throw e;
                }

                ptrarrInputs.push_back(pVariable);
            }
        }
        else if(port->GetType() == eOutletPort)
        {
            for(size_t i = 0; i < ptrarrVariables.size(); i++)
            {
                daeVariable_t* pVariable = ptrarrVariables[i];
                if(pVariable->GetType() == cnAssigned || pVariable->GetType() == cnSomePointsAssigned)
                {
                    daeDeclareException(exInvalidCall);
                    e << "Outlet port variables [" << pVariable->GetCanonicalName() << "] cannot have assigned values (can't be DOFs)";
                    throw e;
                }

                ptrarrOutputs.push_back(pVariable);
            }
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Ports can be either inlet or outlet; inlet/outlet ports are not supported [" << port->GetCanonicalName() << "]";
            throw e;
        }
    }

    // DOFs
    // Nota bene:
    //   Here we have a problem: inlet ports are marked as DOFs, so there is an overlap between the DOF and Input variables)
    /*
    for(std::map<std::string, daeVariable_t*>::iterator iter = mapVariables.begin(); iter != mapVariables.end(); iter++)
    {
        daeVariable_t* pVariable = iter->second;
        if(pVariable->GetType() == cnAssigned)
        {
            ptrarrDOFs.push_back(pVariable);
        }
        else if(pVariable->GetType() == cnSomePointsAssigned)
        {
            std::cout << "Variable: " << pVariable->GetCanonicalName() << " has only some points assigned" << std::endl;
        }
    }
    */

    /* Achtung, Achtung!!
     * Sort by name so that every platform get identical vectors! */
    std::sort(ptrarrParameters.begin(), ptrarrParameters.end(), compareCanonicalNames<daeParameter_t*>);
    std::sort(ptrarrInputs.begin(),     ptrarrInputs.end(),     compareCanonicalNames<daeVariable_t*>);
    std::sort(ptrarrOutputs.begin(),    ptrarrOutputs.end(),    compareCanonicalNames<daeVariable_t*>);
    std::sort(ptrarrSTNs.begin(),       ptrarrSTNs.end(),       compareCanonicalNames<daeSTN_t*>);
}

void daeModel::GetFMIInterface(std::map<size_t, daeFMI2Object_t>& mapInterface)
{
    std::vector<daeParameter_t*> ptrarrParameters;
    std::vector<daeVariable_t*>  ptrarrInputs;
    std::vector<daeVariable_t*>  ptrarrOutputs;
    std::vector<daeSTN_t*>       ptrarrSTNs;
    std::map<size_t, std::vector<size_t> > mapDomainsIndexes;
    std::map<size_t, std::vector<size_t> >::iterator iter;

    GetCoSimulationInterface(ptrarrParameters,
                             ptrarrInputs,
                             ptrarrOutputs,
                             ptrarrSTNs);

    size_t counter = 0;
    for(size_t i = 0; i < ptrarrParameters.size(); i++)
    {
        daeParameter* pParameter = dynamic_cast<daeParameter*>(ptrarrParameters[i]);
        mapDomainsIndexes.clear();
        pParameter->GetDomainsIndexesMap(mapDomainsIndexes, 0);
        size_t noPoints = pParameter->GetNumberOfPoints();

        for(iter = mapDomainsIndexes.begin(); iter != mapDomainsIndexes.end(); iter++)
        {
            daeFMI2Object_t fmi;
            fmi.type        = "Parameter";
            fmi.parameter   = pParameter;
            fmi.reference   = counter;
            fmi.indexes     = iter->second;
            fmi.name        = daeGetStrippedRelativeName(this, pParameter);
            if(noPoints > 1)
                fmi.name  += "(" + toString(iter->second, ",") + ")";
            fmi.description = pParameter->GetDescription();
            fmi.units       = pParameter->GetUnits().toString();
            mapInterface[counter] = fmi;
            counter++;
        }
    }

    for(size_t i = 0; i < ptrarrSTNs.size(); i++)
    {
        daeSTN* pSTN = dynamic_cast<daeSTN*>(ptrarrSTNs[i]);

        daeFMI2Object_t fmi;
        fmi.reference      = counter;
        fmi.stn            = pSTN;
        //fmi.indexes - >Not relevant in this context
        fmi.name           = daeGetStrippedRelativeName(this, pSTN);
        fmi.type           = "STN";
        fmi.description    = pSTN->GetDescription();
        fmi.units          = ""; // Doesn't exist
        mapInterface[counter] = fmi;
        counter++;
    }

    for(size_t i = 0; i < ptrarrInputs.size(); i++)
    {
        daeVariable* pVariable = dynamic_cast<daeVariable*>(ptrarrInputs[i]);
        mapDomainsIndexes.clear();
        pVariable->GetDomainsIndexesMap(mapDomainsIndexes, 0);
        size_t noPoints = pVariable->GetNumberOfPoints();

        for(iter = mapDomainsIndexes.begin(); iter != mapDomainsIndexes.end(); iter++)
        {
            daeFMI2Object_t fmi;
            fmi.type      = "Input";
            fmi.variable  = pVariable;
            fmi.reference = counter;
            fmi.indexes   = iter->second;
            fmi.name      = daeGetStrippedRelativeName(this, pVariable);
            if(noPoints > 1)
                fmi.name  += "(" + toString(iter->second, ",") + ")";
            fmi.description = pVariable->GetDescription();
            fmi.units       = pVariable->GetVariableType()->GetUnits().toString();
            mapInterface[counter] = fmi;
            counter++;
        }
    }

    for(size_t i = 0; i < ptrarrOutputs.size(); i++)
    {
        daeVariable* pVariable = dynamic_cast<daeVariable*>(ptrarrOutputs[i]);
        mapDomainsIndexes.clear();
        pVariable->GetDomainsIndexesMap(mapDomainsIndexes, 0);
        size_t noPoints = pVariable->GetNumberOfPoints();

        for(iter = mapDomainsIndexes.begin(); iter != mapDomainsIndexes.end(); iter++)
        {
            daeFMI2Object_t fmi;
            fmi.type      = "Output";
            fmi.variable  = pVariable;
            fmi.reference = counter;
            fmi.indexes   = iter->second;
            fmi.name      = daeGetStrippedRelativeName(this, pVariable);
            if(noPoints > 1)
                fmi.name  += "(" + toString(iter->second, ",") + ")";
            fmi.description = pVariable->GetDescription();
            fmi.units       = pVariable->GetVariableType()->GetUnits().toString();
            mapInterface[counter] = fmi;
            counter++;
        }
    }
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

    //SetModelAndCanonicalName(pDomain);
    pDomain->SetModel(this);
    dae_push_back(m_ptrarrDomains, pDomain);
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

    //SetModelAndCanonicalName(pVariable);
    pVariable->SetModel(this);
    dae_push_back(m_ptrarrVariables, pVariable);
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

    //SetModelAndCanonicalName(pParameter);
    pParameter->SetModel(this);
    dae_push_back(m_ptrarrParameters, pParameter);
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
//	if(CheckName(m_ptrarrComponents, strName))
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Child Model [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
//		throw e;
//	}

    //SetModelAndCanonicalName(pModel);
    pModel->SetModel(this);
    dae_push_back(m_ptrarrComponents, pModel);
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

    //SetModelAndCanonicalName(pPort);
    pPort->SetModel(this);
    dae_push_back(m_ptrarrPorts, pPort);
}

void daeModel::AddEventPort(daeEventPort* pPort)
{
    std::string strName = pPort->GetName();
    if(strName.empty())
    {
        daeDeclareException(exInvalidCall);
        e << "EventPort name cannot be empty";
        throw e;
    }
    if(CheckName(m_ptrarrEventPorts, strName))
    {
        daeDeclareException(exInvalidCall);
        e << "EventPort [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
        throw e;
    }

    //SetModelAndCanonicalName(pPort);
    pPort->SetModel(this);
    dae_push_back(m_ptrarrEventPorts, pPort);
}

void daeModel::AddOnEventAction(daeOnEventActions* pOnEventAction)
{
    std::string strName = pOnEventAction->GetName();
    if(strName.empty())
    {
        daeDeclareException(exInvalidCall);
        e << "OnEventAction name cannot be empty";
        throw e;
    }
    if(CheckName(m_ptrarrOnEventActions, strName))
    {
        daeDeclareException(exInvalidCall);
        e << "OnEventAction [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
        throw e;
    }

    pOnEventAction->SetModel(this);
    dae_push_back(m_ptrarrOnEventActions, pOnEventAction);
}

void daeModel::AddOnConditionAction(daeOnConditionActions* pOnConditionAction)
{
    std::string strName = pOnConditionAction->GetName();
    if(strName.empty())
    {
        daeDeclareException(exInvalidCall);
        e << "OnConditionAction name cannot be empty";
        throw e;
    }
    if(CheckName(m_ptrarrOnConditionActions, strName))
    {
        daeDeclareException(exInvalidCall);
        e << "OnConditionAction [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
        throw e;
    }

    pOnConditionAction->SetModel(this);
    dae_push_back(m_ptrarrOnConditionActions, pOnConditionAction);
}

void daeModel::AddPortConnection(daePortConnection* pPortConnection)
{
    //SetModelAndCanonicalName(pPortConnection);
    pPortConnection->SetModel(this);
    dae_push_back(m_ptrarrPortConnections, pPortConnection);
}

void daeModel::AddEventPortConnection(daeEventPortConnection* pEventPortConnection)
{
    //SetModelAndCanonicalName(pEventPortConnection);
    pEventPortConnection->SetModel(this);
    dae_push_back(m_ptrarrEventPortConnections, pEventPortConnection);
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

    //SetModelAndCanonicalName(pPortArray);
    pPortArray->SetModel(this);
    dae_push_back(m_ptrarrPortArrays, pPortArray);
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
    if(CheckName(m_ptrarrComponentArrays, strName))
    {
        daeDeclareException(exInvalidCall);
        e << "ModelArray [" << strName << "] already exists in the model [" << GetCanonicalName() << "]";
        throw e;
    }

    //SetModelAndCanonicalName(pModelArray);
    pModelArray->SetModel(this);
    dae_push_back(m_ptrarrComponentArrays, pModelArray);
}

void daeModel::AddExternalFunction(daeExternalFunction_t* pExternalFunction)
{
    //pExternalFunction->SetModel(this);
    dae_push_back(m_ptrarrExternalFunctions, pExternalFunction);
}

void daeModel::AddEquation(daeEquation* pEquation)
{
    //SetModelAndCanonicalName(pEquation);
    pEquation->SetModel(this);
    dae_push_back(m_ptrarrEquations, pEquation);
}

void daeModel::AddDomain(daeDomain& rDomain, const string& strName, const unit& units, string strDescription)
{
    rDomain.SetUnits(units);
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

void daeModel::AddParameter(daeParameter& rParameter, const string& strName, const unit& units, string strDescription)
{
    rParameter.SetName(strName);
    rParameter.SetDescription(strDescription);
    rParameter.SetUnits(units);
    AddParameter(&rParameter);
}

daeState unknownState = daeState();

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

    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
    if(pCurrentState == &unknownState)
    {
        daeDeclareException(exInvalidCall);
        e << "New STN() should be created within some other state using a call to STATE(), IF(), ELSE_IF() and ELSE() functions"
          << " - not directly after the previous STN() call";
        throw e;
    }

    pSTN->SetName(strName);
    pSTN->SetModel(this);
    //SetModelAndCanonicalName(pSTN);

    if(!pCurrentState) // Add a top-level STN
    {
        pSTN->m_pParentState = NULL;
        dae_push_back(m_ptrarrSTNs, pSTN);
    }
    else // Add a nested STN to the current State
    {
        pSTN->m_pParentState = pCurrentState;
        pCurrentState->AddNestedSTN(pSTN);
    }

    return pSTN;
}

daeIF* daeModel::AddIF(const string& strName)
{
    daeIF* pIF = new daeIF;

    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
    if(pCurrentState == &unknownState)
    {
        daeDeclareException(exInvalidCall);
        e << "New IF() should be created within some other state using a call to STATE(), IF(), ELSE_IF() and ELSE() functions"
          << " - not directly after the previous STN() call";
        throw e;
    }

    string strIFName = strName;

    if(!pCurrentState) // Add a top-level IF
    {
        if(strIFName.empty())
            strIFName = "IF_" + toString<size_t>(m_ptrarrSTNs.size());
        pIF->m_pParentState = NULL;
        dae_push_back(m_ptrarrSTNs, pIF);
    }
    else // Add a nested IF to the current State
    {
        if(strIFName.empty())
            strIFName = "IF_" + toString<size_t>(m_ptrarrSTNs.size()) + "_" + toString<size_t>(pCurrentState->m_ptrarrSTNs.size());
        pIF->m_pParentState = pCurrentState;
        pCurrentState->AddNestedSTN(pIF);
    }

    pIF->SetName(strIFName);
    pIF->SetModel(this);
    //SetModelAndCanonicalName(pIF);

    return pIF;
}

void daeModel::IF(const daeCondition& rCondition, real_t dEventTolerance, const string& strIFName, const string& strIFDescription,
                                                                          const string& strStateName, const string& strStateDescription)
{
// Create daeIF and push it on the stack
    daeIF* _pIF = AddIF(strIFName);
    _pIF->SetDescription(strIFDescription);
    m_ptrarrStackSTNs.push(_pIF);
    //std::cout << "Set current STN: " << _pIF->GetCanonicalName() << std::endl;

// Create a new state and push it on the stack
    daeState* _pState = _pIF->AddState(strStateName);
    _pState->SetDescription(strStateDescription);
    m_ptrarrStackStates.push(_pState);
    //std::cout << "Set current state: " << _pState->GetCanonicalName() << std::endl;

    daeOnConditionActions* _pOnConditionActions = new daeOnConditionActions();
    _pOnConditionActions->Create_IF(_pState, rCondition, dEventTolerance);
}

void daeModel::ELSE_IF(const daeCondition& rCondition, real_t dEventTolerance, const string& strStateName, const string& strStateDescription)
{
// Current STN must exist!!!
    daeSTN* pCurrentSTN = (m_ptrarrStackSTNs.empty() ? NULL : m_ptrarrStackSTNs.top());
    if(!pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "Before calling ELSE_IF() an IF block must be started with a call to IF() function";
        throw e;
    }
    //std::cout << "Current STN: " << pCurrentSTN->GetCanonicalName() << std::endl;

// Current state must exist!!!
    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
    if(!pCurrentState)
    {
        daeDeclareException(exInvalidCall);
        e << "ELSE_IF() can only be called after a call to IF() or another ELSE_IF()";
        throw e;
    }

// Current STN and the parent STN of the current state must be the same object!!
    if(pCurrentState->GetSTN() != pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "ELSE_IF() can only be called after a call to IF() or another ELSE_IF()";
        throw e;
    }

// Parrent of the current state must be an IF block!!
    if(typeid(*pCurrentSTN) != typeid(daeIF))
    {
        daeDeclareException(exInvalidCall);
        e << "ELSE_IF() can only be called after a call to IF() or another ELSE_IF()";
        throw e;
    }
    daeIF* _pIF = dynamic_cast<daeIF*>(pCurrentSTN);
    if(!_pIF)
        daeDeclareAndThrowException(exInvalidPointer);

// Create a new state, remove the top from the stack and push the new one
    daeState* _pState = _pIF->AddState(strStateName);
    _pState->SetDescription(strStateDescription);
    m_ptrarrStackStates.pop();
    m_ptrarrStackStates.push(_pState);
    //std::cout << "Set current state: " << _pState->GetCanonicalName() << std::endl;

    daeOnConditionActions* _pOnConditionActions = new daeOnConditionActions();
    _pOnConditionActions->Create_IF(_pState, rCondition, dEventTolerance);
}

void daeModel::ELSE(const string& strStateDescription)
{
// Current STN must exist!!!
    daeSTN* pCurrentSTN = (m_ptrarrStackSTNs.empty() ? NULL : m_ptrarrStackSTNs.top());
    if(!pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "Before calling ELSE() an IF block must be started with a call to IF() function";
        throw e;
    }
    //std::cout << "Current STN: " << pCurrentSTN->GetCanonicalName() << std::endl;

// Current state must exist!!!
    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
    if(!pCurrentState)
    {
        daeDeclareException(exInvalidCall);
        e << "ELSE() can only be called after a call to IF() or another ELSE_IF()";
        throw e;
    }

// Current STN and the parent STN of the current state must be the same object!!
    if(pCurrentState->GetSTN() != pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "ELSE() can only be called after a call to IF() or another ELSE_IF()";
        throw e;
    }

// Parrent of the current state must be an IF block!!
    if(typeid(*pCurrentSTN) != typeid(daeIF))
    {
        daeDeclareException(exInvalidCall);
        e << "ELSE_IF() can only be called after a call to IF() or another ELSE_IF()";
        throw e;
    }
    daeIF* _pIF = dynamic_cast<daeIF*>(pCurrentSTN);

// Create a new state, remove the top from the stack and push the new one
    daeState* _pState = _pIF->CreateElse("");
    _pState->SetDescription(strStateDescription);
    m_ptrarrStackStates.pop();
    m_ptrarrStackStates.push(_pState);
    //std::cout << "Set current state: " << _pState->GetCanonicalName() << std::endl;
}

void daeModel::END_IF(void)
{
// Current STN must exist!!!
    daeSTN* pCurrentSTN = (m_ptrarrStackSTNs.empty() ? NULL : m_ptrarrStackSTNs.top());
    if(!pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "Before calling END_IF() an IF block must be started with a call to IF() function";
        throw e;
    }

// Current state must exist!!!
    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
    if(!pCurrentState)
    {
        daeDeclareException(exInvalidCall);
        e << "END_IF() can only be called after a call to IF() or another ELSE_IF()";
        throw e;
    }

// Current STN and the parent STN of the current state must be the same object!!
    if(pCurrentState->GetSTN() != pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "END_IF can only be called after a call to IF() or ELSE_IF()";
        throw e;
    }

// Parrent of the current state must be an IF block!!
    if(typeid(*pCurrentSTN) != typeid(daeIF))
    {
        daeDeclareException(exInvalidCall);
        e << "ELSE_IF() can only be called after a call to IF() or another ELSE_IF()";
        throw e;
    }
    daeIF* _pIF = dynamic_cast<daeIF*>(pCurrentSTN);

// Finalize the IF block
    _pIF->FinalizeDeclaration();

// Remove the top state from the stack
    m_ptrarrStackStates.pop();
// Remove the top STN from the stack
    m_ptrarrStackSTNs.pop();
}

daeSTN* daeModel::STN(const string& strName, const string& strDescription)
{
// Create daeSTN and push it on the stack
    daeSTN* _pSTN = AddSTN(strName);
    _pSTN->SetDescription(strDescription);
    m_ptrarrStackSTNs.push(_pSTN);
    //std::cout << "Set current STN: " << _pSTN->GetCanonicalName() << std::endl;

    m_ptrarrStackStates.push(&unknownState);
    //std::cout << "Set current state: Unknown" << std::endl;

    return _pSTN;
}

daeState* daeModel::STATE(const string& strName, const string& strDescription)
{
// Current STN must exist!!!
    daeSTN* pCurrentSTN = (m_ptrarrStackSTNs.empty() ? NULL : m_ptrarrStackSTNs.top());
    if(!pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "Before calling STATE() a STN must be started with a call to STN() function";
        throw e;
    }
    //std::cout << "Current STN: " << pCurrentSTN->GetCanonicalName() << std::endl;

// The current STN must be daeSTN
    if(typeid(*pCurrentSTN) != typeid(daeSTN))
    {
        daeDeclareException(exInvalidCall);
        e << "STATE() can only be called after a call to STN()";
        throw e;
    }

    daeState* pCurrentState = NULL;
    if(pCurrentSTN->m_ptrarrStates.empty()) // We are adding the first state: there is no current state
    {
        pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
        if(!pCurrentState)
        {
            daeDeclareException(exInvalidCall);
            e << "STATE() can only be called after a call to STN()";
            throw e;
        }
        if(pCurrentState != &unknownState)
        {
            daeDeclareException(exInvalidCall);
            e << "STATE() can only be called after a call to STN()";
            throw e;
        }
    }
    else // We are adding additional state: the current state must exist
    {
        pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
        if(!pCurrentState)
        {
            daeDeclareException(exInvalidCall);
            e << "STATE() can only be called after a call to STN() or after the previous STATE() call";
            throw e;
        }

    // Current STN and the parent STN of the current state must be the same object!!
        if(pCurrentState->GetSTN() != pCurrentSTN)
        {
            daeDeclareException(exInvalidCall);
            e << "STATE() can only be called after a call to STN()";
            throw e;
        }
    }

// Create a new state
    daeState* pState = pCurrentSTN->AddState(strName);
    pState->SetDescription(strDescription);

// Pop the previous state from the stack
    m_ptrarrStackStates.pop();
// Push the new state on the stack
    m_ptrarrStackStates.push(pState);
    //std::cout << "Set current state: " << pState->GetCanonicalName() << std::endl;

    return pState;
}

void daeModel::END_STN(void)
{
// Current STN must exist!!!
    daeSTN* pCurrentSTN = (m_ptrarrStackSTNs.empty() ? NULL : m_ptrarrStackSTNs.top());
    if(!pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "Before calling END_STN() a STN must be started with a call to STN() function";
        throw e;
    }

// Current state must exist!!!
    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
    if(!pCurrentState)
    {
        daeDeclareException(exInvalidCall);
        e << "END_STN() can only be called after a call to STATE()";
        throw e;
    }
    if(pCurrentState == &unknownState)
    {
        daeDeclareException(exInvalidCall);
        e << "END_STN() can only be called after a call to STN() and one or more STATE() calls - not right after STN() call";
        throw e;
    }

// Current STN and the parent STN of the current state must be the same object!!
    if(pCurrentState->GetSTN() != pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "END_STN can only be called after a call to STN() and STATE()";
        throw e;
    }

// Parrent of the current state must be STN!!
    if(typeid(*pCurrentSTN) != typeid(daeSTN))
    {
        daeDeclareException(exInvalidCall);
        e << "END_STN() can only be called after a call to STN() and STATE()";
        throw e;
    }

    pCurrentSTN->FinalizeDeclaration();

// Remove the top state from the stack
    m_ptrarrStackStates.pop();
// Remove the top STN from the stack
    m_ptrarrStackSTNs.pop();
}

void daeModel::SWITCH_TO(const string& strState, const daeCondition& rCondition, real_t dEventTolerance)
{
// Current STN must exist!!!
    daeSTN* pCurrentSTN = (m_ptrarrStackSTNs.empty() ? NULL : m_ptrarrStackSTNs.top());
    if(!pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "Before calling SWITCH_TO() a STN must be started with a call to STN() function";
        throw e;
    }

// Current state must exist!!!
    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
    if(!pCurrentState)
    {
        daeDeclareException(exInvalidCall);
        e << "Before calling SWITCH_TO() a new state must be started with a call to STATE() function";
        throw e;
    }
    if(pCurrentState == &unknownState)
    {
        daeDeclareException(exInvalidCall);
        e << "SWITCH_TO() can only be called after a call to STATE() - not right after STN() call";
        throw e;
    }

// Current STN and the parent STN of the current state must be the same object!!
    if(pCurrentState->GetSTN() != pCurrentSTN)
    {
        daeDeclareException(exInvalidCall);
        e << "SWITCH_TO() can only be called after a call to STN()";
        throw e;
    }

// Parrent of the current state must be STN!!
    if(typeid(*pCurrentSTN) != typeid(daeSTN))
    {
        daeDeclareException(exInvalidCall);
        e << "SWITCH_TO() can only be called after a call to STN()";
        throw e;
    }

    daeOnConditionActions* _pOnConditionActions = new daeOnConditionActions();
    _pOnConditionActions->Create_SWITCH_TO(pCurrentState, strState, rCondition, dEventTolerance);
}

void daeModel::ON_CONDITION(const daeCondition&								rCondition,
                            vector< pair<string, string> >&					arrSwitchToStates,
                            vector< pair<daeVariableWrapper, adouble> >&	arrSetVariables,
                            vector< pair<daeEventPort*, adouble> >&			arrTriggerEvents,
                            vector<daeAction*>&								ptrarrUserDefinedActions,
                            real_t											dEventTolerance)
{
    size_t i;
    string strSTN;
    string strStateTo;
    daeAction* pAction;
    daeEventPort* pEventPort;
    pair<daeVariableWrapper, adouble> p;
    pair<string, string> p1;
    pair<daeEventPort*, adouble> p3;
    daeVariableWrapper variable;
    adouble value;
    vector<daeAction*> ptrarrActions;

    daeSTN* pCurrentSTN = (m_ptrarrStackSTNs.empty() ? NULL : m_ptrarrStackSTNs.top());

// ChangeState
    for(i = 0; i < arrSwitchToStates.size(); i++)
    {
        p1 = arrSwitchToStates[i];
        strSTN     = p1.first;
        strStateTo = p1.second;
        if(strStateTo.empty())
            daeDeclareAndThrowException(exInvalidCall);

        pAction = new daeAction(string("actionChangeState_") + strSTN + "_" + strStateTo, this, strSTN, strStateTo, string(""));
        ptrarrActions.push_back(pAction);
    }

// TriggerEvents
    for(i = 0; i < arrTriggerEvents.size(); i++)
    {
        p3 = arrTriggerEvents[i];
        pEventPort = p3.first;
        value      = p3.second;
        if(!pEventPort)
            daeDeclareAndThrowException(exInvalidPointer);

        pAction = new daeAction(string("actionTriggerEvent_") + pEventPort->GetName(), this, pEventPort, value, string(""));

        ptrarrActions.push_back(pAction);
    }

// SetVariables
    for(i = 0; i < arrSetVariables.size(); i++)
    {
        p = arrSetVariables[i];
        variable = p.first;
        value    = p.second;

        string name = ReplaceAll(string("actionSetVariable_") + variable.GetName(), '.', '_');
        pAction = new daeAction(name, this, variable, value, string(""));

        ptrarrActions.push_back(pAction);
    }

    daeOnConditionActions* _pOnConditionActions = new daeOnConditionActions();

    if(pCurrentSTN)
    {
    // If we are in the IF block throw an exception
        if(typeid(*pCurrentSTN) == typeid(daeIF))
        {
            daeDeclareException(exInvalidCall);
            e << "ON_CONDITION() cannot be called from the IF block";
            throw e;
        }

    // Current state must exist!!!
        daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
        if(!pCurrentState)
        {
            daeDeclareException(exInvalidCall);
            e << "Before calling ON_CONDITION() a new state must be started with a call to STATE() function";
            throw e;
        }
        if(pCurrentState == &unknownState)
        {
            daeDeclareException(exInvalidCall);
            e << "ON_CONDITION() can only be called after a call to STATE() - not right after STN() call";
            throw e;
        }

        _pOnConditionActions->Create_ON_CONDITION(pCurrentState, this, rCondition, ptrarrActions, ptrarrUserDefinedActions, dEventTolerance);
    }
    else
    {
        _pOnConditionActions->Create_ON_CONDITION(NULL, this, rCondition, ptrarrActions, ptrarrUserDefinedActions, dEventTolerance);
    }
}

void daeModel::ON_EVENT(daeEventPort*									pTriggerEventPort,
                        vector< pair<string, string> >&					arrSwitchToStates,
                        vector< pair<daeVariableWrapper, adouble> >&	arrSetVariables,
                        vector< pair<daeEventPort*, adouble> >&			arrTriggerEvents,
                        vector<daeAction*>&								ptrarrUserDefinedActions)
{
    size_t i;
    daeAction* pAction;
    string strSTN;
    string strStateTo;
    daeEventPort* pEventPort;
    pair<string, string> p1;
    pair<daeVariableWrapper, adouble> p2;
    pair<daeEventPort*, adouble> p3;
    adouble value;
    daeVariableWrapper variable;
    std::vector<daeAction*> ptrarrOnEventActions;

    if(!pTriggerEventPort)
        daeDeclareAndThrowException(exInvalidPointer);

    daeSTN* pCurrentSTN = (m_ptrarrStackSTNs.empty() ? NULL : m_ptrarrStackSTNs.top());

/*  ACHTUNG, ACHTUNG!!!
    We SHOULD be able to have OnEvent actions EVEN on the OUTLET event ports!!
    Therefore, the type check below is commented out.

    if(pTriggerEventPort->GetType() != eInletPort)
    {
        daeDeclareException(exInvalidCall);
        e << "ON_EVENT actions can only be set for inlet event ports, in model " << GetCanonicalName();
        throw e;
    }
*/

// ChangeState
    for(i = 0; i < arrSwitchToStates.size(); i++)
    {
        p1 = arrSwitchToStates[i];
        strSTN     = p1.first;
        strStateTo = p1.second;
        if(strStateTo.empty())
            daeDeclareAndThrowException(exInvalidCall);

        pAction = new daeAction(string("actionChangeState_") + strSTN + "_" + strStateTo, this, strSTN, strStateTo, string(""));
        ptrarrOnEventActions.push_back(pAction);
    }

// TriggerEvents
    for(i = 0; i < arrTriggerEvents.size(); i++)
    {
        p3 = arrTriggerEvents[i];
        pEventPort = p3.first;
        value      = p3.second;
        if(!pEventPort)
            daeDeclareAndThrowException(exInvalidPointer);

        pAction = new daeAction(string("actionTriggerEvent_") + pEventPort->GetName(), this, pEventPort, value, string(""));
        ptrarrOnEventActions.push_back(pAction);
    }

// SetVariables
    for(i = 0; i < arrSetVariables.size(); i++)
    {
        p2 = arrSetVariables[i];
        variable = p2.first;
        value    = p2.second;

        pAction = new daeAction(string("actionSetVariable_") + variable.GetName(), this, variable, value, string(""));
        ptrarrOnEventActions.push_back(pAction);
    }

    daeOnEventActions* pOnEventAction;
    if(pCurrentSTN)
    {
    // If we are in the IF block throw an exception
        if(typeid(*pCurrentSTN) == typeid(daeIF))
        {
            daeDeclareException(exInvalidCall);
            e << "ON_EVENT() can only be called after a call to STN() not IF()";
            throw e;
        }

    // Current state must exist!!!
        daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());
        if(!pCurrentState)
        {
            daeDeclareException(exInvalidCall);
            e << "Before calling ON_EVENT() a new state must be started with a call to STATE() function";
            throw e;
        }
        if(pCurrentState == &unknownState)
        {
            daeDeclareException(exInvalidCall);
            e << "ON_EVENT() can only be called after a call to STATE() - not right after STN() call";
            throw e;
        }

        pOnEventAction = new daeOnEventActions(pTriggerEventPort, pCurrentState, ptrarrOnEventActions, ptrarrUserDefinedActions, string(""));
    }
    else
    {
        pOnEventAction = new daeOnEventActions(pTriggerEventPort, this, ptrarrOnEventActions, ptrarrUserDefinedActions, string(""));

    // Attach ONLY those OnEventActions that belong to the model; others will be set during the active state changes
        pTriggerEventPort->Attach(pOnEventAction);
    }
}

void daeModel::BuildExpressions(daeBlock* pBlock)
{
    size_t i, k, m;
    daeModel* pModel;
    daeModelArray* pModelArray;
    daeOnConditionActions* pOnConditionActions;
    pair<size_t, daeExpressionInfo> pairExprInfo;

    daeExecutionContext EC;
    EC.m_pDataProxy					= m_pDataProxy.get();
    EC.m_pBlock						= pBlock;
    EC.m_eEquationCalculationMode	= eCreateFunctionsIFsSTNs;

// I have to set this since Create_adouble called from adSetup nodes needs it
    daeModel* pTopLevelModel = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
    pTopLevelModel->PropagateGlobalExecutionContext(&EC);

    size_t nIndexInModel = 0;
    for(k = 0; k < m_ptrarrOnConditionActions.size(); k++)
    {
        pOnConditionActions = m_ptrarrOnConditionActions[k];
        if(!pOnConditionActions)
            daeDeclareAndThrowException(exInvalidPointer);

    // Fill array with expressions of the form: left - right,
    // made of the conditional expressions, like: left >= right ... etc etc
        m_pDataProxy->SetGatherInfo(true);
        pBlock->SetInitializeMode(true);
            pOnConditionActions->m_Condition.BuildExpressionsArray(&EC);
            for(m = 0; m < pOnConditionActions->m_Condition.m_ptrarrExpressions.size(); m++)
            {
                pairExprInfo.first                        = nIndexInModel;
                pairExprInfo.second.m_pExpression         = pOnConditionActions->m_Condition.m_ptrarrExpressions[m];
                pairExprInfo.second.m_pOnConditionActions = pOnConditionActions;

                pOnConditionActions->m_mapExpressionInfos.insert(pairExprInfo);
                nIndexInModel++;
            }
        m_pDataProxy->SetGatherInfo(false);
        pBlock->SetInitializeMode(false);
    }

// Restore it to NULL
    pTopLevelModel->PropagateGlobalExecutionContext(NULL);

// Next, BuildExpressions in each child-model
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->BuildExpressions(pBlock);
    }

// Finally, BuildExpressions in each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->BuildExpressions(pBlock);
    }
}

bool daeModel::CheckDiscontinuities(void)
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;
    daeOnConditionActions* pOnConditionActions;
    pair<size_t, daeExpressionInfo> pairExprInfo;
    map<size_t, daeExpressionInfo>::iterator iter;

    daeExecutionContext EC;
    EC.m_pDataProxy					= m_pDataProxy.get();
    EC.m_pBlock						= m_pDataProxy->GetBlock();
    EC.m_eEquationCalculationMode	= eCalculate;

    for(i = 0; i < m_ptrarrOnConditionActions.size(); i++)
    {
        pOnConditionActions = m_ptrarrOnConditionActions[i];
        if(pOnConditionActions->m_Condition.Evaluate(&EC)) // There is a discontinuity, therefore return true
        {
            m_pDataProxy->SetLastSatisfiedCondition(&pOnConditionActions->m_Condition);
            return true;
        }
    }

    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        if(pModel->CheckDiscontinuities()) // There is a discontinuity, therefore return true
            return true;
    }

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        if(pModelArray->CheckDiscontinuities()) // There is a discontinuity, therefore return true
            return true;
    }

    return false;
}

void daeModel::AddExpressionsToBlock(daeBlock* pBlock)
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;
    daeOnConditionActions* pOnConditionActions;
    pair<size_t, daeExpressionInfo> pairExprInfo;
    map<size_t, daeExpressionInfo>::iterator iter;

    for(i = 0; i < m_ptrarrOnConditionActions.size(); i++)
    {
        pOnConditionActions = m_ptrarrOnConditionActions[i];

        for(iter = pOnConditionActions->m_mapExpressionInfos.begin(); iter != pOnConditionActions->m_mapExpressionInfos.end(); iter++)
        {
            pairExprInfo		= *iter;
            pairExprInfo.first	= pBlock->m_mapExpressionInfos.size();
            pBlock->m_mapExpressionInfos.insert(pairExprInfo);
        }
    }

    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->AddExpressionsToBlock(pBlock);
    }

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->AddExpressionsToBlock(pBlock);
    }
}

void daeModel::ExecuteOnConditionActions(void)
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;
    daeOnConditionActions* pOnConditionActions;
    pair<size_t, daeExpressionInfo> pairExprInfo;
    map<size_t, daeExpressionInfo>::iterator iter;

    daeExecutionContext EC;
    EC.m_pDataProxy					= m_pDataProxy.get();
    EC.m_pBlock						= m_pDataProxy->GetBlock();
    EC.m_eEquationCalculationMode	= eCalculate;

    for(i = 0; i < m_ptrarrOnConditionActions.size(); i++)
    {
        pOnConditionActions = m_ptrarrOnConditionActions[i];

        if(pOnConditionActions->m_Condition.Evaluate(&EC))
        {
            if(m_pDataProxy->PrintInfo())
                LogMessage(string("The condition: ") + pOnConditionActions->GetConditionAsString() + string(" is satisfied"), 0);

            pOnConditionActions->Execute();
            break;
        }
    }

    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->ExecuteOnConditionActions();
    }

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->ExecuteOnConditionActions();
    }
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

void daeModel::AddEventPort(daeEventPort& rPort, const string& strName, daeePortType ePortType, string strDescription)
{
    rPort.SetName(strName);
    rPort.SetDescription(strDescription);
    rPort.SetType(ePortType);
    AddEventPort(&rPort);
}

void daeModel::AddOnEventAction(daeOnEventActions& rOnEventAction, const string& strName, string strDescription)
{
    rOnEventAction.SetName(strName);
    rOnEventAction.SetDescription(strDescription);
    AddOnEventAction(&rOnEventAction);
}

daeEquation* daeModel::CreateEquation(const string& strName, string strDescription, real_t dScaling)
{
    string strEqName;
    daeState* pCurrentState = (m_ptrarrStackStates.empty() ? NULL : m_ptrarrStackStates.top());

    daeEquation* pEquation = new daeEquation();

    pEquation->SetDescription(strDescription);
    pEquation->SetScaling(dScaling);

    if(!pCurrentState)
    {
        strEqName = (strName.empty() ? "Equation_" + toString<size_t>(m_ptrarrEquations.size()) : strName);
        pEquation->SetName(strEqName);
        AddEquation(pEquation);
    }
    else
    {
        strEqName = (strName.empty() ? "Equation_" + toString<size_t>(pCurrentState->m_ptrarrEquations.size()) : strName);
        pEquation->SetName(strEqName);
        pCurrentState->AddEquation(pEquation);
    }

    return pEquation;
}

void daeModel::ConnectPorts(daePort* pPortFrom, daePort* pPortTo)
{
    daePortConnection* pPortConnection = new daePortConnection(pPortFrom, pPortTo);
    string strName = pPortFrom->GetName() + "_" + pPortTo->GetName();
    pPortConnection->SetName(strName);
    AddPortConnection(pPortConnection);
}

void daeModel::ConnectEventPorts(daeEventPort* pPortFrom, daeEventPort* pPortTo)
{
// Here, portFrom (inlet) is observer and portTo (outlet) is subject
// When the outlet port sends an event its function Notify() is called which in turn calls the function Update() in the portFrom.
// portFrom then calls its function Notify which calls Update() in all attached observers (daeAction).
    if(pPortFrom->GetType() != eInletPort)
        daeDeclareAndThrowException(exInvalidCall);
    if(pPortTo->GetType() != eOutletPort)
        daeDeclareAndThrowException(exInvalidCall);

    daeEventPortConnection* pEventPortConnection = new daeEventPortConnection(pPortFrom, pPortTo);
    string strName = pPortFrom->GetName() + "_" + pPortTo->GetName();
    pEventPortConnection->SetName(strName);
    AddEventPortConnection(pEventPortConnection);
}

void daeModel::PropagateDomain(daeDomain& propagatedDomain)
{
    size_t i;
    daePort* pPort;
    daeModel* pModel;
    daeModelArray* pModelArray;
    daePortArray* pPortArray;
    daeDomain* pDomain;

// First, propagate domain in this model
    for(i = 0; i < m_ptrarrDomains.size(); i++)
    {
        pDomain = m_ptrarrDomains[i];
        if(pDomain->GetName() == propagatedDomain.GetName())
        {
            if(propagatedDomain.GetType() == eArray)
            {
                pDomain->CreateArray(propagatedDomain.GetNumberOfPoints());
            }
            else if(propagatedDomain.GetType() == eStructuredGrid)
            {
                pDomain->CreateStructuredGrid(/*propagatedDomain.GetDiscretizationMethod(),
                                              propagatedDomain.GetDiscretizationOrder(), */
                                              propagatedDomain.GetNumberOfIntervals(),
                                              propagatedDomain.GetLowerBound(),
                                              propagatedDomain.GetUpperBound());
            }
            else if(propagatedDomain.GetType() == eUnstructuredGrid)
            {
                pDomain->CreateUnstructuredGrid(propagatedDomain.GetCoordinates());
            }
            else
                daeDeclareAndThrowException(exInvalidCall);
        }
    }

// Then, propagate domain in the contained ports
    for(i = 0; i < m_ptrarrPorts.size(); i++)
    {
        pPort = m_ptrarrPorts[i];
        pPort->PropagateDomain(propagatedDomain);
    }

// Next, propagate domain in the child-models
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->PropagateDomain(propagatedDomain);
    }

// Next, propagate domain in each portarray
    for(i = 0; i < m_ptrarrPortArrays.size(); i++)
    {
        pPortArray = m_ptrarrPortArrays[i];
        pPortArray->PropagateDomain(propagatedDomain);
    }

// Finally, propagate domain in the modelarrays
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->PropagateDomain(propagatedDomain);
    }
}

void daeModel::PropagateParameter(daeParameter& propagatedParameter)
{
    size_t i;
    daePort* pPort;
    daeModel* pModel;
    daeModelArray* pModelArray;
    daePortArray* pPortArray;
    daeParameter* pParameter;
    std::vector<quantity> quantities;

// First, propagate parameter in this model
    for(i = 0; i < m_ptrarrParameters.size(); i++)
    {
        pParameter = m_ptrarrParameters[i];
        if(pParameter->GetName() == propagatedParameter.GetName())
        {
            propagatedParameter.GetValues(quantities);
            pParameter->SetValues(quantities);
        }
    }

// Then, propagate parameter in the contained ports
    for(i = 0; i < m_ptrarrPorts.size(); i++)
    {
        pPort = m_ptrarrPorts[i];
        pPort->PropagateParameter(propagatedParameter);
    }

// Next, propagate parameter in the child-models
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->PropagateParameter(propagatedParameter);
    }

// Next, propagate parameter in each portarray
    for(i = 0; i < m_ptrarrPortArrays.size(); i++)
    {
        pPortArray = m_ptrarrPortArrays[i];
        pPortArray->PropagateParameter(propagatedParameter);
    }

// Finally, propagate parameter in the modelarrays
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->PropagateParameter(propagatedParameter);
    }
}

boost::shared_ptr<daeDataProxy_t> daeModel::GetDataProxy(void) const
{
    return m_pDataProxy;
}

void daeModel::RemoveModel(daeModel* pObject)
{
    m_ptrarrComponents.Remove(pObject);
}

void daeModel::RemoveEquation(daeEquation* pObject)
{
    m_ptrarrEquations.Remove(pObject);
}

void daeModel::RemoveSTN(daeSTN* pObject)
{
    m_ptrarrSTNs.Remove(pObject);
}

void daeModel::RemovePortConnection(daePortConnection* pObject)
{
    m_ptrarrPortConnections.Remove(pObject);
}

void daeModel::RemoveEventPortConnection(daeEventPortConnection* pObject)
{
    m_ptrarrEventPortConnections.Remove(pObject);
}

void daeModel::RemoveDomain(daeDomain* pObject)
{
    m_ptrarrDomains.Remove(pObject);
}

void daeModel::RemoveParameter(daeParameter* pObject)
{
    m_ptrarrParameters.Remove(pObject);
}

void daeModel::RemoveVariable(daeVariable* pObject)
{
    m_ptrarrVariables.Remove(pObject);
}

void daeModel::RemovePort(daePort* pObject)
{
    m_ptrarrPorts.Remove(pObject);
}

void daeModel::RemoveEventPort(daeEventPort* pObject)
{
    m_ptrarrEventPorts.Remove(pObject);
}

void daeModel::RemoveOnEventAction(daeOnEventActions* pObject)
{
    m_ptrarrOnEventActions.Remove(pObject);
}

void daeModel::RemoveOnConditionAction(daeOnConditionActions* pObject)
{
    m_ptrarrOnConditionActions.Remove(pObject);
}

void daeModel::RemovePortArray(daePortArray* pObject)
{
    m_ptrarrPortArrays.Remove(pObject);
}

void daeModel::RemoveModelArray(daeModelArray* pObject)
{
    m_ptrarrComponentArrays.Remove(pObject);
}

void daeModel::RemoveExternalFunction(daeExternalFunction_t* pObject)
{
    m_ptrarrExternalFunctions.Remove(pObject);
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

// Set reporting on for ports
    for(i = 0; i < m_ptrarrPorts.size(); i++)
    {
        pPort = m_ptrarrPorts[i];
        pPort->SetReportingOn(bOn);
    }

// Set reporting on for child models
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->SetReportingOn(bOn);
    }

// Set reporting on for each portarray
    for(i = 0; i < m_ptrarrPortArrays.size(); i++)
    {
        pPortArray = m_ptrarrPortArrays[i];
        pPortArray->SetReportingOn(bOn);
    }

// Set reporting on for each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->SetReportingOn(bOn);
    }
}

//void daeModel::DeclareData()
//{
//	daeDeclareException(exNotImplemented);
//	e << "DeclareData() function MUST be implemented in daeModel derived classes, model [" << GetCanonicalName() << "]";
//	throw e;
//}

void daeModel::DeclareEquations()
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;

// Then, create equations for each child-model
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->DeclareEquations();
    }

// Finally, create equations for each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->DeclareEquations();
    }
}

void daeModel::DeclareEquationsBase()
{
    daeDeclareException(exNotImplemented);
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
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->CreatePortConnectionEquations();
    }

// Finally, create port connection equations for each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->CreatePortConnectionEquations();
    }
}

void daeModel::PropagateDataProxy(boost::shared_ptr<daeDataProxy_t> pDataProxy)
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;

    m_pDataProxy = pDataProxy;

    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->PropagateDataProxy(pDataProxy);
    }

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->PropagateDataProxy(pDataProxy);
    }
}

void daeModel::PropagateGlobalExecutionContext(daeExecutionContext* pExecutionContext)
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;

    m_pExecutionContextForGatherInfo = pExecutionContext;

    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->PropagateGlobalExecutionContext(pExecutionContext);
    }

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
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

    daeModel* pTopLevelModel = dynamic_cast<daeModel*>(m_pDataProxy->GetTopLevelModel());
    pTopLevelModel->PropagateGlobalExecutionContext(&EC);

        if(m_ptrarrStackSTNs.size() > 0)
            daeDeclareAndThrowException(exInvalidCall);
        while(m_ptrarrStackStates.size() > 0)
            daeDeclareAndThrowException(exInvalidCall);

    // Declare equations in this model
        DeclareEquations();

    // Create ports' variable equality equations
        CreatePortConnectionEquations();

        if(m_ptrarrStackSTNs.size() > 0)
            daeDeclareAndThrowException(exInvalidCall);
        while(m_ptrarrStackStates.size() > 0)
            daeDeclareAndThrowException(exInvalidCall);

    // Create indexes in DEDIs (they are not created in the moment of declaration!)
        InitializeDEDIs();

    // Create runtime condition nodes based on setup nodes
        InitializeSTNs();
        InitializeOnEventAndOnConditionActions();

    m_pDataProxy->SetGatherInfo(false);
    pTopLevelModel->PropagateGlobalExecutionContext(NULL);
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
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        if(!pModel)
            daeDeclareAndThrowException(exInvalidPointer);
        pModel->InitializeDEDIs();
    }

// Finally, InitializeDEDIs for equations in each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        if(!pModelArray)
            daeDeclareAndThrowException(exInvalidPointer);
        pModelArray->InitializeDEDIs();
    }
}

//void daeModel::BuildUpPortConnectionEquations()
//{
//	if(!m_pDataProxy)
//		daeDeclareAndThrowException(exInvalidPointer);

//	daeExecutionContext EC;
//	EC.m_pDataProxy               = m_pDataProxy.get();
//	EC.m_eEquationCalculationMode = eCreateFunctionsIFsSTNs;

//	m_pDataProxy->SetGatherInfo(true);
//	PropagateGlobalExecutionContext(&EC);
//		CreatePortConnectionEquations();
//	m_pDataProxy->SetGatherInfo(false);
//	PropagateGlobalExecutionContext(NULL);
//}

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
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->InitializeParameters();
    }

// Next, initialize all parameters in each portarray
    for(i = 0; i < m_ptrarrPortArrays.size(); i++)
    {
        pPortArray = m_ptrarrPortArrays[i];
        pPortArray->InitializeParameters();
    }

// Finally, initialize all parameters in the modelarrays
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
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
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->Create();
    }

// Finally, initialize all arrays in the child-models
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
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
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
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
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->SetVariablesStartingIndex(_currentVariablesIndex);
        pModelArray->InitializeVariables();
        _currentVariablesIndex = pModelArray->_currentVariablesIndex;
    }

    m_nTotalNumberOfVariables = _currentVariablesIndex - m_nVariablesStartingIndex;
}

void daeModel::InitializeOnEventAndOnConditionActions(void)
{
    size_t i;
    daeOnEventActions* pOnEventActions;
    daeOnConditionActions* pOnConditionActions;
    daeModel* pModel;
    daeModelArray* pModelArray;

    // Initialize OnEventActions
    for(i = 0; i < m_ptrarrOnEventActions.size(); i++)
    {
        pOnEventActions = m_ptrarrOnEventActions[i];
        if(!pOnEventActions)
            daeDeclareAndThrowException(exInvalidPointer);

        pOnEventActions->Initialize();
    }

// Initialize OnConditionActions
    for(i = 0; i < m_ptrarrOnConditionActions.size(); i++)
    {
        pOnConditionActions = m_ptrarrOnConditionActions[i];
        if(!pOnConditionActions)
            daeDeclareAndThrowException(exInvalidPointer);

        pOnConditionActions->Initialize();
    }

// Next, initialize OnEventActions in each child-model
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->InitializeOnEventAndOnConditionActions();
    }

// Finally, initialize OnEventActions in each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->InitializeOnEventAndOnConditionActions();
    }
}

void daeModel::InitializeSTNs(void)
{
    size_t i;
    daeSTN* pSTN;
    daeModel* pModel;
    daeModelArray* pModelArray;

    // Initialize STNs
    for(i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        pSTN = m_ptrarrSTNs[i];
        if(!pSTN)
            daeDeclareAndThrowException(exInvalidPointer);

        pSTN->InitializeOnEventAndOnConditionActions();
    }

// Next, initialize STNs in each child-model
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->InitializeSTNs();
    }

// Finally, initialize STNs in each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
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
    daeExternalFunction_t* pExternalFunction;
    vector<daeEquationExecutionInfo*> ptrarrEqnExecutionInfosCreated;

// First, create EqnExecInfos info for equations in this model
    for(i = 0; i < m_ptrarrEquations.size(); i++)
    {
        pEquation = m_ptrarrEquations[i];
        if(!pEquation)
            daeDeclareAndThrowException(exInvalidPointer);

    // Create EqnExecInfos, call GatherInfo for each of them, and add them to the model
        pEquation->CreateEquationExecutionInfos(this, ptrarrEqnExecutionInfosCreated, true);
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
            pEquation->CreateEquationExecutionInfos(this, ptrarrEqnExecutionInfosCreated, true);
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
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->InitializeEquations();
    }

// Finally, gather info for equations in each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->InitializeEquations();
    }
}

void daeModel::CreateOverallIndex_BlockIndex_VariableNameMap(std::map<size_t, std::pair<size_t, string> >& mapOverallIndex_BlockIndex_VariableName,
                                                             const std::map<size_t, size_t>& mapOverallIndex_BlockIndex)
{
    size_t i, nOverallIndex, nBlockIndex;
    string strName;
    daePort* pPort;
    daeModel* pModel;
    daeVariable* pVariable;
    daeModelArray* pModelArray;
    daePortArray* pPortArray;

    std::map<size_t, std::vector<size_t> > mapDomainsIndexes;
    std::map<size_t, std::vector<size_t> >::iterator iterDomainsIndexes;
    std::map<size_t, size_t>::const_iterator iter;
    std::pair<size_t, string> p;

    for(i = 0; i < m_ptrarrVariables.size(); i++)
    {
        pVariable = m_ptrarrVariables[i];

        // Get <index within variable, domain points> map
        mapDomainsIndexes.clear();
        pVariable->GetDomainsIndexesMap(mapDomainsIndexes, 0);

        for(iterDomainsIndexes = mapDomainsIndexes.begin(); iterDomainsIndexes != mapDomainsIndexes.end(); iterDomainsIndexes++)
        {
            nOverallIndex = pVariable->m_nOverallIndex + iterDomainsIndexes->first;

            iter = mapOverallIndex_BlockIndex.find(nOverallIndex);
            if(iter != mapOverallIndex_BlockIndex.end()) // if found
                nBlockIndex = iter->second;
            else
                nBlockIndex = ULONG_MAX;

            if(iterDomainsIndexes->second.empty())
                strName = pVariable->GetCanonicalName();
            else
                strName = pVariable->GetCanonicalName() + "(" + toString(iterDomainsIndexes->second, ",") + ")";

            p.first  = nBlockIndex;
            p.second = strName;
            mapOverallIndex_BlockIndex_VariableName[nOverallIndex] = p;
        }
    }

// Then, initialize all variables in the contained ports
    for(i = 0; i < m_ptrarrPorts.size(); i++)
    {
        pPort = m_ptrarrPorts[i];
        pPort->CreateOverallIndex_BlockIndex_VariableNameMap(mapOverallIndex_BlockIndex_VariableName, mapOverallIndex_BlockIndex);
    }

// Then, initialize all variables in the child-models
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->CreateOverallIndex_BlockIndex_VariableNameMap(mapOverallIndex_BlockIndex_VariableName, mapOverallIndex_BlockIndex);
    }

// Next, initialize all variables in the portarrays
    for(i = 0; i < m_ptrarrPortArrays.size(); i++)
    {
        pPortArray = m_ptrarrPortArrays[i];
        pPortArray->CreateOverallIndex_BlockIndex_VariableNameMap(mapOverallIndex_BlockIndex_VariableName, mapOverallIndex_BlockIndex);
    }

// Finally, initialize all variables in the modelarrays
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->CreateOverallIndex_BlockIndex_VariableNameMap(mapOverallIndex_BlockIndex_VariableName, mapOverallIndex_BlockIndex);
    }
}

void daeModel::CollectAllSTNsAsVector(vector<daeSTN*>& ptrarrSTNs) const
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;

// Fill array ptrarrSTNs with all STNs in the model
    dae_add_vector(m_ptrarrSTNs, ptrarrSTNs);

/////////////////////////////////////////////////////////////////////////////////////
// What about STNs nested within states???
// States should take care of them when asked to calculate residuals, conditions etc ...,
// I guess...
/////////////////////////////////////////////////////////////////////////////////////

// Then, fill it with STNs in each child-model
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->CollectAllSTNsAsVector(ptrarrSTNs);
    }

// Finally, ill it with STNs in each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->CollectAllSTNsAsVector(ptrarrSTNs);
    }
}

void daeModel::CollectAllDomains(std::map<dae::string, daeDomain_t*>& mapDomains) const
{
    // Insert objects from the model
    for(std::vector<daeDomain*>::const_iterator iter = Domains().begin(); iter != Domains().end(); iter++)
        mapDomains[(*iter)->GetCanonicalName()] = *iter;

    // Insert objects from the ports
    for(std::vector<daePort*>::const_iterator piter = Ports().begin(); piter != Ports().end(); piter++)
        for(std::vector<daeDomain*>::const_iterator citer = (*piter)->Domains().begin(); citer != (*piter)->Domains().end(); citer++)
            mapDomains[(*citer)->GetCanonicalName()] = *citer;

    // Insert objects from the model arrays
    for(std::vector<daeModelArray*>::const_iterator maiter = ModelArrays().begin(); maiter != ModelArrays().end(); maiter++)
        (*maiter)->CollectAllDomains(mapDomains);

    // Insert objects from the port arrays
    for(std::vector<daePortArray*>::const_iterator paiter = PortArrays().begin(); paiter != PortArrays().end(); paiter++)
        (*paiter)->CollectAllDomains(mapDomains);
}

void daeModel::CollectAllParameters(std::map<dae::string, daeParameter_t*>& mapParameters) const
{
    // Insert objects from the model
    for(std::vector<daeParameter*>::const_iterator iter = Parameters().begin(); iter != Parameters().end(); iter++)
        mapParameters[(*iter)->GetCanonicalName()] = *iter;

    // Insert objects from the ports
    for(std::vector<daePort*>::const_iterator piter = Ports().begin(); piter != Ports().end(); piter++)
        for(std::vector<daeParameter*>::const_iterator citer = (*piter)->Parameters().begin(); citer != (*piter)->Parameters().end(); citer++)
            mapParameters[(*citer)->GetCanonicalName()] = *citer;

    // Insert objects from the model arrays
    for(std::vector<daeModelArray*>::const_iterator maiter = ModelArrays().begin(); maiter != ModelArrays().end(); maiter++)
        (*maiter)->CollectAllParameters(mapParameters);

    // Insert objects from the port arrays
    for(std::vector<daePortArray*>::const_iterator paiter = PortArrays().begin(); paiter != PortArrays().end(); paiter++)
        (*paiter)->CollectAllParameters(mapParameters);
}

void daeModel::CollectAllVariables(std::map<dae::string, daeVariable_t*>& mapVariables) const
{
    // Insert objects from the model
    for(std::vector<daeVariable*>::const_iterator iter = Variables().begin(); iter != Variables().end(); iter++)
        mapVariables[(*iter)->GetCanonicalName()] = *iter;

    // Insert objects from the ports
    for(std::vector<daePort*>::const_iterator piter = Ports().begin(); piter != Ports().end(); piter++)
        for(std::vector<daeVariable*>::const_iterator citer = (*piter)->Variables().begin(); citer != (*piter)->Variables().end(); citer++)
            mapVariables[(*citer)->GetCanonicalName()] = *citer;

    // Insert objects from the child models (units)
    for(std::vector<daeModel*>::const_iterator miter = Models().begin(); miter != Models().end(); miter++)
        (*miter)->CollectAllVariables(mapVariables);

    // Insert objects from the model arrays
    for(std::vector<daeModelArray*>::const_iterator maiter = ModelArrays().begin(); maiter != ModelArrays().end(); maiter++)
        (*maiter)->CollectAllVariables(mapVariables);

    // Insert objects from the port arrays
    for(std::vector<daePortArray*>::const_iterator paiter = PortArrays().begin(); paiter != PortArrays().end(); paiter++)
        (*paiter)->CollectAllVariables(mapVariables);
}

void daeModel::CollectAllSTNs(std::map<dae::string, daeSTN_t*>& mapSTNs) const
{
    // Insert objects from the model
    for(std::vector<daeSTN*>::const_iterator iter = STNs().begin(); iter != STNs().end(); iter++)
    {
        if((*iter)->GetType() == eSTN)
            mapSTNs[(*iter)->GetCanonicalName()] = *iter;

        (*iter)->CollectAllSTNs(mapSTNs);
    }

    // Insert objects from the child models (units)
    for(std::vector<daeModel*>::const_iterator miter = Models().begin(); miter != Models().end(); miter++)
        (*miter)->CollectAllSTNs(mapSTNs);

    // Insert objects from the model arrays
    for(std::vector<daeModelArray*>::const_iterator maiter = ModelArrays().begin(); maiter != ModelArrays().end(); maiter++)
        (*maiter)->CollectAllSTNs(mapSTNs);
}

void daeModel::CollectAllPorts(std::map<dae::string, daePort_t*>& mapPorts) const
{
    // Insert objects from the model
    for(std::vector<daePort*>::const_iterator iter = Ports().begin(); iter != Ports().end(); iter++)
        mapPorts[(*iter)->GetCanonicalName()] = *iter;

    // Insert objects from the child models (units)
    for(std::vector<daeModel*>::const_iterator miter = Models().begin(); miter != Models().end(); miter++)
        (*miter)->CollectAllPorts(mapPorts);

    // Insert objects from the model arrays
    for(std::vector<daeModelArray*>::const_iterator maiter = ModelArrays().begin(); maiter != ModelArrays().end(); maiter++)
        (*maiter)->CollectAllPorts(mapPorts);
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
        pSTN->CollectEquationExecutionInfos(ptrarrEquationExecutionInfo);
    }

// Then, fill it with execution info in each child-model
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->CollectEquationExecutionInfosFromSTNs(ptrarrEquationExecutionInfo);
    }

// Finally, fill it with execution info in each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->CollectEquationExecutionInfosFromSTNs(ptrarrEquationExecutionInfo);
    }
}

void daeModel::CollectEquationExecutionInfosFromModels(vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;

// Fill array ptrarrEquationExecutionInfo with all execution info in the model
    dae_add_vector(m_ptrarrEquationExecutionInfos, ptrarrEquationExecutionInfo);

//	std::cout << GetName() << " Test" << std::endl;
//	dae_capacity_check(ptrarrEquationExecutionInfo);
//	std::cout << GetName() << ": m_ptrarrEquationExecutionInfos: " << m_ptrarrEquationExecutionInfos.size() << std::endl;
//	std::cout << GetName() << ": ptrarrEquationExecutionInfo: " << ptrarrEquationExecutionInfo.size() << std::endl;

// Then, fill it with execution info in each child-model
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->CollectEquationExecutionInfosFromModels(ptrarrEquationExecutionInfo);
    }

// Finally, fill it with execution info in each modelarray
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->CollectEquationExecutionInfosFromModels(ptrarrEquationExecutionInfo);
    }
}

daeBlock* daeModel::DoBlockDecomposition(void)
{
    daeBlock* pBlock;
    //vector<daeEquationExecutionInfo*> ptrarrEEIfromModels, ptrarrEEIfromSTNs;
    vector<daeEquationExecutionInfo*> ptrarrAllEquationExecutionInfosInModel;
    daeEquationExecutionInfo *pEquationExec;

    if(!m_pDataProxy)
        daeDeclareAndThrowException(exInvalidPointer);

/***********************************************************************************
    Populate vector with all existing equation execution infos
    Nota bene:
      m_ptrarrEEIfromSTNs includes only EEI from the current active states in all STNs!
************************************************************************************/
    m_ptrarrEEIfromModels.clear();
    m_ptrarrEEIfromSTNs.clear();
    CollectEquationExecutionInfosFromModels(m_ptrarrEEIfromModels);
    CollectEquationExecutionInfosFromSTNs(m_ptrarrEEIfromSTNs);

    dae_add_vector(m_ptrarrEEIfromSTNs,   ptrarrAllEquationExecutionInfosInModel);
    dae_add_vector(m_ptrarrEEIfromModels, ptrarrAllEquationExecutionInfosInModel);

    dae_capacity_check(ptrarrAllEquationExecutionInfosInModel);
    dae_capacity_check(m_ptrarrEEIfromSTNs);
    dae_capacity_check(m_ptrarrEEIfromModels);
//	std::cout << "m_ptrarrEEIfromSTNs: " << m_ptrarrEEIfromSTNs.size() << std::endl;
//	std::cout << "m_ptrarrEEIfromModels: " << m_ptrarrEEIfromModels.size() << std::endl;

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
    Build-up the block (only one at the moment)
************************************************************************************/
    size_t i, k;
    daeSTN* pSTN;
    size_t nNoEquations = ptrarrAllEquationExecutionInfosInModel.size();

    pBlock = new daeBlock;
    pBlock->SetName(string("Block N-1"));
    pBlock->SetDataProxy(m_pDataProxy.get());
    pBlock->m_nNumberOfEquations      = nNoEquations;
    pBlock->m_nTotalNumberOfVariables = m_pDataProxy->GetTotalNumberOfVariables();

// Here I reserve memory for m_ptrarrEquationExecutionInfos vector
    pBlock->m_ptrarrEquationExecutionInfos.reserve(m_ptrarrEEIfromModels.size());

    for(i = 0; i < nNoEquations; i++)
    {
        pEquationExec = ptrarrAllEquationExecutionInfosInModel[i];

        pBlock->AddVariables(pEquationExec->m_mapIndexes);
    }

////////////////////////////////////////////////////////////////////////
// BUG!!!! 30.07.2009
// A sta sa STNovima iz child modela i modelarrays????
// 31.07.2009 I corrected the code and now I use ALL STNs
////////////////////////////////////////////////////////////////////////

    m_ptrarrAllSTNs.clear();
    CollectAllSTNsAsVector(m_ptrarrAllSTNs);

    map<size_t, size_t> mapVariableIndexes;
    for(i = 0; i < m_ptrarrAllSTNs.size(); i++)
    {
        pSTN = m_ptrarrAllSTNs[i];
        if(!pSTN)
            daeDeclareAndThrowException(exInvalidPointer);

        pSTN->CollectVariableIndexes(mapVariableIndexes);
    }

    // Now add variable indexes from STNs/IFs to the block
    pBlock->AddVariables(mapVariableIndexes);

    if(pBlock->m_mapVariableIndexes.size() == pBlock->m_nTotalNumberOfVariables)
    {
    // There are no assigned variables (DOFs) - we can set the block indexes to be equal to the overall ones
        std::map<size_t, size_t>::iterator iter;
        for(iter = pBlock->m_mapVariableIndexes.begin(); iter != pBlock->m_mapVariableIndexes.end(); iter++)
            iter->second = iter->first;
    }

    // Populate the daeEquationsIndexes object from the block
    for(i = 0; i < m_ptrarrEEIfromModels.size(); i++)
    {
        pEquationExec = m_ptrarrEEIfromModels[i];
        size_t nEquationIndex = pBlock->m_EquationsIndexes.m_mapOverallIndexes_Equations.size();

        std::vector<size_t> arrOI;
        arrOI.reserve(pEquationExec->m_mapIndexes.size());
        for(std::map<size_t, size_t>::const_iterator iter = pEquationExec->m_mapIndexes.begin(); iter != pEquationExec->m_mapIndexes.end(); iter++)
            arrOI.push_back(iter->first);
        pBlock->m_EquationsIndexes.m_mapOverallIndexes_Equations.insert(std::make_pair(nEquationIndex, arrOI));
    }
//    for(i = 0; i < m_ptrarrAllSTNs.size(); i++)
//    {
//        pSTN = m_ptrarrAllSTNs[i];
//        size_t nSTNIndex = pBlock->m_EquationsIndexes.m_mapOverallIndexes_STNs.size();
//        daeEquationsIndexes ei;
//        pBlock->m_EquationsIndexes.m_mapOverallIndexes_STNs[nSTNIndex] = ei;

//        pSTN->CollectVariableIndexes(mapVariableIndexes);
//    }

    {
        // Nota bene:
        //   ptrarrAllEquationExecutionInfosInModel includes EEI from all models and from the current active states from all STNs!
        using namespace boost;

        std::size_t oi, bi;
        typedef adjacency_list<vecS,
                               vecS,
                               undirectedS,
                               property<vertex_color_t, default_color_type, property<vertex_degree_t,int> >
                              > Graph;
        typedef graph_traits<Graph>::vertex_descriptor Vertex;
        typedef graph_traits<Graph>::vertices_size_type size_type;

        size_t Neqns = m_ptrarrEEIfromModels.size();
        size_t Nvars = pBlock->m_mapVariableIndexes.size();
//        if(Nvars != Neqns)
//        {
//            std::cout << "Nvars = " << Nvars << " Neqns = " << Neqns << std::endl;
//            daeDeclareAndThrowException(exInvalidCall);
//        }

        Graph G(Nvars);
        for(int ei = 0; ei < Neqns; ei++)
        {
            daeEquationExecutionInfo* pEEI = m_ptrarrEEIfromModels[ei];
            for(std::map<size_t,size_t>::const_iterator cit = pEEI->m_mapIndexes.begin(); cit != pEEI->m_mapIndexes.end(); cit++)
            {
                oi = cit->first;

                // Block indexes in daeEquationExecutionInfos are invalid at this point - use the map from the daeBlock
                std::map<size_t,size_t>::const_iterator bi_it = pBlock->m_mapVariableIndexes.find(oi);
                if(bi_it == pBlock->m_mapVariableIndexes.end())
                    daeDeclareAndThrowException(exInvalidCall);
                bi = bi_it->second;

                std::cout << ei << "," << bi << std::endl;

                add_edge(ei, bi, G);
            }
        }

        graph_traits<Graph>::vertex_iterator ui, ui_end;

        property_map<Graph,vertex_degree_t>::type deg = get(vertex_degree, G);
        for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui)
          deg[*ui] = degree(*ui, G);

        property_map<Graph, vertex_index_t>::type index_map = get(vertex_index, G);
        std::cout << "Original ordering: " << std::endl;
        for(size_t k = 0; k < Nvars; k++)
            std::cout << index_map[k] << ", ";
        std::cout << std::endl;

        std::cout << "Original bandwidth: " << bandwidth(G) << std::endl;

        std::vector<Vertex> inv_perm(num_vertices(G));
        std::vector<size_type> perm(num_vertices(G));

        cuthill_mckee_ordering(G, inv_perm.rbegin(), get(vertex_color, G), make_degree_map(G));

        std::cout << "Cuthill McKee ordering: " << std::endl;
        for(std::vector<Vertex>::const_iterator i = inv_perm.begin(); i != inv_perm.end(); ++i)
            std::cout << index_map[*i] << ", ";
        std::cout << std::endl;

        for(size_type c = 0; c != inv_perm.size(); ++c)
            perm[index_map[inv_perm[c]]] = c;
        std::cout << std::endl;

        // This basically means to put old block index k to new position perm[k]
        std::cout << "Permutation: " << std::endl;
        for(size_t k = 0; k < Nvars; k++)
            std::cout << perm[k] << ", ";
        std::cout << std::endl;

        std::cout << "Cuthill McKee bandwidth: " << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0])) << std::endl;

        // Perform actual permutations
        std::vector<daeEquationExecutionInfo*> ptrarrEEIfromModels;
        ptrarrEEIfromModels.resize(m_ptrarrEEIfromModels.size());
        for(std::map<size_t,size_t>::iterator it = pBlock->m_mapVariableIndexes.begin(); it != pBlock->m_mapVariableIndexes.end(); it++)
        {
            std::cout << it->first << ": " << it->second << " -> " << perm[it->second] << std::endl;

            //it->second = perm[it->second];
            //ptrarrEEIfromModels[perm[it->second]] = m_ptrarrEEIfromModels[it->second];
        }
        //m_ptrarrEEIfromModels = ptrarrEEIfromModels;
    }

    return pBlock;
}

void daeModel::PopulateBlockIndexes(daeBlock* pBlock)
{
    /* If we change block indexes (for instance when doing data partitioning for c++(MPI) code generator)
     * then the following data need to be updated as well:
     *   - daeBlock::m_mapVariableIndexes (the values are block indexes)
     *   - all daeEquationExecutionInfo::m_mapIndexes (the values are block indexes)
     *   - all daeEquationExecutionInfo::m_mapJacobianExpressions (the keys are block indexes)
     *   - all adRuntimeVariableNodes/adVariableNodeArrays
     * Therefore, the function DoBlockDecomposition is split and a new one has been added InitializeStage6
     * where all these block indexes are populated. A new function has been added to daeSimulation_t
     * that can be overloaded by the users to manipulate the block indexes, if necessary.
     */

    daeSTN* pSTN;
    size_t nEquationIndex;
    vector<string> strarrErrors;
    daeEquationExecutionInfo *pEquationExec;

    if(!pBlock)
        daeDeclareAndThrowException(exInvalidPointer);

    // Fill all daeEquationExecutionInfo::m_mapIndexes/m_mapJacobianExpressions
    // with block indexes from daeBlock::m_mapVariableIndexes
    map<size_t, size_t>::iterator iter, iterIndexInBlock;

    nEquationIndex = 0;
    for(size_t k = 0; k < m_ptrarrEEIfromModels.size(); k++)
    {
        pEquationExec = m_ptrarrEEIfromModels[k];
        if(!pEquationExec)
            daeDeclareAndThrowException(exInvalidPointer);

        pEquationExec->m_nEquationIndexInBlock = nEquationIndex;
        //pEquationExec->m_pBlock = pBlock;
        pBlock->AddEquationExecutionInfo(pEquationExec);
//----------------->
    // Here we have to associate overall variable indexes in equation to corresponding indexes in the block
    // m_mapIndexes<OverallIndex, BlockIndex>
        for(iter = pEquationExec->m_mapIndexes.begin(); iter != pEquationExec->m_mapIndexes.end(); iter++)
        {
        // Try to find OverallIndex in the map of BlockIndexes
            iterIndexInBlock = pBlock->m_mapVariableIndexes.find((*iter).first);
            if(iterIndexInBlock == pBlock->m_mapVariableIndexes.end())
            {
                daeDeclareException(exInvalidCall);
                e << "Cannot find overall variable index [" << toString<size_t>((*iter).first) << "] in model " << GetCanonicalName();
                throw e;
            }
            (*iter).second = (*iterIndexInBlock).second;
        }
//------------------->
        nEquationIndex++;
    }

    pBlock->m_ptrarrSTNs = m_ptrarrAllSTNs;
    for(size_t i = 0; i < m_ptrarrAllSTNs.size(); i++)
    {
        pSTN = m_ptrarrAllSTNs[i];
        if(!pSTN)
            daeDeclareAndThrowException(exInvalidPointer);

        if(pSTN->m_ptrarrStates.size() == 0)
        {
            daeDeclareException(exInvalidCall);
            e << "Number of states is 0 in STN " << pSTN->GetCanonicalName();
            throw e;
        }

        pSTN->SetIndexesWithinBlockToEquationExecutionInfos(pBlock, nEquationIndex);
    }

// Now, after associating overall and block indexes build Jacobian expressions, if requested
// That will also associate block indexes in adRuntimeVariable/adRuntimeTimeDerivative with those in the block
// because the function Evaluate() for runtime nodes will be called for the first time here.
    for(size_t i = 0; i < m_ptrarrEEIfromModels.size(); i++)
    {
        pEquationExec = m_ptrarrEEIfromModels[i];

        if(pEquationExec->m_pEquation->m_bBuildJacobianExpressions)
            pEquationExec->BuildJacobianExpressions();
    }

    for(size_t i = 0; i < m_ptrarrAllSTNs.size(); i++)
    {
        pSTN = m_ptrarrAllSTNs[i];
        pSTN->BuildJacobianExpressions();
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

    // These are not needed anymore
    m_ptrarrEEIfromModels.clear();
    m_ptrarrEEIfromSTNs.clear();
    m_ptrarrAllSTNs.clear();
}

void daeModel::SetDefaultInitialGuesses(void)
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;
    daePortArray* pPortArray;
    daeVariable* pVariable;
    const daeVariableType_t* pVariableType;

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

    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        if(!pModel)
            daeDeclareAndThrowException(exInvalidPointer);
        pModel->SetDefaultInitialGuesses();
    }

    for(i = 0; i < m_ptrarrPortArrays.size(); i++)
    {
        pPortArray = m_ptrarrPortArrays[i];
        pPortArray->SetDefaultInitialGuesses();
    }

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->SetDefaultInitialGuesses();
    }
}

void daeModel::SetDefaultAbsoluteTolerances()
{
    size_t i;
    daeModel* pModel;
    daeVariable* pVariable;
    const daeVariableType_t* pVariableType;
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

    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        if(!pModel)
            daeDeclareAndThrowException(exInvalidPointer);
        pModel->SetDefaultAbsoluteTolerances();
    }

    for(i = 0; i < m_ptrarrPortArrays.size(); i++)
    {
        pPortArray = m_ptrarrPortArrays[i];
        pPortArray->SetDefaultAbsoluteTolerances();
    }

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->SetDefaultAbsoluteTolerances();
    }
}

size_t daeModel::GetNumberOfSTNs(void) const
{
    return m_ptrarrSTNs.size();
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
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
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
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
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
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        if(!pModel)
            daeDeclareAndThrowException(exInvalidPointer);

        pPort = pModel->FindPort(nID);
        if(pPort)
            return pPort;
    }

    return NULL;
}

daeEventPort* daeModel::FindEventPort(unsigned long nID) const
{
    size_t i;
    daeModel* pModel;
    daeEventPort* pEventPort;

    for(i = 0; i < m_ptrarrEventPorts.size(); i++)
    {
        pEventPort = m_ptrarrEventPorts[i];
        if(pEventPort && pEventPort->m_nID == nID)
            return pEventPort;
    }
// Look in child models' event ports
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        if(!pModel)
            daeDeclareAndThrowException(exInvalidPointer);

        pEventPort = pModel->FindEventPort(nID);
        if(pEventPort)
            return pEventPort;
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

void daeModel::InitializeModel(const std::string& jsonInit)
{
    size_t i;
    daeModel* pModel;
    daeModelArray* pModelArray;

    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        pModel->InitializeModel(jsonInit);
    }

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
        pModelArray->InitializeModels(jsonInit);
    }
}

void daeModel::InitializeStage1(void)
{
// Create DataProxy and propagate it to all child models
    m_pDataProxy.reset(new daeDataProxy_t);
    PropagateDataProxy(m_pDataProxy);
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
    if(m_nTotalNumberOfVariables == 0)
        daeDeclareAndThrowException(exInvalidCall);

// Initialize DataProxy
    m_pDataProxy->Initialize(this, pLog, m_nTotalNumberOfVariables);

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

daeBlock_t* daeModel::InitializeStage5(void)
{
// Do block decomposition (if requested)
    daeBlock* pBlock = DoBlockDecomposition();
    return pBlock;
}

void daeModel::InitializeStage6(daeBlock_t* ptrBlock)
{
    daeBlock* pBlock = dynamic_cast<daeBlock*>(ptrBlock);
    PopulateBlockIndexes(pBlock);
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

daeeModelType daeModel::GetModelType() const
{
    daeeEquationType eType;
    vector<daeEquationExecutionInfo*> ptrarrEEIfromModels;
    vector<daeEquationExecutionInfo*> ptrarrEEIfromSTNs;
    vector<daeEquationExecutionInfo*>::iterator c_iterator;

    CollectEquationExecutionInfosFromModels(ptrarrEEIfromModels);
    CollectEquationExecutionInfosFromSTNs(ptrarrEEIfromSTNs);

    bool bHasAlgebraic    = false;
    bool bHasExplicitDiff = false;
    bool bHasImplicitDiff = false;
    bool bIsModelDynamic  = IsModelDynamic();

    for(c_iterator = ptrarrEEIfromModels.begin(); c_iterator != ptrarrEEIfromModels.end(); c_iterator++)
    {
        eType = (*c_iterator)->GetEquationType();

        if(eType == eImplicitODE)
            bHasImplicitDiff = true;
        else if(eType == eExplicitODE)
            bHasExplicitDiff = true;
        else if(eType == eAlgebraic)
            bHasAlgebraic = true;
    }

    for(c_iterator = ptrarrEEIfromSTNs.begin(); c_iterator != ptrarrEEIfromSTNs.end(); c_iterator++)
    {
        eType = (*c_iterator)->GetEquationType();

        if(eType == eImplicitODE)
            bHasImplicitDiff = true;
        else if(eType == eExplicitODE)
            bHasExplicitDiff = true;
        else if(eType == eAlgebraic)
            bHasAlgebraic = true;
    }

    if(bHasExplicitDiff && !bHasAlgebraic && !bHasImplicitDiff)
        return eODE;
    else if(bHasAlgebraic && !bIsModelDynamic && !bHasExplicitDiff && !bHasImplicitDiff)
        return eSteadyState;
    else
        return eDAE;
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
        if(m_pDataProxy->GetVariableType(i) == cnAlgebraic)
            mi.m_nNumberOfStateVariables++;
        else if(m_pDataProxy->GetVariableType(i) == cnDifferential)
            mi.m_nNumberOfInitialConditions++;
        else if(m_pDataProxy->GetVariableType(i) == cnAssigned)
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

//	std::cout << "sizeof(daeDomain) = "							<< sizeof(daeDomain) << std::endl;
//	std::cout << "sizeof(daeParameter) = "						<< sizeof(daeParameter) << std::endl;
//	std::cout << "sizeof(daeVariable) = "						<< sizeof(daeVariable) << std::endl;
//	std::cout << "sizeof(daeEquation) = "						<< sizeof(daeEquation) << std::endl;
//	std::cout << "sizeof(daePort) = "							<< sizeof(daePort) << std::endl;
//	std::cout << "sizeof(daeEventPort) = "						<< sizeof(daeEventPort) << std::endl;
//	std::cout << "sizeof(daeModel) = "							<< sizeof(daeModel) << std::endl;
//	std::cout << "sizeof(daeSTN) = "							<< sizeof(daeSTN) << std::endl;
//	std::cout << "sizeof(daePortConnection) = "					<< sizeof(daePortConnection) << std::endl;
//	std::cout << "sizeof(*this) = "								<< sizeof(*this) << std::endl;

    dae_capacity_check(m_ptrarrDomains);
    dae_capacity_check(m_ptrarrParameters);
    dae_capacity_check(m_ptrarrVariables);
    dae_capacity_check(m_ptrarrEquations);
    dae_capacity_check(m_ptrarrSTNs);
    dae_capacity_check(m_ptrarrPorts);
    dae_capacity_check(m_ptrarrEventPorts);
    dae_capacity_check(m_ptrarrComponents);
    dae_capacity_check(m_ptrarrOnEventActions);
    dae_capacity_check(m_ptrarrOnConditionActions);
    dae_capacity_check(m_ptrarrEquationExecutionInfos);
    dae_capacity_check(m_ptrarrComponentArrays);
    dae_capacity_check(m_ptrarrPortArrays);
    dae_capacity_check(m_ptrarrPortConnections);
    dae_capacity_check(m_ptrarrEventPortConnections);

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
    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
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
    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
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

    for(i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pModel = m_ptrarrComponents[i];
        if(!pModel)
            daeDeclareAndThrowException(exInvalidPointer);
        nNoEqns += pModel->GetTotalNumberOfEquations();
    }

    for(i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pModelArray = m_ptrarrComponentArrays[i];
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

const std::vector<daePort*>& daeModel::Ports() const
{
    return m_ptrarrPorts;
}

const std::vector<daeEventPort*>& daeModel::EventPorts() const
{
    return m_ptrarrEventPorts;
}

const std::vector<daeModel*>& daeModel::Models() const
{
    return m_ptrarrComponents;
}

const std::vector<daeDomain*>& daeModel::Domains() const
{
    return m_ptrarrDomains;
}

const std::vector<daeVariable*>& daeModel::Variables() const
{
    return m_ptrarrVariables;
}

const std::vector<daeParameter*>& daeModel::Parameters() const
{
    return m_ptrarrParameters;
}

const std::vector<daeEquation*>& daeModel::Equations() const
{
    return m_ptrarrEquations;
}

const std::vector<daeSTN*>& daeModel::STNs() const
{
    return m_ptrarrSTNs;
}

const std::vector<daeOnEventActions*>& daeModel::OnEventActions() const
{
    return m_ptrarrOnEventActions;
}

const std::vector<daeOnConditionActions*>& daeModel::OnConditionActions() const
{
    return m_ptrarrOnConditionActions;
}

const std::vector<daePortConnection*>& daeModel::PortConnections() const
{
    return m_ptrarrPortConnections;
}

const std::vector<daeEventPortConnection*>& daeModel::EventPortConnections() const
{
    return m_ptrarrEventPortConnections;
}

const std::vector<daePortArray*>& daeModel::PortArrays() const
{
    return m_ptrarrPortArrays;
}

const std::vector<daeModelArray*>& daeModel::ModelArrays() const
{
    return m_ptrarrComponentArrays;
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
    for(size_t i = 0; i < m_ptrarrComponents.size(); i++)
        ptrarrModels.push_back(m_ptrarrComponents[i]);
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

void daeModel::GetEventPortConnections(vector<daeEventPortConnection_t*>& ptrarrEventPortConnections)
{
    ptrarrEventPortConnections.clear();
    for(size_t i = 0; i < m_ptrarrEventPortConnections.size(); i++)
        ptrarrEventPortConnections.push_back(m_ptrarrEventPortConnections[i]);
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
    for(size_t i = 0; i < m_ptrarrComponentArrays.size(); i++)
        ptrarrModelArrays.push_back(m_ptrarrComponentArrays[i]);
}

void daeModel::GetEquationExecutionInfos(vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfos)
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

daeObject_t* daeModel::FindObjectFromRelativeName(string& strRelativeName)
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

daeObject_t* daeModel::FindObjectFromRelativeName(vector<string>& strarrNames)
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

daeObject_t* daeModel::FindObject(string& strName)
{
    daeObject_t* pObject;

    pObject = FindModel(strName);
    if(pObject)
        return pObject;

    pObject = FindPort(strName);
    if(pObject)
        return pObject;

    pObject = FindEventPort(strName);
    if(pObject)
        return pObject;

    pObject = FindParameter(strName);
    if(pObject)
        return pObject;

    pObject = FindDomain(strName);
    if(pObject)
        return pObject;

    pObject = FindVariable(strName);
    if(pObject)
        return pObject;

    pObject = FindSTN(strName);
    if(pObject)
        return pObject;

    return NULL;
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

daeEventPort_t* daeModel::FindEventPort(string& strName)
{
    daeEventPort* pObject;
    for(size_t i = 0; i < m_ptrarrEventPorts.size(); i++)
    {
        pObject = m_ptrarrEventPorts[i];
        if(pObject->m_strShortName == strName)
            return pObject;
    }
    return NULL;
}

daeModel_t* daeModel::FindModel(string& strName)
{
    daeModel* pObject;
    for(size_t i = 0; i < m_ptrarrComponents.size(); i++)
    {
        pObject = m_ptrarrComponents[i];
        if(pObject->m_strShortName == strName)
            return pObject;
    }
    return NULL;
}

daeSTN_t* daeModel::FindSTN(string& strName)
{
    daeSTN* pObject;
    for(size_t i = 0; i < m_ptrarrSTNs.size(); i++)
    {
        pObject = m_ptrarrSTNs[i];
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
    for(size_t i = 0; i < m_ptrarrComponentArrays.size(); i++)
    {
        pObject = m_ptrarrComponentArrays[i];
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

/*********************************************************************************************
    daeGetVariableAndIndexesFromNode
**********************************************************************************************/
void daeGetVariableAndIndexesFromNode(adouble& a, daeVariable** variable, std::vector<size_t>& narrDomainIndexes)
{
    size_t i, n;

    adSetupVariableNode* node = dynamic_cast<adSetupVariableNode*>(a.node.get());
    if(!node)
    {
        daeDeclareException(exInvalidCall);
        e << "The first argument of daeGetVariableAndIndexesFromNode() function "
          << "can only be a variable or a distributed variable with constant indexes";
        throw e;
    }

    *variable = node->m_pVariable;
    if(!(*variable))
        daeDeclareAndThrowException(exInvalidPointer)

    n = node->m_arrDomains.size();
    for(i = 0; i < n; i++)
    {
    // Only constant indexes are supported here!!!
        if(node->m_arrDomains[i].m_eType == eConstantIndex)
        {
            if(node->m_arrDomains[i].m_nIndex == ULONG_MAX)
                daeDeclareAndThrowException(exInvalidCall);

            narrDomainIndexes.push_back(node->m_arrDomains[i].m_nIndex);
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "The first argument of daeGetVariableAndIndexesFromNode() function "
              << "can only be a variable or a distributed variable with constant indexes";
            throw e;
        }
    }
}


}
}
