#ifndef DAE_DOCSTRINGS_H
#define DAE_DOCSTRINGS_H

/******************************************************************
                            daeModel
******************************************************************/
/*
const char* DOCSTR_Template = ". \n\n"
    "*Arguments:* \n"
    " -  (string): . \n"
    " -  (float): . \n"
    " -  (object): . \n"
    "*Returns:* \n"
    "    object \n\n"
    "*Raises:* \n"
    "    RuntimeError \n";

Code snippet:
    "    documentation:: \n\n"
    "        def any(iterable): \n"
    "            for element in iterable: \n"
    "                if element: \n"
    "                    return True \n"
    "            return False \n\n";
*/
const char* DOCSTR_global_daeGetConfig = ""; 
const char* DOCSTR_global_daeVersion = "";
const char* DOCSTR_global_daeVersionMajor = "";
const char* DOCSTR_global_daeVersionMinor = "";
const char* DOCSTR_global_daeVersionBuild = "";

const char* DOCSTR_daeVariableWrapper_ = "";

const char* DOCSTR_daeConfig_ = "";

const char* DOCSTR_adouble = "Class adouble operates on values/derivatives of domains, parameters and variables. "
         "It supports basic mathematical operators (+, -, *, /, **), comparison operators (<, <=, >, >=, ==, !=), " 
         "and logical operators (and, or, not). Operands can be instances of adouble or float values.";
    const char* DOCSTR_adouble_init = "";
    const char* DOCSTR_adouble_Value = "Value";
    const char* DOCSTR_adouble_Derivative = "Derivative";
    const char* DOCSTR_adouble_GatherInfo = "Internally used by the framework.";
    const char* DOCSTR_adouble_Node = "Contains the equation evaluation node.";

const char* DOCSTR_adouble_array = "Class adouble_array operates on arrays of values/derivatives of domains, parameters and variables. "
         "It supports basic mathematical operators (+, -, *, /, **). Operands can be instances of adouble_array, adouble or float values.";
    const char* DOCSTR_adouble_array_init = "";
    const char* DOCSTR_adouble_array_GatherInfo = "Used internally by the framework.";
    const char* DOCSTR_adouble_array_Node = "Contains the equation evaluation node.";
    const char* DOCSTR_adouble_array_Resize = "Resizes the adouble_array object to the new size.";
    const char* DOCSTR_adouble_array_len = "Returns the size of the adouble_array object.";
    const char* DOCSTR_adouble_array_getitem = "Gets an adouble object at the specified integer index.";
    const char* DOCSTR_adouble_array_setitem = "Sets an adouble object at the specified integer index.";
    const char* DOCSTR_adouble_array_items = "Returns an iterator over adouble items in adouble_array object.";
    const char* DOCSTR_adouble_array_call = "Returns an adouble object at the specified daeDomainIndex (used in equation residals)";
    const char* DOCSTR_adouble_array_FromList = "Returns adouble_array object from the list of adouble objects";
    const char* DOCSTR_adouble_array_FromNumpyArray = "Returns adouble_array object from the ndarray of adouble objects";

const char* DOCSTR_daeCondition = "";
    const char* DOCSTR_daeCondition_EventTolerance = "";
    const char* DOCSTR_daeCondition_SetupNode = "";
    const char* DOCSTR_daeCondition_RuntimeNode = "";
    const char* DOCSTR_daeCondition_Expressions = "";
        
const char* DOCSTR_dt = "";  
const char* DOCSTR_d = "";  
const char* DOCSTR_Time = "";  
const char* DOCSTR_InverseTimeStep = "";
const char* DOCSTR_Constant_c = "";
const char* DOCSTR_Constant_q = "";  
const char* DOCSTR_Array = "";  

const char* DOCSTR_Sum = "";  
const char* DOCSTR_Product = "";  
const char* DOCSTR_Integral = "";  
const char* DOCSTR_Min_adarr = "";  
const char* DOCSTR_Max_adarr = "";  
const char* DOCSTR_Average = "";  

const char* DOCSTR_daeVariableType = "";
    const char* DOCSTR_daeVariableType_init = "";
    const char* DOCSTR_daeVariableType_Name = "";
    const char* DOCSTR_daeVariableType_Units = "";
    const char* DOCSTR_daeVariableType_LowerBound = "";
    const char* DOCSTR_daeVariableType_UpperBound = "";
    const char* DOCSTR_daeVariableType_InitialGuess = "";
    const char* DOCSTR_daeVariableType_AbsoluteTolerance = "";

const char* DOCSTR_daeFMI2Object_t = "";
    const char* DOCSTR_daeFMI2Object_t_reference = "";
    const char* DOCSTR_daeFMI2Object_t_name = "";
    const char* DOCSTR_daeFMI2Object_t_type = "";
    const char* DOCSTR_daeFMI2Object_t_description = "";
    const char* DOCSTR_daeFMI2Object_t_units = "";
    const char* DOCSTR_daeFMI2Object_t_indexes = "";

const char* DOCSTR_daeObject = "";  
    const char* DOCSTR_daeObject_ID = "";  
    const char* DOCSTR_daeObject_Version = "";  
    const char* DOCSTR_daeObject_Library = "";  
    const char* DOCSTR_daeObject_Name = "";  
    const char* DOCSTR_daeObject_Description = "";  
    const char* DOCSTR_daeObject_CanonicalName = "";  
    const char* DOCSTR_daeObject_Model = "";  
    const char* DOCSTR_daeObject_GetNameRelativeToParentModel = "";  
    const char* DOCSTR_daeObject_GetStrippedName = "";  
    const char* DOCSTR_daeObject_GetStrippedNameRelativeToParentModel = "";  

const char* DOCSTR_global_daeIsValidObjectName = "";  
const char* DOCSTR_global_daeGetRelativeName1 = "";  
const char* DOCSTR_global_daeGetRelativeName2 = "";  
const char* DOCSTR_global_daeGetStrippedRelativeName = "";  

const char* DOCSTR_daeDomainIndex = "";  
    const char* DOCSTR_daeDomainIndex_init1 = "";  
    const char* DOCSTR_daeDomainIndex_init2 = "";  
    const char* DOCSTR_daeDomainIndex_init3 = "";  
    const char* DOCSTR_daeDomainIndex_init4 = "";  
    const char* DOCSTR_daeDomainIndex_Type = "";  
    const char* DOCSTR_daeDomainIndex_Index = "";  
    const char* DOCSTR_daeDomainIndex_DEDI = "";  
    const char* DOCSTR_daeDomainIndex_Increment = "";  

const char* DOCSTR_daeIndexRange = "";  
    const char* DOCSTR_daeIndexRange_init1 = "";  
    const char* DOCSTR_daeIndexRange_init2 = "";  
    const char* DOCSTR_daeIndexRange_init3 = "";  
    const char* DOCSTR_daeIndexRange_NoPoints = "";  
    const char* DOCSTR_daeIndexRange_Domain = "";  
    const char* DOCSTR_daeIndexRange_Type = "";  
    const char* DOCSTR_daeIndexRange_StartIndex = "";  
    const char* DOCSTR_daeIndexRange_EndIndex = "";  
    const char* DOCSTR_daeIndexRange_Step = "";  

const char* DOCSTR_daeArrayRange = "";  
    const char* DOCSTR_daeArrayRange_ = "";  
    const char* DOCSTR_daeArrayRange_init1 = "";  
    const char* DOCSTR_daeArrayRange_init2 = "";  
    const char* DOCSTR_daeArrayRange_NoPoints = "";  
    const char* DOCSTR_daeArrayRange_Type = "";  
    const char* DOCSTR_daeArrayRange_Range = "";  
    const char* DOCSTR_daeArrayRange_DomainIndex = "";  

const char* DOCSTR_daeDomain = "";  
    const char* DOCSTR_daeDomain_init = "";  
    const char* DOCSTR_daeDomain_init1 = "";  
    const char* DOCSTR_daeDomain_init2 = "";  
    const char* DOCSTR_daeDomain_Type = "";  
    const char* DOCSTR_daeDomain_NumberOfIntervals = "";  
    const char* DOCSTR_daeDomain_NumberOfPoints = "";  
    const char* DOCSTR_daeDomain_DiscretizationMethod = "";  
    const char* DOCSTR_daeDomain_DiscretizationOrder = "";  
    const char* DOCSTR_daeDomain_LowerBound = "";  
    const char* DOCSTR_daeDomain_UpperBound = "";  
    const char* DOCSTR_daeDomain_Units = "";  
    const char* DOCSTR_daeDomain_npyPoints = "";  
    const char* DOCSTR_daeDomain_Coordinates = "";
    const char* DOCSTR_daeDomain_Points = "";  
    const char* DOCSTR_daeDomain_CreateArray = "";  
    const char* DOCSTR_daeDomain_CreateStructuredGrid = "";
    const char* DOCSTR_daeDomain_CreateUnstructuredGrid = "";
    const char* DOCSTR_daeDomain_getitem = "";
    const char* DOCSTR_daeDomain_call = "";  
    const char* DOCSTR_daeDomain_array = "";  

const char* DOCSTR_daeDEDI = "";  
    const char* DOCSTR_daeDEDI_Domain = "";  
    const char* DOCSTR_daeDEDI_DomainPoints = "";  
    const char* DOCSTR_daeDEDI_DomainBounds = "";  
    const char* DOCSTR_daeDEDI_call = "";  

const char* DOCSTR_daeParameter = "";  
    const char* DOCSTR_daeParameter_init = "";  
    const char* DOCSTR_daeParameter_init1 = "";  
    const char* DOCSTR_daeParameter_init2 = "";  
    const char* DOCSTR_daeParameter_Units = "";  
    const char* DOCSTR_daeParameter_Domains = "";  
    const char* DOCSTR_daeParameter_ReportingOn = "";  
    const char* DOCSTR_daeParameter_npyValues = "";  
    const char* DOCSTR_daeParameter_NumberOfPoints = "";  
    const char* DOCSTR_daeParameter_DistributeOnDomain = "";  
    const char* DOCSTR_daeParameter_GetDomainsIndexesMap = "";  

const char* DOCSTR_daeVariable = "";  
    const char* DOCSTR_daeVariable_init = "";  
    const char* DOCSTR_daeVariable_init1 = "";  
    const char* DOCSTR_daeVariable_init2 = "";  
    const char* DOCSTR_daeVariable_Domains = "";
    const char* DOCSTR_daeVariable_Type = "";
    const char* DOCSTR_daeVariable_VariableType = "";
    const char* DOCSTR_daeVariable_ReportingOn = "";  
    const char* DOCSTR_daeVariable_OverallIndex = "";  
    const char* DOCSTR_daeVariable_NumberOfPoints = "";  
    const char* DOCSTR_daeVariable_npyValues = "";  
    const char* DOCSTR_daeVariable_npyTimeDerivatives = "";
    const char* DOCSTR_daeVariable_npyIDs = "";
    const char* DOCSTR_daeVariable_npyGatheredIDs = "";
    const char* DOCSTR_daeVariable_DistributeOnDomain = "";  
    const char* DOCSTR_daeVariable_GetDomainIndexesMap = "";  

const char* DOCSTR_daeModelExportContext = "";  
    const char* DOCSTR_daeModelExportContext_init = "";  
    const char* DOCSTR_daeModelExportContext_PythonIndentLevel = "";  
    const char* DOCSTR_daeModelExportContext_ExportDefinition = "";  

const char* DOCSTR_daePort = "";  
    const char* DOCSTR_daePort_init = "";  
    const char* DOCSTR_daePort_Type = "";  
    const char* DOCSTR_daePort_Domains = "";  
    const char* DOCSTR_daePort_Parameters = "";  
    const char* DOCSTR_daePort_Variables = "";  
    const char* DOCSTR_daePort_SetReportingOn = "";  
    const char* DOCSTR_daePort_Export = "";  
    
const char* DOCSTR_daeEventPort = "";  
    const char* DOCSTR_daeEventPort_init = "";  
    const char* DOCSTR_daeEventPort_Type = "";  
    const char* DOCSTR_daeEventPort_EventData = "";  
    const char* DOCSTR_daeEventPort_RecordEvents = "";  
    const char* DOCSTR_daeEventPort_Events = "";  
    const char* DOCSTR_daeEventPort_call = "";  
    const char* DOCSTR_daeEventPort_SendEvent = "";  
    const char* DOCSTR_daeEventPort_ReceiveEvent = "";  

const char* DOCSTR_daeAction = "";  
    const char* DOCSTR_daeAction_init = "";  
    const char* DOCSTR_daeAction_Type = "";  
    const char* DOCSTR_daeAction_STN = "";  
    const char* DOCSTR_daeAction_StateTo = "";  
    const char* DOCSTR_daeAction_SendEventPort = "";  
    const char* DOCSTR_daeAction_VariableWrapper = "";  
    const char* DOCSTR_daeAction_SetupNode = "";  
    const char* DOCSTR_daeAction_RuntimeNode = "";  
    const char* DOCSTR_daeAction_Execute = "";  

const char* DOCSTR_daeOptimizationVariable = "";  
    const char* DOCSTR_daeOptimizationVariable_init = "";  
    const char* DOCSTR_daeOptimizationVariable_Name = "";  
    const char* DOCSTR_daeOptimizationVariable_Type = "";  
    const char* DOCSTR_daeOptimizationVariable_Value = "";  
    const char* DOCSTR_daeOptimizationVariable_LowerBound = "";  
    const char* DOCSTR_daeOptimizationVariable_UpperBound = "";  
    const char* DOCSTR_daeOptimizationVariable_StartingPoint = "";  

const char* DOCSTR_daeObjectiveFunction = "";  
    const char* DOCSTR_daeObjectiveFunction_init = "";  
    const char* DOCSTR_daeObjectiveFunction_Name = "";  
    const char* DOCSTR_daeObjectiveFunction_Residual = "";  
    const char* DOCSTR_daeObjectiveFunction_Value = "";  
    const char* DOCSTR_daeObjectiveFunction_Gradients = "";  
    const char* DOCSTR_daeObjectiveFunction_AbsTolerance = "";  

const char* DOCSTR_daeOptimizationConstraint = "";  
    const char* DOCSTR_daeOptimizationConstraint_init = "";  
    const char* DOCSTR_daeOptimizationConstraint_Name = "";  
    const char* DOCSTR_daeOptimizationConstraint_Type = "";  
    const char* DOCSTR_daeOptimizationConstraint_Residual = "";  
    const char* DOCSTR_daeOptimizationConstraint_Value = "";  
    const char* DOCSTR_daeOptimizationConstraint_Gradients = "";  
    const char* DOCSTR_daeOptimizationConstraint_AbsTolerance = "";  

const char* DOCSTR_daeMeasuredVariable = "";  
    const char* DOCSTR_daeMeasuredVariable_init = "";  
    const char* DOCSTR_daeMeasuredVariable_Name = "";  
    const char* DOCSTR_daeMeasuredVariable_Residual = "";  
    const char* DOCSTR_daeMeasuredVariable_Value = "";  
    const char* DOCSTR_daeMeasuredVariable_Gradients = "";  
    const char* DOCSTR_daeMeasuredVariable_AbsTolerance = "";  
    
const char* DOCSTR_daeOnEventActions = "";
    const char* DOCSTR_daeOnEventActions_EventPort = "";
    const char* DOCSTR_daeOnEventActions_Actions = "";
    const char* DOCSTR_daeOnEventActions_UserDefinedActions = "";
    const char* DOCSTR_daeOnEventActions_Execute = "";
    
const char* DOCSTR_daeOnConditionActions = "";  
    const char* DOCSTR_daeOnConditionActions_Condition = "";  
    const char* DOCSTR_daeOnConditionActions_Actions = "";  
    const char* DOCSTR_daeOnConditionActions_UserDefinedActions = "";  
    const char* DOCSTR_daeOnConditionActions_Execute = "";  

const char* DOCSTR_daeExecutionContext = "";

const char* DOCSTR_daeModel = "Base model class.";
    const char* DOCSTR_daeModel_init = "Constructor...";
    const char* DOCSTR_daeModel_Domains = "A list of domains in the model.";
    const char* DOCSTR_daeModel_Parameters = "A list of parameters in the model.";
    const char* DOCSTR_daeModel_Variables = "A list of variables in the model.";
    const char* DOCSTR_daeModel_Equations = "A list of equations in the model.";
    const char* DOCSTR_daeModel_Ports = "A list of ports in the model.";
    const char* DOCSTR_daeModel_EventPorts = "A list of event ports in the model.";
    const char* DOCSTR_daeModel_OnEventActions = "A list of OnEvent actions in the model.";
    const char* DOCSTR_daeModel_OnConditionActions = "A list of OnCondition actions in the model.";
    const char* DOCSTR_daeModel_STNs = "A list of state transition networks in the model.";
    const char* DOCSTR_daeModel_Components = "A list of components in the model.";
    const char* DOCSTR_daeModel_PortArrays = "A list of arrays of ports in the model.";
    const char* DOCSTR_daeModel_ComponentArrays = "A list of arrays of components in the model.";
    const char* DOCSTR_daeModel_PortConnections = "A list of port connections in the model.";
    const char* DOCSTR_daeModel_EventPortConnections = "A list of event port connections in the model.";
    const char* DOCSTR_daeModel_InitialConditionMode = "A mode used to calculate initial conditions ...";
    const char* DOCSTR_daeModel_IsModelDynamic = "Boolean flag that determines whether the model is synamic or steady-state.";
    const char* DOCSTR_daeModel_ModelType = "A type of the model ().";
    const char* DOCSTR_daeModel_OverallIndex_BlockIndex_VariableNameMap = "";

    const char* DOCSTR_daeModel_CreateEquation = "Creates a new equation. Used to add equations to models or "
                                                 "states in state transition networks";
    const char* DOCSTR_daeModel_CreateFiniteElementEquation = "";
    const char* DOCSTR_daeModel_DeclareEquations = "User-defined function where all model equations ans state transition networks "
                                                   "are declared. Must be always implemented in derived classes.";
    const char* DOCSTR_daeModel_UpdateEquations = "";
    const char* DOCSTR_daeModel_InitializeModel = "";
    const char* DOCSTR_daeModel_ConnectPorts = "Connects two ports.";
    const char* DOCSTR_daeModel_ConnectEventPorts = "Connects two event ports.";
    const char* DOCSTR_daeModel_SetReportingOn = "Switches the reporting of the model variables/parameters to the data reporter on or off.";
    const char* DOCSTR_daeModel_IF = "Creates a reversible state transition network and adds the first state.";
    const char* DOCSTR_daeModel_ELSE_IF = "Adds a new state to a reversible state transition network.";
    const char* DOCSTR_daeModel_ELSE = "Adds the last state to a reversible state transition network.";
    const char* DOCSTR_daeModel_END_IF = "Finalises a reversible state transition network.";
    
    const char* DOCSTR_daeModel_STN = ".";
    const char* DOCSTR_daeModel_STATE = ".";
    const char* DOCSTR_daeModel_END_STN = ".";
    const char* DOCSTR_daeModel_SWITCH_TO = ".";
    const char* DOCSTR_daeModel_ON_CONDITION = ".";
    const char* DOCSTR_daeModel_ON_EVENT = ".";
    const char* DOCSTR_daeModel_GetCoSimulationInterface = "";
    const char* DOCSTR_daeModel_GetFMIInterface = "";
    const char* DOCSTR_daeModel_SaveModelReport = ".";
    const char* DOCSTR_daeModel_SaveRuntimeModelReport = ".";
    const char* DOCSTR_daeModel_ExportObjects = ".";
    const char* DOCSTR_daeModel_Export = ".";

const char* DOCSTR_daeMatrix = "";

const char* DOCSTR_daeArray = "";

const char* DOCSTR_daeSparseMatrixRowIterator = "";

const char* DOCSTR_daeSparseMatrixRowIterator_python = "";

const char* DOCSTR_daeFiniteElementModel = "";
    const char* DOCSTR_daeFiniteElementModel_UpdateEquations = "";

const char* DOCSTR_daeFiniteElementObject = "";
    const char* DOCSTR_daeFiniteElementObject_AssembleSystem = "";
    const char* DOCSTR_daeFiniteElementObject_ReAssembleSystem = "";
    const char* DOCSTR_daeFiniteElementObject_NeedsReAssembling = "";
    const char* DOCSTR_daeFiniteElementObject_RowIndices = "";
    const char* DOCSTR_daeFiniteElementObject_SystemMatrix = "";
    const char* DOCSTR_daeFiniteElementObject_SystemMatrix_dt = "";
    const char* DOCSTR_daeFiniteElementObject_SystemRHS = "";
    const char* DOCSTR_daeFiniteElementObject_GetObjectInfo = "";
    const char* DOCSTR_daeFiniteElementObject_GetNumberOfPointsInDomainOmega = "";

const char* DOCSTR_daeEquationExecutionInfo = "";  
    const char* DOCSTR_daeEquationExecutionInfo_Node = "";  
    const char* DOCSTR_daeEquationExecutionInfo_Name = "";  
    const char* DOCSTR_daeEquationExecutionInfo_VariableIndexes = "";  
    const char* DOCSTR_daeEquationExecutionInfo_EquationType = "";  
    const char* DOCSTR_daeEquationExecutionInfo_JacobianExpressions = "";
    const char* DOCSTR_daeEquationExecutionInfo_EquationIndex = "";
    const char* DOCSTR_daeEquationExecutionInfo_Equation = "";  

const char* DOCSTR_daeFiniteElementEquationExecutionInfo = "";
    const char* DOCSTR_daeFiniteElementEquationExecutionInfo_Node = "";

const char* DOCSTR_daeEquation = "";  
    const char* DOCSTR_daeEquation_init = "";  
    const char* DOCSTR_daeEquation_Residual = "";  
    const char* DOCSTR_daeEquation_Scaling = "";  
    const char* DOCSTR_daeEquation_BuildJacobianExpressions = "";
    const char* DOCSTR_daeEquation_CheckUnitConsistency = "";
    const char* DOCSTR_daeEquation_EquationExecutionInfos = "";  
    const char* DOCSTR_daeEquation_DistributedEquationDomainInfos = "";  
    const char* DOCSTR_daeEquation_EquationType = "";  
    const char* DOCSTR_daeEquation_DistributeOnDomain1 = "";  
    const char* DOCSTR_daeEquation_DistributeOnDomain2 = "";  

const char* DOCSTR_daeFiniteElementEquation = "";
    const char* DOCSTR_daeFiniteElementEquation_EquationExecutionInfos = "";

const char* DOCSTR_daePortConnection = "";  
    const char* DOCSTR_daePortConnection_PortFrom = "";  
    const char* DOCSTR_daePortConnection_PortTo = "";  
    const char* DOCSTR_daePortConnection_Equations = "";  

const char* DOCSTR_daeEventPortConnection = "";  
    const char* DOCSTR_daeEventPortConnection_PortFrom = "";  
    const char* DOCSTR_daeEventPortConnection_PortTo = "";  

const char* DOCSTR_daeState = "";  
    const char* DOCSTR_daeState_init = "";  
    const char* DOCSTR_daeState_Equations = "";  
    const char* DOCSTR_daeState_StateTransitions = "";  
    const char* DOCSTR_daeState_NestedSTNs = "";  
    const char* DOCSTR_daeState_OnConditionActions = "";  
    const char* DOCSTR_daeState_OnEventActions = "";

const char* DOCSTR_daeSTN = "";  
    const char* DOCSTR_daeSTN_init = "";  
    const char* DOCSTR_daeSTN_ActiveState = "";  
    const char* DOCSTR_daeSTN_States = "";  
    const char* DOCSTR_daeSTN_Type = "";

const char* DOCSTR_daeIF = "";  
    const char* DOCSTR_daeIF_init = "";  

const char* DOCSTR_daeScalarExternalFunction = "";  
    const char* DOCSTR_daeScalarExternalFunction_init = "";  
    const char* DOCSTR_daeScalarExternalFunction_Name = "";  
    const char* DOCSTR_daeScalarExternalFunction_Calculate = "";  
    const char* DOCSTR_daeScalarExternalFunction_call = "";  

const char* DOCSTR_daeVectorExternalFunction = "";  
    const char* DOCSTR_daeVectorExternalFunction_init = "";  
    const char* DOCSTR_daeVectorExternalFunction_Name = "";  
    const char* DOCSTR_daeVectorExternalFunction_Calculate = "";  
    const char* DOCSTR_daeVectorExternalFunction_call = "";  

const char* DOCSTR_daeLog_t = "";  
    const char* DOCSTR_daeLog_t_init = "";
    const char* DOCSTR_daeLog_t_Name = "";
    const char* DOCSTR_daeLog_t_Enabled = "";  
    const char* DOCSTR_daeLog_t_PrintProgress = "";  
    const char* DOCSTR_daeLog_t_Indent = "";  
    const char* DOCSTR_daeLog_t_Progress = "";  
    const char* DOCSTR_daeLog_t_IndentString = "";  
    const char* DOCSTR_daeLog_t_PercentageDone = "";  
    const char* DOCSTR_daeLog_t_ETA = "";  
    const char* DOCSTR_daeLog_t_Message = "";  
    const char* DOCSTR_daeLog_t_JoinMessages = "";  
    const char* DOCSTR_daeLog_t_IncreaseIndent = "";  
    const char* DOCSTR_daeLog_t_DecreaseIndent = "";  

const char* DOCSTR_daeBaseLog = "";  
    const char* DOCSTR_daeBaseLog_init = "";
    const char* DOCSTR_daeBaseLog_Message = "";  
    const char* DOCSTR_daeBaseLog_SetProgress = "";  
    const char* DOCSTR_daeBaseLog_IncreaseIndent = "";  
    const char* DOCSTR_daeBaseLog_DecreaseIndent = "";  

const char* DOCSTR_daeDelegateLog = "";
    const char* DOCSTR_daeDelegateLog_init = "";
    const char* DOCSTR_daeDelegateLog_Message = "";
    const char* DOCSTR_daeDelegateLog_AddLog = "";
    const char* DOCSTR_daeDelegateLog_Logs = "";
    
const char* DOCSTR_daeFileLog = "";  
    const char* DOCSTR_daeFileLog_init = "";  
    const char* DOCSTR_daeFileLog_Filename = "";
    const char* DOCSTR_daeFileLog_Message = "";  

const char* DOCSTR_daeStdOutLog = "";  
    const char* DOCSTR_daeStdOutLog_init = "";  
    const char* DOCSTR_daeStdOutLog_Message = "";  

const char* DOCSTR_daeTCPIPLog = "";  
    const char* DOCSTR_daeTCPIPLog_init = "";  
    const char* DOCSTR_daeTCPIPLog_Connect = "";  
    const char* DOCSTR_daeTCPIPLog_Disconnect = "";  
    const char* DOCSTR_daeTCPIPLog_IsConnected = "";  
    const char* DOCSTR_daeTCPIPLog_Message = "";  

const char* DOCSTR_daeTCPIPLogServer = "";  
    const char* DOCSTR_daeTCPIPLogServer_init = "";  
    const char* DOCSTR_daeTCPIPLogServer_Start = "";  
    const char* DOCSTR_daeTCPIPLogServer_Stop = "";  
    const char* DOCSTR_daeTCPIPLogServer_MessageReceived = "";  


#endif
