#ifndef DAE_DOCSTRINGS_H
#define DAE_DOCSTRINGS_H

const char* DOCSTR_daeDataReporter_t = "";
    const char* DOCSTR_daeDataReporter_t_Name = "";
    const char* DOCSTR_daeDataReporter_t_Connect = "";
    const char* DOCSTR_daeDataReporter_t_Disconnect = "";
    const char* DOCSTR_daeDataReporter_t_IsConnected = "";
    const char* DOCSTR_daeDataReporter_t_StartRegistration = "";
    const char* DOCSTR_daeDataReporter_t_RegisterDomain = "";
    const char* DOCSTR_daeDataReporter_t_RegisterVariable = "";
    const char* DOCSTR_daeDataReporter_t_EndRegistration = "";
    const char* DOCSTR_daeDataReporter_t_StartNewResultSet = "";
    const char* DOCSTR_daeDataReporter_t_SendVariable = "";
    const char* DOCSTR_daeDataReporter_t_EndOfData = "";

const char* DOCSTR_daeDataReporterDomain = "";
    const char* DOCSTR_daeDataReporterDomain_init = "";
    const char* DOCSTR_daeDataReporterDomain_Name = "";
    const char* DOCSTR_daeDataReporterDomain_Type = "";
    const char* DOCSTR_daeDataReporterDomain_NumberOfPoints = "";
    const char* DOCSTR_daeDataReporterDomain_Points = "";
    const char* DOCSTR_daeDataReporterDomain_getitem = "";
    const char* DOCSTR_daeDataReporterDomain_setitem = "";

const char* DOCSTR_daeDataReporterVariable = "";
    const char* DOCSTR_daeDataReporterVariable_init = "";
    const char* DOCSTR_daeDataReporterVariable_Name = "";
    const char* DOCSTR_daeDataReporterVariable_NumberOfPoints = "";
    const char* DOCSTR_daeDataReporterVariable_NumberOfDomains = "";
    const char* DOCSTR_daeDataReporterVariable_Domains = "";
    const char* DOCSTR_daeDataReporterVariable_AddDomain = "";

const char* DOCSTR_daeDataReporterVariableValue = "";
    const char* DOCSTR_daeDataReporterVariableValue_init = "";
    const char* DOCSTR_daeDataReporterVariableValue_Name = "";
    const char* DOCSTR_daeDataReporterVariableValue_NumberOfPoints = "";
    const char* DOCSTR_daeDataReporterVariableValue_Values = "";
    const char* DOCSTR_daeDataReporterVariableValue_getitem = "";
    const char* DOCSTR_daeDataReporterVariableValue_setitem = "";

const char* DOCSTR_daeDataOut = "";
    const char* DOCSTR_daeDataOut_SendVariable = "";

const char* DOCSTR_daeDataReporterFunctor = "";
    const char* DOCSTR_daeDataReporterFunctor_SendVariable = "";

const char* DOCSTR_daeBlackHoleDataReporter = "";

const char* DOCSTR_daeDelegateDataReporter = "";
    const char* DOCSTR_daeDelegateDataReporter_AddDataReporter = "";
    const char* DOCSTR_daeDelegateDataReporter_DataReporters = "";

const char* DOCSTR_daeDataReporterLocal = "";
    const char* DOCSTR_daeDataReporterLocal_Process = "";

const char* DOCSTR_daeNoOpDataReporter = "";

const char* DOCSTR_daeDataReporterFile = "";
    const char* DOCSTR_daeDataReporterFile_WriteDataToFile = "";

const char* DOCSTR_daeTEXTFileDataReporter = "";
const char* DOCSTR_daeTEXTFileDataReporter_WriteDataToFile = "";

const char* DOCSTR_daeDataReporterRemote = "";
    const char* DOCSTR_daeDataReporterRemote_SendMessage = "";
    
const char* DOCSTR_daeTCPIPDataReporter = "";
    const char* DOCSTR_daeTCPIPDataReporter_SendMessage = "";

const char* DOCSTR_daeDataReceiverDomain = "";
    const char* DOCSTR_daeDataReceiverDomain_init = "";
    const char* DOCSTR_daeDataReceiverDomain_Name = "";
    const char* DOCSTR_daeDataReceiverDomain_Type = "";
    const char* DOCSTR_daeDataReceiverDomain_NumberOfPoints = "";
    const char* DOCSTR_daeDataReceiverDomain_Points = "";
    const char* DOCSTR_daeDataReceiverDomain_Coordinates = "";
    const char* DOCSTR_daeDataReceiverDomain_getitem = "";
    const char* DOCSTR_daeDataReceiverDomain_setitem = "";

const char* DOCSTR_daeDataReceiverVariableValue = "";
    const char* DOCSTR_daeDataReceiverVariableValue_init = "";
    const char* DOCSTR_daeDataReceiverVariableValue_Time = "";
    const char* DOCSTR_daeDataReceiverVariableValue_getitem = "";
    const char* DOCSTR_daeDataReceiverVariableValue_setitem = "";

const char* DOCSTR_daeDataReceiverVariable = "";
    const char* DOCSTR_daeDataReceiverVariable_init = "";
    const char* DOCSTR_daeDataReceiverVariable_Name = "";
    const char* DOCSTR_daeDataReceiverVariable_NumberOfPoints = "";
    const char* DOCSTR_daeDataReceiverVariable_Domains = "";
    const char* DOCSTR_daeDataReceiverVariable_TimeValues = "";
    const char* DOCSTR_daeDataReceiverVariable_Values = "";
    const char* DOCSTR_daeDataReceiverVariable_AddDomain = "";
    const char* DOCSTR_daeDataReceiverVariable_AddVariableValue = "";
    
const char* DOCSTR_daeDataReceiverProcess = "";
    const char* DOCSTR_daeDataReceiverProcess_init = "";
    const char* DOCSTR_daeDataReceiverProcess_Name = "";
    const char* DOCSTR_daeDataReceiverProcess_Domains = "";
    const char* DOCSTR_daeDataReceiverProcess_Variables = "";
    const char* DOCSTR_daeDataReceiverProcess_dictDomains = "";
    const char* DOCSTR_daeDataReceiverProcess_dictVariables = "";
    const char* DOCSTR_daeDataReceiverProcess_RegisterDomain = "";
    const char* DOCSTR_daeDataReceiverProcess_RegisterVariable = "";
    const char* DOCSTR_daeDataReceiverProcess_FindVariable = "";

    
    
const char* DOCSTR_daeDataReceiver_t = "";
    const char* DOCSTR_daeDataReceiver_t_Start = "";
    const char* DOCSTR_daeDataReceiver_t_Stop = "";
    const char* DOCSTR_daeDataReceiver_t_Process = "";

const char* DOCSTR_daeTCPIPDataReceiver = "";
    const char* DOCSTR_daeTCPIPDataReceiver_Start = "";
    const char* DOCSTR_daeTCPIPDataReceiver_Stop = "";
    const char* DOCSTR_daeTCPIPDataReceiver_Process = "";

const char* DOCSTR_daeTCPIPDataReceiverServer = "";
    const char* DOCSTR_daeTCPIPDataReceiverServer_init = "";
    const char* DOCSTR_daeTCPIPDataReceiverServer_Start = "";
    const char* DOCSTR_daeTCPIPDataReceiverServer_Stop = "";
    const char* DOCSTR_daeTCPIPDataReceiverServer_IsConnected = "";
    const char* DOCSTR_daeTCPIPDataReceiverServer_DataReceivers = "";
    
#endif
