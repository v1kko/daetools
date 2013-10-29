#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include "docstrings.h"
#include <noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyDataReporting)
{
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    
    docstring_options doc_options(true, true, false);

/**************************************************************
	daeDataReporter auxuliary classes
***************************************************************/
	class_<daeDataReporterDomain, boost::noncopyable>("daeDataReporterDomain", DOCSTR_daeDataReporterDomain, no_init)
		.def(init<string, daeeDomainType, size_t>(( arg("self"), 
                                                    arg("name"), 
                                                    arg("type"), 
                                                    arg("numberOfPoints") 
                                                  ), DOCSTR_daeDataReporterDomain_init))
		
		.def_readonly("Name",				&daeDataReporterDomain::m_strName,          DOCSTR_daeDataReporterDomain_Name)
		.def_readonly("Type",				&daeDataReporterDomain::m_eType,            DOCSTR_daeDataReporterDomain_Type)		
		.def_readonly("NumberOfPoints",		&daeDataReporterDomain::m_nNumberOfPoints,  DOCSTR_daeDataReporterDomain_NumberOfPoints)
		
		.add_property("Points",				&daepython::GetDataReporterDomainPoints,    DOCSTR_daeDataReporterDomain_Points)
		;

	class_<daeDataReporterVariable, boost::noncopyable>("daeDataReporterVariable", DOCSTR_daeDataReporterVariable, no_init)
		.def(init<string, size_t>(( arg("self"), 
                                    arg("name"), 
                                    arg("numberOfPoints") 
                                  ), DOCSTR_daeDataReporterVariable_init))
		
		.def_readonly("Name",				&daeDataReporterVariable::m_strName,            DOCSTR_daeDataReporterVariable_Name)
		.def_readonly("NumberOfPoints",		&daeDataReporterVariable::m_nNumberOfPoints,    DOCSTR_daeDataReporterVariable_NumberOfPoints)
		
		.add_property("NumberOfDomains",	&daeDataReporterVariable::GetNumberOfDomains,   DOCSTR_daeDataReporterVariable_NumberOfDomains)
		.add_property("Domains",			&daepython::GetDataReporterDomains,             DOCSTR_daeDataReporterVariable_Domains)

		.def("AddDomain",					&daeDataReporterVariable::AddDomain, ( arg("self"), 
                                                                                   arg("domainName") 
                                                                                 ), DOCSTR_daeDataReporterVariable_AddDomain)
		;

	class_<daeDataReporterVariableValue, boost::noncopyable>("daeDataReporterVariableValue", DOCSTR_daeDataReporterVariableValue, no_init)
		.def(init<string, size_t>(( arg("self"), 
                                    arg("name"), 
                                    arg("numberOfPoints") 
                                  ), DOCSTR_daeDataReporterVariableValue_init))
		
		.def_readonly("Name",				&daeDataReporterVariableValue::m_strName,           DOCSTR_daeDataReporterVariableValue_Name)
		.def_readonly("NumberOfPoints",		&daeDataReporterVariableValue::m_nNumberOfPoints,   DOCSTR_daeDataReporterVariableValue_NumberOfPoints)

		.add_property("Values",				&daepython::GetNumPyArrayDataReporterVariableValue, DOCSTR_daeDataReporterVariableValue_Values)
		
		.def("__getitem__",					&daeDataReporterVariableValue::GetValue,            ( arg("self"), arg("index") ), DOCSTR_daeDataReporterVariableValue_getitem)
		.def("__setitem__",					&daeDataReporterVariableValue::SetValue,            ( arg("self"), arg("index"), arg("value") ), DOCSTR_daeDataReporterVariableValue_setitem)
		;

/**************************************************************
    daeDataReporter_xxx
***************************************************************/
	class_<daepython::daeDataReporterWrapper, boost::noncopyable>("daeDataReporter_t", DOCSTR_daeDataReporter_t, no_init)
        .add_property("Name",		&daeDataReporter_t::GetName, DOCSTR_daeDataReporter_t_Name)
        .def("Connect",				pure_virtual(&daeDataReporter_t::Connect),           ( arg("self"),
                                                                                           arg("connectionString"), 
                                                                                           arg("processName") 
                                                                                         ), DOCSTR_daeDataReporter_t_Connect)
		.def("Disconnect",			pure_virtual(&daeDataReporter_t::Disconnect),        ( arg("self") ), DOCSTR_daeDataReporter_t_Disconnect)
		.def("IsConnected",			pure_virtual(&daeDataReporter_t::IsConnected),       ( arg("self") ), DOCSTR_daeDataReporter_t_IsConnected)
		.def("StartRegistration",	pure_virtual(&daeDataReporter_t::StartRegistration), ( arg("self") ), DOCSTR_daeDataReporter_t_StartRegistration)
		.def("RegisterDomain",		pure_virtual(&daeDataReporter_t::RegisterDomain),    ( arg("self"), 
                                                                                           arg("domain") 
                                                                                         ), DOCSTR_daeDataReporter_t_RegisterDomain)
		.def("RegisterVariable",	pure_virtual(&daeDataReporter_t::RegisterVariable),  ( arg("self"), 
                                                                                           arg("variable") 
                                                                                         ), DOCSTR_daeDataReporter_t_RegisterVariable)
		.def("EndRegistration",		pure_virtual(&daeDataReporter_t::EndRegistration),   ( arg("self") ), DOCSTR_daeDataReporter_t_EndRegistration)
		.def("StartNewResultSet",	pure_virtual(&daeDataReporter_t::StartNewResultSet), ( arg("self"),
                                                                                           arg("time") 
                                                                                         ), DOCSTR_daeDataReporter_t_StartNewResultSet)
        .def("SendVariable",	    pure_virtual(&daeDataReporter_t::SendVariable),      ( arg("self"), 
                                                                                           arg("variableValue") 
                                                                                         ), DOCSTR_daeDataReporter_t_SendVariable)
		.def("EndOfData",		    pure_virtual(&daeDataReporter_t::EndOfData),         ( arg("self") ), DOCSTR_daeDataReporter_t_EndOfData)
		;


	class_<daeBlackHoleDataReporter, bases<daeDataReporter_t>, boost::noncopyable>("daeBlackHoleDataReporter", no_init)
        .def(init<>(( arg("self") )))
            
		.def("Connect",				&daeBlackHoleDataReporter::Connect)
		.def("Disconnect",			&daeBlackHoleDataReporter::Disconnect)
		.def("IsConnected",			&daeBlackHoleDataReporter::IsConnected)
		.def("StartRegistration",	&daeBlackHoleDataReporter::StartRegistration)
		.def("RegisterDomain",		&daeBlackHoleDataReporter::RegisterDomain)
		.def("RegisterVariable",	&daeBlackHoleDataReporter::RegisterVariable)
		.def("EndRegistration",		&daeBlackHoleDataReporter::EndRegistration)
		.def("StartNewResultSet",	&daeBlackHoleDataReporter::StartNewResultSet)
		.def("EndOfData",	    	&daeBlackHoleDataReporter::EndOfData)
		.def("SendVariable",	  	&daeBlackHoleDataReporter::SendVariable)  
		;
	
	class_<daepython::daeDelegateDataReporterWrapper, bases<daeDataReporter_t>, boost::noncopyable>("daeDelegateDataReporter", DOCSTR_daeDelegateDataReporter, no_init)
        .def(init<>(( arg("self") )))
            
		.def("Connect",				&daeDataReporter_t::Connect,			&daepython::daeDelegateDataReporterWrapper::def_Connect)
		.def("Disconnect",			&daeDataReporter_t::Disconnect,			&daepython::daeDelegateDataReporterWrapper::def_Disconnect)
		.def("IsConnected",			&daeDataReporter_t::IsConnected,		&daepython::daeDelegateDataReporterWrapper::def_IsConnected)
		.def("StartRegistration",	&daeDataReporter_t::StartRegistration,	&daepython::daeDelegateDataReporterWrapper::def_StartRegistration)
		.def("RegisterDomain",		&daeDataReporter_t::RegisterDomain,		&daepython::daeDelegateDataReporterWrapper::def_RegisterDomain)
		.def("RegisterVariable",	&daeDataReporter_t::RegisterVariable,	&daepython::daeDelegateDataReporterWrapper::def_RegisterVariable)
		.def("EndRegistration",		&daeDataReporter_t::EndRegistration,	&daepython::daeDelegateDataReporterWrapper::def_EndRegistration)
		.def("StartNewResultSet",	&daeDataReporter_t::StartNewResultSet,	&daepython::daeDelegateDataReporterWrapper::def_StartNewResultSet)
		.def("EndOfData",	    	&daeDataReporter_t::EndOfData,		    &daepython::daeDelegateDataReporterWrapper::def_EndOfData)
		.def("SendVariable",	  	&daeDataReporter_t::SendVariable,		&daepython::daeDelegateDataReporterWrapper::def_SendVariable)
		
        .def("AddDataReporter",	  	&daeDelegateDataReporter::AddDataReporter, ( arg("self"), arg("dataReporter") ), DOCSTR_daeDelegateDataReporter_AddDataReporter)
        
        .add_property("DataReporters",  &daepython::daeDelegateDataReporterWrapper::GetDataReporters, DOCSTR_daeDelegateDataReporter_DataReporters)
		;

	class_<daepython::daeDataReporterLocalWrapper, bases<daeDataReporter_t>, boost::noncopyable>("daeDataReporterLocal", DOCSTR_daeDataReporterLocal, no_init)
        .def(init<>(( arg("self") )))
            
		.def("Connect",				&daeDataReporter_t::Connect,			&daepython::daeDataReporterLocalWrapper::def_Connect)
		.def("Disconnect",			&daeDataReporter_t::Disconnect,			&daepython::daeDataReporterLocalWrapper::def_Disconnect)
		.def("IsConnected",			&daeDataReporter_t::IsConnected,		&daepython::daeDataReporterLocalWrapper::def_IsConnected)
		.def("StartRegistration",	&daeDataReporter_t::StartRegistration,	&daepython::daeDataReporterLocalWrapper::def_StartRegistration)
		.def("RegisterDomain",		&daeDataReporter_t::RegisterDomain,		&daepython::daeDataReporterLocalWrapper::def_RegisterDomain)
		.def("RegisterVariable",	&daeDataReporter_t::RegisterVariable,	&daepython::daeDataReporterLocalWrapper::def_RegisterVariable)
		.def("EndRegistration",		&daeDataReporter_t::EndRegistration,	&daepython::daeDataReporterLocalWrapper::def_EndRegistration)
		.def("StartNewResultSet",	&daeDataReporter_t::StartNewResultSet,	&daepython::daeDataReporterLocalWrapper::def_StartNewResultSet)
		.def("EndOfData",	    	&daeDataReporter_t::EndOfData,		    &daepython::daeDataReporterLocalWrapper::def_EndOfData)
		.def("SendVariable",	  	&daeDataReporter_t::SendVariable,		&daepython::daeDataReporterLocalWrapper::def_SendVariable)
            
		.add_property("Process",	make_function(&daeDataReporterLocal::GetProcess, return_internal_reference<>()), DOCSTR_daeDataReporterLocal_Process )
        
        .add_property("dictDomains",		&daepython::daeDataReporterLocalWrapper::GetDomainsAsDict)
        .add_property("dictVariables",		&daepython::daeDataReporterLocalWrapper::GetVariablesAsDict)
		;
	
	class_<daeNoOpDataReporter, bases<daeDataReporterLocal>, boost::noncopyable>("daeNoOpDataReporter", DOCSTR_daeNoOpDataReporter, no_init)
        .def(init<>(( arg("self") )))
            
		.def("Connect",				&daeNoOpDataReporter::Connect)
		.def("Disconnect",			&daeNoOpDataReporter::Disconnect)
		.def("IsConnected",			&daeNoOpDataReporter::IsConnected)
		.def("StartRegistration",	&daeNoOpDataReporter::StartRegistration)
		.def("RegisterDomain",		&daeNoOpDataReporter::RegisterDomain)
		.def("RegisterVariable",	&daeNoOpDataReporter::RegisterVariable)
		.def("EndRegistration",		&daeNoOpDataReporter::EndRegistration)
		.def("StartNewResultSet",	&daeNoOpDataReporter::StartNewResultSet)
		.def("EndOfData",	    	&daeNoOpDataReporter::EndOfData)
		.def("SendVariable",	  	&daeNoOpDataReporter::SendVariable)  
		;

	class_<daepython::daeDataReporterFileWrapper, bases<daeDataReporterLocal>, boost::noncopyable>("daeDataReporterFile", DOCSTR_daeDataReporterFile, no_init)
        .def(init<>(( arg("self") )))
            
		.def("Connect",				&daeDataReporter_t::Connect,			&daepython::daeDataReporterFileWrapper::def_Connect)
		.def("Disconnect",			&daeDataReporter_t::Disconnect,			&daepython::daeDataReporterFileWrapper::def_Disconnect)
		.def("IsConnected",			&daeDataReporter_t::IsConnected,		&daepython::daeDataReporterFileWrapper::def_IsConnected)
		.def("StartRegistration",	&daeDataReporter_t::StartRegistration,	&daepython::daeDataReporterFileWrapper::def_StartRegistration)
		.def("RegisterDomain",		&daeDataReporter_t::RegisterDomain,		&daepython::daeDataReporterFileWrapper::def_RegisterDomain)
		.def("RegisterVariable",	&daeDataReporter_t::RegisterVariable,	&daepython::daeDataReporterFileWrapper::def_RegisterVariable)
		.def("EndRegistration",		&daeDataReporter_t::EndRegistration,	&daepython::daeDataReporterFileWrapper::def_EndRegistration)
		.def("StartNewResultSet",	&daeDataReporter_t::StartNewResultSet,	&daepython::daeDataReporterFileWrapper::def_StartNewResultSet)
		.def("EndOfData",	    	&daeDataReporter_t::EndOfData,		    &daepython::daeDataReporterFileWrapper::def_EndOfData)
		.def("SendVariable",	  	&daeDataReporter_t::SendVariable,		&daepython::daeDataReporterFileWrapper::def_SendVariable)
            
		.def("WriteDataToFile",	    &daeFileDataReporter::WriteDataToFile,	
                                    &daepython::daeDataReporterFileWrapper::WriteDataToFile,
                                    ( arg("self") ), DOCSTR_daeDataReporterFile_WriteDataToFile)
		;

	class_<daepython::daeTEXTFileDataReporterWrapper, bases<daeFileDataReporter>, boost::noncopyable>("daeTEXTFileDataReporter", DOCSTR_daeTEXTFileDataReporter, no_init)
        .def(init<>(( arg("self") )))
            
		.def("Connect",				&daeDataReporter_t::Connect,			&daepython::daeTEXTFileDataReporterWrapper::def_Connect)
		.def("Disconnect",			&daeDataReporter_t::Disconnect,			&daepython::daeTEXTFileDataReporterWrapper::def_Disconnect)
		.def("IsConnected",			&daeDataReporter_t::IsConnected,		&daepython::daeTEXTFileDataReporterWrapper::def_IsConnected)
		.def("StartRegistration",	&daeDataReporter_t::StartRegistration,	&daepython::daeTEXTFileDataReporterWrapper::def_StartRegistration)
		.def("RegisterDomain",		&daeDataReporter_t::RegisterDomain,		&daepython::daeTEXTFileDataReporterWrapper::def_RegisterDomain)
		.def("RegisterVariable",	&daeDataReporter_t::RegisterVariable,	&daepython::daeTEXTFileDataReporterWrapper::def_RegisterVariable)
		.def("EndRegistration",		&daeDataReporter_t::EndRegistration,	&daepython::daeTEXTFileDataReporterWrapper::def_EndRegistration)
		.def("StartNewResultSet",	&daeDataReporter_t::StartNewResultSet,	&daepython::daeTEXTFileDataReporterWrapper::def_StartNewResultSet)
		.def("EndOfData",	    	&daeDataReporter_t::EndOfData,		    &daepython::daeTEXTFileDataReporterWrapper::def_EndOfData)
		.def("SendVariable",	  	&daeDataReporter_t::SendVariable,		&daepython::daeTEXTFileDataReporterWrapper::def_SendVariable)
		.def("WriteDataToFile",	    &daeFileDataReporter::WriteDataToFile,	
                                    &daepython::daeTEXTFileDataReporterWrapper::def_WriteDataToFile,
                                    ( arg("self") ), DOCSTR_daeTEXTFileDataReporter_WriteDataToFile)
		;

	class_<daepython::daeDataReporterRemoteWrapper, bases<daeDataReporter_t>, boost::noncopyable>("daeDataReporterRemote", DOCSTR_daeDataReporterRemote, no_init)
        .def(init<>(( arg("self") )))
            
		.def("Connect",				&daeDataReporter_t::Connect,			&daepython::daeDataReporterRemoteWrapper::def_Connect)
		.def("Disconnect",			&daeDataReporter_t::Disconnect,			&daepython::daeDataReporterRemoteWrapper::def_Disconnect)
		.def("IsConnected",			&daeDataReporter_t::IsConnected,		&daepython::daeDataReporterRemoteWrapper::def_IsConnected)
		.def("StartRegistration",	&daeDataReporter_t::StartRegistration,	&daepython::daeDataReporterRemoteWrapper::def_StartRegistration)
		.def("RegisterDomain",		&daeDataReporter_t::RegisterDomain,		&daepython::daeDataReporterRemoteWrapper::def_RegisterDomain)
		.def("RegisterVariable",	&daeDataReporter_t::RegisterVariable,	&daepython::daeDataReporterRemoteWrapper::def_RegisterVariable)
		.def("EndRegistration",		&daeDataReporter_t::EndRegistration,	&daepython::daeDataReporterRemoteWrapper::def_EndRegistration)
		.def("StartNewResultSet",	&daeDataReporter_t::StartNewResultSet,	&daepython::daeDataReporterRemoteWrapper::def_StartNewResultSet)
		.def("EndOfData",	    	&daeDataReporter_t::EndOfData,		    &daepython::daeDataReporterRemoteWrapper::def_EndOfData)
		.def("SendVariable",	  	&daeDataReporter_t::SendVariable,		&daepython::daeDataReporterRemoteWrapper::def_SendVariable)
		
        .def("SendMessage",	    	&daeDataReporterRemote::SendMessage,    
                                    &daepython::daeDataReporterRemoteWrapper::SendMessage,
                                    ( arg("self"), arg("message") ), DOCSTR_daeDataReporterRemote_SendMessage)
		; 
	
	class_<daepython::daeTCPIPDataReporterWrapper, bases<daeDataReporterRemote>, boost::noncopyable>("daeTCPIPDataReporter", DOCSTR_daeTCPIPDataReporter, no_init)
        .def(init<>(( arg("self") )))
            
		.def("Connect",				&daeDataReporter_t::Connect,			&daepython::daeTCPIPDataReporterWrapper::def_Connect)
		.def("Disconnect",			&daeDataReporter_t::Disconnect,			&daepython::daeTCPIPDataReporterWrapper::def_Disconnect)
		.def("IsConnected",			&daeDataReporter_t::IsConnected,		&daepython::daeTCPIPDataReporterWrapper::def_IsConnected)
		.def("StartRegistration",	&daeDataReporter_t::StartRegistration,	&daepython::daeTCPIPDataReporterWrapper::def_StartRegistration)
		.def("RegisterDomain",		&daeDataReporter_t::RegisterDomain,		&daepython::daeTCPIPDataReporterWrapper::def_RegisterDomain)
		.def("RegisterVariable",	&daeDataReporter_t::RegisterVariable,	&daepython::daeTCPIPDataReporterWrapper::def_RegisterVariable)
		.def("EndRegistration",		&daeDataReporter_t::EndRegistration,	&daepython::daeTCPIPDataReporterWrapper::def_EndRegistration)
		.def("StartNewResultSet",	&daeDataReporter_t::StartNewResultSet,	&daepython::daeTCPIPDataReporterWrapper::def_StartNewResultSet)
		.def("EndOfData",	    	&daeDataReporter_t::EndOfData,		    &daepython::daeTCPIPDataReporterWrapper::def_EndOfData)
		.def("SendVariable",	  	&daeDataReporter_t::SendVariable,		&daepython::daeTCPIPDataReporterWrapper::def_SendVariable)
		
        .def("SendMessage",	    	&daeDataReporterRemote::SendMessage,	
                                    &daepython::daeTCPIPDataReporterWrapper::def_SendMessage,
                                    ( arg("self"), arg("message") ), DOCSTR_daeTCPIPDataReporter_SendMessage)
		; 
		
/**************************************************************
	daeDataReceiver
***************************************************************/
	class_<daeDataReceiverDomain, boost::noncopyable>("daeDataReceiverDomain", DOCSTR_daeDataReceiverDomain, no_init)
		.def(init<string, daeeDomainType, size_t>(( arg("self"), 
                                                    arg("name"), 
                                                    arg("type"), 
                                                    arg("numberOfPoints") 
                                                  ), DOCSTR_daeDataReceiverDomain_init))
		
        .def_readonly("Name",				&daeDataReceiverDomain::m_strName,              DOCSTR_daeDataReceiverDomain_Name)
        .def_readonly("Type",				&daeDataReceiverDomain::m_eType,                DOCSTR_daeDataReceiverDomain_Type)
        .def_readonly("NumberOfPoints",		&daeDataReceiverDomain::m_nNumberOfPoints,      DOCSTR_daeDataReceiverDomain_NumberOfPoints)
		
        .add_property("Points",				&daepython::GetDataReceiverDomainPoints,        DOCSTR_daeDataReceiverDomain_Points)
        .add_property("Coordinates",		&daepython::GetDataReceiverDomainCoordinates,   DOCSTR_daeDataReceiverDomain_Coordinates)
        ;

	class_<daeDataReceiverVariableValue, boost::noncopyable>("daeDataReceiverVariableValue", DOCSTR_daeDataReceiverVariableValue, no_init)  
		.def(init<real_t, size_t>(( arg("self"), 
                                    arg("time"), 
                                    arg("numberOfPoints") 
                                  ), DOCSTR_daeDataReceiverVariableValue_init))
		
		.def_readonly("Time",				&daeDataReceiverVariableValue::m_dTime, DOCSTR_daeDataReceiverVariableValue_Time)

		.def("__getitem__",					&daeDataReceiverVariableValue::GetValue, ( arg("self"), arg("index") ), DOCSTR_daeDataReceiverVariableValue_getitem)
		.def("__setitem__",					&daeDataReceiverVariableValue::SetValue, ( arg("self"), arg("index"), arg("value") ), DOCSTR_daeDataReceiverVariableValue_setitem)
		;

	class_<daeDataReceiverVariable, boost::noncopyable>("daeDataReceiverVariable", DOCSTR_daeDataReceiverVariable, no_init)
		.def(init<string, size_t>(( arg("self"), 
                                    arg("name"), 
                                    arg("numberOfPoints") 
                                  ), DOCSTR_daeDataReceiverVariable_init)) 

		.def_readonly("Name",				&daeDataReceiverVariable::m_strName,            DOCSTR_daeDataReceiverVariable_Name)
		.def_readonly("NumberOfPoints",		&daeDataReceiverVariable::m_nNumberOfPoints,    DOCSTR_daeDataReceiverVariable_NumberOfPoints)
		
		.add_property("Domains",			&daepython::GetDomainsDataReceiverVariable,     DOCSTR_daeDataReceiverVariable_Domains)
		.add_property("TimeValues",			&daepython::GetTimeValuesDataReceiverVariable,  DOCSTR_daeDataReceiverVariable_TimeValues)
		.add_property("Values",				&daepython::GetNumPyArrayDataReceiverVariable,  DOCSTR_daeDataReceiverVariable_Values)
		
		.def("AddDomain",					&daeDataReceiverVariable::AddDomain, 
                                            ( arg("self"), arg("domain") ), DOCSTR_daeDataReceiverVariable_AddDomain)
		.def("AddVariableValue",			&daeDataReceiverVariable::AddVariableValue, 
                                            ( arg("self"), arg("variableValue") ), DOCSTR_daeDataReceiverVariable_AddVariableValue)
		;

	class_<daeDataReceiverProcess, boost::noncopyable>("daeDataReceiverProcess", DOCSTR_daeDataReceiverProcess, no_init)
		.def(init<string>(( arg("self"), 
                            arg("name") 
                          ), DOCSTR_daeDataReceiverProcess_init))

		.def_readonly("Name",				&daeDataReceiverProcess::m_strName, DOCSTR_daeDataReceiverProcess_Name)
 
		.add_property("Domains",			&daepython::GetDomainsDataReporterProcess,   DOCSTR_daeDataReceiverProcess_Domains)
		.add_property("Variables",			&daepython::GetVariablesDataReporterProcess, DOCSTR_daeDataReceiverProcess_Variables)
            
        .add_property("dictDomains",		&daepython::GetDomainsAsDictDataReporterProcess,   DOCSTR_daeDataReceiverProcess_dictDomains)
        .add_property("dictVariables",		&daepython::GetVariablesAsDictDataReporterProcess, DOCSTR_daeDataReceiverProcess_dictVariables)
		
		.def("RegisterDomain",				&daeDataReceiverProcess::RegisterDomain, 
                                            ( arg("self"), arg("domain") ), DOCSTR_daeDataReceiverProcess_RegisterDomain)
		.def("RegisterVariable",			&daeDataReceiverProcess::RegisterVariable, 
                                            ( arg("self"), arg("variable") ), DOCSTR_daeDataReceiverProcess_RegisterVariable)
		.def("FindVariable",				&daeDataReceiverProcess::FindVariable, return_internal_reference<>(), 
                                            ( arg("self"), arg("variableName") ), DOCSTR_daeDataReceiverProcess_FindVariable)
		;
  
	class_<daepython::daeDataReceiverWrapper, boost::noncopyable>("daeDataReceiver_t", DOCSTR_daeDataReceiver_t, no_init)
        .add_property("Process",	make_function(&daeDataReceiver_t::GetProcess, return_internal_reference<>()), 
                                    DOCSTR_daeDataReceiver_t_Process)
		
        .def("Start",				pure_virtual(&daeDataReceiver_t::Start),    
                                    ( arg("self") ), DOCSTR_daeDataReceiver_t_Start)
		.def("Stop",				pure_virtual(&daeDataReceiver_t::Stop),     
                                    ( arg("self") ), DOCSTR_daeDataReceiver_t_Stop)
		;

    class_<daeTCPIPDataReceiver, bases<daeDataReceiver_t>, boost::noncopyable>("daeTCPIPDataReceiver", DOCSTR_daeTCPIPDataReceiver, no_init)
		.def("Start",				&daeTCPIPDataReceiver::Start,    
                                    ( arg("self") ), DOCSTR_daeTCPIPDataReceiver_Start)
		.def("Stop",				&daeTCPIPDataReceiver::Stop,     
                                    ( arg("self") ), DOCSTR_daeTCPIPDataReceiver_Stop)
		;

	class_<daepython::daeTCPIPDataReceiverServerWrapper, boost::noncopyable>("daeTCPIPDataReceiverServer", DOCSTR_daeTCPIPDataReceiverServer, no_init)
        .def(init<int>(( arg("self") ), DOCSTR_daeTCPIPDataReceiverServer_init))
            
		.add_property("DataReceivers",	&daepython::daeTCPIPDataReceiverServerWrapper::GetDataReceivers,         
                                        DOCSTR_daeTCPIPDataReceiverServer_DataReceivers)

        .def("Start",					&daepython::daeTCPIPDataReceiverServerWrapper::Start_, 
                                        ( arg("self") ), DOCSTR_daeTCPIPDataReceiverServer_Start)
        .def("Stop",					&daepython::daeTCPIPDataReceiverServerWrapper::Stop_, 
                                        ( arg("self") ), DOCSTR_daeTCPIPDataReceiverServer_Stop)
        .def("IsConnected",				&daepython::daeTCPIPDataReceiverServerWrapper::IsConnected_, 
                                        ( arg("self") ), DOCSTR_daeTCPIPDataReceiverServer_IsConnected)
	;

}
