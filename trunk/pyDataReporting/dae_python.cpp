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
	daeDataReporter
***************************************************************/
	class_<daeDataReporterDomain>("daeDataReporterDomain")
		.def(init<string, daeeDomainType, size_t>())
		
		.def_readonly("Name",				&daeDataReporterDomain::m_strName)
		.def_readonly("Type",				&daeDataReporterDomain::m_eType)		
		.def_readonly("NumberOfPoints",		&daeDataReporterDomain::m_nNumberOfPoints)
		
		.add_property("Points",				&daepython::GetDataReporterDomainPoints)

		.def("__getitem__",					&daeDataReporterDomain::GetPoint)
		.def("__setitem__",					&daeDataReporterDomain::SetPoint)
		;

	class_<daeDataReporterVariable>("daeDataReporterVariable")
		.def(init<string, size_t>())
		
		.def_readonly("Name",				&daeDataReporterVariable::m_strName)
		.def_readonly("NumberOfPoints",		&daeDataReporterVariable::m_nNumberOfPoints)
		
		.add_property("NumberOfDomains",	&daeDataReporterVariable::GetNumberOfDomains)
		.add_property("Domains",			&daepython::GetDataReporterDomains)

		.def("AddDomain",					&daeDataReporterVariable::AddDomain)
		;

	class_<daeDataReporterVariableValue>("daeDataReporterVariableValue")
		.def(init<string, size_t>())
		
		.def_readonly("Name",				&daeDataReporterVariableValue::m_strName)
		.def_readonly("NumberOfPoints",		&daeDataReporterVariableValue::m_nNumberOfPoints)

		.add_property("Values",				&daepython::GetNumPyArrayDataReporterVariableValue)
		
		.def("__getitem__",					&daeDataReporterVariableValue::GetValue)
		.def("__setitem__",					&daeDataReporterVariableValue::SetValue)
		;

	class_<daepython::daeDataReporterWrapper, boost::noncopyable>("daeDataReporter_t"/*, no_init*/)
		.def("Connect",				pure_virtual(&daeDataReporter_t::Connect))
		.def("Disconnect",			pure_virtual(&daeDataReporter_t::Disconnect))
		.def("IsConnected",			pure_virtual(&daeDataReporter_t::IsConnected))
		.def("StartRegistration",	pure_virtual(&daeDataReporter_t::StartRegistration))
		.def("RegisterDomain",		pure_virtual(&daeDataReporter_t::RegisterDomain))
		.def("RegisterVariable",	pure_virtual(&daeDataReporter_t::RegisterVariable))
		.def("EndRegistration",		pure_virtual(&daeDataReporter_t::EndRegistration))
		.def("StartNewResultSet",	pure_virtual(&daeDataReporter_t::StartNewResultSet))
		.def("EndOfData",		    pure_virtual(&daeDataReporter_t::EndOfData))
		.def("SendVariable",	    pure_virtual(&daeDataReporter_t::SendVariable))
		;


	class_<daeBlackHoleDataReporter, bases<daeDataReporter_t>, boost::noncopyable>("daeBlackHoleDataReporter")
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
	
	class_<daepython::daeDelegateDataReporterWrapper, bases<daeDataReporter_t>, boost::noncopyable>("daeDelegateDataReporter")
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
		.def("AddDataReporter",	  	&daeDelegateDataReporter::AddDataReporter)
		;

	class_<daepython::daeDataReporterLocalWrapper, bases<daeDataReporter_t>, boost::noncopyable>("daeDataReporterLocal")
		.def("Connect",				&daeDataReporter_t::Connect,			&daepython::daeDataReporterLocalWrapper::Connect)
		.def("Disconnect",			&daeDataReporter_t::Disconnect,			&daepython::daeDataReporterLocalWrapper::Disconnect)
		.def("IsConnected",			&daeDataReporter_t::IsConnected,		&daepython::daeDataReporterLocalWrapper::IsConnected)
		.def("StartRegistration",	&daeDataReporter_t::StartRegistration,	&daepython::daeDataReporterLocalWrapper::def_StartRegistration)
		.def("RegisterDomain",		&daeDataReporter_t::RegisterDomain,		&daepython::daeDataReporterLocalWrapper::def_RegisterDomain)
		.def("RegisterVariable",	&daeDataReporter_t::RegisterVariable,	&daepython::daeDataReporterLocalWrapper::def_RegisterVariable)
		.def("EndRegistration",		&daeDataReporter_t::EndRegistration,	&daepython::daeDataReporterLocalWrapper::def_EndRegistration)
		.def("StartNewResultSet",	&daeDataReporter_t::StartNewResultSet,	&daepython::daeDataReporterLocalWrapper::def_StartNewResultSet)
		.def("EndOfData",	    	&daeDataReporter_t::EndOfData,		    &daepython::daeDataReporterLocalWrapper::def_EndOfData)
		.def("SendVariable",	  	&daeDataReporter_t::SendVariable,		&daepython::daeDataReporterLocalWrapper::def_SendVariable)
		.add_property("Process",	make_function(&daeDataReporterLocal::GetProcess, return_internal_reference<>()) )
		;
	
	class_<daeNoOpDataReporter, bases<daeDataReporterLocal>, boost::noncopyable>("daeNoOpDataReporter")
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

	class_<daepython::daeDataReporterFileWrapper, bases<daeDataReporterLocal>, boost::noncopyable>("daeDataReporterFile")
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
		.def("WriteDataToFile",	    &daeFileDataReporter::WriteDataToFile,	&daepython::daeDataReporterFileWrapper::WriteDataToFile)
		;

	class_<daepython::daeTEXTFileDataReporterWrapper, bases<daeFileDataReporter>, boost::noncopyable>("daeTEXTFileDataReporter")
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
		.def("WriteDataToFile",	    &daeFileDataReporter::WriteDataToFile,	&daepython::daeTEXTFileDataReporterWrapper::def_WriteDataToFile)
		;

	class_<daepython::daeDataReporterRemoteWrapper, bases<daeDataReporter_t>, boost::noncopyable>("daeDataReporterRemote")
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
		.def("SendMessage",	    	&daeDataReporterRemote::SendMessage,    &daepython::daeDataReporterRemoteWrapper::SendMessage)
		; 
	
	class_<daepython::daeTCPIPDataReporterWrapper, bases<daeDataReporterRemote>, boost::noncopyable>("daeTCPIPDataReporter")
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
		.def("SendMessage",	    	&daeDataReporterRemote::SendMessage,	&daepython::daeTCPIPDataReporterWrapper::def_SendMessage)
		; 
		
/**************************************************************
	daeDataReceiver
***************************************************************/
	class_<daeDataReceiverDomain, boost::noncopyable>("daeDataReceiverDomain")
		.def(init<string, daeeDomainType, size_t>())
		
		.def_readonly("Name",				&daeDataReceiverDomain::m_strName)
		.def_readonly("Type",				&daeDataReceiverDomain::m_eType)
		.def_readonly("NumberOfPoints",		&daeDataReceiverDomain::m_nNumberOfPoints)
		
		.add_property("Points",				&daepython::GetDataReceiverDomainPoints)
		
		.def("__getitem__",					&daeDataReceiverDomain::GetPoint)
		.def("__setitem__",					&daeDataReceiverDomain::SetPoint)
		;

	class_<daeDataReceiverVariableValue, boost::noncopyable>("daeDataReceiverVariableValue")  
		.def(init<real_t, size_t>())
		
		.def_readonly("Time",				&daeDataReceiverVariableValue::m_dTime)

		.def("__getitem__",					&daeDataReceiverVariableValue::GetValue)
		.def("__setitem__",					&daeDataReceiverVariableValue::SetValue)
		;

	class_<daeDataReceiverVariable, boost::noncopyable>("daeDataReceiverVariable")
		.def(init<string, size_t>()) 

		.def_readonly("Name",				&daeDataReceiverVariable::m_strName)
		.def_readonly("NumberOfPoints",		&daeDataReceiverVariable::m_nNumberOfPoints)
		
		.add_property("Domains",			&daepython::GetDomainsDataReceiverVariable)
		.add_property("TimeValues",			&daepython::GetTimeValuesDataReceiverVariable)
		.add_property("Values",				&daepython::GetNumPyArrayDataReceiverVariable)
		
		.def("AddDomain",					&daeDataReceiverVariable::AddDomain)
		.def("AddVariableValue",			&daeDataReceiverVariable::AddVariableValue)
		;

	class_<daeDataReporterProcess, boost::noncopyable>("daeDataReporterProcess")
		.def(init<string>())

		.def_readonly("Name",				&daeDataReporterProcess::m_strName)
 
		.add_property("Domains",			&daepython::GetDomainsDataReporterProcess)
		.add_property("Variables",			&daepython::GetVariablesDataReporterProcess)
		
		.def("RegisterDomain",				&daeDataReporterProcess::RegisterDomain)
		.def("RegisterVariable",			&daeDataReporterProcess::RegisterVariable)
		.def("FindVariable",				&daeDataReporterProcess::FindVariable, return_internal_reference<>())
		;
  
	class_<daepython::daeDataReceiverWrapper, boost::noncopyable>("daeDataReceiver_t", no_init)
		.def("Start",				pure_virtual(&daeDataReceiver_t::Start))
		.def("Stop",				pure_virtual(&daeDataReceiver_t::Stop))
		.def("GetProcess",			pure_virtual(&daeDataReceiver_t::GetProcess), return_internal_reference<>())
		;

	class_<daepython::daeTCPIPDataReceiverServerWrapper, boost::noncopyable>("daeTCPIPDataReceiverServer", init<int>())
		.add_property("NumberOfProcesses",		&daepython::daeTCPIPDataReceiverServerWrapper::GetNumberOfProcesses)
		.add_property("NumberOfDataReceivers",	&daepython::daeTCPIPDataReceiverServerWrapper::GetNumberOfDataReceivers)
		//.add_property("DataReceivers",			&daepython::daeTCPIPDataReceiverServerWrapper::GetDataReceivers)
		//.add_property("Processes",				&daepython::daeTCPIPDataReceiverServerWrapper::GetProcesses)

        .def("Start",					&daepython::daeTCPIPDataReceiverServerWrapper::Start_)
        .def("Stop",					&daepython::daeTCPIPDataReceiverServerWrapper::Stop_)
		.def("GetProcess",				&daepython::daeTCPIPDataReceiverServerWrapper::GetProcess, return_internal_reference<>())
		.def("GetDataReceiver",			&daepython::daeTCPIPDataReceiverServerWrapper::GetDataReceiver, return_internal_reference<>())
		.def("IsConnected",				&daepython::daeTCPIPDataReceiverServerWrapper::IsConnected_)
	;

}
