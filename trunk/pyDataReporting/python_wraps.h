#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#include <python.h>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/python/call_method.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "../dae_develop.h"
#include "../DataReporting/datareporters.h"

namespace daepython
{
/*******************************************************
	daeDataReporter
*******************************************************/
boost::python::list GetDataReporterDomains(daeDataReporterVariable& Variable);

boost::python::list GetDataReporterDomainPoints(daeDataReporterDomain& Domain);
		
boost::python::numeric::array GetNumPyArrayDataReporterVariableValue(daeDataReporterVariableValue& var);
	
class daeDataReporterWrapper : public daeDataReporter_t,
	                           public boost::python::wrapper<daeDataReporter_t>
{
public:
	daeDataReporterWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
		return this->get_override("Connect")(strConnectString, strProcessName);
	}
	bool Disconnect(void)
	{
		return this->get_override("Disconnect")();
	}
	bool IsConnected(void)
	{
		return this->get_override("IsConnected")();
	}
	bool StartRegistration(void)
	{
		return this->get_override("StartRegistration")();
	}
	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
		return this->get_override("RegisterDomain")(pDomain);
	}
	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
		return this->get_override("RegisterVariable")(pVariable);
	}
	bool EndRegistration(void)
	{
		return this->get_override("EndRegistration")();
	}
	bool StartNewResultSet(real_t dTime)
	{
		return this->get_override("StartNewResultSet")(dTime);
	}
	bool EndOfData(void)
	{
		return this->get_override("EndOfData")();
	}
	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
		return this->get_override("SendVariable")(pVariableValue);
	}
};

class daeDataReporterLocalWrapper : public daeDataReporterLocal,
	                                public boost::python::wrapper<daeDataReporterLocal>
{
public:
	daeDataReporterLocalWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
		return this->get_override("Connect")(strConnectString, strProcessName);
	}
	bool Disconnect(void)
	{
		return this->get_override("Disconnect")();
	}
	bool IsConnected(void)
	{
		return this->get_override("IsConnected")();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeDataReporterLocal::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeDataReporterLocal::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeDataReporterLocal::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeDataReporterLocal::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeDataReporterLocal::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeDataReporterLocal::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeDataReporterLocal::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeDataReporterLocal::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeDataReporterLocal::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeDataReporterLocal::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeDataReporterLocal::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeDataReporterLocal::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeDataReporterLocal::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeDataReporterLocal::SendVariable(pVariableValue);
	}
};

class daeDataReporterFileWrapper : public daeFileDataReporter,
	                               public boost::python::wrapper<daeFileDataReporter>
{
public:
	daeDataReporterFileWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeFileDataReporter::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeFileDataReporter::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeFileDataReporter::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeFileDataReporter::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeFileDataReporter::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeFileDataReporter::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeFileDataReporter::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeFileDataReporter::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeFileDataReporter::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeFileDataReporter::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeFileDataReporter::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeFileDataReporter::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeFileDataReporter::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeFileDataReporter::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeFileDataReporter::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeFileDataReporter::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeFileDataReporter::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeFileDataReporter::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeFileDataReporter::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeFileDataReporter::SendVariable(pVariableValue);
	}

	void WriteDataToFile(void)
	{
        this->get_override("WriteDataToFile")();
	}
};

class daeTEXTFileDataReporterWrapper : public daeTEXTFileDataReporter,
	                                   public boost::python::wrapper<daeTEXTFileDataReporter>
{
public:
	daeTEXTFileDataReporterWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeTEXTFileDataReporter::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeTEXTFileDataReporter::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeTEXTFileDataReporter::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeTEXTFileDataReporter::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeTEXTFileDataReporter::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeTEXTFileDataReporter::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeTEXTFileDataReporter::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeTEXTFileDataReporter::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeTEXTFileDataReporter::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeTEXTFileDataReporter::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeTEXTFileDataReporter::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeTEXTFileDataReporter::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeTEXTFileDataReporter::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeTEXTFileDataReporter::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeTEXTFileDataReporter::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeTEXTFileDataReporter::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeTEXTFileDataReporter::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeTEXTFileDataReporter::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeTEXTFileDataReporter::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeTEXTFileDataReporter::SendVariable(pVariableValue);
	}

	void WriteDataToFile(void)
	{
        if(boost::python::override f = this->get_override("WriteDataToFile"))
            f();
		else
			this->daeTEXTFileDataReporter::WriteDataToFile();
	}
	void def_WriteDataToFile(void)
	{
        this->daeTEXTFileDataReporter::WriteDataToFile();
	}
};

class daeDelegateDataReporterWrapper : public daeDelegateDataReporter,
	                                   public boost::python::wrapper<daeDelegateDataReporter>
{
public:
	daeDelegateDataReporterWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeDelegateDataReporter::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeDelegateDataReporter::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeDelegateDataReporter::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeDelegateDataReporter::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
			return f();
		else
			return this->daeDelegateDataReporter::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return daeDelegateDataReporter::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeDelegateDataReporter::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeDelegateDataReporter::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeDelegateDataReporter::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeDelegateDataReporter::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeDelegateDataReporter::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeDelegateDataReporter::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeDelegateDataReporter::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeDelegateDataReporter::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeDelegateDataReporter::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeDelegateDataReporter::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeDelegateDataReporter::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeDelegateDataReporter::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeDelegateDataReporter::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeDelegateDataReporter::SendVariable(pVariableValue);
	}
};

class daeDataReporterRemoteWrapper : public daeDataReporterRemote,
	                                 public boost::python::wrapper<daeDataReporterRemote>
{
public:
	daeDataReporterRemoteWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeDataReporterRemote::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeDataReporterRemote::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeDataReporterRemote::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeDataReporterRemote::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeDataReporterRemote::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeDataReporterRemote::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeDataReporterRemote::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeDataReporterRemote::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeDataReporterRemote::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeDataReporterRemote::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeDataReporterRemote::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeDataReporterRemote::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeDataReporterRemote::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeDataReporterRemote::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeDataReporterRemote::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeDataReporterRemote::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeDataReporterRemote::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeDataReporterRemote::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeDataReporterRemote::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeDataReporterRemote::SendVariable(pVariableValue);
	}

	bool SendMessage(const string& strMessage)
	{
        return this->get_override("SendMessage")(strMessage);
	}
};

class daeTCPIPDataReporterWrapper : public daeTCPIPDataReporter,
	                                public boost::python::wrapper<daeTCPIPDataReporter>
{
public:
	daeTCPIPDataReporterWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeTCPIPDataReporter::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeTCPIPDataReporter::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeTCPIPDataReporter::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeTCPIPDataReporter::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeTCPIPDataReporter::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeTCPIPDataReporter::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeTCPIPDataReporter::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeTCPIPDataReporter::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeTCPIPDataReporter::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeTCPIPDataReporter::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeTCPIPDataReporter::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeTCPIPDataReporter::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeTCPIPDataReporter::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeTCPIPDataReporter::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeTCPIPDataReporter::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeTCPIPDataReporter::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeTCPIPDataReporter::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeTCPIPDataReporter::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeTCPIPDataReporter::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeTCPIPDataReporter::SendVariable(pVariableValue);
	}

	bool SendMessage(const string& strMessage)
	{
        if(boost::python::override f = this->get_override("SendMessage"))
            return f(strMessage);
		else
			return this->daeTCPIPDataReporter::SendMessage(strMessage);
	}
	bool def_SendMessage(const string& strMessage)
	{
        return this->daeTCPIPDataReporter::SendMessage(strMessage);
	}
};
	
/*******************************************************
	daeDataReceiver
*******************************************************/
class daeDataReceiverWrapper : public daeDataReceiver_t,
	                           public boost::python::wrapper<daeDataReceiver_t>
{
public:
	daeDataReceiverWrapper(void)
	{
	}

	bool Start(void)
	{
		return this->get_override("Start")();
	}
	
	bool Stop(void)
	{
		return this->get_override("Stop")();
	}
	
	daeDataReporterProcess*	GetProcess(void)
	{
		return this->get_override("GetProcess")();
	}

	void GetProcessName(string& strProcessName)
	{
	}
	
	void GetDomains(std::vector<const daeDataReceiverDomain*>& ptrarrDomains) const
	{
	}
	
	void GetVariables(std::map<string, const daeDataReceiverVariable*>& ptrmapVariables) const
	{
	}
};

class daeTCPIPDataReceiverWrapper : public daeTCPIPDataReceiver,
	                                public boost::python::wrapper<daeTCPIPDataReceiver>
{
public:
	bool Start()
	{
        if(boost::python::override f = this->get_override("Start"))
            return f();
		else
			return this->daeTCPIPDataReceiver::Start();
	}
	bool def_Start()
	{
        return this->daeTCPIPDataReceiver::Start();
	}

	bool Stop()
	{
        if(boost::python::override f = this->get_override("Stop"))
            return f();
		else
			return this->daeTCPIPDataReceiver::Stop();
	}
	bool def_Stop()
	{
        return this->daeTCPIPDataReceiver::Stop();
	}

	daeDataReporterProcess* GetProcess()
	{
        if(boost::python::override f = this->get_override("GetProcess"))
            return f();
		else
			return this->daeTCPIPDataReceiver::GetProcess();
	}
	daeDataReporterProcess* def_GetProcess()
	{
        return this->daeTCPIPDataReceiver::GetProcess();
	}
};
	
class daeTCPIPDataReceiverServerWrapper : public daeTCPIPDataReceiverServer,
	                                      public boost::python::wrapper<daeTCPIPDataReceiverServer>
{
public:
	daeTCPIPDataReceiverServerWrapper(int nPort) : daeTCPIPDataReceiverServer(nPort)
	{
	}

    void Start_(void)
    {
        this->daeTCPIPDataReceiverServer::Start();
    }

    void Stop_(void)
    {
        this->daeTCPIPDataReceiverServer::Stop();
    }

	bool IsConnected_(void)
	{
        return this->daeTCPIPDataReceiverServer::IsConnected();
	}

    size_t GetNumberOfDataReceivers(void)
	{
		return m_ptrarrDataReceivers.size();
	}
	
	daeDataReceiver_t* GetDataReceiver(size_t nIndex)
	{
		return m_ptrarrDataReceivers[nIndex];
	}

	boost::python::list GetDataReceivers(void)
	{
		boost::python::list l;
		daeTCPIPDataReceiver* obj;
	
		for(size_t i = 0; i < m_ptrarrDataReceivers.size(); i++)
		{
			obj = m_ptrarrDataReceivers[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	size_t GetNumberOfProcesses(void)
	{
		return m_ptrarrDataReceivers.size();
	}
	
	daeDataReporterProcess* GetProcess(size_t nIndex)
	{
		return m_ptrarrDataReceivers[nIndex]->GetProcess();
	}
	
	boost::python::list GetProcesses(void)
	{
		boost::python::list l;
		boost::python::object o;
		daeTCPIPDataReceiver* pDataReceiver;
		daeDataReporterProcess* obj;
	
		for(size_t i = 0; i < m_ptrarrDataReceivers.size(); i++)
		{
			pDataReceiver = m_ptrarrDataReceivers[i];
			obj = pDataReceiver->GetProcess();
			l.append(boost::ref(obj));
		}
		return l;
	}
};	

class daeHybridDataReporterReceiverWrapper : public daeHybridDataReporterReceiver,
	                                         public boost::python::wrapper<daeHybridDataReporterReceiver>
{
public:
	daeHybridDataReporterReceiverWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeHybridDataReporterReceiver::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeHybridDataReporterReceiver::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeHybridDataReporterReceiver::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeHybridDataReporterReceiver::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeHybridDataReporterReceiver::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeHybridDataReporterReceiver::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeHybridDataReporterReceiver::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeHybridDataReporterReceiver::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeHybridDataReporterReceiver::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeHybridDataReporterReceiver::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeHybridDataReporterReceiver::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeHybridDataReporterReceiver::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeHybridDataReporterReceiver::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeHybridDataReporterReceiver::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeHybridDataReporterReceiver::SendVariable(pVariableValue);
	}

	bool Start()
	{
        if(boost::python::override f = this->get_override("Start"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::Start();
	}
	bool def_Start()
	{
        return this->daeHybridDataReporterReceiver::Start();
	}

	bool Stop()
	{
        if(boost::python::override f = this->get_override("Stop"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::Stop();
	}
	bool def_Stop()
	{
        return this->daeHybridDataReporterReceiver::Stop();
	}

	daeDataReporterProcess* GetProcess()
	{
        if(boost::python::override f = this->get_override("GetProcess"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::GetProcess();
	}
	daeDataReporterProcess* def_GetProcess()
	{
        return this->daeHybridDataReporterReceiver::GetProcess();
	}
};


/*******************************************************
	daeDataReceiverDomain
*******************************************************/
boost::python::list GetDataReceiverDomainPoints(daeDataReceiverDomain& domain);

/*******************************************************
	daeDataReceiverVariable
*******************************************************/
boost::python::numeric::array GetNumPyArrayDataReceiverVariable(daeDataReceiverVariable& var);
boost::python::numeric::array GetTimeValuesDataReceiverVariable(daeDataReceiverVariable& var);

boost::python::list GetDomainsDataReceiverVariable(daeDataReceiverVariable& var);

/*******************************************************
	daeDataReporterProcess
*******************************************************/
boost::python::list GetDomainsDataReporterProcess(daeDataReporterProcess& process);
boost::python::list GetVariablesDataReporterProcess(daeDataReporterProcess& process);

}

#endif
