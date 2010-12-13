#include "stdafx.h"
#include "datareporters.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include "../Core/helpers.h"
#include <boost/cstdint.hpp>
using namespace std;

namespace dae
{
namespace datareporting
{

daeTCPIPDataReporter::daeTCPIPDataReporter()
{
}

daeTCPIPDataReporter::~daeTCPIPDataReporter()
{
	Disconnect();
}

bool daeTCPIPDataReporter::Connect(const string& strConnectString, const string& strProcessName)
{
	size_t nPort;
	string strIPAddress;
	boost::system::error_code ec;

	if(strConnectString.empty())
	{
		daeConfig& cfg = daeConfig::GetConfig();
		strIPAddress = cfg.Get<string>("daetools.datareporting.tcpipDataReceiverAddress", "127.0.0.1");
		nPort        = cfg.Get<int>("daetools.datareporting.tcpipDataReceiverPort", 50000);
	}
	else
	{
		vector<string> arrAddress = dae::ParseString(strConnectString, ':');
		if(arrAddress.size() != 2)
		{
			cout << "Cannot parse the connection string: " << strConnectString << endl;
			return false;
		}
		strIPAddress = arrAddress[0];
		nPort        = dae::fromString<size_t>(arrAddress[1]);
	}
	
	m_ptcpipSocket.reset(new tcp::socket(m_ioService));
	
	tcp::endpoint endpoint(boost::asio::ip::address::from_string(strIPAddress), nPort);
	m_ptcpipSocket->connect(endpoint, ec);
	
	if(ec)
	{
//		if(ec.value() == boost::system::errc::broken_pipe ||
//		   ec.value() == boost::system::errc::address_in_use ||
//		   ec.value() == boost::system::errc::bad_address ||
//		   ec.value() == boost::system::errc::address_family_not_supported ||
//		   ec.value() == boost::system::errc::address_not_available ||
//		   ec.value() == boost::system::errc::already_connected ||
//		   ec.value() == boost::system::errc::connection_refused ||
//		   ec.value() == boost::system::errc::connection_aborted ||
//		   ec.value() == boost::system::errc::network_down ||
//		   ec.value() == boost::system::errc::network_unreachable ||
//		   ec.value() == boost::system::errc::host_unreachable ||
//		   ec.value() == boost::system::errc::broken_pipe )
		cout << "Cannot connect TCPIPDataReporter: " << ec.message() << endl;
		return false;
	}

	m_strConnectionString = strConnectString;
	m_strProcessName      = strProcessName;
	
	return true;
}

bool daeTCPIPDataReporter::IsConnected()
{
	if(m_ptcpipSocket && m_ptcpipSocket->is_open())
		return true;
	else
		return false;
}

bool daeTCPIPDataReporter::Disconnect()
{
//	boost::system::error_code ec;
//
//	if(!IsConnected())
//		return true;
//	
//	m_ptcpipSocket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
//	if(ec)
//	{
//		cout << "Error while shutting down daeTCPIPDataReporter: " << ec.message() << endl;
//		return false;
//	}
//
//	m_ptcpipSocket->close(ec);
//	if(ec)
//	{
//		cout << "Error while closing daeTCPIPDataReporter: " << ec.message() << endl;
//		return false;
//	}

	m_ptcpipSocket.reset();
	
	return true;
}

bool daeTCPIPDataReporter::SendMessage(const string& strMessage)
{
	boost::system::error_code ec;
	
// Gather the message data
	boost::int32_t msgSize = strMessage.size();

// First send the size
	boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(&msgSize, sizeof(msgSize)), boost::asio::transfer_all(), ec);
	if(ec)
	{
		cout << "Error while writing the message size in daeTCPIPDataReporter: " << ec.message() << endl;
		return false;
	}

// Then send the message
	size_t nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(strMessage.c_str(), msgSize), boost::asio::transfer_all(), ec);
	if(ec)
	{
		cout << "Error while writing the message in daeTCPIPDataReporter: " << ec.message() << endl;
		return false;
	}

	return ((boost::int32_t)nSent == msgSize);
}

/*
bool daeTCPIPDataReporter::StartRegistration(void)
{
	if(!IsConnected())
		return false;

	boost::int32_t msgSize, nameSize;
	stringstream s(ios_base::out|ios_base::in|ios_base::binary);
	
// Send cStartRegistration flag
	s.write((const char*)(void*)&cStartRegistration, sizeof(char));
	
// Send the size of the ProcessName, and the ProcessName itself
	nameSize = m_strProcessName.size();
	s.write((const char*)(void*)&nameSize, sizeof(nameSize));
	s.write(m_strProcessName.c_str(), nameSize);

// Flush the buffer
	s.flush();
	
// Gather the message data
	string strMessage = s.str();
	msgSize = strMessage.size();

// First send the size
	boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(&msgSize, sizeof(msgSize)), boost::asio::transfer_all());

// Then send the message
	size_t nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(strMessage.c_str(), msgSize), boost::asio::transfer_all());

	return true;
}
	
bool daeTCPIPDataReporter::EndRegistration(void)
{
	if(!IsConnected())
		return false;

	boost::int32_t msgSize;
	stringstream s(ios_base::out|ios_base::in|ios_base::binary);
	
// Send cEndRegistration flag
	s.write((const char*)(void*)&cEndRegistration, sizeof(char));
	
// Send the dummy byte
	char nothing = 0;
	s.write((const char*)(void*)&nothing, sizeof(nothing));

// Flush the buffer
	s.flush();
	
// Gather the message data
	string strMessage = s.str();
	msgSize = strMessage.size();

// First send the size
	boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(&msgSize, sizeof(msgSize)), boost::asio::transfer_all());

// Then send the message
	size_t nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(strMessage.c_str(), msgSize), boost::asio::transfer_all());

	return true;
}

bool daeTCPIPDataReporter::RegisterDomain(const daeDataReporterDomain* pDomain)
{
	if(!IsConnected())
		return false;

	boost::int32_t msgSize, nameSize, noPoints, type;
	stringstream s(ios_base::out|ios_base::in|ios_base::binary);
	
// Send RegisterDomain flag
	s.write((const char*)(void*)&cRegisterDomain, sizeof(char));
	
// Send the size of the Name, and the Name itself
	nameSize = pDomain->m_strName.size();
	s.write((const char*)(void*)&nameSize, sizeof(nameSize));
	s.write(pDomain->m_strName.c_str(), nameSize);
	
// Send the domain type
	type = pDomain->m_eType;
	s.write((const char*)(void*)&type, sizeof(type));

// Send the number of points
	noPoints = pDomain->m_nNumberOfPoints;
	s.write((const char*)(void*)&noPoints, sizeof(noPoints));
	
// Send the points
	s.write((const char*)(void*)pDomain->m_pPoints, pDomain->m_nNumberOfPoints * sizeof(real_t));
	
// Flush the buffer
	s.flush();
	
// Gather the message data
	string strMessage = s.str();
	msgSize = strMessage.size();

// First send the size
	boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(&msgSize, sizeof(msgSize)), boost::asio::transfer_all());

// Then send the message
	size_t nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(strMessage.c_str(), msgSize), boost::asio::transfer_all());

	return true;
}

bool daeTCPIPDataReporter::RegisterVariable(const daeDataReporterVariable* pVariable)
{
	if(!IsConnected())
		return false;

	boost::int32_t msgSize, nameSize, domainsSize, noPoints;
	stringstream s(ios_base::out|ios_base::in|ios_base::binary);
	
// Send RegisterVariable flag
	s.write((const char*)(void*)&cRegisterVariable, sizeof(char));

// Send the size of the Name, and the Name itself
	nameSize = pVariable->m_strName.size();
	s.write((const char*)(void*)&nameSize, sizeof(nameSize));
	s.write(pVariable->m_strName.c_str(), nameSize);

// Send the number of points
	noPoints = pVariable->m_nNumberOfPoints;
	s.write((const char*)(void*)&noPoints, sizeof(noPoints));
	
// Send the size of the array with domain names, and array itself
	domainsSize = pVariable->m_strarrDomains.size();
	s.write((const char*)(void*)&domainsSize, sizeof(domainsSize));	
	for(size_t i = 0; i < domainsSize; i++)
	{
		nameSize = pVariable->m_strarrDomains[i].size();
		s.write((const char*)(void*)&nameSize,         sizeof(nameSize));
		s.write(pVariable->m_strarrDomains[i].c_str(), nameSize);
	}
	
// Flush the buffer
	s.flush();
	
// Gather the message data
	string strMessage = s.str();
	msgSize = strMessage.size();

// First send the size
	boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(&msgSize, sizeof(msgSize)), boost::asio::transfer_all());

// Then send the message
	size_t nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(strMessage.c_str(), msgSize), boost::asio::transfer_all());

	return true;
}

bool daeTCPIPDataReporter::StartNewResultSet(real_t dTime)
{
	if(!IsConnected())
		return false;

	m_dCurrentTime = dTime;

	boost::int32_t msgSize;
	stringstream s(ios_base::out|ios_base::in|ios_base::binary);
	
// Send cStartNewTime flag
	s.write((const char*)(void*)&cStartNewTime, sizeof(char));
	
// Send the Time
	s.write((const char*)(void*)&dTime, sizeof(dTime));

// Flush the buffer
	s.flush();
	
// Gather the message data
	string strMessage = s.str();
	msgSize = strMessage.size();

// First send the size
	boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(&msgSize, sizeof(msgSize)), boost::asio::transfer_all());

// Then send the message
	size_t nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(strMessage.c_str(), msgSize), boost::asio::transfer_all());

	return true;
}

bool daeTCPIPDataReporter::EndOfData()
{
	if(!IsConnected())
		return false;

	m_dCurrentTime = -1;

	boost::int32_t msgSize;
	stringstream s(ios_base::out|ios_base::in|ios_base::binary);
	
// Send cEndOfData flag
	s.write((const char*)(void*)&cEndOfData, sizeof(char));
	
// Send the dummy byte
	char nothing = 0;
	s.write((const char*)(void*)&nothing, sizeof(nothing));

// Flush the buffer
	s.flush();
	
// Gather the message data
	string strMessage = s.str();
	msgSize = strMessage.size();

// First send the size
	boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(&msgSize, sizeof(msgSize)), boost::asio::transfer_all());

// Then send the message
	size_t nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(strMessage.c_str(), msgSize), boost::asio::transfer_all());

	return true;
}

bool daeTCPIPDataReporter::SendVariable(const daeDataReporterVariableValue* pVariableValue)
{
	if(!IsConnected())
		return false;
	if(m_dCurrentTime < 0)
		return false;

	boost::int32_t msgSize, nameSize, noPoints;
	stringstream s(ios_base::out|ios_base::in|ios_base::binary);
	
// Send cSendVariable flag
	s.write((const char*)(void*)&cSendVariable, sizeof(char));

// Send the size of the Name, and the Name itself
	nameSize = pVariableValue->m_strName.size();
	s.write((const char*)(void*)&nameSize, sizeof(nameSize));
	s.write(pVariableValue->m_strName.c_str(), nameSize);

// Send the number of points
	noPoints = pVariableValue->m_nNumberOfPoints;
	s.write((const char*)(void*)&noPoints, sizeof(noPoints));
	
// Send the points
	s.write((const char*)(void*)pVariableValue->m_pValues, pVariableValue->m_nNumberOfPoints * sizeof(real_t));
	
// Flush the buffer
	s.flush();
	
// Gather the message data
	string strMessage = s.str();
	msgSize = strMessage.size();

// First send the size
	boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(&msgSize, sizeof(msgSize)), boost::asio::transfer_all());

// Then send the message
	size_t nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(strMessage.c_str(), msgSize), boost::asio::transfer_all());

	return true;
}
*/

}
}
