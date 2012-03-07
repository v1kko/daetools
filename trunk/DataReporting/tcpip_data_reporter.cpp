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
	if(IsConnected())
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
	
	return SendProcessName(strProcessName);
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

	if(!IsConnected())
		return false;

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


}
}
