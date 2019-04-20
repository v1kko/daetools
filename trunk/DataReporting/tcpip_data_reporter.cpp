#include "stdafx.h"
#include "datareporters.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include "../Core/helpers.h"
#include <boost/cstdint.hpp>
#include <boost/process/spawn.hpp>
using namespace std;

namespace daetools
{
namespace datareporting
{

daeTCPIPDataReporter::daeTCPIPDataReporter()
{
    m_strName = "TCPIPDataReporter";
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

    daeConfig& cfg = daeConfig::GetConfig();

    // Parse the connect string.
    if(strConnectString.empty())
    {
        strIPAddress = cfg.GetString("daetools.datareporting.tcpipDataReceiverAddress", "127.0.0.1");
        nPort        = cfg.GetInteger("daetools.datareporting.tcpipDataReceiverPort", 50000);
    }
    else
    {
        vector<string> arrAddress = daetools::ParseString(strConnectString, ':');
        if(arrAddress.size() != 2)
        {
            cout << "Cannot parse the connection string: " << strConnectString << endl;
            return false;
        }
        strIPAddress = arrAddress[0];
        nPort        = daetools::fromString<size_t>(arrAddress[1]);
    }

    // Create the TCP/IP socket.
    m_ptcpipSocket.reset(new tcp::socket(m_ioService));

    // Try to connect.
    int numberOfRetries     = cfg.GetInteger("daetools.datareporting.tcpipNumberOfRetries",     10);
    int retryAfterMilliSecs = cfg.GetInteger("daetools.datareporting.tcpipRetryAfterMilliSecs", 1000);

    // Make numberOfRetries attempts to connect.
    // Wait for retryAfter milliseconds before the next attempt.
    bool plotterStarted = false;
    for(int i = 0; i < numberOfRetries; i++)
    {
        ec = boost::asio::error::host_not_found;
        tcp::endpoint endpoint(boost::asio::ip::address::from_string(strIPAddress), nPort);
        m_ptcpipSocket->connect(endpoint, ec);

        // Break the for loop if the connection has been established.
        if(!ec)
            break;

        // Failed to connect (it could be that the DAE Plotter application has not been started).
        // Thus, first time we fail to connect try to start the DAE Plotter application and
        // retry to connect after some time.
        if(ec && !plotterStarted)
        {
#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
            boost::process::spawn("python -m daetools.dae_plotter.plotter");
#elif defined(__linux__)
            boost::process::spawn("python -m daetools.dae_plotter.plotter");
#elif defined(__unix__) || defined(__APPLE__)
            // There is a problem with the boost::process::spawn function in macOS.
#endif
            plotterStarted = true;
        }

        // Wait for retryAfterMilliSecs ms before the next attempt.
        boost::this_thread::sleep(boost::posix_time::milliseconds(retryAfterMilliSecs));
    }
    if(ec)
    {
        cout << "Cannot connect TCPIPDataReporter: " << ec.message() << endl;
        return false;
    }

    m_strConnectString = strConnectString;
    m_strProcessName   = strProcessName;

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
    std::int32_t msgSize = strMessage.size();

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

    return ((std::int32_t)nSent == msgSize);
}


}
}
