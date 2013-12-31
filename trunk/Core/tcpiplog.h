/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_TCPIP_LOG_H
#define DAE_TCPIP_LOG_H

#if defined(__MINGW32__)
#include <winsock2.h>	// Problem with: error Winsock.h has aready been included
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include "base_logging.h"
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
using boost::asio::ip::tcp;

namespace dae
{
namespace logging
{
/********************************************************************
	daeTCPIPLog
*********************************************************************/
class daeTCPIPLog : public daeBaseLog
{
public:
	daeTCPIPLog(void);
	virtual ~daeTCPIPLog(void);

public:
	virtual void Message(const string& strMessage, size_t nSeverity);
    
public:
    bool Connect(const string& strIPAddress, int nPort);
	bool Disconnect(void);
	bool IsConnected(void);

protected:
	int						       m_nPort;
	string					       m_strIPAddress;
	boost::shared_ptr<tcp::socket> m_ptcpipSocket;
	boost::asio::io_service        m_ioService;
};

/********************************************************************
	daeTCPIPLogServer
*********************************************************************/
class daeTCPIPLogServer
{
public:
	daeTCPIPLogServer(int nPort);
	virtual ~daeTCPIPLogServer(void);
	
public:
    virtual void MessageReceived(const char* strMessage);

public:
    int  GetPort() const;
    void Start(void);
    void Stop(void);

protected:
    void thread(void);

public:
	int						         m_nPort;
	boost::asio::io_service          m_ioService;
	tcp::acceptor			         m_acceptor;
	tcp::socket                      m_tcpipSocket;
	boost::shared_ptr<boost::thread> m_pThread;
};


}
}

#endif // TCPIPLOG_H
