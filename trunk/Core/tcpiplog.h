#ifndef DAE_TCPIP_LOG_H
#define DAE_TCPIP_LOG_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include "log.h"
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
using boost::asio::ip::tcp;

namespace dae
{
namespace logging
{
/********************************************************************
	daeTCPIPLog
*********************************************************************/
class daeTCPIPLog : public daeLog_t
{
public:
	daeTCPIPLog(string strIPAddress, int nPort);
	virtual ~daeTCPIPLog(void);

public:
	virtual void Message(const string& strMessage, size_t nSeverity);

protected:
	int						m_nPort;
	string					m_strIPAddress;
	tcp::socket*            m_ptcpipSocket;
	boost::asio::io_service m_ioService;
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
	virtual void MessageReceived(const char* strMessage) = 0;
	void thread(void);

public:
	int						m_nPort;
	boost::asio::io_service m_ioService;
	tcp::acceptor			m_acceptor;
	tcp::socket				m_tcpipSocket;
	boost::thread*			m_pThread;
};


}
}

#endif // TCPIPLOG_H
