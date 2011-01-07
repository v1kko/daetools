#include "stdafx.h"
#include "tcpiplog.h"

namespace dae
{
namespace logging
{
/********************************************************************
	daeTCPIPLog
*********************************************************************/
daeTCPIPLog::daeTCPIPLog(string strIPAddress, int nPort)
{
	m_nPort	       = nPort;
	m_strIPAddress = strIPAddress;

	m_ptcpipSocket = new tcp::socket(m_ioService);
	tcp::endpoint endpoint(boost::asio::ip::address::from_string(strIPAddress), nPort);
	boost::system::error_code ec;
	m_ptcpipSocket->connect(endpoint, ec);
	
	if(ec)
	{
		cout << "Error connecting TCPIPLog: " << ec.message() << endl;
	}
}

daeTCPIPLog::~daeTCPIPLog(void)
{
	m_ptcpipSocket->shutdown(boost::asio::ip::tcp::socket::shutdown_both);
	m_ptcpipSocket->close();
	if(m_ptcpipSocket)
		delete m_ptcpipSocket;
	m_ptcpipSocket = NULL;
}

void daeTCPIPLog::Message(const string& strMessage, size_t nSeverity)
{
	if(m_ptcpipSocket && m_ptcpipSocket->is_open() && m_bEnabled)
	{
		if(strMessage.empty())
			return;
		
	// Add indent
		string msg = m_strIndent + strMessage;
		
	// Gather the message data
		boost::int32_t msgSize = msg.size();
	
	// First send the size
		size_t nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(&msgSize, sizeof(msgSize)), boost::asio::transfer_all());
	
	// Then send the message
		nSent = boost::asio::write(*m_ptcpipSocket, boost::asio::buffer(msg.c_str(), msgSize), boost::asio::transfer_all());
	}
}

/********************************************************************
	daeTCPIPLogServer
*********************************************************************/
daeTCPIPLogServer::daeTCPIPLogServer(int nPort) : m_nPort(nPort), 
											      m_ioService(), 
											      m_acceptor(m_ioService, tcp::endpoint(tcp::v4(), nPort)), 
											      m_tcpipSocket(m_ioService)
{
	m_pThread = new boost::thread(boost::bind(&daeTCPIPLogServer::thread, this));
}

daeTCPIPLogServer::~daeTCPIPLogServer(void)
{
	if(m_pThread)
	{
		delete m_pThread;
		m_pThread = NULL;
	}
}

void daeTCPIPLogServer::thread()
{
	try
	{
		m_acceptor.accept(m_tcpipSocket);

		boost::system::error_code error;
		boost::int32_t msgSize = 0, 
					   nRead   = 0;
		
		for(;;)
		{
			if(!m_tcpipSocket.is_open())
			{
				cout << "Socket disconnected!" << endl;
				return;
			}
			
			nRead = 0;
			nRead = boost::asio::read(m_tcpipSocket, 
									  boost::asio::buffer(&msgSize, sizeof(msgSize)), 
									  boost::asio::transfer_at_least(sizeof(msgSize)), 
									  error); 
			if(nRead == 0)
			{
				cout << "TCPIPLog socket disconnected, closing the receiving socket..." << endl;
				return;
			}
			if(nRead != sizeof(msgSize))
			{
				cout << "Cannot read the message length (4 bytes long); Read = " << nRead << "bytes; "
					 << "Error: " << error.message() << endl;
				return;
			}
			
			//cout << "Message size is " << msgSize << endl;
			if(msgSize <= 0)
			{
				cout << "Message size is 0!!" << endl;
				return;
			}
			
			char* data = new char[msgSize+1];
			data[msgSize] = '\0';
			nRead = 0;
			nRead = boost::asio::read(m_tcpipSocket, 
									  boost::asio::buffer(data, msgSize), 
									  boost::asio::transfer_at_least(msgSize),
									  error); 
			if(nRead == 0)
			{
				cout << "Reporting cocket disconnected, closing the receiving socket..." << endl;
				return;
			}
			if(nRead != msgSize)
			{
				delete[] data;
				data = NULL;
				cout << "Message not read completely; " << nRead << " bytes has been read; "
					 << "Error: " << error.message() << endl;
				return;				
			}
				
			//cout << "Successfully read " << msgSize << " bytes" << endl;
			//cout << "Message is: ";
			
			this->MessageReceived(data);

			delete[] data;
			data = NULL;
		}
	}
	catch (std::exception& e)
	{
		cout << "Exception in thread: " << e.what() << "\n";
	}
}

}
}
