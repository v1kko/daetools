#if !defined(TCPIP_REPORTER_H)
#define TCPIP_REPORTER_H


#include "reporterinterface.h"

namespace drDataReporter
{
const int cMaxMessageSize = 65535;

class drClientMessageParser
{
public:
	drClientMessageParser(){}
	virtual ~drClientMessageParser(){}

public:
	virtual bool MessageFromClient(unsigned char* pData, size_t iSize) = 0;
};

enum
{
  SERVER_ID = 100,
  SOCKET_ID,
  TCPIP_SERVER_WINDOW_ID,
  TCPIP_CLIENT_WINDOW_ID
};

template <class T>
class tcpPtrVector : public std::vector<T>
{
public:
	virtual ~tcpPtrVector()
	{
		EmptyAndFreeMemory();
	};

	virtual void EmptyAndFreeMemory()
	{
		T pType;

		for(size_t i = 0; i < std::vector<T>::size(); i++)
		{
			pType = std::vector<T>::at(i);
			if(pType)
				delete pType;
			pType = NULL;
		}
		std::vector<T>::clear();
	};	
};



class tcpClientEventHandler: public wxEvtHandler
{
public:
    tcpClientEventHandler();
	virtual ~tcpClientEventHandler();

public:
	void OnSocketEvent(wxSocketEvent& event);

public:
	wxSocketClient* m_tcpipClient;
	wxStreamBuffer* m_pStreamBuffer;

private:
	DECLARE_EVENT_TABLE()
};

class tcpReportingClient : public drReportingClient
{
public:
	tcpReportingClient();
	virtual ~tcpReportingClient();

public:
	virtual bool ConnectToServer(string strServer, unsigned int nPort);
	virtual bool Disconnect();
	virtual bool SendData(unsigned char* pData, size_t iLength);

public:
    tcpClientEventHandler*	m_pEventHandler;
	wxSocketClient*			m_tcpipClient;
	string					m_strServer;
	unsigned int			m_nPort;

	long					m_nReadWriteTimeout;
	long					m_nConnectionTimeout;
};

class tcpServerEventHandler: public wxEvtHandler
{
public:
    tcpServerEventHandler();
	virtual ~tcpServerEventHandler();

public:
	void OnServerEvent(wxSocketEvent& event);
	void OnSocketEvent(wxSocketEvent& event);

public:
	wxSocketServer*				m_tcpipServer;
	tcpPtrVector<wxSocketBase*>	m_ptrarrClients;
	long						m_nReadWriteTimeout;

	unsigned char*				m_pReadBuffer;
	drClientMessageParser*		m_pDataParser;

private:
	DECLARE_EVENT_TABLE()
};

class tcpReportingServer : public drReportingServer
{
public:
	tcpReportingServer();
	virtual ~tcpReportingServer();

public:
	virtual bool Start(unsigned int nPort);
	virtual bool Stop();
	virtual void SetParser(drClientMessageParser* pClientMessageParser);

public:
    tcpServerEventHandler*	m_pEventHandler;
	wxSocketServer*			m_tcpipServer;
	unsigned int			m_nPort;
	drClientMessageParser*	m_pDataParser;
};

}



#endif
