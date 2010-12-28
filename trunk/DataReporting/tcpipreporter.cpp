#include "stdafx.h"
#include "tcpipreporter.h"


namespace drDataReporter
{

tcpReportingClient::tcpReportingClient()
{
	m_nReadWriteTimeout  = 20;
	m_nConnectionTimeout = 20;

	m_tcpipClient	= NULL;
	m_strServer		= "localhost";
	m_nPort			= 5000;

	m_tcpipClient	= new wxSocketClient();
	//m_pEventHandler = new tcpClientEventHandler;
	//m_pEventHandler->m_tcpipClient = m_tcpipClient;

}

tcpReportingClient::~tcpReportingClient()
{
	if(m_tcpipClient)
		m_tcpipClient->Destroy();
	//if(m_pEventHandler)
	//	delete m_pEventHandler;
}

bool tcpReportingClient::ConnectToServer(string strServer, unsigned int nPort)
{
	wxIPV4address IPAddress;

	m_strServer = strServer;
	m_nPort     = nPort;

	if(m_tcpipClient->IsConnected())
		m_tcpipClient->Close();

	// Setup the event handler and subscribe to most events
	//m_tcpipClient->SetEventHandler(*m_pEventHandler, SOCKET_ID);
	//m_tcpipClient->SetNotify(wxSOCKET_CONNECTION_FLAG |
	//						 wxSOCKET_INPUT_FLAG      |
	//						 wxSOCKET_LOST_FLAG);
	//m_tcpipClient->Notify(true);

	IPAddress.Hostname(strServer);
	IPAddress.Service(nPort);

	m_tcpipClient->Connect(IPAddress, false);

	m_tcpipClient->WaitOnConnect(m_nConnectionTimeout);
	if(!m_tcpipClient->IsConnected())
	{
		m_tcpipClient->Close();
		wxMessageBox(wxT("Can't connect to the specified host"), wxT("Alert !"));
		return false;
	}

	m_tcpipClient->SetFlags(wxSOCKET_WAITALL);
	m_tcpipClient->SetTimeout(m_nReadWriteTimeout);

	return true;
}

bool tcpReportingClient::Disconnect()
{
	if(m_tcpipClient && m_tcpipClient->IsConnected())
		m_tcpipClient->Close();
	return true;
}

bool tcpReportingClient::SendData(unsigned char* pData, size_t iLength)
{
	wxUint32 iRead;
	char szConfirmBuffer[4];
	szConfirmBuffer[3] = '\0';

	m_tcpipClient->WriteMsg(pData, (wxUint32)iLength);
	if(m_tcpipClient->Error())
	{
		wxSocketError e = m_tcpipClient->LastError();
		wxMessageBox(wxT("WriteMsg failed"));
		return false;
	}

	// Wait until data available (will also return if the connection is lost)
	if(!m_tcpipClient->WaitForRead())
	{
		wxMessageBox(wxT("SendData: No acknowledge from the server"));
		return false;
	}

	iRead = 3;
	if(m_tcpipClient->IsData())
	{
		m_tcpipClient->ReadMsg(&szConfirmBuffer, iRead);
		if(m_tcpipClient->Error())
		{
			wxMessageBox(wxT("ReadMsg failed"));
			return false;
		}
		wxMessageBox(szConfirmBuffer);
		if(strcmp(szConfirmBuffer, wxT("{0}")) != 0)
			return false;
	}

	return true;
}

BEGIN_EVENT_TABLE(tcpClientEventHandler, wxEvtHandler)
	EVT_SOCKET(SOCKET_ID,  tcpClientEventHandler::OnSocketEvent)
END_EVENT_TABLE()


tcpClientEventHandler::tcpClientEventHandler() 
{
	m_tcpipClient = NULL;
}

tcpClientEventHandler::~tcpClientEventHandler()
{
}

void tcpClientEventHandler::OnSocketEvent(wxSocketEvent& event)
{
	wxString s = _("OnSocketEvent: ");

	switch(event.GetSocketEvent())
	{
		case wxSOCKET_INPUT      : s.Append(_("wxSOCKET_INPUT\n")); break;
		case wxSOCKET_LOST       : s.Append(_("wxSOCKET_LOST\n")); break;
		case wxSOCKET_CONNECTION : s.Append(_("wxSOCKET_CONNECTION\n")); break;
		default                  : s.Append(_("Unexpected event !\n")); break;
	}
	wxMessageBox(s);
}

/******************************************************************
* TCP/IP Server section
*******************************************************************/
tcpReportingServer::tcpReportingServer()
{
	m_nPort			= 5000;
	m_tcpipServer	= NULL;

    m_pEventHandler = new tcpServerEventHandler;
	m_pDataParser   = NULL;
}

tcpReportingServer::~tcpReportingServer()
{
	if(m_tcpipServer)
		m_tcpipServer->Destroy();
	if(m_pEventHandler)
		delete m_pEventHandler;
}

bool tcpReportingServer::Start(unsigned int nPort)
{
	wxIPV4address IPAddress;

	m_nPort = nPort;
	IPAddress.Service(m_nPort);
	if(!IPAddress.AnyAddress())
		return false;

	if(m_tcpipServer)
	{
		m_tcpipServer->Close();
		m_tcpipServer->Destroy();
		m_tcpipServer = NULL;
	}
	m_tcpipServer = new wxSocketServer(IPAddress);
	if(!m_tcpipServer)
		return false;

	m_pEventHandler->m_tcpipServer = m_tcpipServer;
	m_pEventHandler->m_pDataParser = m_pDataParser;

	// We use Ok() here to see if the server is really listening
	if(!m_tcpipServer->Ok())
		return false;

	// Setup the event handler and subscribe to connection events
	m_tcpipServer->SetEventHandler(*m_pEventHandler, SERVER_ID);
	m_tcpipServer->SetNotify(wxSOCKET_CONNECTION_FLAG);
	m_tcpipServer->Notify(true);

	return true;
}

bool tcpReportingServer::Stop()
{
	if(!m_tcpipServer)
		return false;
	m_tcpipServer->Close();
	return true;
}

void tcpReportingServer::SetParser(drClientMessageParser* pClientMessageParser)
{
	m_pDataParser = pClientMessageParser;
}



BEGIN_EVENT_TABLE(tcpServerEventHandler, wxEvtHandler)
	EVT_SOCKET(SERVER_ID,  tcpServerEventHandler::OnServerEvent)
	EVT_SOCKET(SOCKET_ID,  tcpServerEventHandler::OnSocketEvent)
END_EVENT_TABLE()


tcpServerEventHandler::tcpServerEventHandler() 
{
	m_pReadBuffer		= new unsigned char[cMaxMessageSize];
	m_nReadWriteTimeout	= 20;
	m_tcpipServer		= NULL;
	m_pDataParser		= NULL;
}

tcpServerEventHandler::~tcpServerEventHandler()
{
	if(m_pReadBuffer)
		delete[] m_pReadBuffer;
}

void tcpServerEventHandler::OnServerEvent(wxSocketEvent& event)
{
	wxSocketBase *pSocket;

	if(event.GetSocketEvent() == wxSOCKET_CONNECTION)
	{
		//wxMessageBox(wxT("New client connection accepted"));
	}
	else
	{
		wxMessageBox(wxT("Unknown event"));
		return;
	}

	pSocket = m_tcpipServer->Accept(false);
	if(!pSocket)
	{
		wxMessageBox(wxT("New client rejected"));
		return;
	}
	//m_ptrarrClients.push_back(pSocket);

	pSocket->SetEventHandler(*this, SOCKET_ID);
	pSocket->SetNotify(wxSOCKET_INPUT_FLAG | wxSOCKET_LOST_FLAG);
	pSocket->Notify(true);

	pSocket->SetFlags(wxSOCKET_WAITALL);
	pSocket->SetTimeout(m_nReadWriteTimeout);

}

void tcpServerEventHandler::OnSocketEvent(wxSocketEvent& event)
{
	wxSocketBase* pSocket = event.GetSocket();

	bool bSuccess;
	wxUint32 iRead, iWrite;
	char* szConfirmBuffer;

	if(!m_pDataParser)
	{
		wxMessageBox(wxT("Data parser is not set"));
		return;
	}

	switch(event.GetSocketEvent())
	{
	case wxSOCKET_INPUT:
		// We disable input events, so that the test doesn't trigger wxSocketEvent again.
		pSocket->SetNotify(wxSOCKET_LOST_FLAG);

		// We don't need to set flags because ReadMsg and WriteMsg are not affected by them anyway.

		// Read the message
		pSocket->ReadMsg(m_pReadBuffer, cMaxMessageSize);
		if(pSocket->Error())
		{
			wxSocketError e = pSocket->LastError();
			wxMessageBox(wxT("ReadMsg failed"));
			return;
		}
		iRead = pSocket->LastCount();

		bSuccess = m_pDataParser->MessageFromClient(m_pReadBuffer, iRead);

		if(!pSocket->WaitForWrite())
		{
			wxMessageBox(wxT("Cannot write response from the server"));
			return;
		}

		iWrite = 3;
		szConfirmBuffer = (char*)(bSuccess ? wxT("{1}") : wxT("{0}"));
		pSocket->WriteMsg(szConfirmBuffer, iWrite);
		if(pSocket->Error())
		{
			wxSocketError e = pSocket->LastError();
			wxMessageBox(wxT("WriteMsg confirmation failed"));
			return;
		}

		// Enable input events again.
		pSocket->SetNotify(wxSOCKET_LOST_FLAG | wxSOCKET_INPUT_FLAG);
		break;

	case wxSOCKET_LOST:
			// Destroy() should be used instead of delete wherever possible,
			// due to the fact that wxSocket uses 'delayed events' (see the
			// documentation for wxPostEvent) and we don't want an event to
			// arrive to the event handler (the frame, here) after the socket
			// has been deleted. Also, we might be doing some other thing with
			// the socket at the same time; for example, we might be in the
			// middle of a test or something. Destroy() takes care of all
			// this for us.

			pSocket->Destroy();
			//wxMessageBox(wxT("Client lost"));
			break;
	default: 
		wxMessageBox(wxT("Unexpected event..."));

	}
}

}
