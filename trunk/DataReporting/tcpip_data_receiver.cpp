#include "stdafx.h"
#include "datareporters.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <boost/cstdint.hpp>

namespace daetools
{
namespace datareporting
{
/*********************************************************************
    daeTCPIPDataReceiverServer
*********************************************************************/
daeTCPIPDataReceiverServer::daeTCPIPDataReceiverServer(int nPort) : m_nPort(nPort),
                                                                    m_ioService(),
                                                                    m_acceptor(m_ioService, tcp::endpoint(tcp::v4(), nPort))
{
}

daeTCPIPDataReceiverServer::~daeTCPIPDataReceiverServer()
{
    Stop();
}

void daeTCPIPDataReceiverServer::Stop(void)
{
    boost::system::error_code ec;
    if(m_acceptor.is_open())
    {
        m_acceptor.cancel(ec);
        m_acceptor.close();
    }
}

void daeTCPIPDataReceiverServer::Start(void)
{
    if(m_pThread)
    {
        daeDeclareException(exInvalidCall);
        e << "TCPIP DataReceiver server has already been started";
        throw e;
    }

    m_pThread = std::shared_ptr<boost::thread>(new boost::thread(boost::bind(&daeTCPIPDataReceiverServer::StartThread, this)));
}

void daeTCPIPDataReceiverServer::StartThread(void)
{
    StartAccept();
    m_ioService.run();
}

bool daeTCPIPDataReceiverServer::IsConnected(void)
{
    if(m_acceptor.is_open())
        return true;
    else
        return false;
}

void daeTCPIPDataReceiverServer::StartAccept(void)
{
    m_tcpipSocket = std::shared_ptr<tcp::socket>(new tcp::socket(m_ioService));
    m_acceptor.async_accept(*m_tcpipSocket, bind(&daeTCPIPDataReceiverServer::HandleAccept, this, boost::asio::placeholders::error));
}

void daeTCPIPDataReceiverServer::HandleAccept(const boost::system::error_code& error)
{
    // This function is called when a client is connected and when ioService is cancelled!!

    if(!error)
    {
        daeTCPIPDataReceiver* pDataReceiver = new daeTCPIPDataReceiver(m_tcpipSocket);
        m_ptrarrDataReceivers.push_back(pDataReceiver);
        m_tcpipSocket.reset();

        pDataReceiver->Start();

        StartAccept();
    }
}

/*********************************************************************
    daeTCPIPDataReceiver
*********************************************************************/
daeTCPIPDataReceiver::daeTCPIPDataReceiver(void)
{
}

daeTCPIPDataReceiver::daeTCPIPDataReceiver(std::shared_ptr<tcp::socket> ptcpipSocket)
{
    m_tcpipSocket = ptcpipSocket;
}

daeTCPIPDataReceiver::~daeTCPIPDataReceiver()
{
    Stop();
}

void daeTCPIPDataReceiver::GetProcessName(string& strProcessName)
{
    strProcessName = m_drProcess.m_strName;
}

bool daeTCPIPDataReceiver::Start(void)
{
    if(m_pThread)
    {
        daeDeclareException(exInvalidCall);
        e << "TCPIP DataReceiver server has already been started";
        throw e;
    }

    m_pThread.reset(new boost::thread(boost::bind(&daeTCPIPDataReceiver::thread, this)));
    return true;
}

bool daeTCPIPDataReceiver::Stop(void)
{
    m_pThread.reset();
    return true;
}

void daeTCPIPDataReceiver::ParseMessage(unsigned char* data, std::int32_t msgSize)
{
    std::int32_t i, curPos, nameSize, unitsSize;
    char cFlag = data[0];
    curPos = 1;

    try
    {
        if(cFlag == cSendProcessName)
        {
            char* szName;

        // Read size of the name and move the pointer
            memcpy(&nameSize, &data[curPos], sizeof(nameSize));
            curPos += sizeof(nameSize);

        // Read the name and move the pointer
            szName = new char[nameSize+1];
            szName[nameSize] = '\0';
            memcpy(szName, &data[curPos], nameSize);
            m_drProcess.m_strName.assign(szName);
            delete[] szName;
        }
        else if(cFlag == cStartRegistration)
        {
        }
        else if(cFlag == cEndRegistration)
        {
        }
        else if(cFlag == cRegisterDomain)
        {
            std::int32_t noPoints, type;

            daeDataReceiverDomain* pDomain = new daeDataReceiverDomain;
            m_drProcess.m_ptrarrRegisteredDomains.push_back(pDomain);

        // Read size of the name and move the pointer
            memcpy(&nameSize, &data[curPos], sizeof(nameSize));
            curPos += sizeof(nameSize);

        // Read the name and move the pointer
            char* szName = new char[nameSize+1];
            szName[nameSize] = '\0';
            memcpy(szName, &data[curPos], nameSize);
            pDomain->m_strName.assign(szName);
            delete[] szName;
            curPos += nameSize;

        // Read size of the Units and move the pointer
            memcpy(&unitsSize, &data[curPos], sizeof(unitsSize));
            curPos += sizeof(unitsSize);

        // Read the Units and move the pointer
            char* szUnits = new char[unitsSize+1];
            szUnits[unitsSize] = '\0';
            memcpy(szUnits, &data[curPos], unitsSize);
            pDomain->m_strUnits.assign(szUnits);
            delete[] szUnits;
            curPos += unitsSize;

        // Read the domain type and move the pointer
            memcpy(&type, &data[curPos], sizeof(type));
            pDomain->m_eType = (daeeDomainType)type;
            curPos += sizeof(type);

        // Read the number of points and move the pointer
            memcpy(&noPoints, &data[curPos], sizeof(noPoints));
            pDomain->m_nNumberOfPoints = (size_t)noPoints;
            curPos += sizeof(noPoints);
            if(pDomain->m_nNumberOfPoints <= 0)
                return;

        // Read the points
            if(pDomain->m_eType == eUnstructuredGrid)
            {
                //std::cout << pDomain->m_strName << " points: " << std::endl;

                daePoint point;
                pDomain->m_arrCoordinates.resize(pDomain->m_nNumberOfPoints);
                for(size_t i = 0; i < pDomain->m_nNumberOfPoints; i++)
                {
                    memcpy(&point.x, &data[curPos], sizeof(double));
                    curPos += sizeof(double);

                    memcpy(&point.y, &data[curPos], sizeof(double));
                    curPos += sizeof(double);

                    memcpy(&point.z, &data[curPos], sizeof(double));
                    curPos += sizeof(double);

                    pDomain->m_arrCoordinates[i] = point;

                    //std::cout << pDomain->m_arrCoordinates[i].x << ", " <<  pDomain->m_arrCoordinates[i].y << std::endl;
                }
            }
            else
            {
                pDomain->m_arrPoints.resize(pDomain->m_nNumberOfPoints);
                double* fdata = (double*)(void*)(&data[curPos]);
                pDomain->m_arrPoints.assign(fdata, fdata + pDomain->m_nNumberOfPoints);
                curPos += pDomain->m_nNumberOfPoints * sizeof(double);
            }
        }
        else if(cFlag == cRegisterParameter)
        {
        }
        else if(cFlag == cRegisterVariable)
        {
            char* szName;
            size_t j, k;
            string strDomainName;
            std::int32_t noPoints;
            std::int32_t domainsSize;
            daeDataReceiverDomain* pDomain;
            daeDataReceiverVariable* pVariable = new daeDataReceiverVariable;

        // Read size of the name and move the pointer
            memcpy(&nameSize, &data[curPos], sizeof(nameSize));
            curPos += sizeof(nameSize);

        // Read the name and move the pointer
            szName = new char[nameSize+1];
            szName[nameSize] = '\0';
            memcpy(szName, &data[curPos], nameSize);
            pVariable->m_strName.assign(szName);
            delete[] szName;
            curPos += nameSize;

        // Read size of the Units and move the pointer
            memcpy(&unitsSize, &data[curPos], sizeof(unitsSize));
            curPos += sizeof(unitsSize);

        // Read the Units and move the pointer
            char* szUnits = new char[unitsSize+1];
            szUnits[unitsSize] = '\0';
            memcpy(szUnits, &data[curPos], unitsSize);
            pVariable->m_strUnits.assign(szUnits);
            delete[] szUnits;
            curPos += unitsSize;

        // Read the number of points and move the pointer
            memcpy(&noPoints, &data[curPos], sizeof(noPoints));
            pVariable->m_nNumberOfPoints = (size_t)noPoints;
            curPos += sizeof(noPoints);
            if(pVariable->m_nNumberOfPoints <= 0)
                return;

        // Read the domains
            memcpy(&domainsSize, &data[curPos], sizeof(domainsSize));
            curPos += sizeof(domainsSize);

            for(i = 0; i < domainsSize; i++)
            {
                memcpy(&nameSize, &data[curPos], sizeof(nameSize));
                curPos += sizeof(nameSize);

                szName = new char[nameSize+1];
                szName[nameSize] = '\0';
                memcpy(szName, &data[curPos], nameSize);
                strDomainName.assign(szName);
                delete[] szName;
                curPos += nameSize;

                bool bFound = false;
                for(j = 0; j < m_drProcess.m_ptrarrRegisteredDomains.size(); j++)
                {
                    pDomain = m_drProcess.m_ptrarrRegisteredDomains[j];
                    if(pDomain->m_strName == strDomainName)
                    {
                        pVariable->m_ptrarrDomains.push_back(pDomain);
                        bFound = true;
                        break;
                    }
                }
                if(!bFound)
                {
                    delete pVariable;
                    daeDeclareException(exRuntimeCheck);
                    e << "Cannot register variable: [" << pVariable->m_strName
                      << "]; cannot find the following domain: [" << strDomainName << "]";
                    throw e;
                    return;
                }
            }

        // Check if number of points match
            noPoints = 1;
            for(k = 0; k < pVariable->m_ptrarrDomains.size(); k++)
                noPoints *= pVariable->m_ptrarrDomains[k]->m_nNumberOfPoints;

            if(noPoints != (std::int32_t)pVariable->m_nNumberOfPoints)
            {
                delete pVariable;
                daeDeclareException(exRuntimeCheck);
                e << "Number of points in variable: [" << pVariable->m_strName << "] does not match the expected one "
                  << "(" << noPoints << " vs. " << pVariable->m_nNumberOfPoints << ")";
                throw e;
                return;
            }

        // Finally add the variable to the map
            std::pair<string, daeDataReceiverVariable*> p(pVariable->m_strName, pVariable);
            m_drProcess.m_ptrmapRegisteredVariables.insert(p);
        }
        else if(cFlag == cStartNewTime)
        {
            m_dCurrentTime = *((real_t*)&data[1]);
        }
        else if(cFlag == cSendVariable)
        {
            char* szName;
            string strVariableName;
            std::int32_t noPoints;

        // Read size of the name and move the pointer
            memcpy(&nameSize, &data[curPos], sizeof(nameSize));
            curPos += sizeof(nameSize);

        // Read the name and move the pointer
            szName = new char[nameSize+1];
            szName[nameSize] = '\0';
            memcpy(szName, &data[curPos], nameSize);
            strVariableName.assign(szName);
            delete[] szName;
            curPos += nameSize;

        // Find the variable by its name
            std::map<string, daeDataReceiverVariable*>::iterator iter = m_drProcess.m_ptrmapRegisteredVariables.find(strVariableName);
            if(iter == m_drProcess.m_ptrmapRegisteredVariables.end())
            {
                daeDeclareException(exRuntimeCheck);
                e << "Cannot find variable: [" << strVariableName << "] among the registered ones";
                throw e;
                return;
            }
            daeDataReceiverVariable* pVariable = (*iter).second;
            if(!pVariable)
                return;

        // Create value
            daeDataReceiverVariableValue* pValue = new daeDataReceiverVariableValue;

        // Set the time
            pValue->m_dTime = m_dCurrentTime;

        // Read the number of points
            memcpy(&noPoints, &data[curPos], sizeof(noPoints));
            curPos += sizeof(noPoints);

            if(noPoints != (std::int32_t)pVariable->m_nNumberOfPoints)
            {
                delete pValue;
                daeDeclareException(exRuntimeCheck);
                e << "Number of points in variable: [" << pVariable->m_strName << "] does not match the expected one "
                  << "(" << noPoints << " vs. " << pVariable->m_nNumberOfPoints << ")";
                throw e;
                return;
            }

        // Read the points
            pValue->m_pValues = new real_t[noPoints];
            memcpy(pValue->m_pValues, &data[curPos], sizeof(real_t)*noPoints);

        // Finally add the value to the array (do not use dae_push_back)
            pVariable->m_ptrarrValues.push_back(pValue);
        }
        else if(cFlag == cEndOfData)
        {
        }
        else
        {
            daeDeclareException(exRuntimeCheck);
            e << "Error parsing the message; Unexpected flag: [" << cFlag << "] found";
            throw e;
        }
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    catch(...)
    {
        std::cout << "Unexpected error in daeTCPIPDataReceiver" << std::endl;
    }
}

void daeTCPIPDataReceiver::thread(void)
{
    try
    {
        boost::system::error_code ec;
        std::int32_t msgSize = 0,
                       nRead   = 0;

        for(;;)
        {
            if(!m_tcpipSocket || !m_tcpipSocket->is_open())
            {
                cout << "Socket disconnected!" << endl;
                return;
            }

            nRead = 0;
            nRead = boost::asio::read(*m_tcpipSocket,
                                      boost::asio::buffer(&msgSize, sizeof(msgSize)),
                                      boost::asio::transfer_at_least(sizeof(msgSize)),
                                      ec);
            if(ec)
            {
//				if(ec.value() == boost::system::errc::broken_pipe ||
//				   ec.value() == boost::system::errc::address_in_use ||
//				   ec.value() == boost::system::errc::bad_address ||
//				   ec.value() == boost::system::errc::address_family_not_supported ||
//				   ec.value() == boost::system::errc::address_not_available ||
//				   ec.value() == boost::system::errc::already_connected ||
//				   ec.value() == boost::system::errc::connection_refused ||
//				   ec.value() == boost::system::errc::connection_aborted ||
//				   ec.value() == boost::system::errc::network_down ||
//				   ec.value() == boost::system::errc::network_unreachable ||
//				   ec.value() == boost::system::errc::host_unreachable ||
//				   ec.value() == boost::system::errc::broken_pipe ) // Do nothing
//					return;
//				else
//					cout << "Error in daeTCPIPDataReceiver: " << ec.message() << endl;
                return;
            }
            if(nRead == 0)
            {
                cout << "Client disconnected, closing the receiving socket..." << endl;
                return;
            }
            if(nRead != sizeof(msgSize))
            {
                cout << "Cannot read the message length (4 bytes long); Read = " << nRead << "bytes; "
                     << "Error: " << ec.message() << endl;
                return;
            }

            //cout << "Message size is " << msgSize << endl;
            if(msgSize <= 0)
            {
                cout << "Message size is 0!!" << endl;
                return;
            }

            unsigned char* data = new unsigned char[msgSize];
            nRead = 0;
            nRead = boost::asio::read(*m_tcpipSocket,
                                      boost::asio::buffer(data, msgSize),
                                      boost::asio::transfer_at_least(msgSize),
                                      ec);
            if(ec)
            {
                //cout << "Error while reading the message in daeTCPIPDataReceiver: " << ec.message() << endl;
                delete[] data;
                return;
            }
            if(nRead == 0)
            {
                cout << "Reporting cocket disconnected, closing the receiving socket..." << endl;
                delete[] data;
                return;
            }
            if(nRead != msgSize)
            {
                delete[] data;
                cout << "Message not read completely; " << nRead << " bytes has been read; "
                     << "Error: " << ec.message() << endl;
                return;
            }

            //cout << "Successfully read " << msgSize << " bytes" << endl;
            //cout << "Message is: ";

            ParseMessage(data, msgSize);

            delete[] data;
            data = NULL;
        }
    }
    catch (std::exception& e)
    {
        cout << "Exception in thread: " << e.what() << "\n";
    }
}

daeDataReceiverProcess*	daeTCPIPDataReceiver::GetProcess(void)
{
    return &m_drProcess;
}

void daeTCPIPDataReceiver::GetDomains(vector<const daeDataReceiverDomain*>& ptrarrDomains) const
{
    for(size_t i = 0; i < m_drProcess.m_ptrarrRegisteredDomains.size(); i++)
        ptrarrDomains.push_back(m_drProcess.m_ptrarrRegisteredDomains[i]);
}

void daeTCPIPDataReceiver::GetVariables(map<string, const daeDataReceiverVariable*>& ptrmappVariables) const
{
    map<string, daeDataReceiverVariable*>::const_iterator iter;
    for(iter = m_drProcess.m_ptrmapRegisteredVariables.begin(); iter != m_drProcess.m_ptrmapRegisteredVariables.end(); iter++)
    {
        std::pair<string, const daeDataReceiverVariable*> p((*iter).first, (*iter).second);
        ptrmappVariables.insert(p);
    }
}


}
}
