#if !defined(DATA_REPORTERS_H)
#define DATA_REPORTERS_H

#if defined(__MINGW32__)
#include <winsock2.h>	// Problem with: error Winsock.h has aready been included
#endif

#include "datareporting_class_factory.h"
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include "../config.h"
using boost::asio::ip::tcp;

namespace dae
{
namespace datareporting
{
/*********************************************************************
    daeBlackHoleDataReporter
*********************************************************************/
class DAE_DATAREPORTERS_API daeBlackHoleDataReporter : public daeDataReporter_t
{
public:
    daeBlackHoleDataReporter(void){}
    virtual ~daeBlackHoleDataReporter(void){}

public:
    virtual std::string GetName() const {return "BlackHoleDataReporter";}
    virtual void SetName(const std::string& strName) {}
    virtual std::string GetConnectString() const {return "";}
    virtual void SetConnectString(const std::string& strConnectString) {}
    virtual std::string GetProcessName() const {return "";}
    virtual void SetProcessName(const std::string& strProcessName) {}

    virtual bool Connect(const string& strConnectString, const string& strProcessName){return true;}
    virtual bool Disconnect(void){return true;}
    virtual bool IsConnected(void){return true;}
    virtual bool StartRegistration(void){return true;}
    virtual bool RegisterDomain(const daeDataReporterDomain* pDomain){return true;}
    virtual bool RegisterVariable(const daeDataReporterVariable* pVariable){return true;}
    virtual bool EndRegistration(void){return true;}
    virtual bool StartNewResultSet(real_t dTime){return true;}
    virtual bool EndOfData(void){return true;}
    virtual bool SendVariable(const daeDataReporterVariableValue* pVariableValue){return true;}
};

/*********************************************************************
    daeDataReporterLocal (ABSTRACT)
*********************************************************************/
class DAE_DATAREPORTERS_API daeDataReporterLocal : public daeDataReporter_t
{
public:
    daeDataReporterLocal(void);
    virtual ~daeDataReporterLocal(void);

public:
    virtual std::string GetName() const;
    virtual void SetName(const std::string& strName);
    virtual std::string GetConnectString() const;
    virtual void SetConnectString(const std::string& strConnectString);
    virtual std::string GetProcessName() const;
    virtual void SetProcessName(const std::string& strProcessName);

    virtual bool StartRegistration(void);
    virtual bool RegisterDomain(const daeDataReporterDomain* pDomain);
    virtual bool RegisterVariable(const daeDataReporterVariable* pVariable);
    virtual bool EndRegistration(void);
    virtual bool StartNewResultSet(real_t dTime);
    virtual bool EndOfData(void);
    virtual bool SendVariable(const daeDataReporterVariableValue* pVariableValue);

    daeDataReceiverProcess*	GetProcess(void);

public:
    std::string                 m_strName;
    string                      m_strConnectString;
    string                      m_strProcessName;
    real_t						m_dCurrentTime;
    daeDataReceiverProcess		m_drProcess;
};

/*********************************************************************
    daeFileDataReporter (ABSTRACT)
*********************************************************************/
class DAE_DATAREPORTERS_API daeFileDataReporter : public daeDataReporterLocal
{
public:
    daeFileDataReporter(void);
    virtual ~daeFileDataReporter(void);

public:
    virtual bool Connect(const string& strConnectString, const string& strProcessName);
    virtual bool Disconnect(void);
    virtual bool IsConnected(void);

    // By default does nothing; has to be overridden in derived classes
    virtual void WriteDataToFile(void);

public:
    std::ofstream	of;
    string			m_strFilename;
};

/*********************************************************************
    daeNoOpDataReporter
*********************************************************************/
class DAE_DATAREPORTERS_API daeNoOpDataReporter : public daeDataReporterLocal
{
public:
    daeNoOpDataReporter(void);
    virtual ~daeNoOpDataReporter(void);

public:
    virtual bool Connect(const string& strConnectString, const string& strProcessName);
    virtual bool Disconnect(void);
    virtual bool IsConnected(void);
};

/*********************************************************************
    daeTEXTFileDataReporter
*********************************************************************/
class DAE_DATAREPORTERS_API daeTEXTFileDataReporter : public daeFileDataReporter
{
public:
    daeTEXTFileDataReporter(void);
    virtual ~daeTEXTFileDataReporter(void);

public:
    virtual void WriteDataToFile(void);
};

/*********************************************************************
    daeHTMLFileDataReporter
*********************************************************************/
class DAE_DATAREPORTERS_API daeHTMLFileDataReporter : public daeFileDataReporter
{
public:
    daeHTMLFileDataReporter(void);
    virtual ~daeHTMLFileDataReporter(void);

public:
    virtual void WriteDataToFile(void);
};

/*********************************************************************
    daeDelegateDataReporter
*********************************************************************/
class DAE_DATAREPORTERS_API daeDelegateDataReporter : public daeDataReporter_t
{
public:
    daeDelegateDataReporter(void);
    virtual ~daeDelegateDataReporter(void);

public:
    virtual std::string GetName() const;
    virtual void SetName(const std::string& strName) {}
    virtual std::string GetConnectString() const {return "";}
    virtual void SetConnectString(const std::string& strConnectString) {}
    virtual std::string GetProcessName() const {return "";}
    virtual void SetProcessName(const std::string& strProcessName) {}

    virtual bool Connect(const string& strConnectString, const string& strProcessName);
    virtual bool Disconnect(void);
    virtual bool IsConnected(void);
    virtual bool StartRegistration(void);
    virtual bool RegisterDomain(const daeDataReporterDomain* pDomain);
    virtual bool RegisterVariable(const daeDataReporterVariable* pVariable);
    virtual bool EndRegistration(void);
    virtual bool StartNewResultSet(real_t dTime);
    virtual bool EndOfData(void);
    virtual bool SendVariable(const daeDataReporterVariableValue* pVariableValue);

public:
    void AddDataReporter(daeDataReporter_t* pDataReporter);

protected:
    std::vector<daeDataReporter_t*> m_ptrarrDataReporters;
};

/*********************************************************************
    daeHybridDataReporterReceiver
*********************************************************************/
//class DAE_DATAREPORTERS_API daeHybridDataReporterReceiver : public daeDataReporterLocal,
//														    public daeDataReceiver_t
//{
//public:
//	daeHybridDataReporterReceiver(void);
//	virtual ~daeHybridDataReporterReceiver(void);

//public:
//// daeDataReporter_t part
//	virtual bool Connect(const string& strConnectString, const string& strProcessName);
//	virtual bool Disconnect(void);
//	virtual bool IsConnected(void);

//// daeDataReceiver_t part
//	virtual bool	Start(void);
//	virtual bool	Stop(void);
//	virtual void	GetProcessName(string& strProcessName);
//	virtual void	GetDomains(std::vector<const daeDataReceiverDomain*>& ptrarrDomains) const;
//	virtual void	GetVariables(std::map<string, const daeDataReceiverVariable*>& ptrmappVariables) const;
//	virtual daeDataReceiverProcess*	GetProcess(void);
//};

/*********************************************************************
    daeMessageFormatter
*********************************************************************/
class daeMessageFormatter
{
public:
    daeMessageFormatter()
    {
    }

    ~daeMessageFormatter()
    {
    }

public:
// Message formatting routines
    string SendProcessName(const string& strProcessName)
    {
        boost::int32_t nameSize;
        std::stringstream s(std::ios_base::out|std::ios_base::in|std::ios_base::binary);

    // Send cStartRegistration flag
        s.write((const char*)(void*)&cSendProcessName, sizeof(char));

    // Send the size of the ProcessName, and the ProcessName itself
        nameSize = strProcessName.size();
        s.write((const char*)(void*)&nameSize, sizeof(nameSize));
        s.write(strProcessName.c_str(), nameSize);

    // Flush the buffer
        s.flush();

        return s.str();
    }

    string StartRegistration(void)
    {
        std::stringstream s(std::ios_base::out|std::ios_base::in|std::ios_base::binary);

    // Send cStartRegistration flag
        s.write((const char*)(void*)&cStartRegistration, sizeof(char));

    // Send the dummy byte
        char nothing = 0;
        s.write((const char*)(void*)&nothing, sizeof(nothing));

    // Flush the buffer
        s.flush();

        return s.str();
    }

    string EndRegistration(void)
    {
        std::stringstream s(std::ios_base::out|std::ios_base::in|std::ios_base::binary);

    // Send cEndRegistration flag
        s.write((const char*)(void*)&cEndRegistration, sizeof(char));

    // Send the dummy byte
        char nothing = 0;
        s.write((const char*)(void*)&nothing, sizeof(nothing));

    // Flush the buffer
        s.flush();

        return s.str();
    }

    string RegisterDomain(const daeDataReporterDomain* pDomain)
    {
        boost::int32_t nameSize, unitsSize, noPoints, type;
        std::stringstream s(std::ios_base::out|std::ios_base::in|std::ios_base::binary);

    // Send RegisterDomain flag
        s.write((const char*)(void*)&cRegisterDomain, sizeof(char));

    // Send the size of the Name, and the Name itself
        nameSize = pDomain->m_strName.size();
        s.write((const char*)(void*)&nameSize, sizeof(nameSize));
        s.write(pDomain->m_strName.c_str(), nameSize);

    // Send the size of the Units, and the Units itself
        unitsSize = pDomain->m_strUnits.size();
        s.write((const char*)(void*)&unitsSize, sizeof(unitsSize));
        s.write(pDomain->m_strUnits.c_str(), unitsSize);

    // Send the domain type
        type = pDomain->m_eType;
        s.write((const char*)(void*)&type, sizeof(type));

    // Send the number of points
        noPoints = pDomain->m_nNumberOfPoints;
        s.write((const char*)(void*)&noPoints, sizeof(noPoints));

    // Send the points
        if(pDomain->m_eType == eUnstructuredGrid)
        {
            for(size_t i= 0; i < pDomain->m_nNumberOfPoints; i++)
            {
                s.write((const char*)(void*)(&pDomain->m_arrCoordinates[i].x), sizeof(double));
                s.write((const char*)(void*)(&pDomain->m_arrCoordinates[i].y), sizeof(double));
                s.write((const char*)(void*)(&pDomain->m_arrCoordinates[i].z), sizeof(double));
            }
        }
        else
        {
            double point;
            for(size_t i= 0; i < pDomain->m_nNumberOfPoints; i++)
            {
                point = pDomain->m_arrPoints[i];
                s.write((const char*)(void*)(&point), sizeof(double));
            }
        }

    // Flush the buffer
        s.flush();

        return s.str();
    }

    string RegisterVariable(const daeDataReporterVariable* pVariable)
    {
        boost::int32_t nameSize, unitsSize, domainsSize, noPoints;
        std::stringstream s(std::ios_base::out|std::ios_base::in|std::ios_base::binary);

    // Send RegisterVariable flag
        s.write((const char*)(void*)&cRegisterVariable, sizeof(char));

    // Send the size of the Name, and the Name itself
        nameSize = pVariable->m_strName.size();
        s.write((const char*)(void*)&nameSize, sizeof(nameSize));
        s.write(pVariable->m_strName.c_str(), nameSize);

    // Send the size of the Units, and the Units itself
        unitsSize = pVariable->m_strUnits.size();
        s.write((const char*)(void*)&unitsSize, sizeof(unitsSize));
        s.write(pVariable->m_strUnits.c_str(), unitsSize);

    // Send the number of points
        noPoints = pVariable->m_nNumberOfPoints;
        s.write((const char*)(void*)&noPoints, sizeof(noPoints));

    // Send the size of the array with domain names, and array itself
        domainsSize = pVariable->m_strarrDomains.size();
        s.write((const char*)(void*)&domainsSize, sizeof(domainsSize));
        for(size_t i = 0; i < (size_t)domainsSize; i++)
        {
            nameSize = pVariable->m_strarrDomains[i].size();
            s.write((const char*)(void*)&nameSize,         sizeof(nameSize));
            s.write(pVariable->m_strarrDomains[i].c_str(), nameSize);
        }

    // Flush the buffer
        s.flush();

        return s.str();
    }

    string StartNewResultSet(real_t dTime)
    {
        std::stringstream s(std::ios_base::out|std::ios_base::in|std::ios_base::binary);

    // Send cStartNewTime flag
        s.write((const char*)(void*)&cStartNewTime, sizeof(char));

    // Send the Time
        s.write((const char*)(void*)&dTime, sizeof(dTime));

    // Flush the buffer
        s.flush();

        return s.str();
    }

    string EndOfData()
    {
        std::stringstream s(std::ios_base::out|std::ios_base::in|std::ios_base::binary);

    // Send cEndOfData flag
        s.write((const char*)(void*)&cEndOfData, sizeof(char));

    // Send the dummy byte
        char nothing = 0;
        s.write((const char*)(void*)&nothing, sizeof(nothing));

    // Flush the buffer
        s.flush();

        return s.str();
    }

    string SendVariable(const daeDataReporterVariableValue* pVariableValue)
    {
        boost::int32_t nameSize, noPoints;
        std::stringstream s(std::ios_base::out|std::ios_base::in|std::ios_base::binary);

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

        return s.str();
    }

// Message parsing routines (NOT USED ANYWHERE!!!??? THEREFORE, COMMENTED OUT)
/*
    char GetMessageType(const string& strMessage)
    {
        return strMessage[0];
    }

    void StartRegistration(const string& strMessage, daeDataReceiverProcess& drProcess)
    {
        boost::int32_t curPos, nameSize;
        const char* data = strMessage.c_str();

        char cFlag = data[0];
        curPos = 1;

        if(cFlag == cStartRegistration)
        {
            char* szName;

        // Read size of the name and move the pointer
            memcpy(&nameSize, &data[curPos], sizeof(nameSize));
            curPos += sizeof(nameSize);

        // Read the name and move the pointer
            szName = new char[nameSize+1];
            szName[nameSize] = '\0';
            memcpy(szName, &data[curPos], nameSize);
            drProcess.m_strName.assign(szName);
            delete[] szName;
        }
    }

    void EndRegistration(const string& strMessage, daeDataReceiverProcess& drProcess)
    {
        boost::int32_t msgSize, curPos;
        const char* data = strMessage.c_str();
        msgSize = strMessage.size();

        char cFlag = data[0];
        curPos = 1;

    }

    void RegisterDomain(const string& strMessage, daeDataReceiverProcess& drProcess)
    {
        boost::int32_t curPos, nameSize, unitsSize;
        const char* data = strMessage.c_str();

        char cFlag = data[0];
        curPos = 1;

        boost::int32_t noPoints, type;

        daeDataReceiverDomain* pDomain = new daeDataReceiverDomain;
        drProcess.m_ptrarrRegisteredDomains.push_back(pDomain);

    // Read size of the Name and move the pointer
        memcpy(&nameSize, &data[curPos], sizeof(nameSize));
        curPos += sizeof(nameSize);

    // Read the Name and move the pointer
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

    void RegisterVariable(const string& strMessage, daeDataReceiverProcess& drProcess)
    {
        boost::int32_t i, msgSize, curPos, nameSize, unitsSize;
        const char* data = strMessage.c_str();
        msgSize = strMessage.size();

        char cFlag = data[0];
        curPos = 1;

        char* szName;
        size_t j, k;
        string strDomainName;
        boost::int32_t noPoints;
        boost::int32_t domainsSize;
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
            for(j = 0; j < drProcess.m_ptrarrRegisteredDomains.size(); j++)
            {
                pDomain = drProcess.m_ptrarrRegisteredDomains[j];
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

        if(noPoints != (boost::int32_t)pVariable->m_nNumberOfPoints)
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
        drProcess.m_ptrmapRegisteredVariables.insert(p);
    }

    void StartNewTime(const string& strMessage, real_t& dCurrentTime)
    {
        const char* data = strMessage.c_str();
        char cFlag = data[0];

        dCurrentTime = *((real_t*)&data[1]);
    }

    void SendVariable(const string& strMessage, daeDataReceiverProcess& drProcess, real_t dCurrentTime)
    {
        boost::int32_t msgSize, curPos, nameSize;
        const char* data = strMessage.c_str();
        msgSize = strMessage.size();

        char cFlag = data[0];
        curPos = 1;

        char* szName;
        string strVariableName;
        boost::int32_t noPoints;

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
        std::map<string, daeDataReceiverVariable*>::iterator iter = drProcess.m_ptrmapRegisteredVariables.find(strVariableName);
        if(iter == drProcess.m_ptrmapRegisteredVariables.end())
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
        pValue->m_dTime = dCurrentTime;

    // Read the number of points
        memcpy(&noPoints, &data[curPos], sizeof(noPoints));
        curPos += sizeof(noPoints);

        if(noPoints != (boost::int32_t)pVariable->m_nNumberOfPoints)
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

    // Finally add the value to the array
        pVariable->m_ptrarrValues.push_back(pValue);
    }

    void EndOfData(const string& strMessage, daeDataReceiverProcess& drProcess)
    {
        boost::int32_t i, msgSize, curPos, nameSize;
        const char* data = strMessage.c_str();
        msgSize = strMessage.size();

        char cFlag = data[0];
        curPos = 1;

    }
*/
};

/*********************************************************************
    daeDataReporterRemote (ABSTRACT)
*********************************************************************/
class DAE_DATAREPORTERS_API daeDataReporterRemote : public daeDataReporter_t
{
public:
    daeDataReporterRemote(void);
    virtual ~daeDataReporterRemote(void);

public:
    virtual std::string GetName() const;
    virtual void SetName(const std::string& strName);
    virtual std::string GetConnectString() const;
    virtual void SetConnectString(const std::string& strConnectString);
    virtual std::string GetProcessName() const;
    virtual void SetProcessName(const std::string& strProcessName);

    bool Connect(const string& strConnectString, const string& strProcessName);
    bool Disconnect(void);
    bool IsConnected(void);
    bool StartRegistration(void);
    bool RegisterDomain(const daeDataReporterDomain* pDomain);
    bool RegisterVariable(const daeDataReporterVariable* pVariable);
    bool EndRegistration(void);
    bool StartNewResultSet(real_t dTime);
    bool EndOfData(void);
    bool SendVariable(const daeDataReporterVariableValue* pVariableValue);

    bool SendProcessName(const string& strProcessName);

// By default does nothing; has to be overridden in derived classes
    virtual bool SendMessage(const string& strMessage);

protected:
    std::string              m_strName;
    string					 m_strConnectString;
    string					 m_strProcessName;
    real_t                   m_dCurrentTime;
    daeMessageFormatter		 m_msgFormatter;
};

/*********************************************************************
    daeTCPIPDataReporter
*********************************************************************/
class DAE_DATAREPORTERS_API daeTCPIPDataReporter : public daeDataReporterRemote
{
public:
    daeTCPIPDataReporter(void);
    virtual ~daeTCPIPDataReporter(void);

public:
    bool Connect(const string& strConnectString, const string& strProcessName);
    bool Disconnect(void);
    bool IsConnected(void);
    bool SendMessage(const string& strMessage);

//	virtual bool StartRegistration(void);
//	virtual bool RegisterDomain(const daeDataReporterDomain* pDomain);
//	virtual bool RegisterVariable(const daeDataReporterVariable* pVariable);
//	virtual bool EndRegistration(void);
//	virtual bool StartNewResultSet(real_t dTime);
//	virtual bool EndOfData(void);
//	virtual bool SendVariable(const daeDataReporterVariableValue* pVariableValue);

protected:
//	string					 m_strConnectionString;
//	string					 m_strProcessName;
//	real_t                   m_dCurrentTime;
    boost::shared_ptr<tcp::socket> m_ptcpipSocket;
    boost::asio::io_service        m_ioService;
};

/*********************************************************************
    daeTCPIPDataReceiver
*********************************************************************/
class DAE_DATAREPORTERS_API daeTCPIPDataReceiver : public daeDataReceiver_t
{
public:
    daeTCPIPDataReceiver(void);
    daeTCPIPDataReceiver(boost::shared_ptr<tcp::socket> ptcpipSocket);
    virtual ~daeTCPIPDataReceiver(void);

public:
    virtual bool	Start(void);
    virtual bool	Stop(void);
    virtual void	GetProcessName(string& strProcessName);
    virtual void	GetDomains(std::vector<const daeDataReceiverDomain*>& ptrarrDomains) const;
    virtual void	GetVariables(std::map<string, const daeDataReceiverVariable*>& ptrmappVariables) const;
    virtual daeDataReceiverProcess*	GetProcess(void);

    void thread(void);
    void ParseMessage(unsigned char* data, boost::int32_t msgSize);

protected:
    boost::shared_ptr<tcp::socket>	 m_tcpipSocket;
    real_t							 m_dCurrentTime;
    daeDataReceiverProcess			 m_drProcess;
    boost::shared_ptr<boost::thread> m_pThread;
};

/*********************************************************************
    daeTCPIPDataReceiverServer
*********************************************************************/
class DAE_DATAREPORTERS_API daeTCPIPDataReceiverServer
{
public:
    daeTCPIPDataReceiverServer(int nPort);
    virtual ~daeTCPIPDataReceiverServer(void);

public:
    void Start(void);
    void StartAccept(void);
    void HandleAccept(const boost::system::error_code& error);
    void Stop(void);
    void StartThread(void);
    bool IsConnected(void);

public:
    int									m_nPort;
    boost::asio::io_service				m_ioService;
    tcp::acceptor						m_acceptor;
    boost::shared_ptr<tcp::socket>		m_tcpipSocket;
    boost::shared_ptr<boost::thread>	m_pThread;
    daePtrVector<daeTCPIPDataReceiver*> m_ptrarrDataReceivers;
};

}
}
#endif
