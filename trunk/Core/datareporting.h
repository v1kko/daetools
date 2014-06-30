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
#ifndef DAE_DATA_REPORTING_H
#define DAE_DATA_REPORTING_H

#include "definitions.h"
#include "core.h"
using namespace dae::core;

namespace dae
{
namespace datareporting
{
/*********************************************************************
	daeDataReporterDomain
*********************************************************************/
const char cSendProcessName   = 0;
const char cStartRegistration = 1;
const char cEndRegistration   = 2;
const char cRegisterDomain    = 3;
const char cRegisterParameter = 4;
const char cRegisterVariable  = 5;
const char cStartNewTime      = 6;
const char cSendVariable      = 7;
const char cEndOfData         = 8;

class daeDataReporterDomain
{
public:
	daeDataReporterDomain(void)
	{
		m_eType           = eDTUnknown;
		m_nNumberOfPoints = 0;
    }
	
	daeDataReporterDomain(const string& strName, daeeDomainType eType, size_t nNoPoints)
	{
		m_strName         = strName;
		m_eType           = eType;
		m_nNumberOfPoints = nNoPoints;
        if(m_eType == eUnstructuredGrid)
            m_arrCoordinates.resize(nNoPoints);
        else
            m_arrPoints.resize(nNoPoints);
    }
	
	daeDataReporterDomain(const daeDataReporterDomain& drd) // Deep copy
	{
		m_strName         = drd.m_strName;
		m_eType           = drd.m_eType;
		m_nNumberOfPoints = drd.m_nNumberOfPoints;
        if(m_eType == eUnstructuredGrid)
            m_arrCoordinates = drd.m_arrCoordinates;
        else
            m_arrPoints = drd.m_arrPoints;
	}
	
	~daeDataReporterDomain(void)
	{
	}

	std::iostream& operator << (std::iostream& io) const
    {
		io << m_strName << (int)m_eType << m_nNumberOfPoints;
        if(m_eType == eUnstructuredGrid)
        {
            for(size_t i = 0; i < m_nNumberOfPoints; i++)
                io << m_arrCoordinates[i];
        }
        else
        {
            for(size_t i = 0; i < m_nNumberOfPoints; i++)
                io << m_arrPoints[i];
        }
        return io;
	}
	
	std::iostream& operator >> (std::iostream& io)
	{
		int e;
		io >> m_strName;
		io >> e;
		m_eType = (daeeDomainType)e;
		io >> m_nNumberOfPoints;
        if(m_eType == eUnstructuredGrid)
        {
            m_arrCoordinates.resize(m_nNumberOfPoints);
            for(size_t i = 0; i < m_nNumberOfPoints; i++)
            {
                io >> m_arrCoordinates[i].x;
                io >> m_arrCoordinates[i].y;
                io >> m_arrCoordinates[i].x;
            }
        }
        else
        {
            m_arrPoints.resize(m_nNumberOfPoints);
            for(size_t i = 0; i < m_nNumberOfPoints; i++)
                io >> m_arrPoints[i];
        }
		return io;
	}

public:
    string                  m_strName;
    daeeDomainType          m_eType;
    size_t                  m_nNumberOfPoints;
    std::vector<real_t>     m_arrPoints;
    std::vector<daePoint>   m_arrCoordinates;
};

/*********************************************************************
	daeDataReporterVariable
*********************************************************************/
class daeDataReporterVariable
{
public:
	daeDataReporterVariable(void)
	{
		m_nNumberOfPoints = 0;
	}
	
	daeDataReporterVariable(const string& strName, size_t nNoPoints)
	{
		m_strName         = strName;
		m_nNumberOfPoints = nNoPoints;
	}
	
	daeDataReporterVariable(const daeDataReporterVariable& drv) // Deep copy
	{
		m_strName         = drv.m_strName;
		m_nNumberOfPoints = drv.m_nNumberOfPoints;
		for(size_t i = 0; i < drv.m_strarrDomains.size(); i++)
			m_strarrDomains.push_back(drv.m_strarrDomains[i]);
	}

	~daeDataReporterVariable(void)
	{
	}

	void AddDomain(const string& strDomain)
	{
		m_strarrDomains.push_back(strDomain);
	}

	string GetDomain(size_t n) const
	{
		if(n >= m_strarrDomains.size())
			daeDeclareAndThrowException(exInvalidCall); 

		return m_strarrDomains[n];
	}

	size_t GetNumberOfDomains(void) const
	{
		return m_strarrDomains.size();
	}

	std::iostream& operator << (std::iostream& io) const
	{
		io << m_strName << m_nNumberOfPoints;
		io << m_strarrDomains.size();
		for(size_t i = 0; i < m_strarrDomains.size(); i++)
			io << m_strarrDomains[i];
		return io;
	}
	
	std::iostream& operator >> (std::iostream& io)
	{
		size_t n;
		string strDomain;
		io >> m_strName;
		io >> m_nNumberOfPoints;
		io >> n;
		if(n > 0)
		{
			m_strarrDomains.resize(n);
			for(size_t i = 0; i < n; i++)
			{
				io >> strDomain;
				m_strarrDomains[i] = strDomain;
			}
		}
		return io;
	}

public:
	string				m_strName;
	size_t				m_nNumberOfPoints;
	std::vector<string>	m_strarrDomains;
};

/*********************************************************************
	daeDataReporterVariableValue
*********************************************************************/
class daeDataReporterVariableValue
{
public:
	daeDataReporterVariableValue(void)
	{
		m_nNumberOfPoints = 0;
		m_pValues         = NULL;
	}
	
	daeDataReporterVariableValue(const string& strName, size_t nNoPoints)
	{
		m_strName         = strName;
		m_nNumberOfPoints = nNoPoints;
		m_pValues = new real_t[nNoPoints];
	}
	
	daeDataReporterVariableValue(const daeDataReporterVariableValue& drvv) // Deep copy
	{
		m_strName         = drvv.m_strName;
		m_nNumberOfPoints = drvv.m_nNumberOfPoints;
		m_pValues = new real_t[m_nNumberOfPoints];
		for(size_t i = 0; i < m_nNumberOfPoints; i++)
			m_pValues[i] = drvv.m_pValues[i];
	}

	~daeDataReporterVariableValue(void)
	{
		if(m_pValues)
			delete[] m_pValues;
	}

	real_t GetValue(size_t n) const
	{
		if(!m_pValues)
			daeDeclareAndThrowException(exInvalidPointer); 
		if(n >= m_nNumberOfPoints)
			daeDeclareAndThrowException(exInvalidCall); 

		return m_pValues[n];
	}

	void SetValue(size_t n, real_t value)
	{
		if(!m_pValues)
			daeDeclareAndThrowException(exInvalidPointer); 
		if(n >= m_nNumberOfPoints)
			daeDeclareAndThrowException(exInvalidCall); 

		m_pValues[n] = value;
	}

	std::iostream& operator << (std::iostream& io) const
	{
		io << m_strName << m_nNumberOfPoints;
		for(size_t i = 0; i < m_nNumberOfPoints; i++)
			io << m_pValues[i];
		return io;
	}
	
	std::iostream& operator >> (std::iostream& io)
	{
		io >> m_strName;
		io >> m_nNumberOfPoints;
		m_pValues = new real_t[m_nNumberOfPoints];
		for(size_t i = 0; i < m_nNumberOfPoints; i++)
			io >> m_pValues[i];
		return io;
	}

public:
	string	m_strName;
	size_t	m_nNumberOfPoints;
	real_t*	m_pValues;
};

/*********************************************************************
	daeDataReporter_t
*********************************************************************/
class daeDataReporter_t
{
public:
	virtual ~daeDataReporter_t(void){}

public:
    virtual std::string GetName() const                                                 = 0;
    virtual void SetName(const std::string& strName)                                    = 0;
    virtual std::string GetConnectString() const                                        = 0;
    virtual void SetConnectString(const std::string& strConnectString)                  = 0;
    virtual std::string GetProcessName() const                                          = 0;
    virtual void SetProcessName(const std::string& strProcessName)                      = 0;

	virtual bool Connect(const string& strConnectString, const string& strProcessName)	= 0;
	virtual bool Disconnect(void)														= 0;
	virtual bool IsConnected(void)														= 0;
	virtual bool StartRegistration(void)												= 0;
	virtual bool RegisterDomain(const daeDataReporterDomain* pDomain)					= 0;
	virtual bool RegisterVariable(const daeDataReporterVariable* pVariable)				= 0;
	virtual bool EndRegistration(void)													= 0;
	virtual bool StartNewResultSet(real_t dTime)										= 0;
	virtual bool EndOfData(void)														= 0;
	virtual bool SendVariable(const daeDataReporterVariableValue* pVariableValue)		= 0;
};

/*********************************************************************
	daeDataReceiverVariableValue
*********************************************************************/
class daeDataReceiverDomain
{
public:
	daeDataReceiverDomain(void)
	{
		m_eType           = eDTUnknown;
		m_nNumberOfPoints = 0;
	}
	
	daeDataReceiverDomain(const string& strName, daeeDomainType eType, size_t nNoPoints)
	{
		m_strName         = strName;
		m_eType           = eType;
		m_nNumberOfPoints = nNoPoints;
        if(m_eType == eUnstructuredGrid)
            m_arrCoordinates.resize(nNoPoints);
        else
            m_arrPoints.resize(nNoPoints);
	}

	daeDataReceiverDomain(const daeDataReceiverDomain& drd) // Deep copy
	{
		m_strName         = drd.m_strName;
		m_eType           = drd.m_eType;
		m_nNumberOfPoints = drd.m_nNumberOfPoints;
        if(m_eType == eUnstructuredGrid)
            m_arrCoordinates = drd.m_arrCoordinates;
        else
            m_arrPoints = drd.m_arrPoints;
	}
	
	~daeDataReceiverDomain(void)
	{
	}

    string                  m_strName;
    daeeDomainType          m_eType;
    size_t                  m_nNumberOfPoints;
    std::vector<real_t>     m_arrPoints;
    std::vector<daePoint>   m_arrCoordinates;
};

/*********************************************************************
	daeDataReceiverVariableValue
*********************************************************************/
class daeDataReceiverVariableValue
{
public:
	daeDataReceiverVariableValue(void)
	{
		m_dTime    = -1;
		m_pValues  = NULL;
	}
	
	daeDataReceiverVariableValue(real_t time, size_t n)
	{
		m_dTime    = -1;
		m_pValues  = new real_t[n];
	}

	~daeDataReceiverVariableValue(void)
	{
		if(m_pValues)
			delete[] m_pValues;
	}

	real_t GetValue(size_t n) const
	{
		if(!m_pValues)
			daeDeclareAndThrowException(exInvalidPointer); 

		return m_pValues[n];
	}
	
	void SetValue(size_t n, real_t value)
	{
		if(!m_pValues)
			daeDeclareAndThrowException(exInvalidPointer); 

		m_pValues[n] = value;
	}

	real_t	m_dTime;
	real_t*	m_pValues;
};

/*********************************************************************
	daeDataReceiverVariable
*********************************************************************/
class daeDataReceiverVariable
{
public:
	daeDataReceiverVariable(void)
	{
		m_nNumberOfPoints = 0;
	}
	
	daeDataReceiverVariable(const daeDataReceiverVariable& var) // Deep copy
	{
		size_t i, k;
		daeDataReceiverVariableValue *pValue, *pOrigValue;
		
		m_strName         = var.m_strName;
		m_nNumberOfPoints = var.m_nNumberOfPoints;
		m_ptrarrDomains   = var.m_ptrarrDomains;
		
		for(i = 0; i < var.m_ptrarrValues.size(); i++)
		{
			pOrigValue = var.m_ptrarrValues[i];
			
			pValue = new daeDataReceiverVariableValue;
			m_ptrarrValues.push_back(pValue);
			
			pValue->m_dTime = pOrigValue->m_dTime;
			pValue->m_pValues = new real_t[m_nNumberOfPoints];
			memcpy(pValue->m_pValues, pOrigValue->m_pValues, m_nNumberOfPoints * sizeof(real_t));
		}
	}
	
	daeDataReceiverVariable(const string& strName, size_t nNumberOfPoints)
	{
		m_strName         = strName;
		m_nNumberOfPoints = nNumberOfPoints;
	}
	
	void AddDomain(daeDataReceiverDomain* pDomain)
	{
		m_ptrarrDomains.push_back(pDomain);
	}

	void AddVariableValue(const daeDataReceiverVariableValue* pVariableValue)
	{
		daeDataReceiverVariableValue *pValue;

		pValue = new daeDataReceiverVariableValue;
		m_ptrarrValues.push_back(pValue);
		
		pValue->m_dTime = pVariableValue->m_dTime;
		pValue->m_pValues = new real_t[m_nNumberOfPoints];
		memcpy(pValue->m_pValues, pVariableValue->m_pValues, m_nNumberOfPoints * sizeof(real_t));
	}

	string										m_strName;
	size_t										m_nNumberOfPoints;
	std::vector<daeDataReceiverDomain*>			m_ptrarrDomains;
	daePtrVector<daeDataReceiverVariableValue*>	m_ptrarrValues;
};

/*********************************************************************
	daeDataReceiverProcess
*********************************************************************/
class daeDataReceiverProcess
{
public:
	daeDataReceiverProcess(void)
	{
	}
	
	daeDataReceiverProcess(const string& strName)
	{
		m_strName = strName;
	}

	void RegisterDomain(const daeDataReceiverDomain* pDomain)
	{
		daeDataReceiverDomain* pDom = new daeDataReceiverDomain(*pDomain);
		m_ptrarrRegisteredDomains.push_back(pDom);
	}

	void RegisterVariable(const daeDataReceiverVariable* pVariable)
	{
		daeDataReceiverVariable *pVar = new daeDataReceiverVariable(*pVariable);
		std::pair<string, daeDataReceiverVariable*> p(pVar->m_strName, pVar);
		m_ptrmapRegisteredVariables.insert(p);
	}

	daeDataReceiverVariable* FindVariable(const string& strName)
	{
		daeDataReceiverVariable* pVariable;
		
		std::map<string, daeDataReceiverVariable*>::iterator iter = m_ptrmapRegisteredVariables.find(strName);
		if(iter == m_ptrmapRegisteredVariables.end())
			return NULL;
		else
			return (*iter).second;
	}

	string										m_strName;
	daePtrVector<daeDataReceiverDomain*>		m_ptrarrRegisteredDomains;
	daePtrMap<string, daeDataReceiverVariable*>	m_ptrmapRegisteredVariables;
};

/*********************************************************************
	daeDataReceiver_t
*********************************************************************/
// Now Observers can be attached to the DataReceiver and be informed 
// about the data just arrived in DataReceiver
class daeDataReceiver_t : public daeSubject<daeDataReceiver_t>
{
public:
	virtual ~daeDataReceiver_t(void){}

public:
	virtual bool					Start(void)																				= 0;
	virtual bool					Stop(void)																				= 0;
	virtual daeDataReceiverProcess*	GetProcess(void)																		= 0;
	virtual void					GetProcessName(string& strProcessName)													= 0;
	virtual void					GetDomains(std::vector<const daeDataReceiverDomain*>& ptrarrDomains) const				= 0;
	virtual void					GetVariables(std::map<string, const daeDataReceiverVariable*>& ptrmapVariables) const	= 0;
};

/******************************************************************
	daeDataReportingClassFactory_t
*******************************************************************/
class daeDataReportingClassFactory_t
{
public:
	virtual ~daeDataReportingClassFactory_t(void){}

public:
    virtual string   GetName(void) const			= 0;
    virtual string   GetDescription(void) const		= 0;
    virtual string   GetAuthorInfo(void) const		= 0;
    virtual string   GetLicenceInfo(void) const		= 0;
    virtual string   GetVersion(void) const			= 0;

    virtual daeDataReceiver_t*	CreateDataReceiver(const string& strClass)	= 0;
	virtual daeDataReporter_t*	CreateDataReporter(const string& strClass)	= 0;

	virtual void SupportedDataReceivers(std::vector<string>& strarrClasses)		= 0;
	virtual void SupportedDataReporters(std::vector<string>& strarrClasses)		= 0;
};
typedef daeDataReportingClassFactory_t* (*pfnGetDataReportingClassFactory)(void);


}
}

#endif
