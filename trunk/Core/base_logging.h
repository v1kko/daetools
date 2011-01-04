#ifndef DAE_BASE_LOGGING_H
#define DAE_BASE_LOGGING_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include "log.h"
#include "../config.h"

namespace dae
{
namespace logging
{
/********************************************************************
	daeBaseLog
*********************************************************************/
class daeBaseLog : public daeLog_t
{
public:
	daeBaseLog(void)
	{
		m_bEnabled = true;
		m_nIndent  = 0;
		
		daeConfig& cfg  = daeConfig::GetConfig();
		m_strIndentUnit = cfg.Get<std::string>("daetools.core.logIndent", "\t");
	}

	virtual ~daeBaseLog(void)
	{
	}

public:
	virtual void Message(const string& strMessage, size_t nSeverity)
	{
		if(strMessage.empty())
			return;
		
		if(m_bEnabled)
			m_strarrMessages.push_back(m_strIndent + strMessage);
	}
	
	virtual string GetIndentString(void) const
	{
		return m_strIndent;
	}
	
	virtual void SetEnabled(bool bEnabled)
	{
		m_bEnabled = bEnabled;
	}	

	virtual bool GetEnabled(void) const
	{
		return m_bEnabled;
	}
	
	virtual void SetIndent(size_t nIndent)
	{
		m_nIndent = nIndent;
		UpdateIndent();
	}
	
	virtual size_t GetIndent(void) const
	{
		return m_nIndent;
	}
	
	virtual void IncreaseIndent(size_t nOffset)
	{
		m_nIndent += nOffset;
		UpdateIndent();
	}
	
	virtual void DecreaseIndent(size_t nOffset)
	{
		if(m_nIndent > 0)
		{
			m_nIndent -= nOffset;
			UpdateIndent();
		}
	}
	
	void UpdateIndent(void)
	{
		m_strIndent.clear();
		for(size_t i = 0; i < m_nIndent; i++)
			m_strIndent += m_strIndentUnit;		
	}

protected:
	std::vector<string> m_strarrMessages;
	bool				m_bEnabled;
	size_t				m_nIndent;
	std::string			m_strIndent;
	std::string			m_strIndentUnit;
};

/********************************************************************
	daeStdOutLog
*********************************************************************/
class daeStdOutLog : public daeBaseLog
{
public:
	daeStdOutLog()
	{
	}

	virtual ~daeStdOutLog(void)
	{
	}

public:
	virtual void Message(const string& strMessage, size_t nSeverity)
	{
		if(strMessage.empty())
			return;
		
		if(m_bEnabled)
		{
			std::cout << m_strIndent + strMessage << std::endl;
			std::cout.flush();
		}
	}
};

/********************************************************************
	daeFileLog
*********************************************************************/
class daeFileLog : public daeBaseLog
{
public:
	daeFileLog(const string& strFileName)
	{
		file.open(strFileName.c_str());
	}

	virtual ~daeFileLog(void)
	{
		file.flush();
		file.close();
	}

public:
	virtual void Message(const string& strMessage, size_t nSeverity)
	{
		if(strMessage.empty())
			return;
		
		if(m_bEnabled)
		{
			if(file.is_open())
				file << m_strIndent + strMessage << std::endl;
		}
	}

protected:
	std::ofstream file;
};


daeLog_t* daeCreateFileLog(const string& strFileName);
daeLog_t* daeCreateStdOutLog(void);
daeLog_t* daeCreateTCPIPLog(const string& strIPAddress, int nPort);
//daeLog_t* daeCreateTCPIPLogServer(int nPort);

}
}

#endif

