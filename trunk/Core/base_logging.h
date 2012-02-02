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
#ifndef DAE_BASE_LOGGING_H
#define DAE_BASE_LOGGING_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <math.h>
#include "log.h"
#include "../config.h"
#include <boost/format.hpp>
#include <boost/timer.hpp>

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
		m_bEnabled  = true;
		m_nIndent   = 0;
		m_dProgress = 0;
		
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
	
	virtual void SetProgress(real_t nProgress)
	{
		m_dProgress = nProgress;
	}

	virtual real_t GetProgress(void) const
	{
		return m_dProgress;
	}
	
	virtual string JoinMessages(const std::string& join = std::string("\n")) const
	{
		string strAllMessages;
		for(size_t i = 0; i < m_strarrMessages.size(); i++)
			strAllMessages += m_strarrMessages[i] + join;
		return strAllMessages;
	}
	
	virtual string GetETA(void) const
	{
		double dt, left, secs;
		int days, hours, mins;
		
		dt = start.elapsed();
		if(m_dProgress < 100 && m_dProgress > 0)
			left = 100.0 * dt / m_dProgress - dt;
		else
			left = 0.0;
		
		days  = int(::floor(left / 86400));
		left  = left - days * 86400;
		hours = int(::floor(left / 3600));
		left  = left - hours * 3600;
		mins  = int(::floor(left / 60));
		secs  = double(left - mins * 60);
		if(days > 0)
			return (boost::format("ETA: [%02dd %02dh %02dm %04.1fs]\r") % days % hours % mins % secs).str();
		else if(hours > 0)
			return (boost::format("ETA: [%02dh %02dm %04.1fs]\r") % hours % mins % secs).str();
		else if(mins > 0)
			return (boost::format("ETA: [%02dm %04.1fs]\r") % mins % secs).str();
		else
			return (boost::format("ETA: [%04.1fs]\r") % secs).str();
	}
	
	virtual string GetPercentageDone(void) const
	{
		return (boost::format("%.2f%%") % m_dProgress).str();
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
	real_t				m_dProgress;
	std::string			m_strIndent;
	std::string			m_strIndentUnit;
	boost::timer        start;
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
	        string msg = (boost::format("%-30s") % (m_strIndent + strMessage)).str();
			std::cout << msg << std::endl;
	        std::cout << (boost::format(" %s %s\r") % GetPercentageDone() % GetETA()).str();
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

