#ifndef DAE_BASE_LOGGING_H
#define DAE_BASE_LOGGING_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include "log.h"

namespace dae
{
namespace logging
{
/********************************************************************
	daeSimpleLog
*********************************************************************/
class daeSimpleLog : public daeLog_t
{
public:
	daeSimpleLog()
	{
	}

	virtual ~daeSimpleLog(void)
	{
	}

public:
	virtual void Message(const string& strMessage, size_t nSeverity)
	{
		if(strMessage.empty())
			return;
		m_strarrMessages.push_back(strMessage);
	}

protected:
	std::vector<string> m_strarrMessages;
};

/********************************************************************
	daeStdOutLog
*********************************************************************/
class daeStdOutLog : public daeLog_t
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
		std::cout << strMessage << std::endl;
		std::cout.flush();
	}
};

/********************************************************************
	daeFileLog
*********************************************************************/
class daeFileLog : public daeLog_t
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
		if(file.is_open())
			file << strMessage << std::endl;
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

