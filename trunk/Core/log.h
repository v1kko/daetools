#ifndef DAE_LOG_H
#define DAE_LOG_H

#include "definitions.h"

namespace dae 
{
namespace logging 
{

enum daeeLogMessageType
{
	eInformation = 0,
	eWarning,
	eError
};

class daeLog_t
{
public:
	virtual ~daeLog_t(void){}

public:
        virtual void Message(const string& strMessage, size_t nSeverity) = 0;
};


}
}


#endif
