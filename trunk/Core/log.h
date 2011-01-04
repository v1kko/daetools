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
	virtual void		Message(const string& strMessage, size_t nSeverity) = 0;
	virtual std::string	GetIndentString(void) const							= 0;
	virtual void		SetEnabled(bool bEnabled)							= 0;
	virtual bool		GetEnabled(void) const								= 0;
	virtual void		SetIndent(size_t nIndent)							= 0;
	virtual size_t		GetIndent(void) const								= 0;
	virtual void		IncreaseIndent(size_t nOffset)						= 0;
	virtual void		DecreaseIndent(size_t nOffset)						= 0;
};


}
}


#endif
