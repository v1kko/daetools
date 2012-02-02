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
	virtual void		Message(const string& strMessage, size_t nSeverity)		= 0;
	virtual string		JoinMessages(const string& join = string("\n")) const	= 0;
	virtual std::string	GetIndentString(void) const								= 0;
	virtual void		SetEnabled(bool bEnabled)								= 0;
	virtual bool		GetEnabled(void) const									= 0;
	virtual void		SetIndent(size_t nIndent)								= 0;
	virtual size_t		GetIndent(void) const									= 0;
	virtual void		IncreaseIndent(size_t nOffset)							= 0;
	virtual void		DecreaseIndent(size_t nOffset)							= 0;
	virtual void		SetProgress(real_t nProgress)							= 0;
	virtual real_t		GetProgress(void) const									= 0;
	virtual std::string	GetETA(void) const										= 0;
	virtual std::string	GetPercentageDone(void) const							= 0;
};


}
}


#endif
