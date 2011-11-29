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
#ifndef DAE_EVENT_HANDLING_H
#define DAE_EVENT_HANDLING_H

#include "coreimpl.h"
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/thread.hpp>

namespace dae
{
namespace core
{
/******************************************************************
	daeRemoteEventReceiver
*******************************************************************/
class daeRemoteEventReceiver : public daeSubject<daeEventPort_t>
{
public:
	daeRemoteEventReceiver();
	virtual ~daeRemoteEventReceiver();
	
public:
	virtual void Notify(void* data);
	void EventsReceiver();

protected:
	boost::shared_ptr<boost::interprocess::message_queue>	mqEventQueue;
	boost::shared_ptr<boost::interprocess::message_queue>	mqStatusQueue;
	boost::shared_ptr<boost::thread>						m_pThread;
};

/******************************************************************
	daeRemoteEventSender
*******************************************************************/
class daeRemoteEventSender : public daeObserver<daeEventPort_t>
{
public:
	daeRemoteEventSender(/*daeRemoteEventReceiver* pReceiver*/);
	virtual ~daeRemoteEventSender();
	
public:
	virtual void Update(daeEventPort_t *pSubject, void* data);

protected:
	//daeRemoteEventReceiver* m_pReceiver;
	boost::shared_ptr<boost::interprocess::message_queue>	mqEventQueue;
	boost::shared_ptr<boost::interprocess::message_queue>	mqStatusQueue;
};

}
}

#endif
