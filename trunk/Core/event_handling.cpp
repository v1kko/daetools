#include "stdafx.h"
#include "event_handling.h"

namespace dae 
{
namespace core 
{
/******************************************************************
	daeRemoteEventReceiver
*******************************************************************/
daeRemoteEventReceiver::daeRemoteEventReceiver()
{
	m_pThread.reset();
	m_pThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&daeRemoteEventReceiver::EventsReceiver, this)));
}

daeRemoteEventReceiver::~daeRemoteEventReceiver()
{
	m_pThread->join();
	m_pThread.reset();
}

void daeRemoteEventReceiver::EventsReceiver()
{
	real_t data;
	unsigned int priority;
    std::size_t recvd_size;
	int iSuccess = 1;

	try
	{
		//Create a message_queue.
		mqEventQueue.reset (new boost::interprocess::message_queue(boost::interprocess::open_only, "EventQueue"));
		//mqStatusQueue.reset(new boost::interprocess::message_queue(boost::interprocess::open_only, "StatusQueue"));
		
		// Receive events in a loop
		while(true)
		{
			mqEventQueue->receive(&data, sizeof(data), recvd_size, priority);
			Notify(&data);
			//mqStatusQueue->send(&iSuccess, sizeof(iSuccess), 0);
		}
	}
	catch(boost::interprocess::interprocess_exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

void daeRemoteEventReceiver::Notify(void* data)
{
	real_t dEventData = *((real_t*)data);
	std::cout << "    daeRemoteEventReceiver: Notify called with data = " << dEventData << std::endl;
	
	daeSubject<daeEventPort_t>::Notify(data);
}

/******************************************************************
	daeRemoteEventSender
*******************************************************************/
daeRemoteEventSender::daeRemoteEventSender(/*daeRemoteEventReceiver* pReceiver*/)
{
	//m_pReceiver = pReceiver;
	try
	{
		//Erase previous message queue
		boost::interprocess::message_queue::remove("EventQueue");
		//boost::interprocess::message_queue::remove("StatusQueue");
		
		//Create a message_queue.
		mqEventQueue.reset (new boost::interprocess::message_queue(boost::interprocess::create_only, "EventQueue",  1, sizeof(real_t)));
		//mqStatusQueue.reset(new boost::interprocess::message_queue(boost::interprocess::create_only, "StatusQueue", 1, sizeof(int)));
	}
	catch(boost::interprocess::interprocess_exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

daeRemoteEventSender::~daeRemoteEventSender()
{
	try
	{
		boost::interprocess::message_queue::remove("EventQueue");
		//boost::interprocess::message_queue::remove("StatusQueue");
		mqEventQueue.reset();
		//mqStatusQueue.reset();
	}
	catch(boost::interprocess::interprocess_exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

void daeRemoteEventSender::Update(daeEventPort_t *pSubject, void* data)
{
	int iSuccess;
	unsigned int priority;
    std::size_t recvd_size;

	real_t dEventData = *((real_t*)data);
	std::cout << "    daeRemoteEventSender: update received from: " << pSubject->GetCanonicalName() << ", dEventData = " << dEventData << std::endl;
	
	//m_pReceiver->Notify(data);
	try
	{
		mqEventQueue->send(&dEventData, sizeof(dEventData), 0);
		//mqStatusQueue->receive(&iSuccess, sizeof(iSuccess), recvd_size, priority);
		std::cout << "    daeRemoteEventSender: successfuly sent the message" << std::endl;
	}
	catch(boost::interprocess::interprocess_exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}


}
}
