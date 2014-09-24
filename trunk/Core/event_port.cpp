#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
using namespace boost;

namespace dae 
{
namespace core 
{

/*********************************************************************************************
	daeEventPort
**********************************************************************************************/
daeEventPort::daeEventPort(void)
{
	m_bRecordEvents = true;
	m_dEventData    = 0;
}

daeEventPort::daeEventPort(string strName, daeePortType eType, daeModel* pModel, const string& strDescription)
{
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	pModel->AddEventPort(*this, strName, eType, strDescription);
	
	m_bRecordEvents = true;
	m_dEventData    = 0;
}

daeEventPort::~daeEventPort(void)
{
}

void daeEventPort::Clone(const daeEventPort& rObject)
{
	m_ePortType  = rObject.m_ePortType;
	m_dEventData = rObject.m_dEventData;
}

daeePortType daeEventPort::GetType(void) const
{
	return m_ePortType;
}

void daeEventPort::SetType(daeePortType eType)
{
	m_ePortType = eType;
}

// Called by daeAction::Execute() to trigger an event
void daeEventPort::SendEvent(real_t data)
{
	if(m_ePortType != eOutletPort)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_dEventData = data;
	
	if(m_bRecordEvents)
	{
        real_t time = m_pModel->m_pDataProxy->GetCurrentTime_();
		m_listEvents.push_back(std::make_pair(time, m_dEventData));
	}
	
	//std::cout << "    Event sent from the outlet port: " << GetCanonicalName() << ", data = " << data << std::endl;
	
// Observers in this case are inlet event ports
	Notify(&data);
}

// Called by the outlet event port that this port is attached to
void daeEventPort::Update(daeEventPort_t* pSubject, void* data)
{
	if(m_ePortType != eInletPort)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_dEventData = *((real_t*)data);
	
	if(m_bRecordEvents)
	{
        real_t time = m_pModel->m_pDataProxy->GetCurrentTime_();
		m_listEvents.push_back(std::make_pair(time, m_dEventData));
	}
	
	//std::cout << "    Update received in inlet port: " << GetCanonicalName() << ", dEventData = " << m_dEventData << std::endl;
	
// Observers in this case are OnEventActions
	Notify(data);
}

void daeEventPort::ReceiveEvent(real_t data)
{
	if(m_ePortType != eInletPort)
		daeDeclareAndThrowException(exInvalidPointer);

    m_dEventData = data;
	
	if(m_bRecordEvents)
	{
        real_t time = m_pModel->m_pDataProxy->GetCurrentTime_();
		m_listEvents.push_back(std::make_pair(time, m_dEventData));
	}
	
// Observers in this case are OnEventActions
	Notify(&data);
}

void daeEventPort::Initialize(void)
{	
}

real_t daeEventPort::GetEventData(void)
{
	return m_dEventData;
}

bool daeEventPort::GetRecordEvents() const
{
	return m_bRecordEvents;
}

void daeEventPort::SetRecordEvents(bool bRecordEvents)
{
	m_bRecordEvents = bRecordEvents;
}

const std::list< std::pair<real_t, real_t> >& daeEventPort::GetListOfEvents(void) const
{
	return m_listEvents;
}

adouble daeEventPort::operator()(void)
{
	adouble a;
	a.setGatherInfo(true);
	a.node = adNodePtr(new adEventPortDataNode(this));
	return a;
}

bool daeEventPort::CheckObject(std::vector<string>& strarrErrors) const
{
	bool bReturn = true;
	
	bReturn = daeObject::CheckObject(strarrErrors);
	
	return bReturn;
}

void daeEventPort::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeObject::Open(pTag);

	strName = "PortType";
	OpenEnum(pTag, strName, m_ePortType);
}

void daeEventPort::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "PortType";
	SaveEnum(pTag, strName, m_ePortType);
}
	
void daeEventPort::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
}

}
}
