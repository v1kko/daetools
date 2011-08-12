#include "stdafx.h"
#include "coreimpl.h"
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
}

daeEventPort::daeEventPort(string strName, daeePortType eType, daeModel* pModel, const string& strDescription)
{
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	pModel->AddEventPort(*this, strName, eType, strDescription);
}

daeEventPort::~daeEventPort(void)
{
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
void daeEventPort::SendEvent(void* data)
{
	if(m_ePortType != eOutletPort)
		daeDeclareAndThrowException(exInvalidPointer);
	
	std::cout << "Event sent from the outlet port: " << GetName() << std::endl;
	
// Observers in this case are inlet event ports
	Notify(data);
}

// Called by the outlet event port that this port is attached to
void daeEventPort::Update(daeEventPort_t* pSubject, void* data)
{
	if(m_ePortType != eInletPort)
		daeDeclareAndThrowException(exInvalidPointer);
	
	std::cout << "Update received in inlet port: " << GetName() << std::endl;
	
// Observers in this case are actions
	Notify(data);
}

void daeEventPort::Initialize(void)
{
	
}

bool daeEventPort::CheckObject(std::vector<string>& strarrErrors) const
{
	return true;
}

void daeEventPort::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeObject::Open(pTag);

	strName = "Type";
	OpenEnum(pTag, strName, m_ePortType);
}

void daeEventPort::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "Type";
	SaveEnum(pTag, strName, m_ePortType);

//	strName = "ActionsRefs";
//	pTag->SaveObjectRefArray(strName, m_ptrDomains);
}
	
void daeEventPort::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
}

}
}
