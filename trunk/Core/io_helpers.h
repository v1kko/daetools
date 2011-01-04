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
#ifndef DAE_IO_FUNCTION_HELPERS_H
#define DAE_IO_FUNCTION_HELPERS_H

namespace dae
{
namespace io
{
/****************************************************************************************
	daeSetModelAndCanonicalName
*****************************************************************************************/
template<class TYPE>
class daeSetModelAndCanonicalNameDelegate : public daeOnOpenObjectArrayDelegate_t<TYPE> 
{
public:
	daeSetModelAndCanonicalNameDelegate(daeObject* pParent, daeModel* pModel)
	{
		m_pParent = pParent;
		m_pModel  = pModel;
	}

	void BeforeOpenObject(TYPE* pObject)
	{
		if(!pObject)
			daeDeclareAndThrowException(exInvalidPointer);
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		pObject->SetModel(m_pModel);
	}

	void AfterOpenObject(TYPE* pObject)
	{
		if(!pObject)
			daeDeclareAndThrowException(exInvalidPointer);
		if(!m_pParent)
			daeDeclareAndThrowException(exInvalidPointer);

		string strName = m_pParent->GetCanonicalName() + "." + pObject->GetName();
		pObject->SetCanonicalName(strName);
	}
	
	void AfterAllObjectsOpened(void)
	{
	}

public:
	daeObject* m_pParent;
	daeModel*  m_pModel;
};

/****************************************************************************************
	daeFindDomainByID
*****************************************************************************************/
class daeFindDomainByID : public daeOnOpenRefDelegate_t<daeDomain, unsigned long>
 
{
public:
	daeFindDomainByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daeDomain* FindObjectByReference(unsigned long& nID)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return m_pModel->FindDomain(nID);
	}

public:
	daeModel* m_pModel;
};

}
}

#endif
