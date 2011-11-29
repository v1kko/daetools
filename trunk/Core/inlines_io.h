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

/****************************************************************************************
	daeSetModelAndCanonicalName
*****************************************************************************************/
template<class TYPE>
class daeSetModelAndCanonicalNameDelegate : public io::daeOnOpenObjectArrayDelegate_t<TYPE>
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
class daeFindDomainByID : public io::daeOnOpenRefDelegate_t<daeDomain> 
{
public:
	daeFindDomainByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daeDomain* FindObjectByID(size_t nID)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return m_pModel->FindDomain(nID);
	}

public:
	daeModel* m_pModel;
};

/****************************************************************************************
	daeFindPortByID
*****************************************************************************************/
class daeFindPortByID : public io::daeOnOpenRefDelegate_t<daePort>
{
public:
	daeFindPortByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daePort* FindObjectByID(size_t nID)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return m_pModel->FindPort(nID);
	}

public:
	daeModel* m_pModel;
};

/****************************************************************************************
	daeFindEventPortByID
*****************************************************************************************/
class daeFindEventPortByID : public io::daeOnOpenRefDelegate_t<daeEventPort>
{
public:
	daeFindEventPortByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daeEventPort* FindObjectByID(size_t nID)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return m_pModel->FindEventPort(nID);
	}

public:
	daeModel* m_pModel;
};

/****************************************************************************************
	daeFindVariableByID
*****************************************************************************************/
class daeFindVariableByID : public io::daeOnOpenRefDelegate_t<daeVariable>
{
public:
	daeFindVariableByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daeVariable* FindObjectByID(size_t nID)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return m_pModel->FindVariable(nID);
	}

public:
	daeModel* m_pModel;
};

/****************************************************************************************
	daeFindBlockByID
*****************************************************************************************/
class daeFindBlockByID : public io::daeOnOpenRefDelegate_t<daeBlock> 
{
public:
	daeFindBlockByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daeBlock* FindObjectByID(size_t /*nID*/)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return NULL;
	}

public:
	daeModel* m_pModel;
};

/****************************************************************************************
	daeFindModelByID
*****************************************************************************************/
class daeFindModelByID : public io::daeOnOpenRefDelegate_t<daeModel> 
{
public:
	daeFindModelByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daeModel* FindObjectByID(size_t /*nID*/)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return NULL;
	}

public:
	daeModel* m_pModel;
};

/****************************************************************************************
	daeFindEquationByID
*****************************************************************************************/
class daeFindEquationByID : public io::daeOnOpenRefDelegate_t<daeEquation> 
{
public:
	daeFindEquationByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daeEquation* FindObjectByID(size_t /*nID*/)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return NULL;
	}

public:
	daeModel* m_pModel;
};

/****************************************************************************************
	daeFindStateByID
*****************************************************************************************/
class daeFindStateByID : public io::daeOnOpenRefDelegate_t<daeState> 
{
public:
	daeFindStateByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daeState* FindObjectByID(size_t /*nID*/)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return NULL;
	}

public:
	daeModel* m_pModel;
};

/****************************************************************************************
	daeFindSTNByID
*****************************************************************************************/
class daeFindSTNByID : public io::daeOnOpenRefDelegate_t<daeSTN> 
{
public:
	daeFindSTNByID(daeModel* pModel)
	{
		m_pModel = pModel;
	}

	daeSTN* FindObjectByID(size_t /*nID*/)
	{
		if(!m_pModel)
			daeDeclareAndThrowException(exInvalidPointer);

		return NULL;
	}

public:
	daeModel* m_pModel;
};
