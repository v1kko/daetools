/******************************************************************
	daeEquation0
*******************************************************************/
template<class MODEL>
class daeEquation0 : public daeEquation
{
public:
	daeDeclareDynamicClass(daeEquation0)
	daeEquation0(const string& strName, daeModel* pModel, adouble (MODEL::*pfn)(void))
	{
		m_strShortName	= strName;
		m_pModel		= pModel;
		pfnCalculate	= pfn;
		m_eEquationDefinitionMode = eMemberFunctionPointer;
		m_eEquationEvaluationMode = eFunctionEvaluation;
	}

protected:
	virtual adouble Calculate()
	{
		MODEL* pModel = dynamic_cast<MODEL*>(m_pModel);
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		return (pModel->*pfnCalculate)();
	}
	
public:
	bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!daeEquation::CheckObject(strarrErrors))
			bCheck = false;
	
	// Check definition mode
		if(m_eEquationDefinitionMode != eMemberFunctionPointer)
		{
			strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check number of distributed equation domain infos
		if(m_ptrarrDistributedEquationDomainInfos.size() != 0)
		{
			strError = "Invalid number of distributed equation domain infos in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check residual function pointer
		if(!pfnCalculate)
		{
			strError = "Invalid residual function pointer in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	adouble	(MODEL::*pfnCalculate)();
};

/******************************************************************
	daeEquation1
*******************************************************************/
template<class MODEL>
class daeEquation1 : public daeEquation
{
public:
	daeDeclareDynamicClass(daeEquation1)
	daeEquation1(const string& strName, daeModel* pModel, adouble (MODEL::*pfn)(size_t))
	{
		m_strShortName	= strName;
		m_pModel		= pModel;
		pfnCalculate	= pfn;
		m_eEquationDefinitionMode = eMemberFunctionPointer;
		m_eEquationEvaluationMode = eFunctionEvaluation;
	}

protected:
	virtual adouble Calculate(size_t nDomain1)
	{
		MODEL* pModel = dynamic_cast<MODEL*>(m_pModel);
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		return (pModel->*pfnCalculate)(nDomain1);
	}
	
public:
	bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!daeEquation::CheckObject(strarrErrors))
			bCheck = false;
	
	// Check definition mode
		if(m_eEquationDefinitionMode != eMemberFunctionPointer)
		{
			strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check number of distributed equation domain infos
		if(m_ptrarrDistributedEquationDomainInfos.size() != 1)
		{
			strError = "Invalid number of distributed equation domain infos in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check residual function pointer
		if(!pfnCalculate)
		{
			strError = "Invalid residual function pointer in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	adouble	(MODEL::*pfnCalculate)(size_t);
};

/******************************************************************
daeEquation2
*******************************************************************/
template<class MODEL>
class daeEquation2 : public daeEquation
{
public:
	daeDeclareDynamicClass(daeEquation2)
	daeEquation2(const string& strName, daeModel* pModel, adouble (MODEL::*pfn)(size_t, size_t))
	{
		m_strShortName	= strName;
		m_pModel		= pModel;
		pfnCalculate	= pfn;
		m_eEquationDefinitionMode = eMemberFunctionPointer;
		m_eEquationEvaluationMode = eFunctionEvaluation;
	}

protected:
	virtual adouble Calculate(size_t nDomain1, size_t nDomain2)
	{
		MODEL* pModel = dynamic_cast<MODEL*>(m_pModel);
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		return (pModel->*pfnCalculate)(nDomain1, nDomain2);
	}

public:
	bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!daeEquation::CheckObject(strarrErrors))
			bCheck = false;
	
	// Check definition mode
		if(m_eEquationDefinitionMode != eMemberFunctionPointer)
		{
			strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check number of distributed equation domain infos
		if(m_ptrarrDistributedEquationDomainInfos.size() != 2)
		{
			strError = "Invalid number of distributed equation domain infos in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check residual function pointer
		if(!pfnCalculate)
		{
			strError = "Invalid residual function pointer in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	adouble	(MODEL::*pfnCalculate)(size_t, size_t);
};

/******************************************************************
	daeEquation3
*******************************************************************/
template<class MODEL>
class daeEquation3 : public daeEquation
{
public:
	daeDeclareDynamicClass(daeEquation3)
	daeEquation3(const string& strName, daeModel* pModel, adouble (MODEL::*pfn)(size_t, size_t, size_t))
	{
		m_strShortName	= strName;
		m_pModel		= pModel;
		pfnCalculate	= pfn;
		m_eEquationDefinitionMode = eMemberFunctionPointer;
		m_eEquationEvaluationMode = eFunctionEvaluation;
	}

protected:
	virtual adouble	Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3)
	{
		MODEL* pModel = dynamic_cast<MODEL*>(m_pModel);
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		return (pModel->*pfnCalculate)(nDomain1, nDomain2, nDomain3);
	}

public:
	bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!daeEquation::CheckObject(strarrErrors))
			bCheck = false;
	
	// Check definition mode
		if(m_eEquationDefinitionMode != eMemberFunctionPointer)
		{
			strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check number of distributed equation domain infos
		if(m_ptrarrDistributedEquationDomainInfos.size() != 3)
		{
			strError = "Invalid number of distributed equation domain infos in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check residual function pointer
		if(!pfnCalculate)
		{
			strError = "Invalid residual function pointer in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	adouble	(MODEL::*pfnCalculate)(size_t, size_t, size_t);
};

/******************************************************************
	daeEquation4
*******************************************************************/
template<class MODEL>
class daeEquation4 : public daeEquation
{
public:
	daeDeclareDynamicClass(daeEquation4)
	daeEquation4(const string& strName, daeModel* pModel, adouble (MODEL::*pfn)(size_t, size_t, size_t, size_t))
	{
		m_strShortName	= strName;
		m_pModel		= pModel;
		pfnCalculate	= pfn;
		m_eEquationDefinitionMode = eMemberFunctionPointer;
		m_eEquationEvaluationMode = eFunctionEvaluation;
	}

protected:
	virtual adouble	Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4)
	{
		MODEL* pModel = dynamic_cast<MODEL*>(m_pModel);
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		return (pModel->*pfnCalculate)(nDomain1, nDomain2, nDomain3, nDomain4);
	}

public:
	bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!daeEquation::CheckObject(strarrErrors))
			bCheck = false;
	
	// Check definition mode
		if(m_eEquationDefinitionMode != eMemberFunctionPointer)
		{
			strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check number of distributed equation domain infos
		if(m_ptrarrDistributedEquationDomainInfos.size() != 4)
		{
			strError = "Invalid number of distributed equation domain infos in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check residual function pointer
		if(!pfnCalculate)
		{
			strError = "Invalid residual function pointer in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	adouble	(MODEL::*pfnCalculate)(size_t, size_t, size_t, size_t);
};

/******************************************************************
	daeEquation5
*******************************************************************/
template<class MODEL>
class daeEquation5 : public daeEquation
{
public:
	daeDeclareDynamicClass(daeEquation5)
	daeEquation5(const string& strName, daeModel* pModel, adouble (MODEL::*pfn)(size_t, size_t, size_t, size_t, size_t))
	{
		m_strShortName	= strName;
		m_pModel		= pModel;
		pfnCalculate	= pfn;
		m_eEquationDefinitionMode = eMemberFunctionPointer;
		m_eEquationEvaluationMode = eFunctionEvaluation;
	}

protected:
	virtual adouble	Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5)
	{
		MODEL* pModel = dynamic_cast<MODEL*>(m_pModel);
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		return (pModel->*pfnCalculate)(nDomain1, nDomain2, nDomain3, nDomain4, nDomain5);
	}

public:
	bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!daeEquation::CheckObject(strarrErrors))
			bCheck = false;
	
	// Check definition mode
		if(m_eEquationDefinitionMode != eMemberFunctionPointer)
		{
			strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check number of distributed equation domain infos
		if(m_ptrarrDistributedEquationDomainInfos.size() != 5)
		{
			strError = "Invalid number of distributed equation domain infos in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check residual function pointer
		if(!pfnCalculate)
		{
			strError = "Invalid residual function pointer in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	adouble	(MODEL::*pfnCalculate)(size_t, size_t, size_t, size_t, size_t);
};

/******************************************************************
	daeEquation6
*******************************************************************/
template<class MODEL>
class daeEquation6 : public daeEquation
{
public:
	daeDeclareDynamicClass(daeEquation6)
	daeEquation6(const string& strName, daeModel* pModel, adouble (MODEL::*pfn)(size_t, size_t, size_t, size_t, size_t, size_t))
	{
		m_strShortName	= strName;
		m_pModel		= pModel;
		pfnCalculate	= pfn;
		m_eEquationDefinitionMode = eMemberFunctionPointer;
		m_eEquationEvaluationMode = eFunctionEvaluation;
	}

protected:
	virtual adouble	Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6)
	{
		MODEL* pModel = dynamic_cast<MODEL*>(m_pModel);
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		return (pModel->*pfnCalculate)(nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6);
	}

public:
	bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!daeEquation::CheckObject(strarrErrors))
			bCheck = false;
	
	// Check definition mode
		if(m_eEquationDefinitionMode != eMemberFunctionPointer)
		{
			strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check number of distributed equation domain infos
		if(m_ptrarrDistributedEquationDomainInfos.size() != 6)
		{
			strError = "Invalid number of distributed equation domain infos in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check residual function pointer
		if(!pfnCalculate)
		{
			strError = "Invalid residual function pointer in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	adouble	(MODEL::*pfnCalculate)(size_t, size_t, size_t, size_t, size_t, size_t);
};

/******************************************************************
	daeEquation7
*******************************************************************/
template<class MODEL>
class daeEquation7 : public daeEquation
{
public:
	daeDeclareDynamicClass(daeEquation7)
	daeEquation7(const string& strName, daeModel* pModel, adouble (MODEL::*pfn)(size_t, size_t, size_t, size_t, size_t, size_t, size_t))
	{
		m_strShortName	= strName;
		m_pModel		= pModel;
		pfnCalculate	= pfn;
		m_eEquationDefinitionMode = eMemberFunctionPointer;
		m_eEquationEvaluationMode = eFunctionEvaluation;
	}

protected:
	virtual adouble	Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7)
	{
		MODEL* pModel = dynamic_cast<MODEL*>(m_pModel);
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		return (pModel->*pfnCalculate)(nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6, nDomain7);
	}

public:
	bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!daeEquation::CheckObject(strarrErrors))
			bCheck = false;
	
	// Check definition mode
		if(m_eEquationDefinitionMode != eMemberFunctionPointer)
		{
			strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check number of distributed equation domain infos
		if(m_ptrarrDistributedEquationDomainInfos.size() != 7)
		{
			strError = "Invalid number of distributed equation domain infos in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check residual function pointer
		if(!pfnCalculate)
		{
			strError = "Invalid residual function pointer in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	adouble	(MODEL::*pfnCalculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t);
};

/******************************************************************
	daeEquation8
*******************************************************************/
template<class MODEL>
class daeEquation8 : public daeEquation
{
public:
	daeDeclareDynamicClass(daeEquation8)
	daeEquation8(const string& strName, daeModel* pModel, adouble (MODEL::*pfn)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t))
	{
		m_strShortName	= strName;
		m_pModel		= pModel;
		pfnCalculate	= pfn;
		m_eEquationDefinitionMode = eMemberFunctionPointer;
		m_eEquationEvaluationMode = eFunctionEvaluation;
	}

protected:
	virtual adouble	Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7, size_t nDomain8)
	{
		MODEL* pModel = dynamic_cast<MODEL*>(m_pModel);
		if(!pModel)
			daeDeclareAndThrowException(exInvalidPointer);
		return (pModel->*pfnCalculate)(nDomain1, nDomain2, nDomain3, nDomain4, nDomain5, nDomain6, nDomain7, nDomain8);
	}

public:
	bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!daeEquation::CheckObject(strarrErrors))
			bCheck = false;
	
	// Check definition mode
		if(m_eEquationDefinitionMode != eMemberFunctionPointer)
		{
			strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check number of distributed equation domain infos
		if(m_ptrarrDistributedEquationDomainInfos.size() != 8)
		{
			strError = "Invalid number of distributed equation domain infos in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check residual function pointer
		if(!pfnCalculate)
		{
			strError = "Invalid residual function pointer in equation [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	adouble	(MODEL::*pfnCalculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
};

/******************************************************************
	daeModel
*******************************************************************/
template<typename Model>
daeEquation* daeModel::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)())
{
	dae::core::daeEquation0<Model>* pEquation;
	pEquation = new daeEquation0<Model>(strFunctionName, this, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<typename Model>
daeEquation* daeModel::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation1<Model>(strFunctionName, this, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<typename Model>
daeEquation* daeModel::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation2<Model>(strFunctionName, this, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<typename Model>
daeEquation* daeModel::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation3<Model>(strFunctionName, this, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<typename Model>
daeEquation* daeModel::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation4<Model>(strFunctionName, this, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<typename Model>
daeEquation* daeModel::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation5<Model>(strFunctionName, this, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<typename Model>
daeEquation* daeModel::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation6<Model>(strFunctionName, this, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<typename Model>
daeEquation* daeModel::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation7<Model>(strFunctionName, this, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<typename Model>
daeEquation* daeModel::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation8<Model>(strFunctionName, this, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

/******************************************************************
	daeState
*******************************************************************/
template<class Model>
daeEquation* daeState::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)())
{
	daeEquation* pEquation;
	pEquation = new daeEquation0<Model>(strFunctionName, m_pModel, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<class Model>
daeEquation* daeState::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation1<Model>(strFunctionName, m_pModel, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<class Model>
daeEquation* daeState::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation2<Model>(strFunctionName, m_pModel, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<class Model>
daeEquation* daeState::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation3<Model>(strFunctionName, m_pModel, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<class Model>
daeEquation* daeState::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation4<Model>(strFunctionName, m_pModel, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<class Model>
daeEquation* daeState::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation5<Model>(strFunctionName, m_pModel, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<class Model>
daeEquation* daeState::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation6<Model>(strFunctionName, m_pModel, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<class Model>
daeEquation* daeState::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation7<Model>(strFunctionName, m_pModel, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}

template<class Model>
daeEquation* daeState::AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t))
{
	daeEquation* pEquation;
	pEquation = new daeEquation8<Model>(strFunctionName, m_pModel, Calculate);
	this->AddEquation(pEquation);
	return pEquation;
}



