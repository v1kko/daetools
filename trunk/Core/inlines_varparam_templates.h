// These are compiler "cheating" functions.
// Their purpose is to allow program to compile 
// only if acceptable arguments have been sent.
// They should forbid calls to: operator(), dt(), d(), d2() 
// with any arguments but simple integers and daeDEDI*
inline size_t CreateSize_tIndex(size_t arg)
{
	return arg;
}

inline size_t CreateSize_tIndex(daeDEDI* /*arg*/)
{
	daeDeclareAndThrowException(exInvalidCall);
	return ULONG_MAX;
}

inline size_t CreateSize_tIndex(daeDomainIndex& /*arg*/)
{
	daeDeclareAndThrowException(exInvalidCall);
	return ULONG_MAX;
}

inline daeDomainIndex CreateDomainIndex(daeDEDI* arg)
{
	return daeDomainIndex(arg);
}

inline daeDomainIndex CreateDomainIndex(size_t arg)
{
	return daeDomainIndex(arg);
}

inline daeDomainIndex CreateDomainIndex(daeDomainIndex& arg)
{
	return arg;
}

/******************************************************************
	daeParameter
*******************************************************************/
inline adouble daeParameter::operator()(void)
{
	if(m_ptrDomains.size() != 0)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get parameter call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 0";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in parameter [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
		return this->CreateSetupParameter((const daeDomainIndex*)NULL, 0);
	else
		return this->Create_adouble((const size_t*)NULL, 0);
}

template<typename TYPE1>
adouble	daeParameter::operator()(TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get parameter call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in parameter [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
		return this->CreateSetupParameter(indexes, 1);
	}
	else
	{
		size_t indexes[1] = {dae::core::CreateSize_tIndex(d1)};
		return this->Create_adouble(indexes, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get parameter call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in parameter [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
						             dae::core::CreateDomainIndex(d2) 
									};
		return this->CreateSetupParameter(indexes, 2);
	}
	else
	{
		size_t indexes[2] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2) 
					        };
		return this->Create_adouble(indexes, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get parameter call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in parameter [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3) 
									};
		return this->CreateSetupParameter(indexes, 3);
	}
	else
	{
		size_t indexes[3] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3) 
					        };
		return this->Create_adouble(indexes, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get parameter call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in parameter [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4) 
									};
		return this->CreateSetupParameter(indexes, 4);
	}
	else
	{
		size_t indexes[4] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4) 
					        };
		return this->Create_adouble(indexes, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get parameter call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in parameter [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5) 
									};
		return this->CreateSetupParameter(indexes, 5);
	}
	else
	{
		size_t indexes[5] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5) 
					        };
		return this->Create_adouble(indexes, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get parameter call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in parameter [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[6] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6)
									};
		return this->CreateSetupParameter(indexes, 6);
	}
	else
	{
		size_t indexes[6] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6)
					        };
		return this->Create_adouble(indexes, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get parameter call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in parameter [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[7] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7)
									};
		return this->CreateSetupParameter(indexes, 7);
	}
	else
	{
		size_t indexes[7] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7)
					        };
		return this->Create_adouble(indexes, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get parameter call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in parameter [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[8] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7), 
									 dae::core::CreateDomainIndex(d8)
									};
		return this->CreateSetupParameter(indexes, 8);
	}
	else
	{
		size_t indexes[8] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7), 
							 dae::core::CreateSize_tIndex(d8)
					        };
		return this->Create_adouble(indexes, 8);
	}
}

/******************************************************************
	daeVariable
*******************************************************************/
inline adouble daeVariable::operator()(void)
{
	if(m_ptrDomains.size() != 0)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get value call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 0";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
		return this->CreateSetupVariable((const daeDomainIndex*)NULL, 0);
	else
		return this->Create_adouble((const size_t*)NULL, 0);
}

template<typename TYPE1>
adouble	daeVariable::operator()(TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get value call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
		return this->CreateSetupVariable(indexes, 1);
	}
	else
	{
		size_t indexes[1] = {dae::core::CreateSize_tIndex(d1)};
		return this->Create_adouble(indexes, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get value call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2) 
									};
		return this->CreateSetupVariable(indexes, 2);
	}
	else
	{
		size_t indexes[2] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2) 
					        };
		return this->Create_adouble(indexes, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get value call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3) 
									};
		return this->CreateSetupVariable(indexes, 3);
	}
	else
	{
		size_t indexes[3] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3) 
					        };
		return this->Create_adouble(indexes, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get value call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4) 
									};
		return this->CreateSetupVariable(indexes, 4);
	}
	else
	{
		size_t indexes[4] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4) 
					        };
		return this->Create_adouble(indexes, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get value call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5) 
									};
		return this->CreateSetupVariable(indexes, 5);
	}
	else
	{
		size_t indexes[5] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5) 
					        };
		return this->Create_adouble(indexes, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get value call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[6] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6)
									};
		return this->CreateSetupVariable(indexes, 6);
	}
	else
	{
		size_t indexes[6] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6)
					        };
		return this->Create_adouble(indexes, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get value call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[7] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7)
									};
		return this->CreateSetupVariable(indexes, 7);
	}
	else
	{
		size_t indexes[7] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7)
					        };
		return this->Create_adouble(indexes, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid get value call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[8] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7), 
									 dae::core::CreateDomainIndex(d8)
									};
		return this->CreateSetupVariable(indexes, 8);
	}
	else
	{
		size_t indexes[8] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7), 
							 dae::core::CreateSize_tIndex(d8)
					        };
		return this->Create_adouble(indexes, 8);
	}
}


inline adouble daeVariable::dt(void)
{
	if(m_ptrDomains.size() != 0)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 0";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
		return this->CreateSetupTimeDerivative((const daeDomainIndex*)NULL, 0);
	else
		return this->Calculate_dt((const size_t*)NULL, 0);
}

template<typename TYPE1>
adouble	daeVariable::dt(TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
		return this->CreateSetupTimeDerivative(indexes, 1);
	}
	else
	{
		size_t indexes[1] = {dae::core::CreateSize_tIndex(d1)};
		return this->Calculate_dt(indexes, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2) 
									};
		return this->CreateSetupTimeDerivative(indexes, 2);
	}
	else
	{
		size_t indexes[2] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2) 
					        };
		return this->Calculate_dt(indexes, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3) 
									};
		return this->CreateSetupTimeDerivative(indexes, 3);
	}
	else
	{
		size_t indexes[3] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3) 
					        };
		return this->Calculate_dt(indexes, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4) 
									};
		return this->CreateSetupTimeDerivative(indexes, 4);
	}
	else
	{
		size_t indexes[4] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4) 
					        };
		return this->Calculate_dt(indexes, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5) 
									};
		return this->CreateSetupTimeDerivative(indexes, 5);
	}
	else
	{
		size_t indexes[5] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5) 
					        };
		return this->Calculate_dt(indexes, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[6] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6)
									};
		return this->CreateSetupTimeDerivative(indexes, 6);
	}
	else
	{
		size_t indexes[6] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6)
					        };
		return this->Calculate_dt(indexes, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[7] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7)
									};
		return this->CreateSetupTimeDerivative(indexes, 7);
	}
	else
	{
		size_t indexes[7] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7)
					        };
		return this->Calculate_dt(indexes, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[8] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7), 
									 dae::core::CreateDomainIndex(d8)
									};
		return this->CreateSetupTimeDerivative(indexes, 8);
	}
	else
	{
		size_t indexes[8] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7), 
							 dae::core::CreateSize_tIndex(d8)
					        };
		return this->Calculate_dt(indexes, 8);
	}
}


template<typename TYPE1>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
		return this->CreateSetupPartialDerivative(1, rDomain, indexes, 1);
	}
	else
	{
		size_t indexes[1] = {dae::core::CreateSize_tIndex(d1)};
		return this->partial(1, rDomain, indexes, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2) 
									};
		return this->CreateSetupPartialDerivative(1, rDomain, indexes, 2);
	}
	else
	{
		size_t indexes[2] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2) 
					        };
		return this->partial(1, rDomain, indexes, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3) 
									};
		return this->CreateSetupPartialDerivative(1, rDomain, indexes, 3);
	}
	else
	{
		size_t indexes[3] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3) 
					        };
		return this->partial(1, rDomain, indexes, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4) 
									};
		return this->CreateSetupPartialDerivative(1, rDomain, indexes, 4);
	}
	else
	{
		size_t indexes[4] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4) 
					        };
		return this->partial(1, rDomain, indexes, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5) 
									};
		return this->CreateSetupPartialDerivative(1, rDomain, indexes, 5);
	}
	else
	{
		size_t indexes[5] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5) 
					        };
		return this->partial(1, rDomain, indexes, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[6] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6)
									};
		return this->CreateSetupPartialDerivative(1, rDomain, indexes, 6);
	}
	else
	{
		size_t indexes[6] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6)
					        };
		return this->partial(1, rDomain, indexes, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[7] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7)
									};
		return this->CreateSetupPartialDerivative(1, rDomain, indexes, 7);
	}
	else
	{
		size_t indexes[7] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7)
					        };
		return this->partial(1, rDomain, indexes, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[8] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7), 
									 dae::core::CreateDomainIndex(d8)
									};
		return this->CreateSetupPartialDerivative(1, rDomain, indexes, 8);
	}
	else
	{
		size_t indexes[8] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7), 
							 dae::core::CreateSize_tIndex(d8)
					        };
		return this->partial(1, rDomain, indexes, 8);
	}
}


template<typename TYPE1>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2 call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
		return this->CreateSetupPartialDerivative(2, rDomain, indexes, 1);
	}
	else
	{
		size_t indexes[1] = {dae::core::CreateSize_tIndex(d1)};
		return this->partial(2, rDomain, indexes, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2 call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2) 
									};
		return this->CreateSetupPartialDerivative(2, rDomain, indexes, 2);
	}
	else
	{
		size_t indexes[2] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2) 
					        };
		return this->partial(2, rDomain, indexes, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2 call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3) 
									};
		return this->CreateSetupPartialDerivative(2, rDomain, indexes, 3);
	}
	else
	{
		size_t indexes[3] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3) 
					        };
		return this->partial(2, rDomain, indexes, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2 call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4) 
									};
		return this->CreateSetupPartialDerivative(2, rDomain, indexes, 4);
	}
	else
	{
		size_t indexes[4] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4) 
					        };
		return this->partial(2, rDomain, indexes, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2 call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5) 
									};
		return this->CreateSetupPartialDerivative(2, rDomain, indexes, 5);
	}
	else
	{
		size_t indexes[5] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5) 
					        };
		return this->partial(2, rDomain, indexes, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2 call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[6] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6)
									};
		return this->CreateSetupPartialDerivative(2, rDomain, indexes, 6);
	}
	else
	{
		size_t indexes[6] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6)
					        };
		return this->partial(2, rDomain, indexes, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2 call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[7] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7)
									};
		return this->CreateSetupPartialDerivative(2, rDomain, indexes, 7);
	}
	else
	{
		size_t indexes[7] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7)
					        };
		return this->partial(2, rDomain, indexes, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2 call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
		throw e;
	}
	if(!m_pModel)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid parent model in variable [" << m_strCanonicalName << "]";
		throw e;
	}

// During creation of the equations and STNs I should always create adSetupXXX nodes
// Residual functions in daeModel are not called at this stage;
// they are called later on with eGatherInfo mode
	bool bCreateSetupNodes = (m_pModel->m_pExecutionContextForGatherInfo && 
		                     (m_pModel->m_pExecutionContextForGatherInfo->m_eEquationCalculationMode == eCreateFunctionsIFsSTNs));

	if(bCreateSetupNodes)
	{
		daeDomainIndex indexes[8] = {dae::core::CreateDomainIndex(d1), 
									 dae::core::CreateDomainIndex(d2), 
									 dae::core::CreateDomainIndex(d3), 
									 dae::core::CreateDomainIndex(d4), 
									 dae::core::CreateDomainIndex(d5), 
									 dae::core::CreateDomainIndex(d6), 
									 dae::core::CreateDomainIndex(d7), 
									 dae::core::CreateDomainIndex(d8)
									};
		return this->CreateSetupPartialDerivative(2, rDomain, indexes, 8);
	}
	else
	{
		size_t indexes[8] = {dae::core::CreateSize_tIndex(d1), 
						     dae::core::CreateSize_tIndex(d2), 
							 dae::core::CreateSize_tIndex(d3), 
							 dae::core::CreateSize_tIndex(d4), 
							 dae::core::CreateSize_tIndex(d5), 
							 dae::core::CreateSize_tIndex(d6), 
							 dae::core::CreateSize_tIndex(d7), 
							 dae::core::CreateSize_tIndex(d8)
					        };
		return this->partial(2, rDomain, indexes, 8);
	}
}
