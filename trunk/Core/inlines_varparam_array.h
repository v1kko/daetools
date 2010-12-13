// These are compiler "cheating" functions.
// Their purpose is to allow program to compile 
// only if acceptable arguments have been sent.
// They should forbid calls to: Array(), dt_array(), d_array(), d2_array() 
// with any arguments but simple integers, daeDEDI* or daeIndexRange

inline daeArrayRange CreateRange(daeDEDI* arg)
{
	return daeArrayRange(arg);
}

inline daeArrayRange CreateRange(size_t arg)
{
	return daeArrayRange(arg);
}

inline daeArrayRange CreateRange(daeIndexRange arg)
{
	return daeArrayRange(arg);
}

inline daeArrayRange CreateRange(daeArrayRange arg)
{
	return arg;
}

/******************************************************************
	daeParameter
*******************************************************************/
template<typename TYPE1>
adouble_array daeParameter::array(TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
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
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->CreateSetupParameterArray(ranges, 1);
	}
	else
	{
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->Create_adouble_array(ranges, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble_array daeParameter::array(TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
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
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
							      };
		return this->CreateSetupParameterArray(ranges, 2);
	}
	else
	{
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
								  };
		return this->Create_adouble_array(ranges, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble_array daeParameter::array(TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
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
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->CreateSetupParameterArray(ranges, 3);
	}
	else
	{
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->Create_adouble_array(ranges, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble_array daeParameter::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
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
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->CreateSetupParameterArray(ranges, 4);
	}
	else
	{
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->Create_adouble_array(ranges, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble_array daeParameter::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
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
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->CreateSetupParameterArray(ranges, 5);
	}
	else
	{
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->Create_adouble_array(ranges, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble_array daeParameter::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
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
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->CreateSetupParameterArray(ranges, 6);
	}
	else
	{
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->Create_adouble_array(ranges, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble_array daeParameter::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
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
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->CreateSetupParameterArray(ranges, 7);
	}
	else
	{
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->Create_adouble_array(ranges, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble_array daeParameter::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
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
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->CreateSetupParameterArray(ranges, 8);
	}
	else
	{
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->Create_adouble_array(ranges, 8);
	}
}

/******************************************************************
	daeVariable
*******************************************************************/
template<typename TYPE1>
adouble_array daeVariable::array(TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
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
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->CreateSetupVariableArray(ranges, 1);
	}
	else
	{
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->Create_adouble_array(ranges, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble_array daeVariable::array(TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
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
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
							   };
		return this->CreateSetupVariableArray(ranges, 2);
	}
	else
	{
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
							   };
		return this->Create_adouble_array(ranges, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble_array daeVariable::array(TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
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
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->CreateSetupVariableArray(ranges, 3);
	}
	else
	{
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->Create_adouble_array(ranges, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble_array daeVariable::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
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
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->CreateSetupVariableArray(ranges, 4);
	}
	else
	{
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->Create_adouble_array(ranges, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble_array daeVariable::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
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
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->CreateSetupVariableArray(ranges, 5);
	}
	else
	{
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->Create_adouble_array(ranges, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble_array daeVariable::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
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
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->CreateSetupVariableArray(ranges, 6);
	}
	else
	{
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->Create_adouble_array(ranges, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble_array daeVariable::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
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
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->CreateSetupVariableArray(ranges, 7);
	}
	else
	{
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->Create_adouble_array(ranges, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble_array daeVariable::array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid Array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
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
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->CreateSetupVariableArray(ranges, 8);
	}
	else
	{
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->Create_adouble_array(ranges, 8);
	}
}


template<typename TYPE1>
adouble_array daeVariable::dt_array(TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
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
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->CreateSetupTimeDerivativeArray(ranges, 1);
	}
	else
	{
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->Calculate_dt_array(ranges, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble_array daeVariable::dt_array(TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
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
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
							   };
		return this->CreateSetupTimeDerivativeArray(ranges, 2);
	}
	else
	{
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
							   };
		return this->Calculate_dt_array(ranges, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble_array daeVariable::dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
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
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->CreateSetupTimeDerivativeArray(ranges, 3);
	}
	else
	{
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->Calculate_dt_array(ranges, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble_array daeVariable::dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
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
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->CreateSetupTimeDerivativeArray(ranges, 4);
	}
	else
	{
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->Calculate_dt_array(ranges, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble_array daeVariable::dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
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
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->CreateSetupTimeDerivativeArray(ranges, 5);
	}
	else
	{
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->Calculate_dt_array(ranges, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble_array daeVariable::dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
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
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->CreateSetupTimeDerivativeArray(ranges, 6);
	}
	else
	{
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->Calculate_dt_array(ranges, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble_array daeVariable::dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
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
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->CreateSetupTimeDerivativeArray(ranges, 7);
	}
	else
	{
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->Calculate_dt_array(ranges, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble_array daeVariable::dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid dt_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
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
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->CreateSetupTimeDerivativeArray(ranges, 8);
	}
	else
	{
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->Calculate_dt_array(ranges, 8);
	}
}


template<typename TYPE1>
adouble_array daeVariable::d_array(const daeDomain_t& rDomain, TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
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
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->CreateSetupPartialDerivativeArray(1, rDomain, ranges, 1);
	}
	else
	{
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->partial_array(1, rDomain, ranges, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble_array daeVariable::d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
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
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
							   };
		return this->CreateSetupPartialDerivativeArray(1, rDomain, ranges, 2);
	}
	else
	{
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
							   };
		return this->partial_array(1, rDomain, ranges, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble_array daeVariable::d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
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
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->CreateSetupPartialDerivativeArray(1, rDomain, ranges, 3);
	}
	else
	{
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->partial_array(1, rDomain, ranges, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble_array daeVariable::d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
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
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->CreateSetupPartialDerivativeArray(1, rDomain, ranges, 4);
	}
	else
	{
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->partial_array(1, rDomain, ranges, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble_array daeVariable::d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
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
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->CreateSetupPartialDerivativeArray(1, rDomain, ranges, 5);
	}
	else
	{
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->partial_array(1, rDomain, ranges, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble_array daeVariable::d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
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
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->CreateSetupPartialDerivativeArray(1, rDomain, ranges, 6);
	}
	else
	{
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->partial_array(1, rDomain, ranges, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble_array daeVariable::d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
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
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->CreateSetupPartialDerivativeArray(1, rDomain, ranges, 7);
	}
	else
	{
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->partial_array(1, rDomain, ranges, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble_array daeVariable::d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
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
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->CreateSetupPartialDerivativeArray(1, rDomain, ranges, 8);
	}
	else
	{
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->partial_array(1, rDomain, ranges, 8);
	}
}


template<typename TYPE1>
adouble_array daeVariable::d2_array(const daeDomain_t& rDomain, TYPE1 d1)
{
	if(m_ptrDomains.size() != 1)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 1";
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
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->CreateSetupPartialDerivativeArray(2, rDomain, ranges, 1);
	}
	else
	{
		daeArrayRange ranges[1] = {dae::core::CreateRange(d1)};
		return this->partial_array(2, rDomain, ranges, 1);
	}
}

template<typename TYPE1, typename TYPE2>
adouble_array daeVariable::d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2)
{
	if(m_ptrDomains.size() != 2)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 2";
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
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
							   };
		return this->CreateSetupPartialDerivativeArray(2, rDomain, ranges, 2);
	}
	else
	{
		daeArrayRange ranges[2] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2) 
							   };
		return this->partial_array(2, rDomain, ranges, 2);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble_array daeVariable::d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	if(m_ptrDomains.size() != 3)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 3";
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
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->CreateSetupPartialDerivativeArray(2, rDomain, ranges, 3);
	}
	else
	{
		daeArrayRange ranges[3] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3) 
							   };
		return this->partial_array(2, rDomain, ranges, 3);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble_array daeVariable::d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	if(m_ptrDomains.size() != 4)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 4";
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
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->CreateSetupPartialDerivativeArray(2, rDomain, ranges, 4);
	}
	else
	{
		daeArrayRange ranges[4] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4) 
							   };
		return this->partial_array(2, rDomain, ranges, 4);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble_array daeVariable::d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	if(m_ptrDomains.size() != 5)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 5";
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
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->CreateSetupPartialDerivativeArray(2, rDomain, ranges, 5);
	}
	else
	{
		daeArrayRange ranges[5] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5) 
							   };
		return this->partial_array(2, rDomain, ranges, 5);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble_array daeVariable::d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
{
	if(m_ptrDomains.size() != 6)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 6";
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
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->CreateSetupPartialDerivativeArray(2, rDomain, ranges, 6);
	}
	else
	{
		daeArrayRange ranges[6] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6)
							   };
		return this->partial_array(2, rDomain, ranges, 6);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble_array daeVariable::d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
{
	if(m_ptrDomains.size() != 7)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 7";
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
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->CreateSetupPartialDerivativeArray(2, rDomain, ranges, 7);
	}
	else
	{
		daeArrayRange ranges[7] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7)
							   };
		return this->partial_array(2, rDomain, ranges, 7);
	}
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble_array daeVariable::d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
{
	if(m_ptrDomains.size() != 8)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid d2_array call for [" << m_strCanonicalName << "], number of domains is " << m_ptrDomains.size() << " - given 8";
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
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->CreateSetupPartialDerivativeArray(2, rDomain, ranges, 8);
	}
	else
	{
		daeArrayRange ranges[8] = {dae::core::CreateRange(d1), 
								   dae::core::CreateRange(d2), 
								   dae::core::CreateRange(d3), 
								   dae::core::CreateRange(d4), 
								   dae::core::CreateRange(d5), 
								   dae::core::CreateRange(d6), 
								   dae::core::CreateRange(d7), 
								   dae::core::CreateRange(d8)
							   };
		return this->partial_array(2, rDomain, ranges, 8);
	}
}
