// These are compiler "cheating" functions.
// Their purpose is to allow program to compile 
// only if acceptable arguments have been sent.
// They should forbid calls to: operator(), dt(), d(), d2() 
// with any arguments but simple integers, daeDEDI* and incremented daeDEDI*
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
	return this->CreateSetupParameter((const daeDomainIndex*)NULL, 0);
}

template<typename TYPE1>
adouble	daeParameter::operator()(TYPE1 d1)
{
	daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
	return this->CreateSetupParameter(indexes, 1);
}

template<typename TYPE1, typename TYPE2>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2)
{
	daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
					             dae::core::CreateDomainIndex(d2) 
								};
	return this->CreateSetupParameter(indexes, 2);
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3) 
								};
	return this->CreateSetupParameter(indexes, 3);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4) 
								};
	return this->CreateSetupParameter(indexes, 4);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4), 
								 dae::core::CreateDomainIndex(d5) 
								};
	return this->CreateSetupParameter(indexes, 5);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeParameter::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
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

/******************************************************************
	daeVariable
*******************************************************************/
inline adouble daeVariable::operator()(void)
{
	return this->CreateSetupVariable((const daeDomainIndex*)NULL, 0);
}

template<typename TYPE1>
adouble	daeVariable::operator()(TYPE1 d1)
{
	daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
	return this->CreateSetupVariable(indexes, 1);
}

template<typename TYPE1, typename TYPE2>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2)
{
	daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2) 
								};
	return this->CreateSetupVariable(indexes, 2);
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3) 
								};
	return this->CreateSetupVariable(indexes, 3);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4) 
								};
	return this->CreateSetupVariable(indexes, 4);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4), 
								 dae::core::CreateDomainIndex(d5) 
								};
	return this->CreateSetupVariable(indexes, 5);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeVariable::operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
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


inline adouble daeVariable::dt(void)
{
	return this->CreateSetupTimeDerivative((const daeDomainIndex*)NULL, 0);
}

template<typename TYPE1>
adouble	daeVariable::dt(TYPE1 d1)
{
	daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
	return this->CreateSetupTimeDerivative(indexes, 1);
}

template<typename TYPE1, typename TYPE2>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2)
{
	daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2) 
								};
	return this->CreateSetupTimeDerivative(indexes, 2);
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3) 
								};
	return this->CreateSetupTimeDerivative(indexes, 3);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4) 
								};
	return this->CreateSetupTimeDerivative(indexes, 4);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4), 
								 dae::core::CreateDomainIndex(d5) 
								};
	return this->CreateSetupTimeDerivative(indexes, 5);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeVariable::dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
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


template<typename TYPE1>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1)
{
	daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
	return this->CreateSetupPartialDerivative(1, rDomain, indexes, 1);
}

template<typename TYPE1, typename TYPE2>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2)
{
	daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2) 
								};
	return this->CreateSetupPartialDerivative(1, rDomain, indexes, 2);
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3) 
								};
	return this->CreateSetupPartialDerivative(1, rDomain, indexes, 3);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4) 
								};
	return this->CreateSetupPartialDerivative(1, rDomain, indexes, 4);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4), 
								 dae::core::CreateDomainIndex(d5) 
								};
	return this->CreateSetupPartialDerivative(1, rDomain, indexes, 5);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeVariable::d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
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


template<typename TYPE1>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1)
{
	daeDomainIndex indexes[1] = {dae::core::CreateDomainIndex(d1)};
	return this->CreateSetupPartialDerivative(2, rDomain, indexes, 1);
}

template<typename TYPE1, typename TYPE2>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2)
{
	daeDomainIndex indexes[2] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2) 
								};
	return this->CreateSetupPartialDerivative(2, rDomain, indexes, 2);
}

template<typename TYPE1, typename TYPE2, typename TYPE3>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3)
{
	daeDomainIndex indexes[3] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3) 
								};
	return this->CreateSetupPartialDerivative(2, rDomain, indexes, 3);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4)
{
	daeDomainIndex indexes[4] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4) 
								};
	return this->CreateSetupPartialDerivative(2, rDomain, indexes, 4);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5)
{
	daeDomainIndex indexes[5] = {dae::core::CreateDomainIndex(d1), 
								 dae::core::CreateDomainIndex(d2), 
								 dae::core::CreateDomainIndex(d3), 
								 dae::core::CreateDomainIndex(d4), 
								 dae::core::CreateDomainIndex(d5) 
								};
	return this->CreateSetupPartialDerivative(2, rDomain, indexes, 5);
}

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7)
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

template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
adouble	daeVariable::d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8)
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
