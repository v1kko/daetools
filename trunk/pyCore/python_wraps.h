#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <string>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/python/call_method.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "../dae_develop.h"
#include "../Core/base_logging.h"
#include "../Core/tcpiplog.h"

namespace daepython
{
/*******************************************************
	String functions
*******************************************************/
string daeVariableType_str(const daeVariableType& self) ;
string daeDomain_str(const daeDomain& self);
string daeParameter_str(const daeParameter& self);
string daeVariable_str(const daeVariable& self);
string daePort_str(const daePort& self);
string daeModel_str(const daeModel& self);
string daeEquation_str(const daeEquation& self);
string daeDEDI_str(const daeDEDI& self);
string adouble_repr(const adouble& self);

/*******************************************************
	Common functions
*******************************************************/
daeDomainIndex CreateDomainIndex(boost::python::object& o);
daeArrayRange  CreateArrayRange(boost::python::object& o, daeDomain* pDomain);

//void daeSaveModel(daeModel& rModel, string strFileName);

boost::python::object daeGetConfig(void);
bool        GetBoolean(daeConfig& self, const std::string& strPropertyPath);
real_t      GetFloat(daeConfig& self, const std::string& strPropertyPath);
int         GetInteger(daeConfig& self, const std::string& strPropertyPath);
std::string GetString(daeConfig& self, const std::string& strPropertyPath);
bool        GetBoolean1(daeConfig& self, const std::string& strPropertyPath, const bool defValue);
real_t      GetFloat1(daeConfig& self, const std::string& strPropertyPath, const real_t defValue);
int         GetInteger1(daeConfig& self, const std::string& strPropertyPath, const int defValue);
std::string GetString1(daeConfig& self, const std::string& strPropertyPath, const std::string defValue);

void SetBoolean(daeConfig& self, const std::string& strPropertyPath, bool value);
void SetFloat(daeConfig& self, const std::string& strPropertyPath, real_t value);
void SetInteger(daeConfig& self, const std::string& strPropertyPath, int value);
void SetString(daeConfig& self, const std::string& strPropertyPath, std::string value);

std::string daeConfig__str__(daeConfig& self);
boost::python::object daeConfig__contains__(daeConfig& self, boost::python::object key);
boost::python::object daeConfig_has_key(daeConfig& self, boost::python::object key);
boost::python::object daeConfig__getitem__(daeConfig& self, boost::python::object key);
void                  daeConfig__setitem__(daeConfig& self, boost::python::object key, boost::python::object value);

/*******************************************************
	adouble
*******************************************************/
const adouble ad_exp(const adouble &a);
const adouble ad_log(const adouble &a);
const adouble ad_sqrt(const adouble &a);
const adouble ad_sin(const adouble &a);
const adouble ad_cos(const adouble &a);
const adouble ad_tan(const adouble &a);
const adouble ad_asin(const adouble &a);
const adouble ad_acos(const adouble &a);
const adouble ad_atan(const adouble &a);

const adouble ad_sinh(const adouble &a);
const adouble ad_cosh(const adouble &a);
const adouble ad_tanh(const adouble &a);
const adouble ad_asinh(const adouble &a);
const adouble ad_acosh(const adouble &a);
const adouble ad_atanh(const adouble &a);
const adouble ad_atan2(const adouble &a, const adouble &b);

const adouble ad_pow1(const adouble &a, real_t v);
const adouble ad_pow2(const adouble &a, const adouble &b);
const adouble ad_pow3(real_t v, const adouble &a);
const adouble ad_log10(const adouble &a);

const adouble ad_abs(const adouble &a);
const adouble ad_ceil(const adouble &a);
const adouble ad_floor(const adouble &a);
const adouble ad_max1(const adouble &a, const adouble &b);
const adouble ad_max2(real_t v, const adouble &a);
const adouble ad_max3(const adouble &a, real_t v);
const adouble ad_min1(const adouble &a, const adouble &b);
const adouble ad_min2(real_t v, const adouble &a);
const adouble ad_min3(const adouble &a, real_t v);

const adouble ad_Constant_q(const quantity& q);
const adouble ad_Constant_c(real_t c);
const adouble_array adarr_Array(boost::python::list Values);

/*******************************************************
	adouble_array
*******************************************************/
const adouble_array adarr_exp(const adouble_array& a);
const adouble_array adarr_sqrt(const adouble_array& a);
const adouble_array adarr_log(const adouble_array& a);
const adouble_array adarr_log10(const adouble_array& a);
const adouble_array adarr_abs(const adouble_array& a);
const adouble_array adarr_floor(const adouble_array& a);
const adouble_array adarr_ceil(const adouble_array& a);
const adouble_array adarr_sin(const adouble_array& a);
const adouble_array adarr_cos(const adouble_array& a);
const adouble_array adarr_tan(const adouble_array& a);
const adouble_array adarr_asin(const adouble_array& a);
const adouble_array adarr_acos(const adouble_array& a);
const adouble_array adarr_atan(const adouble_array& a);

/*******************************************************
	daeObject
*******************************************************/
string daeGetRelativeName_1(const daeObject* parent, const daeObject* child);
string daeGetRelativeName_2(const string& strParent, const string& strChild);

/*******************************************************
	daeDomain
*******************************************************/
boost::python::numeric::array GetNumPyArrayDomain(daeDomain& domain);
adouble_array DomainArray1(daeDomain& domain);
adouble_array DomainArray2(daeDomain& domain, boost::python::slice s);
daeIndexRange FunctionCallDomain1(daeDomain& domain, int start, int end, int step);
daeIndexRange FunctionCallDomain2(daeDomain& domain, boost::python::list l);
daeIndexRange FunctionCallDomain3(daeDomain& domain);
boost::python::list GetDomainPoints(daeDomain& domain);
void SetDomainPoints(daeDomain& domain, boost::python::list l);

daeIndexRange* __init__daeIndexRange(daeDomain* pDomain, boost::python::list CustomPoints);

/*******************************************************
	daeParameter
*******************************************************/
class daeParameterWrapper : public daeParameter,
	                        public boost::python::wrapper<daeParameter>
{
public:
	daeParameterWrapper(void)
	{
	}

	daeParameterWrapper(string strName, const unit& units, daeModel* pModel, string strDescription = "", boost::python::list domains = boost::python::list())
		: daeParameter(strName, units, pModel, strDescription)
	{
		daeDomain* pDomain;
		boost::python::ssize_t n = boost::python::len(domains);
		m_ptrDomains.resize(n);
		for(boost::python::ssize_t i = 0; i < n; i++)
		{
			pDomain = boost::python::extract<daeDomain*>(domains[i]);
			m_ptrDomains[i] = pDomain;
		}
	}

	daeParameterWrapper(string strName, const unit& units, daePort* pPort, string strDescription = "", boost::python::list domains = boost::python::list())
		: daeParameter(strName, units, pPort, strDescription)
	{
		daeDomain* pDomain;
		boost::python::ssize_t n = boost::python::len(domains);
		m_ptrDomains.resize(n);
		for(boost::python::ssize_t i = 0; i < n; i++)
		{
			pDomain = boost::python::extract<daeDomain*>(domains[i]);
			m_ptrDomains[i] = pDomain;
		}
	}

public:
	boost::python::list GetDomains(void)
	{
		boost::python::list l;
		daeDomain* obj;

		for(size_t i = 0; i < m_ptrDomains.size(); i++)
		{
			obj = m_ptrDomains[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	real_t GetParameterValue0()
	{
		return GetValue();
	}

	real_t GetParameterValue1(size_t n1)
	{
		return GetValue(n1);
	}

	real_t GetParameterValue2(size_t n1, size_t n2)
	{
		return GetValue(n1, n2);
	}

	real_t GetParameterValue3(size_t n1, size_t n2, size_t n3)
	{
		return GetValue(n1, n2, n3);
	}

	real_t GetParameterValue4(size_t n1, size_t n2, size_t n3, size_t n4)
	{
		return GetValue(n1, n2, n3, n4);
	}

	real_t GetParameterValue5(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
	{
		return GetValue(n1, n2, n3, n4, n5);
	}

	real_t GetParameterValue6(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
	{
		return GetValue(n1, n2, n3, n4, n5, n6);
	}

	real_t GetParameterValue7(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
	{
		return GetValue(n1, n2, n3, n4, n5, n6, n7);
	}

	real_t GetParameterValue8(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
	{
		return GetValue(n1, n2, n3, n4, n5, n6, n7, n8);
	}

	quantity GetParameterQuantity0()
	{
		return GetQuantity();
	}

	quantity GetParameterQuantity1(size_t n1)
	{
		return GetQuantity(n1);
	}

	quantity GetParameterQuantity2(size_t n1, size_t n2)
	{
		return GetQuantity(n1, n2);
	}

	quantity GetParameterQuantity3(size_t n1, size_t n2, size_t n3)
	{
		return GetQuantity(n1, n2, n3);
	}

	quantity GetParameterQuantity4(size_t n1, size_t n2, size_t n3, size_t n4)
	{
		return GetQuantity(n1, n2, n3, n4);
	}

	quantity GetParameterQuantity5(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
	{
		return GetQuantity(n1, n2, n3, n4, n5);
	}

	quantity GetParameterQuantity6(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
	{
		return GetQuantity(n1, n2, n3, n4, n5, n6);
	}

	quantity GetParameterQuantity7(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
	{
		return GetQuantity(n1, n2, n3, n4, n5, n6, n7);
	}

	quantity GetParameterQuantity8(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
	{
		return GetQuantity(n1, n2, n3, n4, n5, n6, n7, n8);
	}
};

boost::python::numeric::array GetNumPyArrayParameter(daeParameter& param);

adouble FunctionCallParameter0(daeParameter& param);
adouble FunctionCallParameter1(daeParameter& param, boost::python::object o1);
adouble FunctionCallParameter2(daeParameter& param, boost::python::object o1, boost::python::object o2);
adouble FunctionCallParameter3(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble FunctionCallParameter4(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble FunctionCallParameter5(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble FunctionCallParameter6(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble FunctionCallParameter7(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble FunctionCallParameter8(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

void SetParameterValue0(daeParameter& param, real_t value);
void SetParameterValue1(daeParameter& param, size_t n1, real_t value);
void SetParameterValue2(daeParameter& param, size_t n1, size_t n2, real_t value);
void SetParameterValue3(daeParameter& param, size_t n1, size_t n2, size_t n3, real_t value);
void SetParameterValue4(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, real_t value);
void SetParameterValue5(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value);
void SetParameterValue6(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value);
void SetParameterValue7(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value);
void SetParameterValue8(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value);

void SetParameterQuantity0(daeParameter& param, const quantity& q);
void SetParameterQuantity1(daeParameter& param, size_t n1, const quantity& q);
void SetParameterQuantity2(daeParameter& param, size_t n1, size_t n2, const quantity& q);
void SetParameterQuantity3(daeParameter& param, size_t n1, size_t n2, size_t n3, const quantity& q);
void SetParameterQuantity4(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& q);
void SetParameterQuantity5(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& q);
void SetParameterQuantity6(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& q);
void SetParameterQuantity7(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& q);
void SetParameterQuantity8(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& q);

void SetParameterValues(daeParameter& param, real_t values);
void qSetParameterValues(daeParameter& param, const quantity& q);

adouble_array ParameterArray1(daeParameter& param, boost::python::object o1);
adouble_array ParameterArray2(daeParameter& param, boost::python::object o1, boost::python::object o2);
adouble_array ParameterArray3(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble_array ParameterArray4(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble_array ParameterArray5(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble_array ParameterArray6(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble_array ParameterArray7(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble_array ParameterArray8(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

/*******************************************************
	daeVariable_Wrapper
*******************************************************/
class daeVariable_Wrapper : public daeVariable,
	                        public boost::python::wrapper<daeVariable>
{
public:
	daeVariable_Wrapper(void)
	{
	}

	daeVariable_Wrapper(string strName, const daeVariableType& varType, daeModel* pModel, string strDescription = "", boost::python::list domains = boost::python::list())
		: daeVariable(strName, varType, pModel, strDescription)
	{
		daeDomain* pDomain;
		boost::python::ssize_t n = boost::python::len(domains);
		m_ptrDomains.resize(n);
		for(boost::python::ssize_t i = 0; i < n; i++)
		{
			pDomain = boost::python::extract<daeDomain*>(domains[i]);
			m_ptrDomains[i] = pDomain;
		}
	}

	daeVariable_Wrapper(string strName, const daeVariableType& varType, daePort* pPort, string strDescription = "", boost::python::list domains = boost::python::list())
		: daeVariable(strName, varType, pPort, strDescription)
	{
		daeDomain* pDomain;
		boost::python::ssize_t n = boost::python::len(domains);
		m_ptrDomains.resize(n);
		for(boost::python::ssize_t i = 0; i < n; i++)
		{
			pDomain = boost::python::extract<daeDomain*>(domains[i]);
			m_ptrDomains[i] = pDomain;
		}
	}

public:
	boost::python::list GetDomains(void)
	{
		boost::python::list l;
		daeDomain* obj;

		for(size_t i = 0; i < m_ptrDomains.size(); i++)
		{
			obj = m_ptrDomains[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	daeVariableType* GetVariableType(void)
	{
		return &this->m_VariableType;
	}

	real_t GetVariableValue0(void)
	{
		return GetValue();
	}

	real_t GetVariableValue1(size_t n1)
	{
		return GetValue(n1);
	}

	real_t GetVariableValue2(size_t n1, size_t n2)
	{
		return GetValue(n1, n2);
	}

	real_t GetVariableValue3(size_t n1, size_t n2, size_t n3)
	{
		return GetValue(n1, n2, n3);
	}

	real_t GetVariableValue4(size_t n1, size_t n2, size_t n3, size_t n4)
	{
		return GetValue(n1, n2, n3, n4);
	}

	real_t GetVariableValue5(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
	{
		return GetValue(n1, n2, n3, n4, n5);
	}

	real_t GetVariableValue6(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
	{
		return GetValue(n1, n2, n3, n4, n5, n6);
	}

	real_t GetVariableValue7(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
	{
		return GetValue(n1, n2, n3, n4, n5, n6, n7);
	}

	real_t GetVariableValue8(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
	{
		return GetValue(n1, n2, n3, n4, n5, n6, n7, n8);
	}

	quantity GetVariableQuantity0()
	{
		return GetQuantity();
	}

	quantity GetVariableQuantity1(size_t n1)
	{
		return GetQuantity(n1);
	}

	quantity GetVariableQuantity2(size_t n1, size_t n2)
	{
		return GetQuantity(n1, n2);
	}

	quantity GetVariableQuantity3(size_t n1, size_t n2, size_t n3)
	{
		return GetQuantity(n1, n2, n3);
	}

	quantity GetVariableQuantity4(size_t n1, size_t n2, size_t n3, size_t n4)
	{
		return GetQuantity(n1, n2, n3, n4);
	}

	quantity GetVariableQuantity5(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
	{
		return GetQuantity(n1, n2, n3, n4, n5);
	}

	quantity GetVariableQuantity6(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
	{
		return GetQuantity(n1, n2, n3, n4, n5, n6);
	}

	quantity GetVariableQuantity7(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
	{
		return GetQuantity(n1, n2, n3, n4, n5, n6, n7);
	}

	quantity GetVariableQuantity8(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
	{
		return GetQuantity(n1, n2, n3, n4, n5, n6, n7, n8);
	}
};

boost::python::numeric::array GetNumPyArrayVariable(daeVariable& var);

adouble VariableFunctionCall0(daeVariable& var);
adouble VariableFunctionCall1(daeVariable& var, boost::python::object o1);
adouble VariableFunctionCall2(daeVariable& var, boost::python::object o1, boost::python::object o2);
adouble VariableFunctionCall3(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble VariableFunctionCall4(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble VariableFunctionCall5(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble VariableFunctionCall6(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble VariableFunctionCall7(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble VariableFunctionCall8(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

adouble Get_dt0(daeVariable& var);
adouble Get_dt1(daeVariable& var, boost::python::object o1);
adouble Get_dt2(daeVariable& var, boost::python::object o1, boost::python::object o2);
adouble Get_dt3(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble Get_dt4(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble Get_dt5(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble Get_dt6(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble Get_dt7(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble Get_dt8(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

adouble Get_d1(daeVariable& var, daeDomain& d, boost::python::object o1);
adouble Get_d2(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2);
adouble Get_d3(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble Get_d4(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble Get_d5(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble Get_d6(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble Get_d7(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble Get_d8(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

adouble Get_d21(daeVariable& var, daeDomain& d, boost::python::object o1);
adouble Get_d22(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2);
adouble Get_d23(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble Get_d24(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble Get_d25(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble Get_d26(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble Get_d27(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble Get_d28(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

adouble_array VariableArray1(daeVariable& var, boost::python::object o1);
adouble_array VariableArray2(daeVariable& var, boost::python::object o1, boost::python::object o2);
adouble_array VariableArray3(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble_array VariableArray4(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble_array VariableArray5(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble_array VariableArray6(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble_array VariableArray7(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble_array VariableArray8(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

adouble_array Get_dt_array1(daeVariable& var, boost::python::object o1);
adouble_array Get_dt_array2(daeVariable& var, boost::python::object o1, boost::python::object o2);
adouble_array Get_dt_array3(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble_array Get_dt_array4(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble_array Get_dt_array5(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble_array Get_dt_array6(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble_array Get_dt_array7(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble_array Get_dt_array8(daeVariable& var, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

adouble_array Get_d_array1(daeVariable& var, daeDomain& d, boost::python::object o1);
adouble_array Get_d_array2(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2);
adouble_array Get_d_array3(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble_array Get_d_array4(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble_array Get_d_array5(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble_array Get_d_array6(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble_array Get_d_array7(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble_array Get_d_array8(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

adouble_array Get_d2_array1(daeVariable& var, daeDomain& d, boost::python::object o1);
adouble_array Get_d2_array2(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2);
adouble_array Get_d2_array3(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble_array Get_d2_array4(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble_array Get_d2_array5(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble_array Get_d2_array6(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble_array Get_d2_array7(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble_array Get_d2_array8(daeVariable& var, daeDomain& d, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

void SetVariableValue0(daeVariable& var, real_t value);
void SetVariableValue1(daeVariable& var, size_t n1, real_t value);
void SetVariableValue2(daeVariable& var, size_t n1, size_t n2, real_t value);
void SetVariableValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value);
void SetVariableValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value);
void SetVariableValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value);
void SetVariableValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value);
void SetVariableValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value);
void SetVariableValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value);

void AssignValue0(daeVariable& var, real_t value);
void AssignValue1(daeVariable& var, size_t n1, real_t value);
void AssignValue2(daeVariable& var, size_t n1, size_t n2, real_t value);
void AssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value);
void AssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value);
void AssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value);
void AssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value);
void AssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value);
void AssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value);

void ReAssignValue0(daeVariable& var, real_t value);
void ReAssignValue1(daeVariable& var, size_t n1, real_t value);
void ReAssignValue2(daeVariable& var, size_t n1, size_t n2, real_t value);
void ReAssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value);
void ReAssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value);
void ReAssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value);
void ReAssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value);
void ReAssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value);
void ReAssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value);

void SetInitialCondition0(daeVariable& var, real_t value);
void SetInitialCondition1(daeVariable& var, size_t n1, real_t value);
void SetInitialCondition2(daeVariable& var, size_t n1, size_t n2, real_t value);
void SetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value);
void SetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value);
void SetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value);
void SetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value);
void SetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value);
void SetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value);

void ReSetInitialCondition0(daeVariable& var, real_t value);
void ReSetInitialCondition1(daeVariable& var, size_t n1, real_t value);
void ReSetInitialCondition2(daeVariable& var, size_t n1, size_t n2, real_t value);
void ReSetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value);
void ReSetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value);
void ReSetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value);
void ReSetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value);
void ReSetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value);
void ReSetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value);

void SetInitialGuess0(daeVariable& var, real_t value);
void SetInitialGuess1(daeVariable& var, size_t n1, real_t value);
void SetInitialGuess2(daeVariable& var, size_t n1, size_t n2, real_t value);
void SetInitialGuess3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value);
void SetInitialGuess4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value);
void SetInitialGuess5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value);
void SetInitialGuess6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value);
void SetInitialGuess7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value);
void SetInitialGuess8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value);

void qSetVariableValue0(daeVariable& var, const quantity& q);
void qSetVariableValue1(daeVariable& var, size_t n1, const quantity& q);
void qSetVariableValue2(daeVariable& var, size_t n1, size_t n2, const quantity& q);
void qSetVariableValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& q);
void qSetVariableValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& q);
void qSetVariableValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& q);
void qSetVariableValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& q);
void qSetVariableValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& q);
void qSetVariableValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& q);

void qAssignValue0(daeVariable& var, const quantity& value);
void qAssignValue1(daeVariable& var, size_t n1, const quantity& value);
void qAssignValue2(daeVariable& var, size_t n1, size_t n2, const quantity& value);
void qAssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& value);
void qAssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& value);
void qAssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& value);
void qAssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& value);
void qAssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& value);
void qAssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& value);

void qReAssignValue0(daeVariable& var, const quantity& value);
void qReAssignValue1(daeVariable& var, size_t n1, const quantity& value);
void qReAssignValue2(daeVariable& var, size_t n1, size_t n2, const quantity& value);
void qReAssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& value);
void qReAssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& value);
void qReAssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& value);
void qReAssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& value);
void qReAssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& value);
void qReAssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& value);

void qSetInitialCondition0(daeVariable& var, const quantity& q);
void qSetInitialCondition1(daeVariable& var, size_t n1, const quantity& q);
void qSetInitialCondition2(daeVariable& var, size_t n1, size_t n2, const quantity& q);
void qSetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& q);
void qSetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& q);
void qSetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& q);
void qSetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& q);
void qSetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& q);
void qSetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& q);

void qReSetInitialCondition0(daeVariable& var, const quantity& q);
void qReSetInitialCondition1(daeVariable& var, size_t n1, const quantity& q);
void qReSetInitialCondition2(daeVariable& var, size_t n1, size_t n2, const quantity& q);
void qReSetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& q);
void qReSetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& q);
void qReSetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& q);
void qReSetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& q);
void qReSetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& q);
void qReSetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& q);

void qSetInitialGuess0(daeVariable& var, const quantity& q);
void qSetInitialGuess1(daeVariable& var, size_t n1, const quantity& q);
void qSetInitialGuess2(daeVariable& var, size_t n1, size_t n2, const quantity& q);
void qSetInitialGuess3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& q);
void qSetInitialGuess4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& q);
void qSetInitialGuess5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& q);
void qSetInitialGuess6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& q);
void qSetInitialGuess7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& q);
void qSetInitialGuess8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& q);

void AssignValues(daeVariable& var, real_t values);
void qAssignValues(daeVariable& var, const quantity& q);

void ReAssignValues(daeVariable& var, real_t values);
void qReAssignValues(daeVariable& var, const quantity& q);

void SetInitialConditions(daeVariable& var, real_t values);
void qSetInitialConditions(daeVariable& var, const quantity& q);

void ReSetInitialConditions(daeVariable& var, real_t values);
void qReSetInitialConditions(daeVariable& var, const quantity& q);

void SetInitialGuesses(daeVariable& var, real_t values);
void qSetInitialGuesses(daeVariable& var, const quantity& q);

/*******************************************************
	daeActionWrapper
*******************************************************/
class daeActionWrapper : public daeAction,
                         public boost::python::wrapper<daeAction>
{
public:
	daeActionWrapper(void)
	{
		m_eActionType = eUserDefinedAction;
	}

	void Execute(void)
	{
        this->get_override("Execute")();
	}
};


/*******************************************************
	daePort
*******************************************************/
class daePortWrapper : public daePort,
	                   public boost::python::wrapper<daePort>
{
public:
	daePortWrapper(void)
	{
	}

	daePortWrapper(string strName, daeePortType eType, daeModel* parent, string strDescription = "")
		: daePort(strName, eType, parent, strDescription)
	{
	}

public:
	boost::python::list GetDomains(void)
	{
		boost::python::list l;
		daeDomain* obj;

		for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
		{
			obj = m_ptrarrDomains[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetParameters(void)
	{
		boost::python::list l;
		daeParameter* obj;

		for(size_t i = 0; i < m_ptrarrParameters.size(); i++)
		{
			obj = m_ptrarrParameters[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetVariables(void)
	{
		boost::python::list l;
		daeVariable* obj;

		for(size_t i = 0; i < m_ptrarrVariables.size(); i++)
		{
            //obj = dynamic_cast<daeVariable_Wrapper*>(m_ptrarrVariables[i]);
			obj = m_ptrarrVariables[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	std::string GetObjectClassName(void) const
	{
		boost::python::reference_existing_object::apply<const daePort*>::type converter;
		PyObject* pyobj = converter( this );
		boost::python::object obj = boost::python::object( boost::python::handle<>( pyobj ) );
		boost::python::object o_class = obj.attr("__class__");
		string name = boost::python::extract<string>(o_class.attr("__name__"));
		return name;
	}
};

/*******************************************************
	daeEventPort
*******************************************************/
boost::python::list GetEventPortEventsList(daeEventPort& self);

/*******************************************************
	daeEquation
*******************************************************/
//daeEquation* __init__daeEquation(const string& strName, daeModel& model);
daeDEDI* DistributeOnDomain1(daeEquation& eq, daeDomain& rDomain, daeeDomainBounds eDomainBounds);
daeDEDI* DistributeOnDomain2(daeEquation& eq, daeDomain& rDomain, boost::python::list l);

/*******************************************************
	daeModel
*******************************************************/
class daeModelWrapper : public daeModel,
	                    public boost::python::wrapper<daeModel>
{
public:
	daeModelWrapper(void)
	{
	}

	daeModelWrapper(string strName, daeModel* pModel = NULL, string strDescription = "") : daeModel(strName, pModel, strDescription)
	{
	}

	void IF(const daeCondition& rCondition, real_t dEventTolerance = 0.0)
	{
		daeModel::IF(rCondition, dEventTolerance);
	}

	void ELSE_IF(const daeCondition& rCondition, real_t dEventTolerance = 0.0)
	{
		daeModel::ELSE_IF(rCondition, dEventTolerance);
	}

	void ELSE(void)
	{
		daeModel::ELSE();
	}

	void END_IF(void)
	{
		daeModel::END_IF();
	}

	daeSTN* STN(const string& strSTN)
	{
		return daeModel::STN(strSTN);
	}

	daeState* STATE(const string& strState)
	{
		return daeModel::STATE(strState);
	}

	void END_STN(void)
	{
		daeModel::END_STN();
	}

	void SWITCH_TO(const string& strState, const daeCondition& rCondition, real_t dEventTolerance = 0.0)
	{
		daeModel::SWITCH_TO(strState, rCondition, dEventTolerance);
	}

    void ON_CONDITION(const daeCondition& rCondition,
                      const string& strStateTo               = string(),
                      boost::python::list setVariableValues  = boost::python::list(),
	                  boost::python::list triggerEvents      = boost::python::list(),
					  boost::python::list userDefinedActions = boost::python::list(),
                      real_t dEventTolerance = 0.0)
    {
		daeAction* pAction;
        daeEventPort* pEventPort;
        vector< pair<daeVariableWrapper, adouble> > arrSetVariables;
        vector< pair<daeEventPort*, adouble> > arrTriggerEvents;
		vector<daeAction*> ptrarrUserDefinedActions;
        boost::python::ssize_t i, n;
        boost::python::tuple t;

        n = boost::python::len(setVariableValues);
        for(i = 0; i < n; i++)
        {
            t = boost::python::extract<boost::python::tuple>(setVariableValues[i]);
            if(boost::python::len(t) != 2)
                daeDeclareAndThrowException(exInvalidCall);

            boost::python::object var = boost::python::extract<boost::python::object>(t[0]);
            boost::python::object o   = boost::python::extract<boost::python::object>(t[1]);
			
			boost::python::extract<daeVariable*> pvar(var);
			boost::python::extract<adouble>      avar(var);
			
			boost::python::extract<real_t>  dValue(o);
			boost::python::extract<adouble> aValue(o);
            
            pair<daeVariableWrapper, adouble> p;

			if(pvar.check())
			{
				p.first = daeVariableWrapper(*pvar());
			}
			else if(avar.check())
			{
				adouble a = avar();
				p.first = daeVariableWrapper(a);
			}
			else
			{
				daeDeclareException(exInvalidCall);
				e << "Invalid setVariableValues argument in ON_CONDITION function";
				throw e;
			}

			if(aValue.check())
			{
				p.second = aValue();
			}
			else if(dValue.check())
			{
				p.second = adouble(dValue());
			}
			else
			{
				daeDeclareException(exInvalidCall);
				e << "Invalid setVariableValues argument in ON_CONDITION function";
				throw e;
			}

			arrSetVariables.push_back(p);
        }

        n = boost::python::len(triggerEvents);
        for(i = 0; i < n; i++)
        {
            t = boost::python::extract<boost::python::tuple>(triggerEvents[i]);
            if(boost::python::len(t) != 2)
                daeDeclareAndThrowException(exInvalidCall);

            pEventPort              = boost::python::extract<daeEventPort*>(t[0]);
            boost::python::object o = boost::python::extract<boost::python::object>(t[1]);
			
			boost::python::extract<real_t>  dValue(o);
			boost::python::extract<adouble> aValue(o);

            pair<daeEventPort*, adouble> p;
			
			p.first = pEventPort;
			if(aValue.check())
			{
				p.second = aValue();
			}
			else if(dValue.check())
			{
				p.second = adouble(dValue());
			}
			else
			{
				daeDeclareException(exInvalidCall);
				e << "Invalid trigger events argument in ON_CONDITION function";
				throw e;
			}

            arrTriggerEvents.push_back(p);
        }

        n = boost::python::len(userDefinedActions);
        for(i = 0; i < n; i++)
        {
            pAction = boost::python::extract<daeAction*>(userDefinedActions[i]);
			if(!pAction)
				daeDeclareAndThrowException(exInvalidPointer);
			
            ptrarrUserDefinedActions.push_back(pAction);
		}

        daeModel::ON_CONDITION(rCondition,
                               strStateTo,
                               arrSetVariables,
                               arrTriggerEvents,
							   ptrarrUserDefinedActions,
                               dEventTolerance);
    }

    void ON_EVENT(daeEventPort* pTriggerEventPort,
                  boost::python::list switchToStates     = boost::python::list(),
	              boost::python::list setVariableValues  = boost::python::list(),
                  boost::python::list triggerEvents      = boost::python::list(),
				  boost::python::list userDefinedActions = boost::python::list())
    {
		daeAction* pAction;
        daeEventPort* pEventPort;
        string strSTN;
        string strStateTo;
        vector< pair<string, string> > arrSwitchToStates;
        vector< pair<daeVariableWrapper, adouble> > arrSetVariables;
        vector< pair<daeEventPort*, adouble> > arrTriggerEvents;
		vector<daeAction*> ptrarrUserDefinedActions;
        boost::python::ssize_t i, n;
        boost::python::tuple t;

        n = boost::python::len(switchToStates);
        for(i = 0; i < n; i++)
        {
            t = boost::python::extract<boost::python::tuple>(switchToStates[i]);
            if(boost::python::len(t) != 2)
                daeDeclareAndThrowException(exInvalidCall);

            strSTN     = boost::python::extract<string>(t[0]);
            strStateTo = boost::python::extract<string>(t[1]);
            
			pair<string, string> p(strSTN, strStateTo);
            arrSwitchToStates.push_back(p);
        }

        n = boost::python::len(setVariableValues);
        for(i = 0; i < n; i++)
        {
            t = boost::python::extract<boost::python::tuple>(setVariableValues[i]);
            if(boost::python::len(t) != 2)
                daeDeclareAndThrowException(exInvalidCall);

			boost::python::object var = boost::python::extract<boost::python::object>(t[0]);
            boost::python::object o   = boost::python::extract<boost::python::object>(t[1]);
			
			boost::python::extract<daeVariable*> pvar(var);
			boost::python::extract<adouble>      avar(var);

			boost::python::extract<real_t>  dValue(o);
			boost::python::extract<adouble> aValue(o);
            
			pair<daeVariableWrapper, adouble> p;

			if(pvar.check())
			{
				p.first = daeVariableWrapper(*pvar());
			}
			else if(avar.check())
			{
				adouble a = avar();
				p.first = daeVariableWrapper(a);
			}
			else
			{
				daeDeclareException(exInvalidCall);
				e << "Invalid setVariableValues argument in ON_CONDITION function";
				throw e;
			}

			if(aValue.check())
			{
				p.second = aValue();
			}
			else if(dValue.check())
			{
				p.second = adouble(dValue());
			}
			else
			{
				daeDeclareException(exInvalidCall);
				e << "Invalid setVariableValues argument in ON_EVENT function";
				throw e;
			}

            arrSetVariables.push_back(p);
        }

        n = boost::python::len(triggerEvents);
        for(i = 0; i < n; i++)
        {
            t = boost::python::extract<boost::python::tuple>(triggerEvents[i]);
            if(boost::python::len(t) != 2)
                daeDeclareAndThrowException(exInvalidCall);

            pEventPort              = boost::python::extract<daeEventPort*>(t[0]);
            boost::python::object o = boost::python::extract<boost::python::object>(t[1]);
			
			boost::python::extract<real_t>  dValue(o);
			boost::python::extract<adouble> aValue(o);

            pair<daeEventPort*, adouble> p;
			
			p.first = pEventPort;
			if(aValue.check())
			{
				p.second = aValue();
			}
			else if(dValue.check())
			{
				p.second = adouble(dValue());
			}
			else
			{
				daeDeclareException(exInvalidCall);
				e << "Invalid triggerEvents argument in ON_EVENT function";
				throw e;
			}

            arrTriggerEvents.push_back(p);
        }

        n = boost::python::len(userDefinedActions);
        for(i = 0; i < n; i++)
        {
            pAction = boost::python::extract<daeAction*>(userDefinedActions[i]);
			if(!pAction)
				daeDeclareAndThrowException(exInvalidPointer);
			
            ptrarrUserDefinedActions.push_back(pAction);
		} 

        daeModel::ON_EVENT(pTriggerEventPort,
                           arrSwitchToStates,
                           arrSetVariables,
                           arrTriggerEvents,
						   ptrarrUserDefinedActions);
    }
	
	daeEquation* CreateEquation1(string strName, string strDescription, real_t dScaling)
	{
		return daeModel::CreateEquation(strName, strDescription, dScaling);
	}

	daeEquation* CreateEquation2(string strName, string strDescription)
	{
		return daeModel::CreateEquation(strName, strDescription);
	}

	daeEquation* CreateEquation3(string strName)
	{
		return daeModel::CreateEquation(strName);
	}

	void DeclareData()
	{
		// Must be declared since I dont need it in python
	}

	void DeclareEquations(void)
	{
        if(boost::python::override f = this->get_override("DeclareEquations"))
            f();
		else
			this->daeModel::DeclareEquations();
	}
	void def_DeclareEquations(void)
	{
        this->daeModel::DeclareEquations();
	}

	boost::python::list GetDomains(void)
	{
		boost::python::list l;
		daeDomain* obj;

		for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
		{
			obj = m_ptrarrDomains[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetParameters(void)
	{
		boost::python::list l;
		daeParameter* obj;

		for(size_t i = 0; i < m_ptrarrParameters.size(); i++)
		{
			obj = m_ptrarrParameters[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetVariables(void)
	{
		boost::python::list l;
		daeVariable* obj;

		for(size_t i = 0; i < m_ptrarrVariables.size(); i++)
		{
			obj = m_ptrarrVariables[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetPorts(void)
	{
		boost::python::list l;
		daePort* obj;

		for(size_t i = 0; i < m_ptrarrPorts.size(); i++)
		{
			obj = m_ptrarrPorts[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetEventPorts(void)
	{
		boost::python::list l;
		daeEventPort* obj;

		for(size_t i = 0; i < m_ptrarrEventPorts.size(); i++)
		{
			obj = m_ptrarrEventPorts[i];
			l.append(boost::ref(obj));
		}
		return l;
	}
	
	boost::python::list GetOnEventActions(void)
	{
		boost::python::list l;
		daeOnEventActions* obj;

		for(size_t i = 0; i < m_ptrarrOnEventActions.size(); i++)
		{
			obj = m_ptrarrOnEventActions[i];
			l.append(boost::ref(obj));
		}
		return l;
	}
	
	boost::python::list GetPortArrays(void)
	{
		boost::python::list l;
		daePortArray* obj;

		for(size_t i = 0; i < m_ptrarrPortArrays.size(); i++)
		{
			obj = m_ptrarrPortArrays[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetChildModels(void)
	{
		boost::python::list l;
		daeModel* obj;

		for(size_t i = 0; i < m_ptrarrModels.size(); i++)
		{
			obj = m_ptrarrModels[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetChildModelArrays(void)
	{
		boost::python::list l;
		daeModelArray* obj;

		for(size_t i = 0; i < m_ptrarrModelArrays.size(); i++)
		{
			obj = m_ptrarrModelArrays[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetSTNs(void)
	{
		boost::python::list l;
		daeSTN* obj;

		for(size_t i = 0; i < m_ptrarrSTNs.size(); i++)
		{
			obj = m_ptrarrSTNs[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	string ExportObjects(boost::python::list objects, daeeModelLanguage eLanguage) const
	{
		daeObject* pObject;
		daeExportable_t* pExportable;
		std::vector<daeExportable_t*> ptrarrObjects;
		boost::python::ssize_t n = boost::python::len(objects);
		for(boost::python::ssize_t i = 0; i < n; i++)
		{
			pObject = boost::python::extract<daeObject*>(objects[i]);
			pExportable = dynamic_cast<daeExportable_t*>(pObject);
			ptrarrObjects.push_back(pExportable);
		}

		return daeModel::ExportObjects(ptrarrObjects, eLanguage);
	}

	std::string GetObjectClassName(void) const
	{
		boost::python::reference_existing_object::apply<const daeModel*>::type converter;
		PyObject* pyobj = converter( this );
		boost::python::object obj = boost::python::object( boost::python::handle<>( pyobj ) );
		boost::python::object o_class = obj.attr("__class__");
		string name = boost::python::extract<string>(o_class.attr("__name__"));
		return name;
	}

};

/*******************************************************
	daeState
*******************************************************/
class daeStateWrapper : public daeState,
                        public boost::python::wrapper<daeState>
{
public:
	daeStateWrapper(void)
	{
	}

public:
	boost::python::list GetEquations(void)
	{
		boost::python::list l;
		daeEquation* obj;

		for(size_t i = 0; i < m_ptrarrEquations.size(); i++)
		{
			obj = m_ptrarrEquations[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetStateTransitions(void)
	{
		boost::python::list l;
		daeStateTransition* obj;

		for(size_t i = 0; i < m_ptrarrStateTransitions.size(); i++)
		{
			obj = m_ptrarrStateTransitions[i];
			l.append(boost::ref(obj));
		}
		return l;
	}

	boost::python::list GetNestedSTNs(void)
	{
		boost::python::list l;
		daeSTN* obj;

		for(size_t i = 0; i < m_ptrarrSTNs.size(); i++)
		{
			obj = m_ptrarrSTNs[i];
			l.append(boost::ref(obj));
		}
		return l;
	}
};

/*******************************************************
	daeSTN
*******************************************************/
boost::python::list GetStatesSTN(daeSTN& stn);

/*******************************************************
	daeIF
*******************************************************/
class daeIFWrapper : public daeIF,
                     public boost::python::wrapper<daeIF>
{
public:
	daeIFWrapper(void)
	{
	}
};

/*******************************************************
	daeScalarExternalFunctionWrapper
*******************************************************/
class daeScalarExternalFunctionWrapper : public daeScalarExternalFunction,
                                         public boost::python::wrapper<daeScalarExternalFunction>
{
public:
	daeScalarExternalFunctionWrapper(const string& strName, daeModel* pModel, const unit& units, boost::python::dict arguments)
	    : daeScalarExternalFunction(strName, pModel, units)
	{
		string name;
		boost::python::ssize_t i, n;
		boost::python::tuple t;
		daeExternalFunctionArgumentMap_t mapArguments;
		
		boost::python::list items = arguments.items();
		n = boost::python::len(items);
		
		for(i = 0; i < n; i++)
		{
			t = boost::python::extract<boost::python::tuple>(items[i]);
			name  = boost::python::extract<string>(t[0]);
			boost::python::extract<adouble> get_adouble(t[1]);
			boost::python::extract<adouble_array> get_adouble_array(t[1]);
			
			if(get_adouble.check())
				mapArguments[name] = get_adouble();
			else if(get_adouble_array.check())
				mapArguments[name] = get_adouble_array();
		}
		SetArguments(mapArguments);
	}
	
	boost::python::object Calculate_(boost::python::tuple arguments, boost::python::dict values)
	{
		daeDeclareAndThrowException(exNotImplemented);
		return boost::python::object();
	}

	adouble Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const
	{
		boost::python::tuple arguments;
		boost::python::dict values;

        if(boost::python::override f = this->get_override("Calculate"))
        {
            for(daeExternalFunctionArgumentValueMap_t::iterator iter = mapValues.begin(); iter != mapValues.end(); iter++)
			{
				adouble*              ad    = boost::get<adouble>              (&iter->second);
				std::vector<adouble>* adarr = boost::get<std::vector<adouble> >(&iter->second);
				
				if(ad)
					values[iter->first] = *ad;
				else if(adarr)
					values[iter->first] = *adarr;
				else
					daeDeclareAndThrowException(exInvalidCall);
			}
			
			boost::python::object res = f(*arguments, **values);
			boost::python::extract<adouble> get_adouble(res);
			if(get_adouble.check())
			{
				return get_adouble();
			}
			else
			{
				daeDeclareAndThrowException(exInvalidCall);
				return adouble();
			}			
		}
		else
		{
			return daeScalarExternalFunctionWrapper::Calculate(mapValues);
		}
	}
};

/*******************************************************
	daeVectorExternalFunctionWrapper
*******************************************************/
class daeVectorExternalFunctionWrapper : public daeVectorExternalFunction,
                                         public boost::python::wrapper<daeVectorExternalFunction>
{
public:
	daeVectorExternalFunctionWrapper(const string& strName, daeModel* pModel, const unit& units, size_t nNumberofArguments, boost::python::dict arguments)
	    : daeVectorExternalFunction(strName, pModel, units, nNumberofArguments)
	{
		string name;
		boost::python::ssize_t i, n;
		boost::python::tuple t;
		daeExternalFunctionArgumentMap_t mapArguments;
		
		boost::python::list items = arguments.items();
		n = boost::python::len(items);
		
		for(i = 0; i < n; i++)
		{
			t = boost::python::extract<boost::python::tuple>(items[i]);
			name  = boost::python::extract<string>(t[0]);
			boost::python::extract<adouble> get_adouble(t[1]);
			boost::python::extract<adouble_array> get_adouble_array(t[1]);
			
			if(get_adouble.check())
				mapArguments[name] = get_adouble();
			else if(get_adouble_array.check())
				mapArguments[name] = get_adouble_array();
		}
		SetArguments(mapArguments);
	}
	
	boost::python::list Calculate_(boost::python::tuple arguments, boost::python::dict values)
	{
		daeDeclareAndThrowException(exNotImplemented);
		return boost::python::list();
	}

	std::vector<adouble> Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const
	{
		std::vector<adouble> arrResults;
		boost::python::list results;
		boost::python::tuple arguments;
		boost::python::dict values;
        
		if(boost::python::override f = this->get_override("Calculate"))
        {
            for(daeExternalFunctionArgumentValueMap_t::iterator iter = mapValues.begin(); iter != mapValues.end(); iter++)
			{
				adouble*              ad    = boost::get<adouble>              (&iter->second);
				std::vector<adouble>* adarr = boost::get<std::vector<adouble> >(&iter->second);
				
				if(ad)
					values[iter->first] = *ad;
				else if(adarr)
					values[iter->first] = *adarr;
				else
					daeDeclareAndThrowException(exInvalidCall);
			}
			
			results = f(*arguments, **values);
			
			boost::python::ssize_t n = boost::python::len(results);
			arrResults.resize(n);
			
			for(boost::python::ssize_t i = 0; i < n; i++)
			{
				boost::python::extract<adouble> get_adouble(results[i]);
				if(get_adouble.check())
					arrResults[i] = get_adouble();
				else
					daeDeclareAndThrowException(exInvalidCall);
			}
			return arrResults;
		}
		else
		{
			return daeVectorExternalFunctionWrapper::Calculate(mapValues);
		}
	}
};


/*******************************************************
	daeStateTransition
*******************************************************/
class daeStateTransitionWrapper : public daeStateTransition,
                                  public boost::python::wrapper<daeStateTransition>
{
public:
	daeStateTransitionWrapper(void)
	{
	}

public:
	daeCondition GetCondition(void)
	{
		return m_Condition;
	}
};


/*******************************************************
	daeObjectiveFunction, daeOptimizationConstraint
*******************************************************/
boost::python::numeric::array GetGradientsObjectiveFunction(daeObjectiveFunction& o);
boost::python::numeric::array GetGradientsOptimizationConstraint(daeOptimizationConstraint& o);
boost::python::numeric::array GetGradientsMeasuredVariable(daeMeasuredVariable& o);

/*******************************************************
	daeLog
*******************************************************/
class daeLogWrapper : public daeLog_t,
	                  public boost::python::wrapper<daeLog_t>
{
public:
	void Message(const string& strMessage, size_t nSeverity)
	{
		this->get_override("Message")(strMessage, nSeverity);
	}
};

class daeBaseLogWrapper : public daeBaseLog,
	                      public boost::python::wrapper<daeBaseLog>
{
public:
	daeBaseLogWrapper(void)
	{
	}

	void Message(const string& strMessage, size_t nSeverity)
	{
        if(boost::python::override f = this->get_override("Message"))
            f(strMessage, nSeverity);
		else
			this->daeBaseLog::Message(strMessage, nSeverity);
	}

	void def_Message(const string& strMessage, size_t nSeverity)
	{
        this->daeBaseLog::Message(strMessage, nSeverity);
	}

	void SetProgress(real_t dProgress)
	{
        if(boost::python::override f = this->get_override("SetProgress"))
            f(dProgress);
		else
			this->daeBaseLog::SetProgress(dProgress);
	}

	void def_SetProgress(real_t dProgress)
	{
        this->daeBaseLog::SetProgress(dProgress);
	}
};

class daeFileLogWrapper : public daeFileLog,
	                      public boost::python::wrapper<daeFileLog>
{
public:
	daeFileLogWrapper(string strFileName) : daeFileLog(strFileName)
	{
	}

	void Message(const string& strMessage, size_t nSeverity)
	{
        if(boost::python::override f = this->get_override("Message"))
            f(strMessage, nSeverity);
		else
			this->daeFileLog::Message(strMessage, nSeverity);
	}

	void def_Message(const string& strMessage, size_t nSeverity)
	{
        this->daeFileLog::Message(strMessage, nSeverity);
	}
};

class daeStdOutLogWrapper : public daeStdOutLog,
	                        public boost::python::wrapper<daeStdOutLog>
{
public:
	daeStdOutLogWrapper(void){}

	void Message(const string& strMessage, size_t nSeverity)
	{
        if(boost::python::override f = this->get_override("Message"))
            f(strMessage, nSeverity);
		else
			this->daeStdOutLog::Message(strMessage, nSeverity);
	}

	void def_Message(const string& strMessage, size_t nSeverity)
	{
        this->daeStdOutLog::Message(strMessage, nSeverity);
	}
};

class daeTCPIPLogWrapper : public daeTCPIPLog,
	                       public boost::python::wrapper<daeTCPIPLog>
{
public:
	daeTCPIPLogWrapper(string strIPAddress, int nPort) : daeTCPIPLog(strIPAddress, nPort)
	{
	}

	void Message(const string& strMessage, size_t nSeverity)
	{
        if(boost::python::override f = this->get_override("Message"))
            f(strMessage, nSeverity);
		else
			this->daeTCPIPLog::Message(strMessage, nSeverity);
	}

	void def_Message(const string& strMessage, size_t nSeverity)
	{
        this->daeTCPIPLog::Message(strMessage, nSeverity);
	}
};

class thread_locker
{
public:
	thread_locker()
	{
	//	if(thread_support::enabled())
			m_gstate = PyGILState_Ensure();
	}

	~thread_locker()
	{
	//	if(boost::thread_support::enabled())
			PyGILState_Release(m_gstate);
	}
	PyGILState_STATE m_gstate;
};

class daeTCPIPLogServerWrapper : public daeTCPIPLogServer,
	                             public boost::python::wrapper<daeTCPIPLogServer>
{
public:
	daeTCPIPLogServerWrapper(int nPort) : daeTCPIPLogServer(nPort)
	{
	}

	virtual void MessageReceived(const char* szMessage)
	{
		thread_locker lock;
		if(boost::python::override f = this->get_override("MessageReceived"))
		{
			f(szMessage);
		}
		else
		{
			daeDeclareException(exNotImplemented);
			e << "daeTCPIPLogServer::MessageReceived() function must be implemented in the derived class";
			throw e;
		}
	}
};

}

#endif
