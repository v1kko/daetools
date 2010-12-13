#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#include <string>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "../dae_develop.h"
#include "../DataReporters/datareporters.h"
#include "../Simulation/dyn_simulation.h"
#include "../Solver/ida_solver.h"
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
	
/*******************************************************
	Common functions
*******************************************************/
daeDomainIndex CreateDomainIndex(boost::python::object& o);
daeArrayRange  CreateArrayRange(boost::python::object& s);

void daeSaveModel(daeModel& rModel, string strFileName);
		
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

	daeParameterWrapper(string strName, daeeParameterType eType, daeModel* pModel, string strDescription = "")
		: daeParameter(strName, eType, pModel, strDescription)
	{
	}

	daeParameterWrapper(string strName, daeeParameterType eType, daePort* pPort, string strDescription = "")
		: daeParameter(strName, eType, pPort, strDescription)
	{
	}

public:
	boost::python::list GetDomains(void)
	{
		boost::python::list l;
		daeDomain* pDomain;
	
		for(size_t i = 0; i < m_ptrDomains.size(); i++)
		{
			pDomain = m_ptrDomains[i];
			l.append(pDomain);
		}
		return l;
	}

	real_t GetParameterValue0()
	{
		return GetValue();
	}
	
	real_t GetParameterValue1(real_t n1)
	{
		return GetValue(n1);
	}
	
	real_t GetParameterValue2(real_t n1, real_t n2)
	{
		return GetValue(n1, n2);
	}
	
	real_t GetParameterValue3(real_t n1, real_t n2, real_t n3)
	{
		return GetValue(n1, n2, n3);
	}
	
	real_t GetParameterValue4(real_t n1, real_t n2, real_t n3, real_t n4)
	{
		return GetValue(n1, n2, n3, n4);
	}
	
	real_t GetParameterValue5(real_t n1, real_t n2, real_t n3, real_t n4, real_t n5)
	{
		return GetValue(n1, n2, n3, n4, n5);
	}
	
	real_t GetParameterValue6(real_t n1, real_t n2, real_t n3, real_t n4, real_t n5, real_t n6)
	{
		return GetValue(n1, n2, n3, n4, n5, n6);
	}
	
	real_t GetParameterValue7(real_t n1, real_t n2, real_t n3, real_t n4, real_t n5, real_t n6, real_t n7)
	{
		return GetValue(n1, n2, n3, n4, n5, n6, n7);
	}
	
	real_t GetParameterValue8(real_t n1, real_t n2, real_t n3, real_t n4, real_t n5, real_t n6, real_t n7, real_t n8)
	{
		return GetValue(n1, n2, n3, n4, n5, n6, n7, n8);
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

//real_t GetParameterValue0(daeParameter& param);
//real_t GetParameterValue1(daeParameter& param, size_t n1);
//real_t GetParameterValue2(daeParameter& param, size_t n1, size_t n2);
//real_t GetParameterValue3(daeParameter& param, size_t n1, size_t n2, size_t n3);
//real_t GetParameterValue4(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4);
//real_t GetParameterValue5(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5);
//real_t GetParameterValue6(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6);
//real_t GetParameterValue7(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7);
//real_t GetParameterValue8(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8);

void SetParameterValue0(daeParameter& param, real_t value);
void SetParameterValue1(daeParameter& param, size_t n1, real_t value);
void SetParameterValue2(daeParameter& param, size_t n1, size_t n2, real_t value);
void SetParameterValue3(daeParameter& param, size_t n1, size_t n2, size_t n3, real_t value);
void SetParameterValue4(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, real_t value);
void SetParameterValue5(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value);
void SetParameterValue6(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value);
void SetParameterValue7(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value);
void SetParameterValue8(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value);

adouble_array ParameterArray1(daeParameter& param, boost::python::object o1);
adouble_array ParameterArray2(daeParameter& param, boost::python::object o1, boost::python::object o2);
adouble_array ParameterArray3(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3);
adouble_array ParameterArray4(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4);
adouble_array ParameterArray5(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5);
adouble_array ParameterArray6(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6);
adouble_array ParameterArray7(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7);
adouble_array ParameterArray8(daeParameter& param, boost::python::object o1, boost::python::object o2, boost::python::object o3, boost::python::object o4, boost::python::object o5, boost::python::object o6, boost::python::object o7, boost::python::object o8);

/*******************************************************
	daeVariable
*******************************************************/
class daeVariableWrapper : public daeVariable,
	                       public boost::python::wrapper<daeVariable>
{
public:
	daeVariableWrapper(void)
	{
	}

	daeVariableWrapper(string strName, const daeVariableType& varType, daeModel* pModel, string strDescription = "")
		: daeVariable(strName, varType, pModel, strDescription)
	{
	}

	daeVariableWrapper(string strName, const daeVariableType& varType, daePort* pPort, string strDescription = "")
		: daeVariable(strName, varType, pPort, strDescription)
	{
	}

public:
	boost::python::list GetDomains(void)
	{
		boost::python::list l;
		daeDomain* pDomain;
	
		for(size_t i = 0; i < m_ptrDomains.size(); i++)
		{
			pDomain = m_ptrDomains[i];
			l.append(pDomain);
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
	
	real_t GetVariableValue1(real_t n1)
	{
		return GetValue(n1);
	}
	
	real_t GetVariableValue2(real_t n1, real_t n2)
	{
		return GetValue(n1, n2);
	}
	
	real_t GetVariableValue3(real_t n1, real_t n2, real_t n3)
	{
		return GetValue(n1, n2, n3);
	}
	
	real_t GetVariableValue4(real_t n1, real_t n2, real_t n3, real_t n4)
	{
		return GetValue(n1, n2, n3, n4);
	}
	
	real_t GetVariableValue5(real_t n1, real_t n2, real_t n3, real_t n4, real_t n5)
	{
		return GetValue(n1, n2, n3, n4, n5);
	}
	
	real_t GetVariableValue6(real_t n1, real_t n2, real_t n3, real_t n4, real_t n5, real_t n6)
	{
		return GetValue(n1, n2, n3, n4, n5, n6);
	}
	
	real_t GetVariableValue7(real_t n1, real_t n2, real_t n3, real_t n4, real_t n5, real_t n6, real_t n7)
	{
		return GetValue(n1, n2, n3, n4, n5, n6, n7);
	}
	
	real_t GetVariableValue8(real_t n1, real_t n2, real_t n3, real_t n4, real_t n5, real_t n6, real_t n7, real_t n8)
	{
		return GetValue(n1, n2, n3, n4, n5, n6, n7, n8);
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
		daeDomain* pDomain;
	
		for(size_t i = 0; i < m_ptrarrDomains.size(); i++)
		{
			pDomain = m_ptrarrDomains[i];
			l.append(pDomain);
		}
		return l;
	}
	
	boost::python::list GetParameters(void)
	{
		boost::python::list l;
		daeParameter* pParameter;
	
		for(size_t i = 0; i < m_ptrarrParameters.size(); i++)
		{
			pParameter = m_ptrarrParameters[i];
			l.append(pParameter);
		}
		return l;
	}
	
	boost::python::list GetVariables(void)
	{
		boost::python::list l;
		daeVariable* pVariable;
	
		for(size_t i = 0; i < m_ptrarrVariables.size(); i++)
		{
			pVariable = m_ptrarrVariables[i];
			l.append(pVariable);
		}
		return l;
	}
};

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

	daeEquation* CreateEquation1(string strName, string strDescription)
	{
		return daeModel::CreateEquation(strName, strDescription);
	}

	daeEquation* CreateEquation2(string strName)
	{
		return daeModel::CreateEquation(strName, "");
	}
	
	void ConnectPorts(daePort& pPortFrom, daePort& pPortTo)
	{
		daeModel::ConnectPorts(&pPortFrom, &pPortTo);
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
			l.append(obj);
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
			l.append(obj);
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
			l.append(obj);
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
			l.append(obj);
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
			l.append(obj);
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
			l.append(obj);
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
			l.append(obj);
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
			l.append(obj);
		}
		return l;
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
			l.append(obj);
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
			l.append(obj);
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
			l.append(obj);
		}
		return l;
	}
};

/*******************************************************
	daeSTN
*******************************************************/
class daeSTNWrapper : public daeSTN,
                      public boost::python::wrapper<daeSTN>
{
public:
	daeSTNWrapper(void)
	{
	}

public:
	boost::python::list GetStates(void)
	{
		boost::python::list l;
		daeState* obj;
	
		for(size_t i = 0; i < m_ptrarrStates.size(); i++)
		{
			obj = m_ptrarrStates[i];
			l.append(obj);
		}
		return l;
	}
	
    daeState* GetActState(void)
	{
		return m_pActiveState;
	}
	
	void SetActiveState(const string& strStateName)
	{
		daeState* pState = daeSTN::FindState(strStateName);
		if(!pState)
			daeDeclareAndThrowException(exInvalidPointer);
		
		daeSTN::SetActiveState(pState);
	}
	
};

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
	daeState* GetStateFrom(void)
	{
		return m_pStateFrom;
	}
	
	daeState* GetStateTo(void)
	{
		return m_pStateTo;
	}

	daeCondition GetCondition(void)
	{
		return m_Condition;
	}
};

/*******************************************************
	daeLog
*******************************************************/
class daeLogWrapper : public daeLog_t,
	                  public boost::python::wrapper<daeLog_t>
{
public:
	void Message(const string& strMessage, size_t nSeverity)
	{
		this->get_override("Message")();
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

/*******************************************************
	daeDataReporter
*******************************************************/
boost::python::list GetDataReporterDomains(daeDataReporterVariable& Variable);

boost::python::list GetDataReporterDomainPoints(daeDataReporterDomain& Domain);
		
boost::python::numeric::array GetNumPyArrayDataReporterVariableValue(daeDataReporterVariableValue& var);
	
class daeDataReporterWrapper : public daeDataReporter_t,
	                           public boost::python::wrapper<daeDataReporter_t>
{
public:
	daeDataReporterWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
		return this->get_override("Connect")(strConnectString, strProcessName);
	}
	bool Disconnect(void)
	{
		return this->get_override("Disconnect")();
	}
	bool IsConnected(void)
	{
		return this->get_override("IsConnected")();
	}
	bool StartRegistration(void)
	{
		return this->get_override("StartRegistration")();
	}
	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
		return this->get_override("RegisterDomain")(pDomain);
	}
	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
		return this->get_override("RegisterVariable")(pVariable);
	}
	bool EndRegistration(void)
	{
		return this->get_override("EndRegistration")();
	}
	bool StartNewResultSet(real_t dTime)
	{
		return this->get_override("StartNewResultSet")(dTime);
	}
	bool EndOfData(void)
	{
		return this->get_override("EndOfData")();
	}
	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
		return this->get_override("SendVariable")(pVariableValue);
	}
};

class daeDataReporterLocalWrapper : public daeDataReporterLocal,
	                                public boost::python::wrapper<daeDataReporterLocal>
{
public:
	daeDataReporterLocalWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
		return this->get_override("Connect")(strConnectString, strProcessName);
	}
	bool Disconnect(void)
	{
		return this->get_override("Disconnect")();
	}
	bool IsConnected(void)
	{
		return this->get_override("IsConnected")();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeDataReporterLocal::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeDataReporterLocal::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeDataReporterLocal::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeDataReporterLocal::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeDataReporterLocal::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeDataReporterLocal::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeDataReporterLocal::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeDataReporterLocal::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeDataReporterLocal::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeDataReporterLocal::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeDataReporterLocal::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeDataReporterLocal::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeDataReporterLocal::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeDataReporterLocal::SendVariable(pVariableValue);
	}
};

class daeDataReporterFileWrapper : public daeFileDataReporter,
	                               public boost::python::wrapper<daeFileDataReporter>
{
public:
	daeDataReporterFileWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeFileDataReporter::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeFileDataReporter::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeFileDataReporter::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeFileDataReporter::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeFileDataReporter::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeFileDataReporter::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeFileDataReporter::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeFileDataReporter::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeFileDataReporter::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeFileDataReporter::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeFileDataReporter::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeFileDataReporter::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeFileDataReporter::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeFileDataReporter::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeFileDataReporter::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeFileDataReporter::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeFileDataReporter::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeFileDataReporter::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeFileDataReporter::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeFileDataReporter::SendVariable(pVariableValue);
	}

	void WriteDataToFile(void)
	{
        this->get_override("WriteDataToFile")();
	}
};

class daeTEXTFileDataReporterWrapper : public daeTEXTFileDataReporter,
	                                   public boost::python::wrapper<daeTEXTFileDataReporter>
{
public:
	daeTEXTFileDataReporterWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeTEXTFileDataReporter::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeTEXTFileDataReporter::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeTEXTFileDataReporter::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeTEXTFileDataReporter::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeTEXTFileDataReporter::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeTEXTFileDataReporter::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeTEXTFileDataReporter::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeTEXTFileDataReporter::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeTEXTFileDataReporter::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeTEXTFileDataReporter::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeTEXTFileDataReporter::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeTEXTFileDataReporter::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeTEXTFileDataReporter::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeTEXTFileDataReporter::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeTEXTFileDataReporter::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeTEXTFileDataReporter::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeTEXTFileDataReporter::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeTEXTFileDataReporter::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeTEXTFileDataReporter::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeTEXTFileDataReporter::SendVariable(pVariableValue);
	}

	void WriteDataToFile(void)
	{
        if(boost::python::override f = this->get_override("WriteDataToFile"))
            f();
		else
			this->daeTEXTFileDataReporter::WriteDataToFile();
	}
	void def_WriteDataToFile(void)
	{
        this->daeTEXTFileDataReporter::WriteDataToFile();
	}
};

class daeDelegateDataReporterWrapper : public daeDelegateDataReporter,
	                                   public boost::python::wrapper<daeDelegateDataReporter>
{
public:
	daeDelegateDataReporterWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeDelegateDataReporter::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeDelegateDataReporter::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeDelegateDataReporter::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeDelegateDataReporter::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeDelegateDataReporter::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeDelegateDataReporter::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeDelegateDataReporter::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeDelegateDataReporter::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeDelegateDataReporter::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeDelegateDataReporter::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeDelegateDataReporter::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeDelegateDataReporter::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeDelegateDataReporter::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeDelegateDataReporter::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeDelegateDataReporter::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeDelegateDataReporter::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeDelegateDataReporter::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeDelegateDataReporter::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeDelegateDataReporter::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeDelegateDataReporter::SendVariable(pVariableValue);
	}
//	
//	void AddDataReporter(daeDataReporter_t* pDataReporter)
//	{
//        if(boost::python::override f = this->get_override("AddDataReporter"))
//            f(pDataReporter);
//		else
//			this->daeDelegateDataReporter::AddDataReporter(pDataReporter);
//	}
//	void def_AddDataReporter(daeDataReporter_t* pDataReporter)
//	{
//        this->daeDelegateDataReporter::AddDataReporter(pDataReporter);
//	}
};

class daeDataReporterRemoteWrapper : public daeDataReporterRemote,
	                                 public boost::python::wrapper<daeDataReporterRemote>
{
public:
	daeDataReporterRemoteWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeDataReporterRemote::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeDataReporterRemote::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeDataReporterRemote::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeDataReporterRemote::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeDataReporterRemote::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeDataReporterRemote::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeDataReporterRemote::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeDataReporterRemote::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeDataReporterRemote::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeDataReporterRemote::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeDataReporterRemote::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeDataReporterRemote::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeDataReporterRemote::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeDataReporterRemote::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeDataReporterRemote::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeDataReporterRemote::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeDataReporterRemote::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeDataReporterRemote::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeDataReporterRemote::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeDataReporterRemote::SendVariable(pVariableValue);
	}

	bool SendMessage(const string& strMessage)
	{
        return this->get_override("SendMessage")(strMessage);
	}
};

class daeTCPIPDataReporterWrapper : public daeTCPIPDataReporter,
	                                public boost::python::wrapper<daeTCPIPDataReporter>
{
public:
	daeTCPIPDataReporterWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeTCPIPDataReporter::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeTCPIPDataReporter::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeTCPIPDataReporter::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeTCPIPDataReporter::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeTCPIPDataReporter::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeTCPIPDataReporter::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeTCPIPDataReporter::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeTCPIPDataReporter::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeTCPIPDataReporter::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeTCPIPDataReporter::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeTCPIPDataReporter::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeTCPIPDataReporter::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeTCPIPDataReporter::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeTCPIPDataReporter::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeTCPIPDataReporter::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeTCPIPDataReporter::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeTCPIPDataReporter::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeTCPIPDataReporter::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeTCPIPDataReporter::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeTCPIPDataReporter::SendVariable(pVariableValue);
	}

	bool SendMessage(const string& strMessage)
	{
        if(boost::python::override f = this->get_override("SendMessage"))
            return f(strMessage);
		else
			return this->daeTCPIPDataReporter::SendMessage(strMessage);
	}
	bool def_SendMessage(const string& strMessage)
	{
        return this->daeTCPIPDataReporter::SendMessage(strMessage);
	}
};
	
/*******************************************************
	daeDataReceiver
*******************************************************/
class daeDataReceiverWrapper : public daeDataReceiver_t,
	                           public boost::python::wrapper<daeDataReceiver_t>
{
public:
	daeDataReceiverWrapper(void)
	{
	}

	bool Start(void)
	{
		return this->get_override("Start")();
	}
	
	bool Stop(void)
	{
		return this->get_override("Stop")();
	}
	
	daeDataReporterProcess*	GetProcess(void)
	{
		return this->get_override("GetProcess")();
	}

	void GetProcessName(string& strProcessName)
	{
	}
	
	void GetDomains(std::vector<const daeDataReceiverDomain*>& ptrarrDomains) const
	{
	}
	
	void GetVariables(std::map<string, const daeDataReceiverVariable*>& ptrmapVariables) const
	{
	}
};

class daeTCPIPDataReceiverWrapper : public daeTCPIPDataReceiver,
	                                public boost::python::wrapper<daeTCPIPDataReceiver>
{
public:
	bool Start()
	{
        if(boost::python::override f = this->get_override("Start"))
            return f();
		else
			return this->daeTCPIPDataReceiver::Start();
	}
	bool def_Start()
	{
        return this->daeTCPIPDataReceiver::Start();
	}

	bool Stop()
	{
        if(boost::python::override f = this->get_override("Stop"))
            return f();
		else
			return this->daeTCPIPDataReceiver::Stop();
	}
	bool def_Stop()
	{
        return this->daeTCPIPDataReceiver::Stop();
	}

	daeDataReporterProcess* GetProcess()
	{
        if(boost::python::override f = this->get_override("GetProcess"))
            return f();
		else
			return this->daeTCPIPDataReceiver::GetProcess();
	}
	daeDataReporterProcess* def_GetProcess()
	{
        return this->daeTCPIPDataReceiver::GetProcess();
	}
};
	
class daeTCPIPDataReceiverServerWrapper : public daeTCPIPDataReceiverServer,
	                                      public boost::python::wrapper<daeTCPIPDataReceiverServer>
{
public:
	daeTCPIPDataReceiverServerWrapper(int nPort) : daeTCPIPDataReceiverServer(nPort)
	{
	}

    void Start_(void)
    {
        this->daeTCPIPDataReceiverServer::Start();
    }

    void Stop_(void)
    {
        this->daeTCPIPDataReceiverServer::Stop();
    }

	bool IsConnected_(void)
	{
        return this->daeTCPIPDataReceiverServer::IsConnected();
	}

    size_t GetNumberOfDataReceivers(void)
	{
		return m_ptrarrDataReceivers.size();
	}
	
	daeDataReceiver_t* GetDataReceiver(size_t nIndex)
	{
		return m_ptrarrDataReceivers[nIndex];
	}

	boost::python::list GetDataReceivers(void)
	{
		boost::python::list l;
		daeTCPIPDataReceiver* pDataReceiver;
	
		for(size_t i = 0; i < m_ptrarrDataReceivers.size(); i++)
		{
			pDataReceiver = m_ptrarrDataReceivers[i];
			l.append(pDataReceiver);
		}
		return l;
	}

	size_t GetNumberOfProcesses(void)
	{
		return m_ptrarrDataReceivers.size();
	}
	
	daeDataReporterProcess* GetProcess(size_t nIndex)
	{
		return m_ptrarrDataReceivers[nIndex]->GetProcess();
	}
	
	boost::python::list GetProcesses(void)
	{
		boost::python::list l;
		boost::python::object o;
		daeTCPIPDataReceiver* pDataReceiver;
		daeDataReporterProcess* pDataReporterProcess;
	
		for(size_t i = 0; i < m_ptrarrDataReceivers.size(); i++)
		{
			pDataReceiver = m_ptrarrDataReceivers[i];
			pDataReporterProcess = pDataReceiver->GetProcess();
			l.append(boost::python::object(pDataReporterProcess));
		}
		return l;
	}
};	

class daeHybridDataReporterReceiverWrapper : public daeHybridDataReporterReceiver,
	                                         public boost::python::wrapper<daeHybridDataReporterReceiver>
{
public:
	daeHybridDataReporterReceiverWrapper(void)
	{
	}

	bool Connect(const string& strConnectString, const string& strProcessName)
	{
        if(boost::python::override f = this->get_override("Connect"))
            return f(strConnectString, strProcessName);
		else
			return this->daeHybridDataReporterReceiver::Connect(strConnectString, strProcessName);
	}
	bool def_Connect(const string& strConnectString, const string& strProcessName)
	{
        return this->daeHybridDataReporterReceiver::Connect(strConnectString, strProcessName);
	}

	bool Disconnect(void)
	{
        if(boost::python::override f = this->get_override("Disconnect"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::Disconnect();
	}
	bool def_Disconnect(void)
	{
        return this->daeHybridDataReporterReceiver::Disconnect();
	}

	bool IsConnected(void)
	{
        if(boost::python::override f = this->get_override("IsConnected"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::IsConnected();
	}
	bool def_IsConnected(void)
	{
        return this->daeHybridDataReporterReceiver::IsConnected();
	}

	bool StartRegistration(void)
	{
        if(boost::python::override f = this->get_override("StartRegistration"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::StartRegistration();
	}
	bool def_StartRegistration(void)
	{
        return this->daeHybridDataReporterReceiver::StartRegistration();
	}

	bool RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        if(boost::python::override f = this->get_override("RegisterDomain"))
            return f(pDomain);
		else
			return this->daeHybridDataReporterReceiver::RegisterDomain(pDomain);
	}
	bool def_RegisterDomain(const daeDataReporterDomain* pDomain)
	{
        return this->daeHybridDataReporterReceiver::RegisterDomain(pDomain);
	}

	bool RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        if(boost::python::override f = this->get_override("RegisterVariable"))
            return f(pVariable);
		else
			return this->daeHybridDataReporterReceiver::RegisterVariable(pVariable);
	}
	bool def_RegisterVariable(const daeDataReporterVariable* pVariable)
	{
        return this->daeHybridDataReporterReceiver::RegisterVariable(pVariable);
	}

	bool EndRegistration(void)
	{
        if(boost::python::override f = this->get_override("EndRegistration"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::EndRegistration();
	}
	bool def_EndRegistration(void)
	{
        return this->daeHybridDataReporterReceiver::EndRegistration();
	}

	bool StartNewResultSet(real_t dTime)
	{
        if(boost::python::override f = this->get_override("StartNewResultSet"))
            return f(dTime);
		else
			return this->daeHybridDataReporterReceiver::StartNewResultSet(dTime);
	}
	bool def_StartNewResultSet(real_t dTime)
	{
        return this->daeHybridDataReporterReceiver::StartNewResultSet(dTime);
	}

	bool EndOfData(void)
	{
        if(boost::python::override f = this->get_override("EndOfData"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::EndOfData();
	}
	bool def_EndOfData(void)
	{
        return this->daeHybridDataReporterReceiver::EndOfData();
	}

	bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        if(boost::python::override f = this->get_override("SendVariable"))
            return f(pVariableValue);
		else
			return this->daeHybridDataReporterReceiver::SendVariable(pVariableValue);
	}
	bool def_SendVariable(const daeDataReporterVariableValue* pVariableValue)
	{
        return this->daeHybridDataReporterReceiver::SendVariable(pVariableValue);
	}

	bool Start()
	{
        if(boost::python::override f = this->get_override("Start"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::Start();
	}
	bool def_Start()
	{
        return this->daeHybridDataReporterReceiver::Start();
	}

	bool Stop()
	{
        if(boost::python::override f = this->get_override("Stop"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::Stop();
	}
	bool def_Stop()
	{
        return this->daeHybridDataReporterReceiver::Stop();
	}

	daeDataReporterProcess* GetProcess()
	{
        if(boost::python::override f = this->get_override("GetProcess"))
            return f();
		else
			return this->daeHybridDataReporterReceiver::GetProcess();
	}
	daeDataReporterProcess* def_GetProcess()
	{
        return this->daeHybridDataReporterReceiver::GetProcess();
	}
};


/*******************************************************
	daeDataReceiverDomain
*******************************************************/
boost::python::list GetDataReceiverDomainPoints(daeDataReceiverDomain& domain);

/*******************************************************
	daeDataReceiverVariable
*******************************************************/
boost::python::numeric::array GetNumPyArrayDataReceiverVariable(daeDataReceiverVariable& var);
boost::python::numeric::array GetTimeValuesDataReceiverVariable(daeDataReceiverVariable& var);

boost::python::list GetDomainsDataReceiverVariable(daeDataReceiverVariable& var);

/*******************************************************
	daeDataReporterProcess
*******************************************************/
boost::python::list GetDomainsDataReporterProcess(daeDataReporterProcess& process);
boost::python::list GetVariablesDataReporterProcess(daeDataReporterProcess& process);

/*******************************************************
	daeDAESolver
*******************************************************/
class daeDAESolverWrapper : public daeDAESolver_t,
	                        public boost::python::wrapper<daeDAESolver_t>
{
public:
	daeDAESolverWrapper(void){}

	void Initialize(daeBlock_t* pBlock, daeLog_t* pLog)
	{
		this->get_override("Initialize")(pBlock, pLog);
	}
	
	real_t Solve(real_t dTime, bool bStopAtDiscontinuity)
	{
		return this->get_override("Solve")(dTime, bStopAtDiscontinuity);
	}
	
	daeBlock_t* GetBlock(void) const
	{
		return this->get_override("GetBlock")();
	}
	
	daeLog_t* GetLog(void) const
	{
		return this->get_override("GetLog")();
	}
};


class daeIDASolverWrapper : public daeIDASolver,
	                        public boost::python::wrapper<daeIDASolver>
{
public:
	daeIDASolverWrapper(void)
	{
	}

	void Initialize(daeBlock_t* pBlock, daeLog_t* pLog, daeeInitialConditionMode eMode)
	{
        if(boost::python::override f = this->get_override("Initialize"))
            f(pBlock, pLog, eMode);
		else
			this->daeIDASolver::Initialize(pBlock, pLog, eMode);
	}
	void def_Initialize(daeBlock_t* pBlock, daeLog_t* pLog, daeeInitialConditionMode eMode)
	{
        this->daeIDASolver::Initialize(pBlock, pLog, eMode);
	}
	
	real_t Solve(real_t dTime, daeeStopCriterion eStop)
	{
        if(boost::python::override f = this->get_override("Solve"))
            return f(dTime, eStop);
		else
			return this->daeIDASolver::Solve(dTime, eStop);
	}
	real_t def_Solve(real_t dTime, daeeStopCriterion eStop)
	{
        return this->daeIDASolver::Solve(dTime, eStop);
	}
	
//	boost::python::tuple GetSparseMatrixData(void)
//	{
//		boost::python::list ia;
//		boost::python::list ja;
//		int i, NNZ;
//		int *IA, *JA;
//		
//		daeIDASolver::GetSparseMatrixData(NNZ, &IA, &JA);
//
//		if(NNZ == 0)
//			return boost::python::make_tuple(0, 0, ia, ja);
//		
//		for(i = 0; i < m_nNumberOfEquations+1; i++)
//			ia.append(IA[i]);
//
//		for(i = 0; i < NNZ; i++)
//			ja.append(JA[i]);
//
//		return boost::python::make_tuple(m_nNumberOfEquations, NNZ, ia, ja);
//	}
	

};

/*******************************************************
	daeDynamicSimulation
*******************************************************/
class daeActivityWrapper : public daeActivity_t,
	                       public boost::python::wrapper<daeActivity_t>
{
public:
	daeModel_t* GetModel(void) const
	{
		return this->get_override("GetModel")();
	}

	void SetModel(daeModel_t* pModel)
	{
		this->get_override("SetModel")(pModel);
	}

	daeDataReporter_t* GetDataReporter(void) const
	{
		return this->get_override("GetDataReporter")();
	}

	daeLog_t* GetLog(void) const
	{
		return this->get_override("GetLog")();
	}

	void Run(void)
	{
		this->get_override("Run")();
	}
};

class daeDynamicActivityWrapper : public daeDynamicActivity_t,
	                              public boost::python::wrapper<daeDynamicActivity_t>
{
public:
	void ReportData(void) const
	{
        this->get_override("ReportData")();
	}

	void SetTimeHorizon(real_t dTimeHorizon)
	{
        this->get_override("SetTimeHorizon")(dTimeHorizon);
	}
	
	real_t GetTimeHorizon(void) const
	{
        return this->get_override("GetTimeHorizon")();
	}
	
	void SetReportingInterval(real_t dReportingInterval)
	{
        this->get_override("SetReportingInterval")(dReportingInterval);
	}
	
	real_t GetReportingInterval(void) const
	{
        return this->get_override("GetReportingInterval")();
	}
	
	void Pause(void)
	{
		this->get_override("Pause")();
	}

	void Resume(void)
	{
		this->get_override("Resume")();
	}

	void Stop(void)
	{
		this->get_override("Stop")();
	}
};

class daeDynamicSimulationWrapper : public daeDynamicSimulation_t,
	                                public boost::python::wrapper<daeDynamicSimulation_t>
{
public:
	void Initialize(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog)
	{
        this->get_override("Initialize")(pDAESolver, pDataReporter, pLog);
	}
	
	void Reinitialize(void)
	{
        this->get_override("Reinitialize")();
	}

	void SolveInitial(void)
	{
        this->get_override("SolveInitial")();
	}
	
	daeDAESolver_t* GetDAESolver(void) const
	{
        return this->get_override("GetDAESolver")();
	}
	
	void SetUpParametersAndDomains(void)
	{
        this->get_override("SetUpParametersAndDomains")();
	}

	void SetUpVariables(void)
	{
        this->get_override("SetUpVariables")();
	}
	
	real_t Integrate(daeeStopCriterion eStopCriterion)
	{
        return this->get_override("Integrate")(eStopCriterion);
	}
	
	real_t IntegrateForTimeInterval(real_t time_interval)
	{
        return this->get_override("IntegrateForTimeInterval")(time_interval);
	}
	
	real_t IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion)
	{
        return this->get_override("IntegrateUntilTime")(time, eStopCriterion);
	}
	
};

class daeDefaultDynamicSimulationWrapper : public daeDynamicSimulation,
	                                       public boost::python::wrapper<daeDynamicSimulation>
{
public:
	daeDefaultDynamicSimulationWrapper()
	{
	}

	boost::python::object GetModel_(void) const
	{
		return model;
	}

	void SetModel_(boost::python::object Model)
	{
		model = Model;
		daeModel* pModel = boost::python::extract<daeModel*>(Model);
		this->daeDynamicSimulation::SetModel(pModel);
	}

	void SetUpParametersAndDomains(void)
	{
        if(boost::python::override f = this->get_override("SetUpParametersAndDomains"))
			f();
		else
			this->daeDynamicSimulation::SetUpParametersAndDomains();
	}
	void def_SetUpParametersAndDomains(void)
	{
		this->daeDynamicSimulation::SetUpParametersAndDomains();
	}

	void SetUpVariables(void)
	{
        if(boost::python::override f = this->get_override("SetUpVariables"))
            f();
		else
			this->daeDynamicSimulation::SetUpVariables();
	}
	void def_SetUpVariables(void)
	{
		this->daeDynamicSimulation::SetUpVariables();
	}

	void Run(void)
	{
        if(boost::python::override f = this->get_override("Run"))
			f();
 		else
	       return daeDynamicSimulation::Run();
	}
	void def_Run(void)
	{
        this->daeDynamicSimulation::Run();
	}

public:
	boost::python::object model;	
};

}

#endif
