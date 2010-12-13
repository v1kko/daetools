#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#define NO_IMPORT_ARRAY
#include <numpy/core/include/numpy/noprefix.h>
using namespace std;
using namespace boost;
using namespace boost::python;
  
namespace daepython
{
/*******************************************************
	Common
*******************************************************/
daeDomainIndex CreateDomainIndex(object& o)
{
	extract<size_t>   size(o);
	extract<daeDEDI*> DEDI(o);
	
	if(size.check())
	{
		size_t n = size();
		return daeDomainIndex(n);
	}
	else if(DEDI.check())
	{
		daeDEDI* pDEDI = DEDI();
		return daeDomainIndex(pDEDI);
	}
	else
	{
		daeDeclareException(exInvalidCall); 
		e << "Invalid argument" ;
		throw e;
		return daeDomainIndex();
	}
}

daeArrayRange CreateArrayRange(object& o)
{
	extract<size_t>        get_size_t(o);
	extract<daeDEDI*>      get_DEDI(o);
	extract<daeIndexRange> get_IndexRange(o);
	
// We have only the first number (start) so it must be integer or daeDEDI
	if(get_DEDI.check())
	{
		daeDEDI* pDEDI = get_DEDI();
		return daeArrayRange(pDEDI);
	}
	else if(get_size_t.check())
	{
		size_t n = get_size_t();
		return daeArrayRange(n);
	}
	else if(get_IndexRange.check())
	{
		daeIndexRange ir = get_IndexRange();
		return daeArrayRange(ir);
	}
	else
	{
		daeDeclareException(exInvalidCall); 
		e << "Invalid argument" ;
		throw e;
		return daeArrayRange();
	}
}

void daeSaveModel(daeModel& rModel, string strFileName)
{
	dae::core::daeSaveModel(&rModel, strFileName);
}

/*******************************************************
	__str__ funkcije
*******************************************************/
string daeVariableType_str(const daeVariableType& self) 
{
	string str;
	str += self.GetName() + string(", ");
	str += toString(self.GetUnits()) + string(", ");
	str += string("[") + toString(self.GetLowerBound()) + string(", ") + toString(self.GetUpperBound()) + string("], ");
	str += toString(self.GetInitialGuess()) + string(", ");
	str += toString(self.GetAbsoluteTolerance());
	return str;
}

string daeDomain_str(const daeDomain& self) 
{
//	cout << "called daeDomain_str" << endl;
	string str;
	str += self.GetCanonicalName();
	return str;
}

string daeParameter_str(const daeParameter& self) 
{
	string str;
	str += self.GetCanonicalName();
	return str;
}

string daeVariable_str(const daeVariable& self) 
{
	string str;
	str += self.GetCanonicalName();
	return str;
}

string daePort_str(const daePort& self) 
{
	string str;
	str += self.GetCanonicalName();
	return str;
}

string daeModel_str(const daeModel& self) 
{
	string str;
	str += self.GetCanonicalName();
	return str;
}

string daeEquation_str(const daeEquation& self) 
{
	string str;
	str += self.GetCanonicalName();
	return str;
}

string daeDEDI_str(const daeDEDI& self)
{
	string str;
	str += self.GetCanonicalName();
	return str;
}

/*******************************************************
	adouble
*******************************************************/
const adouble ad_exp(const adouble &a)
{
	return exp(a);
}
const adouble ad_log(const adouble &a)
{
	return log(a);
}
const adouble ad_sqrt(const adouble &a)
{
	return sqrt(a);
}
const adouble ad_sin(const adouble &a)
{
	return sin(a);
}
const adouble ad_cos(const adouble &a)
{
	return cos(a);
}
const adouble ad_tan(const adouble &a)
{
	return tan(a);
}
const adouble ad_asin(const adouble &a)
{
	return asin(a);
}
const adouble ad_acos(const adouble &a)
{
	return acos(a);
}
const adouble ad_atan(const adouble &a)
{
	return atan(a);
}
const adouble ad_pow1(const adouble &a, real_t v)
{
	return pow(a,v);
}
const adouble ad_pow2(const adouble &a, const adouble &b)
{
	return pow(a,b);
}
const adouble ad_pow3(real_t v, const adouble &a)
{
	return pow(v,a);
}
const adouble ad_log10(const adouble &a)
{
	return log10(a);
}
const adouble ad_abs(const adouble &a)
{
	return abs(a);
}
const adouble ad_ceil(const adouble &a)
{
	return ceil(a);
}
const adouble ad_floor(const adouble &a)
{
	return floor(a);
}
const adouble ad_max1(const adouble &a, const adouble &b)
{
	return max(a,b);
}
const adouble ad_max2(real_t v, const adouble &a)
{
	return max(v,a);
}
const adouble ad_max3(const adouble &a, real_t v)
{
	return max(a,v);
}
const adouble ad_min1(const adouble &a, const adouble &b)
{
	return min(a,b);
}
const adouble ad_min2(real_t v, const adouble &a)
{
	return min(v,a);
}
const adouble ad_min3(const adouble &a, real_t v)
{
	return min(a,v);
}

/*******************************************************
	adouble_array
*******************************************************/
const adouble_array adarr_exp(const adouble_array& a)
{
	return exp(a);
}
const adouble_array adarr_sqrt(const adouble_array& a)
{
	return sqrt(a);
}
const adouble_array adarr_log(const adouble_array& a)
{
	return log(a);
}
const adouble_array adarr_log10(const adouble_array& a)
{
	return log10(a);
}
const adouble_array adarr_abs(const adouble_array& a)
{
	return abs(a);
}
const adouble_array adarr_floor(const adouble_array& a)
{
	return floor(a);
}
const adouble_array adarr_ceil(const adouble_array& a)
{
	return ceil(a);
}
const adouble_array adarr_sin(const adouble_array& a)
{
	return sin(a);
}
const adouble_array adarr_cos(const adouble_array& a)
{
	return cos(a);
}
const adouble_array adarr_tan(const adouble_array& a)
{
	return tan(a);
}
const adouble_array adarr_asin(const adouble_array& a)
{
	return asin(a);
}
const adouble_array adarr_acos(const adouble_array& a)
{
	return acos(a);
}
const adouble_array adarr_atan(const adouble_array& a)
{
	return atan(a);
}


/*******************************************************
	daeDomain
*******************************************************/
python::numeric::array GetNumPyArrayDomain(daeDomain& domain)
{
	size_t nType;
	npy_intp dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = domain.GetNumberOfPoints();
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	for(size_t k = 0; k < dimensions; k++)
		values[k] = domain.GetPoint(k);

	return numpy_array;
}

boost::python::list GetDomainPoints(daeDomain& domain)
{
	boost::python::list l;

	for(size_t i = 0; i < domain.GetNumberOfPoints(); i++)
		l.append(domain.GetPoint(i));

	return l;
}
	
void SetDomainPoints(daeDomain& domain, boost::python::list l)
{
	real_t point;
	std::vector<real_t> darrPoints;
	
	boost::python::ssize_t n = boost::python::len(l);
	for(boost::python::ssize_t i = 0; i < n; i++) 
	{
		point = extract<real_t>(l[i]);
		darrPoints.push_back(point);
	}

	domain.SetPoints(darrPoints);
}

adouble_array DomainArray1(daeDomain& domain)
{
	return domain.array();
}

adouble_array DomainArray2(daeDomain& domain, slice s)
{
	extract<int> get_start(s.start());
	extract<int> get_end(s.stop());
	extract<int> get_step(s.step());

	int start = get_start.check() ? get_start() : 0;
	int end   = get_end.check()   ? get_end() : domain.GetNumberOfPoints()-1;
	int step  = get_step.check()  ? get_step() : 1;

	return domain.array(start, end, step);
}

daeIndexRange* __init__daeIndexRange(daeDomain* pDomain, boost::python::list CustomPoints)
{
	size_t index;
	std::vector<size_t> narrCustomPoints;
	
	boost::python::ssize_t n = boost::python::len(CustomPoints);
	for(boost::python::ssize_t i = 0; i < n; i++) 
	{
		index = extract<size_t>(CustomPoints[i]);
		narrCustomPoints.push_back(index);
	}
	return new daeIndexRange(pDomain, narrCustomPoints);
}

daeIndexRange FunctionCallDomain1(daeDomain& domain, int start, int end, int step)
{
	return domain(start, end, step);
}

daeIndexRange FunctionCallDomain2(daeDomain& domain, boost::python::list l)
{
	size_t index;
	std::vector<size_t> narrDomainIndexes;
	boost::python::ssize_t n = boost::python::len(l);
	for(boost::python::ssize_t i = 0; i < n; i++) 
	{
		index = extract<size_t>(l[i]);
		narrDomainIndexes.push_back(index);
	}
	
	return domain(narrDomainIndexes);
}

daeIndexRange FunctionCallDomain3(daeDomain& domain)
{
	return domain();
}

/*******************************************************
	daeParameter
*******************************************************/
python::numeric::array GetNumPyArrayParameter(daeParameter& param)
{
	size_t nType, nDomains, nTotalSize;
	real_t* data;
	npy_intp* dimensions;
	vector<daeDomain_t*> ptrarrDomains;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	data = param.GetValuePointer();
	param.GetDomains(ptrarrDomains);
	nDomains = ptrarrDomains.size();
	dimensions = new npy_intp[nDomains];
	nTotalSize = 1;
	for(size_t i = 0; i < nDomains; i++)
	{
		dimensions[i] = ptrarrDomains[i]->GetNumberOfPoints();
		nTotalSize *= dimensions[i];
	}
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(nDomains, dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	for(size_t k = 0; k < nTotalSize; k++)
		values[k] = data[k];

	delete[] dimensions;
	return numpy_array;
}

adouble FunctionCallParameter0(daeParameter& param)
{
	return param();
}

adouble FunctionCallParameter1(daeParameter& param, object o1)
{
	return param(CreateDomainIndex(o1));
}

adouble FunctionCallParameter2(daeParameter& param, object o1, object o2)
{
	return param(CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble FunctionCallParameter3(daeParameter& param, object o1, object o2, object o3)
{
	return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble FunctionCallParameter4(daeParameter& param, object o1, object o2, object o3, object o4)
{
	return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble FunctionCallParameter5(daeParameter& param, object o1, object o2, object o3, object o4, object o5)
{
	return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble FunctionCallParameter6(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble FunctionCallParameter7(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble FunctionCallParameter8(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

void SetParameterValue0(daeParameter& param, real_t value)
{
	param.SetValue(value);
}

void SetParameterValue1(daeParameter& param, size_t n1, real_t value)
{
	param.SetValue(n1, value);
}

void SetParameterValue2(daeParameter& param, size_t n1, size_t n2, real_t value)
{
	param.SetValue(n1, n2, value);
}

void SetParameterValue3(daeParameter& param, size_t n1, size_t n2, size_t n3, real_t value)
{
	param.SetValue(n1, n2, n3, value);
}

void SetParameterValue4(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
	param.SetValue(n1, n2, n3, n4, value);
}

void SetParameterValue5(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
	param.SetValue(n1, n2, n3, n4, n5, value);
}

void SetParameterValue6(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
	param.SetValue(n1, n2, n3, n4, n5, n6, value);
}

void SetParameterValue7(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
	param.SetValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void SetParameterValue8(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
	param.SetValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

adouble_array ParameterArray1(daeParameter& param, object o1)
{
	return param.array(CreateArrayRange(o1));
}

adouble_array ParameterArray2(daeParameter& param, object o1, object o2)
{
	return param.array(CreateArrayRange(o1), CreateArrayRange(o2));
}

adouble_array ParameterArray3(daeParameter& param, object o1, object o2, object o3)
{
	return param.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3));
}

adouble_array ParameterArray4(daeParameter& param, object o1, object o2, object o3, object o4)
{
	return param.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4));
}

adouble_array ParameterArray5(daeParameter& param, object o1, object o2, object o3, object o4, object o5)
{
	return param.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5));
}

adouble_array ParameterArray6(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return param.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6));
}

adouble_array ParameterArray7(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return param.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7));
}

adouble_array ParameterArray8(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return param.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7), CreateArrayRange(o8));
}

/*******************************************************
	daeVariable
*******************************************************/
python::numeric::array GetNumPyArrayVariable(daeVariable& var)
{
	size_t nType, nDomains, nTotalSize;
	real_t* data;
	npy_intp* dimensions;
	vector<daeDomain_t*> ptrarrDomains;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	data = var.GetValuePointer();
	var.GetDomains(ptrarrDomains);
	nDomains = ptrarrDomains.size();
	dimensions = new npy_intp[nDomains];
	nTotalSize = 1;
	for(size_t i = 0; i < nDomains; i++)
	{
		dimensions[i] = ptrarrDomains[i]->GetNumberOfPoints();
		nTotalSize *= dimensions[i];
	}
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(nDomains, dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	for(size_t k = 0; k < nTotalSize; k++)
		values[k] = data[k];

	delete[] dimensions;
	return numpy_array;
}

adouble VariableFunctionCall0(daeVariable& var)
{
	return var();
}

adouble VariableFunctionCall1(daeVariable& var, object o1)
{
	return var(CreateDomainIndex(o1));
}

adouble VariableFunctionCall2(daeVariable& var, object o1, object o2)
{
	return var(CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble VariableFunctionCall3(daeVariable& var, object o1, object o2, object o3)
{
	return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble VariableFunctionCall4(daeVariable& var, object o1, object o2, object o3, object o4)
{
	return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble VariableFunctionCall5(daeVariable& var, object o1, object o2, object o3, object o4, object o5)
{
	return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble VariableFunctionCall6(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble VariableFunctionCall7(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble VariableFunctionCall8(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

void AssignValue0(daeVariable& var, real_t value)
{
	var.AssignValue(value);
}

void AssignValue1(daeVariable& var, size_t n1, real_t value)
{
	var.AssignValue(n1, value);
}

void AssignValue2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
	var.AssignValue(n1, n2, value);
}

void AssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
	var.AssignValue(n1, n2, n3, value);
}

void AssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
	var.AssignValue(n1, n2, n3, n4, value);
}

void AssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
	var.AssignValue(n1, n2, n3, n4, n5, value);
}

void AssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
	var.AssignValue(n1, n2, n3, n4, n5, n6, value);
}

void AssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
	var.AssignValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void AssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
	var.AssignValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void ReAssignValue0(daeVariable& var, real_t value)
{
	var.ReAssignValue(value);
}

void ReAssignValue1(daeVariable& var, size_t n1, real_t value)
{
	var.ReAssignValue(n1, value);
}

void ReAssignValue2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
	var.ReAssignValue(n1, n2, value);
}

void ReAssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
	var.ReAssignValue(n1, n2, n3, value);
}

void ReAssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
	var.ReAssignValue(n1, n2, n3, n4, value);
}

void ReAssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
	var.ReAssignValue(n1, n2, n3, n4, n5, value);
}

void ReAssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
	var.ReAssignValue(n1, n2, n3, n4, n5, n6, value);
}

void ReAssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
	var.ReAssignValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void ReAssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
	var.ReAssignValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

adouble Get_dt0(daeVariable& var)
{
	return var.dt();
}

adouble Get_dt1(daeVariable& var, object o1)
{
	return var.dt(CreateDomainIndex(o1));
}

adouble Get_dt2(daeVariable& var, object o1, object o2)
{
	return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble Get_dt3(daeVariable& var, object o1, object o2, object o3)
{
	return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble Get_dt4(daeVariable& var, object o1, object o2, object o3, object o4)
{
	return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble Get_dt5(daeVariable& var, object o1, object o2, object o3, object o4, object o5)
{
	return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble Get_dt6(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble Get_dt7(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble Get_dt8(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

adouble Get_d1(daeVariable& var, daeDomain& d, object o1)
{
	return var.d(d, CreateDomainIndex(o1));
}

adouble Get_d2(daeVariable& var, daeDomain& d, object o1, object o2)
{
	return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble Get_d3(daeVariable& var, daeDomain& d, object o1, object o2, object o3)
{
	return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble Get_d4(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4)
{
	return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble Get_d5(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5)
{
	return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble Get_d6(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble Get_d7(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble Get_d8(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

adouble Get_d21(daeVariable& var, daeDomain& d, object o1)
{
	return var.d2(d, CreateDomainIndex(o1));
}

adouble Get_d22(daeVariable& var, daeDomain& d, object o1, object o2)
{
	return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble Get_d23(daeVariable& var, daeDomain& d, object o1, object o2, object o3)
{
	return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble Get_d24(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4)
{
	return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble Get_d25(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5)
{
	return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble Get_d26(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble Get_d27(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble Get_d28(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

adouble_array VariableArray1(daeVariable& var, object o1)
{
	return var.array(CreateArrayRange(o1));
}

adouble_array VariableArray2(daeVariable& var, object o1, object o2)
{
	return var.array(CreateArrayRange(o1), CreateArrayRange(o2));
}

adouble_array VariableArray3(daeVariable& var, object o1, object o2, object o3)
{
	return var.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3));
}

adouble_array VariableArray4(daeVariable& var, object o1, object o2, object o3, object o4)
{
	return var.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4));
}

adouble_array VariableArray5(daeVariable& var, object o1, object o2, object o3, object o4, object o5)
{
	return var.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5));
}

adouble_array VariableArray6(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return var.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6));
}

adouble_array VariableArray7(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return var.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7));
}

adouble_array VariableArray8(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return var.array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7), CreateArrayRange(o8));
}

adouble_array Get_dt_array1(daeVariable& var, object o1)
{
	return var.dt_array(CreateArrayRange(o1));
}

adouble_array Get_dt_array2(daeVariable& var, object o1, object o2)
{
	return var.dt_array(CreateArrayRange(o1), CreateArrayRange(o2));
}

adouble_array Get_dt_array3(daeVariable& var, object o1, object o2, object o3)
{
	return var.dt_array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3));
}

adouble_array Get_dt_array4(daeVariable& var, object o1, object o2, object o3, object o4)
{
	return var.dt_array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4));
}

adouble_array Get_dt_array5(daeVariable& var, object o1, object o2, object o3, object o4, object o5)
{
	return var.dt_array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5));
}

adouble_array Get_dt_array6(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return var.dt_array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6));
}

adouble_array Get_dt_array7(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return var.dt_array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7));
}

adouble_array Get_dt_array8(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return var.dt_array(CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7), CreateArrayRange(o8));
}


adouble_array Get_d_array1(daeVariable& var, daeDomain& d, object o1)
{
	return var.d_array(d, CreateArrayRange(o1));
}

adouble_array Get_d_array2(daeVariable& var, daeDomain& d, object o1, object o2)
{
	return var.d_array(d, CreateArrayRange(o1), CreateArrayRange(o2));
}

adouble_array Get_d_array3(daeVariable& var, daeDomain& d, object o1, object o2, object o3)
{
	return var.d_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3));
}

adouble_array Get_d_array4(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4)
{
	return var.d_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4));
}

adouble_array Get_d_array5(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5)
{
	return var.d_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5));
}

adouble_array Get_d_array6(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return var.d_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6));
}

adouble_array Get_d_array7(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return var.d_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7));
}

adouble_array Get_d_array8(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return var.d_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7), CreateArrayRange(o8));
}

adouble_array Get_d2_array1(daeVariable& var, daeDomain& d, object o1)
{
	return var.d2_array(d, CreateArrayRange(o1));
}

adouble_array Get_d2_array2(daeVariable& var, daeDomain& d, object o1, object o2)
{
	return var.d2_array(d, CreateArrayRange(o1), CreateArrayRange(o2));
}

adouble_array Get_d2_array3(daeVariable& var, daeDomain& d, object o1, object o2, object o3)
{
	return var.d2_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3));
}

adouble_array Get_d2_array4(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4)
{
	return var.d2_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4));
}

adouble_array Get_d2_array5(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5)
{
	return var.d2_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5));
}

adouble_array Get_d2_array6(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6)
{
	return var.d2_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6));
}

adouble_array Get_d2_array7(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
	return var.d2_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7));
}

adouble_array Get_d2_array8(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
	return var.d2_array(d, CreateArrayRange(o1), CreateArrayRange(o2), CreateArrayRange(o3), CreateArrayRange(o4), CreateArrayRange(o5), CreateArrayRange(o6), CreateArrayRange(o7), CreateArrayRange(o8));
}

void SetVariableValue0(daeVariable& var, real_t value)
{
	var.SetValue(value);
}

void SetVariableValue1(daeVariable& var, size_t n1, real_t value)
{
	var.SetValue(n1, value);
}

void SetVariableValue2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
	var.SetValue(n1, n2, value);
}

void SetVariableValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
	var.SetValue(n1, n2, n3, value);
}

void SetVariableValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
	var.SetValue(n1, n2, n3, n4, value);
}

void SetVariableValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
	var.SetValue(n1, n2, n3, n4, n5, value);
}

void SetVariableValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
	var.SetValue(n1, n2, n3, n4, n5, n6, value);
}

void SetVariableValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
	var.SetValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void SetVariableValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
	var.SetValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void SetInitialGuess0(daeVariable& var, real_t value)
{
	var.SetInitialGuess(value);
}

void SetInitialGuess1(daeVariable& var, size_t n1, real_t value)
{
	var.SetInitialGuess(n1, value);
}

void SetInitialGuess2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
	var.SetInitialGuess(n1, n2, value);
}

void SetInitialGuess3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
	var.SetInitialGuess(n1, n2, n3, value);
}

void SetInitialGuess4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
	var.SetInitialGuess(n1, n2, n3, n4, value);
}

void SetInitialGuess5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
	var.SetInitialGuess(n1, n2, n3, n4, n5, value);
}

void SetInitialGuess6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
	var.SetInitialGuess(n1, n2, n3, n4, n5, n6, value);
}

void SetInitialGuess7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
	var.SetInitialGuess(n1, n2, n3, n4, n5, n6, n7, value);
}

void SetInitialGuess8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
	var.SetInitialGuess(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void SetInitialCondition0(daeVariable& var, real_t value)
{
	var.SetInitialCondition(value);
}

void SetInitialCondition1(daeVariable& var, size_t n1, real_t value)
{
	var.SetInitialCondition(n1, value);
}

void SetInitialCondition2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
	var.SetInitialCondition(n1, n2, value);
}

void SetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
	var.SetInitialCondition(n1, n2, n3, value);
}

void SetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
	var.SetInitialCondition(n1, n2, n3, n4, value);
}

void SetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
	var.SetInitialCondition(n1, n2, n3, n4, n5, value);
}

void SetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
	var.SetInitialCondition(n1, n2, n3, n4, n5, n6, value);
}

void SetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
	var.SetInitialCondition(n1, n2, n3, n4, n5, n6, n7, value);
}

void SetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
	var.SetInitialCondition(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void ReSetInitialCondition0(daeVariable& var, real_t value)
{
	var.ReSetInitialCondition(value);
}

void ReSetInitialCondition1(daeVariable& var, size_t n1, real_t value)
{
	var.ReSetInitialCondition(n1, value);
}

void ReSetInitialCondition2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
	var.ReSetInitialCondition(n1, n2, value);
}

void ReSetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
	var.ReSetInitialCondition(n1, n2, n3, value);
}

void ReSetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
	var.ReSetInitialCondition(n1, n2, n3, n4, value);
}

void ReSetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
	var.ReSetInitialCondition(n1, n2, n3, n4, n5, value);
}

void ReSetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
	var.ReSetInitialCondition(n1, n2, n3, n4, n5, n6, value);
}

void ReSetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
	var.ReSetInitialCondition(n1, n2, n3, n4, n5, n6, n7, value);
}

void ReSetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
	var.ReSetInitialCondition(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

/*******************************************************
	daeEquation
*******************************************************/
//daeEquation* __init__daeEquation(const string& strName, daeModel& model)
//{
//	daeEquation* pEquation = model.AddEquation(strName);
//	return pEquation;
//}
daeDEDI* DistributeOnDomain1(daeEquation& eq, daeDomain& rDomain, daeeDomainBounds eDomainBounds)
{
	return eq.DistributeOnDomain(rDomain, eDomainBounds);
}

daeDEDI* DistributeOnDomain2(daeEquation& eq, daeDomain& rDomain, boost::python::list l)
{
	size_t index;
	std::vector<size_t> narrDomainIndexes;
	boost::python::ssize_t n = boost::python::len(l);
	for(boost::python::ssize_t i = 0; i < n; i++) 
	{
		index = extract<size_t>(l[i]);
		narrDomainIndexes.push_back(index);
	}

	 return eq.DistributeOnDomain(rDomain, narrDomainIndexes);
}

/*******************************************************
	daeDataReporterVariable
*******************************************************/
boost::python::list GetDataReporterDomains(daeDataReporterVariable& Variable)
{
	boost::python::list l;

	for(size_t i = 0; i < Variable.m_strarrDomains.size(); i++)
		l.append(Variable.m_strarrDomains[i]);
	return l;
}

boost::python::list GetDataReporterDomainPoints(daeDataReporterDomain& Domain)
{
	boost::python::list l;

	for(size_t i = 0; i < Domain.m_nNumberOfPoints; i++)
		l.append(Domain.m_pPoints[i]);
	return l;
}

python::numeric::array GetNumPyArrayDataReporterVariableValue(daeDataReporterVariableValue& var)
{
	size_t i, nType;
	npy_intp dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = var.m_nNumberOfPoints;
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	memcpy(values, var.m_pValues, sizeof(real_t)*var.m_nNumberOfPoints);

	return numpy_array;
}

/*******************************************************
	daeDataReceiverDomain
*******************************************************/
boost::python::list GetDataReceiverDomainPoints(daeDataReceiverDomain& domain)
{
	python::list l;
	for(size_t i = 0; i < domain.m_nNumberOfPoints; i++)
		l.append(domain.m_pPoints[i]);
	return l;
}

/*******************************************************
	daeDataReceiverVariable
*******************************************************/
python::numeric::array GetNumPyArrayDataReceiverVariable(daeDataReceiverVariable& var)
{
	size_t i, nType, nDomains, nTotalSize, nTimeSize;
	npy_intp* dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	nDomains = var.m_ptrarrDomains.size();
	dimensions = new npy_intp[nDomains + 1];
	nTimeSize = var.m_ptrarrValues.size();
	nTotalSize = nTimeSize;
	dimensions[0] = nTimeSize;
	for(i = 0; i < nDomains; i++)
	{
		dimensions[i+1] = var.m_ptrarrDomains[i]->m_nNumberOfPoints;
		nTotalSize *= dimensions[i+1];
	}
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(nDomains+1, dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	for(i = 0; i < nTimeSize; i++)
	{
		memcpy(values, var.m_ptrarrValues[i]->m_pValues, sizeof(real_t)*var.m_nNumberOfPoints);
		values += var.m_nNumberOfPoints;
	}

	delete[] dimensions;
	return numpy_array;
}

python::numeric::array GetTimeValuesDataReceiverVariable(daeDataReceiverVariable& var)
{
	size_t nType;
	npy_intp dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = var.m_ptrarrValues.size();
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	for(size_t k = 0; k < dimensions; k++)
		values[k] = var.m_ptrarrValues[k]->m_dTime;

	return numpy_array;
}

python::list GetDomainsDataReceiverVariable(daeDataReceiverVariable& var)
{
	python::list l;
	daeDataReceiverDomain* pDomain;
	size_t i, nDomains;

	nDomains = var.m_ptrarrDomains.size();
	
	for(i = 0; i < nDomains; i++)
	{
		pDomain = var.m_ptrarrDomains[i];
		l.append(pDomain);
	}
	return l;
}

/*******************************************************
	daeDataReporterProcess
*******************************************************/
python::list GetDomainsDataReporterProcess(daeDataReporterProcess& process)
{
	python::list l;
	daeDataReceiverDomain* pDomain;

	for(size_t i = 0; i < process.m_ptrarrRegisteredDomains.size(); i++)
	{
		pDomain = process.m_ptrarrRegisteredDomains[i];
		l.append(pDomain);
	}
	return l;
}

python::list GetVariablesDataReporterProcess(daeDataReporterProcess& process)
{
	python::list l;
	daeDataReceiverVariable* pVariable;

	map<string, daeDataReceiverVariable*>::const_iterator iter;
	for(iter = process.m_ptrmapRegisteredVariables.begin(); iter != process.m_ptrmapRegisteredVariables.end(); iter++)
	{
		pVariable = (*iter).second;
		l.append(pVariable);
	}

	return l;
}

}
