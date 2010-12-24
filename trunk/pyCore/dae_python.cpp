#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <numpy/core/include/numpy/noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyCore)
{
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  
/**************************************************************
    Enums
***************************************************************/
    enum_<daeeDomainType>("daeeDomainType")
		.value("eDTUnknown",	dae::core::eDTUnknown)
		.value("eArray",		dae::core::eArray)
		.value("eDistributed",	dae::core::eDistributed)
		.export_values()
		;

	enum_<daeeParameterType>("daeeParameterType")
		.value("ePTUnknown",	dae::core::ePTUnknown)
		.value("eReal",			dae::core::eReal)
		.value("eInteger",		dae::core::eInteger)
		.value("eBool",			dae::core::eBool)
		.export_values()
	;

	enum_<daeePortType>("daeePortType")
		.value("eUnknownPort",	dae::core::eUnknownPort)
		.value("eInletPort",	dae::core::eInletPort)
		.value("eOutletPort",	dae::core::eOutletPort)
		.export_values()
	;

	enum_<daeeDiscretizationMethod>("daeeDiscretizationMethod")
		.value("eDMUnknown",	dae::core::eDMUnknown)
		.value("eCFDM",			dae::core::eCFDM)
		.value("eFFDM",			dae::core::eFFDM)
		.value("eBFDM",			dae::core::eBFDM)
		.value("eCustomDM",		dae::core::eCustomDM)
		.export_values()
		;

	enum_<daeeDomainBounds>("daeeDomainBounds")
		.value("eDBUnknown",		dae::core::eDBUnknown)
		.value("eOpenOpen",			dae::core::eOpenOpen)
		.value("eOpenClosed",		dae::core::eOpenClosed)
		.value("eClosedOpen",		dae::core::eClosedOpen)
		.value("eClosedClosed",		dae::core::eClosedClosed)
		.value("eLowerBound",		dae::core::eLowerBound)
		.value("eUpperBound",		dae::core::eUpperBound)
		.export_values()
	;

	enum_<daeeInitialConditionMode>("daeeInitialConditionMode")
		.value("eICTUnknown",				dae::core::eICTUnknown)
		.value("eAlgebraicValuesProvided",	dae::core::eAlgebraicValuesProvided)
		.value("eSteadyState",				dae::core::eSteadyState)
		.export_values()
	;

	enum_<daeeDomainIndexType>("daeeDomainIndexType")
		.value("eDITUnknown",		dae::core::eDITUnknown)
		.value("eConstantIndex",	dae::core::eConstantIndex)
		.value("eDomainIterator",	dae::core::eDomainIterator)
		.export_values()
	;

	enum_<daeeRangeType>("daeeRangeType")
		.value("eRaTUnknown",			dae::core::eRaTUnknown)
		.value("eRangeConstantIndex",	dae::core::eRangeConstantIndex)
		.value("eRangeDomainIterator",	dae::core::eRangeDomainIterator)
		.value("eRange",				dae::core::eRange)
		.export_values()
	;

	enum_<daeIndexRangeType>("daeIndexRangeType")
		.value("eIRTUnknown",			dae::core::eIRTUnknown)
		.value("eAllPointsInDomain",	dae::core::eAllPointsInDomain)
		.value("eRangeOfIndexes",		dae::core::eRangeOfIndexes)
		.value("eCustomRange",			dae::core::eCustomRange)
		.export_values()
	;
	
/**************************************************************
	Global functions
***************************************************************/
	def("daeGetConfig", &daepython::daeGetConfig);		
	 
/**************************************************************
	Classes
***************************************************************/
	class_<daeConfig>("daeConfig")
		.def("GetBoolean",	   &daepython::GetBoolean1)
		.def("GetFloat",	   &daepython::GetFloat1)
		.def("GetInteger",	   &daepython::GetInteger1)
		.def("GetString",	   &daepython::GetString1)

		.def("GetBoolean",	   &daepython::GetBoolean)
		.def("GetFloat",	   &daepython::GetFloat)
		.def("GetInteger",	   &daepython::GetInteger)
		.def("GetString",	   &daepython::GetString)

		.def("Reload",         &daeConfig::Reload)
	;

	class_<daeCondition>("daeCondition")
		.add_property("EventTolerance",	&daeCondition::GetEventTolerance, &daeCondition::SetEventTolerance)
		.def(!self)
		.def(self | self)
		.def(self & self)
	;

	class_<adouble>("adouble")
		.add_property("Value",		&adouble::getValue,      &adouble::setValue)
		.add_property("Derivative",	&adouble::getDerivative, &adouble::setDerivative)

		.def(- self)
		.def(+ self)

		.def(self + self)
		.def(self - self)
		.def(self * self)
		.def(self / self)
		.def(pow(self, self))
		.def(self == self)
		.def(self <  self)
		.def(self <= self)
		.def(self >  self)
		.def(self >= self)
		.def(self != self)

		.def(self + real_t())
		.def(self - real_t())
		.def(self * real_t())
		.def(self / real_t())
		.def(pow(self, real_t()))
		.def(self == real_t())
		.def(self <  real_t())
		.def(self <= real_t())
		.def(self >  real_t())
		.def(self >= real_t())
		.def(self != real_t())

		.def(real_t() + self)
		.def(real_t() - self)
		.def(real_t() * self)
		.def(real_t() / self)
		.def(pow(real_t(), self))
		.def(real_t() == self)
		.def(real_t() <  self)
		.def(real_t() <= self)
		.def(real_t() >  self)
		.def(real_t() >= self)
		.def(real_t() != self)
		;
	def("Exp",   &daepython::ad_exp);
	def("Log",   &daepython::ad_log);
	def("Sqrt",  &daepython::ad_sqrt);
	def("Sin",   &daepython::ad_sin);
	def("Cos",   &daepython::ad_cos);
	def("Tan",   &daepython::ad_tan);
	def("ASin",  &daepython::ad_asin);
	def("ACos",  &daepython::ad_acos);
	def("ATan",  &daepython::ad_atan);
	def("Log10", &daepython::ad_log10);
	def("Abs",   &daepython::ad_abs);
	def("Ceil",  &daepython::ad_ceil);
	def("Floor", &daepython::ad_floor);
	def("Min",   &daepython::ad_min1);
	def("Min",   &daepython::ad_min2);
	def("Min",   &daepython::ad_min3);
	def("Max",   &daepython::ad_max1);
	def("Max",   &daepython::ad_max2);
	def("Max",   &daepython::ad_max3);
	def("Pow",   &daepython::ad_pow1);
	def("Pow",   &daepython::ad_pow2);
	def("Pow",   &daepython::ad_pow3);
	
	class_<adouble_array>("adouble_array")
		.def("__getitem__", &adouble_array::GetItem)
		.def(- self)
		.def(self + self)
		.def(self - self)
		.def(self * self)
		.def(self / self)
		
		.def(self + real_t())
		.def(self - real_t())
		.def(self * real_t())
		.def(self / real_t())
		.def(real_t() + self)
		.def(real_t() - self)
		.def(real_t() * self)
		.def(real_t() / self)

		.def(self + adouble())
		.def(self - adouble())
		.def(self * adouble())
		.def(self / adouble())
		.def(adouble() + self)
		.def(adouble() - self)
		.def(adouble() * self)
		.def(adouble() / self)
		;    
	def("Exp",		&daepython::adarr_exp);
	def("Log",		&daepython::adarr_log);
	def("Sqrt",		&daepython::adarr_sqrt);
	def("Sin",		&daepython::adarr_sin);
	def("Cos",		&daepython::adarr_cos);
	def("Tan",		&daepython::adarr_tan);
	def("ASin",		&daepython::adarr_asin);
	def("ACos",		&daepython::adarr_acos);
	def("ATan",		&daepython::adarr_atan);
	def("Log10",	&daepython::adarr_log10);
	def("Abs",		&daepython::adarr_abs);
	def("Ceil",		&daepython::adarr_ceil);
	def("Floor",	&daepython::adarr_floor);
 			
	class_<daeVariableType>("daeVariableType")
		.def(init<string, string, real_t, real_t, real_t, real_t>())
		
		.add_property("Name",				&daeVariableType::GetName,				&daeVariableType::SetName)
		.add_property("Units",				&daeVariableType::GetUnits,				&daeVariableType::SetUnits)
		.add_property("LowerBound",			&daeVariableType::GetLowerBound,		&daeVariableType::SetLowerBound)
		.add_property("UpperBound",			&daeVariableType::GetUpperBound,		&daeVariableType::SetUpperBound)
		.add_property("InitialGuess",		&daeVariableType::GetInitialGuess,		&daeVariableType::SetInitialGuess)
		.add_property("AbsoluteTolerance",	&daeVariableType::GetAbsoluteTolerance, &daeVariableType::SetAbsoluteTolerance)
		
		.def("__str__",						&daepython::daeVariableType_str)
		;

	class_<daeObject, boost::noncopyable>("daeObject", no_init)
		.add_property("Name",			&daeObject::GetName,			&daeObject::SetName)
		.add_property("CanonicalName",	&daeObject::GetCanonicalName,	&daeObject::SetCanonicalName)
		.add_property("Description",	&daeObject::GetDescription,		&daeObject::SetDescription)
		;

	class_<daeDomainIndex>("daeDomainIndex")
		.def(init<size_t>())
		.def(init<daeDomainIndex>())
		
		.def_readonly("Type",	&daeDomainIndex::m_eType)
		.def_readonly("Index",	&daeDomainIndex::m_nIndex)
		.def_readonly("DEDI",	&daeDomainIndex::m_pDEDI)
		;

	class_<daeIndexRange>("daeIndexRange")
		.def(init<daeDomain*>())
		.def ("__init__", make_constructor(daepython::__init__daeIndexRange))
		.def(init<daeDomain*, size_t, size_t,size_t>())
		
		.add_property("NoPoints",	&daeIndexRange::GetNoPoints)

		.def_readonly("Domain",		&daeIndexRange::m_pDomain)
		.def_readonly("Type",		&daeIndexRange::m_eType) 
		.def_readonly("StartIndex",	&daeIndexRange::m_iStartIndex)
		.def_readonly("EndIndex",	&daeIndexRange::m_iEndIndex)
		.def_readonly("Step",		&daeIndexRange::m_iStride)
		;

	class_<daeArrayRange>("daeArrayRange")
		.def(init<size_t>())
		.def(init<daeDistributedEquationDomainInfo*>())
		.def(init<daeIndexRange>())
		
		.add_property("NoPoints",	&daeArrayRange::GetNoPoints)

		.def_readonly("Type",	&daeArrayRange::m_eType)
		.def_readonly("Range",	&daeArrayRange::m_Range)
		.def_readonly("Index",	&daeArrayRange::m_nIndex)
		.def_readonly("DEDI",	&daeArrayRange::m_pDEDI)
		;
 
	class_<daeDEDI, bases<daeObject>, boost::noncopyable>("daeDEDI", no_init)
		.def("__str__",		&daepython::daeDEDI_str)
		.def("__call__",	&daeDEDI::operator())
		;

	class_<daeDomain, bases<daeObject> >("daeDomain")
		.def(init<string, daeModel*, optional<string> >())
		.def(init<string, daePort*, optional<string> >())

		.add_property("Type",					&daeDomain::GetType)
		.add_property("NumberOfIntervals",		&daeDomain::GetNumberOfIntervals)
		.add_property("NumberOfPoints",			&daeDomain::GetNumberOfPoints)
		.add_property("DiscretizationMethod",	&daeDomain::GetDiscretizationMethod)
		.add_property("DiscretizationOrder",	&daeDomain::GetDiscretizationOrder)
		.add_property("LowerBound",				&daeDomain::GetLowerBound)
		.add_property("UpperBound",				&daeDomain::GetUpperBound)
		.add_property("Points",					&daepython::GetDomainPoints, &daepython::SetDomainPoints)
 
		.def("__str__",							&daepython::daeDomain_str)
		.def("CreateArray",						&daeDomain::CreateArray)
		.def("CreateDistributed",				&daeDomain::CreateDistributed)
		.def("GetNumPyArray",                   &daepython::GetNumPyArrayDomain)
		.def("__getitem__",						&daeDomain::operator [])
		.def("__call__",						&daepython::FunctionCallDomain1)
		.def("__call__",						&daepython::FunctionCallDomain2) 
		.def("__call__",						&daepython::FunctionCallDomain3)
		//.def("array",							&daepython::DomainArray1)
		//.def("array",							&daepython::DomainArray2)
		.def("GetPoint",						&daeDomain::GetPoint)
		;

	class_<daepython::daeParameterWrapper, bases<daeObject>, boost::noncopyable>("daeParameter")
		.def(init<string, daeeParameterType, daePort*, optional<string> >())
		.def(init<string, daeeParameterType, daeModel*, optional<string> >())

		.add_property("Type",		&daeParameter::GetParameterType)
		.add_property("Domains",	&daepython::daeParameterWrapper::GetDomains)

		.def("__str__",				&daepython::daeParameter_str)
		.def("DistributeOnDomain",	&daeParameter::DistributeOnDomain)
		
		.def("GetValue", &daepython::daeParameterWrapper::GetParameterValue0)
		.def("GetValue", &daepython::daeParameterWrapper::GetParameterValue1)
		.def("GetValue", &daepython::daeParameterWrapper::GetParameterValue2)
		.def("GetValue", &daepython::daeParameterWrapper::GetParameterValue3)
		.def("GetValue", &daepython::daeParameterWrapper::GetParameterValue4)
		.def("GetValue", &daepython::daeParameterWrapper::GetParameterValue5)
		.def("GetValue", &daepython::daeParameterWrapper::GetParameterValue6)
		.def("GetValue", &daepython::daeParameterWrapper::GetParameterValue7)
		.def("GetValue", &daepython::daeParameterWrapper::GetParameterValue8)

		.def("SetValue", &daepython::SetParameterValue0)
		.def("SetValue", &daepython::SetParameterValue1)
		.def("SetValue", &daepython::SetParameterValue2)
		.def("SetValue", &daepython::SetParameterValue3)
		.def("SetValue", &daepython::SetParameterValue4)
		.def("SetValue", &daepython::SetParameterValue5)
		.def("SetValue", &daepython::SetParameterValue6)
		.def("SetValue", &daepython::SetParameterValue7)
		.def("SetValue", &daepython::SetParameterValue8)
		
		.def("__call__", &daepython::FunctionCallParameter0)
		.def("__call__", &daepython::FunctionCallParameter1)
		.def("__call__", &daepython::FunctionCallParameter2)
		.def("__call__", &daepython::FunctionCallParameter3)
		.def("__call__", &daepython::FunctionCallParameter4)
		.def("__call__", &daepython::FunctionCallParameter5)
		.def("__call__", &daepython::FunctionCallParameter6)
		.def("__call__", &daepython::FunctionCallParameter7)
		.def("__call__", &daepython::FunctionCallParameter8)
		
		.def("GetNumPyArray", &daepython::GetNumPyArrayParameter)

		.def("array", &daepython::ParameterArray1)
		.def("array", &daepython::ParameterArray2)
		.def("array", &daepython::ParameterArray3)
		.def("array", &daepython::ParameterArray4)
		.def("array", &daepython::ParameterArray5)
		.def("array", &daepython::ParameterArray6)
		.def("array", &daepython::ParameterArray7)
		.def("array", &daepython::ParameterArray8)
		;
 
	class_<daepython::daeVariableWrapper, bases<daeObject>, boost::noncopyable>("daeVariable")
		.def(init<string, const daeVariableType&, daeModel*, optional<string> >())
		.def(init<string, const daeVariableType&, daePort*, optional<string> >())
		.add_property("Domains",		&daepython::daeVariableWrapper::GetDomains)
		.add_property("VariableType",	make_function(&daepython::daeVariableWrapper::GetVariableType, return_internal_reference<>()) )
		.add_property("ReportingOn",	&daeVariable::GetReportingOn,	&daeVariable::SetReportingOn)

		.def("__str__",					&daepython::daeVariable_str)
		.def("DistributeOnDomain",		&daeVariable::DistributeOnDomain)
		
		.def("__call__", &daepython::VariableFunctionCall0)
		.def("__call__", &daepython::VariableFunctionCall1)
		.def("__call__", &daepython::VariableFunctionCall2)
		.def("__call__", &daepython::VariableFunctionCall3)
		.def("__call__", &daepython::VariableFunctionCall4)
		.def("__call__", &daepython::VariableFunctionCall5)
		.def("__call__", &daepython::VariableFunctionCall6)
		.def("__call__", &daepython::VariableFunctionCall7)
		.def("__call__", &daepython::VariableFunctionCall8)

		.def("GetNumPyArray", &daepython::GetNumPyArrayVariable)
		
		.def("SetValue", &daepython::SetVariableValue0)
		.def("SetValue", &daepython::SetVariableValue1)
		.def("SetValue", &daepython::SetVariableValue2)
		.def("SetValue", &daepython::SetVariableValue3)
		.def("SetValue", &daepython::SetVariableValue4)
		.def("SetValue", &daepython::SetVariableValue5)
		.def("SetValue", &daepython::SetVariableValue6)
		.def("SetValue", &daepython::SetVariableValue7)
		.def("SetValue", &daepython::SetVariableValue8)
		
		.def("GetValue", &daepython::daeVariableWrapper::GetVariableValue0)
		.def("GetValue", &daepython::daeVariableWrapper::GetVariableValue1)
		.def("GetValue", &daepython::daeVariableWrapper::GetVariableValue2)
		.def("GetValue", &daepython::daeVariableWrapper::GetVariableValue3)
		.def("GetValue", &daepython::daeVariableWrapper::GetVariableValue4)
		.def("GetValue", &daepython::daeVariableWrapper::GetVariableValue5)
		.def("GetValue", &daepython::daeVariableWrapper::GetVariableValue6)
		.def("GetValue", &daepython::daeVariableWrapper::GetVariableValue7)
		.def("GetValue", &daepython::daeVariableWrapper::GetVariableValue8)

		.def("AssignValue", &daepython::AssignValue0)
		.def("AssignValue", &daepython::AssignValue1)
		.def("AssignValue", &daepython::AssignValue2)
		.def("AssignValue", &daepython::AssignValue3)
		.def("AssignValue", &daepython::AssignValue4)
		.def("AssignValue", &daepython::AssignValue5)
		.def("AssignValue", &daepython::AssignValue6)
		.def("AssignValue", &daepython::AssignValue7)
		.def("AssignValue", &daepython::AssignValue8)

		.def("ReAssignValue", &daepython::ReAssignValue0)
		.def("ReAssignValue", &daepython::ReAssignValue1)
		.def("ReAssignValue", &daepython::ReAssignValue2)
		.def("ReAssignValue", &daepython::ReAssignValue3)
		.def("ReAssignValue", &daepython::ReAssignValue4)
		.def("ReAssignValue", &daepython::ReAssignValue5)
		.def("ReAssignValue", &daepython::ReAssignValue6)
		.def("ReAssignValue", &daepython::ReAssignValue7)
		.def("ReAssignValue", &daepython::ReAssignValue8)

		.def("SetInitialGuess", &daepython::SetInitialGuess0)
		.def("SetInitialGuess", &daepython::SetInitialGuess1)
		.def("SetInitialGuess", &daepython::SetInitialGuess2)
		.def("SetInitialGuess", &daepython::SetInitialGuess3)
		.def("SetInitialGuess", &daepython::SetInitialGuess4)
		.def("SetInitialGuess", &daepython::SetInitialGuess5)
		.def("SetInitialGuess", &daepython::SetInitialGuess6)
		.def("SetInitialGuess", &daepython::SetInitialGuess7)
		.def("SetInitialGuess", &daepython::SetInitialGuess8)
 
		.def("SetInitialCondition", &daepython::SetInitialCondition0)
		.def("SetInitialCondition", &daepython::SetInitialCondition1)
		.def("SetInitialCondition", &daepython::SetInitialCondition2)
		.def("SetInitialCondition", &daepython::SetInitialCondition3)
		.def("SetInitialCondition", &daepython::SetInitialCondition4)
		.def("SetInitialCondition", &daepython::SetInitialCondition5)
		.def("SetInitialCondition", &daepython::SetInitialCondition6)
		.def("SetInitialCondition", &daepython::SetInitialCondition7)
		.def("SetInitialCondition", &daepython::SetInitialCondition8)

		.def("ReSetInitialCondition", &daepython::ReSetInitialCondition0)
		.def("ReSetInitialCondition", &daepython::ReSetInitialCondition1)
		.def("ReSetInitialCondition", &daepython::ReSetInitialCondition2)
		.def("ReSetInitialCondition", &daepython::ReSetInitialCondition3)
		.def("ReSetInitialCondition", &daepython::ReSetInitialCondition4)
		.def("ReSetInitialCondition", &daepython::ReSetInitialCondition5)
		.def("ReSetInitialCondition", &daepython::ReSetInitialCondition6)
		.def("ReSetInitialCondition", &daepython::ReSetInitialCondition7)
		.def("ReSetInitialCondition", &daepython::ReSetInitialCondition8)

		.def("SetAbsoluteTolerances", &daeVariable::SetAbsoluteTolerances)
		.def("SetInitialGuesses",     &daeVariable::SetInitialGuesses)

		.def("dt", &daepython::Get_dt0)
		.def("dt", &daepython::Get_dt1)
		.def("dt", &daepython::Get_dt2)
		.def("dt", &daepython::Get_dt3)
		.def("dt", &daepython::Get_dt4)
		.def("dt", &daepython::Get_dt5)
		.def("dt", &daepython::Get_dt6)
		.def("dt", &daepython::Get_dt7)
		.def("dt", &daepython::Get_dt8)
		
		.def("d", &daepython::Get_d1)
		.def("d", &daepython::Get_d2)
		.def("d", &daepython::Get_d3)
		.def("d", &daepython::Get_d4)
		.def("d", &daepython::Get_d5)
		.def("d", &daepython::Get_d6)
		.def("d", &daepython::Get_d7)
		.def("d", &daepython::Get_d8)
		
		.def("d2", &daepython::Get_d21)
		.def("d2", &daepython::Get_d22)
		.def("d2", &daepython::Get_d23)
		.def("d2", &daepython::Get_d24)
		.def("d2", &daepython::Get_d25)
		.def("d2", &daepython::Get_d26)
		.def("d2", &daepython::Get_d27)
		.def("d2", &daepython::Get_d28)
		
		.def("array", &daepython::VariableArray1)
		.def("array", &daepython::VariableArray2)
		.def("array", &daepython::VariableArray3)
		.def("array", &daepython::VariableArray4)
		.def("array", &daepython::VariableArray5)
		.def("array", &daepython::VariableArray6)
		.def("array", &daepython::VariableArray7)
		.def("array", &daepython::VariableArray8)
		
		.def("dt_array", &daepython::Get_dt_array1)
		.def("dt_array", &daepython::Get_dt_array2)
		.def("dt_array", &daepython::Get_dt_array3)
		.def("dt_array", &daepython::Get_dt_array4)
		.def("dt_array", &daepython::Get_dt_array5)
		.def("dt_array", &daepython::Get_dt_array6)
		.def("dt_array", &daepython::Get_dt_array7)
		.def("dt_array", &daepython::Get_dt_array8)
		
		.def("d_array", &daepython::Get_d_array1)
		.def("d_array", &daepython::Get_d_array2)
		.def("d_array", &daepython::Get_d_array3)
		.def("d_array", &daepython::Get_d_array4)
		.def("d_array", &daepython::Get_d_array5)
		.def("d_array", &daepython::Get_d_array6)
		.def("d_array", &daepython::Get_d_array7)
		.def("d_array", &daepython::Get_d_array8)
		
		.def("d2_array", &daepython::Get_d2_array1)
		.def("d2_array", &daepython::Get_d2_array2)
		.def("d2_array", &daepython::Get_d2_array3)
		.def("d2_array", &daepython::Get_d2_array4)
		.def("d2_array", &daepython::Get_d2_array5)
		.def("d2_array", &daepython::Get_d2_array6)
		.def("d2_array", &daepython::Get_d2_array7)
		.def("d2_array", &daepython::Get_d2_array8)
		;
 
	class_<daepython::daePortWrapper, bases<daeObject>, boost::noncopyable>("daePort")
		.def(init<string, daeePortType, daeModel*, optional<string> >())
		
		.add_property("Type",			&daePort::GetType)
		.add_property("Domains",		&daepython::daePortWrapper::GetDomains)
		.add_property("Parameters",		&daepython::daePortWrapper::GetParameters)
		.add_property("Variables",		&daepython::daePortWrapper::GetVariables)
		
		.def("__str__",					&daepython::daePort_str)
		.def("SetReportingOn",			&daePort::SetReportingOn)
		;
 
	class_<daepython::daeModelWrapper, bases<daeObject>, boost::noncopyable>("daeModel")
		.def(init<string, optional<daeModel*, string> >())
 	
		.add_property("Domains",				&daepython::daeModelWrapper::GetDomains)
		.add_property("Parameters",				&daepython::daeModelWrapper::GetParameters)
		.add_property("Variables",				&daepython::daeModelWrapper::GetVariables)
		.add_property("Equations",				&daepython::daeModelWrapper::GetEquations)
		.add_property("Ports",					&daepython::daeModelWrapper::GetPorts)
		.add_property("Models",					&daepython::daeModelWrapper::GetChildModels)
		.add_property("PortArrays",				&daepython::daeModelWrapper::GetPortArrays)
		.add_property("ModelArrays",			&daepython::daeModelWrapper::GetChildModelArrays)
		.add_property("STNs",					&daepython::daeModelWrapper::GetSTNs) 
		.add_property("InitialConditionMode",	&daeModel::GetInitialConditionMode, &daeModel::SetInitialConditionMode)
 
		.def("__str__",          &daepython::daeModel_str)
		.def("CreateEquation",   &daepython::daeModelWrapper::CreateEquation1, return_internal_reference<>())
		.def("CreateEquation",   &daepython::daeModelWrapper::CreateEquation2, return_internal_reference<>())
		.def("DeclareEquations", &daeModel::DeclareEquations,  &daepython::daeModelWrapper::def_DeclareEquations)
		.def("ConnectPorts",     &daepython::daeModelWrapper::ConnectPorts)
		.def("SetReportingOn",	 &daeModel::SetReportingOn)
		
		.def("sum",				&daeModel::sum)
		.def("product",         &daeModel::product)
		.def("integral",		&daeModel::integral)
		.def("min",				&daeModel::min) 
		.def("max",				&daeModel::max) 
		.def("average",			&daeModel::average)
		.def("dt",				&daeModel::dt)
		.def("d",				&daeModel::d)

		.def("IF",				&daepython::daeModelWrapper::IF, ( arg("EventTolerance") = 0.0 ) )
		.def("ELSE_IF",			&daepython::daeModelWrapper::ELSE_IF, ( arg("EventTolerance") = 0.0 ) )
		.def("ELSE",			&daepython::daeModelWrapper::ELSE)
		.def("END_IF",			&daepython::daeModelWrapper::END_IF)
 
		.def("STN",				&daepython::daeModelWrapper::STN, return_internal_reference<>())
		.def("STATE",			&daepython::daeModelWrapper::STATE, return_internal_reference<>())
		.def("END_STN",			&daepython::daeModelWrapper::END_STN)
		.def("SWITCH_TO",		&daepython::daeModelWrapper::SWITCH_TO, ( arg("EventTolerance") = 0.0 ) )

		.def("SaveModelReport",			&daeModel::SaveModelReport)
		.def("SaveRuntimeModelReport",	&daeModel::SaveRuntimeModelReport)	
		; 
        
	class_<daeEquation, bases<daeObject>, boost::noncopyable>("daeEquation")
		.def("__str__",				&daepython::daeEquation_str)
		.def("DistributeOnDomain",	&daepython::DistributeOnDomain1, return_internal_reference<>())
		.def("DistributeOnDomain",	&daepython::DistributeOnDomain2, return_internal_reference<>())
		.add_property("Residual",	&daeEquation::GetResidual,  &daeEquation::SetResidual)
		;
 
	class_<daepython::daeStateWrapper, bases<daeObject>, boost::noncopyable>("daeState")
		.add_property("NumberOfStateTransitions",	&daeState::GetNumberOfStateTransitions)
		.add_property("NumberOfNestedSTNs",			&daeState::GetNumberOfSTNs)
		.add_property("NumberOfEquations",			&daeState::GetNumberOfEquations)
		.add_property("Equations",					&daepython::daeStateWrapper::GetEquations)
		.add_property("StateTransitions",			&daepython::daeStateWrapper::GetStateTransitions)
		.add_property("NestedSTNs",					&daepython::daeStateWrapper::GetNestedSTNs)
		;

	class_<daepython::daeStateTransitionWrapper, bases<daeObject>, boost::noncopyable>("daeStateTransition")
		.add_property("StateFrom",	make_function(&daepython::daeStateTransitionWrapper::GetStateFrom, return_internal_reference<>()))
		.add_property("StateTo",	make_function(&daepython::daeStateTransitionWrapper::GetStateTo, return_internal_reference<>()))
		.add_property("Condition",	&daepython::daeStateTransitionWrapper::GetCondition)
		;

    class_<daepython::daeSTNWrapper, bases<daeObject>, boost::noncopyable>("daeSTN")
        .add_property("NumberOfStates",		&daeSTN::GetNumberOfStates)
        .add_property("States",				&daepython::daeSTNWrapper::GetStates)
        .add_property("ActiveState",		make_function(&daepython::daeSTNWrapper::GetActState, return_internal_reference<>()))
        .add_property("ParentState",		make_function(&daeSTN::GetParentState, return_internal_reference<>()))

        .def("SetActiveState",				&daepython::daeSTNWrapper::SetActState)
        ;

	class_<daeObjectiveFunction, bases<daeObject>, boost::noncopyable>("daeObjectiveFunction", no_init)
		.add_property("Residual",	&daeObjectiveFunction::GetResidual,  &daeObjectiveFunction::SetResidual)
		;

	class_<daeOptimizationConstraint, bases<daeObject>, boost::noncopyable>("daeOptimizationConstraint", no_init)
		.add_property("Residual",	&daeOptimizationConstraint::GetResidual,  &daeOptimizationConstraint::SetResidual)
		;

	class_<daepython::daeIFWrapper, bases<daeSTN>, boost::noncopyable>("daeIF")
		;
    
/**************************************************************
	daeLog
***************************************************************/
	class_<daepython::daeLogWrapper, boost::noncopyable>("daeLog_t", no_init)
		.def("Message", pure_virtual(&daeLog_t::Message))
		;
	
	class_<daepython::daeFileLogWrapper, bases<daeLog_t>, boost::noncopyable>("daeFileLog", init<string>())
		.def("Message",	&daeLog_t::Message, &daepython::daeFileLogWrapper::def_Message)
		;
	
	class_<daepython::daeStdOutLogWrapper, bases<daeLog_t>, boost::noncopyable>("daeStdOutLog")
		.def("Message",	&daeLog_t::Message, &daepython::daeStdOutLogWrapper::def_Message)
		;

	class_<daepython::daeTCPIPLogWrapper, bases<daeLog_t>, boost::noncopyable>("daeTCPIPLog", init<string, int>())
		.def("Message",	&daeLog_t::Message, &daepython::daeTCPIPLogWrapper::def_Message)
		;

	class_<daepython::daeTCPIPLogServerWrapper, boost::noncopyable>("daeTCPIPLogServer", init<int>())
		.def("MessageReceived",	pure_virtual(&daeTCPIPLogServer::MessageReceived))
		;

}
