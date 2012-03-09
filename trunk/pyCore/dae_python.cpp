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
		.value("eICTUnknown",					dae::core::eICTUnknown)
		.value("eAlgebraicValuesProvided",		dae::core::eAlgebraicValuesProvided)
		.value("eDifferentialValuesProvided",	dae::core::eDifferentialValuesProvided)
		.value("eQuasySteadyState",				dae::core::eQuasySteadyState)
		.export_values()
	;

	enum_<daeeDomainIndexType>("daeeDomainIndexType")
		.value("eDITUnknown",					dae::core::eDITUnknown)
		.value("eConstantIndex",				dae::core::eConstantIndex)
		.value("eDomainIterator",				dae::core::eDomainIterator)
		.value("eIncrementedDomainIterator",	dae::core::eIncrementedDomainIterator)
		.export_values()
	;

	enum_<daeeRangeType>("daeeRangeType")
		.value("eRaTUnknown",			dae::core::eRaTUnknown)
		.value("eRangeDomainIndex",		dae::core::eRangeDomainIndex)
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

	enum_<daeeOptimizationVariableType>("daeeOptimizationVariableType")
		.value("eIntegerVariable",		dae::core::eIntegerVariable)
		.value("eBinaryVariable",		dae::core::eBinaryVariable)
		.value("eContinuousVariable",	dae::core::eContinuousVariable)
		.export_values()
	;

	enum_<daeeModelLanguage>("daeeModelLanguage")
		.value("eMLNone",	dae::core::eMLNone)
		.value("eCDAE",		dae::core::eCDAE)
		.value("ePYDAE",	dae::core::ePYDAE)
		.export_values()
	;

	enum_<daeeConstraintType>("daeeConstraintType")
		.value("eInequalityConstraint",	dae::core::eInequalityConstraint)
		.value("eEqualityConstraint",	dae::core::eEqualityConstraint)
		.export_values()
	;

/**************************************************************
	Global functions
***************************************************************/
	def("daeGetConfig",		&daepython::daeGetConfig); 
	def("daeVersion",		&dae::daeVersion);
	def("daeVersionMajor",  &dae::daeVersionMajor);
	def("daeVersionMinor",  &dae::daeVersionMinor);
	def("daeVersionBuild",  &dae::daeVersionBuild);

/**************************************************************
	Classes
***************************************************************/
	class_<daeVariableWrapper>("daeVariableWrapper", no_init)
		.def(init<daeVariable&, optional<string> >())
		.def(init<adouble&, optional<string> >())

		.def_readwrite("Name", &daeVariableWrapper::m_strName)  
		.add_property("Value", &daeVariableWrapper::GetValue, &daeVariableWrapper::SetValue)
	;

	class_<daeConfig>("daeConfig")
		.def("GetBoolean",	   &daepython::GetBoolean1)
		.def("GetFloat",	   &daepython::GetFloat1)
		.def("GetInteger",	   &daepython::GetInteger1)
		.def("GetString",	   &daepython::GetString1)
	  
		.def("GetBoolean",	   &daepython::GetBoolean)
		.def("GetFloat",	   &daepython::GetFloat)
		.def("GetInteger",	   &daepython::GetInteger)
		.def("GetString",	   &daepython::GetString)

		.def("SetBoolean",	   &daepython::SetBoolean)
		.def("SetFloat",	   &daepython::SetFloat)
		.def("SetInteger",	   &daepython::SetInteger)
		.def("SetString",	   &daepython::SetString) 

	    .def("Reload",		   &daeConfig::Reload)
        
        .def("has_key",		   &daepython::daeConfig_has_key)
	    .def("__contains__",   &daepython::daeConfig__contains__)
        .def("__getitem__",	   &daepython::daeConfig__getitem__)
        .def("__setitem__",	   &daepython::daeConfig__setitem__)
	        
        .def("__str__",		   &daepython::daeConfig__str__)
	;
/*
  .def("GetBoolean",	   &daepython::GetBoolean1)
  .def("GetFloat",	   &daepython::GetFloat1)
  .def("GetInteger",	   &daepython::GetInteger1)
  .def("GetString",	   &daepython::GetString1)

  .def("GetBoolean",	   &daepython::GetBoolean)
  .def("GetFloat",	   &daepython::GetFloat)
  .def("GetInteger",	   &daepython::GetInteger)
  .def("GetString",	   &daepython::GetString)

  .def("PutBoolean",	   &daepython::PutBoolean)
  .def("PutFloat",	   &daepython::PutFloat)
  .def("PutInteger",	   &daepython::PutInteger)
  .def("PutString",	   &daepython::PutString)
*/  
	class_<daeCondition>("daeCondition")
		.add_property("EventTolerance",	&daeCondition::GetEventTolerance, &daeCondition::SetEventTolerance)
		//.def(!self)
		.def(self | self)
		.def(self & self)
	;

	class_<adouble>("adouble")
		.add_property("Value",		&adouble::getValue,      &adouble::setValue)
		.add_property("Derivative",	&adouble::getDerivative, &adouble::setDerivative) 

        .def("__repr__",     &daepython::adouble_repr)

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

	def("Sinh",  &daepython::ad_sinh);
	def("Cosh",  &daepython::ad_cosh);
	def("Tanh",  &daepython::ad_tanh);
	def("ASinh", &daepython::ad_asinh);
	def("ACosh", &daepython::ad_acosh);
	def("ATanh", &daepython::ad_atanh);
	def("ATan2", &daepython::ad_atan2);

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

	def("Time",			&Time);
	def("Constant",		&daepython::ad_Constant_c);
	def("Constant",		&daepython::ad_Constant_q);
	def("Vector",		&daepython::adarr_Vector);
	
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
		.def(init<string, unit, real_t, real_t, real_t, real_t>())

		.add_property("Name",				&daeVariableType::GetName,				&daeVariableType::SetName)
		.add_property("Units",				&daeVariableType::GetUnits,				&daeVariableType::SetUnits)
		.add_property("LowerBound",			&daeVariableType::GetLowerBound,		&daeVariableType::SetLowerBound)
		.add_property("UpperBound",			&daeVariableType::GetUpperBound,		&daeVariableType::SetUpperBound)
		.add_property("InitialGuess",		&daeVariableType::GetInitialGuess,		&daeVariableType::SetInitialGuess)
		.add_property("AbsoluteTolerance",	&daeVariableType::GetAbsoluteTolerance, &daeVariableType::SetAbsoluteTolerance)

		.def("__str__",						&daepython::daeVariableType_str)
		;

    class_<daeObject, boost::noncopyable>("daeObject", no_init)
        .add_property("Name",           &daeObject::GetName,        &daeObject::SetName)
        .add_property("Description",    &daeObject::GetDescription, &daeObject::SetDescription)
        .add_property("CanonicalName",  &daeObject::GetCanonicalName)

        .def("GetNameRelativeToParentModel",            &daeObject::GetNameRelativeToParentModel)
        .def("GetStrippedName",                         &daeObject::GetStrippedName)
        .def("GetStrippedNameRelativeToParentModel",    &daeObject::GetStrippedNameRelativeToParentModel)
        ;
	def("daeGetRelativeName",            &daepython::daeGetRelativeName_1); 
	def("daeGetRelativeName",            &daepython::daeGetRelativeName_2);
	def("daeGetStrippedRelativeName",    &daeGetStrippedRelativeName);

	class_<daeDomainIndex>("daeDomainIndex")
		.def(init<size_t>())
		.def(init<daeDistributedEquationDomainInfo*>())
		.def(init<daeDistributedEquationDomainInfo*, int>())
		.def(init<daeDomainIndex>())

		.def_readonly("Type",		&daeDomainIndex::m_eType)
		.def_readonly("Index",		&daeDomainIndex::m_nIndex)
		.def_readonly("DEDI",		&daeDomainIndex::m_pDEDI)
		.def_readonly("Increment",	&daeDomainIndex::m_iIncrement)
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
		.def(init<daeDomainIndex>())
		.def(init<daeIndexRange>())

		.add_property("NoPoints",		&daeArrayRange::GetNoPoints)

		.def_readonly("Type",			&daeArrayRange::m_eType)
		.def_readonly("Range",			&daeArrayRange::m_Range)
		.def_readonly("DomainIndex",	&daeArrayRange::m_domainIndex)
		;

	class_<daeDEDI, bases<daeObject>, boost::noncopyable>("daeDEDI", no_init)
		.def("__str__",		&daepython::daeDEDI_str)
		.def("__call__",	&daeDEDI::operator())
		.def(self + size_t())
		.def(self - size_t())  
		;

	class_<daeDomain, bases<daeObject> , boost::noncopyable>("daeDomain")
		.def(init<string, daeModel*, const unit&, optional<string> >())
		.def(init<string, daePort*, const unit&, optional<string> >())

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
		; 

	class_<daepython::daeParameterWrapper, bases<daeObject>, boost::noncopyable>("daeParameter")
		.def(init<string, const unit&, daePort*, optional<string, boost::python::list> >())
		.def(init<string, const unit&, daeModel*, optional<string, boost::python::list> >())

		.add_property("Units",			&daeParameter::GetUnits)
		.add_property("Domains",		&daepython::daeParameterWrapper::GetDomains)
        .add_property("ReportingOn",	&daeParameter::GetReportingOn,	&daeParameter::SetReportingOn)

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

	    .def("GetQuantity", &daepython::daeParameterWrapper::GetParameterQuantity0)
		.def("GetQuantity", &daepython::daeParameterWrapper::GetParameterQuantity1)
		.def("GetQuantity", &daepython::daeParameterWrapper::GetParameterQuantity2)
		.def("GetQuantity", &daepython::daeParameterWrapper::GetParameterQuantity3)
		.def("GetQuantity", &daepython::daeParameterWrapper::GetParameterQuantity4)
		.def("GetQuantity", &daepython::daeParameterWrapper::GetParameterQuantity5)
		.def("GetQuantity", &daepython::daeParameterWrapper::GetParameterQuantity6)
		.def("GetQuantity", &daepython::daeParameterWrapper::GetParameterQuantity7)
		.def("GetQuantity", &daepython::daeParameterWrapper::GetParameterQuantity8)

		.def("SetValue", &daepython::SetParameterQuantity0)  
		.def("SetValue", &daepython::SetParameterQuantity1)
		.def("SetValue", &daepython::SetParameterQuantity2)
		.def("SetValue", &daepython::SetParameterQuantity3)
		.def("SetValue", &daepython::SetParameterQuantity4)
		.def("SetValue", &daepython::SetParameterQuantity5)
		.def("SetValue", &daepython::SetParameterQuantity6)
		.def("SetValue", &daepython::SetParameterQuantity7)
		.def("SetValue", &daepython::SetParameterQuantity8)

		.def("SetValues", &daepython::SetParameterValues)
		.def("SetValues", &daepython::qSetParameterValues)
	        
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

	class_<daepython::daeVariable_Wrapper, bases<daeObject>, boost::noncopyable>("daeVariable")
		.def(init<string, const daeVariableType&, daeModel*, optional<string, boost::python::list> >())
		.def(init<string, const daeVariableType&, daePort*, optional<string, boost::python::list> >())
		.add_property("Domains",		&daepython::daeVariable_Wrapper::GetDomains)
		.add_property("VariableType",	make_function(&daepython::daeVariable_Wrapper::GetVariableType, return_internal_reference<>()) )
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

		.def("GetValue", &daepython::daeVariable_Wrapper::GetVariableValue0)
		.def("GetValue", &daepython::daeVariable_Wrapper::GetVariableValue1)
		.def("GetValue", &daepython::daeVariable_Wrapper::GetVariableValue2)
		.def("GetValue", &daepython::daeVariable_Wrapper::GetVariableValue3)
		.def("GetValue", &daepython::daeVariable_Wrapper::GetVariableValue4)
		.def("GetValue", &daepython::daeVariable_Wrapper::GetVariableValue5)
		.def("GetValue", &daepython::daeVariable_Wrapper::GetVariableValue6)
		.def("GetValue", &daepython::daeVariable_Wrapper::GetVariableValue7)
		.def("GetValue", &daepython::daeVariable_Wrapper::GetVariableValue8)

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

        .def("SetValue", &daepython::qSetVariableValue0)
        .def("SetValue", &daepython::qSetVariableValue1)
        .def("SetValue", &daepython::qSetVariableValue2)
        .def("SetValue", &daepython::qSetVariableValue3)
        .def("SetValue", &daepython::qSetVariableValue4)
        .def("SetValue", &daepython::qSetVariableValue5)
        .def("SetValue", &daepython::qSetVariableValue6)
        .def("SetValue", &daepython::qSetVariableValue7)
        .def("SetValue", &daepython::qSetVariableValue8)

        .def("GetQuantity", &daepython::daeVariable_Wrapper::GetVariableQuantity0)
        .def("GetQuantity", &daepython::daeVariable_Wrapper::GetVariableQuantity1)
        .def("GetQuantity", &daepython::daeVariable_Wrapper::GetVariableQuantity2)
        .def("GetQuantity", &daepython::daeVariable_Wrapper::GetVariableQuantity3)
        .def("GetQuantity", &daepython::daeVariable_Wrapper::GetVariableQuantity4)
        .def("GetQuantity", &daepython::daeVariable_Wrapper::GetVariableQuantity5)
        .def("GetQuantity", &daepython::daeVariable_Wrapper::GetVariableQuantity6)
        .def("GetQuantity", &daepython::daeVariable_Wrapper::GetVariableQuantity7)
        .def("GetQuantity", &daepython::daeVariable_Wrapper::GetVariableQuantity8)

        .def("AssignValue", &daepython::qAssignValue0)
        .def("AssignValue", &daepython::qAssignValue1)
        .def("AssignValue", &daepython::qAssignValue2)
        .def("AssignValue", &daepython::qAssignValue3)
        .def("AssignValue", &daepython::qAssignValue4)
        .def("AssignValue", &daepython::qAssignValue5)
        .def("AssignValue", &daepython::qAssignValue6)
        .def("AssignValue", &daepython::qAssignValue7)
        .def("AssignValue", &daepython::qAssignValue8)

        .def("ReAssignValue", &daepython::qReAssignValue0)
        .def("ReAssignValue", &daepython::qReAssignValue1)
        .def("ReAssignValue", &daepython::qReAssignValue2)
        .def("ReAssignValue", &daepython::qReAssignValue3)
        .def("ReAssignValue", &daepython::qReAssignValue4)
        .def("ReAssignValue", &daepython::qReAssignValue5)
        .def("ReAssignValue", &daepython::qReAssignValue6)
        .def("ReAssignValue", &daepython::qReAssignValue7)
        .def("ReAssignValue", &daepython::qReAssignValue8)

        .def("SetInitialGuess",  &daepython::qSetInitialGuess0)
        .def("SetInitialGuess",  &daepython::qSetInitialGuess1)
        .def("SetInitialGuess",  &daepython::qSetInitialGuess2)
        .def("SetInitialGuess",  &daepython::qSetInitialGuess3)
        .def("SetInitialGuess",  &daepython::qSetInitialGuess4)
        .def("SetInitialGuess",  &daepython::qSetInitialGuess5)
        .def("SetInitialGuess",  &daepython::qSetInitialGuess6)
        .def("SetInitialGuess",  &daepython::qSetInitialGuess7)
        .def("SetInitialGuess",  &daepython::qSetInitialGuess8)

        .def("SetInitialCondition", &daepython::qSetInitialCondition0)
        .def("SetInitialCondition", &daepython::qSetInitialCondition1)
        .def("SetInitialCondition", &daepython::qSetInitialCondition2)
        .def("SetInitialCondition", &daepython::qSetInitialCondition3)
        .def("SetInitialCondition", &daepython::qSetInitialCondition4)
        .def("SetInitialCondition", &daepython::qSetInitialCondition5)
        .def("SetInitialCondition", &daepython::qSetInitialCondition6)
        .def("SetInitialCondition", &daepython::qSetInitialCondition7)
        .def("SetInitialCondition", &daepython::qSetInitialCondition8)

        .def("ReSetInitialCondition", &daepython::qReSetInitialCondition0)
        .def("ReSetInitialCondition", &daepython::qReSetInitialCondition1)
        .def("ReSetInitialCondition", &daepython::qReSetInitialCondition2)
        .def("ReSetInitialCondition", &daepython::qReSetInitialCondition3)
        .def("ReSetInitialCondition", &daepython::qReSetInitialCondition4)
        .def("ReSetInitialCondition", &daepython::qReSetInitialCondition5)
        .def("ReSetInitialCondition", &daepython::qReSetInitialCondition6)
        .def("ReSetInitialCondition", &daepython::qReSetInitialCondition7)  
        .def("ReSetInitialCondition", &daepython::qReSetInitialCondition8)

	    .def("AssignValues",			&daepython::AssignValues)
        .def("ReAssignValues",			&daepython::ReAssignValues)
        .def("SetInitialGuesses",		&daepython::SetInitialGuesses)
        .def("SetInitialConditions",	&daepython::SetInitialConditions)
        .def("ReSetInitialConditions",	&daepython::ReSetInitialConditions)
        
	    .def("AssignValues",			&daepython::qAssignValues)
        .def("ReAssignValues",			&daepython::qReAssignValues)
		.def("SetInitialGuesses",		&daepython::qSetInitialGuesses)
        .def("SetInitialConditions",	&daepython::qSetInitialConditions)
        .def("ReSetInitialConditions",	&daepython::qReSetInitialConditions) 
		
	    .def("SetAbsoluteTolerances",	&daeVariable::SetAbsoluteTolerances)

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

	class_<daeModelExportContext>("daeModelExportContext")
		.def_readonly("PythonIndentLevel",	&daeModelExportContext::m_nPythonIndentLevel)
		.def_readonly("ExportDefinition",	&daeModelExportContext::m_bExportDefinition)
		;

	class_<daepython::daePortWrapper, bases<daeObject>, boost::noncopyable>("daePort")
		.def(init<string, daeePortType, daeModel*, optional<string> >())

		.add_property("Type",			&daePort::GetType)
		.add_property("Domains",		&daepython::daePortWrapper::GetDomains)
		.add_property("Parameters",		&daepython::daePortWrapper::GetParameters)
		.add_property("Variables",		&daepython::daePortWrapper::GetVariables)

		.def("__str__",					&daepython::daePort_str)
		.def("SetReportingOn",			&daePort::SetReportingOn)
		.def("Export",					&daePort::Export)
		;

    class_<daeEventPort, bases<daeObject>, boost::noncopyable>("daeEventPort")
        .def(init<string, daeePortType, daeModel*, optional<string> >())

        .add_property("Type",			&daeEventPort::GetType)
        .add_property("EventData",		&daeEventPort::GetEventData)
	    .add_property("RecordEvents",	&daeEventPort::GetRecordEvents, &daeEventPort::SetRecordEvents)
        .add_property("Events",			&daepython::GetEventPortEventsList)  
		
		.def("__call__",		&daeEventPort::operator())
		.def("SendEvent",		&daeEventPort::SendEvent)  
		.def("ReceiveEvent",	&daeEventPort::ReceiveEvent)  
        ;

	class_<daepython::daeActionWrapper, bases<daeObject>, boost::noncopyable>("daeAction")
		.def("Execute",		pure_virtual(&daepython::daeActionWrapper::Execute))
        ;
 
    class_<daeOptimizationVariable_t, boost::noncopyable>("daeOptimizationVariable_t", no_init)
        ;

    class_<daeObjectiveFunction_t, boost::noncopyable>("daeObjectiveFunction_t", no_init)
        ;

    class_<daeOptimizationConstraint_t, boost::noncopyable>("daeOptimizationConstraint_t", no_init)
        ;

    class_<daeMeasuredVariable_t, boost::noncopyable>("daeMeasuredVariable_t", no_init)
        ;

    class_<daeOptimizationVariable, bases<daeOptimizationVariable_t> >("daeOptimizationVariable")
        .add_property("Name",           &daeOptimizationVariable::GetName)
        .add_property("Type",           &daeOptimizationVariable::GetType,          &daeOptimizationVariable::SetType)
        .add_property("Value",          &daeOptimizationVariable::GetValue,         &daeOptimizationVariable::SetValue)
        .add_property("LowerBound",     &daeOptimizationVariable::GetLB,            &daeOptimizationVariable::SetLB)
        .add_property("UpperBound",     &daeOptimizationVariable::GetUB,            &daeOptimizationVariable::SetUB)
        .add_property("StartingPoint",  &daeOptimizationVariable::GetStartingPoint, &daeOptimizationVariable::SetStartingPoint)
        ;

    class_<daeObjectiveFunction, bases<daeObjectiveFunction_t> >("daeObjectiveFunction")  
        .add_property("Name",           &daeObjectiveFunction::GetName)
        .add_property("Residual",       &daeObjectiveFunction::GetResidual,     &daeObjectiveFunction::SetResidual)
        .add_property("Value",          &daeObjectiveFunction::GetValue)
        .add_property("Gradients",      &daepython::GetGradientsObjectiveFunction)
        //.add_property("AbsTolerance", &daeObjectiveFunction::GetAbsTolerance, &daeObjectiveFunction::SetAbsTolerance)
        ;

    class_<daeOptimizationConstraint, bases<daeOptimizationConstraint_t> >("daeOptimizationConstraint")
        .add_property("Name",           &daeOptimizationConstraint::GetName)
        .add_property("Residual",       &daeOptimizationConstraint::GetResidual,     &daeOptimizationConstraint::SetResidual)
        .add_property("Value",          &daeOptimizationConstraint::GetValue)
        .add_property("Type",           &daeOptimizationConstraint::GetType,            &daeOptimizationConstraint::SetType)
        .add_property("Gradients",      &daepython::GetGradientsOptimizationConstraint)
        //.add_property("AbsTolerance", &daeOptimizationConstraint::GetAbsTolerance, &daeOptimizationConstraint::SetAbsTolerance)
        ;

    class_<daeMeasuredVariable, bases<daeMeasuredVariable_t> >("daeMeasuredVariable")
        .add_property("Name",           &daeMeasuredVariable::GetName)
        .add_property("Residual",       &daeMeasuredVariable::GetResidual,     &daeMeasuredVariable::SetResidual)
        .add_property("Value",          &daeMeasuredVariable::GetValue)
        .add_property("Gradients",      &daepython::GetGradientsMeasuredVariable)
        //.add_property("AbsTolerance", &daeMeasuredVariable::GetAbsTolerance, &daeMeasuredVariable::SetAbsTolerance)
        ;

	class_<daepython::daeModelWrapper, bases<daeObject>, boost::noncopyable>("daeModel")
		.def(init<string, optional<daeModel*, string> >())

		.add_property("Domains",				&daepython::daeModelWrapper::GetDomains)
		.add_property("Parameters",				&daepython::daeModelWrapper::GetParameters)
		.add_property("Variables",				&daepython::daeModelWrapper::GetVariables)
		.add_property("Equations",				&daepython::daeModelWrapper::GetEquations)
		.add_property("Ports",					&daepython::daeModelWrapper::GetPorts)
		.add_property("EventPorts",				&daepython::daeModelWrapper::GetEventPorts)
		.add_property("OnEventActions",			&daepython::daeModelWrapper::GetOnEventActions)
		.add_property("Models",					&daepython::daeModelWrapper::GetChildModels)
		.add_property("PortArrays",				&daepython::daeModelWrapper::GetPortArrays)
		.add_property("ModelArrays",			&daepython::daeModelWrapper::GetChildModelArrays)
		.add_property("STNs",					&daepython::daeModelWrapper::GetSTNs)
		.add_property("InitialConditionMode",	&daeModel::GetInitialConditionMode, &daeModel::SetInitialConditionMode)
		.add_property("IsModelDynamic",			&daeModel::IsModelDynamic)

		.def("__str__",          &daepython::daeModel_str)  
		.def("CreateEquation",   &daepython::daeModelWrapper::CreateEquation1, return_internal_reference<>())
		.def("CreateEquation",   &daepython::daeModelWrapper::CreateEquation2, return_internal_reference<>())
        .def("CreateEquation",   &daepython::daeModelWrapper::CreateEquation3, return_internal_reference<>())
		.def("DeclareEquations", &daeModel::DeclareEquations,  &daepython::daeModelWrapper::def_DeclareEquations)
		.def("ConnectPorts",     &daeModel::ConnectPorts)
		.def("ConnectEventPorts",&daeModel::ConnectEventPorts)
		.def("SetReportingOn",	 &daeModel::SetReportingOn)

		.def("sum",				&daeModel::sum) 
		.def("product",         &daeModel::product) 
		.def("integral",		&daeModel::integral)
		.def("min",				&daeModel::min)
		.def("max",				&daeModel::max)
		.def("average",			&daeModel::average)
		.def("dt",				&daeModel::dt)
		.def("d",				&daeModel::d)

		.def("IF",				&daepython::daeModelWrapper::IF, ( boost::python::arg("eventTolerance") = 0.0 ) )
		.def("ELSE_IF",			&daepython::daeModelWrapper::ELSE_IF, ( boost::python::arg("eventTolerance") = 0.0 ) )
		.def("ELSE",			&daepython::daeModelWrapper::ELSE)
		.def("END_IF",			&daepython::daeModelWrapper::END_IF) 

		.def("STN",				&daepython::daeModelWrapper::STN, return_internal_reference<>())
		.def("STATE",			&daepython::daeModelWrapper::STATE, return_internal_reference<>())
		.def("END_STN",			&daepython::daeModelWrapper::END_STN)
		.def("SWITCH_TO",		&daepython::daeModelWrapper::SWITCH_TO, ( boost::python::arg("eventTolerance") = 0.0 ) )
        .def("ON_CONDITION",    &daepython::daeModelWrapper::ON_CONDITION, ( boost::python::arg("switchTo")           = string(),
	                                                                         boost::python::arg("setVariableValues")  = boost::python::list(),
                                                                             boost::python::arg("triggerEvents")      = boost::python::list(),
                                                                             boost::python::arg("userDefinedActions") = boost::python::list(),
                                                                             boost::python::arg("eventTolerance")     = 0.0) )
        .def("ON_EVENT",		&daepython::daeModelWrapper::ON_EVENT, ( boost::python::arg("switchToStates")     = boost::python::list(),
	                                                                     boost::python::arg("setVariableValues")  = boost::python::list(),
																		 boost::python::arg("triggerEvents")      = boost::python::list(),
																		 boost::python::arg("userDefinedActions") = boost::python::list()) )
		
		.def("SaveModelReport",			&daeModel::SaveModelReport)
		.def("SaveRuntimeModelReport",	&daeModel::SaveRuntimeModelReport)
		.def("ExportObjects",			&daepython::daeModelWrapper::ExportObjects)
		.def("Export",					&daeModel::Export)
		;

	class_<daeEquation, bases<daeObject>, boost::noncopyable>("daeEquation")
		.def("__str__",				&daepython::daeEquation_str)
		.def("DistributeOnDomain",	&daepython::DistributeOnDomain1, return_internal_reference<>())
		.def("DistributeOnDomain",	&daepython::DistributeOnDomain2, return_internal_reference<>())
		.add_property("Residual",	&daeEquation::GetResidual, &daeEquation::SetResidual)
        .add_property("Scaling",	&daeEquation::GetScaling,  &daeEquation::SetScaling)
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
		//.add_property("StateFrom",	make_function(&daepython::daeStateTransitionWrapper::GetStateFrom, return_internal_reference<>()))
		//.add_property("StateTo",	make_function(&daepython::daeStateTransitionWrapper::GetStateTo, return_internal_reference<>()))
		.add_property("Condition",	&daepython::daeStateTransitionWrapper::GetCondition)
		;

    class_<daeSTN, bases<daeObject>, boost::noncopyable>("daeSTN")
        .add_property("ActiveState",	&daeSTN::GetActiveState2, &daeSTN::SetActiveState2)
        .add_property("States",			&daepython::GetStatesSTN)
        ;

	class_<daepython::daeIFWrapper, bases<daeSTN>, boost::noncopyable>("daeIF")
		;
	
	class_<daepython::daeScalarExternalFunctionWrapper, boost::noncopyable>("daeScalarExternalFunction", init<const string&, daeModel*, const unit&, boost::python::dict>())
		.def("Calculate",	pure_virtual(&daepython::daeScalarExternalFunctionWrapper::Calculate_))
		.def("__call__",	&daeScalarExternalFunction::operator())
		;
	
	class_<daepython::daeVectorExternalFunctionWrapper, boost::noncopyable>("daeVectorExternalFunction", init<const string&, daeModel*, const unit&, size_t, boost::python::dict>())
		.def("Calculate",	pure_virtual(&daepython::daeVectorExternalFunctionWrapper::Calculate_)) 
		.def("__call__",	&daeVectorExternalFunction::operator())
		;


/**************************************************************
	daeLog
***************************************************************/
	class_<daepython::daeLogWrapper, boost::noncopyable>("daeLog_t", no_init)
        .add_property("Enabled",		&daeLog_t::GetEnabled,		 &daeLog_t::SetEnabled)
		.add_property("PrintProgress",	&daeLog_t::GetPrintProgress, &daeLog_t::SetPrintProgress)
		.add_property("Indent",			&daeLog_t::GetIndent,		 &daeLog_t::SetIndent)
		.add_property("Progress",		&daeLog_t::GetProgress,		 &daeLog_t::SetProgress)
		.add_property("IndentString",	&daeLog_t::GetIndentString)
        .add_property("PercentageDone",	&daeLog_t::GetPercentageDone)
        .add_property("ETA",			&daeLog_t::GetETA)

		.def("Message",			pure_virtual(&daeLog_t::Message))
		.def("JoinMessages",	pure_virtual(&daeLog_t::JoinMessages))
		.def("IncreaseIndent",	pure_virtual(&daeLog_t::IncreaseIndent))
		.def("DecreaseIndent",	pure_virtual(&daeLog_t::DecreaseIndent))
		;

	class_<daepython::daeBaseLogWrapper, bases<daeLog_t>, boost::noncopyable>("daeBaseLog")
		.def("Message",				&daeLog_t::Message,     &daepython::daeBaseLogWrapper::def_Message)
	    .def("SetProgress",			&daeLog_t::SetProgress, &daepython::daeBaseLogWrapper::def_SetProgress)
		.def("IncreaseIndent",		&daeBaseLog::IncreaseIndent)
		.def("DecreaseIndent",		&daeBaseLog::DecreaseIndent)
		;

	class_<daepython::daeFileLogWrapper, bases<daeBaseLog>, boost::noncopyable>("daeFileLog", init<string>())
		.def("Message",			&daeLog_t::Message, &daepython::daeFileLogWrapper::def_Message)
		;

	class_<daepython::daeStdOutLogWrapper, bases<daeBaseLog>, boost::noncopyable>("daeStdOutLog")
		.def("Message",			&daeLog_t::Message, &daepython::daeStdOutLogWrapper::def_Message)
		;

	class_<daepython::daeTCPIPLogWrapper, bases<daeBaseLog>, boost::noncopyable>("daeTCPIPLog", init<string, int>())
		.def("Message",			&daeLog_t::Message, &daepython::daeTCPIPLogWrapper::def_Message)
		;

	class_<daepython::daeTCPIPLogServerWrapper, boost::noncopyable>("daeTCPIPLogServer", init<int>())
		.def("MessageReceived",	pure_virtual(&daeTCPIPLogServer::MessageReceived))
		;

}
