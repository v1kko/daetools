#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <noprefix.h>
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
    
    enum_<daeeUnaryFunctions>("daeeUnaryFunctions")
        .value("eUFUnknown",dae::core::eUFUnknown)
        .value("eSign",     dae::core::eSign)
        .value("eSqrt",     dae::core::eSqrt)
        .value("eExp",      dae::core::eExp)
        .value("eLog",      dae::core::eLog)
        .value("eLn",       dae::core::eLn)
        .value("eAbs",      dae::core::eAbs)
        .value("eSin",      dae::core::eSin)
        .value("eCos",      dae::core::eCos)
        .value("eTan",      dae::core::eTan)
        .value("eArcSin",	dae::core::eArcSin)
        .value("eArcCos",	dae::core::eArcCos)
        .value("eArcTan",	dae::core::eArcTan)
        .value("eCeil",     dae::core::eCeil)
        .value("eFloor",	dae::core::eFloor)
        .export_values()
    ;
    
    enum_<daeeBinaryFunctions>("daeeBinaryFunctions")
        .value("eBFUnknown",	dae::core::eBFUnknown)
        .value("ePlus",         dae::core::ePlus)
        .value("eMinus",        dae::core::eMinus)
        .value("eMulti",        dae::core::eMulti)
        .value("eDivide",       dae::core::eDivide)
        .value("ePower",        dae::core::ePower)
        .value("eMin",          dae::core::eMin)
        .value("eMax",          dae::core::eMax)
        .export_values()
    ;
    
    enum_<daeeSpecialUnaryFunctions>("daeeSpecialUnaryFunctions")
        .value("eSUFUnknown",	dae::core::eSUFUnknown)
        .value("eSum",          dae::core::eSum)
        .value("eProduct",      dae::core::eProduct)
        .value("eMinInArray",   dae::core::eMinInArray)
        .value("eMaxInArray",   dae::core::eMaxInArray)
        .value("eAverage",      dae::core::eAverage)
        .export_values()
    ;

    enum_<daeeLogicalUnaryOperator>("daeeLogicalUnaryOperator")
        .value("eUOUnknown",	dae::core::eUOUnknown)
        .value("eNot",          dae::core::eNot)
        .export_values()
    ;
    
    enum_<daeeLogicalBinaryOperator>("daeeLogicalBinaryOperator")
        .value("eBOUnknown",	dae::core::eBOUnknown)
        .value("eAnd",          dae::core::eAnd)
        .value("eOr",           dae::core::eOr)
        .export_values()
    ;
    
    enum_<daeeConditionType>("daeeConditionType")
        .value("eCTUnknown",	dae::core::eCTUnknown)
        .value("eNotEQ",        dae::core::eNotEQ)
        .value("eEQ",           dae::core::eEQ)
        .value("eGT",           dae::core::eGT)
        .value("eGTEQ",         dae::core::eGTEQ)
        .value("eLT",           dae::core::eLT)
        .value("eLTEQ",         dae::core::eLTEQ)
        .export_values()
    ;

    enum_<daeeActionType>("daeeActionType")
        .value("eUnknownAction",                    dae::core::eUnknownAction)
        .value("eChangeState",                      dae::core::eChangeState)
        .value("eSendEvent",                        dae::core::eSendEvent)
        .value("eReAssignOrReInitializeVariable",   dae::core::eReAssignOrReInitializeVariable)
        .value("eUserDefinedAction",                dae::core::eUserDefinedAction)
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

//	def("daeBoostVersion",				&dae::daeBoostVersion);
//	def("daeBoostVersionMajor",			&dae::daeBoostVersionMajor);
//	def("daeBoostVersionMinor",			&dae::daeBoostVersionMinor);
//	def("daeBoostVersionBuild",			&dae::daeBoostVersionBuild);
	
/**************************************************************
    Global constants
***************************************************************/
    boost::python::scope().attr("cnAlgebraic")    = cnAlgebraic;
    boost::python::scope().attr("cnDifferential") = cnDifferential;
    boost::python::scope().attr("cnAssigned")     = cnAssigned;
    
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
        
    class_<daeNodeSaveAsContext>("daeNodeSaveAsContext")
        .def(init<daeModel*>())
        .def_readwrite("Model",	&daeNodeSaveAsContext::m_pModel)
    ;

    class_<adNode, boost::noncopyable>("adNode", no_init)
        .def("SaveAsLatex", &adNode::SaveAsLatex)
    ;
    
    class_<adNodeArray, boost::noncopyable>("adNodeArray", no_init)
    ;
    
    class_<condNode, boost::noncopyable>("condNode", no_init)
    ;

    class_<adConstantNode, bases<adNode>, boost::noncopyable>("adConstantNode", no_init)
        .def_readonly("Quantity",	&adConstantNode::m_quantity)
    ;

    class_<adTimeNode, bases<adNode>, boost::noncopyable>("adTimeNode", no_init)
    ;

    class_<adEventPortDataNode, bases<adNode>, boost::noncopyable>("adEventPortDataNode", no_init)
    ;
    
    class_<adUnaryNode, bases<adNode>, boost::noncopyable>("adUnaryNode", no_init)
        .def_readonly("Function",	&adUnaryNode::eFunction)
        .add_property("Node",       make_function(&adUnaryNode::getNodeRawPtr, return_internal_reference<>()))
    ;
    
    class_<adBinaryNode, bases<adNode>, boost::noncopyable>("adBinaryNode", no_init)
        .def_readonly("Function",	&adBinaryNode::eFunction)
        .add_property("LNode",      make_function(&adBinaryNode::getLeftRawPtr,  return_internal_reference<>()))
        .add_property("RNode",      make_function(&adBinaryNode::getRightRawPtr, return_internal_reference<>()))
    ;
    
    class_<adSetupDomainIteratorNode, bases<adNode>, boost::noncopyable>("adSetupDomainIteratorNode", no_init)
    ;
    
    class_<adSetupParameterNode, bases<adNode>, boost::noncopyable>("adSetupParameterNode", no_init)
        .add_property("Parameter",      make_function(&daepython::adSetupParameterNode_Parameter, return_internal_reference<>()))
        .add_property("DomainIndexes",	&daepython::adSetupParameterNode_Domains)
    ;
        
    class_<adSetupVariableNode, bases<adNode>, boost::noncopyable>("adSetupVariableNode", no_init)
        .add_property("Variable",       make_function(&daepython::adSetupVariableNode_Variable, return_internal_reference<>()))
        .add_property("DomainIndexes",	&daepython::adSetupVariableNode_Domains)
    ;
    
    class_<adSetupTimeDerivativeNode, bases<adNode>, boost::noncopyable>("adSetupTimeDerivativeNode", no_init)
        .def_readonly("Order",          &adSetupTimeDerivativeNode::m_nDegree)
        .add_property("Variable",       make_function(&daepython::adSetupTimeDerivativeNode_Variable, return_internal_reference<>()))
        .add_property("DomainIndexes",	&daepython::adSetupTimeDerivativeNode_Domains)
    ;
    
    class_<adSetupPartialDerivativeNode, bases<adNode>, boost::noncopyable>("adSetupPartialDerivativeNode", no_init)
        .def_readonly("Order",          &adSetupPartialDerivativeNode::m_nDegree)
        .add_property("Variable",       make_function(&daepython::adSetupPartialDerivativeNode_Variable, return_internal_reference<>()))
        .add_property("Domain",         make_function(&daepython::adSetupPartialDerivativeNode_Domain,   return_internal_reference<>()))
        .add_property("DomainIndexes",	&daepython::adSetupPartialDerivativeNode_Domains)
    ;

    class_<adSetupExpressionDerivativeNode, bases<adNode>, boost::noncopyable>("adSetupExpressionDerivativeNode", no_init)
    ;
    
    class_<adSetupExpressionPartialDerivativeNode, bases<adNode>, boost::noncopyable>("adSetupExpressionPartialDerivativeNode", no_init)
    ;

    class_<adSetupIntegralNode, bases<adNode>, boost::noncopyable>("adSetupIntegralNode", no_init)
    ;
    
    class_<adSetupSpecialFunctionNode, bases<adNode>, boost::noncopyable>("adSetupSpecialFunctionNode", no_init)
    ;

    class_<adScalarExternalFunctionNode, bases<adNode>, boost::noncopyable>("adScalarExternalFunctionNode", no_init)
    ;
    
    class_<adVectorExternalFunctionNode, bases<adNode>, boost::noncopyable>("adVectorExternalFunctionNode", no_init)
    ;

    
    class_<adDomainIndexNode, bases<adNode>, boost::noncopyable>("adDomainIndexNode", no_init)
        .def_readonly("Index",   &adDomainIndexNode::m_nIndex)
        .add_property("Domain",  make_function(&daepython::adDomainIndexNode_Domain, return_internal_reference<>()))
    ;

    class_<adRuntimeParameterNode, bases<adNode>, boost::noncopyable>("adRuntimeParameterNode", no_init)
        .def_readonly("Value",          &adRuntimeParameterNode::m_dValue)
        .add_property("Parameter",      make_function(&daepython::adRuntimeParameterNode_Parameter, return_internal_reference<>()))
        .add_property("DomainIndexes",	&daepython::adRuntimeParameterNode_Domains)
    ;

    class_<adRuntimeVariableNode, bases<adNode>, boost::noncopyable>("adRuntimeVariableNode", no_init)
        .def_readonly("OverallIndex",   &adRuntimeVariableNode::m_nOverallIndex)
        .def_readonly("BlockIndex",     &adRuntimeVariableNode::m_nBlockIndex)
        .def_readonly("IsAssigned",     &adRuntimeVariableNode::m_bIsAssigned)
            
        .add_property("Variable",       make_function(&daepython::adRuntimeVariableNode_Variable, return_internal_reference<>()))
        .add_property("DomainIndexes",	&daepython::adRuntimeVariableNode_Domains)
    ;
    
    class_<adRuntimeTimeDerivativeNode, bases<adNode>, boost::noncopyable>("adRuntimeTimeDerivativeNode", no_init)
        .def_readonly("Order",          &adRuntimeTimeDerivativeNode::m_nDegree)
        .def_readonly("OverallIndex",   &adRuntimeTimeDerivativeNode::m_nOverallIndex)
        .def_readonly("BlockIndex",     &adRuntimeTimeDerivativeNode::m_nBlockIndex)        
        .add_property("Variable",       make_function(&daepython::adRuntimeTimeDerivativeNode_Variable, return_internal_reference<>()))
        .add_property("DomainIndexes",	&daepython::adRuntimeTimeDerivativeNode_Domains)
    ;

    

    class_<adSingleNodeArray, bases<adNode>, boost::noncopyable>("adSingleNodeArray", no_init)
    ;
    
    class_<adVectorNodeArray, bases<adNode>, boost::noncopyable>("adVectorNodeArray", no_init)
    ;
    
    class_<adUnaryNodeArray, bases<adNode>, boost::noncopyable>("adUnaryNodeArray", no_init)
    ;
    
    class_<adBinaryNodeArray, bases<adNode>, boost::noncopyable>("adBinaryNodeArray", no_init)
    ;
    
    class_<adSetupVariableNodeArray, bases<adNode>, boost::noncopyable>("adSetupVariableNodeArray", no_init)
    ;
    
    class_<adSetupParameterNodeArray, bases<adNode>, boost::noncopyable>("adSetupParameterNodeArray", no_init)
    ;
    
    class_<adSetupTimeDerivativeNodeArray, bases<adNode>, boost::noncopyable>("adSetupTimeDerivativeNodeArray", no_init)
    ;
    
    class_<adSetupPartialDerivativeNodeArray, bases<adNode>, boost::noncopyable>("adSetupPartialDerivativeNodeArray", no_init)
    ;

    
    class_<condUnaryNode, bases<condNode>, boost::noncopyable>("condUnaryNode", no_init)
        .def_readonly("LogicalOperator",	&condUnaryNode::m_eLogicalOperator)
        .add_property("Node",               make_function(&condUnaryNode::getNodeRawPtr, return_internal_reference<>()))
    ;

    class_<condBinaryNode, bases<condNode>, boost::noncopyable>("condBinaryNode", no_init)
        .def_readonly("LogicalOperator",	&condBinaryNode::m_eLogicalOperator)
        .add_property("LNode",              make_function(&condBinaryNode::getLeftRawPtr,  return_internal_reference<>()))
        .add_property("RNode",              make_function(&condBinaryNode::getRightRawPtr, return_internal_reference<>()))
    ;

    class_<condExpressionNode, bases<condNode>, boost::noncopyable>("condExpressionNode", no_init)
        .def_readonly("ConditionType",	&condExpressionNode::m_eConditionType)
        .add_property("LNode",          make_function(&condExpressionNode::getLeftRawPtr,  return_internal_reference<>()))
        .add_property("RNode",          make_function(&condExpressionNode::getRightRawPtr, return_internal_reference<>()))
    ;
    
    class_<daeCondition>("daeCondition")
		.add_property("EventTolerance",	&daeCondition::GetEventTolerance, &daeCondition::SetEventTolerance)
        .add_property("SetupNode",      make_function(&daeCondition::getSetupNodeRawPtr, return_internal_reference<>()))
        .add_property("RuntimeNode",    make_function(&daeCondition::getRuntimeNodeRawPtr, return_internal_reference<>()))
        .add_property("Expressions",    &daepython::daeCondition_GetExpressions)
            
		//.def(!self)
		.def(self | self)
		.def(self & self)
	;

	class_<adouble>("adouble")
		.add_property("Value",		&adouble::getValue,      &adouble::setValue)
		.add_property("Derivative",	&adouble::getDerivative, &adouble::setDerivative) 
        .add_property("Node",       make_function(&adouble::getNodeRawPtr, return_internal_reference<>()))
        .add_property("GatherInfo",	&adouble::getGatherInfo, &adouble::setGatherInfo)

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
	def("exp",   &daepython::ad_exp);
	def("log",   &daepython::ad_log);
	def("sqrt",  &daepython::ad_sqrt);
	def("sin",   &daepython::ad_sin);
	def("cos",   &daepython::ad_cos);
	def("tan",   &daepython::ad_tan);
	def("asin",  &daepython::ad_asin);
	def("acos",  &daepython::ad_acos);
	def("atan",  &daepython::ad_atan);

	def("sinh",  &daepython::ad_sinh);
	def("cosh",  &daepython::ad_cosh);
	def("tanh",  &daepython::ad_tanh);
	def("asinh", &daepython::ad_asinh);
	def("acosh", &daepython::ad_acosh);
	def("atanh", &daepython::ad_atanh);
	def("atan2", &daepython::ad_atan2);

	def("log10", &daepython::ad_log10);
	def("abs",   &daepython::ad_abs);
	def("ceil",  &daepython::ad_ceil);
	def("floor", &daepython::ad_floor);
	def("min",   &daepython::ad_min1);
	def("min",   &daepython::ad_min2);
	def("min",   &daepython::ad_min3);
	def("max",   &daepython::ad_max1);
	def("max",   &daepython::ad_max2);
	def("max",   &daepython::ad_max3);
	def("pow",   &daepython::ad_pow1);
	def("pow",   &daepython::ad_pow2);
	def("pow",   &daepython::ad_pow3);
    
    def("dt",	 &daepython::ad_dt);
    def("d",	 &daepython::ad_d);

	def("Time",			&Time);
	def("Constant",		&daepython::ad_Constant_c);
	def("Constant",		&daepython::ad_Constant_q);
	def("Array",		&daepython::adarr_Array);  
	
	class_<adouble_array>("adouble_array")
        .add_property("GatherInfo",	 &adouble_array::getGatherInfo,	&adouble_array::setGatherInfo)
        .add_property("Node",        make_function(&adouble_array::getNodeRawPtr, return_internal_reference<>()))
        
        .def("Resize",      &adouble_array::Resize)
        .def("__len__",     &adouble_array::GetSize)
		.def("__getitem__", &adouble_array::GetItem)
        .def("__setitem__", &adouble_array::SetItem)
        .def("items",       range< return_value_policy<copy_non_const_reference> >(&adouble_array::begin, &adouble_array::end))

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
	def("exp",		&daepython::adarr_exp);
	def("log",		&daepython::adarr_log);
	def("sqrt",		&daepython::adarr_sqrt);
	def("sin",		&daepython::adarr_sin);
	def("cos",		&daepython::adarr_cos);
	def("tan",		&daepython::adarr_tan);
	def("asin",		&daepython::adarr_asin);
	def("acos",		&daepython::adarr_acos);
	def("atan",		&daepython::adarr_atan);
	def("log10",	&daepython::adarr_log10);
	def("abs",		&daepython::adarr_abs);
	def("ceil",		&daepython::adarr_ceil);
	def("floor",	&daepython::adarr_floor);
      
    def("Sum",		 &daepython::adarr_sum);
    def("Product",   &daepython::adarr_product); 
    def("Integral",  &daepython::adarr_integral);
    def("Min",		 &daepython::adarr_min);
    def("Max",		 &daepython::adarr_max);
    def("Average",	 &daepython::adarr_average);

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
        .add_property("Name",           &daeObject::GetName, &daeObject::SetName)
        .add_property("Description",    &daeObject::GetDescription, &daeObject::SetDescription)
        .add_property("CanonicalName",  &daeObject::GetCanonicalName)
        .add_property("Model",          make_function(&daepython::daeObject_GetModel, return_internal_reference<>()))

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

        .def("__str__",				&daeDomainIndex::GetIndexAsString)
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

        .def("__str__",				&daeIndexRange::ToString)
		;

	class_<daeArrayRange>("daeArrayRange")
		.def(init<daeDomainIndex>())
		.def(init<daeIndexRange>())

		.add_property("NoPoints",		&daeArrayRange::GetNoPoints)

		.def_readonly("Type",			&daeArrayRange::m_eType)
		.def_readonly("Range",			&daeArrayRange::m_Range)
		.def_readonly("DomainIndex",	&daeArrayRange::m_domainIndex)
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
        .add_property("npyPoints",              &daepython::GetNumPyArrayDomain)
        .add_property("Units",					&daeDomain::GetUnits)

		.def("__str__",							&daepython::daeDomain_str)
		.def("CreateArray",						&daeDomain::CreateArray)
		.def("CreateDistributed",				&daeDomain::CreateDistributed)
		.def("__getitem__",						&daeDomain::operator[])
        .def("__call__",						&daeDomain::operator())
//		.def("__call__",						&daepython::FunctionCallDomain1)
//		.def("__call__",						&daepython::FunctionCallDomain2)
//		.def("__call__",						&daepython::FunctionCallDomain3)  
		//.def("array",							&daepython::DomainArray1)
		//.def("array",							&daepython::DomainArray2)
		; 
    
    class_<daeDEDI, bases<daeObject>, boost::noncopyable>("daeDEDI", no_init)
        .add_property("Domain",			make_function(&daepython::daeDEDI_GetDomain, return_internal_reference<>()))
        .add_property("DomainPoints",	&daepython::daeDEDI_GetDomainPoints)
        .add_property("DomainBounds",	&daeDEDI::GetDomainBounds)
		
        .def("__str__",		&daepython::daeDEDI_str)
		.def("__call__",	&daeDEDI::operator())
		.def(self + size_t())
		.def(self - size_t())  
		;

	class_<daepython::daeParameterWrapper, bases<daeObject>, boost::noncopyable>("daeParameter")
		.def(init<string, const unit&, daePort*, optional<string, boost::python::list> >())
		.def(init<string, const unit&, daeModel*, optional<string, boost::python::list> >())

		.add_property("Units",			&daeParameter::GetUnits)
		.add_property("Domains",		&daepython::daeParameterWrapper::GetDomains)
        .add_property("ReportingOn",	&daeParameter::GetReportingOn,	&daeParameter::SetReportingOn)
        .add_property("npyValues",      &daepython::GetNumPyArrayParameter)

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
		
        .add_property("Domains",                    &daepython::daeVariable_Wrapper::GetDomains)
		.add_property("VariableType",               make_function(&daepython::daeVariable_Wrapper::GetVariableType, return_internal_reference<>()) )
		.add_property("ReportingOn",                &daeVariable::GetReportingOn,	&daeVariable::SetReportingOn)
        .add_property("OverallIndex",               &daeVariable::GetOverallIndex)
        .add_property("NumberOfPoints",             &daeVariable::GetNumberOfPoints)
        .add_property("npyValues",                  &daepython::daeVariable_Values)
        .add_property("npyIDs",                     &daepython::daeVariable_IDs)

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
         
        .def("GetDomainsIndexesMap",    &daepython::daeVariable_Wrapper::GetOverallVSDomainsIndexesMap1)

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
        .add_property("Type",            &daepython::daeActionWrapper::GetType)
        .add_property("STN",             make_function(&daepython::daeActionWrapper::GetSTN, return_internal_reference<>()))
        .add_property("StateTo",         make_function(&daepython::daeActionWrapper::GetStateTo, return_internal_reference<>()))
        .add_property("SendEventPort",   make_function(&daepython::daeActionWrapper::GetSendEventPort, return_internal_reference<>()))
        .add_property("VariableWrapper", make_function(&daepython::daeActionWrapper::GetVariableWrapper, return_internal_reference<>()))
        .add_property("SetupNode",       make_function(&daepython::daeActionWrapper::getSetupNodeRawPtr, return_internal_reference<>()))
        .add_property("RuntimeNode",     make_function(&daepython::daeActionWrapper::getRuntimeNodeRawPtr, return_internal_reference<>()))
            
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
        .add_property("STNs",					&daepython::daeModelWrapper::GetSTNs)
		.add_property("Components",				&daepython::daeModelWrapper::GetComponents)
		.add_property("PortArrays",				&daepython::daeModelWrapper::GetPortArrays)
		.add_property("ComponentArrays",		&daepython::daeModelWrapper::GetComponentArrays)
        .add_property("PortConnections",		&daepython::daeModelWrapper::GetPortConnections)
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

    class_<daeEquationExecutionInfo, boost::noncopyable>("daeEquationExecutionInfo", no_init)
        .add_property("Node",	make_function(&daeEquationExecutionInfo::GetEquationEvaluationNodeRawPtr, return_internal_reference<>()))
    ;
    
	class_<daeEquation, bases<daeObject>, boost::noncopyable>("daeEquation")
        .add_property("Residual",                       &daeEquation::GetResidual, &daeEquation::SetResidual)
        .add_property("Scaling",                        &daeEquation::GetScaling,  &daeEquation::SetScaling)
        .add_property("EquationExecutionInfos",	        &daepython::daeEquation_GetEquationExecutionInfos)
        .add_property("DistributedEquationDomainInfos",	&daepython::daeEquation_DistributedEquationDomainInfos)
            
		.def("__str__",				&daepython::daeEquation_str)
		.def("DistributeOnDomain",	&daepython::daeEquation_DistributeOnDomain1, return_internal_reference<>())
		.def("DistributeOnDomain",	&daepython::daeEquation_DistributeOnDomain2, return_internal_reference<>())
		;

    class_<daePortConnection, bases<daeObject>, boost::noncopyable>("daePortConnection", no_init)
        .add_property("PortFrom", make_function(&daepython::daePortConnection_GetPortFrom, return_internal_reference<>()))
        .add_property("PortTo",   make_function(&daepython::daePortConnection_GetPortTo, return_internal_reference<>()))
    ;
    
	class_<daeState, bases<daeObject>, boost::noncopyable>("daeState")
		.add_property("Equations",			&daepython::daeState_GetEquations)
		.add_property("StateTransitions",	&daepython::daeState_GetStateTransitions)
		.add_property("NestedSTNs",			&daepython::daeState_GetNestedSTNs)
		;

	class_<daeStateTransition, bases<daeObject>, boost::noncopyable>("daeStateTransition")
		.add_property("Condition",	make_function(&daepython::daeStateTransition_GetCondition, return_internal_reference<>()))
        .add_property("Actions",	&daepython::daeStateTransition_GetActions)
		;

    class_<daeSTN, bases<daeObject>, boost::noncopyable>("daeSTN")
        .add_property("ActiveState",	&daeSTN::GetActiveState2, &daeSTN::SetActiveState2)
        .add_property("States",			&daepython::GetStatesSTN)
        ;

	class_<daeIF, bases<daeSTN>, boost::noncopyable>("daeIF")
		;
	
	class_<daepython::daeScalarExternalFunctionWrapper, boost::noncopyable>("daeScalarExternalFunction", init<const string&, daeModel*, const unit&, boost::python::dict>())
        .add_property("Name",	&daepython::daeScalarExternalFunctionWrapper::GetName)

        .def("Calculate",	pure_virtual(&daepython::daeScalarExternalFunctionWrapper::Calculate_))
		.def("__call__",	&daeScalarExternalFunction::operator())
		;
	
	class_<daepython::daeVectorExternalFunctionWrapper, boost::noncopyable>("daeVectorExternalFunction", init<const string&, daeModel*, const unit&, size_t, boost::python::dict>())
        .add_property("Name",	&daepython::daeVectorExternalFunctionWrapper::GetName)

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
