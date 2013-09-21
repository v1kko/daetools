#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include "docstrings.h"
#include <noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyCore)
{
	import_array();
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    docstring_options doc_options(true, true, false);

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

    enum_<daeeEquationType>("daeeEquationType")
        .value("eETUnknown",	dae::core::eETUnknown)
        .value("eExplicitODE",	dae::core::eExplicitODE)
        .value("eImplicitODE",  dae::core::eImplicitODE)
        .value("eAlgebraic",    dae::core::eAlgebraic)
        .export_values()
    ;
    
    enum_<daeeModelType>("daeeModelType")
        .value("eMTUnknown",	dae::core::eMTUnknown)
        .value("eSteadyState",	dae::core::eSteadyState)
        .value("eODE",          dae::core::eODE)
        .value("eDAE",          dae::core::eDAE)
        .export_values()
    ;
    
/**************************************************************
	Global functions
***************************************************************/
	def("daeGetConfig",		&daepython::daeGetConfig, DOCSTR_global_daeGetConfig); 
    def("daeVersion",		&dae::daeVersion, ( arg("includeBuild") = false ), DOCSTR_global_daeVersion);
	def("daeVersionMajor",  &dae::daeVersionMajor, DOCSTR_global_daeVersionMajor);
	def("daeVersionMinor",  &dae::daeVersionMinor, DOCSTR_global_daeVersionMinor);
	def("daeVersionBuild",  &dae::daeVersionBuild, DOCSTR_global_daeVersionBuild);

/**************************************************************
    Global constants
***************************************************************/
    boost::python::scope().attr("cnAlgebraic")    = cnAlgebraic;
    boost::python::scope().attr("cnDifferential") = cnDifferential;
    boost::python::scope().attr("cnAssigned")     = cnAssigned;
    
/**************************************************************
	Classes
***************************************************************/
	class_<daeVariableWrapper>("daeVariableWrapper", DOCSTR_daeVariableWrapper_, no_init)
        .def(init<daeVariable&, optional<string> >(( arg("self"), arg("variable"), arg("name") = "" ),  DOCSTR_daeVariableWrapper_))
        .def(init<adouble&, optional<string> >    (( arg("self"), arg("ad"), arg("name") = "" ),        DOCSTR_daeVariableWrapper_))

		.def_readwrite("Name",         &daeVariableWrapper::m_strName,                                  DOCSTR_daeVariableWrapper_)  
		.add_property("Value",         &daeVariableWrapper::GetValue, &daeVariableWrapper::SetValue,    DOCSTR_daeVariableWrapper_)
        .add_property("OverallIndex",  &daeVariableWrapper::GetOverallIndex,                            DOCSTR_daeVariableWrapper_)
        .add_property("VariableType",  &daeVariableWrapper::GetVariableType,                            DOCSTR_daeVariableWrapper_)
        .add_property("DomainIndexes", &daepython::daeVariableWrapper_GetDomainIndexes,                 DOCSTR_daeVariableWrapper_)
        .add_property("Variable",      make_function(&daepython::daeVariableWrapper_GetVariable, return_internal_reference<>()), DOCSTR_daeVariableWrapper_)
	;

    class_<daeConfig>("daeConfig", DOCSTR_daeConfig_)
		.def("GetBoolean",	   &daepython::GetBoolean1, ( arg("self"), arg("propertyPath"), arg("defaultValue") ), DOCSTR_daeConfig_)
		.def("GetFloat",	   &daepython::GetFloat1,   ( arg("self"), arg("propertyPath"), arg("defaultValue") ), DOCSTR_daeConfig_)
		.def("GetInteger",	   &daepython::GetInteger1, ( arg("self"), arg("propertyPath"), arg("defaultValue") ), DOCSTR_daeConfig_)
		.def("GetString",	   &daepython::GetString1,  ( arg("self"), arg("propertyPath"), arg("defaultValue") ), DOCSTR_daeConfig_)
	  
		.def("GetBoolean",	   &daepython::GetBoolean, ( arg("self"), arg("propertyPath") ), DOCSTR_daeConfig_)
		.def("GetFloat",	   &daepython::GetFloat,   ( arg("self"), arg("propertyPath") ), DOCSTR_daeConfig_)
		.def("GetInteger",	   &daepython::GetInteger, ( arg("self"), arg("propertyPath") ), DOCSTR_daeConfig_)
		.def("GetString",	   &daepython::GetString,  ( arg("self"), arg("propertyPath") ), DOCSTR_daeConfig_)

		.def("SetBoolean",	   &daepython::SetBoolean, ( arg("self"), arg("propertyPath"), arg("value") ), DOCSTR_daeConfig_)
		.def("SetFloat",	   &daepython::SetFloat,   ( arg("self"), arg("propertyPath"), arg("value") ), DOCSTR_daeConfig_)
		.def("SetInteger",	   &daepython::SetInteger, ( arg("self"), arg("propertyPath"), arg("value") ), DOCSTR_daeConfig_)
		.def("SetString",	   &daepython::SetString,  ( arg("self"), arg("propertyPath"), arg("value") ), DOCSTR_daeConfig_) 

	    .def("Reload",		   &daeConfig::Reload, ( arg("self") ), DOCSTR_daeConfig_)
        
        .def("has_key",		   &daepython::daeConfig_has_key,     ( arg("self"), arg("propertyPath") ), DOCSTR_daeConfig_)
	    .def("__contains__",   &daepython::daeConfig__contains__, ( arg("self"), arg("propertyPath") ), DOCSTR_daeConfig_)
        .def("__getitem__",	   &daepython::daeConfig__getitem__,  ( arg("self"), arg("propertyPath") ), DOCSTR_daeConfig_)
        .def("__setitem__",	   &daepython::daeConfig__setitem__,  ( arg("self"), arg("propertyPath"), arg("value") ), DOCSTR_daeConfig_)
	        
        .def("__str__",		   &daepython::daeConfig__str__)
        .def("__repr__",	   &daepython::daeConfig__repr__)
	;
        
/*******************************
    Nodes
********************************/
    class_<daeNodeSaveAsContext>("daeNodeSaveAsContext")
        .def(init<daeModel*>())
        .def_readwrite("Model",	&daeNodeSaveAsContext::m_pModel) 
    ; 

    class_<adNode, boost::noncopyable>("adNode", no_init)
        .add_property("IsLinear",               &adNode::IsLinear)
        .add_property("IsFunctionOfVariables",  &adNode::IsFunctionOfVariables)
        .add_property("IsDifferential",         &adNode::IsDifferential)
        .add_property("Quantity",               &adNode::GetQuantity)
        
        .def("SaveAsLatex", &adNode::SaveAsLatex)

        .def("__str__",		   &daepython::adNode__str__)
        .def("__repr__",	   &daepython::adNode__repr__)
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

    class_<adSetupValueInArrayAtIndexNode, bases<adNode>, boost::noncopyable>("adSetupValueInArrayAtIndexNode", no_init)
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
        .add_property("Value",   &daepython::adDomainIndexNode_Value)
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

    class_<adRuntimeSpecialFunctionForLargeArraysNode, bases<adNode>, boost::noncopyable>("adRuntimeSpecialFunctionForLargeArraysNode", no_init)
    ;


    class_<adSingleNodeArray, bases<adNode>, boost::noncopyable>("adSingleNodeArray", no_init)
    ;
    
    class_<adVectorNodeArray, bases<adNodeArray>, boost::noncopyable>("adVectorNodeArray", no_init)
    ;
    
    class_<adUnaryNodeArray, bases<adNodeArray>, boost::noncopyable>("adUnaryNodeArray", no_init)
    ;
    
    class_<adBinaryNodeArray, bases<adNodeArray>, boost::noncopyable>("adBinaryNodeArray", no_init)
    ;
    
    class_<adSetupVariableNodeArray, bases<adNodeArray>, boost::noncopyable>("adSetupVariableNodeArray", no_init)
    ;
    
    class_<adSetupParameterNodeArray, bases<adNodeArray>, boost::noncopyable>("adSetupParameterNodeArray", no_init)
    ;
    
    class_<adSetupTimeDerivativeNodeArray, bases<adNodeArray>, boost::noncopyable>("adSetupTimeDerivativeNodeArray", no_init)
    ;
    
    class_<adSetupPartialDerivativeNodeArray, bases<adNodeArray>, boost::noncopyable>("adSetupPartialDerivativeNodeArray", no_init)
    ;

    class_<adSetupCustomNodeArray, bases<adNodeArray>, boost::noncopyable>("adSetupCustomNodeArray", no_init)
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
    
    class_<daeCondition>("daeCondition", DOCSTR_daeCondition)
		.add_property("EventTolerance",	&daeCondition::GetEventTolerance, &daeCondition::SetEventTolerance,                DOCSTR_daeCondition_EventTolerance)
        .add_property("SetupNode",      make_function(&daeCondition::getSetupNodeRawPtr, return_internal_reference<>()),   DOCSTR_daeCondition_SetupNode)
        .add_property("RuntimeNode",    make_function(&daeCondition::getRuntimeNodeRawPtr, return_internal_reference<>()), DOCSTR_daeCondition_RuntimeNode)
        .add_property("Expressions",    &daepython::daeCondition_GetExpressions,                                           DOCSTR_daeCondition_Expressions)
            
		//.def(!self) 
		.def(self | self)
		.def(self & self)
            
        .def("__str__",  &daepython::daeCondition__str__)
        .def("__repr__", &daepython::daeCondition__repr__)
	;

	class_<adouble>("adouble", DOCSTR_adouble, no_init)
        .def(init< optional<real_t, real_t, bool, adNode*> >(( arg("self"), arg("value") = 0.0, arg("derivative") = 0.0, arg("gatherInfo") = false, arg("node") = NULL ), DOCSTR_adouble_init))
		
        .add_property("Value",		&adouble::getValue,      &adouble::setValue, DOCSTR_adouble_Value)
		.add_property("Derivative",	&adouble::getDerivative, &adouble::setDerivative, DOCSTR_adouble_Derivative) 
        .add_property("Node",       make_function(&adouble::getNodeRawPtr, return_internal_reference<>()), DOCSTR_adouble_Node)
        .add_property("GatherInfo",	&adouble::getGatherInfo, &adouble::setGatherInfo, DOCSTR_adouble_GatherInfo)

        .def("__str__",  &daepython::adouble__str__)
        .def("__repr__", &daepython::adouble__repr__)

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
	def("Ceil",  &daepython::ad_ceil);
	def("Floor", &daepython::ad_floor);
	def("Pow",   &daepython::ad_pow1);
	def("Pow",   &daepython::ad_pow2);
	def("Pow",   &daepython::ad_pow3);
    
// Python built-in functions
    def("Abs",   &daepython::ad_abs);
    def("Min",   &daepython::ad_min1);
	def("Min",   &daepython::ad_min2);
	def("Min",   &daepython::ad_min3);
	def("Max",   &daepython::ad_max1);
	def("Max",   &daepython::ad_max2);
	def("Max",   &daepython::ad_max3);
    
    def("dt",	 &daepython::ad_dt, (arg("ad")), DOCSTR_dt);
    def("d",	 &daepython::ad_d,  (arg("ad")), DOCSTR_d);

	def("Time",			&Time, DOCSTR_Time);
	def("Constant",		&daepython::ad_Constant_c,      (arg("value")),  DOCSTR_Constant_c);
	def("Constant",		&daepython::ad_Constant_q,      (arg("value")),  DOCSTR_Constant_q);
	def("Array",		&daepython::adarr_Array,        (arg("values")), DOCSTR_Array);

	class_<adouble_array>("adouble_array", DOCSTR_adouble_array, no_init)
        .def(init< optional<bool, adNodeArray*> >( ( arg("self"), arg("gatherInfo") = false, arg("node") = NULL ), DOCSTR_adouble_array_init))
            
        .add_property("GatherInfo",	 &adouble_array::getGatherInfo,	&adouble_array::setGatherInfo,                DOCSTR_adouble_array_GatherInfo)
        .add_property("Node",        make_function(&adouble_array::getNodeRawPtr, return_internal_reference<>()), DOCSTR_adouble_array_Node)
        
        .def("Resize",      &adouble_array::Resize, ( arg("self"), arg("newSize") ),                DOCSTR_adouble_array_Resize)
        .def("__len__",     &adouble_array::GetSize, ( arg("self") ),                               DOCSTR_adouble_array_len)
		.def("__getitem__", &adouble_array::GetItem, ( arg("self"), arg("index") ),                 DOCSTR_adouble_array_getitem)
        .def("__setitem__", &adouble_array::SetItem, ( arg("self"), arg("index"), arg("value") ),   DOCSTR_adouble_array_setitem)
        .def("items",       range< return_value_policy<copy_non_const_reference> >(&adouble_array::begin, &adouble_array::end), 
                            DOCSTR_adouble_array_items)
        .def("__call__",    &daepython::adouble_array__call__, ( arg("self"), arg("index") ), DOCSTR_adouble_array_call)

        .def("__str__",     &daepython::adouble_array__str__)
        .def("__repr__",    &daepython::adouble_array__repr__)

        .def("FromList",	   &daepython::adarr_FromList,   (arg("values")), DOCSTR_adouble_array_FromList)
            .staticmethod("FromList")
        .def("FromNumpyArray", &daepython::adarr_FromNumpyArray, (arg("values")), DOCSTR_adouble_array_FromNumpyArray)
            .staticmethod("FromNumpyArray")

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
      
    def("Sum",		 &daepython::adarr_sum,      (arg("adarray"), arg("isLargeArray") = false), DOCSTR_Sum);
    def("Product",   &daepython::adarr_product,  (arg("adarray"), arg("isLargeArray") = false), DOCSTR_Product);
    def("Integral",  &daepython::adarr_integral, (arg("adarray")), DOCSTR_Integral);
    def("Min",		 &daepython::adarr_min,      (arg("adarray")), DOCSTR_Min_adarr);
    def("Max",		 &daepython::adarr_max,      (arg("adarray")), DOCSTR_Max_adarr);
    def("Average",	 &daepython::adarr_average,  (arg("adarray")), DOCSTR_Average);

	class_<daeVariableType>("daeVariableType", DOCSTR_daeVariableType, no_init)
        .def(init<string, unit, real_t, real_t, real_t, real_t>( ( arg("self"), 
                                                                   arg("name"), 
                                                                   arg("units"), 
                                                                   arg("lowerBound"), 
                                                                   arg("upperBound"), 
                                                                   arg("initialGuess"), 
                                                                   arg("absTolerance") 
                                                                  ), DOCSTR_daeVariableType_init))

		.add_property("Name",				&daeVariableType::GetName,				&daeVariableType::SetName,              DOCSTR_daeVariableType_Name)
		.add_property("Units",				&daeVariableType::GetUnits,				&daeVariableType::SetUnits,             DOCSTR_daeVariableType_Units)
		.add_property("LowerBound",			&daeVariableType::GetLowerBound,		&daeVariableType::SetLowerBound,        DOCSTR_daeVariableType_LowerBound)
		.add_property("UpperBound",			&daeVariableType::GetUpperBound,		&daeVariableType::SetUpperBound,        DOCSTR_daeVariableType_UpperBound)
		.add_property("InitialGuess",		&daeVariableType::GetInitialGuess,		&daeVariableType::SetInitialGuess,      DOCSTR_daeVariableType_InitialGuess)
		.add_property("AbsoluteTolerance",	&daeVariableType::GetAbsoluteTolerance, &daeVariableType::SetAbsoluteTolerance, DOCSTR_daeVariableType_AbsoluteTolerance)

        .def("__repr__",					&daepython::daeVariableType__repr__)
		;

    class_<daeObject, boost::noncopyable>("daeObject", DOCSTR_daeObject, no_init)
        // daeSerializable part
        .add_property("ID",             &daeObject::GetID,      DOCSTR_daeObject_ID)
        .add_property("Version",        &daeObject::GetVersion, DOCSTR_daeObject_Version)
        .add_property("Library",        &daeObject::GetLibrary, DOCSTR_daeObject_Library)
        // daeObject part
        .add_property("Name",           &daeObject::GetName, &daeObject::SetName,               DOCSTR_daeObject_Name)
        .add_property("Description",    &daeObject::GetDescription, &daeObject::SetDescription, DOCSTR_daeObject_Description)
        .add_property("CanonicalName",  &daeObject::GetCanonicalName,                           DOCSTR_daeObject_CanonicalName)
        .add_property("Model",          make_function(&daepython::daeObject_GetModel, return_internal_reference<>()), 
                                        DOCSTR_daeObject_Model)

        .def("GetNameRelativeToParentModel",            &daeObject::GetNameRelativeToParentModel,         (arg("self")), DOCSTR_daeObject_GetNameRelativeToParentModel)
        .def("GetStrippedName",                         &daeObject::GetStrippedName,                      (arg("self")), DOCSTR_daeObject_GetStrippedName)
        .def("GetStrippedNameRelativeToParentModel",    &daeObject::GetStrippedNameRelativeToParentModel, (arg("self")), DOCSTR_daeObject_GetStrippedNameRelativeToParentModel)
        ;
    def("daeIsValidObjectName",          &daeIsValidObjectName,            (arg("name")),                 DOCSTR_global_daeIsValidObjectName);
	def("daeGetRelativeName",            &daepython::daeGetRelativeName_1, (arg("parent"), arg("child")), DOCSTR_global_daeGetRelativeName1); 
	def("daeGetRelativeName",            &daepython::daeGetRelativeName_2, (arg("parent"), arg("child")), DOCSTR_global_daeGetRelativeName2);
	def("daeGetStrippedRelativeName",    &daeGetStrippedRelativeName,      (arg("parent"), arg("child")), DOCSTR_global_daeGetStrippedRelativeName);

	class_<daeDomainIndex>("daeDomainIndex", DOCSTR_daeDomainIndex, no_init)
		.def(init<size_t>(( arg("self"), arg("index") ), DOCSTR_daeDomainIndex_init1))
		.def(init<daeDistributedEquationDomainInfo*>(( arg("self"), arg("dedi") ), DOCSTR_daeDomainIndex_init2))
		.def(init<daeDistributedEquationDomainInfo*, int>(( arg("self"), arg("dedi"), arg("increment") ), DOCSTR_daeDomainIndex_init3))
		.def(init<daeDomainIndex>(( arg("self"), arg("domainIndex") ), DOCSTR_daeDomainIndex_init4))

		.def_readonly("Type",		&daeDomainIndex::m_eType,      DOCSTR_daeDomainIndex_Type)
		.def_readonly("Index",		&daeDomainIndex::m_nIndex,     DOCSTR_daeDomainIndex_Index)
		.def_readonly("DEDI",		&daeDomainIndex::m_pDEDI,      DOCSTR_daeDomainIndex_DEDI)
		.def_readonly("Increment",	&daeDomainIndex::m_iIncrement, DOCSTR_daeDomainIndex_Increment)

        .def("__str__",				&daeDomainIndex::GetIndexAsString)
        .def("__repr__",			&daepython::daeDomainIndex__repr__)
		;

	class_<daeIndexRange>("daeIndexRange", DOCSTR_daeIndexRange, no_init)
		.def(init<daeDomain*>(( arg("self"), arg("domain") ), DOCSTR_daeIndexRange_init1))
		.def ("__init__", make_constructor(daepython::__init__daeIndexRange), DOCSTR_daeIndexRange_init2)
		.def(init<daeDomain*, size_t, size_t,size_t>(( arg("self"), arg("domain"), arg("startIndex"), arg("endIndex"), arg("step") ), DOCSTR_daeIndexRange_init3))

        .add_property("Domain",		make_function(&daepython::daeIndexRange_GetDomain, return_internal_reference<>()), 
                                    DOCSTR_daeIndexRange_Domain)
		.add_property("NoPoints",	&daeIndexRange::GetNoPoints,   DOCSTR_daeIndexRange_NoPoints)
		.def_readonly("Type",		&daeIndexRange::m_eType,       DOCSTR_daeIndexRange_Type)
		.def_readonly("StartIndex",	&daeIndexRange::m_iStartIndex, DOCSTR_daeIndexRange_StartIndex)
		.def_readonly("EndIndex",	&daeIndexRange::m_iEndIndex,   DOCSTR_daeIndexRange_EndIndex)
		.def_readonly("Step",		&daeIndexRange::m_iStride,     DOCSTR_daeIndexRange_Step)

        .def("__str__",				&daeIndexRange::ToString)
        .def("__repr__",			&daepython::daeIndexRange__repr__)
		;

	class_<daeArrayRange>("daeArrayRange", DOCSTR_daeArrayRange, no_init)
		.def(init<daeDomainIndex>(( arg("self"), arg("domainIndex") ), DOCSTR_daeArrayRange_init1))
		.def(init<daeIndexRange>(( arg("self"), arg("indexRange") ), DOCSTR_daeArrayRange_init2))

		.add_property("NoPoints",		&daeArrayRange::GetNoPoints,   DOCSTR_daeArrayRange_NoPoints)

		.def_readonly("Type",			&daeArrayRange::m_eType,       DOCSTR_daeArrayRange_Type)
		.def_readonly("Range",			&daeArrayRange::m_Range,       DOCSTR_daeArrayRange_Range)
		.def_readonly("DomainIndex",	&daeArrayRange::m_domainIndex, DOCSTR_daeArrayRange_DomainIndex)
            
        .def("__str__",				&daepython::daeArrayRange__str__)
        .def("__repr__",			&daepython::daeArrayRange__repr__)
		;

	class_<daeDomain, bases<daeObject>, boost::noncopyable>("daeDomain", DOCSTR_daeDomain, no_init)
        //.def(init<>(( arg("self") ), DOCSTR_daeDomain_init))
        .def(init<string, daeModel*, const unit&, optional<string> >(( arg("self"), 
                                                                       arg("name"), 
                                                                       arg("parentModel"), 
                                                                       arg("units"),
                                                                       arg("description") = "" 
                                                                     ), DOCSTR_daeDomain_init1))
		.def(init<string, daePort*, const unit&, optional<string> >((  arg("self"), 
                                                                       arg("name"), 
                                                                       arg("parentPort"),  
                                                                       arg("units"), 
                                                                       arg("description") = "" 
                                                                    ), DOCSTR_daeDomain_init2))

		.add_property("Type",					&daeDomain::GetType,                DOCSTR_daeDomain_Type)
		.add_property("NumberOfIntervals",		&daeDomain::GetNumberOfIntervals,   DOCSTR_daeDomain_NumberOfIntervals)
		.add_property("NumberOfPoints",			&daeDomain::GetNumberOfPoints,      DOCSTR_daeDomain_NumberOfPoints)
		.add_property("DiscretizationMethod",	&daeDomain::GetDiscretizationMethod, DOCSTR_daeDomain_DiscretizationMethod)
		.add_property("DiscretizationOrder",	&daeDomain::GetDiscretizationOrder, DOCSTR_daeDomain_DiscretizationOrder)
		.add_property("LowerBound",				&daeDomain::GetLowerBound,          DOCSTR_daeDomain_LowerBound)
		.add_property("UpperBound",				&daeDomain::GetUpperBound,          DOCSTR_daeDomain_UpperBound) 
        .add_property("Units",					&daeDomain::GetUnits,               DOCSTR_daeDomain_Units)
        .add_property("npyPoints",              &daepython::GetNumPyArrayDomain,    DOCSTR_daeDomain_npyPoints)
        .add_property("Points",					&daepython::GetDomainPoints, 
                                                &daepython::SetDomainPoints, DOCSTR_daeDomain_Points) 

		.def("__str__",	 					    &daepython::daeDomain__str__)
        .def("__repr__",						&daepython::daeDomain__repr__)
		
        .def("CreateArray",						&daeDomain::CreateArray, ( arg("self"), 
                                                                           arg("noIntervals") 
                                                                         ), DOCSTR_daeDomain_CreateArray)
		.def("CreateDistributed",				&daeDomain::CreateDistributed, ( arg("self"),
                                                                                 arg("discretizationMethod"), 
                                                                                 arg("discretizationOrder"), 
                                                                                 arg("numberOfIntervals"),
                                                                                 arg("lowerBound"),
                                                                                 arg("upperBound")
                                                                                ), DOCSTR_daeDomain_CreateDistributed)
		.def("__getitem__",						&daeDomain::operator[], ( arg("self"), arg("index") ), DOCSTR_daeDomain_getitem)
        .def("__call__",						&daeDomain::operator(), ( arg("self"), arg("index") ), DOCSTR_daeDomain_call) 
		.def("array",							&daepython::DomainArray, ( arg("self"), arg("indexes") ), DOCSTR_daeDomain_array)
		; 
    
    class_<daeDEDI, bases<daeObject>, boost::noncopyable>("daeDEDI", DOCSTR_daeDEDI, no_init)
        .add_property("Domain",			make_function(&daepython::daeDEDI_GetDomain, return_internal_reference<>()), DOCSTR_daeDEDI_Domain)
        .add_property("DomainPoints",	&daepython::daeDEDI_GetDomainPoints, DOCSTR_daeDEDI_DomainPoints)
        .add_property("DomainBounds",	&daeDEDI::GetDomainBounds, DOCSTR_daeDEDI_DomainBounds)
		
        .def("__str__",	    &daepython::daeDEDI__str__)
        .def("__repr__",	&daepython::daeDEDI__repr__)
            
		.def("__call__",	&daeDEDI::operator(), ( arg("self") ), DOCSTR_daeDEDI_call)
		.def(self + size_t())
		.def(self - size_t())  
		;

	class_<daepython::daeParameterWrapper, bases<daeObject>, boost::noncopyable>("daeParameter", DOCSTR_daeParameter, no_init)
		.def(init<string, const unit&, daePort*, optional<string, boost::python::list> >(( arg("self"), 
                                                                                           arg("name"), 
                                                                                           arg("units"), 
                                                                                           arg("parentPort"),  
                                                                                           arg("description") = "", 
                                                                                           arg("domains") =  boost::python::list()
                                                                                         ), DOCSTR_daeParameter_init1))
		.def(init<string, const unit&, daeModel*, optional<string, boost::python::list> >(( arg("self"), 
                                                                                            arg("name"), 
                                                                                            arg("units"), 
                                                                                            arg("parentModel"),  
                                                                                            arg("description") = "", 
                                                                                            arg("domains") =  boost::python::list()
                                                                                          ), DOCSTR_daeParameter_init2))

		.add_property("Units",			&daeParameter::GetUnits, DOCSTR_daeParameter_Units)
		.add_property("Domains",		&daepython::daeParameterWrapper::GetDomains, DOCSTR_daeParameter_Domains)
        .add_property("ReportingOn",	&daeParameter::GetReportingOn,	&daeParameter::SetReportingOn, DOCSTR_daeParameter_ReportingOn)
        .add_property("npyValues",      &daepython::GetNumPyArrayParameter, DOCSTR_daeParameter_npyValues)
        .add_property("NumberOfPoints", &daeParameter::GetNumberOfPoints, DOCSTR_daeParameter_NumberOfPoints)

		.def("__repr__",				&daepython::daeParameter__repr__)  
        .def("__str__", 				&daepython::daeParameter__str__)  
            
		.def("DistributeOnDomain",	&daeParameter::DistributeOnDomain, ( arg("self"), arg("domain") ), DOCSTR_daeParameter_DistributeOnDomain)

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
	        
        .def("GetDomainsIndexesMap", &daepython::daeParameterWrapper::GetDomainsIndexesMap1, 
                                     ( arg("self"), arg("indexBase") ), DOCSTR_daeParameter_GetDomainsIndexesMap)
            
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

	class_<daepython::daeVariable_Wrapper, bases<daeObject>, boost::noncopyable>("daeVariable", DOCSTR_daeVariable, no_init)  
		.def(init<string, const daeVariableType&, daeModel*, optional<string, boost::python::list> >(( arg("self"), 
                                                                                                       arg("name"), 
                                                                                                       arg("variableType"), 
                                                                                                       arg("parentPort"),  
                                                                                                       arg("description") = "", 
                                                                                                       arg("domains") =  boost::python::list()
                                                                                                      ), DOCSTR_daeVariable_init1))
		.def(init<string, const daeVariableType&, daePort*, optional<string, boost::python::list> >(( arg("self"), 
                                                                                                      arg("name"), 
                                                                                                      arg("variableType"), 
                                                                                                      arg("parentModel"),  
                                                                                                      arg("description") = "", 
                                                                                                      arg("domains") =  boost::python::list()
                                                                                                    ), DOCSTR_daeVariable_init2))
		
        .add_property("Domains",            &daepython::daeVariable_Wrapper::GetDomains, DOCSTR_daeVariable_Domains)
		.add_property("VariableType",       make_function(&daepython::daeVariable_Wrapper::GetVariableType, return_internal_reference<>()),
                                            DOCSTR_daeVariable_VariabeType)
		.add_property("ReportingOn",        &daeVariable::GetReportingOn,
                                            &daeVariable::SetReportingOn,             DOCSTR_daeVariable_ReportingOn)
        .add_property("OverallIndex",       &daeVariable::GetOverallIndex,            DOCSTR_daeVariable_OverallIndex)
        .add_property("NumberOfPoints",     &daeVariable::GetNumberOfPoints,          DOCSTR_daeVariable_NumberOfPoints)
        .add_property("npyValues",          &daepython::daeVariable_Values,           DOCSTR_daeVariable_npyValues)
        .add_property("npyTimeDerivatives", &daepython::daeVariable_TimeDerivatives,  DOCSTR_daeVariable_npyTimeDerivatives)
        .add_property("npyIDs",             &daepython::daeVariable_IDs,              DOCSTR_daeVariable_npyIDs)

		.def("__str__",	  &daepython::daeVariable__str__)
        .def("__repr__",  &daepython::daeVariable__repr__)
            
		.def("DistributeOnDomain", &daeVariable::DistributeOnDomain, ( arg("self"), arg("domain") ), DOCSTR_daeVariable_DistributeOnDomain)

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
         
        .def("GetDomainsIndexesMap",    &daepython::daeVariable_Wrapper::GetDomainsIndexesMap1, ( arg("self"), arg("indexBase") ), DOCSTR_daeVariable_GetDomainIndexesMap)

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

	class_<daeModelExportContext>("daeModelExportContext", DOCSTR_daeModelExportContext, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeModelExportContext_init))
		.def_readonly("PythonIndentLevel",	&daeModelExportContext::m_nPythonIndentLevel, DOCSTR_daeModelExportContext_PythonIndentLevel)
		.def_readonly("ExportDefinition",	&daeModelExportContext::m_bExportDefinition, DOCSTR_daeModelExportContext_ExportDefinition)
		;

	class_<daepython::daePortWrapper, bases<daeObject>, boost::noncopyable>("daePort", DOCSTR_daePort, no_init)
		.def(init<string, daeePortType, daeModel*, optional<string> >(( arg("self"), 
                                                                        arg("name"), 
                                                                        arg("type"), 
                                                                        arg("parentModel"), 
                                                                        arg("description") = "" 
                                                                      ), DOCSTR_daePort_init))

		.add_property("Type",			&daePort::GetType, DOCSTR_daePort_Type)
		.add_property("Domains",		&daepython::daePortWrapper::GetDomains, DOCSTR_daePort_Domains)
		.add_property("Parameters",		&daepython::daePortWrapper::GetParameters, DOCSTR_daePort_Parameters)
		.add_property("Variables",		&daepython::daePortWrapper::GetVariables, DOCSTR_daePort_Variables)

		.def("__str__",		    		&daepython::daePort__str__)
        .def("__repr__",				&daepython::daePort__repr__)
            
		.def("SetReportingOn",			&daePort::SetReportingOn, ( arg("self"), arg("reportingOn") ), DOCSTR_daePort_SetReportingOn)
		.def("Export",					&daePort::Export, ( arg("self"), arg("content"), arg("language"), arg("context") ), DOCSTR_daePort_Export)
		;

    class_<daeEventPort, bases<daeObject>, boost::noncopyable>("daeEventPort", DOCSTR_daeEventPort, no_init)
        .def(init<string, daeePortType, daeModel*, optional<string> >(( arg("self"), 
                                                                        arg("name"), 
                                                                        arg("type"), 
                                                                        arg("parentModel"), 
                                                                        arg("description")  = ""
                                                                      ), DOCSTR_daePort_init))

        .add_property("Type",			&daeEventPort::GetType, DOCSTR_daeEventPort_Type)
        .add_property("EventData",		&daeEventPort::GetEventData, DOCSTR_daeEventPort_EventData)
	    .add_property("RecordEvents",	&daeEventPort::GetRecordEvents, &daeEventPort::SetRecordEvents, DOCSTR_daeEventPort_RecordEvents)
        .add_property("Events",			&daepython::GetEventPortEventsList, DOCSTR_daeEventPort_Events)  
		
        .def("__str__",		    &daepython::daeEventPort__str__)
        .def("__repr__",		&daepython::daeEventPort__repr__)
		.def("__call__",		&daeEventPort::operator(), ( arg("self") ), DOCSTR_daeEventPort_call)
		.def("SendEvent",		&daeEventPort::SendEvent, ( arg("self"), arg("data") ) , DOCSTR_daeEventPort_SendEvent)  
		.def("ReceiveEvent",	&daeEventPort::ReceiveEvent, ( arg("self"), arg("data") ), DOCSTR_daeEventPort_ReceiveEvent)  
        ;

	class_<daepython::daeActionWrapper, bases<daeObject>, boost::noncopyable>("daeAction", DOCSTR_daeAction, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeAction_init))
        .add_property("Type",            &daepython::daeActionWrapper::GetType, DOCSTR_daeAction_Type)
        .add_property("STN",             make_function(&daepython::daeActionWrapper::GetSTN, return_internal_reference<>()), DOCSTR_daeAction_STN)
        .add_property("StateTo",         make_function(&daepython::daeActionWrapper::GetStateTo, return_internal_reference<>()), DOCSTR_daeAction_StateTo)
        .add_property("SendEventPort",   make_function(&daepython::daeActionWrapper::GetSendEventPort, return_internal_reference<>()), DOCSTR_daeAction_SendEventPort)
        .add_property("VariableWrapper", make_function(&daepython::daeActionWrapper::GetVariableWrapper, return_internal_reference<>()), DOCSTR_daeAction_VariableWrapper)
        .add_property("SetupNode",       make_function(&daepython::daeActionWrapper::getSetupNodeRawPtr, return_internal_reference<>()), DOCSTR_daeAction_SetupNode)
        .add_property("RuntimeNode",     make_function(&daepython::daeActionWrapper::getRuntimeNodeRawPtr, return_internal_reference<>()), DOCSTR_daeAction_RuntimeNode)
            
        .def("__str__",				&daepython::daeActionWrapper::__str__)
        .def("__repr__",			&daepython::daeActionWrapper::__repr__)
            
		.def("Execute",		pure_virtual(&daepython::daeActionWrapper::Execute), ( arg("self") ), DOCSTR_daeAction_Execute)
        ;
 
    class_<daeOptimizationVariable_t, boost::noncopyable>("daeOptimizationVariable_t", no_init)
        ;

    class_<daeObjectiveFunction_t, boost::noncopyable>("daeObjectiveFunction_t", no_init)
        ;

    class_<daeOptimizationConstraint_t, boost::noncopyable>("daeOptimizationConstraint_t", no_init)
        ;

    class_<daeMeasuredVariable_t, boost::noncopyable>("daeMeasuredVariable_t", no_init)
        ;

    class_<daeOptimizationVariable, bases<daeOptimizationVariable_t> >("daeOptimizationVariable", DOCSTR_daeOptimizationVariable, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeOptimizationVariable_init))
            
        .add_property("Name",           &daeOptimizationVariable::GetName, DOCSTR_daeOptimizationVariable)
        .add_property("Type",           &daeOptimizationVariable::GetType,          &daeOptimizationVariable::SetType, DOCSTR_daeOptimizationVariable)
        .add_property("Value",          &daeOptimizationVariable::GetValue,         &daeOptimizationVariable::SetValue, DOCSTR_daeOptimizationVariable)
        .add_property("LowerBound",     &daeOptimizationVariable::GetLB,            &daeOptimizationVariable::SetLB, DOCSTR_daeOptimizationVariable)
        .add_property("UpperBound",     &daeOptimizationVariable::GetUB,            &daeOptimizationVariable::SetUB, DOCSTR_daeOptimizationVariable)
        .add_property("StartingPoint",  &daeOptimizationVariable::GetStartingPoint, &daeOptimizationVariable::SetStartingPoint, DOCSTR_daeOptimizationVariable)
            
        .def("__str__",				&daepython::daeOptimizationVariable__str__)
        .def("__repr__",			&daepython::daeOptimizationVariable__repr__)
        ;

    class_<daeObjectiveFunction, bases<daeObjectiveFunction_t> >("daeObjectiveFunction", DOCSTR_daeObjectiveFunction, no_init)  
        .def(init<>(( arg("self") ), DOCSTR_daeObjectiveFunction_init))
            
        .add_property("Name",           &daeObjectiveFunction::GetName, DOCSTR_daeObjectiveFunction_Name)
        .add_property("Residual",       &daeObjectiveFunction::GetResidual, &daeObjectiveFunction::SetResidual, DOCSTR_daeObjectiveFunction_Residual)
        .add_property("Value",          &daeObjectiveFunction::GetValue, DOCSTR_daeObjectiveFunction_Value)
        .add_property("Gradients",      &daepython::GetGradientsObjectiveFunction, DOCSTR_daeObjectiveFunction_Gradients)
        //.add_property("AbsTolerance", &daeObjectiveFunction::GetAbsTolerance, &daeObjectiveFunction::SetAbsTolerance, DOCSTR_daeObjectiveFunction_AbsTolerance)
            
        .def("__str__",				&daepython::daeObjectiveFunction__str__)
        .def("__repr__",			&daepython::daeObjectiveFunction__repr__)
        ;

    class_<daeOptimizationConstraint, bases<daeOptimizationConstraint_t> >("daeOptimizationConstraint", DOCSTR_daeOptimizationConstraint, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeOptimizationConstraint_init))
            
        .add_property("Name",           &daeOptimizationConstraint::GetName, DOCSTR_daeOptimizationConstraint_Name)
        .add_property("Residual",       &daeOptimizationConstraint::GetResidual, &daeOptimizationConstraint::SetResidual, DOCSTR_daeOptimizationConstraint_Residual)
        .add_property("Value",          &daeOptimizationConstraint::GetValue, DOCSTR_daeOptimizationConstraint_Value)
        .add_property("Type",           &daeOptimizationConstraint::GetType, &daeOptimizationConstraint::SetType, DOCSTR_daeOptimizationConstraint_Type)
        .add_property("Gradients",      &daepython::GetGradientsOptimizationConstraint, DOCSTR_daeOptimizationConstraint_Gradients)
        //.add_property("AbsTolerance", &daeOptimizationConstraint::GetAbsTolerance, &daeOptimizationConstraint::SetAbsTolerance, DOCSTR_daeOptimizationConstraint_AbsTolerance)
            
        .def("__str__",				&daepython::daeOptimizationConstraint__str__)
        .def("__repr__",			&daepython::daeOptimizationConstraint__repr__)
        ;

    class_<daeMeasuredVariable, bases<daeMeasuredVariable_t> >("daeMeasuredVariable", DOCSTR_daeMeasuredVariable, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeMeasuredVariable_init))
            
        .add_property("Name",           &daeMeasuredVariable::GetName, DOCSTR_daeMeasuredVariable_Name)
        .add_property("Residual",       &daeMeasuredVariable::GetResidual, &daeMeasuredVariable::SetResidual, DOCSTR_daeMeasuredVariable_Residual)
        .add_property("Value",          &daeMeasuredVariable::GetValue, DOCSTR_daeMeasuredVariable_Value)
        .add_property("Gradients",      &daepython::GetGradientsMeasuredVariable, DOCSTR_daeMeasuredVariable_Gradients)
        //.add_property("AbsTolerance", &daeMeasuredVariable::GetAbsTolerance, &daeMeasuredVariable::SetAbsTolerance, DOCSTR_daeMeasuredVariable_AbsTolerance)
            
        .def("__str__",				&daepython::daeMeasuredVariable__str__)
        .def("__repr__",			&daepython::daeMeasuredVariable__repr__)
        ;

    class_<daeOnEventActions, bases<daeObject>, boost::noncopyable>("daeOnEventActions", DOCSTR_daeOnEventActions, no_init)
        .add_property("EventPort",          make_function(&daeOnEventActions::GetEventPort, return_internal_reference<>()),
                                                                                               DOCSTR_daeOnEventActions_EventPort)
        .add_property("Actions",            &daepython::daeOnEventActions_Actions,             DOCSTR_daeOnEventActions_Actions)
        .add_property("UserDefinedActions", &daepython::daeOnEventActions_UserDefinedActions,  DOCSTR_daeOnEventActions_UserDefinedActions)
            
        .def("Execute",	&daeOnEventActions::Execute, DOCSTR_daeOnEventActions_Execute)
            
        .def("__str__",	 &daepython::daeOnEventActions__str__)
        .def("__repr__", &daepython::daeOnEventActions__repr__) 
    ;
    
    class_<daeOnConditionActions, bases<daeObject>, boost::noncopyable>("daeOnConditionActions", DOCSTR_daeOnConditionActions, no_init)
		.add_property("Condition",          make_function(&daepython::daeOnConditionActions_Condition, return_internal_reference<>()),
                                                                                                   DOCSTR_daeOnConditionActions_Condition)
        .add_property("Actions",	        &daepython::daeOnConditionActions_Actions,             DOCSTR_daeOnConditionActions_Actions)
        .add_property("UserDefinedActions", &daepython::daeOnConditionActions_UserDefinedActions,  DOCSTR_daeOnConditionActions_UserDefinedActions)
        
        .def("Execute",	&daeOnConditionActions::Execute, DOCSTR_daeOnConditionActions_Execute)
        
        .def("__str__",	 &daepython::daeOnConditionActions__str__)
        .def("__repr__", &daepython::daeOnConditionActions__repr__)
		; 
    
	class_<daepython::daeModelWrapper, bases<daeObject>, boost::noncopyable>("daeModel", DOCSTR_daeModel, no_init)
		.def(init<string, optional<daeModel*, string> >(( arg("self"), 
                                                          arg("name"),
                                                          arg("parentModel") = NULL, 
                                                          arg("description") = ""
                                                        ), DOCSTR_daeModel_init))

		.add_property("Domains",				&daepython::daeModelWrapper::GetDomains,            DOCSTR_daeModel_Domains)
		.add_property("Parameters",				&daepython::daeModelWrapper::GetParameters,         DOCSTR_daeModel_Parameters)
		.add_property("Variables",				&daepython::daeModelWrapper::GetVariables,          DOCSTR_daeModel_Variables)
		.add_property("Equations",				&daepython::daeModelWrapper::GetEquations,          DOCSTR_daeModel_Equations)
		.add_property("Ports",					&daepython::daeModelWrapper::GetPorts,              DOCSTR_daeModel_Ports)
		.add_property("EventPorts",				&daepython::daeModelWrapper::GetEventPorts,         DOCSTR_daeModel_EventPorts)
		.add_property("OnEventActions",			&daepython::daeModelWrapper::GetOnEventActions,     DOCSTR_daeModel_OnEventActions)
        .add_property("OnConditionActions",		&daepython::daeModelWrapper::GetOnConditionActions, DOCSTR_daeModel_OnConditionActions)
        .add_property("STNs",					&daepython::daeModelWrapper::GetSTNs,               DOCSTR_daeModel_STNs)
		.add_property("Components",				&daepython::daeModelWrapper::GetComponents,         DOCSTR_daeModel_Components)
		.add_property("PortArrays",				&daepython::daeModelWrapper::GetPortArrays,         DOCSTR_daeModel_PortArrays)
		.add_property("ComponentArrays",		&daepython::daeModelWrapper::GetComponentArrays,    DOCSTR_daeModel_ComponentArrays)
        .add_property("PortConnections",		&daepython::daeModelWrapper::GetPortConnections,    DOCSTR_daeModel_PortConnections)
        .add_property("EventPortConnections",	&daepython::daeModelWrapper::GetEventPortConnections,DOCSTR_daeModel_EventPortConnections)
        .add_property("IsModelDynamic",			&daeModel::IsModelDynamic,                          DOCSTR_daeModel_IsModelDynamic)
        .add_property("ModelType",	   		    &daeModel::GetModelType,                            DOCSTR_daeModel_ModelType)
        .add_property("InitialConditionMode",	&daeModel::GetInitialConditionMode, &daeModel::SetInitialConditionMode, DOCSTR_daeModel_InitialConditionMode)
        .add_property("OverallIndex_BlockIndex_VariableNameMap", &daepython::daeModelWrapper::GetOverallIndex_BlockIndex_VariableNameMap,
                                                                 DOCSTR_daeModel_OverallIndex_BlockIndex_VariableNameMap)

        .def("__str__",           &daepython::daeModel__str__)
        .def("__repr__",          &daepython::daeModel__repr__)  
            
        .def("CreateEquation",   &daeModel::CreateEquation, return_internal_reference<>(), 
                                 ( arg("self"), arg("name"), arg("description") = "", arg("scaling") = 1.0), DOCSTR_daeModel_CreateEquation)
		.def("DeclareEquations", &daeModel::DeclareEquations,  &daepython::daeModelWrapper::def_DeclareEquations,
                                 ( arg("self") ), DOCSTR_daeModel_DeclareEquations)
		.def("ConnectPorts",     &daeModel::ConnectPorts, 
                                 ( arg("self"), arg("portFrom"), arg("portTo") ), DOCSTR_daeModel_ConnectPorts)
		.def("ConnectEventPorts",&daeModel::ConnectEventPorts, 
                                 ( arg("self"), arg("portFrom"), arg("portTo") ), DOCSTR_daeModel_ConnectEventPorts)
		.def("SetReportingOn",	 &daeModel::SetReportingOn, 
                                 ( arg("self"), arg("reportingOn") ), DOCSTR_daeModel_SetReportingOn)

		.def("IF",				&daepython::daeModelWrapper::IF, 
                                ( arg("self"), arg("condition"), arg("eventTolerance") = 0.0 ), DOCSTR_daeModel_IF)
		.def("ELSE_IF",			&daepython::daeModelWrapper::ELSE_IF, 
                                ( arg("self"), arg("condition"), arg("eventTolerance") = 0.0 ), DOCSTR_daeModel_ELSE_IF)
		.def("ELSE",			&daepython::daeModelWrapper::ELSE,
                                ( arg("self") ), DOCSTR_daeModel_ELSE)
		.def("END_IF",			&daepython::daeModelWrapper::END_IF,
                                ( arg("self") ), DOCSTR_daeModel_END_IF) 

		.def("STN",				&daepython::daeModelWrapper::STN, return_internal_reference<>(),
                                ( arg("self"), arg("stnName") ), DOCSTR_daeModel_STN)
		.def("STATE",			&daepython::daeModelWrapper::STATE, return_internal_reference<>(),
                                ( arg("self"), arg("stateName") ), DOCSTR_daeModel_STATE)
		.def("END_STN",			&daepython::daeModelWrapper::END_STN,
                                ( arg("self") ), DOCSTR_daeModel_END_STN)
		.def("SWITCH_TO",		&daepython::daeModelWrapper::SWITCH_TO, 
                                ( arg("self"), 
                                  arg("targetState"), 
                                  arg("condition"), 
                                  arg("eventTolerance") = 0.0 
                                ), DOCSTR_daeModel_SWITCH_TO)
        .def("ON_CONDITION",    &daepython::daeModelWrapper::ON_CONDITION, 
                                ( arg("self"), 
                                  arg("condition"), 
                                  arg("switchToStates")     = boost::python::list(), 
                                  arg("setVariableValues")  = boost::python::list(),
                                  arg("triggerEvents")      = boost::python::list(),
                                  arg("userDefinedActions") = boost::python::list(),
                                  arg("eventTolerance")     = 0.0
                                ), DOCSTR_daeModel_ON_CONDITION)
        .def("ON_EVENT",		&daepython::daeModelWrapper::ON_EVENT,
                                ( arg("self"),
                                  arg("eventPort"),
                                  arg("switchToStates")     = boost::python::list(),
	                              arg("setVariableValues")  = boost::python::list(),
								  arg("triggerEvents")      = boost::python::list(),
								  arg("userDefinedActions") = boost::python::list() 
                                ), DOCSTR_daeModel_ON_EVENT )
		
		.def("SaveModelReport",			&daeModel::SaveModelReport, 
                                        ( arg("self"), arg("xmlFilename") ), DOCSTR_daeModel_SaveModelReport)
		.def("SaveRuntimeModelReport",	&daeModel::SaveRuntimeModelReport, 
                                        ( arg("self"), arg("xmlFilename") ), DOCSTR_daeModel_SaveRuntimeModelReport)
		.def("ExportObjects",			&daepython::daeModelWrapper::ExportObjects, 
                                        ( arg("self"), arg("objects"), arg("language") ), DOCSTR_daeModel_ExportObjects)
		.def("Export",					&daeModel::Export,
                                        ( arg("self"), arg("content"), arg("language"), arg("modelExportContext") ), DOCSTR_daeModel_Export)
		;

    class_<daeEquationExecutionInfo, boost::noncopyable>("daeEquationExecutionInfo", DOCSTR_daeEquationExecutionInfo, no_init)
        .add_property("Node",	             make_function(&daeEquationExecutionInfo::GetEquationEvaluationNodeRawPtr, return_internal_reference<>()),
                                             DOCSTR_daeEquationExecutionInfo_Node)
        .add_property("Name",                &daeEquationExecutionInfo::GetName, DOCSTR_daeEquationExecutionInfo_Name)
        .add_property("VariableIndexes",     &daepython::daeEquationExecutionInfo_GetVariableIndexes, DOCSTR_daeEquationExecutionInfo_VariableIndexes)
        .add_property("EquationIndex",       &daeEquationExecutionInfo::GetEquationIndexInBlock, DOCSTR_daeEquationExecutionInfo_EquationIndex)
        .add_property("EquationType",	     &daeEquationExecutionInfo::GetEquationType, DOCSTR_daeEquationExecutionInfo_EquationType)
        .add_property("JacobianExpressions", &daepython::daeEquationExecutionInfo_JacobianExpressions, DOCSTR_daeEquationExecutionInfo_JacobianExpressions)
        .add_property("Equation",            make_function(&daeEquationExecutionInfo::GetEquation, return_internal_reference<>()),
                                             DOCSTR_daeEquationExecutionInfo_Equation)
    ;

	class_<daeEquation, bases<daeObject>, boost::noncopyable>("daeEquation", DOCSTR_daeEquation, no_init)
        .add_property("Residual",                       &daeEquation::GetResidual, &daeEquation::SetResidual, DOCSTR_daeEquation_Residual)
        .add_property("Scaling",                        &daeEquation::GetScaling,  &daeEquation::SetScaling, DOCSTR_daeEquation_Scaling)
        .add_property("BuildJacobianExpressions",       &daeEquation::GetBuildJacobianExpressions,  &daeEquation::SetBuildJacobianExpressions, DOCSTR_daeEquation_BuildJacobianExpressions)
        .add_property("CheckUnitsConsistency",          &daeEquation::GetCheckUnitsConsistency,  &daeEquation::SetCheckUnitsConsistency, DOCSTR_daeEquation_CheckUnitConsistency)
        .add_property("EquationExecutionInfos",	        &daepython::daeEquation_GetEquationExecutionInfos, DOCSTR_daeEquation_EquationExecutionInfos)
        .add_property("DistributedEquationDomainInfos",	&daepython::daeEquation_DistributedEquationDomainInfos, DOCSTR_daeEquation_DistributedEquationDomainInfos)
        .add_property("EquationType",	                &daeEquation::GetEquationType, DOCSTR_daeEquation_EquationType)
            
		.def("__str__",				&daepython::daeEquation__str__)
        .def("__repr__",			&daepython::daeEquation__repr__)
            
		.def("DistributeOnDomain",	&daepython::daeEquation_DistributeOnDomain1, return_internal_reference<>(), 
                                    ( arg("self"), arg("domain"), arg("domainBounds"), arg("name") = string("") ), DOCSTR_daeEquation_DistributeOnDomain1)
		.def("DistributeOnDomain",	&daepython::daeEquation_DistributeOnDomain2, return_internal_reference<>(), 
                                    ( arg("self"), arg("domain"), arg("domainIndexes"), arg("name") = string("") ), DOCSTR_daeEquation_DistributeOnDomain1)
	;

    class_<daePortConnection, bases<daeObject>, boost::noncopyable>("daePortConnection", DOCSTR_daePortConnection, no_init)
        .add_property("PortFrom",   make_function(&daepython::daePortConnection_GetPortFrom, return_internal_reference<>()), DOCSTR_daePortConnection_PortFrom)
        .add_property("PortTo",     make_function(&daepython::daePortConnection_GetPortTo, return_internal_reference<>()), DOCSTR_daePortConnection_PortTo)
        .add_property("Equations",	&daepython::daePortConnection_GetEquations, DOCSTR_daePortConnection_Equations)  
            
        .def("__str__",				&daepython::daePortConnection__str__)
        .def("__repr__",			&daepython::daePortConnection__repr__)
    ;
    
    class_<daeEventPortConnection, bases<daeObject>, boost::noncopyable>("daeEventPortConnection", DOCSTR_daeEventPortConnection, no_init)
        .add_property("PortFrom",   make_function(&daepython::daePortConnection_GetPortFrom, return_internal_reference<>()), DOCSTR_daeEventPortConnection_PortFrom)
        .add_property("PortTo",     make_function(&daepython::daePortConnection_GetPortTo, return_internal_reference<>()), DOCSTR_daeEventPortConnection_PortTo)
            
        .def("__str__",				&daepython::daeEventPortConnection__str__)
        .def("__repr__",			&daepython::daeEventPortConnection__repr__)
    ;
    
	class_<daeState, bases<daeObject>, boost::noncopyable>("daeState", DOCSTR_daeState, no_init)
		.add_property("Equations",			 &daepython::daeState_GetEquations,          DOCSTR_daeState_Equations)
		.add_property("NestedSTNs",			 &daepython::daeState_GetNestedSTNs,         DOCSTR_daeState_NestedSTNs)
        .add_property("OnEventActions",      &daepython::daeState_GetOnEventActions,     DOCSTR_daeState_OnEventActions)
        .add_property("OnConditionActions",  &daepython::daeState_GetOnConditionActions, DOCSTR_daeState_OnConditionActions)
            
        .def("__str__",				&daepython::daeState__str__)
        .def("__repr__",			&daepython::daeState__repr__)
    ;

    class_<daeSTN, bases<daeObject>, boost::noncopyable>("daeSTN", DOCSTR_daeSTN, no_init)
        .add_property("ActiveState",	&daeSTN::GetActiveState2, &daeSTN::SetActiveState2, DOCSTR_daeSTN_ActiveState)
        .add_property("States",			&daepython::GetStatesSTN, DOCSTR_daeSTN_States)
        
        .def("__str__",				&daepython::daeSTN__str__)
        .def("__repr__",			&daepython::daeSTN__repr__)
        ;

	class_<daeIF, bases<daeSTN>, boost::noncopyable>("daeIF", DOCSTR_daeIF, no_init)
        .def("__str__",				&daepython::daeIF__str__)
        .def("__repr__",			&daepython::daeIF__repr__)
		;
	
	class_<daepython::daeScalarExternalFunctionWrapper, boost::noncopyable>("daeScalarExternalFunction", DOCSTR_daeScalarExternalFunction, no_init)
        .def(init<const string&, daeModel*, const unit&, boost::python::dict>(( arg("self"), 
                                                                                arg("name"), 
                                                                                arg("parentModel"),
                                                                                arg("units"),
                                                                                arg("arguments")
                                                                              ), DOCSTR_daeScalarExternalFunction_init))
        .add_property("Name",	&daepython::daeScalarExternalFunctionWrapper::GetName, DOCSTR_daeScalarExternalFunction_Name)

        .def("Calculate",	pure_virtual(&daepython::daeScalarExternalFunctionWrapper::Calculate_),
                            ( arg("self"), arg("values") ), DOCSTR_daeScalarExternalFunction_Calculate)
		.def("__call__",	&daeScalarExternalFunction::operator(), ( arg("self") ), DOCSTR_daeScalarExternalFunction_call)
        .def("__str__",		&daepython::daeScalarExternalFunctionWrapper::__str__)
        .def("__repr__",	&daepython::daeScalarExternalFunctionWrapper::__repr__)
		;
	
	class_<daepython::daeVectorExternalFunctionWrapper, boost::noncopyable>("daeVectorExternalFunction", DOCSTR_daeVectorExternalFunction, no_init)
        .def(init<const string&, daeModel*, const unit&, size_t, boost::python::dict>(( arg("self"), 
                                                                                        arg("name"), 
                                                                                        arg("parentModel"), 
                                                                                        arg("units"),
                                                                                        arg("numberOfResults"),
                                                                                        arg("arguments")
                                                                                      ), DOCSTR_daeVectorExternalFunction_init))
        .add_property("Name",	         &daepython::daeVectorExternalFunctionWrapper::GetName)
        .add_property("NumberOfResults", &daepython::daeVectorExternalFunctionWrapper::GetNumberOfResults)

        .def("Calculate",	pure_virtual(&daepython::daeVectorExternalFunctionWrapper::Calculate_), 
                            ( arg("self"), arg("values") ), DOCSTR_daeVectorExternalFunction_Calculate)
		.def("__call__",	&daeVectorExternalFunction::operator(), ( arg("self") ), DOCSTR_daeVectorExternalFunction_call)
        .def("__str__",		&daepython::daeVectorExternalFunctionWrapper::__str__)
        .def("__repr__",	&daepython::daeVectorExternalFunctionWrapper::__repr__)
		;


/**************************************************************
	daeLog
***************************************************************/
	class_<daepython::daeLogWrapper, boost::noncopyable>("daeLog_t", DOCSTR_daeLog_t, no_init)
        .add_property("Enabled",		&daeLog_t::GetEnabled,		 &daeLog_t::SetEnabled, DOCSTR_daeLog_t_Enabled)
		.add_property("PrintProgress",	&daeLog_t::GetPrintProgress, &daeLog_t::SetPrintProgress, DOCSTR_daeLog_t_PrintProgress)
		.add_property("Indent",			&daeLog_t::GetIndent,		 &daeLog_t::SetIndent, DOCSTR_daeLog_t_Indent)
		.add_property("Progress",		&daeLog_t::GetProgress,		 &daeLog_t::SetProgress, DOCSTR_daeLog_t_Progress)
		.add_property("IndentString",	&daeLog_t::GetIndentString, DOCSTR_daeLog_t_IndentString)
        .add_property("PercentageDone",	&daeLog_t::GetPercentageDone, DOCSTR_daeLog_t_PercentageDone)
        .add_property("ETA",			&daeLog_t::GetETA, DOCSTR_daeLog_t_ETA)

		.def("Message",			pure_virtual(&daeLog_t::Message), ( arg("self"), 
                                                                    arg("message"), 
                                                                    arg("severity") 
                                                                  ), DOCSTR_daeLog_t_Message)
        .def("JoinMessages",	pure_virtual(&daeLog_t::JoinMessages),   
                                ( arg("self"), arg("delimiter") = "\n" ), DOCSTR_daeLog_t_JoinMessages)
		.def("IncreaseIndent",	pure_virtual(&daeLog_t::IncreaseIndent), 
                                ( arg("self"), arg("offset") ), DOCSTR_daeLog_t_IncreaseIndent)
		.def("DecreaseIndent",	pure_virtual(&daeLog_t::DecreaseIndent), 
                                ( arg("self"), arg("offset") ), DOCSTR_daeLog_t_DecreaseIndent)
		;

	class_<daepython::daeBaseLogWrapper, bases<daeLog_t>, boost::noncopyable>("daeBaseLog", DOCSTR_daeBaseLog, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeBaseLog_init)) 
            
		.def("Message",				&daeLog_t::Message,     &daepython::daeBaseLogWrapper::def_Message, ( arg("self"), 
                                                                                                          arg("message"), 
                                                                                                          arg("severity") 
                                                                                                        ), DOCSTR_daeBaseLog_Message)
	    .def("SetProgress",			&daeLog_t::SetProgress, &daepython::daeBaseLogWrapper::def_SetProgress, 
                                    ( arg("self"), arg("progress") ), DOCSTR_daeBaseLog_SetProgress)
		.def("IncreaseIndent",		&daeBaseLog::IncreaseIndent, ( arg("self"), arg("offset") ), DOCSTR_daeBaseLog_IncreaseIndent)
		.def("DecreaseIndent",		&daeBaseLog::DecreaseIndent, ( arg("self"), arg("offset") ), DOCSTR_daeBaseLog_DecreaseIndent)
		;

    
    class_<daepython::daeDelegateLogWrapper, bases<daeBaseLog>, boost::noncopyable>("daeDelegateLog", DOCSTR_daeDelegateLog, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeDelegateLog_init)) 
            
        .def("Message",	 &daeLog_t::Message,  &daepython::daeDelegateLogWrapper::def_Message, ( arg("self"), 
                                                                                                arg("message"), 
                                                                                                arg("severity") 
                                                                                              ), DOCSTR_daeDelegateLog_Message)
        .def("AddLog", &daeDelegateLog::AddLog, ( arg("self"), arg("log") ), DOCSTR_daeDelegateLog_AddLog)
        .add_property("Logs",  &daepython::daeDelegateLogWrapper::GetLogs, DOCSTR_daeDelegateLog_Logs)
        ;
    
	class_<daepython::daeFileLogWrapper, bases<daeBaseLog>, boost::noncopyable>("daeFileLog", DOCSTR_daeFileLog, no_init)
        .def(init<string>(( arg("self"), arg("filename") ), DOCSTR_daeFileLog_init))
            
		.def("Message",	&daeLog_t::Message, &daepython::daeFileLogWrapper::def_Message, ( arg("self"), 
                                                                                          arg("message"), 
                                                                                          arg("severity") 
                                                                                        ), DOCSTR_daeFileLog_Message)
		;

	class_<daepython::daeStdOutLogWrapper, bases<daeBaseLog>, boost::noncopyable>("daeStdOutLog", DOCSTR_daeStdOutLog, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeStdOutLog_init))
            
		.def("Message",	&daeLog_t::Message, &daepython::daeStdOutLogWrapper::def_Message, ( arg("self"), 
                                                                                            arg("message"), 
                                                                                            arg("severity") 
                                                                                          ), DOCSTR_daeStdOutLog_Message)
		;

	class_<daepython::daeTCPIPLogWrapper, bases<daeBaseLog>, boost::noncopyable>("daeTCPIPLog", DOCSTR_daeTCPIPLog, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeTCPIPLog_init))
            
        //.def_readonly("Port",         &daeTCPIPLog::m_strIPAddress)
        //.def_readonly("TCPIPAddress", &daeTCPIPLog::m_nPort)

        .def("Connect",     &daeTCPIPLog::Connect, ( arg("self"), arg("tcpipAddress"), arg("port") ), DOCSTR_daeTCPIPLog_Connect)
        .def("IsConnected", &daeTCPIPLog::IsConnected, ( arg("self") ), DOCSTR_daeTCPIPLog_IsConnected)
        .def("Disconnect",  &daeTCPIPLog::Disconnect, ( arg("self") ), DOCSTR_daeTCPIPLog_Disconnect)
		.def("Message",     &daeLog_t::Message, &daepython::daeTCPIPLogWrapper::def_Message, ( arg("self"), 
                                                                                               arg("message"), 
                                                                                               arg("severity") 
                                                                                             ), DOCSTR_daeTCPIPLog_Message)
		;

	class_<daepython::daeTCPIPLogServerWrapper, boost::noncopyable>("daeTCPIPLogServer", DOCSTR_daeTCPIPLogServer, no_init)
        .def(init<int>(( arg("self"), arg("port") ), DOCSTR_daeTCPIPLogServer_init))
            
        .add_property("Port",   &daeTCPIPLogServer::GetPort)
            
        .def("Start",           &daeTCPIPLogServer::Start, ( arg("self") ), DOCSTR_daeTCPIPLogServer_Start)
        .def("Stop",            &daeTCPIPLogServer::Stop,  ( arg("self") ), DOCSTR_daeTCPIPLogServer_Stop)
		.def("MessageReceived",	&daeTCPIPLogServer::MessageReceived, &daepython::daeTCPIPLogServerWrapper::def_MessageReceived, 
                                ( arg("self"), arg("message") ), DOCSTR_daeTCPIPLogServer_MessageReceived)
		;
}
