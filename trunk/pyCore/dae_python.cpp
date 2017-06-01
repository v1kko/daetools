#include "stdafx.h"
#include "python_wraps.h"
#include "docstrings.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
//#include <noprefix.h>
using namespace boost::python;

// Temporary workaround for Visual Studio 2015 update 3
//  Error   LNK2019 unresolved external symbol "class ClassName const volatile * __cdecl boost::get_pointer<class ClassName const volatile *>(...)
#if _MSC_FULL_VER  == 190024210
#if (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
namespace boost
{
#define POINTER_CONVERSION(CLASS_NAME)   template <> CLASS_NAME const volatile * get_pointer(class CLASS_NAME const volatile *c) {return c;}

POINTER_CONVERSION(adNode)
POINTER_CONVERSION(adNodeArray)
POINTER_CONVERSION(condNode)
POINTER_CONVERSION(adouble)
POINTER_CONVERSION(adouble_array)
POINTER_CONVERSION(daeEquation)
POINTER_CONVERSION(daeObject)
POINTER_CONVERSION(daeCondition)
POINTER_CONVERSION(daeEventPort)
POINTER_CONVERSION(daeVariableWrapper)
POINTER_CONVERSION(daeState)
POINTER_CONVERSION(daeSTN)
POINTER_CONVERSION(daeVariableType)
POINTER_CONVERSION(daeDomain)
POINTER_CONVERSION(daeVariable)
POINTER_CONVERSION(daeParameter)
POINTER_CONVERSION(daeScalarExternalFunction)
POINTER_CONVERSION(daeVectorExternalFunction)
POINTER_CONVERSION(daeDistributedEquationDomainInfo)
POINTER_CONVERSION(daeLog_t)
POINTER_CONVERSION(daeMatrix<adouble>)
POINTER_CONVERSION(daeArray<adouble>)
POINTER_CONVERSION(daeConfig)
POINTER_CONVERSION(daeAction)
POINTER_CONVERSION(daeEquationExecutionInfo)
POINTER_CONVERSION(daeOnEventActions)
POINTER_CONVERSION(daeOnConditionActions)
POINTER_CONVERSION(daePort)
POINTER_CONVERSION(daeModel)
POINTER_CONVERSION(daePortConnection)
POINTER_CONVERSION(daeEventPortConnection)
POINTER_CONVERSION(daePortArray)
POINTER_CONVERSION(daeModelArray)
POINTER_CONVERSION(dae::tpp::daeThermoPhysicalPropertyPackage_t)
}
#endif
#endif

BOOST_PYTHON_MODULE(pyCore)
{
    //import_array();
    //boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    docstring_options doc_options(true, true, false);

/**************************************************************
    Enums
***************************************************************/
    enum_<daeeDomainType>("daeeDomainType")
        .value("eDTUnknown",        dae::core::eDTUnknown)
        .value("eArray",            dae::core::eArray)
        .value("eStructuredGrid",   dae::core::eStructuredGrid)
        .value("eUnstructuredGrid",	dae::core::eUnstructuredGrid)
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
        .value("eUpwindCCFV",	dae::core::eUpwindCCFV)
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

    enum_<daeeSTNType>("daeeSTNType")
        .value("eSTNTUnknown",	dae::core::eSTNTUnknown)
        .value("eSTN",          dae::core::eSTN)
        .value("eIF",			dae::core::eIF)
        .export_values()
    ;

    enum_<daeeDomainIndexType>("daeeDomainIndexType")
        .value("eDITUnknown",					dae::core::eDITUnknown)
        .value("eConstantIndex",				dae::core::eConstantIndex)
        .value("eLastPointInDomain",            dae::core::eLastPointInDomain)
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
        .value("eSinh",     dae::core::eSinh)
        .value("eCosh",     dae::core::eCosh)
        .value("eTanh",     dae::core::eTanh)
        .value("eArcSinh",	dae::core::eArcSinh)
        .value("eArcCosh",	dae::core::eArcCosh)
        .value("eArcTanh",	dae::core::eArcTanh)
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
        .value("eArcTan2",      dae::core::eArcTan2)
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

    enum_<daeeThermoPackageBasis>("daeeThermoPackageBasis")
        .value("eMole",	          dae::tpp::eMole)
        .value("eMass",	          dae::tpp::eMass)
        .value("eUndefinedBasis", dae::tpp::eUndefinedBasis)
        .export_values()
    ;

    enum_<daeeThermoPackagePhase>("daeeThermoPackagePhase")
        .value("etppPhaseUnknown",	dae::tpp::etppPhaseUnknown)
        .value("eVapor",            dae::tpp::eVapor)
        .value("eLiquid",           dae::tpp::eLiquid)
        .value("eSolid",            dae::tpp::eSolid)
        .export_values()
    ;

    class_< std::vector<std::string> >("vector_string")
        .def(vector_indexing_suite< std::vector<std::string> >())
    ;
    class_< std::vector<double> >("vector_double")
        .def(vector_indexing_suite< std::vector<double> >())
    ;
    class_< std::vector<float> >("vector_float")
        .def(vector_indexing_suite< std::vector<float> >())
    ;
    class_< std::vector<unsigned int> >("vector_uint")
        .def(vector_indexing_suite< std::vector<unsigned int> >())
    ;
    class_< std::vector<unsigned long> >("vector_ulong")
        .def(vector_indexing_suite< std::vector<unsigned long> >())
    ;
    class_< std::vector<int> >("vector_int")
        .def(vector_indexing_suite< std::vector<int> >())
    ;
    class_< std::vector<long> >("vector_long")
        .def(vector_indexing_suite< std::vector<long> >())
    ;
    class_< std::map<size_t,size_t> >("map_ulong_ulong")
        .def(map_indexing_suite< std::map<size_t,size_t> >())
    ;
    class_< std::map<std::string, daeEquationsIndexes> >("map_string_daeEquationsIndexes")
        .def(map_indexing_suite< std::map<std::string, daeEquationsIndexes> >())
    ;
    class_< std::map< size_t,std::vector<size_t> > >("map_ulong_vector_ulong")
        .def(map_indexing_suite< std::map< size_t,std::vector<size_t> > >())
    ;

/**************************************************************
    Global functions
***************************************************************/
    def("daeSetConfigFile",	&daeSetConfigFile);
    def("daeGetConfig",		&daepython::pydaeGetConfig);
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

    boost::python::scope().attr("cnSomePointsAssigned")                 = cnSomePointsAssigned;
    boost::python::scope().attr("cnSomePointsDifferential")             = cnSomePointsDifferential;
    boost::python::scope().attr("cnMixedAlgebraicAssignedDifferential") = cnMixedAlgebraicAssignedDifferential;

/**************************************************************
    Classes
***************************************************************/
    class_<daeEquationsIndexes, boost::noncopyable>("daeEquationsIndexes", DOCSTR_daeEquationsIndexes, no_init)
        .def_readonly("OverallIndexes_Equations",  &daeEquationsIndexes::m_mapOverallIndexes_Equations)
        .def_readonly("OverallIndexes_STNs",       &daeEquationsIndexes::m_mapOverallIndexes_STNs)
    ;

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

    class_<daeConfig>("daeConfig", DOCSTR_daeConfig_, no_init)
        .add_property("ConfigFileName",	&daeConfig::GetConfigFileName, DOCSTR_daeConfig_)

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
        .def(init< optional<daeModel*> >((arg("self"), arg("model") = NULL)))
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
        .def_readonly("Order",          &adSetupTimeDerivativeNode::m_nOrder)
        .add_property("Variable",       make_function(&daepython::adSetupTimeDerivativeNode_Variable, return_internal_reference<>()))
        .add_property("DomainIndexes",	&daepython::adSetupTimeDerivativeNode_Domains)
    ;

    class_<adSetupPartialDerivativeNode, bases<adNode>, boost::noncopyable>("adSetupPartialDerivativeNode", no_init)
        .def_readonly("Order",          &adSetupPartialDerivativeNode::m_nOrder)
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
        .add_property("ExternalFunction",  make_function(&daepython::adScalarExternalFunctionNode_ExternalFunction, return_internal_reference<>()))
    ;

    class_<adVectorExternalFunctionNode, bases<adNode>, boost::noncopyable>("adVectorExternalFunctionNode", no_init)
        .add_property("ExternalFunction",  make_function(&daepython::adVectorExternalFunctionNode_ExternalFunction, return_internal_reference<>()))
    ;

    class_<adFEMatrixItemNode, bases<adNode>, boost::noncopyable>("adFEMatrixItemNode", no_init)
        .def_readonly("MatrixName",   &adFEMatrixItemNode::m_strMatrixName)
        .def_readonly("Row",          &adFEMatrixItemNode::m_row)
        .def_readonly("Column",       &adFEMatrixItemNode::m_column)
        .add_property("Value",        &daepython::adFEMatrixItemNode_Value)
    ;

    class_<adFEVectorItemNode, bases<adNode>, boost::noncopyable>("adFEVectorItemNode", no_init)
        .def_readonly("VectorName",   &adFEVectorItemNode::m_strVectorName)
        .def_readonly("Row",          &adFEVectorItemNode::m_row)
        .add_property("Value",        &daepython::adFEVectorItemNode_Value)
    ;

    class_<adDomainIndexNode, bases<adNode>, boost::noncopyable>("adDomainIndexNode", no_init)
        .def_readonly("Index",   &adDomainIndexNode::m_nIndex)
        .add_property("Domain",  make_function(&daepython::adDomainIndexNode_Domain, return_internal_reference<>()))
        .add_property("Value",   &daepython::adDomainIndexNode_Value)
    ;

    class_<adRuntimeParameterNode, bases<adNode>, boost::noncopyable>("adRuntimeParameterNode", no_init)
        .add_property("Value",          &daepython::adRuntimeParameterNode_Value)
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
        .def_readonly("Order",          &adRuntimeTimeDerivativeNode::m_nOrder)
        .def_readonly("OverallIndex",   &adRuntimeTimeDerivativeNode::m_nOverallIndex)
        .def_readonly("BlockIndex",     &adRuntimeTimeDerivativeNode::m_nBlockIndex)
        .add_property("Variable",       make_function(&daepython::adRuntimeTimeDerivativeNode_Variable, return_internal_reference<>()))
        .add_property("DomainIndexes",	&daepython::adRuntimeTimeDerivativeNode_Domains)
    ;

    class_<adRuntimeSpecialFunctionForLargeArraysNode, bases<adNode>, boost::noncopyable>("adRuntimeSpecialFunctionForLargeArraysNode", no_init)
        .def_readonly("Function",	&adRuntimeSpecialFunctionForLargeArraysNode::eFunction)
        .add_property("Nodes",      &daepython::adRuntimeSpecialFunctionForLargeArraysNode_RuntimeNodes)
    ;

    class_<adThermoPhysicalPropertyPackageScalarNode, bases<adNode>, boost::noncopyable>("adThermoPhysicalPropertyPackageScalarNode", no_init)
    ;

    class_<adThermoPhysicalPropertyPackageArrayNode, bases<adNode>, boost::noncopyable>("adThermoPhysicalPropertyPackageArrayNode", no_init)
    ;

    class_<adCustomNodeArray, bases<adNodeArray>, boost::noncopyable>("adCustomNodeArray", no_init)
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

    class_<daeCondition>("daeCondition", DOCSTR_daeCondition, no_init)
        // daeSerializable part
        .add_property("ID",             &daeCondition::GetID)
        .add_property("Version",        &daeCondition::GetVersion)
        .add_property("Library",        &daeCondition::GetLibrary)

        .add_property("EventTolerance",	&daeCondition::GetEventTolerance, &daeCondition::SetEventTolerance,                DOCSTR_daeCondition_EventTolerance)
        .add_property("SetupNode",      make_function(&daeCondition::getSetupNodeRawPtr, return_internal_reference<>()),   DOCSTR_daeCondition_SetupNode)
        .add_property("RuntimeNode",    make_function(&daeCondition::getRuntimeNodeRawPtr, return_internal_reference<>()), DOCSTR_daeCondition_RuntimeNode)
        .add_property("Expressions",    &daepython::daeCondition_GetExpressions,                                           DOCSTR_daeCondition_Expressions)

        .def(self | self) // Bitwise operator OR is used in daetools as a logical comparison operator OR
        .def(self & self) // Bitwise operator AND is  used in daetools as a logical comparison operator AND
        .def(~ self)      // Bitwise operator INVERT is used in daetools as a logical negation operator NOT

        .def("SetupNodeAsPlainText",    &daeCondition::SetupNodeAsPlainText)
        .def("SetupNodeAsLatex",        &daeCondition::SetupNodeAsLatex)
        .def("RuntimeNodeAsPlainText",  &daeCondition::RuntimeNodeAsPlainText)
        .def("RuntimeNodeAsLatex",      &daeCondition::RuntimeNodeAsLatex)

        .def("__str__",  &daepython::daeCondition__str__)
        .def("__repr__", &daepython::daeCondition__repr__)
    ;

    class_<adouble>("adouble", DOCSTR_adouble, no_init)
        .def(init< optional<real_t, real_t, bool, adNode*> >(( arg("self"), arg("value") = 0.0, arg("derivative") = 0.0, arg("gatherInfo") = false, arg("node") = NULL ), DOCSTR_adouble_init))

        .add_property("Value",		&adouble::getValue,      &adouble::setValue, DOCSTR_adouble_Value)
        .add_property("Derivative",	&adouble::getDerivative, &adouble::setDerivative, DOCSTR_adouble_Derivative)
        .add_property("Node",       make_function(&adouble::getNodeRawPtr, return_internal_reference<>()), DOCSTR_adouble_Node)
        .add_property("GatherInfo",	&adouble::getGatherInfo, &adouble::setGatherInfo, DOCSTR_adouble_GatherInfo)

        .def("NodeAsPlainText", &adouble::NodeAsPlainText)
        .def("NodeAsLatex",     &adouble::NodeAsLatex)

        .def("__str__",  &daepython::adouble__str__)
        .def("__repr__", &daepython::adouble__repr__)

        .def(- self) // unary -
        .def(+ self) // unary +

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

        // Functions __bool__ and __nonzero__ are also used by operators __and__, __or__ and __not__
        // For instance, the expression 'a1 and a2' is evaluated as:
        //  - in python3: 'bool(a1) and bool(a2)'
        //  - in python2: 'nonzero(a1) and nonzero(a2)'
        .def("__bool__",    &daepython::ad_bool)     // bool(adouble) used in python 3.x
        .def("__nonzero__", &daepython::ad_nonzero)  // nonzero(adouble) used in python 2.x

        // True division operator (/), mostly used by numpy
        .def("__truediv__",  &daepython::ad_true_divide1)   // adouble / adouble
        .def("__truediv__",  &daepython::ad_true_divide2)   // adouble / real_t
        .def("__truediv__",  &daepython::ad_true_divide3)   // real_t  / adouble

        // Floor division operator (//), mostly used by numpy
        .def("__floordiv__", &daepython::ad_floor_divide1)  // adouble // adouble
        .def("__floordiv__", &daepython::ad_floor_divide2)  // adouble // real_t
        .def("__floordiv__", &daepython::ad_floor_divide3)  // real_t  // adouble

    // Math. functions declared as members to enable numpy support
    // For instance, the following will be possible to write in python for scalars:
    //   y = numpy.exp(x())
    // or for arrays:
    //   x = np.empty(n, dtype=object)
    //   x[:] = [self.x(i) for i in range(n)]
    //   y = numpy.exp(x)
        .def("exp",     &daepython::ad_exp)
        .def("log",     &daepython::ad_log)
        .def("log10",   &daepython::ad_log10)
        .def("sqrt",    &daepython::ad_sqrt)
        .def("sin",     &daepython::ad_sin)
        .def("cos",     &daepython::ad_cos)
        .def("tan",     &daepython::ad_tan)
        .def("arcsin",  &daepython::ad_asin)
        .def("arccos",  &daepython::ad_acos)
        .def("arctan",  &daepython::ad_atan)

        .def("sinh",    &daepython::ad_sinh)
        .def("cosh",    &daepython::ad_cosh)
        .def("tanh",    &daepython::ad_tanh)
        .def("arcsinh", &daepython::ad_asinh)
        .def("arccosh", &daepython::ad_acosh)
        .def("arctanh", &daepython::ad_atanh)
        .def("arctan2", &daepython::ad_atan2)
        .def("erf",     &daepython::ad_erf)

        .def("abs",    &daepython::ad_abs)
        .def("fabs",   &daepython::ad_abs)
        .def("ceil",   &daepython::ad_ceil)
        .def("floor",  &daepython::ad_floor)
    ;

    def("Exp",   &daepython::ad_exp);
    def("Log",   &daepython::ad_log);
    def("Log10", &daepython::ad_log10);
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
    def("Erf",   &daepython::ad_erf);

    def("Ceil",  &daepython::ad_ceil);
    def("Floor", &daepython::ad_floor);
    def("Pow",   &daepython::ad_pow1);
    def("Pow",   &daepython::ad_pow2);
    def("Pow",   &daepython::ad_pow3);

    def("Abs",   &daepython::ad_abs);
    def("Min",   &daepython::ad_min1);
    def("Min",   &daepython::ad_min2);
    def("Min",   &daepython::ad_min3);
    def("Max",   &daepython::ad_max1);
    def("Max",   &daepython::ad_max2);
    def("Max",   &daepython::ad_max3);

    def("dt",	 &daepython::ad_dt, (arg("ad")), DOCSTR_dt);
    def("d",	 &daepython::ad_d,  (arg("ad"), arg("domain"), arg("discretizationMethod") = eCFDM, arg("options") = boost::python::dict()), DOCSTR_d);
    def("d2",    &daepython::ad_d2, (arg("ad"), arg("domain"), arg("discretizationMethod") = eCFDM, arg("options") = boost::python::dict()), DOCSTR_d2);

    def("Time",             &Time,                                           DOCSTR_Time);
    def("Constant",         &daepython::ad_Constant_c,      (arg("value")),  DOCSTR_Constant_c);
    def("Constant",         &daepython::ad_Constant_q,      (arg("value")),  DOCSTR_Constant_q);
    def("Array",            &daepython::adarr_Array,        (arg("values")), DOCSTR_Array);


    class_<adouble_array>("adouble_array", DOCSTR_adouble_array, no_init)
        .def(init< optional<bool, adNodeArray*> >( ( arg("self"), arg("gatherInfo") = false, arg("node") = NULL ), DOCSTR_adouble_array_init))

        .add_property("GatherInfo",	 &adouble_array::getGatherInfo,	&adouble_array::setGatherInfo,                DOCSTR_adouble_array_GatherInfo)
        .add_property("Node",        make_function(&adouble_array::getNodeRawPtr, return_internal_reference<>()), DOCSTR_adouble_array_Node)

       /*
        .def("Resize",      &adouble_array::Resize, ( arg("self"), arg("newSize") ),                DOCSTR_adouble_array_Resize)
        .def("__len__",     &adouble_array::GetSize, ( arg("self") ),                               DOCSTR_adouble_array_len)
        .def("__getitem__", &adouble_array::GetItem, ( arg("self"), arg("index") ),                 DOCSTR_adouble_array_getitem)
        .def("__setitem__", &adouble_array::SetItem, ( arg("self"), arg("index"), arg("value") ),   DOCSTR_adouble_array_setitem)
        .def("items",       range< return_value_policy<copy_non_const_reference> >(&adouble_array::begin, &adouble_array::end), DOCSTR_adouble_array_items)
       */

        .def("__call__",    &daepython::adouble_array__call__, ( arg("self"), arg("index") ), DOCSTR_adouble_array_call)

        .def("NodeAsPlainText", &adouble_array::NodeAsPlainText)
        .def("NodeAsLatex",     &adouble_array::NodeAsLatex)

        .def("__str__",  &daepython::adouble_array__str__)
        .def("__repr__", &daepython::adouble_array__repr__)

        .def("FromList",        &daepython::adarr_FromList,       arg("values"), DOCSTR_adouble_array_FromList)
        .staticmethod("FromList")

        .def("FromNumpyArray",  &daepython::adarr_FromNumpyArray, arg("values"), DOCSTR_adouble_array_FromNumpyArray)
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

        // Functions __bool__ and __nonzero__ are also used by operators __and__, __or__ and __not__
        // For instance, the expression 'a1 and a2' is evaluated as:
        //  - in python3: 'bool(a1) and bool(a2)'
        //  - in python2: 'nonzero(a1) and nonzero(a2)'
        .def("__bool__",    &daepython::adarr_bool)     // bool(adouble_array) used in python 3.x
        .def("__nonzero__", &daepython::adarr_nonzero)  // nonzero(adouble_array) used in python 2.x

        // True division operator (/), mostly used by numpy
        .def("__truediv__",  &daepython::adarr_true_divide1)   // adouble_array / adouble_array
        .def("__truediv__",  &daepython::adarr_true_divide2)   // adouble_array / real_t
        .def("__truediv__",  &daepython::adarr_true_divide3)   // real_t  / adouble_array

        // Floor division operator (//), mostly used by numpy
        .def("__floordiv__", &daepython::adarr_floor_divide1)  // adouble_array // adouble
        .def("__floordiv__", &daepython::adarr_floor_divide2)  // adouble_array // real_t
        .def("__floordiv__", &daepython::adarr_floor_divide3)  // real_t  // adouble_array

        .def("exp",     &daepython::adarr_exp)
        .def("log",     &daepython::adarr_log)
        .def("log10",   &daepython::adarr_log10)
        .def("sqrt",    &daepython::adarr_sqrt)
        .def("sin",     &daepython::adarr_sin)
        .def("cos",     &daepython::adarr_cos)
        .def("tan",     &daepython::adarr_tan)
        .def("arcsin",  &daepython::adarr_asin)
        .def("arccos",  &daepython::adarr_acos)
        .def("arctan",  &daepython::adarr_atan)

        .def("sinh",    &daepython::adarr_sinh)
        .def("cosh",    &daepython::adarr_cosh)
        .def("tanh",    &daepython::adarr_tanh)
        .def("arcsinh", &daepython::adarr_asinh)
        .def("arccosh", &daepython::adarr_acosh)
        .def("arctanh", &daepython::adarr_atanh)
        .def("arctan2", &daepython::adarr_atan2)
        .def("erf",     &daepython::adarr_erf)

        .def("abs",    &daepython::adarr_abs)
        .def("fabs",   &daepython::adarr_abs)
        .def("ceil",   &daepython::adarr_ceil)
        .def("floor",  &daepython::adarr_floor)
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

    def("Sinh",  &daepython::adarr_sinh);
    def("Cosh",  &daepython::adarr_cosh);
    def("Tanh",  &daepython::adarr_tanh);
    def("ASinh", &daepython::adarr_asinh);
    def("ACosh", &daepython::adarr_acosh);
    def("ATanh", &daepython::adarr_atanh);
    def("ATan2", &daepython::adarr_atan2);
    def("Erf",   &daepython::adarr_erf);

    def("Sum",		 &daepython::adarr_sum,      (arg("adarray")), DOCSTR_Sum);
    def("Product",   &daepython::adarr_product,  (arg("adarray")), DOCSTR_Product);
    def("Integral",  &daepython::adarr_integral, (arg("adarray")), DOCSTR_Integral);
    def("Min",		 &daepython::adarr_min,      (arg("adarray")), DOCSTR_Min_adarr);
    def("Max",		 &daepython::adarr_max,      (arg("adarray")), DOCSTR_Max_adarr);
    def("Average",	 &daepython::adarr_average,  (arg("adarray")), DOCSTR_Average);

    def("dt_array",	 &daepython::ad_dt_array,    (arg("adarr")), DOCSTR_dt_array);
    def("d_array",	 &daepython::ad_d_array,     (arg("adarr"), arg("domain"), arg("discretizationMethod") = eCFDM, arg("options") = boost::python::dict()), DOCSTR_d_array);
    def("d2_array",	 &daepython::ad_d2_array,    (arg("adarr"), arg("domain"), arg("discretizationMethod") = eCFDM, arg("options") = boost::python::dict()), DOCSTR_d2_array);

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

    class_<daeFMI2Object_t>("daeFMI2Object_t", DOCSTR_daeFMI2Object_t)
        .def_readonly("reference",   &daeFMI2Object_t::reference,   DOCSTR_daeFMI2Object_t_reference)
        .def_readonly("name",        &daeFMI2Object_t::name,        DOCSTR_daeFMI2Object_t_name)
        .def_readonly("type",        &daeFMI2Object_t::type,        DOCSTR_daeFMI2Object_t_type)
        .def_readonly("description", &daeFMI2Object_t::description, DOCSTR_daeFMI2Object_t_description)
        .def_readonly("units",       &daeFMI2Object_t::units,       DOCSTR_daeFMI2Object_t_units)
        .def_readonly("indexes",     &daeFMI2Object_t::indexes,     DOCSTR_daeFMI2Object_t_indexes)
        // Only one valid, depending on "type"
        .add_property("parameter",   make_function(&daepython::daeFMI2Object_t_parameter, return_internal_reference<>()),
                                     DOCSTR_daeFMI2Object_t_parameter)
        .add_property("variable",    make_function(&daepython::daeFMI2Object_t_variable, return_internal_reference<>()),
                                     DOCSTR_daeFMI2Object_t_variable)
        .add_property("stn",         make_function(&daepython::daeFMI2Object_t_stn, return_internal_reference<>()),
                                     DOCSTR_daeFMI2Object_t_stn)
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
    def("daeGetStrippedName",            &daeGetStrippedName,              (arg("name")),                 DOCSTR_global_daeGetStrippedRelativeName);
    def("daeGetStrippedRelativeName",    &daeGetStrippedRelativeName,      (arg("parent"), arg("child")), DOCSTR_global_daeGetStrippedRelativeName);

    class_<daeDomainIndex>("daeDomainIndex", DOCSTR_daeDomainIndex, no_init)
        .def(init<size_t>(( arg("self"), arg("index") ), DOCSTR_daeDomainIndex_init1))
        .def(init<daeDomain*, int>(( arg("self"), arg("domain"), arg("dummy") ), DOCSTR_daeDomainIndex_init2))
        .def(init<daeDistributedEquationDomainInfo*>(( arg("self"), arg("dedi") ), DOCSTR_daeDomainIndex_init3))
        .def(init<daeDistributedEquationDomainInfo*, int>(( arg("self"), arg("dedi"), arg("increment") ), DOCSTR_daeDomainIndex_init4))
        .def(init<daeDomainIndex>(( arg("self"), arg("domainIndex") ), DOCSTR_daeDomainIndex_init5))

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
        //.add_property("DiscretizationMethod",	&daeDomain::GetDiscretizationMethod,DOCSTR_daeDomain_DiscretizationMethod)
        //.add_property("DiscretizationOrder",	&daeDomain::GetDiscretizationOrder, DOCSTR_daeDomain_DiscretizationOrder)
        .add_property("LowerBound",				&daeDomain::GetLowerBound,          DOCSTR_daeDomain_LowerBound)
        .add_property("UpperBound",				&daeDomain::GetUpperBound,          DOCSTR_daeDomain_UpperBound)
        .add_property("Units",					&daeDomain::GetUnits,               DOCSTR_daeDomain_Units)
        .add_property("Points",					&daepython::GetDomainPoints,
                                                &daepython::SetDomainPoints,        DOCSTR_daeDomain_Points)
        .add_property("Coordinates",			&daepython::GetDomainCoordinates,   DOCSTR_daeDomain_Coordinates)

        .def("__str__",	 					    &daepython::daeDomain__str__)
        .def("__repr__",						&daepython::daeDomain__repr__)

        .def("CreateArray",						&daeDomain::CreateArray, ( arg("self"),
                                                                           arg("noPoints")
                                                                         ), DOCSTR_daeDomain_CreateArray)
        .def("CreateStructuredGrid",			&daepython::qCreateStructuredGrid, ( arg("self"),
                                                                                     arg("numberOfIntervals"),
                                                                                     arg("qlowerBound"),
                                                                                     arg("qupperBound")
                                                                                   ), DOCSTR_daeDomain_CreateStructuredGrid)
        .def("CreateStructuredGrid",			&daepython::CreateStructuredGrid, ( arg("self"),
                                                                                    arg("numberOfIntervals"),
                                                                                    arg("lowerBound"),
                                                                                    arg("upperBound")
                                                                                  ), DOCSTR_daeDomain_CreateStructuredGrid)
        .def("CreateUnstructuredGrid",			&daepython::CreateUnstructuredGrid, ( arg("self"),
                                                                                      arg("coordinates")
                                                                                    ), DOCSTR_daeDomain_CreateUnstructuredGrid)
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

    class_<daeParameter, bases<daeObject>, boost::noncopyable>("daeParameter", DOCSTR_daeParameter, no_init)
        .def("__init__", make_constructor(&daepython::daeParameter_init1, default_call_policies(), ( arg("name"),
                                                                                                     arg("units"),
                                                                                                     arg("parentModel"),
                                                                                                     arg("description") = "",
                                                                                                     arg("domains") =  boost::python::list()
                                                                                                   ) ), DOCSTR_daeParameter_init1)
        .def("__init__", make_constructor(&daepython::daeParameter_init2, default_call_policies(), ( arg("name"),
                                                                                                     arg("units"),
                                                                                                     arg("parentPort"),
                                                                                                     arg("description") = "",
                                                                                                     arg("domains") =  boost::python::list()
                                                                                                   ) ), DOCSTR_daeParameter_init2)

        .add_property("Units",			&daeParameter::GetUnits, DOCSTR_daeParameter_Units)
        .add_property("Domains",		&daepython::daeParameter_GetDomains, DOCSTR_daeParameter_Domains)
        .add_property("ReportingOn",	&daeParameter::GetReportingOn,	&daeParameter::SetReportingOn, DOCSTR_daeParameter_ReportingOn)
        .add_property("npyValues",      &daepython::GetNumPyArrayParameter, DOCSTR_daeParameter_npyValues)
        .add_property("NumberOfPoints", &daeParameter::GetNumberOfPoints, DOCSTR_daeParameter_NumberOfPoints)

        .def("__repr__",				&daepython::daeParameter__repr__)
        .def("__str__", 				&daepython::daeParameter__str__)

        .def("DistributeOnDomain",	&daeParameter::DistributeOnDomain, ( arg("self"), arg("domain") ), DOCSTR_daeParameter_DistributeOnDomain)

        .def("GetValue", &daepython::lGetParameterValue)
        .def("GetValue", &daepython::GetParameterValue0)
        .def("GetValue", &daepython::GetParameterValue1)
        .def("GetValue", &daepython::GetParameterValue2)
        .def("GetValue", &daepython::GetParameterValue3)
        .def("GetValue", &daepython::GetParameterValue4)
        .def("GetValue", &daepython::GetParameterValue5)
        .def("GetValue", &daepython::GetParameterValue6)
        .def("GetValue", &daepython::GetParameterValue7)
        .def("GetValue", &daepython::GetParameterValue8)

        .def("SetValue", &daepython::lSetParameterValue)
        .def("SetValue", &daepython::SetParameterValue0)
        .def("SetValue", &daepython::SetParameterValue1)
        .def("SetValue", &daepython::SetParameterValue2)
        .def("SetValue", &daepython::SetParameterValue3)
        .def("SetValue", &daepython::SetParameterValue4)
        .def("SetValue", &daepython::SetParameterValue5)
        .def("SetValue", &daepython::SetParameterValue6)
        .def("SetValue", &daepython::SetParameterValue7)
        .def("SetValue", &daepython::SetParameterValue8)

        .def("GetQuantity", &daepython::lGetParameterQuantity)
        .def("GetQuantity", &daepython::GetParameterQuantity0)
        .def("GetQuantity", &daepython::GetParameterQuantity1)
        .def("GetQuantity", &daepython::GetParameterQuantity2)
        .def("GetQuantity", &daepython::GetParameterQuantity3)
        .def("GetQuantity", &daepython::GetParameterQuantity4)
        .def("GetQuantity", &daepython::GetParameterQuantity5)
        .def("GetQuantity", &daepython::GetParameterQuantity6)
        .def("GetQuantity", &daepython::GetParameterQuantity7)
        .def("GetQuantity", &daepython::GetParameterQuantity8)

        .def("SetValue", &daepython::lSetParameterQuantity)
        .def("SetValue", &daepython::SetParameterQuantity0)
        .def("SetValue", &daepython::SetParameterQuantity1)
        .def("SetValue", &daepython::SetParameterQuantity2)
        .def("SetValue", &daepython::SetParameterQuantity3)
        .def("SetValue", &daepython::SetParameterQuantity4)
        .def("SetValue", &daepython::SetParameterQuantity5)
        .def("SetValue", &daepython::SetParameterQuantity6)
        .def("SetValue", &daepython::SetParameterQuantity7)
        .def("SetValue", &daepython::SetParameterQuantity8)

        .def("SetValues", &daepython::lSetParameterValues)
        .def("SetValues", &daepython::SetParameterValues)
        .def("SetValues", &daepython::qSetParameterValues)

        .def("GetDomainsIndexesMap", &daepython::daeParameter_GetDomainsIndexesMap1,
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

    class_<daeVariable, bases<daeObject>, boost::noncopyable>("daeVariable", DOCSTR_daeVariable, no_init)
        .def("__init__", make_constructor(&daepython::daeVariable_init1, default_call_policies(), ( arg("name"),
                                                                                                    arg("variableType"),
                                                                                                    arg("parentModel"),
                                                                                                    arg("description") = "",
                                                                                                    arg("domains") =  boost::python::list()
                                                                                                  ) ), DOCSTR_daeVariable_init1)
        .def("__init__", make_constructor(&daepython::daeVariable_init2, default_call_policies(), ( arg("name"),
                                                                                                    arg("variableType"),
                                                                                                    arg("parentPort"),
                                                                                                    arg("description") = "",
                                                                                                    arg("domains") =  boost::python::list()
                                                                                                  ) ), DOCSTR_daeVariable_init2)

        .add_property("Domains",            &daepython::daeVariable_GetDomains, DOCSTR_daeVariable_Domains)
        .add_property("Type",               &daeVariable::GetType, DOCSTR_daeVariable_Type)
        .add_property("VariableType",       make_function(&daepython::daeVariable_GetVariableType, return_internal_reference<>()),
                                            DOCSTR_daeVariable_VariableType)
        .add_property("ReportingOn",        &daeVariable::GetReportingOn,
                                            &daeVariable::SetReportingOn,             DOCSTR_daeVariable_ReportingOn)
        .add_property("OverallIndex",       &daeVariable::GetOverallIndex,            DOCSTR_daeVariable_OverallIndex)
        .add_property("BlockIndexes",       &daepython::daeVariable_BlockIndexes,     DOCSTR_daeVariable_BlockIndexes)
        .add_property("NumberOfPoints",     &daeVariable::GetNumberOfPoints,          DOCSTR_daeVariable_NumberOfPoints)
        .add_property("npyValues",          &daepython::daeVariable_Values,           DOCSTR_daeVariable_npyValues)
        .add_property("npyTimeDerivatives", &daepython::daeVariable_TimeDerivatives,  DOCSTR_daeVariable_npyTimeDerivatives)
        .add_property("npyIDs",             &daepython::daeVariable_IDs,              DOCSTR_daeVariable_npyIDs)
        .add_property("npyGatheredIDs",     &daepython::daeVariable_GatheredIDs,      DOCSTR_daeVariable_npyGatheredIDs)

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

        .def("SetValue", &daepython::lSetVariableValue1)
        .def("SetValue", &daepython::lSetVariableValue2)
        .def("SetValue", &daepython::SetVariableValue0)
        .def("SetValue", &daepython::SetVariableValue1)
        .def("SetValue", &daepython::SetVariableValue2)
        .def("SetValue", &daepython::SetVariableValue3)
        .def("SetValue", &daepython::SetVariableValue4)
        .def("SetValue", &daepython::SetVariableValue5)
        .def("SetValue", &daepython::SetVariableValue6)
        .def("SetValue", &daepython::SetVariableValue7)
        .def("SetValue", &daepython::SetVariableValue8)

        .def("GetValue", &daepython::daeVariable_lGetVariableValue)
        .def("GetValue", &daepython::daeVariable_GetVariableValue0)
        .def("GetValue", &daepython::daeVariable_GetVariableValue1)
        .def("GetValue", &daepython::daeVariable_GetVariableValue2)
        .def("GetValue", &daepython::daeVariable_GetVariableValue3)
        .def("GetValue", &daepython::daeVariable_GetVariableValue4)
        .def("GetValue", &daepython::daeVariable_GetVariableValue5)
        .def("GetValue", &daepython::daeVariable_GetVariableValue6)
        .def("GetValue", &daepython::daeVariable_GetVariableValue7)
        .def("GetValue", &daepython::daeVariable_GetVariableValue8)

        .def("AssignValue", &daepython::lAssignValue1)
        .def("AssignValue", &daepython::lAssignValue2)
        .def("AssignValue", &daepython::AssignValue0)
        .def("AssignValue", &daepython::AssignValue1)
        .def("AssignValue", &daepython::AssignValue2)
        .def("AssignValue", &daepython::AssignValue3)
        .def("AssignValue", &daepython::AssignValue4)
        .def("AssignValue", &daepython::AssignValue5)
        .def("AssignValue", &daepython::AssignValue6)
        .def("AssignValue", &daepython::AssignValue7)
        .def("AssignValue", &daepython::AssignValue8)

        .def("ReAssignValue", &daepython::lReAssignValue1)
        .def("ReAssignValue", &daepython::lReAssignValue2)
        .def("ReAssignValue", &daepython::ReAssignValue0)
        .def("ReAssignValue", &daepython::ReAssignValue1)
        .def("ReAssignValue", &daepython::ReAssignValue2)
        .def("ReAssignValue", &daepython::ReAssignValue3)
        .def("ReAssignValue", &daepython::ReAssignValue4)
        .def("ReAssignValue", &daepython::ReAssignValue5)
        .def("ReAssignValue", &daepython::ReAssignValue6)
        .def("ReAssignValue", &daepython::ReAssignValue7)
        .def("ReAssignValue", &daepython::ReAssignValue8)

        .def("SetInitialGuess", &daepython::lSetInitialGuess1)
        .def("SetInitialGuess", &daepython::lSetInitialGuess2)
        .def("SetInitialGuess", &daepython::SetInitialGuess0)
        .def("SetInitialGuess", &daepython::SetInitialGuess1)
        .def("SetInitialGuess", &daepython::SetInitialGuess2)
        .def("SetInitialGuess", &daepython::SetInitialGuess3)
        .def("SetInitialGuess", &daepython::SetInitialGuess4)
        .def("SetInitialGuess", &daepython::SetInitialGuess5)
        .def("SetInitialGuess", &daepython::SetInitialGuess6)
        .def("SetInitialGuess", &daepython::SetInitialGuess7)
        .def("SetInitialGuess", &daepython::SetInitialGuess8)

        .def("SetInitialCondition", &daepython::lSetInitialCondition1)
        .def("SetInitialCondition", &daepython::lSetInitialCondition2)
        .def("SetInitialCondition", &daepython::SetInitialCondition0)
        .def("SetInitialCondition", &daepython::SetInitialCondition1)
        .def("SetInitialCondition", &daepython::SetInitialCondition2)
        .def("SetInitialCondition", &daepython::SetInitialCondition3)
        .def("SetInitialCondition", &daepython::SetInitialCondition4)
        .def("SetInitialCondition", &daepython::SetInitialCondition5)
        .def("SetInitialCondition", &daepython::SetInitialCondition6)
        .def("SetInitialCondition", &daepython::SetInitialCondition7)
        .def("SetInitialCondition", &daepython::SetInitialCondition8)

        .def("ReSetInitialCondition", &daepython::lReSetInitialCondition1)
        .def("ReSetInitialCondition", &daepython::lReSetInitialCondition2)
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

        .def("GetQuantity", &daepython::daeVariable_lGetVariableQuantity)
        .def("GetQuantity", &daepython::daeVariable_GetVariableQuantity0)
        .def("GetQuantity", &daepython::daeVariable_GetVariableQuantity1)
        .def("GetQuantity", &daepython::daeVariable_GetVariableQuantity2)
        .def("GetQuantity", &daepython::daeVariable_GetVariableQuantity3)
        .def("GetQuantity", &daepython::daeVariable_GetVariableQuantity4)
        .def("GetQuantity", &daepython::daeVariable_GetVariableQuantity5)
        .def("GetQuantity", &daepython::daeVariable_GetVariableQuantity6)
        .def("GetQuantity", &daepython::daeVariable_GetVariableQuantity7)
        .def("GetQuantity", &daepython::daeVariable_GetVariableQuantity8)

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

        .def("AssignValues",			&daepython::AssignValues2)
        .def("ReAssignValues",			&daepython::ReAssignValues2)
        .def("SetInitialGuesses",		&daepython::SetInitialGuesses2)
        .def("SetInitialConditions",	&daepython::SetInitialConditions2)
        .def("ReSetInitialConditions",	&daepython::ReSetInitialConditions2)

        .def("AssignValues",			&daepython::qAssignValues)
        .def("ReAssignValues",			&daepython::qReAssignValues)
        .def("SetInitialGuesses",		&daepython::qSetInitialGuesses)
        .def("SetInitialConditions",	&daepython::qSetInitialConditions)
        .def("ReSetInitialConditions",	&daepython::qReSetInitialConditions)

        .def("SetAbsoluteTolerances",	&daeVariable::SetAbsoluteTolerances)

        .def("GetDomainsIndexesMap",    &daepython::daeVariable_GetDomainsIndexesMap,
                                        ( arg("self"), arg("indexBase") ), DOCSTR_daeVariable_GetDomainIndexesMap)

        .def("dt", &daepython::Get_dt0)
        .def("dt", &daepython::Get_dt1)
        .def("dt", &daepython::Get_dt2)
        .def("dt", &daepython::Get_dt3)
        .def("dt", &daepython::Get_dt4)
        .def("dt", &daepython::Get_dt5)
        .def("dt", &daepython::Get_dt6)
        .def("dt", &daepython::Get_dt7)
        .def("dt", &daepython::Get_dt8)

        .def("d", &daepython::Get_d1, (arg("self"), arg("domain"), arg("index1")))
        .def("d", &daepython::Get_d2, (arg("self"), arg("domain"), arg("index1"), arg("index2")))
        .def("d", &daepython::Get_d3, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3")))
        .def("d", &daepython::Get_d4, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4")))
        .def("d", &daepython::Get_d5, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4"), arg("index5")))
        .def("d", &daepython::Get_d6, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4"), arg("index5"), arg("index6")))
        .def("d", &daepython::Get_d7, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4"), arg("index5"), arg("index6"), arg("index7")))
        .def("d", &daepython::Get_d8, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4"), arg("index5"), arg("index6"), arg("index7"), arg("index8")))

        .def("d2", &daepython::Get_d21, (arg("self"), arg("domain"), arg("index1")))
        .def("d2", &daepython::Get_d22, (arg("self"), arg("domain"), arg("index1"), arg("index2")))
        .def("d2", &daepython::Get_d23, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3")))
        .def("d2", &daepython::Get_d24, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4")))
        .def("d2", &daepython::Get_d25, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4"), arg("index5")))
        .def("d2", &daepython::Get_d26, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4"), arg("index5"), arg("index6")))
        .def("d2", &daepython::Get_d27, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4"), arg("index5"), arg("index6"), arg("index7")))
        .def("d2", &daepython::Get_d28, (arg("self"), arg("domain"), arg("index1"), arg("index2"), arg("index3"), arg("index4"), arg("index5"), arg("index6"), arg("index7"), arg("index8")))

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

    class_<daePort, bases<daeObject>, boost::noncopyable>("daePort", DOCSTR_daePort, no_init)
        .def(init<string, daeePortType, daeModel*, optional<string> >(( arg("self"),
                                                                        arg("name"),
                                                                        arg("type"),
                                                                        arg("parentModel"),
                                                                        arg("description") = ""
                                                                      ), DOCSTR_daePort_init))

        .add_property("Type",			&daePort::GetType,                  DOCSTR_daePort_Type)

        .add_property("Domains",		&daepython::daePort_GetDomains,     DOCSTR_daePort_Domains)
        .add_property("Parameters",		&daepython::daePort_GetParameters,  DOCSTR_daePort_Parameters)
        .add_property("Variables",		&daepython::daePort_GetVariables,   DOCSTR_daePort_Variables)

        .add_property("dictDomains",	&daepython::daePort_dictDomains,    DOCSTR_daePort_Domains)
        .add_property("dictParameters",	&daepython::daePort_dictParameters, DOCSTR_daePort_Parameters)
        .add_property("dictVariables",	&daepython::daePort_dictVariables,  DOCSTR_daePort_Variables)

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

        .add_property("Type",            &daeAction::GetType, DOCSTR_daeAction_Type)
        .add_property("STN",             make_function(&daeAction::GetSTN, return_internal_reference<>()), DOCSTR_daeAction_STN)
        .add_property("StateTo",         make_function(&daeAction::GetStateTo, return_internal_reference<>()), DOCSTR_daeAction_StateTo)
        .add_property("SendEventPort",   make_function(&daeAction::GetSendEventPort, return_internal_reference<>()), DOCSTR_daeAction_SendEventPort)
        .add_property("VariableWrapper", make_function(&daeAction::GetVariableWrapper, return_internal_reference<>()), DOCSTR_daeAction_VariableWrapper)
        .add_property("SetupNode",       make_function(&daeAction::getSetupNodeRawPtr, return_internal_reference<>()), DOCSTR_daeAction_SetupNode)
        .add_property("RuntimeNode",     make_function(&daeAction::getRuntimeNodeRawPtr, return_internal_reference<>()), DOCSTR_daeAction_RuntimeNode)

        .def("__str__",		&daepython::daeAction__str__)
        .def("__repr__",	&daepython::daeAction__repr__)

        // Virtual function that must be implemented in derived classes in python
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

        .add_property("Name",           &daeOptimizationVariable::GetName,                                                      DOCSTR_daeOptimizationVariable_Name)
        .add_property("Type",           &daeOptimizationVariable::GetType,          &daeOptimizationVariable::SetType,          DOCSTR_daeOptimizationVariable_Type)
        .add_property("Value",          &daeOptimizationVariable::GetValue,         &daeOptimizationVariable::SetValue,         DOCSTR_daeOptimizationVariable_Value)
        .add_property("LowerBound",     &daeOptimizationVariable::GetLB,            &daeOptimizationVariable::SetLB,            DOCSTR_daeOptimizationVariable_LowerBound)
        .add_property("UpperBound",     &daeOptimizationVariable::GetUB,            &daeOptimizationVariable::SetUB,            DOCSTR_daeOptimizationVariable_UpperBound)
        .add_property("StartingPoint",  &daeOptimizationVariable::GetStartingPoint, &daeOptimizationVariable::SetStartingPoint, DOCSTR_daeOptimizationVariable_StartingPoint)
        .add_property("Scaling",        &daeOptimizationVariable::GetScaling,       &daeOptimizationVariable::SetScaling,       DOCSTR_daeOptimizationVariable_Scaling)
        .add_property("Units",          &daeOptimizationVariable::GetUnits,                                                     DOCSTR_daeOptimizationVariable_Units)

        .def("__str__",				&daepython::daeOptimizationVariable__str__)
        .def("__repr__",			&daepython::daeOptimizationVariable__repr__)
    ;

    class_<daeObjectiveFunction, bases<daeObjectiveFunction_t> >("daeObjectiveFunction", DOCSTR_daeObjectiveFunction, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeObjectiveFunction_init))

        .add_property("Name",           &daeObjectiveFunction::GetName, DOCSTR_daeObjectiveFunction_Name)
        .add_property("Residual",       &daeObjectiveFunction::GetResidual, &daeObjectiveFunction::SetResidual, DOCSTR_daeObjectiveFunction_Residual)
        .add_property("Value",          &daeObjectiveFunction::GetValue, DOCSTR_daeObjectiveFunction_Value)
        .add_property("Gradients",      &daepython::GetGradientsObjectiveFunction, DOCSTR_daeObjectiveFunction_Gradients)
        .add_property("AbsTolerance",   &daeObjectiveFunction::GetAbsTolerance, &daeObjectiveFunction::SetAbsTolerance, DOCSTR_daeObjectiveFunction_AbsTolerance)
        .add_property("Scaling",        &daeObjectiveFunction::GetScaling, &daeObjectiveFunction::SetScaling, DOCSTR_daeObjectiveFunction_Scaling)

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
        .add_property("AbsTolerance",   &daeOptimizationConstraint::GetAbsTolerance, &daeOptimizationConstraint::SetAbsTolerance, DOCSTR_daeOptimizationConstraint_AbsTolerance)
        .add_property("Scaling",        &daeOptimizationConstraint::GetScaling, &daeOptimizationConstraint::SetScaling, DOCSTR_daeOptimizationConstraint_Scaling)

        .def("__str__",				&daepython::daeOptimizationConstraint__str__)
        .def("__repr__",			&daepython::daeOptimizationConstraint__repr__)
    ;

    class_<daeMeasuredVariable, bases<daeMeasuredVariable_t> >("daeMeasuredVariable", DOCSTR_daeMeasuredVariable, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeMeasuredVariable_init))

        .add_property("Name",           &daeMeasuredVariable::GetName, DOCSTR_daeMeasuredVariable_Name)
        .add_property("Residual",       &daeMeasuredVariable::GetResidual, &daeMeasuredVariable::SetResidual, DOCSTR_daeMeasuredVariable_Residual)
        .add_property("Value",          &daeMeasuredVariable::GetValue, DOCSTR_daeMeasuredVariable_Value)
        .add_property("Gradients",      &daepython::GetGradientsMeasuredVariable, DOCSTR_daeMeasuredVariable_Gradients)
        .add_property("AbsTolerance",   &daeMeasuredVariable::GetAbsTolerance, &daeMeasuredVariable::SetAbsTolerance, DOCSTR_daeMeasuredVariable_AbsTolerance)
        .add_property("Scaling",        &daeMeasuredVariable::GetScaling,  &daeMeasuredVariable::SetScaling, DOCSTR_daeMeasuredVariable_Scaling)

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


    class_<daeExecutionContext, boost::noncopyable>("daeExecutionContext", DOCSTR_daeExecutionContext, no_init)
    ;

    class_<daepython::daeModelWrapper, bases<daeObject>, boost::noncopyable>("daeModel", DOCSTR_daeModel, no_init)
        .def(init<string, optional<daeModel*, string> >(( arg("self"),
                                                          arg("name"),
                                                          arg("parentModel") = NULL,
                                                          arg("description") = ""
                                                        ), DOCSTR_daeModel_init))

        .add_property("Domains",				&daepython::daeModel_GetDomains,                DOCSTR_daeModel_Domains)
        .add_property("Parameters",				&daepython::daeModel_GetParameters,             DOCSTR_daeModel_Parameters)
        .add_property("Variables",				&daepython::daeModel_GetVariables,              DOCSTR_daeModel_Variables)
        .add_property("Equations",				&daepython::daeModel_GetEquations,              DOCSTR_daeModel_Equations)
        .add_property("Ports",					&daepython::daeModel_GetPorts,                  DOCSTR_daeModel_Ports)
        .add_property("EventPorts",				&daepython::daeModel_GetEventPorts,             DOCSTR_daeModel_EventPorts)
        .add_property("OnEventActions",			&daepython::daeModel_GetOnEventActions,         DOCSTR_daeModel_OnEventActions)
        .add_property("OnConditionActions",		&daepython::daeModel_GetOnConditionActions,     DOCSTR_daeModel_OnConditionActions)
        .add_property("STNs",					&daepython::daeModel_GetSTNs,                   DOCSTR_daeModel_STNs)
        .add_property("Components",				&daepython::daeModel_GetComponents,             DOCSTR_daeModel_Components)
        .add_property("PortArrays",				&daepython::daeModel_GetPortArrays,             DOCSTR_daeModel_PortArrays)
        .add_property("ComponentArrays",		&daepython::daeModel_GetComponentArrays,        DOCSTR_daeModel_ComponentArrays)
        .add_property("PortConnections",		&daepython::daeModel_GetPortConnections,        DOCSTR_daeModel_PortConnections)
        .add_property("EventPortConnections",	&daepython::daeModel_GetEventPortConnections,   DOCSTR_daeModel_EventPortConnections)

        .add_property("dictDomains",				&daepython::daeModel_dictDomains,                DOCSTR_daeModel_Domains)
        .add_property("dictParameters",				&daepython::daeModel_dictParameters,             DOCSTR_daeModel_Parameters)
        .add_property("dictVariables",				&daepython::daeModel_dictVariables,              DOCSTR_daeModel_Variables)
        .add_property("dictEquations",				&daepython::daeModel_dictEquations,              DOCSTR_daeModel_Equations)
        .add_property("dictPorts",					&daepython::daeModel_dictPorts,                  DOCSTR_daeModel_Ports)
        .add_property("dictEventPorts",				&daepython::daeModel_dictEventPorts,             DOCSTR_daeModel_EventPorts)
        .add_property("dictOnEventActions",			&daepython::daeModel_dictOnEventActions,         DOCSTR_daeModel_OnEventActions)
        .add_property("dictOnConditionActions",		&daepython::daeModel_dictOnConditionActions,     DOCSTR_daeModel_OnConditionActions)
        .add_property("dictSTNs",					&daepython::daeModel_dictSTNs,                   DOCSTR_daeModel_STNs)
        .add_property("dictComponents",				&daepython::daeModel_dictComponents,             DOCSTR_daeModel_Components)
        .add_property("dictPortArrays",				&daepython::daeModel_dictPortArrays,             DOCSTR_daeModel_PortArrays)
        .add_property("dictComponentArrays",		&daepython::daeModel_dictComponentArrays,        DOCSTR_daeModel_ComponentArrays)
        .add_property("dictPortConnections",		&daepython::daeModel_dictPortConnections,        DOCSTR_daeModel_PortConnections)
        .add_property("dictEventPortConnections",	&daepython::daeModel_dictEventPortConnections,   DOCSTR_daeModel_EventPortConnections)

        .add_property("IsModelDynamic",			&daeModel::IsModelDynamic,                      DOCSTR_daeModel_IsModelDynamic)
        .add_property("ModelType",	   		    &daeModel::GetModelType,                        DOCSTR_daeModel_ModelType)
        .add_property("InitialConditionMode",	&daeModel::GetInitialConditionMode,
                                                &daeModel::SetInitialConditionMode,             DOCSTR_daeModel_InitialConditionMode)
        .add_property("OverallIndex_BlockIndex_VariableNameMap",
                      &daepython::daeModel_GetOverallIndex_BlockIndex_VariableNameMap,          DOCSTR_daeModel_OverallIndex_BlockIndex_VariableNameMap)

        .def("__str__",           &daepython::daeModel__str__)
        .def("__repr__",          &daepython::daeModel__repr__)

        // Virtual function that must be implemented in derived classes in python
        .def("DeclareEquations", &daepython::daeModelWrapper::DeclareEquations,  &daepython::daeModelWrapper::def_DeclareEquations,
                                 ( arg("self") ), DOCSTR_daeModel_DeclareEquations)

        // Virtual function that can be implemented in derived classes in python if equations need an update (useful for FE models)
        .def("UpdateEquations",  &daepython::daeModelWrapper::UpdateEquations,  &daepython::daeModelWrapper::def_UpdateEquations,
                                 ( arg("self"), arg("executionContext") ), DOCSTR_daeModel_UpdateEquations)

         // Virtual function that can be implemented in derived classes in python to have a generic initialization method (useful for FE models)
        .def("InitializeModel",  &daepython::daeModel_def_InitializeModel,
                                 ( arg("self"), arg("jsonInit") ), DOCSTR_daeModel_InitializeModel)

        .def("CreateEquation",   &daeModel::CreateEquation, return_internal_reference<>(),
                                 ( arg("self"), arg("name"), arg("description") = "", arg("scaling") = 1.0), DOCSTR_daeModel_CreateEquation)

        .def("ConnectPorts",     &daeModel::ConnectPorts,
                                 ( arg("self"), arg("portFrom"), arg("portTo") ), DOCSTR_daeModel_ConnectPorts)
        .def("ConnectEventPorts",&daeModel::ConnectEventPorts,
                                 ( arg("self"), arg("portFrom"), arg("portTo") ), DOCSTR_daeModel_ConnectEventPorts)
        .def("SetReportingOn",	 &daeModel::SetReportingOn,
                                 ( arg("self"), arg("reportingOn") ), DOCSTR_daeModel_SetReportingOn)

        .def("IF",				&daeModel::IF,
                                ( arg("self"), arg("condition"), arg("eventTolerance") = 0.0, arg("ifName") = "", arg("ifDescription") = "",
                                                                                              arg("stateName") = "", arg("stateDescription") = "" ), DOCSTR_daeModel_IF)
        .def("ELSE_IF",			&daeModel::ELSE_IF,
                                ( arg("self"), arg("condition"), arg("eventTolerance") = 0.0, arg("stateName") = "", arg("stateDescription") = "" ), DOCSTR_daeModel_ELSE_IF)
        .def("ELSE",			&daeModel::ELSE,
                                ( arg("self"), arg("stateDescription") = "" ), DOCSTR_daeModel_ELSE)
        .def("END_IF",			&daeModel::END_IF,
                                ( arg("self") ), DOCSTR_daeModel_END_IF)

        .def("STN",				&daeModel::STN, return_internal_reference<>(),
                                ( arg("self"), arg("stnName"), arg("stnDescription") = "" ), DOCSTR_daeModel_STN)
        .def("STATE",			&daeModel::STATE, return_internal_reference<>(),
                                ( arg("self"), arg("stateName"), arg("stateDescription") = "" ), DOCSTR_daeModel_STATE)
        .def("END_STN",			&daeModel::END_STN,
                                ( arg("self") ), DOCSTR_daeModel_END_STN)
        .def("SWITCH_TO",		&daeModel::SWITCH_TO,
                                ( arg("self"),
                                  arg("targetState"),
                                  arg("condition"),
                                  arg("eventTolerance") = 0.0
                                ), DOCSTR_daeModel_SWITCH_TO)
        .def("ON_CONDITION",    &daepython::daeModel_ON_CONDITION,
                                ( arg("self"),
                                  arg("condition"),
                                  arg("switchToStates")      = boost::python::list(),
                                  arg("setVariableValues")   = boost::python::list(),
                                  arg("triggerEvents")       = boost::python::list(),
                                  arg("userDefinedActions")  = boost::python::list(),
                                  arg("eventTolerance")      = 0.0
                                ), DOCSTR_daeModel_ON_CONDITION)
        .def("ON_EVENT",		&daepython::daeModel_ON_EVENT,
                                ( arg("self"),
                                  arg("eventPort"),
                                  arg("switchToStates")      = boost::python::list(),
                                  arg("setVariableValues")   = boost::python::list(),
                                  arg("triggerEvents")       = boost::python::list(),
                                  arg("userDefinedActions")  = boost::python::list()
                                ), DOCSTR_daeModel_ON_EVENT )

        .def("GetCoSimulationInterface", &daepython::daeModel_GetCoSimulationInterface,
                                         ( arg("self") ), DOCSTR_daeModel_GetCoSimulationInterface)

        .def("GetFMIInterface", &daepython::daeModel_GetFMIInterface,
                                ( arg("self") ), DOCSTR_daeModel_GetFMIInterface)

        .def("SaveModelReport",			&daeModel::SaveModelReport,
                                        ( arg("self"), arg("xmlFilename") ), DOCSTR_daeModel_SaveModelReport)
        .def("SaveRuntimeModelReport",	&daeModel::SaveRuntimeModelReport,
                                        ( arg("self"), arg("xmlFilename") ), DOCSTR_daeModel_SaveRuntimeModelReport)

        .def("PropagateDomain",			&daeModel::PropagateDomain,    DOCSTR_daeModel_PropagateDomain)
        .def("PropagateParameter",		&daeModel::PropagateParameter, DOCSTR_daeModel_PropagateParameter)

        /*
        .def("ExportObjects",			&daepython::daeModelWrapper::ExportObjects,
                                        ( arg("self"), arg("objects"), arg("language") ), DOCSTR_daeModel_ExportObjects)
        .def("Export",					&daeModel::Export,
                                        ( arg("self"), arg("content"), arg("language"), arg("modelExportContext") ), DOCSTR_daeModel_Export)
        */
    ;

    class_<daeMatrix<real_t>, boost::noncopyable>("daeMatrix_real", DOCSTR_daeMatrix, no_init)
        .add_property("n",	 &daeMatrix<real_t>::GetNrows)
        .add_property("m",	 &daeMatrix<real_t>::GetNcols)
        .def("__call__",     &daeMatrix<real_t>::GetItem,  ( arg("self"), arg("row"), arg("column") ))
        .def("GetItem",      &daeMatrix<real_t>::GetItem,  ( arg("self"), arg("row"), arg("column") ))
        .def("SetItem",      &daeMatrix<real_t>::SetItem,  ( arg("self"), arg("row"), arg("column"), arg("value") ))
    ;

    class_<daeArray<real_t>, boost::noncopyable>("daeArray_real", DOCSTR_daeArray, no_init)
        .add_property("n",	 &daeArray<real_t>::GetSize)
        .add_property("Values",	 &daepython::daeArray_GetValues)
        .def("__getitem__",  &daeArray<real_t>::GetItem,   ( arg("self"), arg("item") ))
        .def("__setitem__",  &daeArray<real_t>::SetItem,   ( arg("self"), arg("item"), arg("value") ))
        .def("GetItem",      &daeArray<real_t>::GetItem,   ( arg("self"), arg("item") ))
        .def("SetItem",      &daeArray<real_t>::SetItem,   ( arg("self"), arg("item"), arg("value") ))
    ;

    class_<daeMatrix<adouble>, boost::noncopyable>("daeMatrix_adouble", DOCSTR_daeMatrix, no_init)
        .add_property("n",	 &daeMatrix<adouble>::GetNrows)
        .add_property("m",	 &daeMatrix<adouble>::GetNcols)
        .def("__call__",     &daeMatrix<adouble>::GetItem,  ( arg("self"), arg("row"), arg("column") ))
        .def("GetItem",      &daeMatrix<adouble>::GetItem,  ( arg("self"), arg("row"), arg("column") ))
        .def("SetItem",      &daeMatrix<adouble>::SetItem,  ( arg("self"), arg("row"), arg("column"), arg("value") ))
    ;

    class_<daeArray<adouble>, boost::noncopyable>("daeArray_adouble", DOCSTR_daeArray, no_init)
        .add_property("n",	 &daeArray<adouble>::GetSize)
        .def("__call__",     &daeArray<adouble>::GetItem,   ( arg("self"), arg("item") ))
        .def("__getitem__",  &daeArray<adouble>::GetItem,   ( arg("self"), arg("item") ))
        .def("__setitem__",  &daeArray<adouble>::SetItem,   ( arg("self"), arg("item"), arg("value") ))
        .def("GetItem",      &daeArray<adouble>::GetItem,   ( arg("self"), arg("item") ))
        .def("SetItem",      &daeArray<adouble>::SetItem,   ( arg("self"), arg("item"), arg("value") ))
    ;

    class_<daepython::daeSparseMatrixRowIteratorWrapper, boost::noncopyable>("daeSparseMatrixRowIterator", DOCSTR_daeSparseMatrixRowIterator, no_init)
        .def("first",           pure_virtual(&daeSparseMatrixRowIterator::first),       ( arg("self") ))
        .def("next",            pure_virtual(&daeSparseMatrixRowIterator::next),        ( arg("self") ))
        .def("isDone",          pure_virtual(&daeSparseMatrixRowIterator::isDone),      ( arg("self") ))
        .def("currentItem",     pure_virtual(&daeSparseMatrixRowIterator::currentItem), ( arg("self") ))
        .def("__iter__",        &daepython::daeSparseMatrixRowIterator_iter, return_value_policy<manage_new_object>(), ( arg("self") ))
    ;

    class_<daepython::daeSparseMatrixRowIterator__iter__, boost::noncopyable>("daeSparseMatrixRowIterator__iter__", DOCSTR_daeSparseMatrixRowIterator, no_init)
        .def("next",  &daepython::daeSparseMatrixRowIterator__iter__::next, ( arg("self") ))
    ;

    class_<daeFiniteElementModel, bases<daeModel>, boost::noncopyable>("daeFiniteElementModel", DOCSTR_daeFiniteElementModel, no_init)
        .def(init<string, daeModel*, string, daeFiniteElementObject_t* >(( arg("self"),
                                                                           arg("name"),
                                                                           arg("parentModel"),
                                                                           arg("description"),
                                                                           arg("feObject")
                                                                        ), DOCSTR_daeModel_init))

        .def("UpdateEquations",  &daeFiniteElementModel::UpdateEquations, ( arg("self"), arg("executionContext") ), DOCSTR_daeModel_UpdateEquations)
        .def("DeclareEquations", &daeFiniteElementModel::DeclareEquations, ( arg("self") ), DOCSTR_daeModel_DeclareEquations)
    ;

    class_<daeFiniteElementVariableInfo>("daeFiniteElementVariableInfo")
        .def_readonly("VariableName",          &daeFiniteElementVariableInfo::m_strName)
        .def_readonly("VariableDescription",   &daeFiniteElementVariableInfo::m_strDescription)
        .def_readonly("Multiplicity",          &daeFiniteElementVariableInfo::m_nMultiplicity)
        .def_readonly("m_nNumberOfDOFs",       &daeFiniteElementVariableInfo::m_nNumberOfDOFs)
    ;

    //class_< std::vector<daeFiniteElementVariableInfo> >("vector_daeFiniteElementVariableInfo")
    //    .def(vector_indexing_suite< std::vector<daeFiniteElementVariableInfo> >())
    //;

    class_<daeFiniteElementObjectInfo>("daeFiniteElementObjectInfo")
        .def_readonly("NumberOfDOFsPerVariable",    &daeFiniteElementObjectInfo::m_nNumberOfDOFsPerVariable)
        .def_readonly("TotalNumberDOFs",            &daeFiniteElementObjectInfo::m_nTotalNumberDOFs)
        .def_readonly("VariableInfos",              &daeFiniteElementObjectInfo::m_VariableInfos)
    ;

    class_<daepython::daeFiniteElementObjectWrapper, boost::noncopyable>("daeFiniteElementObject_t", DOCSTR_daeFiniteElementObject, no_init)
        .def("AssembleSystem",      pure_virtual(&daeFiniteElementObject_t::AssembleSystem),    ( arg("self") ), DOCSTR_daeFiniteElementObject_AssembleSystem)
        .def("ReAssembleSystem",    pure_virtual(&daeFiniteElementObject_t::ReAssembleSystem),  ( arg("self") ), DOCSTR_daeFiniteElementObject_ReAssembleSystem)
        .def("NeedsReAssembling",   pure_virtual(&daeFiniteElementObject_t::NeedsReAssembling), ( arg("self") ), DOCSTR_daeFiniteElementObject_NeedsReAssembling)

        .def("RowIndices",          pure_virtual(&daeFiniteElementObject_t::RowIndices), return_value_policy<manage_new_object>(),
                                    ( arg("self"), arg("row") ), DOCSTR_daeFiniteElementObject_RowIndices)

        .def("Asystem",             pure_virtual(&daeFiniteElementObject_t::Asystem), return_value_policy<manage_new_object>(),
                                    ( arg("self") ), DOCSTR_daeFiniteElementObject_SystemMatrix)

        .def("Msystem",             pure_virtual(&daeFiniteElementObject_t::Msystem), return_value_policy<manage_new_object>(),
                                    ( arg("self") ), DOCSTR_daeFiniteElementObject_SystemMatrix_dt)

        .def("Fload",               pure_virtual(&daeFiniteElementObject_t::Fload), return_value_policy<manage_new_object>(),
                                    ( arg("self") ), DOCSTR_daeFiniteElementObject_SystemRHS)

        .def("GetObjectInfo",       pure_virtual(&daeFiniteElementObject_t::GetObjectInfo),
                                    ( arg("self") ), DOCSTR_daeFiniteElementObject_GetObjectInfo)
    ;

    class_<daeEquationExecutionInfo, boost::noncopyable>("daeEquationExecutionInfo", DOCSTR_daeEquationExecutionInfo, no_init)
        .add_property("Node",	             make_function(&daepython::daeEquationExecutionInfo_GetNode, return_internal_reference<>()), DOCSTR_daeEquationExecutionInfo_Node)
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

    class_<daeFiniteElementEquation, bases<daeEquation>, boost::noncopyable>("daeFiniteElementEquation", DOCSTR_daeEquation, no_init)
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
        .add_property("ActiveState", &daepython::daeSTN_GetActiveState,
                                     &daepython::daeSTN_SetActiveState,     DOCSTR_daeSTN_ActiveState)
        .add_property("States",      &daepython::daeSTN_States,             DOCSTR_daeSTN_States)
        .add_property("dictStates",  &daepython::daeSTN_dictStates,         DOCSTR_daeSTN_States)
        .add_property("Type",	     &daeSTN::GetType, &daeSTN::SetType,    DOCSTR_daeSTN_Type)

        .def("__str__",				&daepython::daeSTN__str__)
        .def("__repr__",			&daepython::daeSTN__repr__)
    ;

    class_<daeIF, bases<daeSTN>, boost::noncopyable>("daeIF", DOCSTR_daeIF, no_init)
        .def("__str__",				&daepython::daeIF__str__)
        .def("__repr__",			&daepython::daeIF__repr__)
    ;

    class_<dae::tpp::daeThermoPhysicalPropertyPackage_t, boost::noncopyable>("daeThermoPhysicalPropertyPackage_t", no_init)
    ;

    class_<daeThermoPhysicalPropertyPackage, boost::noncopyable>("daeThermoPhysicalPropertyPackage", DOCSTR_daeThermoPhysicalPropertyPackage, no_init)
        .def(init<const string&, daeModel*, const string&>(( arg("self"),
                                                             arg("name"),
                                                             arg("parentModel"),
                                                             arg("descripton") = ""
                                                           )))

        .def("Load_CapeOpen_TPP", &daepython::CapeOpen_LoadPackage, (  arg("self"),
                                                                       arg("packageManager"),
                                                                       arg("packageName"),
                                                                       arg("compoundIDs"),
                                                                       arg("compoundCASNumbers"),
                                                                       arg("availablePhases"),
                                                                       arg("defaultBasis") = dae::tpp::eMole,
                                                                       arg("options") = boost::python::dict()
                                                                     ))

        .def("Load_CoolProp_TPP", &daepython::CoolProp_LoadPackage, (  arg("self"),
                                                                       arg("compoundIDs"),
                                                                       arg("compoundCASNumbers"),
                                                                       arg("availablePhases"),
                                                                       arg("defaultBasis") = dae::tpp::eMole,
                                                                       arg("options") = boost::python::dict()
                                                                     ))

        .add_property("TPPName", &daeThermoPhysicalPropertyPackage::GetTPPName)

        // ICapeThermoCompounds interface
        .def("GetCompoundConstant",	&daepython::GetCompoundConstant, (  arg("self"),
                                                                        arg("property"),
                                                                        arg("compound")
                                                                      ))
        .def("GetTDependentProperty",	&daepython::GetTDependentProperty, ( arg("self"),
                                                                             arg("property"),
                                                                             arg("temperature"),
                                                                             arg("compound")
                                                                           ))
        .def("GetPDependentProperty",	&daepython::GetPDependentProperty, ( arg("self"),
                                                                             arg("property"),
                                                                             arg("pressure"),
                                                                             arg("compound")
                                                                           ))

        // ICapeThermoPropertyRoutine interface
        .def("CalcSinglePhaseScalarProperty",	&daepython::CalcSinglePhaseScalarProperty, ( arg("self"),
                                                                                             arg("property"),
                                                                                             arg("pressure"),
                                                                                             arg("temperature"),
                                                                                             arg("composition"),
                                                                                             arg("phase"),
                                                                                             arg("basis") = dae::tpp::eMole
                                                                                           ))

        .def("CalcSinglePhaseVectorProperty",	&daepython::CalcSinglePhaseVectorProperty, ( arg("self"),
                                                                                             arg("property"),
                                                                                             arg("pressure"),
                                                                                             arg("temperature"),
                                                                                             arg("composition"),
                                                                                             arg("phase"),
                                                                                             arg("basis") = dae::tpp::eMole
                                                                                           ))

        .def("CalcTwoPhaseScalarProperty",	&daepython::CalcTwoPhaseScalarProperty, ( arg("self"),
                                                                                      arg("property"),
                                                                                      arg("pressure1"),
                                                                                      arg("temperature1"),
                                                                                      arg("composition1"),
                                                                                      arg("phase1"),
                                                                                      arg("pressure2"),
                                                                                      arg("temperature2"),
                                                                                      arg("composition2"),
                                                                                      arg("phase2"),
                                                                                      arg("basis") = dae::tpp::eMole
                                                                                    ))

        .def("CalcTwoPhaseVectorProperty",	&daepython::CalcTwoPhaseVectorProperty, ( arg("self"),
                                                                                      arg("property"),
                                                                                      arg("pressure1"),
                                                                                      arg("temperature1"),
                                                                                      arg("composition1"),
                                                                                      arg("phase1"),
                                                                                      arg("pressure2"),
                                                                                      arg("temperature2"),
                                                                                      arg("composition2"),
                                                                                      arg("phase2"),
                                                                                      arg("basis") = dae::tpp::eMole
                                                                                    ))



        .def("_GetCompoundConstant",	&daepython::_GetCompoundConstant, ( arg("self"),
                                                                            arg("property"),
                                                                            arg("compound")
                                                                           ))

        .def("_GetTDependentProperty",	&daepython::_GetTDependentProperty, ( arg("self"),
                                                                              arg("property"),
                                                                              arg("temperature"),
                                                                              arg("compound")
                                                                            ))

        .def("_GetPDependentProperty",	&daepython::_GetPDependentProperty, ( arg("self"),
                                                                              arg("property"),
                                                                              arg("pressure"),
                                                                              arg("compound")
                                                                             ))

        .def("_CalcSinglePhaseScalarProperty",	&daepython::_CalcSinglePhaseScalarProperty, (  arg("self"),
                                                                                               arg("property"),
                                                                                               arg("pressure"),
                                                                                               arg("temperature"),
                                                                                               arg("composition"),
                                                                                               arg("phase"),
                                                                                               arg("basis") = dae::tpp::eMole
                                                                                             ))

        .def("_CalcSinglePhaseVectorProperty",	&daepython::_CalcSinglePhaseVectorProperty, (  arg("self"),
                                                                                               arg("property"),
                                                                                               arg("pressure"),
                                                                                               arg("temperature"),
                                                                                               arg("composition"),
                                                                                               arg("phase"),
                                                                                               arg("basis") = dae::tpp::eMole
                                                                                             ))

        .def("_CalcTwoPhaseScalarProperty",	&daepython::_CalcTwoPhaseScalarProperty, (  arg("self"),
                                                                                        arg("property"),
                                                                                        arg("pressure1"),
                                                                                        arg("temperature1"),
                                                                                        arg("composition1"),
                                                                                        arg("phase1"),
                                                                                        arg("pressure2"),
                                                                                        arg("temperature2"),
                                                                                        arg("composition2"),
                                                                                        arg("phase2"),
                                                                                        arg("basis") = dae::tpp::eMole
                                                                                      ))

        .def("_CalcTwoPhaseVectorProperty",	&daepython::_CalcTwoPhaseVectorProperty, (  arg("self"),
                                                                                        arg("property"),
                                                                                        arg("pressure1"),
                                                                                        arg("temperature1"),
                                                                                        arg("composition1"),
                                                                                        arg("phase1"),
                                                                                        arg("pressure2"),
                                                                                        arg("temperature2"),
                                                                                        arg("composition2"),
                                                                                        arg("phase2"),
                                                                                        arg("basis") = dae::tpp::eMole
                                                                                      ))

        .def("__str__",		&daepython::daeThermoPhysicalPropertyPackage__str__)
        .def("__repr__",	&daepython::daeThermoPhysicalPropertyPackage__repr__)
    ;

    class_<daepython::daeScalarExternalFunctionWrapper, boost::noncopyable>("daeScalarExternalFunction", DOCSTR_daeScalarExternalFunction, no_init)
        .def(init<const string&, daeModel*, const unit&, boost::python::dict>(( arg("self"),
                                                                                arg("name"),
                                                                                arg("parentModel"),
                                                                                arg("units"),
                                                                                arg("arguments")
                                                                              ), DOCSTR_daeScalarExternalFunction_init))
        .add_property("Name",	&daeScalarExternalFunction::GetName, DOCSTR_daeScalarExternalFunction_Name)

        // Virtual function that must be implemented in derived classes in python
        .def("Calculate",	pure_virtual(&daepython::daeScalarExternalFunctionWrapper::Calculate_),
                            ( arg("self"), arg("values") ), DOCSTR_daeScalarExternalFunction_Calculate)

        .def("__call__",	&daeScalarExternalFunction::operator(), ( arg("self") ), DOCSTR_daeScalarExternalFunction_call)

        .def("__str__",		&daepython::daeScalarExternalFunction__str__)
        .def("__repr__",	&daepython::daeScalarExternalFunction__repr__)
    ;

    class_<daepython::daeVectorExternalFunctionWrapper, boost::noncopyable>("daeVectorExternalFunction", DOCSTR_daeVectorExternalFunction, no_init)
        .def(init<const string&, daeModel*, const unit&, size_t, boost::python::dict>(( arg("self"),
                                                                                        arg("name"),
                                                                                        arg("parentModel"),
                                                                                        arg("units"),
                                                                                        arg("numberOfResults"),
                                                                                        arg("arguments")
                                                                                      ), DOCSTR_daeVectorExternalFunction_init))
        .add_property("Name",	         &daeVectorExternalFunction::GetName)
        .add_property("NumberOfResults", &daeVectorExternalFunction::GetNumberOfResults)

        // Virtual function that must be implemented in derived classes in python
        .def("Calculate",	pure_virtual(&daepython::daeVectorExternalFunctionWrapper::Calculate_),
                            ( arg("self"), arg("values") ), DOCSTR_daeVectorExternalFunction_Calculate)

        .def("__call__",	&daeVectorExternalFunction::operator(), ( arg("self") ), DOCSTR_daeVectorExternalFunction_call)
        .def("__str__",		&daepython::daeVectorExternalFunction__str__)
        .def("__repr__",	&daepython::daeVectorExternalFunction__repr__)
    ;


/**************************************************************
    daeLog
***************************************************************/
    class_<daepython::daeLogWrapper, boost::noncopyable>("daeLog_t", DOCSTR_daeLog_t, no_init)
        .add_property("Name",           &daeLog_t::GetName,                                         DOCSTR_daeLog_t_Name)
        .add_property("Enabled",		&daeLog_t::GetEnabled,		 &daeLog_t::SetEnabled,         DOCSTR_daeLog_t_Enabled)
        .add_property("PrintProgress",	&daeLog_t::GetPrintProgress, &daeLog_t::SetPrintProgress,   DOCSTR_daeLog_t_PrintProgress)
        .add_property("Indent",			&daeLog_t::GetIndent,		 &daeLog_t::SetIndent,          DOCSTR_daeLog_t_Indent)
        .add_property("Progress",		&daeLog_t::GetProgress,		 &daeLog_t::SetProgress,        DOCSTR_daeLog_t_Progress)
        .add_property("IndentString",	&daeLog_t::GetIndentString,                                 DOCSTR_daeLog_t_IndentString)
        .add_property("PercentageDone",	&daeLog_t::GetPercentageDone,                               DOCSTR_daeLog_t_PercentageDone)
        .add_property("ETA",			&daeLog_t::GetETA,                                          DOCSTR_daeLog_t_ETA)

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

        .add_property("Filename",           &daeFileLog::GetFilename, DOCSTR_daeFileLog_Filename)
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
