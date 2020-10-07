#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include "docstrings.h"
//#include <noprefix.h>
using namespace boost::python;

// Temporary workaround for Visual Studio 2015 update 3
//  Error   LNK2019 unresolved external symbol "class ClassName const volatile * __cdecl boost::get_pointer<class ClassName const volatile *>(...)
#if _MSC_VER == 1900
#if (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
namespace boost
{
#define POINTER_CONVERSION(CLASS_NAME)   template <> CLASS_NAME const volatile * get_pointer(class CLASS_NAME const volatile *c) {return c;}

POINTER_CONVERSION(csGraphPartitioner_t)
POINTER_CONVERSION(csGraphPartitioner_Metis)
POINTER_CONVERSION(csGraphPartitioner_2D_Npde)
POINTER_CONVERSION(csGraphPartitioner_Simple)
POINTER_CONVERSION(csModel_t)
POINTER_CONVERSION(csNumber_t)
POINTER_CONVERSION(csModelBuilder_t)
}
#endif
#endif

BOOST_PYTHON_MODULE(pyOpenCS)
{
    docstring_options doc_options(true, true, false);

    /*******************************
        csLog_t classes
    ********************************/
    class_<daepython::csLog_Wrapper, boost::noncopyable>("csLog_t", DOCSTR_csLog_t)
        .def("GetName",     pure_virtual(&daepython::csLog_Wrapper::GetName),     ( arg("self") ),                  DOCSTR_csLog_GetName)
        .def("Connect",     pure_virtual(&daepython::csLog_Wrapper::Connect),     ( arg("self"), arg("rank") ),     DOCSTR_csLog_Connect)
        .def("Disconnect",  pure_virtual(&daepython::csLog_Wrapper::Disconnect),  ( arg("self") ),                  DOCSTR_csLog_Disconnect)
        .def("IsConnected", pure_virtual(&daepython::csLog_Wrapper::IsConnected), ( arg("self") ),                  DOCSTR_csLog_IsConnected)
        .def("Message",     pure_virtual(&daepython::csLog_Wrapper::Message),     ( arg("self"), arg("mesage") ),   DOCSTR_csLog_Message)
    ;
    def("createLog_StdOut",   &cs::createLog_StdOut);
    def("createLog_TextFile", &cs::createLog_TextFile, ( arg("fileName") ));

    /*******************************
        csDataReporter_t classes
    ********************************/
    class_<daepython::csDataReporter_Wrapper, boost::noncopyable>("csDataReporter_t", DOCSTR_csDataReporter_t)
        .def("GetName",             pure_virtual(&daepython::csDataReporter_Wrapper::GetName),           ( arg("self") ),                       DOCSTR_csDataReporter_GetName)
        .def("Connect",             pure_virtual(&daepython::csDataReporter_Wrapper::Connect),           ( arg("self"), arg("rank") ),          DOCSTR_csDataReporter_Connect)
        .def("Disconnect",          pure_virtual(&daepython::csDataReporter_Wrapper::Disconnect),        ( arg("self") ),                       DOCSTR_csDataReporter_Disconnect)
        .def("IsConnected",         pure_virtual(&daepython::csDataReporter_Wrapper::IsConnected),       ( arg("self") ),                       DOCSTR_csDataReporter_IsConnected)
        .def("RegisterVariables",   pure_virtual(&daepython::csDataReporter_Wrapper::RegisterVariables), ( arg("self"), arg("variableNames") ), DOCSTR_csDataReporter_RegisterVariables)
        .def("StartNewResultSet",   pure_virtual(&daepython::csDataReporter_Wrapper::StartNewResultSet), ( arg("self"), arg("time") ),          DOCSTR_csDataReporter_StartNewResultSet)
        .def("EndOfData",           pure_virtual(&daepython::csDataReporter_Wrapper::EndOfData),         ( arg("self") ),                       DOCSTR_csDataReporter_EndOfData)
        .def("SendVariables",       pure_virtual(&daepython::csDataReporter_Wrapper::SendVariables),     ( arg("self"), arg("values") ),        DOCSTR_csDataReporter_SendVariables)
        .def("SendDerivatives",     pure_virtual(&daepython::csDataReporter_Wrapper::SendDerivatives),   ( arg("self"), arg("derivatives") ),   DOCSTR_csDataReporter_SendDerivatives)
    ;
    def("createDataReporter_CSV",   &cs::createDataReporter_CSV, ( arg("fileNameValues"),
                                                                   arg("fileNameDerivatives"),
                                                                   arg("delimiter")  = ';',
                                                                   arg("format")     = "fixed",
                                                                   arg("precision")  = 15));
    def("createDataReporter_HDF5",  &cs::createDataReporter_HDF5, ( arg("fileNameValues"), arg("fileNameDerivatives") ));

    /*******************************
        csGraphPartitioner_t classes
    ********************************/
    class_<daepython::csGraphPartitioner_Wrapper, boost::noncopyable>("csGraphPartitioner_t", DOCSTR_csGraphPartitioner_t)
        .def("GetName",     pure_virtual(&daepython::csGraphPartitioner_Wrapper::GetName), ( arg("self") ), DOCSTR_csGraphPartitioner_GetName)
        .def("Partition",   pure_virtual(&daepython::csGraphPartitioner_Wrapper::Partition_), ( arg("self"),
                                                                                                arg("Npe"),
                                                                                                arg("Nvertices"),
                                                                                                arg("Nconstraints"),
                                                                                                arg("rowIndices"),
                                                                                                arg("colIndices"),
                                                                                                arg("vertexWeights")
                                                                                              ), DOCSTR_csGraphPartitioner_Partition)
    ;
    register_ptr_to_python< std::shared_ptr<csGraphPartitioner_t> >();

    def("createGraphPartitioner_Simple",  &cs::createGraphPartitioner_Simple);
    def("createGraphPartitioner_Metis",   &cs::createGraphPartitioner_Metis, ( arg("algorithm") ));
    def("createGraphPartitioner_2D_Npde", &cs::createGraphPartitioner_2D_Npde, ( arg("Nx"),
                                                                                 arg("Ny"),
                                                                                 arg("Npde"),
                                                                                 arg("Npex_Npey_ratio") = 1.0 ));
/*
    class_<csGraphPartitioner_Simple, std::shared_ptr<csGraphPartitioner_t>, bases<csGraphPartitioner_t>, boost::noncopyable>("csGraphPartitioner_Simple", no_init)
        .def(init<>(( arg("self") )))
        .def("GetName", &csGraphPartitioner_Simple::GetName)
    ;

    enum_<cs::MetisRoutine>("MetisRoutine")
        .value("PartGraphKway",        cs::PartGraphKway)
        .value("PartGraphRecursive",   cs::PartGraphRecursive)
        .export_values()
        ;

    class_<csGraphPartitioner_Metis, std::shared_ptr<csGraphPartitioner_t>, bases<csGraphPartitioner_t>, boost::noncopyable>("csGraphPartitioner_Metis", no_init)
        .def("__init__", make_constructor(daepython::csGraphPartitioner_Metis_constructor, ( arg("self"), arg("algorithm") )) )
        .def("GetName", &csGraphPartitioner_Metis::GetName)
    ;

    class_<csGraphPartitioner_2D_Npde, std::shared_ptr<csGraphPartitioner_t>, bases<csGraphPartitioner_t>, boost::noncopyable>("csGraphPartitioner_2D_Npde", no_init)
        .def(init<int, int, int, double>(( arg("self"),
                                           arg("Nx"),
                                           arg("Ny"),
                                           arg("Npde"),
                                           arg("Npex_Npey_ratio") = 1.0
                                        )))
        .def("GetName", &csGraphPartitioner_2D_Npde::GetName)
    ;
*/

    /*******************************
        csNumber_t
    ********************************/
    class_<csNumber_t>("csNumber_t", no_init)
        .def(init<>(( arg("self") )))
        .def(init<real_t>(( arg("self"), arg("value") )))

        .def("__str__",		&daepython::csNumber_str)
        .def("__repr__",	&daepython::csNumber_repr)

        .def(- self) // unary -
        .def(+ self) // unary +

        .def(self + self)
        .def(self - self)
        .def(self * self)
        .def(self / self)
        .def(pow(self, self))

        .def(self + real_t())
        .def(self - real_t())
        .def(self * real_t())
        .def(self / real_t())
        .def(pow(self, real_t()))

        .def(real_t() + self)
        .def(real_t() - self)
        .def(real_t() * self)
        .def(real_t() / self)
        .def(pow(real_t(), self))

        // True division operator (/), mostly used by numpy
        .def("__truediv__",  &daepython::true_divide1)   // csNumber_t / csNumber_t
        .def("__truediv__",  &daepython::true_divide2)   // csNumber_t / real_t
        .def("__truediv__",  &daepython::true_divide3)   // real_t  / csNumber_t

        // Floor division operator (//), mostly used by numpy
        .def("__floordiv__", &daepython::floor_divide1)  // csNumber_t // csNumber_t
        .def("__floordiv__", &daepython::floor_divide2)  // csNumber_t // real_t
        .def("__floordiv__", &daepython::floor_divide3)  // real_t  // csNumber_t

        // Math. functions declared as members to enable numpy support
        // For instance, the following will be possible to write in python:
        //   y = numpy.exp(csNumber_t)
        .def("exp",     &cs::exp)
        .def("log",     &cs::log)
        .def("log10",   &cs::log10)
        .def("sqrt",    &cs::sqrt)
        .def("sin",     &cs::sin)
        .def("cos",     &cs::cos)
        .def("tan",     &cs::tan)
        .def("arcsin",  &cs::asin)
        .def("arccos",  &cs::acos)
        .def("arctan",  &cs::atan)

        .def("sinh",    &cs::sinh)
        .def("cosh",    &cs::cosh)
        .def("tanh",    &cs::tanh)
        .def("arcsinh", &cs::asinh) // arcsinh - not asinh
        .def("arccosh", &cs::acosh) // arccosh - not acosh
        .def("arctanh", &cs::atanh) // arctanh - not atanh
        .def("arctan2", &cs::atan2) // arctan2 - not atan2
        .def("erf",     &cs::erf)

        .def("abs",     &cs::fabs)
        .def("fabs",    &cs::fabs)
        .def("ceil",    &cs::ceil)
        .def("floor",   &cs::floor)
    ;
    def("exp",     &cs::exp);
    def("log",     &cs::log);
    def("log10",   &cs::log10);
    def("sqrt",    &cs::sqrt);
    def("sin",     &cs::sin);
    def("cos",     &cs::cos);
    def("tan",     &cs::tan);
    def("asin",    &cs::asin);
    def("acos",    &cs::acos);
    def("atan",    &cs::atan);
    def("sinh",    &cs::sinh);
    def("cosh",    &cs::cosh);
    def("tanh",    &cs::tanh);
    def("asinh",   &cs::asinh);
    def("acosh",   &cs::acosh);
    def("atanh",   &cs::atanh);
    def("atan2",   &cs::atan2);
    def("erf",     &cs::erf);
    def("abs",     &cs::fabs);
    def("fabs",    &cs::fabs);
    def("ceil",    &cs::ceil);
    def("floor",   &cs::floor);
    def("min",     &cs::min);
    def("max",     &cs::max);

    /*******************************
        csModelBuilder_t
    ********************************/
    class_<csModel_t, std::shared_ptr<csModel_t>, boost::noncopyable>("csModel_t", no_init)
    ;

    class_<csModelBuilder_t, boost::noncopyable>("csModelBuilder_t", no_init)
        .def(init<>(( arg("self") )))

        .add_property("Time",                     make_function(&daepython::GetTime, return_internal_reference<>()) )
        .add_property("DegreesOfFreedom",         &daepython::GetDegreesOfFreedom)
        .add_property("Variables",                &daepython::GetVariables)
        .add_property("TimeDerivatives",          &daepython::GetTimeDerivatives)
        .add_property("SimulationOptions",        &daepython::GetSimulationOptions,       &daepython::SetSimulationOptions)
        .add_property("ModelEquations",           &daepython::GetModelEquations,          &daepython::SetModelEquations)
        .add_property("VariableValues",           &daepython::GetVariableValues,          &daepython::SetVariableValues)
        .add_property("VariableTimeDerivatives",  &daepython::GetVariableTimeDerivatives, &daepython::SetVariableTimeDerivatives)
        .add_property("DegreeOfFreedomValues",    &daepython::GetDegreeOfFreedomValues,   &daepython::SetDegreeOfFreedomValues)
        .add_property("VariableNames",            &daepython::GetVariableNames,           &daepython::SetVariableNames)
        .add_property("VariableTypes",            &daepython::GetVariableTypes,           &daepython::SetVariableTypes)
        .add_property("AbsoluteTolerances",       &daepython::GetAbsoluteTolerances,      &daepython::SetAbsoluteTolerances)

        .def("Initialize_ODE_System",	&csModelBuilder_t::Initialize_ODE_System, ( arg("self"),
                                                                                    arg("noVariables"),
                                                                                    arg("noDofs"),
                                                                                    arg("defaultVariableValue") = 0.0,
                                                                                    arg("defaultAbsoluteTolerance") = 1e-5,
                                                                                    arg("defaultVariableName") = "x",
                                                                                    arg("defaultDOFName") = "y"
                                                                                  ) )
        .def("Initialize_DAE_System",	&csModelBuilder_t::Initialize_DAE_System, ( arg("self"),
                                                                                    arg("noVariables"),
                                                                                    arg("noDofs"),
                                                                                    arg("defaultVariableValue") = 0.0,
                                                                                    arg("defaultVariableTimeDerivative") = 0.0,
                                                                                    arg("defaultAbsoluteTolerance") = 1e-5,
                                                                                    arg("defaultVariableName") = "x",
                                                                                    arg("defaultDOFName") = "y"
                                                                                  ) )

        .def("Initialise_DAETools_DAE_System",  &daepython::Initialise_DAETools_DAE_System, ( arg("self"), arg("data") ))

        .def("PartitionSystem", &daepython::PartitionSystem, ( arg("self"),
                                                               arg("Npe"),
                                                               arg("graphPartitioner"),
                                                               arg("balancingConstraints")  = boost::python::list(),
                                                               arg("logPartitionResults")   = false,
                                                               arg("unaryOperationsFlops")  = boost::python::dict(),
                                                               arg("binaryOperationsFlops") = boost::python::dict()
                                                             ) )
        .def("ExportModels", &daepython::ExportModels, ( arg("models"),
                                                         arg("outputDirectory"),
                                                         arg("simulationOptions")
                                                        ) )
        .staticmethod("ExportModels")

        .def("GetDefaultSimulationOptions_DAE", &daepython::GetDefaultSimulationOptions_DAE)
        .staticmethod("GetDefaultSimulationOptions_DAE")

        .def("GetDefaultSimulationOptions_ODE", &daepython::GetDefaultSimulationOptions_ODE)
        .staticmethod("GetDefaultSimulationOptions_ODE")
    ;

    /*******************************
        csSimulate functions
    ********************************/
    def("csSimulate", &daepython::csSimulate_model, ( arg("model"),
                                                      arg("simulationOptions"),
                                                      arg("simulationDirectory")
                                                    ) );

    def("csSimulate", &daepython::csSimulate_dir, ( arg("inputFilesDirectory") ) );

}
