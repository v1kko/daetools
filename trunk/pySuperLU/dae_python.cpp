#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include "stdafx.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "docstrings.h"
#include "../LA_SuperLU/superlu_la_solver.h"
using namespace boost::python;
using namespace dae::solver;

// Temporary workaround for Visual Studio 2015 update 3
//  Error   LNK2019 unresolved external symbol "class ClassName const volatile * __cdecl boost::get_pointer<class ClassName const volatile *>(...)
#if _MSC_VER == 1900
#if (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
namespace boost
{
#define POINTER_CONVERSION(CLASS_NAME)   template <> CLASS_NAME const volatile * get_pointer(class CLASS_NAME const volatile *c) {return c;}

POINTER_CONVERSION(daeLASolver_t)
}
#endif
#endif

template<typename KEY, typename VALUE>
boost::python::dict getDictFromMapByValue(std::map<KEY,VALUE>& mapItems)
{
    boost::python::dict res;
    typename std::map<KEY,VALUE>::iterator iter;

    for(iter = mapItems.begin(); iter != mapItems.end(); iter++)
    {
        KEY   key = iter->first;
        VALUE val = iter->second;
        res[key] = val;
    }

    return res;
}

#ifdef daeSuperLU
using namespace dae::solver::superlu;
static boost::python::dict GetCallStats(superlu::daeSuperLUSolver& self)
{
    std::map<std::string, call_stats::TimeAndCount> stats = self.GetCallStats();
    return getDictFromMapByValue(stats);
}

BOOST_PYTHON_MODULE(pySuperLU)
#endif

#ifdef daeSuperLU_MT
using namespace dae::solver::superlu_mt;
static boost::python::dict GetCallStats(superlu_mt::daeSuperLUSolver& self)
{
    std::map<std::string, call_stats::TimeAndCount> stats = self.GetCallStats();
    return getDictFromMapByValue(stats);
}

BOOST_PYTHON_MODULE(pySuperLU_MT)
#endif
{
    docstring_options doc_options(true, true, false);

/**************************************************************
    LA Solver
***************************************************************/
    class_<daeLASolver_t, boost::noncopyable>("daeLASolver_t", no_init)
        .def("SaveAsXPM",	pure_virtual(&daeLASolver_t::SaveAsXPM))

        .def("GetOption_bool",   pure_virtual(&daeLASolver_t::GetOption_bool))
        .def("GetOption_int",    pure_virtual(&daeLASolver_t::GetOption_int))
        .def("GetOption_float",	 pure_virtual(&daeLASolver_t::GetOption_float))
        .def("GetOption_string", pure_virtual(&daeLASolver_t::GetOption_string))

        .def("SetOption_bool",   pure_virtual(&daeLASolver_t::SetOption_bool))
        .def("SetOption_int",    pure_virtual(&daeLASolver_t::SetOption_int))
        .def("SetOption_float",  pure_virtual(&daeLASolver_t::SetOption_float))
        .def("SetOption_string", pure_virtual(&daeLASolver_t::SetOption_string))
    ;

#ifndef daeSuperLU_CUDA
    enum_<yes_no_t>("yes_no_t")
        .value("NO",	NO)
        .value("YES",	YES)
        .export_values()
    ;

    enum_<colperm_t>("colperm_t")
        .value("NATURAL",			NATURAL)
        .value("MMD_ATA",			MMD_ATA)
        .value("MMD_AT_PLUS_A",		MMD_AT_PLUS_A)
        .value("COLAMD",			COLAMD)
        .value("METIS_AT_PLUS_A",	METIS_AT_PLUS_A)
        .export_values()
    ;
#endif

#ifdef daeSuperLU_MT
    class_<superlumt_options_t, boost::noncopyable>("superlumt_options_t", no_init)
        .def_readwrite("nprocs",			&superlumt_options_t::nprocs)
        .def_readwrite("panel_size",		&superlumt_options_t::panel_size)
        .def_readwrite("relax",				&superlumt_options_t::relax)
        .def_readwrite("ColPerm",			&superlumt_options_t::ColPerm)
        .def_readwrite("diag_pivot_thresh",	&superlumt_options_t::diag_pivot_thresh)
        .def_readwrite("drop_tol",			&superlumt_options_t::drop_tol)
        .def_readwrite("PrintStat",			&superlumt_options_t::PrintStat)
    ;

    class_<daeSuperLUSolver, bases<daeLASolver_t>, boost::noncopyable>("daeSuperLU_MT_Solver")
        .add_property("Name",               &daeSuperLUSolver::GetName, DOCSTR_daeSuperLUSolver_Name)
        .add_property("Options",            make_function(&daeSuperLUSolver::GetOptions, return_internal_reference<>()), DOCSTR_daeSuperLUSolver_Options)
        .add_property("CallStats",   &GetCallStats)

        .def("SaveAsXPM",				&daeSuperLUSolver::SaveAsXPM,
                                        ( arg("self"), arg("xpmFilename") ), DOCSTR_daeSuperLUSolver_SaveAsXPM)
        .def("SaveAsMatrixMarketFile",	&daeSuperLUSolver::SaveAsMatrixMarketFile,
                                        ( arg("self"), arg("filename"), arg("matrixName"), arg("description") ), DOCSTR_daeSuperLUSolver_SaveAsMatrixMarketFile)
    ;
#endif


#ifdef daeSuperLU
    enum_<rowperm_t>("rowperm_t")
        .value("NOROWPERM",	NOROWPERM)
        .value("LargeDiag",	LargeDiag)
        .value("MY_PERMR",	MY_PERMR)
        .export_values()
    ;

    class_<superlu_options_t, boost::noncopyable>("superlu_options_t", no_init)
        .def_readwrite("Equil",             &superlu_options_t::Equil)
        .def_readwrite("ColPerm",			&superlu_options_t::ColPerm)
        .def_readwrite("DiagPivotThresh",   &superlu_options_t::DiagPivotThresh)
        .def_readwrite("RowPerm",           &superlu_options_t::RowPerm)
        .def_readwrite("PivotGrowth",	    &superlu_options_t::PivotGrowth)
        .def_readwrite("ConditionNumber",   &superlu_options_t::ConditionNumber)
        .def_readwrite("PrintStat",	        &superlu_options_t::PrintStat)
    ;

    class_<daeSuperLUSolver, bases<daeLASolver_t>, boost::noncopyable>("daeSuperLU_Solver")
        .add_property("Name",             &daeSuperLUSolver::GetName, DOCSTR_daeSuperLUSolver_Name)
        .add_property("Options",          make_function(&daeSuperLUSolver::GetOptions, return_internal_reference<>()), DOCSTR_daeSuperLUMTSolver_Options)
        .add_property("CallStats", &GetCallStats)

        .def("SaveAsXPM",				&daeSuperLUSolver::SaveAsXPM,
                                        ( arg("self"), arg("xpmFilename") ), DOCSTR_daeSuperLUSolver_SaveAsXPM)
        .def("SaveAsMatrixMarketFile",	&daeSuperLUSolver::SaveAsMatrixMarketFile,
                                        ( arg("self"), arg("filename"), arg("matrixName"), arg("description") ), DOCSTR_daeSuperLUSolver_SaveAsMatrixMarketFile)

        .def("GetOption_bool",   &daeSuperLUSolver::GetOption_bool,   ( boost::python::arg("self"), boost::python::arg("name") ))
        .def("GetOption_int",    &daeSuperLUSolver::GetOption_int,    ( boost::python::arg("self"), boost::python::arg("name") ))
        .def("GetOption_float",	 &daeSuperLUSolver::GetOption_float,  ( boost::python::arg("self"), boost::python::arg("name") ))
        .def("GetOption_string", &daeSuperLUSolver::GetOption_string, ( boost::python::arg("self"), boost::python::arg("name") ))

        .def("SetOption_bool",   &daeSuperLUSolver::SetOption_bool,   ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        .def("SetOption_int",    &daeSuperLUSolver::SetOption_int,    ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        .def("SetOption_float",  &daeSuperLUSolver::SetOption_float,  ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        .def("SetOption_string", &daeSuperLUSolver::SetOption_string, ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
    ;

#endif

#ifdef daeSuperLU
    def("daeCreateSuperLUSolver", daeCreateSuperLUSolver, return_value_policy<manage_new_object>(), DOCSTR_daeCreateSuperLUSolver);
#endif

#ifdef daeSuperLU_MT
    def("daeCreateSuperLUSolver", daeCreateSuperLU_MTSolver, return_value_policy<manage_new_object>(), DOCSTR_daeCreateSuperLUSolver);
#endif

}
