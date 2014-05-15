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

#ifdef daeSuperLU
using namespace dae::solver::superlu;
BOOST_PYTHON_MODULE(pySuperLU)
#endif

#ifdef daeSuperLU_MT
using namespace dae::solver::superlu_mt;
BOOST_PYTHON_MODULE(pySuperLU_MT)
#endif	

#ifdef daeSuperLU_CUDA
BOOST_PYTHON_MODULE(pySuperLU_CUDA)
#endif
{
    docstring_options doc_options(true, true, false);
    
/**************************************************************
	LA Solver
***************************************************************/
	class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
		.def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
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
	
	class_<daeSuperLUSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeSuperLU_MT_Solver")
        .add_property("Name",			&daeSuperLUSolver::GetName, DOCSTR_daeSuperLUSolver_Name)
        .add_property("Options",		make_function(&daeSuperLUSolver::GetOptions, return_internal_reference<>()), DOCSTR_daeSuperLUSolver_Options)
        .def("SaveAsXPM",				&daeSuperLUSolver::SaveAsXPM,
                                        ( arg("self"), arg("xpmFilename") ), DOCSTR_daeSuperLUSolver_SaveAsXPM)
        .def("SaveAsMatrixMarketFile",	&daeSuperLUSolver::SaveAsMatrixMarketFile,
                                        ( arg("self"), arg("filename"), arg("matrixName"), arg("description") ), DOCSTR_daeSuperLUSolver_SaveAsMatrixMarketFile)
        .def("SetOpenBLASNoThreads",	&daeSuperLUSolver::SetOpenBLASNoThreads,
                                        ( boost::python::arg("self"), boost::python::arg("noThreads") ))
        ;
#endif
	
#ifdef daeSuperLU_CUDA
	class_<daeSuperLUSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeSuperLU_CUDA_Solver")
        .add_property("Name",			&daeSuperLUSolver::GetName, DOCSTR_daeSuperLUSolver_Name)
        .def("SaveAsXPM",				&daeSuperLUSolver::SaveAsXPM,
                                        ( arg("self"), arg("xpmFilename") ), DOCSTR_daeSuperLUSolver_SaveAsXPM)
        .def("SaveAsMatrixMarketFile",	&daeSuperLUSolver::SaveAsMatrixMarketFile,
                                        ( arg("self"), arg("filename"), arg("matrixName"), arg("description") ), DOCSTR_daeSuperLUSolver_SaveAsMatrixMarketFile)
		;
#endif


#ifdef daeSuperLU
    enum_<IterRefine_t>("IterRefine_t")
		.value("NOREFINE",	NOREFINE)
		.value("SINGLE",	SINGLE)
		.value("DOUBLE",	DOUBLE)
		.value("EXTRA",		EXTRA)
		.export_values()
	;
	
    enum_<rowperm_t>("rowperm_t")
		.value("NOROWPERM",	NOROWPERM)
		.value("LargeDiag",	LargeDiag)
		.value("MY_PERMR",	MY_PERMR)
		.export_values()
	;

	class_<superlu_options_t, boost::noncopyable>("superlu_options_t", no_init)
		.def_readwrite("ColPerm",			&superlu_options_t::ColPerm)
		.def_readwrite("DiagPivotThresh",	&superlu_options_t::DiagPivotThresh)
		.def_readwrite("RowPerm",			&superlu_options_t::RowPerm)
		.def_readwrite("PrintStat",			&superlu_options_t::PrintStat)
		//.def_readwrite("IterRefine",		&superlu_options_t::IterRefine)
        //.def_readwrite("Equil",			&superlu_options_t::Equil)
		;

	class_<daeSuperLUSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeSuperLU_Solver")
        .add_property("Name",			&daeSuperLUSolver::GetName, DOCSTR_daeSuperLUSolver_Name)
        .add_property("Options",		make_function(&daeSuperLUSolver::GetOptions, return_internal_reference<>()), DOCSTR_daeSuperLUMTSolver_Options)

        .def("SaveAsXPM",				&daeSuperLUSolver::SaveAsXPM,
                                        ( arg("self"), arg("xpmFilename") ), DOCSTR_daeSuperLUSolver_SaveAsXPM)
        .def("SaveAsMatrixMarketFile",	&daeSuperLUSolver::SaveAsMatrixMarketFile,
                                        ( arg("self"), arg("filename"), arg("matrixName"), arg("description") ), DOCSTR_daeSuperLUSolver_SaveAsMatrixMarketFile)
        .def("SetOpenBLASNoThreads",	&daeSuperLUSolver::SetOpenBLASNoThreads,
                                        ( boost::python::arg("self"), boost::python::arg("noThreads") ))
        ;
	
#endif
	
#ifdef daeSuperLU
    def("daeCreateSuperLUSolver", daeCreateSuperLUSolver, return_value_policy<manage_new_object>(), DOCSTR_daeCreateSuperLUSolver);
#endif

#ifdef daeSuperLU_MT
    def("daeCreateSuperLUSolver", daeCreateSuperLU_MTSolver, return_value_policy<manage_new_object>(), DOCSTR_daeCreateSuperLUSolver);
#endif

}
