#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "../LA_SuperLU/superlu_la_solver.h"
using namespace boost::python;
using namespace dae::solver;

#ifdef daeSuperLU
BOOST_PYTHON_MODULE(pySuperLU)
#endif

#ifdef daeSuperLU_MT
BOOST_PYTHON_MODULE(pySuperLU_MT)
#endif	

#ifdef daeSuperLU_CUDA
BOOST_PYTHON_MODULE(pySuperLU_CUDA)
#endif
{
/**************************************************************
	LA Solver
***************************************************************/
	class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
		.def("Create",		pure_virtual(&daeIDALASolver_t::Create))
		.def("Reinitialize",pure_virtual(&daeIDALASolver_t::Reinitialize))
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
	
	class_<daeSuperLUSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeSuperLUSolver")
		.add_property("Name",			&daeSuperLUSolver::GetName)
		.def("Create",					&daeSuperLUSolver::Create)
		.def("Reinitialize",			&daeSuperLUSolver::Reinitialize)
		.def("SaveAsXPM",				&daeSuperLUSolver::SaveAsXPM)
		.def("SaveAsMatrixMarketFile",	&daeSuperLUSolver::SaveAsMatrixMarketFile)
		.def("GetOptions",				&daeSuperLUSolver::GetOptions, return_value_policy<reference_existing_object>())
		;
#endif
	
#ifdef daeSuperLU_CUDA
	class_<daeSuperLUSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeSuperLUSolver")
		.add_property("Name",			&daeSuperLUSolver::GetName)
		.def("Create",					&daeSuperLUSolver::Create)
		.def("Reinitialize",			&daeSuperLUSolver::Reinitialize)
		.def("SaveAsXPM",				&daeSuperLUSolver::SaveAsXPM)
		.def("SaveAsMatrixMarketFile",	&daeSuperLUSolver::SaveAsMatrixMarketFile)
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
		//.def_readwrite("Equil",				&superlu_options_t::Equil)
		;

	class_<daeSuperLUSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeSuperLUSolver")
		.add_property("Name",			&daeSuperLUSolver::GetName)
		.def("Create",					&daeSuperLUSolver::Create)
		.def("Reinitialize",			&daeSuperLUSolver::Reinitialize)
		.def("SaveAsXPM",				&daeSuperLUSolver::SaveAsXPM)
		.def("SaveAsMatrixMarketFile",	&daeSuperLUSolver::SaveAsMatrixMarketFile)
		.def("GetOptions",				&daeSuperLUSolver::GetOptions, return_value_policy<reference_existing_object>())
		;
	
#endif
	
	def("daeCreateSuperLUSolver", daeCreateSuperLUSolver, return_value_policy<reference_existing_object>());

}
