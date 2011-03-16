#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "superlu_la_solver.h"
using namespace boost::python;
using namespace dae::solver;

BOOST_PYTHON_MODULE(pySuperLU)
{
/**************************************************************
	LA Solver
***************************************************************/
//
//	typedef struct {
//		fact_t        Fact;
//		yes_no_t      Equil;
//		colperm_t     ColPerm;
//		trans_t       Trans;
//		IterRefine_t  IterRefine;
//		double        DiagPivotThresh;
//		yes_no_t      SymmetricMode;
//		yes_no_t      PivotGrowth;
//		yes_no_t      ConditionNumber;
//		rowperm_t     RowPerm;
//		int 	  ILU_DropRule;
//		double	  ILU_DropTol;    /* threshold for dropping */
//		double	  ILU_FillFactor; /* gamma in the secondary dropping */
//		norm_t	  ILU_Norm;       /* infinity-norm, 1-norm, or 2-norm */
//		double	  ILU_FillTol;    /* threshold for zero pivot perturbation */
//		milu_t	  ILU_MILU;
//		double	  ILU_MILU_Dim;   /* Dimension of PDE (if available) */
//		yes_no_t      ParSymbFact;
//		yes_no_t      ReplaceTinyPivot; /* used in SuperLU_DIST */
//		yes_no_t      SolveInitialized;
//		yes_no_t      RefineInitialized;
//		yes_no_t      PrintStat;
//	} superlu_options_t;
//
//	typedef enum {NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD,
//			  METIS_AT_PLUS_A, PARMETIS, ZOLTAN, MY_PERMC}      colperm_t;
//	typedef enum {NOTRANS, TRANS, CONJ}                             trans_t;
//	typedef enum {NOEQUIL, ROW, COL, BOTH}                          DiagScale_t;
//	typedef enum {NOREFINE, SINGLE=1, DOUBLE, EXTRA}                IterRefine_t;

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
		.value("PARMETIS",			PARMETIS)
		.value("ZOLTAN",			ZOLTAN)
		.value("MY_PERMC",			MY_PERMC)
		.export_values()
	;
	
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
		.def_readwrite("Equil",				&superlu_options_t::Equil)
		.def_readwrite("ColPerm",			&superlu_options_t::ColPerm)
		.def_readwrite("IterRefine",		&superlu_options_t::IterRefine)
		.def_readwrite("DiagPivotThresh",	&superlu_options_t::DiagPivotThresh)
		.def_readwrite("RowPerm",			&superlu_options_t::RowPerm)
		;

	class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
		.def("Create",		pure_virtual(&daeIDALASolver_t::Create))
		.def("Reinitialize",pure_virtual(&daeIDALASolver_t::Reinitialize))
		.def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
		;

	class_<daeSuperLUSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeSuperLUSolver")
		.def("Create",		&daeSuperLUSolver::Create)
		.def("Reinitialize",&daeSuperLUSolver::Reinitialize)
		.def("SaveAsXPM",	&daeSuperLUSolver::SaveAsXPM)
		.def("GetOptions",	&daeSuperLUSolver::GetOptions, return_value_policy<reference_existing_object>())
		;

	def("daeCreateSuperLUSolver", daeCreateSuperLUSolver, return_value_policy<reference_existing_object>());

}
