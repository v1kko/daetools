#include "stdafx.h"
#include "nlpsolver.h"
#include "../Activity/simulation.h"
#include <stdio.h>
#include <time.h>
using namespace std;

#ifdef DAE_USE_OPEN_BLAS
extern "C" void openblas_set_num_threads(int);
#endif

namespace daetools
{
namespace nlpsolver
{
daeNLPSolver_t* daeCreateNLOPTSolver(const string& algorithm)
{
	return new daeNLOPTSolver(algorithm);
}

double function(unsigned n, const double* x, double* grad, void* data)
{
	double obj_value;
	daeNLOPTSolver* nloptsolver = (daeNLOPTSolver*)data;
	if(!nloptsolver)
		daeDeclareAndThrowException(exInvalidPointer);
	
	obj_value = nloptsolver->eval_f(n, x);
	if(grad)
		nloptsolver->eval_grad_f(n, x, grad);
	
	return obj_value;
}

// Evaluates constraint sent through data* pointer
double constraint(unsigned n, const double *x, double *grad, void *data)
{
	double c_value;
	nloptData* nlpdata = (nloptData*)data;
	if(!nlpdata)
		daeDeclareAndThrowException(exInvalidPointer);
	
	c_value = nlpdata->nloptsolver->eval_g(nlpdata->constraint, n, x);
	if(grad)
		nlpdata->nloptsolver->eval_grad_g(nlpdata->constraint, n, x, grad);
	
	return c_value;
}

/******************************************************************
	daeNLOPTSolver
*******************************************************************/
daeNLOPTSolver::daeNLOPTSolver(nlopt_algorithm algorithm)
{
// Set OpenBLAS to use only one thread (OpenBLAS can't decide based on the matrix size)
// It can be changed later on by the user
    SetOpenBLASNoThreads(1);

    m_nlopt			  = NULL;
	m_nlopt_algorithm = algorithm;
}

daeNLOPTSolver::daeNLOPTSolver(const string& algorithm)
{
// Set OpenBLAS to use only one thread (OpenBLAS can't decide based on the matrix size)
// It can be changed later on by the user
    SetOpenBLASNoThreads(1);

    m_nlopt	= NULL;
	SetAlgorithm(algorithm);
}

daeNLOPTSolver::~daeNLOPTSolver(void)
{
	if(m_nlopt)
		nlopt_destroy(m_nlopt);
}

void daeNLOPTSolver::SetOpenBLASNoThreads(int n)
{
#ifdef DAE_USE_OPEN_BLAS
    openblas_set_num_threads(n);
#endif
}

void daeNLOPTSolver::Initialize(daeOptimization_t* pOptimization,
                                daeSimulation_t* pSimulation, 
								daeDAESolver_t* pDAESolver, 
								daeDataReporter_t* pDataReporter, 
                                daeLog_t* pLog,
                                const std::string& initializationFile)
{
    daeNLPCommon::Init(pOptimization, pSimulation, pDAESolver, pDataReporter, pLog, initializationFile);
	
	daeNLPCommon::CheckProblem(m_ptrarrOptVariables);

	size_t Nv = m_ptrarrOptVariables.size();
	if(Nv == 0)
		daeDeclareAndThrowException(exInvalidCall);
	
	m_darrTempStorage.resize(Nv);
	
	m_nlopt = nlopt_create(m_nlopt_algorithm, Nv);
	if(!m_nlopt)
		daeDeclareAndThrowException(exInvalidPointer);
	
	SetOptimizationVariables();
	SetConstraints();
	
	daeConfig& cfg = daeConfig::GetConfig();
	double xtol_rel   = cfg.GetFloat("daetools.NLOPT.xtol_rel",   1E-6);
	double xtol_abs   = cfg.GetFloat("daetools.NLOPT.xtol_abs",   1E-6);
	double ftol_rel   = cfg.GetFloat("daetools.NLOPT.ftol_rel",   1E-6);
	double ftol_abs   = cfg.GetFloat("daetools.NLOPT.ftol_abs",   1E-6);
	m_bPrintInfo      = cfg.GetBoolean  ("daetools.NLOPT.printInfo",  false);
	int maxeval       = cfg.GetInteger   ("daetools.NLOPT.ftol_abs",  1000);
	double maxtime    = cfg.GetFloat("daetools.NLOPT.maxtime",   0);
	
	nlopt_set_xtol_rel(m_nlopt, xtol_rel);	
	nlopt_set_xtol_abs1(m_nlopt, xtol_abs);	
	nlopt_set_ftol_rel(m_nlopt, ftol_rel);
	nlopt_set_ftol_abs(m_nlopt, ftol_abs);
	nlopt_set_maxeval(m_nlopt, maxeval);
	nlopt_set_maxtime(m_nlopt, maxtime);
	
	nlopt_set_min_objective(m_nlopt, function, this);
	
	if(m_bPrintInfo) 
	{
		PrintVariablesTypes();
		PrintVariablesLinearity();
		PrintBoundsInfo();
		PrintConstraintsLinearity();
		PrintStartingPoint();
		PrintBoundsInfo();
	}
}

void daeNLOPTSolver::PrintOptions(void)
{
	m_pLog->Message(string("NLOPT options:"), 0);
	m_pLog->Message(string("  xtol_rel: ") + toStringFormatted<real_t>(nlopt_get_xtol_rel(m_nlopt), -1, 1, true), 0);
	//m_pLog->Message(string("  xtol_abs: ") + toStringFormatted<real_t>(nlopt_get_xtol_abs(m_nlopt), -1, 1, true), 0);
	m_pLog->Message(string("  ftol_rel: ") + toStringFormatted<real_t>(nlopt_get_ftol_rel(m_nlopt), -1, 1, true), 0);
	m_pLog->Message(string("  ftol_abs: ") + toStringFormatted<real_t>(nlopt_get_ftol_abs(m_nlopt), -1, 1, true), 0);	
	m_pLog->Message(string("  maxtime: ") + toStringFormatted<real_t>(nlopt_get_maxtime(m_nlopt), -1, 1, true), 0);	
	m_pLog->Message(string("  maxeval: ") + toString(nlopt_get_maxeval(m_nlopt)), 0);	
}

void daeNLOPTSolver::Solve(void)
{
	size_t j;
	double optResult;

	if(!m_nlopt)
		daeDeclareAndThrowException(exInvalidPointer);

	nlopt_result status = nlopt_optimize(m_nlopt, &m_darrX[0], &optResult);
	if(status < 0 || status == NLOPT_MAXEVAL_REACHED || status == NLOPT_MAXTIME_REACHED) 
	{
		daeDeclareException(exMiscellanous);
		e << "Optimization faled! (" << CreateNLOPTErrorMessage(status) << ")";
		throw e;
	}

	PrintSolution(&m_darrX[0], optResult, status);
}

std::string daeNLOPTSolver::GetName(void) const
{
	return string("NLOPT MINLP");
}

double daeNLOPTSolver::get_xtol_rel(void) const
{
	if(!m_nlopt)
	{
		daeDeclareException(exInvalidCall);
		e << "NLOPT options cannot be get/set before the optimization is initialized.";
		throw e;
	}
	return nlopt_get_xtol_rel(m_nlopt);	
}

double daeNLOPTSolver::get_xtol_abs(void) const
{
	if(!m_nlopt)
	{
		daeDeclareException(exInvalidCall);
		e << "NLOPT options cannot be get/set before the optimization is initialized.";
		throw e;
	}
//	return nlopt_get_xtol_abs(m_nlopt);	
	return 0;
}

double daeNLOPTSolver::get_ftol_rel(void) const
{
	if(!m_nlopt)
	{
		daeDeclareException(exInvalidCall);
		e << "NLOPT options cannot be get/set before the optimization is initialized.";
		throw e;
	}
	return nlopt_get_ftol_rel(m_nlopt);	
}

double daeNLOPTSolver::get_ftol_abs(void) const
{
	if(!m_nlopt)
	{
		daeDeclareException(exInvalidCall);
		e << "NLOPT options cannot be get/set before the optimization is initialized.";
		throw e;
	}
	return nlopt_get_ftol_abs(m_nlopt);	
}

void daeNLOPTSolver::set_xtol_rel(double tol)
{
	if(!m_nlopt)
	{
		daeDeclareException(exInvalidCall);
		e << "NLOPT options cannot be get/set before the optimization is initialized.";
		throw e;
	}
	nlopt_set_xtol_rel(m_nlopt, tol);	
}

void daeNLOPTSolver::set_xtol_abs(double tol)
{
	if(!m_nlopt)
	{
		daeDeclareException(exInvalidCall);
		e << "NLOPT options cannot be get/set before the optimization is initialized.";
		throw e;
	}
	nlopt_set_xtol_abs1(m_nlopt, tol);	
}

void daeNLOPTSolver::set_ftol_rel(double tol)
{
	if(!m_nlopt)
	{
		daeDeclareException(exInvalidCall);
		e << "NLOPT options cannot be get/set before the optimization is initialized.";
		throw e;
	}
	nlopt_set_ftol_rel(m_nlopt, tol);	
}

void daeNLOPTSolver::set_ftol_abs(double tol)
{
	if(!m_nlopt)
	{
		daeDeclareException(exInvalidCall);
		e << "NLOPT options cannot be get/set before the optimization is initialized.";
		throw e;
	}
	nlopt_set_ftol_abs(m_nlopt, tol);	
}

double daeNLOPTSolver::eval_f(unsigned n, const double* x)
{
	double value;
	try
	{
		CheckAndRun(x);
		daeNLPCommon::Calculate_fobj(value);
	}
	catch(std::exception& e) 	 
	{ 	 
		m_pLog->Message(string("Exception occurred: ") + e.what(), 0); 	 
		return 0;
	}
	
	return value;
}

void daeNLOPTSolver::eval_grad_f(unsigned n, const double* x, double* grad_f)
{
	try
	{
		CheckAndRun(x);
		daeNLPCommon::Calculate_fobj_gradient(grad_f);
	}
	catch(std::exception& e) 	 
	{ 	 
		m_pLog->Message(string("Exception occurred: ") + e.what(), 0); 	 
		for(size_t i = 0; i < n; i++)
			grad_f[i] = 0;
	}
}

double daeNLOPTSolver::eval_g(daeOptimizationConstraint_t* pConstraint, unsigned n, const double* x)
{
	double value;
	
	try
	{
		CheckAndRun(x);
		daeNLPCommon::Calculate_g(pConstraint, value);
	}
	catch(std::exception& e) 	 
	{ 	 
		m_pLog->Message(string("Exception occurred: ") + e.what(), 0); 	 
		return 0;
	}

	return value;
}

void daeNLOPTSolver::eval_grad_g(daeOptimizationConstraint_t* pConstraint,  unsigned n, const double* x, double* grad_g) 
{
	try
	{
		CheckAndRun(x);
		daeNLPCommon::Calculate_g_gradient(pConstraint, grad_g);
	}
	catch(std::exception& e) 	 
	{ 	 
		m_pLog->Message(string("Exception occurred: ") + e.what(), 0); 	 
		for(size_t i = 0; i < n; i++)
			grad_g[i] = 0;
	}
}

void daeNLOPTSolver::SetConstraints(void)
{
	size_t i, j;
	daeOptimizationConstraint_t* pConstraint;

	if(!m_nlopt)
		daeDeclareAndThrowException(exInvalidPointer);

	daeConfig& cfg = daeConfig::GetConfig();
	double tolerance = cfg.GetFloat("daetools.NLOPT.constr_tol", 1E-6);

	size_t Nc = m_ptrarrConstraints.size();
	m_arrConstraintData.resize(Nc);
	
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
		
		m_arrConstraintData[i].nloptsolver = this;
		m_arrConstraintData[i].constraint  = pConstraint;
	
		if(pConstraint->GetType() == eInequalityConstraint)
		{
			nlopt_add_inequality_constraint(m_nlopt, constraint, &m_arrConstraintData[i], tolerance);
		}
		else if(pConstraint->GetType() == eEqualityConstraint)
		{
			nlopt_add_equality_constraint(m_nlopt, constraint, &m_arrConstraintData[i], tolerance);
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented);
		}
	}
}

void daeNLOPTSolver::SetOptimizationVariables(void)
{
	size_t i;
	daeOptimizationVariable_t* pOptVariable;
	
	if(!m_nlopt)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t Nv = m_ptrarrOptVariables.size();

	m_darrLBs.resize(Nv, 0);
	m_darrUBs.resize(Nv, 0);
	m_darrX.resize(Nv, 0);
	m_darrLastX.resize(Nv, -1e30);

	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		
		if(pOptVariable->GetType() == eContinuousVariable)
		{
			m_darrLBs[i] = pOptVariable->GetLB();
			m_darrUBs[i] = pOptVariable->GetUB();
			m_darrX[i]   = pOptVariable->GetStartingPoint();
		}
		else
		{
			daeDeclareException(exNotImplemented);
			e << "NLOPT cannot handle integer variables";
			throw e;
		}
	}
	nlopt_set_lower_bounds(m_nlopt, &m_darrLBs[0]);
	nlopt_set_upper_bounds(m_nlopt, &m_darrUBs[0]);
}

void daeNLOPTSolver::CheckAndRun(const double* x)
{
	size_t i;
	bool bPreviousRun;
	daeOptimizationVariable_t* pOptVariable;
    static real_t tolerance = std::numeric_limits<real_t>::denorm_min();
	
// If all values are equal then do not run again
	bPreviousRun = true;
	for( i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
        // If any opt. variable is different from the previous run
        // execute the simulation. Otherwise skip running it.
        if(::fabs(m_darrLastX[i] - x[i]) > tolerance)
		{
			bPreviousRun = false;
			break;
		}
	}

    try
    {
        if(!bPreviousRun)
        {
            daeNLPCommon::CopyOptimizationVariablesToSimulationAndRun(x);

            for(size_t i = 0; i < m_ptrarrOptVariables.size(); i++)
            {
                pOptVariable = m_ptrarrOptVariables[i];
                pOptVariable->SetValue(x[i]);
            }
        }
    }
    catch(std::exception& e)
    {
        m_pLog->Message(string("Exception occurred: ") + e.what(), 0);
    }
}

string daeNLOPTSolver::CreateNLOPTErrorMessage(nlopt_result status)
{
	string strMessage;

	if(status == NLOPT_FAILURE)
		strMessage = "Generic failure code";	
	else if(status == NLOPT_INVALID_ARGS)
		strMessage = "Invalid arguments";	
	else if(status == NLOPT_OUT_OF_MEMORY)
		strMessage = "Ran out of memory";	
	else if(status == NLOPT_ROUNDOFF_LIMITED)
		strMessage = "Halted because roundoff errors limited progress";	
	else if(status == NLOPT_FORCED_STOP)
		strMessage = "The user requested termination";	
	else if(status == NLOPT_STOPVAL_REACHED)
		strMessage = "stopval was reached";	
	else if(status == NLOPT_FTOL_REACHED)
		strMessage = "ftol_rel or ftol_abs was reached";	
	else if(status == NLOPT_XTOL_REACHED)
		strMessage = "xtol_rel or xtol_abs was reached";	
	else if(status == NLOPT_MAXEVAL_REACHED)
		strMessage = "maxeval was reached";	
	else if(status == NLOPT_MAXTIME_REACHED)
		strMessage = "maxtime was reached";	
	
	return strMessage;
}
	
void daeNLOPTSolver::PrintSolution(const double* x, double obj_value, nlopt_result status)
{
	m_pLog->Message(" ", 0);
	m_pLog->Message("    Optimal Solution Found!", 0);
	m_pLog->Message("    " + CreateNLOPTErrorMessage(status), 0);
	m_pLog->Message(" ", 0);
	
	daeNLPCommon::PrintSolution(obj_value, x, NULL);
}

void daeNLOPTSolver::SetAlgorithm(string algorithm)
{
	if(algorithm == "NLOPT_GN_DIRECT")
		m_nlopt_algorithm = NLOPT_GN_DIRECT;
	else if(algorithm == "NLOPT_GN_DIRECT_L")
		m_nlopt_algorithm = NLOPT_GN_DIRECT_L;
	else if(algorithm == "NLOPT_GN_DIRECT_L_RAND")
		m_nlopt_algorithm = NLOPT_GN_DIRECT_L_RAND;
	else if(algorithm == "NLOPT_GN_DIRECT_NOSCAL")
		m_nlopt_algorithm = NLOPT_GN_DIRECT_NOSCAL;
	else if(algorithm == "NLOPT_GN_DIRECT_L_NOSCAL")
		m_nlopt_algorithm = NLOPT_GN_DIRECT_L_NOSCAL;
	else if(algorithm == "NLOPT_GN_DIRECT_L_RAND_NOSCAL")
		m_nlopt_algorithm = NLOPT_GN_DIRECT_L_RAND_NOSCAL;

	else if(algorithm == "NLOPT_GN_ORIG_DIRECT")
		m_nlopt_algorithm = NLOPT_GN_ORIG_DIRECT;
	else if(algorithm == "NLOPT_GN_ORIG_DIRECT_L")
		m_nlopt_algorithm = NLOPT_GN_ORIG_DIRECT_L;
	
	else if(algorithm == "NLOPT_GD_STOGO")
		m_nlopt_algorithm = NLOPT_GD_STOGO;
	else if(algorithm == "NLOPT_GD_STOGO_RAND")
		m_nlopt_algorithm = NLOPT_GD_STOGO_RAND;
	
	else if(algorithm == "NLOPT_LD_LBFGS_NOCEDAL")
		m_nlopt_algorithm = NLOPT_LD_LBFGS_NOCEDAL;
	
	else if(algorithm == "NLOPT_LD_LBFGS")
		m_nlopt_algorithm = NLOPT_LD_LBFGS;
	
	else if(algorithm == "NLOPT_LN_PRAXIS")
		m_nlopt_algorithm = NLOPT_LN_PRAXIS;
	
	else if(algorithm == "NLOPT_LD_VAR1")
		m_nlopt_algorithm = NLOPT_LD_VAR1;
	else if(algorithm == "NLOPT_LD_VAR2")
		m_nlopt_algorithm = NLOPT_LD_VAR2;
	
	else if(algorithm == "NLOPT_LD_TNEWTON")
		m_nlopt_algorithm = NLOPT_LD_TNEWTON;
	else if(algorithm == "NLOPT_LD_TNEWTON_RESTART")
		m_nlopt_algorithm = NLOPT_LD_TNEWTON_RESTART;
	else if(algorithm == "NLOPT_LD_TNEWTON_PRECOND")
		m_nlopt_algorithm = NLOPT_LD_TNEWTON_PRECOND;
	else if(algorithm == "NLOPT_LD_TNEWTON_PRECOND_RESTART")
		m_nlopt_algorithm = NLOPT_LD_TNEWTON_PRECOND_RESTART;
	
	else if(algorithm == "NLOPT_GN_CRS2_LM")
		m_nlopt_algorithm = NLOPT_GN_CRS2_LM;
	
	else if(algorithm == "NLOPT_GN_MLSL")
		m_nlopt_algorithm = NLOPT_GN_MLSL;
	else if(algorithm == "NLOPT_GD_MLSL")
		m_nlopt_algorithm = NLOPT_GD_MLSL;
	else if(algorithm == "NLOPT_GN_MLSL_LDS")
		m_nlopt_algorithm = NLOPT_GN_MLSL_LDS;
	else if(algorithm == "NLOPT_GD_MLSL_LDS")
		m_nlopt_algorithm = NLOPT_GD_MLSL_LDS;
	
	else if(algorithm == "NLOPT_LD_MMA")
		m_nlopt_algorithm = NLOPT_LD_MMA;
	
	else if(algorithm == "NLOPT_LN_COBYLA")
		m_nlopt_algorithm = NLOPT_LN_COBYLA;
	
	else if(algorithm == "NLOPT_LN_NEWUOA")
		m_nlopt_algorithm = NLOPT_LN_NEWUOA;
	else if(algorithm == "NLOPT_LN_NEWUOA_BOUND")
		m_nlopt_algorithm = NLOPT_LN_NEWUOA_BOUND;
	
	else if(algorithm == "NLOPT_LN_NELDERMEAD")
		m_nlopt_algorithm = NLOPT_LN_NELDERMEAD;
	else if(algorithm == "NLOPT_LN_SBPLX")
		m_nlopt_algorithm = NLOPT_LN_SBPLX;
	
	else if(algorithm == "NLOPT_LN_AUGLAG")
		m_nlopt_algorithm = NLOPT_LN_AUGLAG;
	else if(algorithm == "NLOPT_LD_AUGLAG")
		m_nlopt_algorithm = NLOPT_LD_AUGLAG;
	else if(algorithm == "NLOPT_LN_AUGLAG_EQ")
		m_nlopt_algorithm = NLOPT_LN_AUGLAG_EQ;
	else if(algorithm == "NLOPT_LD_AUGLAG_EQ")
		m_nlopt_algorithm = NLOPT_LD_AUGLAG_EQ;
	
	else if(algorithm == "NLOPT_LN_BOBYQA")
		m_nlopt_algorithm = NLOPT_LN_BOBYQA;
	
	else if(algorithm == "NLOPT_GN_ISRES")
		m_nlopt_algorithm = NLOPT_GN_ISRES;
	
	/* new variants that require local_optimizer to be set;
	   not with older constants for backwards compatibility */
	else if(algorithm == "NLOPT_AUGLAG")
		m_nlopt_algorithm = NLOPT_AUGLAG;
	else if(algorithm == "NLOPT_AUGLAG_EQ")
		m_nlopt_algorithm = NLOPT_AUGLAG_EQ;
	else if(algorithm == "NLOPT_G_MLSL")
		m_nlopt_algorithm = NLOPT_G_MLSL;
	else if(algorithm == "NLOPT_G_MLSL_LDS")
		m_nlopt_algorithm = NLOPT_G_MLSL_LDS;
	
	else if(algorithm == "NLOPT_LD_SLSQP")
		m_nlopt_algorithm = NLOPT_LD_SLSQP;
	else
		daeDeclareAndThrowException(exNotImplemented);
}


}
}
