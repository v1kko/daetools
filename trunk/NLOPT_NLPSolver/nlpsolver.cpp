#include "stdafx.h"
#include "nlpsolver.h"
#include "../Activity/simulation.h"
#include <stdio.h>
#include <time.h>
using namespace std;

namespace dae
{
namespace nlpsolver
{
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
daeNLOPTSolver::daeNLOPTSolver(void)
{
	m_pSimulation	     = NULL;
	m_pDAESolver		 = NULL;
	m_pDataReporter		 = NULL;
	m_pLog			     = NULL;

	m_nlopt				 = NULL;
	m_nlopt_algorithm    = NLOPT_NUM_ALGORITHMS;
	m_iRunCounter        = 0;
	
	daeConfig& cfg = daeConfig::GetConfig();
	m_bPrintInfo = cfg.Get<bool>("daetools.NLOPT.printInfo", false);
}

daeNLOPTSolver::~daeNLOPTSolver(void)
{
	if(m_nlopt)
		nlopt_destroy(m_nlopt);
}

void daeNLOPTSolver::Initialize(daeSimulation_t* pSimulation, 
								daeDAESolver_t* pDAESolver, 
								daeDataReporter_t* pDataReporter, 
								daeLog_t* pLog)
{
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pLog)
		daeDeclareAndThrowException(exInvalidPointer);

// Check data reporter
	if(!pDataReporter->IsConnected())
	{
		daeDeclareException(exInvalidCall);
		e << "Data Reporter is not connected \n";
		throw e;
	}

	m_pSimulation   = pSimulation;
	m_pDAESolver    = pDAESolver;
	m_pDataReporter	= pDataReporter;
	m_pLog			= pLog;
	
	m_pSimulation->GetOptimizationConstraints(m_ptrarrConstraints);
	m_pSimulation->GetOptimizationVariables(m_ptrarrOptVariables);
	m_pObjectiveFunction = m_pSimulation->GetObjectiveFunction();

	if(!m_pObjectiveFunction || m_ptrarrOptVariables.empty())
		daeDeclareAndThrowException(exInvalidPointer);
		
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
	double xtolerance = cfg.Get<double>("daetools.NLOPT.opt_variables_tolerance", 1E-5);
	double ftol_rel   = cfg.Get<double>("daetools.NLOPT.ftol_rel", 1E-5);
	double ftol_abs   = cfg.Get<double>("daetools.NLOPT.ftol_abs", 1E-5);
	
	nlopt_set_xtol_rel(m_nlopt, xtolerance);	
	nlopt_set_ftol_rel(m_nlopt, ftol_rel);
	nlopt_set_ftol_abs(m_nlopt, ftol_abs);
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

double daeNLOPTSolver::eval_f(unsigned n, const double* x)
{
	double value;

	CopyOptimizationVariablesToSimulationAndRun(x);
	
	value = m_pObjectiveFunction->GetValue();
	
	if(m_bPrintInfo) 
	{
		string strMessage = "Fobj: ";
		strMessage += toStringFormatted<real_t>(value, -1, 10, true);
		m_pLog->Message(strMessage, 0);
	}
	
	return value;
}

void daeNLOPTSolver::eval_grad_f(unsigned n, const double* x, double* grad_f)
{
	CopyOptimizationVariablesToSimulationAndRun(x);

	daeMatrix<real_t>& matSens = m_pDAESolver->GetSensitivities();
	if(n != matSens.GetNrows())
		daeDeclareAndThrowException(exInvalidCall)
	
	// Set all values to 0
	::memset(grad_f, 0, n * sizeof(double));
	
	m_pObjectiveFunction->GetGradients(matSens, grad_f, n);
	
	if(m_bPrintInfo) 
	{
		string strMessage;
		m_pLog->Message("Fobj gradient: ", 0);
		for(size_t j = 0; j < n; j++)
			strMessage += toStringFormatted<real_t>(grad_f[j], -1, 10, true) + " ";
		m_pLog->Message(strMessage, 0);
	}
}

double daeNLOPTSolver::eval_g(daeOptimizationConstraint_t* pConstraint, unsigned n, const double* x)
{
	double value;
	
	CopyOptimizationVariablesToSimulationAndRun(x);

	value = pConstraint->GetValue();
	
	if(m_bPrintInfo) 
	{
		string strMessage;
		m_pLog->Message(pConstraint->GetName() + " value: ", 0);
		strMessage += toStringFormatted<real_t>(value, -1, 10, true) + " ";
		m_pLog->Message(strMessage, 0);
	}

	return value;
}

void daeNLOPTSolver::eval_grad_g(daeOptimizationConstraint_t* pConstraint,  unsigned n, const double* x, double* grad_gh) 
{
	size_t j, paramIndex;
	std::vector<size_t> narrOptimizationVariablesIndexes;

	CopyOptimizationVariablesToSimulationAndRun(x);

	daeMatrix<real_t>& matSens = m_pDAESolver->GetSensitivities();
	if(n != matSens.GetNrows())
		daeDeclareAndThrowException(exInvalidCall)

	size_t Nv = m_ptrarrOptVariables.size();

// Set all values to 0
	::memset(grad_gh, 0, Nv * sizeof(double));
	
// Call GetGradients to fill the array m_pdTempStorage with gradients
// ONLY the values for indexes in the current constraint are set!! The rest is left as it is (zero)
	pConstraint->GetGradients(matSens, grad_gh, Nv);

	if(m_bPrintInfo) 
	{
		string strMessage;
		m_pLog->Message(pConstraint->GetName() + " gradient: ", 0);
		for(j = 0; j < Nv; j++)
			strMessage += toStringFormatted<real_t>(grad_gh[j], -1, 10, true) + " ";
		m_pLog->Message(strMessage, 0);
	}
}

void daeNLOPTSolver::SetConstraints(void)
{
	size_t i, j;
	daeOptimizationConstraint_t* pConstraint;

	if(!m_nlopt)
		daeDeclareAndThrowException(exInvalidPointer);

	daeConfig& cfg = daeConfig::GetConfig();
	double tolerance = cfg.Get<double>("daetools.NLOPT.constraints_tolerance", 1E-5);

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
	m_darrLastX.resize(Nv, 0);

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

void daeNLOPTSolver::CopyOptimizationVariablesToSimulationAndRun(const double* x)
{
	size_t i;
	bool bAlreadyRun;
	daeOptimizationVariable_t* pOptVariable;
	
// If all values are equal then do not run again
	bAlreadyRun = true;
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		if(m_darrLastX[i] != x[i])
		{
			bAlreadyRun = false;
			break;
		}
	}
	
	m_pLog->IncreaseIndent(1);
	m_pLog->Message(string("Starting the run No. ") + toString(m_iRunCounter + 1) + string(" ..."), 0);
	
// Print before Run
	if(m_bPrintInfo) 
	{
		m_pLog->Message("Values before Run", 0);
		PrintObjectiveFunction();
		PrintOptimizationVariables();
		PrintConstraints();
	}
	
	if(m_iRunCounter == 0)
	{
	// 1. Re-assign the optimization variables
		for(i = 0; i < m_ptrarrOptVariables.size(); i++)
		{
			pOptVariable = m_ptrarrOptVariables[i];
			pOptVariable->SetValue(x[i]);
		}
		
	// 2. Calculate initial conditions
		m_pSimulation->SolveInitial();
		
	// 3. Run the simulation
		m_pSimulation->Run();
	}
	else
	{		
	// 1. Set again the initial conditions, values, tolerances, active states etc
		m_pSimulation->SetUpVariables();
		
	// 2. Re-assign the optimization variables
		for(i = 0; i < m_ptrarrOptVariables.size(); i++)
		{
			pOptVariable = m_ptrarrOptVariables[i];
			pOptVariable->SetValue(x[i]);
		}
			
	// 3. Reset simulation and DAE solver
		m_pSimulation->Reset();
	
	// 4. Calculate initial conditions
		m_pSimulation->SolveInitial();
	
	// 5. Run the simulation
		m_pSimulation->Run();
	}
	
	m_iRunCounter++;
	
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
		m_darrLastX[i] = x[i];
	  
// Print After Run
	if(m_bPrintInfo) 
	{
		m_pLog->Message("Values after Run", 0);
		PrintObjectiveFunction();
		PrintOptimizationVariables();
		PrintConstraints();
	}
	
	m_pLog->Message(string(" "), 0);
	m_pLog->DecreaseIndent(1);
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
		strMessage = "Halted because of a forced termination: the user called nlopt_force_stop";	
	else if(status == NLOPT_STOPVAL_REACHED)
		strMessage = "Optimization stopped because stopval was reached";	
	else if(status == NLOPT_FTOL_REACHED)
		strMessage = "Optimization stopped because ftol_rel or ftol_abs was reached";	
	else if(status == NLOPT_XTOL_REACHED)
		strMessage = "Optimization stopped because xtol_rel or xtol_abs was reached";	
	else if(status == NLOPT_MAXEVAL_REACHED)
		strMessage = "Optimization stopped because maxeval was reached";	
	else if(status == NLOPT_MAXTIME_REACHED)
		strMessage = "Optimization stopped because maxtime was reached";	
	
	return strMessage;
}
	
void daeNLOPTSolver::PrintSolution(const double* x, double obj_value, nlopt_result status)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable_t* pOptVariable;
	daeOptimizationConstraint_t* pConstraint;
		
	strMessage  = "Optimal Solution Found! (";	
	strMessage += CreateNLOPTErrorMessage(status);	
	strMessage += ")";	
	
	m_pLog->Message(string(" "), 0);
	m_pLog->Message("    " + strMessage, 0);
	m_pLog->Message(string(" "), 0);
	m_pLog->Message(string(" "), 0);
	
	m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted("Objective function", 25) +  
				 toStringFormatted("Final value",        16) +
				 toStringFormatted("Type",               5);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted(m_pObjectiveFunction->GetName(), 25) +   
				 toStringFormatted(obj_value,                       16, 6, true) +
				 toStringFormatted((m_pObjectiveFunction->IsLinear() ? "L" : "NL"), 5);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string(" "), 0);

	m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted("Optimization variable", 25) +  
				 toStringFormatted("Final value",           16) +
				 toStringFormatted("Lower bound",           16) +
				 toStringFormatted("Upper bound",           16);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
			
		strMessage = toStringFormatted(pOptVariable->GetName(), 25)          +   
					 toStringFormatted(x[i],                    16, 6, true) +
		             toStringFormatted(pOptVariable->GetLB(),   16, 6, true) + 
		             toStringFormatted(pOptVariable->GetUB(),   16, 6, true);
		m_pLog->Message(strMessage, 0);
	}
	m_pLog->Message(string(" "), 0);

	m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted("Constraint",  25) +  
				 toStringFormatted("Final value", 16) +
				 toStringFormatted("Type",         5);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
			
		strMessage = toStringFormatted(pConstraint->GetName(),                 25)          + 
					 toStringFormatted(pConstraint->GetValue(),                16, 6, true) +
					 toStringFormatted((pConstraint->IsLinear() ? "L" : "NL"),  5);
		m_pLog->Message(strMessage, 0);
	}	
	m_pLog->Message(string(" "), 0);
}

void daeNLOPTSolver::PrintObjectiveFunction(void)
{
	string strMessage;
	real_t obj_value = m_pObjectiveFunction->GetValue();
	strMessage = "Fobj = " + toStringFormatted<real_t>(obj_value, -1, 10, true);
	m_pLog->Message(strMessage, 0);
}

void daeNLOPTSolver::PrintOptimizationVariables(void)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable_t* pOptVariable;

	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
			
		strMessage = pOptVariable->GetName() + 
					 " = " + 
					 toStringFormatted<real_t>(pOptVariable->GetValue(), -1, 10, true);
		m_pLog->Message(strMessage, 0);
	}	
}

void daeNLOPTSolver::PrintConstraints(void)
{
	size_t i;
	string strMessage;
	daeOptimizationConstraint_t* pConstraint;

	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
		if(!pConstraint)
			daeDeclareAndThrowException(exInvalidPointer)
			
		strMessage = pConstraint->GetName() + " = " + 
					 toStringFormatted<real_t>(pConstraint->GetValue(), -1, 10, true);
		m_pLog->Message(strMessage, 0);
	}	
}

void daeNLOPTSolver::PrintVariablesTypes(void)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable_t* pOptVariable;
	
	m_pLog->Message(string("Variable types:"), 0);
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		
		if(pOptVariable->GetType() == eIntegerVariable)
			strMessage = "INTEGER";
		else if(pOptVariable->GetType() == eBinaryVariable)
			strMessage = "BINARY";
		else if(pOptVariable->GetType() == eContinuousVariable)
			strMessage = "CONTINUOUS";
		else
			daeDeclareAndThrowException(exNotImplemented)
					
		m_pLog->Message(pOptVariable->GetName() + " = " + strMessage, 0);
	}	
}

void daeNLOPTSolver::PrintVariablesLinearity(void)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable_t* pOptVariable;
	
	m_pLog->Message(string("Variable linearity:"), 0);
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		
		if(pOptVariable->GetType() == eIntegerVariable)
			strMessage = "LINEAR";
		else if(pOptVariable->GetType() == eBinaryVariable)
			strMessage = "LINEAR";
		else if(pOptVariable->GetType() == eContinuousVariable)
			strMessage = "NON_LINEAR";
		else
			daeDeclareAndThrowException(exNotImplemented)
					
		m_pLog->Message(pOptVariable->GetName() + " = " + strMessage, 0);
	}	
}


void daeNLOPTSolver::PrintConstraintsLinearity(void)
{
	size_t i;
	string strMessage;
	daeOptimizationConstraint_t* pConstraint;
	
	m_pLog->Message(string("Constraints linearity:"), 0);
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
	
	// Here I access SetupNode!! Is it wise??
		if(pConstraint->IsLinear())
			strMessage = "LINEAR";
		else
			strMessage = "NON_LINEAR";
		
		m_pLog->Message(pConstraint->GetName() + " = " + strMessage, 0);
	}
}

void daeNLOPTSolver::PrintBoundsInfo(void)
{
	size_t i;
	string strMessage;
	daeOptimizationConstraint_t* pConstraint;
	daeOptimizationVariable_t* pOptVariable;

	m_pLog->Message(string("Variables bounds:"), 0);
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
			
		strMessage = pOptVariable->GetName() + " bounds = [" + 
					 toStringFormatted<real_t>(pOptVariable->GetLB(), -1, 10, true) + 
					 ", " + 
					 toStringFormatted<real_t>(pOptVariable->GetUB(), -1, 10, true) + 
					 "]";
		m_pLog->Message(strMessage, 0);
	}	

//	m_pLog->Message(string("Constraints bounds:"), 0);
//	for(i = 0; i < m_ptrarrConstraints.size(); i++)
//	{
//		pConstraint = m_ptrarrConstraints[i];
//	
//		strMessage = pConstraint->GetName() + " bounds = [" + 
//					 toStringFormatted<real_t>(pConstraint->GetLB(), -1, 10, true) + 
//					 ", " + 
//					 toStringFormatted<real_t>(pConstraint->GetUB(), -1, 10, true) + 
//					 "]";
//		m_pLog->Message(strMessage, 0);
//	}
}

void daeNLOPTSolver::PrintStartingPoint(void)
{
	size_t i;
	daeOptimizationVariable_t* pOptVariable;

	m_pLog->Message(string("Starting point:"), 0);
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
			
		m_pLog->Message(pOptVariable->GetName() + 
						" = " + 
						toStringFormatted<real_t>(pOptVariable->GetStartingPoint(), -1, 10, true), 0);
	}	
}

void daeNLOPTSolver::SetAlgorithm(nlopt_algorithm algorithm)
{
	m_nlopt_algorithm = algorithm;
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

nlopt_algorithm daeNLOPTSolver::GetAlgorithm(void)
{
	return m_nlopt_algorithm;
}


}
}
