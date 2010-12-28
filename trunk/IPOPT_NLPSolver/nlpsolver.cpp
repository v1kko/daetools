#include "stdafx.h"
#include "nlpsolver.h"
#include "../Activity/dyn_simulation.h"
#include <stdio.h>
#include <time.h>
using namespace std;

namespace dae
{
namespace nlpsolver
{
/******************************************************************
	daeNLP
*******************************************************************/
daeNLP::daeNLP(daeSimulation_t*   pSimulation, 
			   daeDAESolver_t*    pDAESolver, 
			   daeDataReporter_t* pDataReporter, 
			   daeLog_t*          pLog)
{
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
	if(!pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer)
	if(!pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer)
	if(!pLog)
		daeDeclareAndThrowException(exInvalidPointer)
	
	m_pSimulation	     = pSimulation;
	m_pDAESolver		 = pDAESolver;
	m_pDataReporter		 = pDataReporter;
	m_pLog			     = pLog;
	
	m_iRunCounter = 0;
}

daeNLP::~daeNLP(void)
{
}

bool daeNLP::get_nlp_info(Index& n, 
						  Index& m, 
						  Index& nnz_jac_g,
						  Index& nnz_h_lag, 
						  IndexStyleEnum& index_style)
{
	size_t i;
	daeOptimizationConstraint* pConstraint;
	daeOptimizationVariable* pOptVariable;
	
	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
		
// Get the objective function, opt. variables and constraints from the simulation 
	pSimulation->GetOptimizationConstraints(m_ptrarrConstraints);
	pSimulation->GetOptimizationVariables(m_ptrarrOptVariables);
	m_pObjectiveFunction = pSimulation->GetObjectiveFunction();

	if(!m_pObjectiveFunction || m_ptrarrOptVariables.empty())
		return false;
				
// Generate the array of opt. variables indexes
	m_narrOptimizationVariableIndexes.clear();
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
			
		m_narrOptimizationVariableIndexes.push_back(pOptVariable->GetIndex());
	}	
			
// Set the number of opt. variables and constraints
	n = m_narrOptimizationVariableIndexes.size();
	m = m_ptrarrConstraints.size();
	index_style = TNLP::C_STYLE;
	
// Set the jacobian number of non-zeroes
	nnz_jac_g = 0;
	nnz_h_lag = 0;
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
	
		nnz_jac_g += pConstraint->m_narrOptimizationVariablesIndexes.size();
	}
	
	return true;
}

bool daeNLP::get_bounds_info(Index n, 
							 Number* x_l, 
							 Number* x_u,
							 Index m, 
							 Number* g_l, 
							 Number* g_u)
{
	size_t i, j;
	daeOptimizationConstraint* pConstraint;
	daeOptimizationVariable* pOptVariable;

	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
			
		x_l[i] = pOptVariable->m_dLB;
		x_u[i] = pOptVariable->m_dUB;
	}	

	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
	
		if(pConstraint->m_eConstraintType == eInequalityConstraint)
		{
			g_l[i] = pConstraint->m_dLB;
			g_u[i] = pConstraint->m_dUB;
		}
		else if(pConstraint->m_eConstraintType == eEqualityConstraint)
		{
			g_l[i] = pConstraint->m_dValue;
			g_u[i] = pConstraint->m_dValue;
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented)
		}
	}

	return true;
}

bool daeNLP::get_starting_point(Index n, 
								bool init_x,
								Number* x,
								bool init_z, 
								Number* z_L, 
								Number* z_U,
								Index m, 
								bool init_lambda,
								Number* lambda)
{
	size_t i;
	daeOptimizationVariable* pOptVariable;

// I am interested ONLY in initial values for opt. variables (x) 
	if(init_x)
	{
		for(i = 0; i < m_ptrarrOptVariables.size(); i++)
		{
			pOptVariable = m_ptrarrOptVariables[i];
				
			x[i] = pOptVariable->m_dDefaultValue;
		}	
		return true;
	}
	else
	{
		return false;
	}
}

bool daeNLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
	try
	{
		if(new_x)
		{
			CopyOptimizationVariablesToSimulationAndRun(x);
			m_iRunCounter++;
		}
		
		obj_value = m_pObjectiveFunction->m_pObjectiveVariable->GetValue();
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeIPOPT: ") + e.what(), 0);
		return false;
	}
	
	return true;
}

bool daeNLP::eval_grad_f(Index n, 
						 const Number* x, 
						 bool new_x, 
						 Number* grad_f)
{
	size_t j;

	try
	{
		if(new_x)
		{
			CopyOptimizationVariablesToSimulationAndRun(x);
			m_iRunCounter++;
		}

		daeMatrix<real_t>& matSens = m_pDAESolver->GetSensitivities();
		if(n != matSens.GetNrows())
			daeDeclareAndThrowException(exInvalidCall)
		
		// Set all values to 0
		::memset(grad_f, 0, n * sizeof(Number));
		
		// Iterate and set only the values for the opt. variable indexes in the objective function
		for(j = 0; j < m_pObjectiveFunction->m_narrOptimizationVariablesIndexes.size(); j++)
		{
			grad_f[j] = matSens.GetItem(m_pObjectiveFunction->m_narrOptimizationVariablesIndexes[j], // Sensitivity parameter index
								        m_pObjectiveFunction->m_nEquationIndexInBlock);              // Equation index
		}
		/*
		cout << endl;
		cout << "Gradient Fobj: ";
		for(j = 0; j < n; j++)
			cout << toStringFormatted<real_t>(grad_f[j], 14, 4, true) << " ";
		cout << endl;
		cout.flush();
		*/
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeIPOPT: ") + e.what(), 0);
		return false;
	}

	return true;
}

bool daeNLP::eval_g(Index n, 
					const Number* x, 
					bool new_x, 
					Index m, 
					Number* g)
{
	size_t i;
	daeOptimizationConstraint* pConstraint;

	if(m != m_ptrarrConstraints.size())
		daeDeclareAndThrowException(exInvalidCall)

	try
	{
		if(new_x)
		{
			CopyOptimizationVariablesToSimulationAndRun(x);
			m_iRunCounter++;
		}

		for(i = 0; i < m_ptrarrConstraints.size(); i++)
		{
			pConstraint = m_ptrarrConstraints[i];
				
			g[i] = pConstraint->m_pConstraintVariable->GetValue();
		}
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeIPOPT: ") + e.what(), 0);
		return false;
	}

	return true;
}

bool daeNLP::eval_jac_g(Index n, 
						const Number* x, 
						bool new_x,
						Index m, 
						Index nele_jac, 
						Index* iRow, 
						Index *jCol,
						Number* values)
{
	size_t i, j, counter;
	daeOptimizationConstraint* pConstraint;

	try
	{
		if(values == NULL) 
		{
		// Return the structure only
			counter = 0;
			for(i = 0; i < m_ptrarrConstraints.size(); i++)
			{
				pConstraint = m_ptrarrConstraints[i];
					
			// Add indexes for the current constraint
			// Achtung: m_narrOptimizationVariablesIndexes must previously be sorted (in function Initialize)
				for(j = 0; j < pConstraint->m_narrOptimizationVariablesIndexes.size(); j++)
				{
					iRow[counter] = i; // The row number is 'i' (the current constraint)
					jCol[counter] = pConstraint->m_narrOptimizationVariablesIndexes[j];
					counter++;
				}
			}
			
			if(nele_jac != counter)
				daeDeclareAndThrowException(exInvalidCall)
		}
		else
		{
		// Return the values
			if(new_x)
			{
				CopyOptimizationVariablesToSimulationAndRun(x);
				m_iRunCounter++;
			}
	
			daeMatrix<real_t>& matSens = m_pDAESolver->GetSensitivities();
			if(n != matSens.GetNrows())
				daeDeclareAndThrowException(exInvalidCall)

			// Set all values to 0
			::memset(values, 0, nele_jac * sizeof(Number));

			counter = 0;
			for(i = 0; i < m_ptrarrConstraints.size(); i++)
			{
				pConstraint = m_ptrarrConstraints[i];
				if(!pConstraint)
					daeDeclareAndThrowException(exInvalidPointer)
											
				for(j = 0; j < pConstraint->m_narrOptimizationVariablesIndexes.size(); j++)
				{
					values[counter] = matSens.GetItem(pConstraint->m_narrOptimizationVariablesIndexes[j], // Sensitivity parameter index
											          pConstraint->m_nEquationIndexInBlock );             // Equation index
					counter++;
				}
			}
			if(nele_jac != counter)
				daeDeclareAndThrowException(exInvalidCall)
			/*	
			cout << endl;
			cout << "Jacobian constraints: ";
			for(j = 0; j < nele_jac; j++)
				cout << toStringFormatted<real_t>(values[j], 14, 4, true) << " ";
			cout << endl;
			cout.flush();
			*/
		}
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeIPOPT: ") + e.what(), 0);
		return false;
	}

	return true;
}

bool daeNLP::eval_h(Index n, 
					const Number* x, 
					bool new_x,
					Number obj_factor, 
					Index m, 
					const Number* lambda,
					bool new_lambda, 
					Index nele_hess, 
					Index* iRow,
					Index* jCol, 
					Number* values)
{
	return false;
}

bool daeNLP::intermediate_callback( AlgorithmMode mode,
									Index iter, 
									Number obj_value,
									Number inf_pr, 
									Number inf_du,
									Number mu, 
									Number d_norm,
									Number regularization_size,
									Number alpha_du, 
									Number alpha_pr,
									Index ls_trials,
									const IpoptData* ip_data,
									IpoptCalculatedQuantities* ip_cq)
{
	string strError;

//	cout << "Iteration: " << iter << endl;
	
//	cout << "Mode: " << iter;
//	if(mode == Solve_Succeeded)
//		strError = "Solve_Succeeded.";	
//	else if(mode == Solved_To_Acceptable_Level)
//		strError = "Solved_To_Acceptable_Level.";	
//	else if(mode == Infeasible_Problem_Detected)
//		strError = "Infeasible_Problem_Detected.";	
//	else if(mode ==  Search_Direction_Becomes_Too_Small)
//		strError = "Search_Direction_Becomes_Too_Small.";	
//	else if(mode ==  Diverging_Iterates)
//		strError = "Diverging_Iterates.";	
//	else if(mode ==  User_Requested_Stop)
//		strError = "User_Requested_Stop.";	
//	else if(mode ==  Feasible_Point_Found)
//		strError = "Feasible_Point_Found.";	
//	else if(mode ==  Maximum_Iterations_Exceeded)
//		strError = "Maximum_Iterations_Exceeded.";	
//	else if(mode ==  Restoration_Failed)
//		strError = "Restoration_Failed.";	
//	else if(mode ==  Error_In_Step_Computation)
//		strError = "Error_In_Step_Computation.";	
//	else if(mode ==  Maximum_CpuTime_Exceeded)
//		strError = "Maximum_CpuTime_Exceeded.";	
//	else if(mode ==  Not_Enough_Degrees_Of_Freedom)
//		strError = "Not_Enough_Degrees_Of_Freedom.";	
//	else if(mode ==  Invalid_Problem_Definition)
//		strError = "Invalid_Problem_Definition.";	
//	else if(mode ==  Invalid_Option)
//		strError = "Invalid_Option.";	
//	else if(mode ==  Invalid_Number_Detected)
//		strError = "Invalid_Number_Detected.";	
//	else if(mode ==  Unrecoverable_Exception)
//		strError = "Unrecoverable_Exception.";	
//	else if(mode ==  NonIpopt_Exception_Thrown)
//		strError = "NonIpopt_Exception_Thrown.";	
//	else if(mode ==  Insufficient_Memory)
//		strError = "Insufficient_Memory.";	
//	else if(mode ==  Internal_Error)
//		strError = "Internal_Error.";	
//	else
//		strError = "Unknown.";	
		
//	cout << endl;
	
//	cout << "obj_value: " << obj_value << endl;
	
	return true;
}

void daeNLP::finalize_solution(SolverReturn status,
								 Index n, 
								 const Number* x, 
								 const Number* z_L, 
								 const Number* z_U,
								 Index m, 
								 const Number* g, 
								 const Number* lambda,
								 Number obj_value,
								 const IpoptData* ip_data,
								 IpoptCalculatedQuantities* ip_cq)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable* pOptVariable;
	daeOptimizationConstraint* pConstraint;
		
	if(status == SUCCESS)
		strMessage = "Optimal Solution Found!";	
	else if(status == MAXITER_EXCEEDED)
		strMessage = "Optimization failed: Maximum number of iterations exceeded.";
	else if(status == STOP_AT_TINY_STEP)
		strMessage = "Optimization failed: Algorithm proceeds with very little progress.";
	else if(status ==  STOP_AT_ACCEPTABLE_POINT)
		strMessage = "Optimization failed: Algorithm stopped at a point that was converged, not to “desired” tolerances, but to “acceptable” tolerances (see the acceptable-... options).";
	else if(status ==  LOCAL_INFEASIBILITY)
		strMessage = "Optimization failed: Algorithm converged to a point of local infeasibility. Problem may be infeasible.";
	else if(status ==  USER_REQUESTED_STOP)
		strMessage = "Optimization failed: The user call-back function intermediate callback (see Section 3.3.4) returned false, i.e., the user code requested a premature termination of the optimization.";
	else if(status ==  DIVERGING_ITERATES)
		strMessage = "Optimization failed: It seems that the iterates diverge.";
	else if(status ==  RESTORATION_FAILURE)
		strMessage = "Optimization failed: Restoration phase failed, algorithm doesn’t know how to proceed.";
	else if(status ==  ERROR_IN_STEP_COMPUTATION)
		strMessage = "Optimization failed: An unrecoverable error occurred while Ipopt tried to compute the search direction.";
	else if(status ==  INVALID_NUMBER_DETECTED)
		strMessage = "Optimization failed: Algorithm received an invalid number (such as NaN or Inf) from the NLP; see also option check derivatives for naninf.";
	else if(status ==  INTERNAL_ERROR)
		strMessage = "Optimization failed: An unknown internal error occurred. Please contact the Ipopt authors through the mailing list.";
	
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string(" "), 0);
	
	strMessage = "Objective function value = " + toStringFormatted<real_t>(obj_value, -1, 10, true);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string(" "), 0);

	m_pLog->Message(string("Optimization variables values:"), 0);
	for(i = 0; i < n; i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		if(!pOptVariable || !pOptVariable->m_pVariable)
			daeDeclareAndThrowException(exInvalidPointer)
			
		strMessage = pOptVariable->m_pVariable->GetName() + " = " + toStringFormatted<real_t>(x[i], -1, 10, true);
		m_pLog->Message(strMessage, 0);
	}	
	m_pLog->Message(string(" "), 0);

	m_pLog->Message(string("Final values of the constraints:"), 0);
	for(i = 0; i < m; i++)
	{
		pConstraint = m_ptrarrConstraints[i];
		if(!pConstraint)
			daeDeclareAndThrowException(exInvalidPointer)
			
		strMessage = pConstraint->m_pConstraintVariable->GetName() + " = " + toStringFormatted<real_t>(g[i], -1, 10, true);
		m_pLog->Message(strMessage, 0);
	}	
	m_pLog->Message(string(" "), 0);
}

void daeNLP::CopyOptimizationVariablesToSimulationAndRun(const Number* x)
{
	size_t i, j;
	daeOptimizationVariable* pOptVariable;

// 1. Set again the initial conditions, values, tolerances, active states etc
	m_pSimulation->SetUpVariables();
	
// 2. Re-assign the optimization variables
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		if(!pOptVariable || !pOptVariable->m_pVariable)
			daeDeclareAndThrowException(exInvalidPointer)
			
		pOptVariable->m_pVariable->ReAssignValue(x[i]);
	}	
	
// 3. Reset simulation and DAE solver
	m_pSimulation->Reset();

// 4. Calculate initial conditions
	m_pSimulation->SolveInitial();

// 5. Run the simulation
	m_pSimulation->Run();
}


/******************************************************************
	daeIPOPTSolver
*******************************************************************/
daeIPOPTSolver::daeIPOPTSolver(void)
{
	m_pSimulation	     = NULL;
	m_pDAESolver		 = NULL;
	m_pDataReporter		 = NULL;
	m_pLog			     = NULL;
	
	m_Application = IpoptApplicationFactory();
	
	daeConfig& cfg = daeConfig::GetConfig();

	real_t tol                   = cfg.Get<real_t>("daetools.activity.ipopt_tol",                   1E-6);
	real_t print_level           = cfg.Get<int>("daetools.activity.ipopt_print_level",              0);
	string linear_solver         = cfg.Get<string>("daetools.activity.ipopt_linear_solver",         "mumps");
	string mu_strategy           = cfg.Get<string>("daetools.activity.ipopt_mu_strategy",           "adaptive");
	string hessian_approximation = cfg.Get<string>("daetools.activity.ipopt_hessian_approximation", "limited-memory");

	SetOption("hessian_approximation", hessian_approximation);
	SetOption("tol",			       tol);
	SetOption("linear_solver",         linear_solver);
	SetOption("mu_strategy",           mu_strategy);
	SetOption("print_level",           print_level);
}

daeIPOPTSolver::~daeIPOPTSolver(void)
{
}

void daeIPOPTSolver::Initialize(daeSimulation_t* pSimulation, 
                                daeDAESolver_t* pDAESolver, 
                                daeDataReporter_t* pDataReporter, 
                                daeLog_t* pLog)
{
	time_t start, end;

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
	
	m_NLP = new daeNLP(m_pSimulation, m_pDAESolver, m_pDataReporter, m_pLog);
}

void daeIPOPTSolver::Solve(void)
{
	if(IsNull(m_Application) || IsNull(m_NLP))
	{
		daeDeclareException(exInvalidCall);
		e << "daeIPOPT object has not been initialized.";
		throw e;
	}
		
// Intialize the IpoptApplication and process the options
	ApplicationReturnStatus status;
	status = m_Application->Initialize();
	if(status != Solve_Succeeded) 
	{
		m_pLog->Message("Error during initialization!", 0);
		daeDeclareAndThrowException(exInvalidCall)
	}
	
// Ask Ipopt to solve the problem
	status = m_Application->OptimizeTNLP(m_NLP);
	if(status != Solve_Succeeded) 
	{
		m_pLog->Message("The problem FAILED!", 0);
	}
}

void daeIPOPTSolver::SetOption(const string& strOptionName, const string& strValue)
{
	if(IsNull(m_Application))
	{
		daeDeclareException(exInvalidCall);
		e << "daeIPOPT object has not been initialized.";
		throw e;
	}
	
	m_Application->Options()->SetStringValue(strOptionName, strValue);
}


void daeIPOPTSolver::SetOption(const string& strOptionName, real_t dValue)
{
	if(IsNull(m_Application))
	{
		daeDeclareException(exInvalidCall);
		e << "daeIPOPT object has not been initialized.";
		throw e;
	}
	
	m_Application->Options()->SetNumericValue(strOptionName, dValue);
}

void daeIPOPTSolver::SetOption(const string& strOptionName, int iValue)
{
	if(IsNull(m_Application))
	{
		daeDeclareException(exInvalidCall);
		e << "daeIPOPT object has not been initialized.";
		throw e;
	}
	
	m_Application->Options()->SetIntegerValue(strOptionName, iValue);
}


}
}
