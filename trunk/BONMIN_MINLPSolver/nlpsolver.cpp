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
/******************************************************************
	daeNLP
*******************************************************************/
daeMINLP::daeMINLP(daeSimulation_t*   pSimulation, 
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

	size_t i;
	daeOptimizationVariable* pOptVariable;
	
	daeSimulation* pSim = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSim)
		daeDeclareAndThrowException(exInvalidPointer)
		
// Get the objective function, opt. variables and constraints from the simulation 
// In BONMIN get_nlp_info is called twice. Why is that??
	m_ptrarrConstraints.clear();
	m_ptrarrOptVariables.clear();
	
	pSim->GetOptimizationConstraints(m_ptrarrConstraints);
	pSim->GetOptimizationVariables(m_ptrarrOptVariables);
	m_pObjectiveFunction = pSim->GetObjectiveFunction();

	if(!m_pObjectiveFunction || m_ptrarrOptVariables.empty())
		daeDeclareAndThrowException(exInvalidPointer)
				
// Generate the array of opt. variables indexes
//	m_narrOptimizationVariableIndexes.clear();
//	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
//	{
//		pOptVariable = m_ptrarrOptVariables[i];
			
//		m_narrOptimizationVariableIndexes.push_back(pOptVariable->GetOverallIndex());
//	}
	
	daeConfig& cfg = daeConfig::GetConfig();
	m_bPrintInfo = cfg.Get<bool>("daetools.minlpsolver.printInfo", false);
}

daeMINLP::~daeMINLP(void)
{
}

bool daeMINLP::get_variables_types(Index n, 
								   VariableType* var_types)
{
	size_t i;
	daeOptimizationVariable* pOptVariable;
	
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		
		if(pOptVariable->m_eType == eIntegerVariable)
			var_types[i] = INTEGER;
		else if(pOptVariable->m_eType == eBinaryVariable)
			var_types[i] = BINARY;
		else if(pOptVariable->m_eType == eContinuousVariable)
			var_types[i] = CONTINUOUS;
		else
			daeDeclareAndThrowException(exNotImplemented)
	}
			
	if(m_bPrintInfo) 
		PrintVariablesTypes();

	return true;
}

bool daeMINLP::get_variables_linearity(Index n, 
									   Ipopt::TNLP::LinearityType* var_types)
{
	size_t i;
	daeOptimizationVariable* pOptVariable;
	
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		
		if(pOptVariable->m_eType == eIntegerVariable)
			var_types[i] = Ipopt::TNLP::LINEAR;
		else if(pOptVariable->m_eType == eBinaryVariable)
			var_types[i] = Ipopt::TNLP::LINEAR;
		else if(pOptVariable->m_eType == eContinuousVariable)
			var_types[i] = Ipopt::TNLP::NON_LINEAR;
		else
			daeDeclareAndThrowException(exNotImplemented)
	}
	
	if(m_bPrintInfo) 
		PrintVariablesLinearity();
	
	return true;
}


bool daeMINLP::get_constraints_linearity(Index m, 
									     Ipopt::TNLP::LinearityType* const_types)
{
	size_t i;
	daeOptimizationConstraint* pConstraint;
	
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
	
	// Here I access SetupNode!! Is it wise??
		if(pConstraint->m_pConstraintFunction->GetResidual().node->IsLinear())
			const_types[i] = Ipopt::TNLP::LINEAR;
		else
			const_types[i] = Ipopt::TNLP::NON_LINEAR;
	}
	
	if(m_bPrintInfo) 
		PrintConstraintsLinearity();
	
	return true;
}

bool daeMINLP::get_nlp_info(Index& n, 
						    Index& m, 
						    Index& nnz_jac_g,
						    Index& nnz_h_lag, 
						    TNLP::IndexStyleEnum& index_style)
{
	size_t i;
	daeOptimizationConstraint* pConstraint;
	
/*	
	daeOptimizationVariable* pOptVariable;

	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
		
// Get the objective function, opt. variables and constraints from the simulation 
// In BONMIN get_nlp_info is called twice. Why is that??
	m_ptrarrConstraints.clear();
	m_ptrarrOptVariables.clear();
	
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
*/
	
// Set the number of opt. variables and constraints
	n = m_ptrarrOptVariables.size();
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

bool daeMINLP::get_bounds_info(Index n, 
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
	
	if(m_bPrintInfo) 
		PrintBoundsInfo();

	return true;
}

bool daeMINLP::get_starting_point(Index n, 
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
		
		if(m_bPrintInfo) 
			PrintStartingPoint();
		
		return true;
	}
	else
	{
		return false;
	}
}

bool daeMINLP::eval_f(Index n, 
					  const Number* x, 
					  bool new_x, 
					  Number& obj_value)
{
	try
	{
		if(new_x)
		{
			CopyOptimizationVariablesToSimulationAndRun(x);
			m_iRunCounter++;
		}
		
		obj_value = m_pObjectiveFunction->m_pObjectiveVariable->GetValue();
		
		if(m_bPrintInfo) 
		{
			string strMessage = "Fobj: ";
			strMessage += toStringFormatted<real_t>(obj_value, -1, 10, true);
			m_pLog->Message(strMessage, 0);
		}
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeBONMIN (eval_gf: ") + e.what(), 0);
		return false;
	}
	
	return true;
}

bool daeMINLP::eval_grad_f(Index n, 
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
		
		if(m_bPrintInfo) 
		{
			string strMessage;
			m_pLog->Message("Fobj gradient: ", 0);
			for(j = 0; j < n; j++)
				strMessage += toStringFormatted<real_t>(grad_f[j], -1, 10, true) + " ";
			m_pLog->Message(strMessage, 0);
		}
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeBONMIN (eval_grad_f): ") + e.what(), 0);
		return false;
	}

	return true;
}

bool daeMINLP::eval_g(Index n, 
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
		
		if(m_bPrintInfo) 
		{
			string strMessage;
			m_pLog->Message("Constraints values: ", 0);
			for(size_t j = 0; j < m; j++)
				strMessage += toStringFormatted<real_t>(g[j], -1, 10, true) + " ";
			m_pLog->Message(strMessage, 0);
		}
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeBONMIN (eval_g): ") + e.what(), 0);
		return false;
	}

	return true;
}

bool daeMINLP::eval_gi(Index n, 
					   const Number* x, 
					   bool new_x, 
					   Index i, 
					   Number& gi)
{
	daeOptimizationConstraint* pConstraint;

	try
	{
		if(new_x)
		{
			CopyOptimizationVariablesToSimulationAndRun(x);
			m_iRunCounter++;
		}
		if(i >= m_ptrarrConstraints.size())
			daeDeclareAndThrowException(exInvalidCall)

		pConstraint = m_ptrarrConstraints[i];
		gi = pConstraint->m_pConstraintVariable->GetValue();
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeBONMIN (eval_gi): ") + e.what(), 0);
		return false;
	}

	return true;
}

bool daeMINLP::eval_jac_g(Index n, 
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
			
			if(m_bPrintInfo) 
			{
				string strMessage;
				m_pLog->Message("Constraints Jacobian: ", 0);
				for(j = 0; j < nele_jac; j++)
					strMessage += toStringFormatted<real_t>(values[j], -1, 10, true) + " ";
				m_pLog->Message(strMessage, 0);
			}
		}
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeBONMIN (eval_jac_g): ") + e.what(), 0);
		return false;
	}

	return true;
}

bool daeMINLP::eval_grad_gi(Index n, 
						    const Number* x, 
						    bool new_x,
						    Index i, 
						    Index& nele_grad_gi, 
						    Index* jCol,
						    Number* values)
{
	size_t j, counter;
	daeOptimizationConstraint* pConstraint;

	try
	{
		if(i >= m_ptrarrConstraints.size())
			daeDeclareAndThrowException(exInvalidCall)

		if(values == NULL) 
		{
		// Return the structure only
			
			pConstraint = m_ptrarrConstraints[i];
			if(!pConstraint)
				daeDeclareAndThrowException(exInvalidPointer)
					
		// Add indexes for the current constraint
		// Achtung: m_narrOptimizationVariablesIndexes must previously be sorted (in function Initialize)
			counter = 0;
			for(j = 0; j < pConstraint->m_narrOptimizationVariablesIndexes.size(); j++)
			{
				jCol[counter] = pConstraint->m_narrOptimizationVariablesIndexes[j];
				counter++;
			}
			
			if(nele_grad_gi != counter)
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
			::memset(values, 0, nele_grad_gi * sizeof(Number));

			counter = 0;
			pConstraint = m_ptrarrConstraints[i];
			if(!pConstraint)
				daeDeclareAndThrowException(exInvalidPointer)
										
			for(j = 0; j < pConstraint->m_narrOptimizationVariablesIndexes.size(); j++)
			{
				values[counter] = matSens.GetItem(pConstraint->m_narrOptimizationVariablesIndexes[j], // Sensitivity parameter index
												  pConstraint->m_nEquationIndexInBlock );             // Equation index
				counter++;
			}

			if(nele_grad_gi != counter)
				daeDeclareAndThrowException(exInvalidCall)
				
			if(m_bPrintInfo) 
			{
				string strMessage;
				m_pLog->Message("Constraints gradient: ", 0);
				for(j = 0; j < nele_grad_gi; j++)
					strMessage += " " + toStringFormatted<real_t>(values[j], -1, 10, true);
				m_pLog->Message(strMessage , 0);
			}
		}
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeBONMIN (eval_grad_gi): ") + e.what(), 0);
		return false;
	}

	return true;
}

bool daeMINLP::eval_h(Index n, 
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

bool daeMINLP::intermediate_callback(AlgorithmMode mode,
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
	m_pLog->IncreaseIndent(2);
	m_pLog->Message(string("Iteration: ") + toString(iter), 0);
	m_pLog->DecreaseIndent(2);
	
	if(m_bPrintInfo) 
	{
		PrintObjectiveFunction();
		PrintOptimizationVariables();
		PrintConstraints();
	}
	
	return true;
}

void daeMINLP::finalize_solution(TMINLP::SolverReturn status,
								 Index n, 
								 const Number* x, 
								 Number obj_value)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable* pOptVariable;
	daeOptimizationConstraint* pConstraint;
		
	if(status == TMINLP::SUCCESS)
		strMessage = "Optimal Solution Found!";	
	else if(status == TMINLP::INFEASIBLE)
		strMessage = "Optimization failed: Infeasible problem";
	else if(status == TMINLP::CONTINUOUS_UNBOUNDED)
		strMessage = "Optimization failed: CONTINUOUS_UNBOUNDED";
	else if(status == TMINLP::LIMIT_EXCEEDED)
		strMessage = "Optimization failed: Limit exceeded";
	else if(status == TMINLP::MINLP_ERROR)
		strMessage = "Optimization failed: MINLP error";
	
	m_pLog->Message(string("**************************************************************************"), 0);
	m_pLog->Message("                        " + strMessage, 0);
	m_pLog->Message(string("**************************************************************************"), 0);
	m_pLog->Message(string(" "), 0);
	
	m_pLog->Message(string("--------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted("Objective function", 25) +  
				 toStringFormatted("Final value",        16);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string("--------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted(m_pObjectiveFunction->m_pObjectiveVariable->GetName(), 25) +   
				 toStringFormatted(obj_value,                                             16, 6, true);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string(" "), 0);

	m_pLog->Message(string("--------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted("Optimization variable", 25) +  
				 toStringFormatted("Final value",           16) +
				 toStringFormatted("Lower bound",           16) +
				 toStringFormatted("Upper bound",           16);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string("--------------------------------------------------------------------------"), 0);
	for(i = 0; i < n; i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		if(!pOptVariable)
			daeDeclareAndThrowException(exInvalidPointer)
			
		strMessage = toStringFormatted(pOptVariable->GetName(), 25)          +   
					 toStringFormatted(x[i],                    16, 6, true) +
		             toStringFormatted(pOptVariable->m_dLB,     16, 6, true) + 
		             toStringFormatted(pOptVariable->m_dUB,     16, 6, true);
		m_pLog->Message(strMessage, 0);
	}
	m_pLog->Message(string(" "), 0);

	m_pLog->Message(string("--------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted("Constraint",  25) +  
				 toStringFormatted("Final value", 16) +
				 toStringFormatted("Lower bound", 16) +
				 toStringFormatted("Upper bound", 16);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string("--------------------------------------------------------------------------"), 0);
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
		if(!pConstraint)
			daeDeclareAndThrowException(exInvalidPointer)
			
		strMessage = toStringFormatted(pConstraint->m_pConstraintVariable->GetName(),  25)          + 
					 toStringFormatted(pConstraint->m_pConstraintVariable->GetValue(), 16, 6, true) +
					 toStringFormatted(pConstraint->m_dLB,                             16, 6, true) + 
					 toStringFormatted(pConstraint->m_dUB,                             16, 6, true);
		m_pLog->Message(strMessage, 0);
	}	
	m_pLog->Message(string(" "), 0);
}

void daeMINLP::PrintObjectiveFunction(void)
{
	string strMessage;
	real_t obj_value = m_pObjectiveFunction->m_pObjectiveVariable->GetValue();
	strMessage = "Fobj = " + toStringFormatted<real_t>(obj_value, -1, 10, true);
	m_pLog->Message(strMessage, 0);
}

void daeMINLP::PrintOptimizationVariables(void)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable* pOptVariable;

	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		if(!pOptVariable)
			daeDeclareAndThrowException(exInvalidPointer)
			
		strMessage = pOptVariable->GetName() + 
					 " = " + 
					 toStringFormatted<real_t>(pOptVariable->GetValue(), -1, 10, true);
		m_pLog->Message(strMessage, 0);
	}	
}

void daeMINLP::PrintConstraints(void)
{
	size_t i;
	string strMessage;
	daeOptimizationConstraint* pConstraint;

	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
		if(!pConstraint)
			daeDeclareAndThrowException(exInvalidPointer)
			
		strMessage = pConstraint->m_pConstraintVariable->GetName() + " = " + 
					 toStringFormatted<real_t>(pConstraint->m_pConstraintVariable->GetValue(), -1, 10, true);
		m_pLog->Message(strMessage, 0);
	}	
}

void daeMINLP::PrintVariablesTypes(void)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable* pOptVariable;
	
	m_pLog->Message(string("Variable types:"), 0);
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		
		if(pOptVariable->m_eType == eIntegerVariable)
			strMessage = "INTEGER";
		else if(pOptVariable->m_eType == eBinaryVariable)
			strMessage = "BINARY";
		else if(pOptVariable->m_eType == eContinuousVariable)
			strMessage = "CONTINUOUS";
		else
			daeDeclareAndThrowException(exNotImplemented)
					
		m_pLog->Message(pOptVariable->GetName() + " = " + strMessage, 0);
	}	
}

void daeMINLP::PrintVariablesLinearity(void)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable* pOptVariable;
	
	m_pLog->Message(string("Variable linearity:"), 0);
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		
		if(pOptVariable->m_eType == eIntegerVariable)
			strMessage = "LINEAR";
		else if(pOptVariable->m_eType == eBinaryVariable)
			strMessage = "LINEAR";
		else if(pOptVariable->m_eType == eContinuousVariable)
			strMessage = "NON_LINEAR";
		else
			daeDeclareAndThrowException(exNotImplemented)
					
		m_pLog->Message(pOptVariable->GetName() + " = " + strMessage, 0);
	}	
}


void daeMINLP::PrintConstraintsLinearity(void)
{
	size_t i;
	string strMessage;
	daeOptimizationConstraint* pConstraint;
	
	m_pLog->Message(string("Constraints linearity:"), 0);
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
	
	// Here I access SetupNode!! Is it wise??
		if(pConstraint->m_pConstraintFunction->GetResidual().node->IsLinear())
			strMessage = "LINEAR";
		else
			strMessage = "NON_LINEAR";
		
		m_pLog->Message(pConstraint->m_pConstraintFunction->GetName() + " = " + strMessage, 0);
	}
}

void daeMINLP::PrintBoundsInfo(void)
{
	size_t i;
	string strMessage;
	daeOptimizationConstraint* pConstraint;
	daeOptimizationVariable* pOptVariable;

	m_pLog->Message(string("Variables bounds:"), 0);
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
			
		strMessage = pOptVariable->GetName() + " bounds = [" + 
					 toStringFormatted<real_t>(pOptVariable->m_dLB, -1, 10, true) + 
					 ", " + 
					 toStringFormatted<real_t>(pOptVariable->m_dUB, -1, 10, true) + 
					 "]";
		m_pLog->Message(strMessage, 0);
	}	

	m_pLog->Message(string("Constraints bounds:"), 0);
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
	
		strMessage = pConstraint->m_pConstraintFunction->GetName() + " bounds = [" + 
					 toStringFormatted<real_t>(pConstraint->m_dLB, -1, 10, true) + 
					 ", " + 
					 toStringFormatted<real_t>(pConstraint->m_dUB, -1, 10, true) + 
					 "]";
		m_pLog->Message(strMessage, 0);
	}
}

void daeMINLP::PrintStartingPoint(void)
{
	size_t i;
	daeOptimizationVariable* pOptVariable;

	m_pLog->Message(string("Starting point:"), 0);
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
			
		m_pLog->Message(pOptVariable->GetName() + 
						" = " + 
						toStringFormatted<real_t>(pOptVariable->m_dDefaultValue, -1, 10, true), 0);
	}	
}

void daeMINLP::CopyOptimizationVariablesToSimulationAndRun(const Number* x)
{
	size_t i, j;
	daeOptimizationVariable* pOptVariable;
	
	m_pLog->IncreaseIndent(1);
	m_pLog->Message(string("Starting the run No. ") + toString(m_iRunCounter + 1) + string(" ..."), 0);
	
// Print before Run
//	if(m_bPrintInfo) 
//	{
//		m_pLog->Message("Values before Run", 0);
//		PrintObjectiveFunction();
//		PrintOptimizationVariables();
//		PrintConstraints();
//	}
	
	if(m_iRunCounter == 0)
	{
	// 1. Re-assign the optimization variables
		for(i = 0; i < m_ptrarrOptVariables.size(); i++)
		{
			pOptVariable = m_ptrarrOptVariables[i];
			if(!pOptVariable)
				daeDeclareAndThrowException(exInvalidPointer)
				
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
			if(!pOptVariable)
				daeDeclareAndThrowException(exInvalidPointer)
				
			pOptVariable->SetValue(x[i]);
		}
			
	// 3. Reset simulation and DAE solver
		m_pSimulation->Reset();
	
	// 4. Calculate initial conditions
		m_pSimulation->SolveInitial();
	
	// 5. Run the simulation
		m_pSimulation->Run();
	}
	  
// Print After Run
//	if(m_bPrintInfo) 
//	{
//		m_pLog->Message("Values after Run", 0);
//		PrintObjectiveFunction();
//		PrintOptimizationVariables();
//		PrintConstraints();
//	}
	
	m_pLog->Message(string(" "), 0);
	m_pLog->DecreaseIndent(1);
}

const TMINLP::SosInfo* daeMINLP::sosConstraints() const
{
	return NULL;
}

const TMINLP::BranchingInfo* daeMINLP::branchingInfo() const
{
	return NULL;
}

/******************************************************************
	daeBONMINSolver
*******************************************************************/
daeBONMINSolver::daeBONMINSolver(void)
{
	m_pSimulation	     = NULL;
	m_pDAESolver		 = NULL;
	m_pDataReporter		 = NULL;
	m_pLog			     = NULL;
	
	m_Bonmin.initializeOptionsAndJournalist();
	
// IPOPT options	
	string strValue;
	daeConfig& cfg = daeConfig::GetConfig();

	strValue = cfg.Get<string>("daetools.BONMIN.IPOPT.linearSolver", "mumps");
	SetOption("linear_solver", strValue);

	strValue = cfg.Get<string>("daetools.BONMIN.IPOPT.hessianApproximation", "limited-memory");
	SetOption("hessian_approximation", strValue);
}

daeBONMINSolver::~daeBONMINSolver(void)
{
}

void daeBONMINSolver::Initialize(daeSimulation_t* pSimulation, 
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
	
	m_MINLP = new daeMINLP(m_pSimulation, m_pDAESolver, m_pDataReporter, m_pLog);
	m_Bonmin.initialize(GetRawPtr(m_MINLP));	
}

void daeBONMINSolver::Solve(void)
{
	if(IsNull(m_MINLP))
	{
		daeDeclareException(exInvalidCall);
		e << "daeBONMIN object has not been initialized.";
		throw e;
	}
		
	try 
	{
		Bab bb;
		bb(m_Bonmin);  
	}
	catch(TNLPSolver::UnsolvedError* E) 
	{
		daeDeclareException(exMiscellanous);
		e << "Ipopt has failed to solve a problem";
		throw e;
	}
	catch(OsiTMINLPInterface::SimpleError &E) 
	{
		daeDeclareException(exMiscellanous);
		e << "Exception in " << E.className() << "::" << E.methodName() 
		  << ": " << E.message();
		throw e;
	}
	catch(CoinError &E)
	{
		daeDeclareException(exMiscellanous);
		e << "Exception in " << E.className() << "::" << E.methodName() 
		  << ": " << E.message();
		throw e;
	}
}

void daeBONMINSolver::SetOption(const string& strOptionName, const string& strValue)
{
	m_Bonmin.options()->SetStringValue(strOptionName, strValue);
}

void daeBONMINSolver::SetOption(const string& strOptionName, real_t dValue)
{
	m_Bonmin.options()->SetNumericValue(strOptionName, dValue);
}

void daeBONMINSolver::SetOption(const string& strOptionName, int iValue)
{
	m_Bonmin.options()->SetIntegerValue(strOptionName, iValue);
}

void daeBONMINSolver::ClearOptions(void)
{
	m_Bonmin.options()->clear();
}

void daeBONMINSolver::PrintOptions(void)
{	
	string strOptions;
	m_Bonmin.options()->PrintList(strOptions);
	m_pLog->Message(string("BONMIN options:"), 0);
	m_pLog->Message(strOptions, 0);
}

void daeBONMINSolver::PrintUserOptions(void)
{
	string strOptions;
	m_Bonmin.options()->PrintUserOptions(strOptions);
	m_pLog->Message(string("BONMIN options set by user:"), 0);
	m_pLog->Message(strOptions, 0);
}

void daeBONMINSolver::LoadOptionsFile(const string& strOptionsFile)
{	
	if(strOptionsFile.empty())
		m_Bonmin.readOptionsFile(daeConfig::GetBONMINOptionsFile());
	else
		m_Bonmin.readOptionsFile(strOptionsFile);
}





}
}
