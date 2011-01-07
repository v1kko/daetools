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
	
	m_pdTempStorage = NULL;
	m_iRunCounter   = 0;

//	size_t i;
//	daeOptimizationVariable* pOptVariable;
	
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
		
	size_t n = m_ptrarrOptVariables.size();
	if(n == 0)
		daeDeclareAndThrowException(exInvalidCall)

	m_pdTempStorage = new real_t[n];
				
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
	if(m_pdTempStorage)
	{
		delete[] m_pdTempStorage;
		m_pdTempStorage = NULL;
	}
}

bool daeMINLP::get_variables_types(Index n, 
								   VariableType* var_types)
{
	size_t i;
	daeOptimizationVariable* pOptVariable;
	
	for(i = 0; i < m_ptrarrOptVariables.size(); i++)
	{
		pOptVariable = m_ptrarrOptVariables[i];
		
		if(pOptVariable->GetType() == eIntegerVariable)
			var_types[i] = INTEGER;
		else if(pOptVariable->GetType() == eBinaryVariable)
			var_types[i] = BINARY;
		else if(pOptVariable->GetType() == eContinuousVariable)
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
		
		if(pOptVariable->GetType() == eIntegerVariable)
			var_types[i] = Ipopt::TNLP::LINEAR;
		else if(pOptVariable->GetType() == eBinaryVariable)
			var_types[i] = Ipopt::TNLP::LINEAR;
		else if(pOptVariable->GetType() == eContinuousVariable)
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
	
		if(pConstraint->IsLinear())
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
	std::vector<size_t> narrOptimizationVariablesIndexes;
	
// Set the number of opt. variables and constraints
	n = m_ptrarrOptVariables.size();
	m = m_ptrarrConstraints.size();
	index_style = TNLP::C_STYLE;
	
	if(n == 0)
		daeDeclareAndThrowException(exInvalidCall)
	
// Set the jacobian number of non-zeroes
	nnz_jac_g = 0;
	nnz_h_lag = 0;
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
		
		narrOptimizationVariablesIndexes.clear();
		pConstraint->GetOptimizationVariableIndexes(narrOptimizationVariablesIndexes);
	
		nnz_jac_g += narrOptimizationVariablesIndexes.size();
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
			
		x_l[i] = pOptVariable->GetLB();
		x_u[i] = pOptVariable->GetUB();
	}	

	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
	
		if(pConstraint->GetType() == eInequalityConstraint)
		{
			g_l[i] = pConstraint->GetLB();
			g_u[i] = pConstraint->GetUB();
		}
		else if(pConstraint->GetType() == eEqualityConstraint)
		{
			g_l[i] = pConstraint->GetEqualityValue();
			g_u[i] = pConstraint->GetEqualityValue();
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
				
			x[i] = pOptVariable->GetStartingPoint();
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
		
		obj_value = m_pObjectiveFunction->GetValue();
		
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
		
		m_pObjectiveFunction->GetGradients(matSens, grad_f, n);
		
/*	OLD	
		// Iterate and set only the values for the opt. variable indexes in the objective function
		for(j = 0; j < m_pObjectiveFunction->m_narrOptimizationVariablesIndexes.size(); j++)
		{
			grad_f[j] = matSens.GetItem(m_pObjectiveFunction->m_narrOptimizationVariablesIndexes[j], // Sensitivity parameter index
								        m_pObjectiveFunction->m_nEquationIndexInBlock);              // Equation index
		}
*/		
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
				
			g[i] = pConstraint->GetValue();
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
		gi = pConstraint->GetValue();
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
	size_t i, j, counter, paramIndex;
	daeOptimizationConstraint* pConstraint;
	std::vector<size_t> narrOptimizationVariablesIndexes;

	try
	{
		if(values == NULL) 
		{
		// Return the structure only
			counter = 0;
			for(i = 0; i < m_ptrarrConstraints.size(); i++)
			{
				pConstraint = m_ptrarrConstraints[i];
				
				narrOptimizationVariablesIndexes.clear();
				pConstraint->GetOptimizationVariableIndexes(narrOptimizationVariablesIndexes);
					
			// Add indexes for the current constraint
			// Achtung: m_narrOptimizationVariablesIndexes must previously be sorted (in function Initialize)
				for(j = 0; j < narrOptimizationVariablesIndexes.size(); j++)
				{
					iRow[counter] = i; // The row number is 'i' (the current constraint)
					jCol[counter] = narrOptimizationVariablesIndexes[j];
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
			if(!m_pdTempStorage)
				daeDeclareAndThrowException(exInvalidCall)

			// Set all values to 0
			::memset(values, 0, nele_jac * sizeof(Number));

			counter = 0;
			for(i = 0; i < m_ptrarrConstraints.size(); i++)
			{
				pConstraint = m_ptrarrConstraints[i];
											
				narrOptimizationVariablesIndexes.clear();
				pConstraint->GetOptimizationVariableIndexes(narrOptimizationVariablesIndexes);

			// The function GetGradients needs an array of size n (number of opt. variables)
			// Therefore, first I should to set all values to zero (I dont have to, but just in case)
				::memset(m_pdTempStorage, 0, n * sizeof(real_t));
				
			// Call GetGradients to fill the array m_pdTempStorage with gradients
			// ONLY the values for indexes in the current constraint are set!! The rest is left as it is (zero)
				pConstraint->GetGradients(matSens, m_pdTempStorage, n);

			// Copy the values to the gradients array
				for(j = 0; j < narrOptimizationVariablesIndexes.size(); j++)
				{
					paramIndex = narrOptimizationVariablesIndexes[j];
					values[counter] = m_pdTempStorage[paramIndex];
					counter++;
				}
				
/*	OLD			
				for(j = 0; j < pConstraint->m_narrOptimizationVariablesIndexes.size(); j++)
				{
					values[counter] = matSens.GetItem(pConstraint->m_narrOptimizationVariablesIndexes[j], // Sensitivity parameter index
											          pConstraint->m_nEquationIndexInBlock );             // Equation index
					counter++;
				}
*/
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
	size_t j, counter, paramIndex;
	daeOptimizationConstraint* pConstraint;
	std::vector<size_t> narrOptimizationVariablesIndexes;

	try
	{
		if(i >= m_ptrarrConstraints.size())
			daeDeclareAndThrowException(exInvalidCall)

		if(values == NULL) 
		{
		// Return the structure only
			
			pConstraint = m_ptrarrConstraints[i];
				
			narrOptimizationVariablesIndexes.clear();
			pConstraint->GetOptimizationVariableIndexes(narrOptimizationVariablesIndexes);
					
		// Add indexes for the current constraint
		// Achtung: m_narrOptimizationVariablesIndexes must previously be sorted (in function Initialize)
			counter = 0;
			for(j = 0; j < narrOptimizationVariablesIndexes.size(); j++)
			{
				jCol[counter] = narrOptimizationVariablesIndexes[j];
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
			
			narrOptimizationVariablesIndexes.clear();
			pConstraint->GetOptimizationVariableIndexes(narrOptimizationVariablesIndexes);

		// The function GetGradients needs an array of size n (number of opt. variables)
		// Therefore, first I should to set all values to zero (I dont have to, but just in case)
			::memset(m_pdTempStorage, 0, n * sizeof(real_t));
			
		// Call GetGradients to fill the array m_pdTempStorage with gradients
		// ONLY the values for indexes in the current constraint are set!! The rest is left as it is (zero)
			pConstraint->GetGradients(matSens, m_pdTempStorage, n);

		// Copy the values to the gradients array
			for(j = 0; j < narrOptimizationVariablesIndexes.size(); j++)
			{
				paramIndex = narrOptimizationVariablesIndexes[j];
				values[counter] = m_pdTempStorage[paramIndex];
				counter++;
			}
			if(nele_grad_gi != counter)
				daeDeclareAndThrowException(exInvalidCall)
			
/*			
			for(j = 0; j < pConstraint->m_narrOptimizationVariablesIndexes.size(); j++)
			{
				values[counter] = matSens.GetItem(pConstraint->m_narrOptimizationVariablesIndexes[j], // Sensitivity parameter index
												  pConstraint->m_nEquationIndexInBlock );             // Equation index
				counter++;
			}
			if(nele_grad_gi != counter)
				daeDeclareAndThrowException(exInvalidCall)
*/
				
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
	
	m_pLog->Message(string(" "), 0);
	m_pLog->Message("                        " + strMessage, 0);
	m_pLog->Message(string(" "), 0);
	m_pLog->Message(string(" "), 0);
	
	m_pLog->Message(string("--------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted("Objective function", 25) +  
				 toStringFormatted("Final value",        16);
	m_pLog->Message(strMessage, 0);
	m_pLog->Message(string("--------------------------------------------------------------------------"), 0);
	strMessage = toStringFormatted(m_pObjectiveFunction->GetName(), 25) +   
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
			
		strMessage = toStringFormatted(pOptVariable->GetName(), 25)          +   
					 toStringFormatted(x[i],                    16, 6, true) +
		             toStringFormatted(pOptVariable->GetLB(),     16, 6, true) + 
		             toStringFormatted(pOptVariable->GetUB(),     16, 6, true);
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
			
		strMessage = toStringFormatted(pConstraint->GetName(),  25)          + 
					 toStringFormatted(pConstraint->GetValue(), 16, 6, true) +
					 toStringFormatted(pConstraint->GetLB(),    16, 6, true) + 
					 toStringFormatted(pConstraint->GetUB(),    16, 6, true);
		m_pLog->Message(strMessage, 0);
	}	
	m_pLog->Message(string(" "), 0);
}

void daeMINLP::PrintObjectiveFunction(void)
{
	string strMessage;
	real_t obj_value = m_pObjectiveFunction->GetValue();
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
			
		strMessage = pConstraint->GetName() + " = " + 
					 toStringFormatted<real_t>(pConstraint->GetValue(), -1, 10, true);
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

void daeMINLP::PrintVariablesLinearity(void)
{
	size_t i;
	string strMessage;
	daeOptimizationVariable* pOptVariable;
	
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
		if(pConstraint->IsLinear())
			strMessage = "LINEAR";
		else
			strMessage = "NON_LINEAR";
		
		m_pLog->Message(pConstraint->GetName() + " = " + strMessage, 0);
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
					 toStringFormatted<real_t>(pOptVariable->GetLB(), -1, 10, true) + 
					 ", " + 
					 toStringFormatted<real_t>(pOptVariable->GetUB(), -1, 10, true) + 
					 "]";
		m_pLog->Message(strMessage, 0);
	}	

	m_pLog->Message(string("Constraints bounds:"), 0);
	for(i = 0; i < m_ptrarrConstraints.size(); i++)
	{
		pConstraint = m_ptrarrConstraints[i];
	
		strMessage = pConstraint->GetName() + " bounds = [" + 
					 toStringFormatted<real_t>(pConstraint->GetLB(), -1, 10, true) + 
					 ", " + 
					 toStringFormatted<real_t>(pConstraint->GetUB(), -1, 10, true) + 
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
						toStringFormatted<real_t>(pOptVariable->GetStartingPoint(), -1, 10, true), 0);
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
