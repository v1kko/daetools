#include "stdafx.h"
#include "optimization.h"
#include <stdio.h>
#include <time.h>
#include <IpIpoptApplication.hpp>

namespace dae
{
namespace activity
{
/******************************************************************
	daeNLP
*******************************************************************/
daeNLP::daeNLP(daeSimulation_t*   pSimulation, 
			   daeDAESolver_t*    pDAESolver, 
			   daeDataReporter_t* pDataReporter, 
			   daeLog_t*          pLog)
{
	m_pSimulation	     = pSimulation;
	m_pDAESolver		 = pDAESolver;
	m_pDataReporter		 = pDataReporter;
	m_pLog			     = pLog;
}

daeNLP::~daeNLP(void)
{
}

// returns the size of the problem
bool daeNLP::get_nlp_info(Index& n, 
							Index& m, 
							Index& nnz_jac_g,
							Index& nnz_h_lag, 
							IndexStyleEnum& index_style)
{
	size_t i;
	daeObjectiveFunction* pObjectiveFunction;
	daeOptimizationConstraint* pConstraint;
	daeOptimizationVariable* pOptVariable;
	std::vector<daeOptimizationConstraint*> ptrarrConstraints;
	std::vector<daeOptimizationVariable*> ptrarrOptVariables;
	
	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
		
	pSimulation->GetOptimizationConstraints(ptrarrConstraints);
	pSimulation->GetOptimizationVariables(ptrarrOptVariables);
	pObjectiveFunction = pSimulation->GetObjectiveFunction();
		
// Set the indexes of the optimizationvariables 
	m_narrOptimizationVariableIndexes.clear();
	for(i = 0; i < ptrarrOptVariables.size(); i++)
	{
		pOptVariable = ptrarrOptVariables[i];
		if(!pOptVariable)
			daeDeclareAndThrowException(exInvalidPointer)
			
		m_narrOptimizationVariableIndexes.push_back(pOptVariable->GetIndex());
	}	
	
// Set the number of opt. variables and constraints
	n = m_narrOptimizationVariableIndexes.size();
	m = ptrarrConstraints.size();
	index_style = TNLP::C_STYLE;
	
// Set the jacobian number of non-zeroes
	nnz_jac_g = 0;
	nnz_h_lag = 0;
	for(i = 0; i < ptrarrConstraints.size(); i++)
	{
		pConstraint = ptrarrConstraints[i];
		if(!pConstraint)
			daeDeclareAndThrowException(exInvalidPointer)
	
		nnz_jac_g += pConstraint->m_narrOptimizationVariablesIndexes.size();
	}
	
//	Index idx=0;
//	for (Index row = 0; row < 4; row++)
//	{
//		for (Index col = 0; col <= row; col++)
//		{
//			iRow[idx] = row;
//			jCol[idx] = col;
//			idx++;
//		}
//	}
	
	return true;
}

// returns the variable bounds
bool daeNLP::get_bounds_info(Index n, 
							   Number* x_l, Number* x_u,
							   Index m, 
							   Number* g_l, Number* g_u)
{
	size_t i, j;
	daeOptimizationConstraint* pConstraint;
	daeOptimizationVariable* pOptVariable;
	daeObjectiveFunction* pObjectiveFunction;
	std::vector<daeOptimizationConstraint*> ptrarrConstraints;
	std::vector<daeOptimizationVariable*> ptrarrOptVariables;

	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
	
	pSimulation->GetOptimizationConstraints(ptrarrConstraints);
	pSimulation->GetOptimizationVariables(ptrarrOptVariables);
	pObjectiveFunction = pSimulation->GetObjectiveFunction();

	for(i = 0; i < ptrarrOptVariables.size(); i++)
	{
		pOptVariable = ptrarrOptVariables[i];
		if(!pOptVariable)
			daeDeclareAndThrowException(exInvalidPointer)
			
		x_l[i] = pOptVariable->m_dLB;
		x_u[i] = pOptVariable->m_dUB;
	}	

	for(i = 0; i < ptrarrConstraints.size(); i++)
	{
		pConstraint = ptrarrConstraints[i];
		if(!pConstraint)
			daeDeclareAndThrowException(exInvalidPointer)
	
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

// returns the initial point for the problem
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
//	// Here, we assume we only have starting values for x, if you code
//	// your own NLP, you can provide starting values for the dual variables
//	// if you wish
//	assert(init_x == true);
//	assert(init_z == false);
//	assert(init_lambda == false);
	
	size_t i;
	daeOptimizationVariable* pOptVariable;
	daeObjectiveFunction* pObjectiveFunction;
	std::vector<daeOptimizationConstraint*> ptrarrConstraints;
	std::vector<daeOptimizationVariable*> ptrarrOptVariables;

	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
	
	pSimulation->GetOptimizationConstraints(ptrarrConstraints);
	pSimulation->GetOptimizationVariables(ptrarrOptVariables);
	pObjectiveFunction = pSimulation->GetObjectiveFunction();

	for(i = 0; i < ptrarrOptVariables.size(); i++)
	{
		pOptVariable = ptrarrOptVariables[i];
		if(!pOptVariable)
			daeDeclareAndThrowException(exInvalidPointer)
			
		x[i] = pOptVariable->m_dDefaultValue;
	}	
	
	return true;
}

// returns the value of the objective function
bool daeNLP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
	daeObjectiveFunction* pObjectiveFunction;
	
	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
	
	pObjectiveFunction = pSimulation->GetObjectiveFunction();
	
	try
	{
		if(!new_x)
			CopyOptimizationVariablesToSimulationAndRun(x);
		
		obj_value = pObjectiveFunction->m_pObjectiveVariable->GetValue();
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeIPOPT: ") + e.what(), 0);
		return false;
	}
	
	return true;
}

// return the gradient of the objective function grad_{x} f(x)
bool daeNLP::eval_grad_f(Index n, 
						   const Number* x, 
						   bool new_x, 
						   Number* grad_f)
{
	size_t j;
	daeObjectiveFunction* pObjectiveFunction;
	
	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
	
	pObjectiveFunction = pSimulation->GetObjectiveFunction();

	try
	{
		if(!new_x)
			CopyOptimizationVariablesToSimulationAndRun(x);

		daeMatrix<real_t>& matSens = m_pDAESolver->GetSensitivities();
		if(n != matSens.GetNcols())
			daeDeclareAndThrowException(exInvalidCall)
		
		// Set all values to 0
		::memset(grad_f, 0, n * sizeof(Number));
		
		// Iterate and set only the values for the opt. variable indexes in the objective function
		for(j = 0; j < pObjectiveFunction->m_narrOptimizationVariablesIndexes.size(); j++)
			grad_f[j] = matSens.GetItem(pObjectiveFunction->m_narrOptimizationVariablesIndexes[j], // Sensitivity parameter index
								        pObjectiveFunction->m_nEquationIndexInBlock);              // Equation index
		
//		m_pDAESolver->GetSensitivities(pObjectiveFunction->m_nEquationIndexInBlock, // Indef of the equation in the block 
//									   m_narrOptimizationVariableIndexes,           // Variable indexes to get the sensitivities for 
//									   grad_f);                                     // Pointer on the first value 
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeIPOPT: ") + e.what(), 0);
		return false;
	}

	return true;
}

// return the value of the constraints: g(x)
bool daeNLP::eval_g(Index n, 
					  const Number* x, 
					  bool new_x, 
					  Index m, 
					  Number* g)
{
	size_t i;
	daeOptimizationConstraint* pConstraint;
	std::vector<daeOptimizationConstraint*> ptrarrConstraints;

	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
	
	pSimulation->GetOptimizationConstraints(ptrarrConstraints);

	if(m != ptrarrConstraints.size())
		daeDeclareAndThrowException(exInvalidCall)

	try
	{
		if(!new_x)
			CopyOptimizationVariablesToSimulationAndRun(x);

		for(i = 0; i < ptrarrConstraints.size(); i++)
		{
			pConstraint = ptrarrConstraints[i];
			if(!pConstraint)
				daeDeclareAndThrowException(exInvalidPointer)
				
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

// return the structure or values of the jacobian
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
	daeObjectiveFunction* pObjectiveFunction;
	daeOptimizationConstraint* pConstraint;
	std::vector<daeOptimizationConstraint*> ptrarrConstraints;
	std::vector<daeOptimizationVariable*> ptrarrOptVariables;

	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
	
	pSimulation->GetOptimizationConstraints(ptrarrConstraints);
	pSimulation->GetOptimizationVariables(ptrarrOptVariables);
	pObjectiveFunction = pSimulation->GetObjectiveFunction();

	try
	{
		if(values == NULL) 
		{
		// Return the structure only
			counter = 0;
			for(i = 0; i < ptrarrConstraints.size(); i++)
			{
				pConstraint = ptrarrConstraints[i];
				if(!pConstraint)
					daeDeclareAndThrowException(exInvalidPointer)
					
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
			if(!new_x)
				CopyOptimizationVariablesToSimulationAndRun(x);
	
			daeMatrix<real_t>& matSens = m_pDAESolver->GetSensitivities();
			if(n != matSens.GetNcols())
				daeDeclareAndThrowException(exInvalidCall)

			// Set all values to 0
			::memset(values, 0, nele_jac * sizeof(Number));

			counter = 0;
			for(i = 0; i < ptrarrConstraints.size(); i++)
			{
				pConstraint = ptrarrConstraints[i];
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
		}
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeIPOPT: ") + e.what(), 0);
		return false;
	}

	return true;
}

//return the structure or values of the hessian
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
	daeDeclareAndThrowException(exNotImplemented)

	size_t i, j, counter, Ns;
	daeObjectiveFunction* pObjectiveFunction;
	daeOptimizationConstraint* pConstraint;
	std::vector<daeOptimizationConstraint*> ptrarrConstraints;

	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
	
	pSimulation->GetOptimizationConstraints(ptrarrConstraints);
	pObjectiveFunction = pSimulation->GetObjectiveFunction();

	try
	{
	// Return the structure only
		if(values == NULL) 
		{
			for(i = 0; i < ptrarrConstraints.size(); i++)
			{
				pConstraint = ptrarrConstraints[i];
				if(!pConstraint)
					daeDeclareAndThrowException(exInvalidPointer)
					
//				counter = 0;
//				Ns = pConstraint->m_narrOptimizationVariablesIndexes.size();
//				for(j = 0; j < Ns; j++)
//				{
//					iRow[counter] = 0;
//					jCol[counter] = 0;
//					counter++;
//				}
			}
		}
		else
		{
			if(!new_x)
				CopyOptimizationVariablesToSimulationAndRun(x);
	
		}
	}
	catch(std::exception& e)
	{
		m_pLog->Message(string("Exception occured in daeIPOPT: ") + e.what(), 0);
		return false;
	}


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
	// here is where we would store the solution to variables, or write to a file, etc
	// so we could use the solution.
	
	// For this example, we write the solution to the console
	printf("\n\nSolution of the primal variables, x\n");
	for (Index i=0; i<n; i++) {
		printf("x[%d] = %e\n", i, x[i]);
	}
	
	printf("\n\nSolution of the bound multipliers, z_L and z_U\n");
	for (Index i=0; i<n; i++) {
		printf("z_L[%d] = %e\n", i, z_L[i]);
	}
	for (Index i=0; i<n; i++) {
		printf("z_U[%d] = %e\n", i, z_U[i]);
	}
	
	printf("\n\nObjective value\n");
	printf("f(x*) = %e\n", obj_value);
	
	printf("\nFinal value of the constraints:\n");
	for (Index i=0; i<m ;i++) {
		printf("g(%d) = %e\n", i, g[i]);
	}
}

void daeNLP::CopyOptimizationVariablesToSimulationAndRun(const Number* x)
{
	size_t i, j;
	daeOptimizationVariable* pOptVariable;
	std::vector<daeOptimizationVariable*> ptrarrOptVariables;

	daeSimulation* pSimulation = dynamic_cast<daeSimulation*>(m_pSimulation);
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer)
	
	pSimulation->GetOptimizationVariables(ptrarrOptVariables);

// 1. First reset the simulation and DAE solver in it
	pSimulation->Reset();

// 2. Re-assign the optimization variables
	for(i = 0; i < ptrarrOptVariables.size(); i++)
	{
		pOptVariable = ptrarrOptVariables[i];
		if(!pOptVariable || !pOptVariable->m_pVariable)
			daeDeclareAndThrowException(exInvalidPointer)
			
		pOptVariable->m_pVariable->ReAssignValue(x[i]);
	}	
	  
// I should check which function to call here (or to call both) !!!???
	pSimulation->SolveInitial();
	pSimulation->Run();
}


/******************************************************************
	daeIPOPT
*******************************************************************/
daeIPOPT::daeIPOPT(void)
{
	m_pSimulation	     = NULL;
	m_pNLPSolver		 = NULL;
	m_pDAESolver		 = NULL;
	m_pDataReporter		 = NULL;
	m_pLog			     = NULL;
}

daeIPOPT::~daeIPOPT(void)
{
}

void daeIPOPT::Initialize(daeSimulation_t* pSimulation, 
                          daeNLPSolver_t* pNLPSolver, 
                          daeDAESolver_t* pDAESolver, 
                          daeDataReporter_t* pDataReporter, 
                          daeLog_t* pLog)
{
	time_t start, end;

	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer);
//	if(!pNLPSolver)
//		daeDeclareAndThrowException(exInvalidPointer);
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
	m_pNLPSolver    = pNLPSolver;
	m_pDAESolver    = pDAESolver;
	m_pDataReporter	= pDataReporter;
	m_pLog			= pLog;

	m_NLP = new daeNLP(m_pSimulation, m_pDAESolver, m_pDataReporter, m_pLog);
}

void daeIPOPT::Run(void)
{
	SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
	
	app->Options()->SetStringValue("hessian_approximation", "limited-memory");
	app->Options()->SetNumericValue("tol", 1e-7);
	app->Options()->SetStringValue("linear_solver", "mumps");
	app->Options()->SetStringValue("mu_strategy", "adaptive");
	app->Options()->SetStringValue("output_file", "ipopt.out");

// The following overwrites the default name (ipopt.opt) of the options file
// app->Options()->SetStringValue("option_file_name", "hs071.opt");
	
// Intialize the IpoptApplication and process the options
	ApplicationReturnStatus status;
	status = app->Initialize();
	if(status != Solve_Succeeded) 
	{
		m_pLog->Message("*** Error during initialization!", 0);
		daeDeclareAndThrowException(exInvalidCall)
	}
	
// Ask Ipopt to solve the problem
	status = app->OptimizeTNLP(m_NLP);
	if(status == Solve_Succeeded) 
	{
		m_pLog->Message("*** The problem solved!", 0);
	}
	else
	{
		m_pLog->Message("*** The problem FAILED!", 0);
	}
}

void daeIPOPT::Finalize(void)
{
}





}
}
