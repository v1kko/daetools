#ifndef NLPSOLVER_COMMON_H
#define NLPSOLVER_COMMON_H

#include "../Core/optimization.h"
#include "../Core/helpers.h"

namespace dae
{
namespace nlpsolver
{
class daeNLPCommon
{
public:
	daeNLPCommon(void)
	{		
		m_pSimulation	= NULL;
		m_pDAESolver	= NULL;
		m_pDataReporter	= NULL;
		m_pLog			= NULL;
		m_iRunCounter	= 0;
		m_bPrintInfo	= false;
	}
	
	virtual ~daeNLPCommon(void)
	{		
	}

public:
	void Init(daeSimulation_t*   pSimulation, 
			  daeDAESolver_t*    pDAESolver, 
			  daeDataReporter_t* pDataReporter, 
			  daeLog_t*          pLog)
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
	}
	
	void Calculate_fobj(double& fobj)
	{
		fobj = m_pObjectiveFunction->GetValue();
		
		if(m_bPrintInfo) 
		{
			string strMessage = "Fobj: ";
			strMessage += toStringFormatted<real_t>(fobj, -1, 10, true);
			m_pLog->Message(strMessage, 0);
		}
	}

	void Calculate_fobj_gradient(double* grad_f)
	{
		size_t Nv = m_ptrarrOptVariables.size();
		daeMatrix<real_t>& matSens = m_pDAESolver->GetSensitivities();
		if(Nv != matSens.GetNrows())
			daeDeclareAndThrowException(exInvalidCall)
		
		m_pObjectiveFunction->GetGradients(matSens, grad_f, Nv);
		
		if(m_bPrintInfo) 
		{
			string strMessage;
			m_pLog->Message("Fobj gradient: ", 0);
			for(size_t j = 0; j < Nv; j++)
				strMessage += toStringFormatted<real_t>(grad_f[j], -1, 10, true) + " ";
			m_pLog->Message(strMessage, 0);
		}
	}

	void Calculate_g(daeOptimizationConstraint_t* pConstraint, double& g)
	{
		g = pConstraint->GetValue();
		
		if(m_bPrintInfo) 
		{
			string strMessage;
			m_pLog->Message(pConstraint->GetName() + " value: ", 0);
			strMessage += toStringFormatted<real_t>(g, -1, 10, true) + " ";
			m_pLog->Message(strMessage, 0);
		}
	}

	void Calculate_g_gradient(daeOptimizationConstraint_t* pConstraint, double* grad_g)
	{
		size_t Nv = m_ptrarrOptVariables.size();
		daeMatrix<real_t>& matSens = m_pDAESolver->GetSensitivities();
		if(Nv != matSens.GetNrows())
			daeDeclareAndThrowException(exInvalidCall)
	
		pConstraint->GetGradients(matSens, grad_g, Nv);
	
		if(m_bPrintInfo) 
		{
			string strMessage;
			m_pLog->Message(pConstraint->GetName() + " gradient: ", 0);
			for(size_t j = 0; j < Nv; j++)
				strMessage += toStringFormatted<real_t>(grad_g[j], -1, 10, true) + " ";
			m_pLog->Message(strMessage, 0);
		}		
	}

protected:
	void PrintSolution(double fobj, const double* x, const double* g)
	{
		size_t i;
		string strMessage;
		daeOptimizationVariable_t* pOptVariable;
		daeOptimizationConstraint_t* pConstraint;
			
		m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
		strMessage = toString("Objective function", 25) +  
					 toString("Final value",        16) +
					 toString("Type",               5);
		m_pLog->Message(strMessage, 0);
		m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
		strMessage = toString(m_pObjectiveFunction->GetName(), 25) +   
					 toStringFormatted(fobj, 16, 6, true) +
					 toString((m_pObjectiveFunction->IsLinear() ? "L" : "NL"), 5);
		m_pLog->Message(strMessage, 0);
		m_pLog->Message(string(" "), 0);
	
		m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
		strMessage = toString("Optimization variable", 25) +  
					 toString("Final value",           16) +
					 toString("Lower bound",           16) +
					 toString("Upper bound",           16);
		m_pLog->Message(strMessage, 0);
		m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
		for(i = 0; i < m_ptrarrOptVariables.size(); i++)
		{
			pOptVariable = m_ptrarrOptVariables[i];
				
			strMessage = toString(pOptVariable->GetName(),          25)          +   
						 toStringFormatted(x[i],                    16, 6, true) +
						 toStringFormatted(pOptVariable->GetLB(),   16, 6, true) + 
						 toStringFormatted(pOptVariable->GetUB(),   16, 6, true);
			m_pLog->Message(strMessage, 0);
		}
		m_pLog->Message(string(" "), 0);
	
		m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
		strMessage = toString("Constraint",  25) +  
					 toString("Final value", 16) +
					 toString("Type",         5);
		m_pLog->Message(strMessage, 0);
		m_pLog->Message(string("-------------------------------------------------------------------------------"), 0);
		for(i = 0; i < m_ptrarrConstraints.size(); i++)
		{
			pConstraint = m_ptrarrConstraints[i];
				
			strMessage = toString(pConstraint->GetName(), 25)  + 
						 toStringFormatted(g ? g[i] : pConstraint->GetValue(), 16, 6, true) +
						 toString((pConstraint->IsLinear() ? "L" : "NL"),  5);
			m_pLog->Message(strMessage, 0);
		}	
		m_pLog->Message(string(" "), 0);
	}
	
	void CopyOptimizationVariablesToSimulationAndRun(const double* x)
	{
		size_t i;
		daeOptimizationVariable_t* pOptVariable;
		
		m_pLog->IncreaseIndent(1);
		m_pLog->Message(string("Starting the run No. ") + toString(m_iRunCounter + 1) + string(" ..."), 0);
		
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
	
	void PrintObjectiveFunction(void)
	{
		string strMessage;
		real_t obj_value = m_pObjectiveFunction->GetValue();
		strMessage = "Fobj = " + toStringFormatted<real_t>(obj_value, -1, 10, true);
		m_pLog->Message(strMessage, 0);
	}
	
	void PrintOptimizationVariables(void)
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
	
	void PrintConstraints(void)
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
	
	void PrintVariablesTypes(void)
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
				daeDeclareAndThrowException(exNotImplemented);
						
			m_pLog->Message(pOptVariable->GetName() + " = " + strMessage, 0);
		}	
	}
	
	void PrintVariablesLinearity(void)
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
				daeDeclareAndThrowException(exNotImplemented);
						
			m_pLog->Message(pOptVariable->GetName() + " = " + strMessage, 0);
		}	
	}
		
	void PrintConstraintsLinearity(void)
	{
		size_t i;
		string strMessage;
		daeOptimizationConstraint_t* pConstraint;
		
		m_pLog->Message(string("Constraints linearity:"), 0);
		for(i = 0; i < m_ptrarrConstraints.size(); i++)
		{
			pConstraint = m_ptrarrConstraints[i];
		
			if(pConstraint->IsLinear())
				strMessage = "LINEAR";
			else
				strMessage = "NON_LINEAR";
			
			m_pLog->Message(pConstraint->GetName() + " = " + strMessage, 0);
		}
	}
	
	void PrintBoundsInfo(void)
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
	}
	
	void PrintStartingPoint(void)
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
	
	static void CheckProblem(std::vector<daeOptimizationVariable_t*>& ptrarrOptVariables)
	{
		size_t i;
		daeOptimizationVariable_t* pOptVariable;
		
		for(i = 0; i < ptrarrOptVariables.size(); i++)
		{
			pOptVariable = ptrarrOptVariables[i];
			
			if(pOptVariable->GetType() == eIntegerVariable ||
			   pOptVariable->GetType() == eBinaryVariable)
			{
				daeDeclareException(exRuntimeCheck);
				e << "NLP solvers does not support integer/binary optimization variables - use a MINLP solver";
				throw e;
			}
		}
	}
	
public:
	daeSimulation_t*	m_pSimulation;
	daeDAESolver_t*		m_pDAESolver;
	daeLog_t*			m_pLog;
	daeDataReporter_t*	m_pDataReporter;
	
	real_t*			    m_pdTempStorage;
	
	daeObjectiveFunction_t*					   m_pObjectiveFunction;
	std::vector<daeOptimizationConstraint_t*>  m_ptrarrConstraints;
	std::vector<daeOptimizationVariable_t*>    m_ptrarrOptVariables;
	
	int  m_iRunCounter;
	bool m_bPrintInfo;
};

}
}

#endif
