#ifndef DAE_NLPSOLVER_H
#define DAE_NLPSOLVER_H

#include "stdafx.h"
#include "nlpsolver_class_factory.h"
#include "../Core/coreimpl.h"
#include "../Core/optimization.h"
#include <stdio.h>
#include <time.h>
#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>

using namespace dae::core;
using namespace dae::solver;
using namespace dae::activity;
using namespace Ipopt;

namespace dae
{
namespace nlpsolver
{
/******************************************************************
	daeNLP
*******************************************************************/
class DAE_NLPSOLVER_API daeNLP : public TNLP
{
public:
	daeNLP(daeSimulation_t*   pSimulation, 
		   daeDAESolver_t*    pDAESolver, 
		   daeDataReporter_t* pDataReporter, 
		   daeLog_t*          pLog);
	
	virtual ~daeNLP(void);

public:
	virtual bool get_nlp_info(Index& n, 
							  Index& m, 
							  Index& nnz_jac_g,
							  Index& nnz_h_lag, 
							  IndexStyleEnum& index_style);
  
	virtual bool get_bounds_info(Index n, 
								 Number* x_l, 
								 Number* x_u,
								 Index m, 
								 Number* g_l, 
								 Number* g_u);
  
	virtual bool get_starting_point(Index n, 
									bool init_x, 
									Number* x,
									bool init_z, 
									Number* z_L, 
									Number* z_U,
									Index m, 
									bool init_lambda,
									Number* lambda);
  
	virtual bool eval_f(Index n, 
						const Number* x, 
						bool new_x, 
						Number& obj_value);
  
	virtual bool eval_grad_f(Index n, 
							 const Number* x, 
							 bool new_x, 
							 Number* grad_f);
  
	virtual bool eval_g(Index n, 
						const Number* x, 
						bool new_x, 
						Index m, 
						Number* g);
  
	virtual bool eval_jac_g(Index n, 
							const Number* x,
							bool new_x,
							Index m, 
							Index nele_jac, 
							Index* iRow, 
							Index *jCol,
							Number* values);
  
	virtual bool eval_h(Index n, 
						const Number* x, 
						bool new_x,
						Number obj_factor, 
						Index m, 
						const Number* lambda,
						bool new_lambda, 
						Index nele_hess, 
						Index* iRow,
						Index* jCol, 
						Number* values);
	
	virtual bool intermediate_callback( AlgorithmMode mode,
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
										IpoptCalculatedQuantities* ip_cq);
  
	virtual void finalize_solution(SolverReturn status,
								   Index n, 
								   const Number* x, 
								   const Number* z_L,
								   const Number* z_U,
								   Index m, 
								   const Number* g, 
								   const Number* lambda,
								   Number obj_value,
								   const IpoptData* ip_data,
								   IpoptCalculatedQuantities* ip_cq);
	
protected:
	void CopyOptimizationVariablesToSimulationAndRun(const Number* x);

public:
	daeSimulation_t*	m_pSimulation;
	daeDAESolver_t*		m_pDAESolver;
	daeLog_t*			m_pLog;
	daeDataReporter_t*	m_pDataReporter;
	std::vector<size_t>	m_narrOptimizationVariableIndexes;
	
	daeObjectiveFunction*					 m_pObjectiveFunction;
	std::vector<daeOptimizationConstraint*>  m_ptrarrConstraints;
	std::vector<daeOptimizationVariable*>    m_ptrarrOptVariables;
	
	int m_iRunCounter;
};

/******************************************************************
	daeIPOPT
*******************************************************************/
class DAE_NLPSOLVER_API daeIPOPTSolver : public daeNLPSolver_t
{
public:
	daeIPOPTSolver(void);
	virtual ~daeIPOPTSolver(void);

public:
	virtual void Initialize(daeSimulation_t*   pSimulation, 
							daeDAESolver_t*    pDAESolver, 
							daeDataReporter_t* pDataReporter, 
							daeLog_t*          pLog);
	virtual void Solve(void);
	
	void SetOption(const string& strOptionName, const string& strValue);
	void SetOption(const string& strOptionName, real_t dValue);
	void SetOption(const string& strOptionName, int iValue);
	
protected:
	daeSimulation_t*			m_pSimulation;
	daeDAESolver_t*				m_pDAESolver;
	daeLog_t*					m_pLog;
	daeDataReporter_t*			m_pDataReporter;
	SmartPtr<TNLP>				m_NLP;	
	SmartPtr<IpoptApplication>	m_Application;
};

}
}
#endif
