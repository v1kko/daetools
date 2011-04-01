#ifndef DAE_NLPSOLVER_H
#define DAE_NLPSOLVER_H

#include "stdafx.h"
#include "nlpsolver_class_factory.h"
#include "../Core/optimization.h"
#include "../BONMIN_MINLPSolver/nlp_common.h"
#include <stdio.h>
#include <time.h>
#include <nlopt.h>

#include <iomanip>
#include <fstream>

using namespace dae::core;
using namespace dae::solver;
using namespace dae::activity;

namespace dae
{
namespace nlpsolver
{
double function(unsigned n, const double* x, double* grad, void* data);
double constraint(unsigned n, const double *x, double *grad, void *data);

/******************************************************************
	daeNLOPTSolver
*******************************************************************/
class daeNLOPTSolver;
struct nloptData
{
	daeNLOPTSolver*              nloptsolver;
	daeOptimizationConstraint_t* constraint;
};

class DAE_NLPSOLVER_API daeNLOPTSolver: public daeNLPSolver_t,
                                        public daeNLPCommon
{
public:
	daeNLOPTSolver(void);
	virtual ~daeNLOPTSolver(void);

public:
	virtual void Initialize(daeSimulation_t*   pSimulation, 
							daeDAESolver_t*    pDAESolver, 
							daeDataReporter_t* pDataReporter, 
							daeLog_t*          pLog);
	virtual void Solve(void);
	
	void            SetAlgorithm(nlopt_algorithm algorithm);
	void            SetAlgorithm(string algorithm);
	nlopt_algorithm GetAlgorithm(void);
	
	double eval_f(unsigned n, const double* x);
	void   eval_grad_f(unsigned n, const double* x, double* grad_f);
	double eval_g(daeOptimizationConstraint_t* pConstraint, unsigned n, const double* x);
	void   eval_grad_g(daeOptimizationConstraint_t* pConstraint,  unsigned n, const double* x, double* grad) ;
	
protected:
	string CreateNLOPTErrorMessage(nlopt_result status);
	void PrintSolution(const double* x, double obj_value, nlopt_result status);
	void SetConstraints(void);
	void SetOptimizationVariables(void);
	void CheckAndRun(const double* x);
		
protected:
	nlopt_opt				m_nlopt;
	nlopt_algorithm			m_nlopt_algorithm;

	std::vector<double>		m_darrLBs;
	std::vector<double>		m_darrUBs;
	std::vector<double>		m_darrX;
	std::vector<double>		m_darrLastX;
	std::vector<nloptData>  m_arrConstraintData;
	std::vector<double>		m_darrTempStorage;
};

}
}
#endif
