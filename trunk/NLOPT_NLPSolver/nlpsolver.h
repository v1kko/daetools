#ifndef DAE_NLPSOLVER_H
#define DAE_NLPSOLVER_H

#include "stdafx.h"
#include "nlpsolver_class_factory.h"
#include "../Core/optimization.h"
#include "../nlp_common.h"
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
	daeNLOPTSolver(nlopt_algorithm algorithm);
	daeNLOPTSolver(const string& algorithm);
	virtual ~daeNLOPTSolver(void);

public:
	virtual void Initialize(daeOptimization_t* pOptimization,
                            daeSimulation_t*   pSimulation, 
							daeDAESolver_t*    pDAESolver, 
							daeDataReporter_t* pDataReporter, 
							daeLog_t*          pLog);
	virtual void Solve(void);
	virtual string GetName(void) const;
	
	double eval_f(unsigned n, const double* x);
	void   eval_grad_f(unsigned n, const double* x, double* grad_f);
	double eval_g(daeOptimizationConstraint_t* pConstraint, unsigned n, const double* x);
	void   eval_grad_g(daeOptimizationConstraint_t* pConstraint,  unsigned n, const double* x, double* grad) ;
	
	double get_xtol_rel(void) const;
	double get_xtol_abs(void) const;
	double get_ftol_rel(void) const;
	double get_ftol_abs(void) const;
	double get_maxtime(void) const;
	int    get_maxeval(void) const;

	void set_xtol_rel(double tol);
	void set_xtol_abs(double tol);
	void set_ftol_rel(double tol);
	void set_ftol_abs(double tol);
	void set_maxtime(double time);
	void set_maxeval(int eval);
	
	void PrintOptions(void);
    void SetOpenBLASNoThreads(int n);

protected:
	string CreateNLOPTErrorMessage(nlopt_result status);
	void PrintSolution(const double* x, double obj_value, nlopt_result status);
	void SetConstraints(void);
	void SetOptimizationVariables(void);
	void CheckAndRun(const double* x);
	void SetAlgorithm(string algorithm);
		
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
