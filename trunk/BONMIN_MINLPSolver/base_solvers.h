#ifndef DAE_NLP_BASESOLVERS_H
#define DAE_NLP_BASESOLVERS_H

#include "stdafx.h"
#include "nlpsolver_class_factory.h"
#include "../Core/optimization.h"

using namespace dae::solver;

namespace dae
{
namespace nlpsolver
{
#ifdef daeIPOPT
	daeNLPSolver_t* daeCreateIPOPTSolver(void);
#endif

#ifdef daeBONMIN
	daeNLPSolver_t* daeCreateBONMINSolver(void);
#endif

}
}
#endif
