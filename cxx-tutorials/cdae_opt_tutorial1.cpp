/********************************************************************************
                 DAE Tools: cDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic
*********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
*********************************************************************************/

#include "../dae_develop.h"
#include "../variable_types.h"
namespace vt = variable_types;

using units_pool::s;

daeVariableType gradient_function_t("gradient_function_t", unit(), -1e+100, 1e+100, 0.1, 1e-08);

class modOptTutorial1 : public daeModel
{
daeDeclareDynamicClass(modOptTutorial1)
public:
    daeVariable x1;
    daeVariable x2;
    daeVariable x3;
    daeVariable x4;
    daeVariable dummy;

    modOptTutorial1(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        x1("x1", vt::no_t, this, ""),
        x2("x2", vt::no_t, this, ""),
        x3("x3", vt::no_t, this, ""),
        x4("x4", vt::no_t, this, ""),
        dummy("dummy", vt::no_t, this, "A dummy variable to satisfy the condition that there should be at least one-state variable and one equation in a model")
    {
    }

	void DeclareEquations(void)
    {
        daeEquation* eq;

		eq = CreateEquation("Dummy", "");
		eq->SetResidual( dummy() );
    }
	
};

class simOptTutorial1 : public daeSimulation
{
public:
	modOptTutorial1 M;
	
public:
	simOptTutorial1(void) : M("cdaeOptTutorial1")
	{
		SetModel(&M);
        M.SetDescription("This tutorial introduces IPOPT NLP solver, its setup and options.");
	}

public:
	void SetUpParametersAndDomains(void)
	{
	}

	void SetUpVariables(void)
	{
		M.x1.AssignValue(1);
        M.x2.AssignValue(5);
        M.x3.AssignValue(5);
        M.x4.AssignValue(1);
	}
	
	void SetUpOptimization(void)
	{
		//daeObjectiveFunction* fobj = dynamic_cast<daeObjectiveFunction*>(GetObjectiveFunction());
		daeObjectiveFunction* fobj = GetObjectiveFunction();
		fobj->SetResidual( M.x1() * M.x4() * (M.x1() + M.x2() + M.x3()) + M.x3() );

		daeOptimizationConstraint* c1 = CreateInequalityConstraint("Constraint 1"); // g(x) >= 25:  25 - x1*x2*x3*x4 <= 0
	    c1->SetResidual( Constant(25) - M.x1() * M.x2() * M.x3() * M.x4() );

		daeOptimizationConstraint* c2 = CreateInequalityConstraint("Constraint 2"); // h(x) == 40
	    c2->SetResidual( M.x1() * M.x1() + M.x2() * M.x2() + M.x3() * M.x3() + M.x4() * M.x4() - Constant(40) );
		
		daeOptimizationVariable* x1 = SetContinuousOptimizationVariable(M.x1, 1, 5, 2);
		daeOptimizationVariable* x2 = SetContinuousOptimizationVariable(M.x2, 1, 5, 2);
		daeOptimizationVariable* x3 = SetContinuousOptimizationVariable(M.x3, 1, 5, 2);
		daeOptimizationVariable* x4 = SetContinuousOptimizationVariable(M.x4, 1, 5, 2);
	}
	
};

int main(int argc, char *argv[])
{ 
	boost::scoped_ptr<daeDataReporter_t>	pDataReporter(daeCreateTCPIPDataReporter());
	boost::scoped_ptr<daeIDASolver>			pDAESolver(new daeIDASolver());
	boost::scoped_ptr<daeNLPSolver_t>		pNLPSolver(daeCreateIPOPTSolver());
	boost::scoped_ptr<daeLog_t>				pLog(daeCreateStdOutLog());
	boost::scoped_ptr<daeSimulation_t>		pSimulation(new simOptTutorial1);  
	boost::scoped_ptr<daeOptimization_t>	pOptimization(new daeOptimization());  
	
	if(!pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!pLog)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!pOptimization)
		daeDeclareAndThrowException(exInvalidPointer); 

	time_t rawtime;
	struct tm* timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);	  
	strftime (buffer, 80, " [%d.%m.%Y %H:%M:%S]", timeinfo);
	string simName = pSimulation->GetModel()->GetName() + buffer;
	if(!pDataReporter->Connect(string(""), simName))
		daeDeclareAndThrowException(exInvalidCall); 

	pSimulation->SetReportingInterval(1);
	pSimulation->SetTimeHorizon(5);
	pSimulation->GetModel()->SetReportingOn(true);
	
	pOptimization->Initialize(pSimulation.get(), pNLPSolver.get(), pDAESolver.get(), pDataReporter.get(), pLog.get());
	
	pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
	pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
  
	pOptimization->Run();
	pOptimization->Finalize();
} 

