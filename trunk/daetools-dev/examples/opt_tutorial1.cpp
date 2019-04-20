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

#include <daetools.h>
using namespace daetools::logging;
using namespace daetools::core;
using namespace daetools::solver;
using namespace daetools::datareporting;
using namespace daetools::activity;
namespace vt = daetools::core::variable_types;

using units_pool::s;

daeVariableType gradient_function_t("gradient_function_t", unit(), -1e+100, 1e+100, 0.1, 1e-08);

class modOptTutorial1 : public daeModel
{
public:
    daeVariable x1;
    daeVariable x2;
    daeVariable x3;
    daeVariable x4;
    daeVariable dummy;

    modOptTutorial1(std::string strName, daeModel* pParent = NULL, std::string strDescription = "")
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
    modOptTutorial1 model;

public:
    simOptTutorial1(void) : model("cdaeOptTutorial1")
    {
        SetModel(&model);
    }

public:
    void SetUpParametersAndDomains(void)
    {
    }

    void SetUpVariables(void)
    {
        model.x1.AssignValue(1);
        model.x2.AssignValue(5);
        model.x3.AssignValue(5);
        model.x4.AssignValue(1);
    }

    void SetUpOptimization(void)
    {
        //daeObjectiveFunction* fobj = dynamic_cast<daeObjectiveFunction*>(GetObjectiveFunction());
        daeObjectiveFunction* fobj = GetObjectiveFunction();
        fobj->SetResidual( model.x1() * model.x4() * (model.x1() + model.x2() + model.x3()) + model.x3() );

        daeOptimizationConstraint* c1 = CreateInequalityConstraint("Constraint 1"); // g(x) >= 25:  25 - x1*x2*x3*x4 <= 0
        c1->SetResidual( Constant(25) - model.x1() * model.x2() * model.x3() * model.x4() );

        daeOptimizationConstraint* c2 = CreateInequalityConstraint("Constraint 2"); // h(x) == 40
        c2->SetResidual( model.x1() * model.x1() + model.x2() * model.x2() + model.x3() * model.x3() + model.x4() * model.x4() - Constant(40) );

        daeOptimizationVariable* x1 = SetContinuousOptimizationVariable(model.x1, 1, 5, 2);
        daeOptimizationVariable* x2 = SetContinuousOptimizationVariable(model.x2, 1, 5, 2);
        daeOptimizationVariable* x3 = SetContinuousOptimizationVariable(model.x3, 1, 5, 2);
        daeOptimizationVariable* x4 = SetContinuousOptimizationVariable(model.x4, 1, 5, 2);
    }

};

int main(int argc, char *argv[])
{
    std::unique_ptr<daeDataReporter_t>  pDataReporter(daeCreateTCPIPDataReporter());
    std::unique_ptr<daeDAESolver_t>     pDAESolver   (daeCreateIDASolver());
    std::unique_ptr<daeNLPSolver_t>     pNLPSolver   (daeCreateIPOPTSolver());
    std::unique_ptr<daeLog_t>           pLog         (daeCreateStdOutLog());
    std::unique_ptr<daeSimulation_t>    pSimulation  (new simOptTutorial1);
    std::unique_ptr<daeOptimization_t>	pOptimization(new daeOptimization());

    pDataReporter->Connect("", "cdae_opt_tutorial1-" + daetools::getFormattedDateTime());

    pSimulation->SetReportingInterval(1);
    pSimulation->SetTimeHorizon(5);
    pSimulation->GetModel()->SetReportingOn(true);

    pOptimization->Initialize(pSimulation.get(), pNLPSolver.get(), pDAESolver.get(), pDataReporter.get(), pLog.get());

    pOptimization->Run();
    pOptimization->Finalize();
}

