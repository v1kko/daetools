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

/*
In this example we use the same conduction problem as in the tutorial 1.
Here we introduce:
 - Discontinuous equations (non-symmetrical state transition networks: daeSTN statements)

Here, we have the similar problem as in the tutorial 4. The model is equivalent.
Again we have a piece of copper (a plate) is at one side exposed to the source of heat
and at the other to the surroundings. The process starts at the temperature of 283K.
The metal is allowed to warm up, and then its temperature is kept in the interval
[320 - 340] for at 350 seconds. After 350s the heat source is removed and the metal
cools down slowly again to the ambient temperature.
*/

#include <daetools.h>
using namespace daetools::logging;
using namespace daetools::core;
using namespace daetools::solver;
using namespace daetools::datareporting;
using namespace daetools::activity;
namespace vt = daetools::core::variable_types;

using units_pool::m;
using units_pool::kg;
using units_pool::K;
using units_pool::J;
using units_pool::W;
using units_pool::s;

class modTutorial5 : public daeModel
{
public:
    daeParameter mass;
    daeParameter c_p;
    daeParameter alpha;
    daeParameter A;
    daeParameter T_surr;
    daeVariable Q_in, Q1, Q2;
    daeVariable T;
	daeSTN* stnRegulator;

    modTutorial5(std::string strName, daeModel* pParent = NULL, std::string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        mass("m", kg, this, "Mass of the copper plate"),
        c_p("c_p", J/(kg*K), this, "Specific heat capacity of the plate"),
        alpha("&alpha;", W/((m^2)*K), this, "Heat transfer coefficient"),
        A("A", m^2, this, "Area of the plate"),
        T_surr("T_surr", K, this, "Temperature of the surroundings"),
        Q_in("Q_in", vt::power_t, this, "Power of the heater"),
        T("T", vt::temperature_t, this, "Temperature of the plate")
    {
    }

    void DeclareEquations(void)
    {
        daeEquation* eq;

        eq = CreateEquation("HeatBalance", "Integral heat balance equation");
        eq->SetResidual( mass() * c_p() * T.dt() - Q_in() + alpha() * A() * (T() - T_surr()) );

        /*
        Non-symmetrical STNs in DAE Tools can be created by using STN/STATE/END_STN statements.
        Again, states MUST contain the SAME NUMBER OF EQUATIONS.
        First start with the call to STN("STN_Name") function from daeModel class.
        If you need to change active states in operating procedure in function Run()
        store the stn reference (here in the stnRegulator object).
        After that call, define your states by calling the function STATE("State1") and write
        equations that will be active if this state (called 'State1') is active.
        If there are state transitions, write them by calling the function SWITCH_TO("State2", 'condition').
        This function defines the condition when the state 'State2' becomes the active one.
        Repeat this procedure for all states in the state transition network.
        Finally call the function END_STN() to finalize the state transition network.
        Again, there is an optional argument EventTolerance of the function SWITCH_TO, as explained in tutorial 4.
        */
        std::vector< std::pair<std::string, std::string> >                switchToCooling   = {{"Regulator", "Cooling"}};
        std::vector< std::pair<std::string, std::string> >                switchToHeating   = {{"Regulator", "Heating"}};
        std::vector< std::pair<std::string, std::string> >                switchToHeaterOff = {{"Regulator", "HeaterOff"}};
        std::vector< std::pair<daeVariableWrapper, adouble> >   setVariables;
        std::vector< std::pair<daeEventPort*, adouble> >        triggerEvents;
        std::vector<daeAction*>                                 userDefinedActions;
		
        stnRegulator = STN("Regulator");
        STATE("Heating");
            eq = CreateEquation("Q_in", "The heater is on");
            eq->SetResidual(Q_in() - Constant(1500*W));

            ON_CONDITION(T()    > Constant(340*K), switchToCooling,   setVariables, triggerEvents, userDefinedActions);
			ON_CONDITION(Time() > Constant(350*s), switchToHeaterOff, setVariables, triggerEvents, userDefinedActions);

        STATE("Cooling");
            eq = CreateEquation("Q_in", "The heater is off");
            eq->SetResidual(Q_in());

            ON_CONDITION(Constant(320*K) > T(),    switchToHeating,   setVariables, triggerEvents, userDefinedActions);
            ON_CONDITION(Constant(350*s) < Time(), switchToHeaterOff, setVariables, triggerEvents, userDefinedActions);

        STATE("HeaterOff");
            eq = CreateEquation("Q_in", "The heater is off");
            eq->SetResidual(Q_in());

		END_STN();
    }
};

class simTutorial5 : public daeSimulation
{
public:
    modTutorial5 model;
    
public:
    simTutorial5(void) : model("cdaeTutorial5")
    {
        SetModel(&model);
    }

public:
    void SetUpParametersAndDomains(void)
    {
        model.c_p.SetValue(385 * J/(kg*K));
        model.mass.SetValue(1 * kg);
        model.alpha.SetValue(200 * W/((m^2)*K));
        model.A.SetValue(0.1 * (m^2));
        model.T_surr.SetValue(283 * K);
    }

    void SetUpVariables(void)
    {
        // Set the state active at the beginning (the default is the first declared state; here 'Heating')
        model.stnRegulator->SetActiveState("Heating");

        model.T.SetInitialCondition(283 * K);
    }
};

int main(int argc, char *argv[])
{ 
    std::unique_ptr<daeSimulation_t>    pSimulation  (new simTutorial5);
    std::unique_ptr<daeDataReporter_t>  pDataReporter(daeCreateTCPIPDataReporter());
    std::unique_ptr<daeDAESolver_t>     pDAESolver   (daeCreateIDASolver());
    std::unique_ptr<daeLog_t>           pLog         (daeCreateStdOutLog());
	std::unique_ptr<daeLASolver_t>      pLASolver    (daeCreateSuperLUSolver());

    pDAESolver->SetLASolver(pLASolver.get());

    pDataReporter->Connect("", "cdae_tutorial5-" + daetools::getFormattedDateTime());

    pSimulation->SetReportingInterval(2);
    pSimulation->SetTimeHorizon(500);
    pSimulation->GetModel()->SetReportingOn(true);
    
    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
    pSimulation->SolveInitial();
    pSimulation->Run();
    pSimulation->Finalize();
} 
