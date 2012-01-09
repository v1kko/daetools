/********************************************************************************
                 DAE Tools: cDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
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

#include "variable_types.h"
using units_pool::m;
using units_pool::kg;
using units_pool::K;
using units_pool::J;
using units_pool::W;

class extfnPower : public daeScalarExternalFunction
{
public:
    extfnPower(const string& strName, daeModel* pModel, const unit& units, adouble m, adouble cp, adouble dT) 
		: daeScalarExternalFunction(strName, pModel, units)
	{
		daeExternalFunctionArgumentMap_t mapArguments;
		mapArguments["m"]  = m;
		mapArguments["cp"] = cp;
		mapArguments["dT"] = dT;
		SetArguments(mapArguments);
	}
	
	adouble Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const
	{
        adouble m  = boost::get<adouble>(mapValues["m"]);
        adouble cp = boost::get<adouble>(mapValues["cp"]);
        adouble dT = boost::get<adouble>(mapValues["dT"]);
//		std::cout << (boost::format("m = %1%, cp = %2%, dT = %3%, mcpdT = %4%") % m % cp % dT % (m*cp*dT)).str() << std::endl;
        return m * cp * dT;
	}
};

class modTutorial15 : public daeModel
{
daeDeclareDynamicClass(modTutorial15)
public:
    daeParameter mass;
    daeParameter c_p;
    daeParameter alpha;
    daeParameter A;
    daeParameter T_surr;
    daeVariable Q_in;
    daeVariable T;
	daeSTN* stnRegulator;

    daeVariable Power;
	boost::shared_ptr<extfnPower> P;

    modTutorial15(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        mass("m", kg, this, "Mass of the copper plate"),
        c_p("c_p", J/(kg*K), this, "Specific heat capacity of the plate"),
        alpha("&alpha;", W/((m^2)*K), this, "Heat transfer coefficient"),
        A("A", m^2, this, "Area of the plate"),
        T_surr("T_surr", K, this, "Temperature of the surroundings"),
        Q_in("Q_in", power_t, this, "Power of the heater"),
        T("T", temperature_t, this, "Temperature of the plate"),
		
		Power("Power", power_t, this, "External function power")
    {
    }

    void DeclareEquations(void)
    {
        daeEquation* eq;

        eq = CreateEquation("HeatBalance", "Integral heat balance equation");
        eq->SetResidual( mass() * c_p() * T.dt() - Q_in() + alpha() * A() * (T() - T_surr()) );

		P = boost::shared_ptr<extfnPower>(new extfnPower("Power", this, W, mass(), c_p(), T.dt()));
		
        eq = CreateEquation("ExternalFunction", "");
        eq->SetResidual( Power() - (*P)() ); //mass()* c_p() * T.dt() ); //(*P)() );
		
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
        stnRegulator = STN("Regulator");
        STATE("Heating");
            eq = CreateEquation("Q_in", "The heater is on");
            eq->SetResidual(Q_in() - 1500);

            SWITCH_TO("Cooling", 340 < T());
            SWITCH_TO("HeaterOff", 350 < Time());

        STATE("Cooling");
            eq = CreateEquation("Q_in", "The heater is off");
            eq->SetResidual(Q_in());

            SWITCH_TO("Heating", 320 > T());
            SWITCH_TO("HeaterOff", 350 < Time());

        STATE("HeaterOff");
            eq = CreateEquation("Q_in", "The heater is off");
            eq->SetResidual(Q_in());

        END_STN();
    }
};

class simTutorial15 : public daeSimulation
{
public:
    modTutorial15 m;
    
public:
    simTutorial15(void) : m("cdaeTutorial15")
    {
        SetModel(&m);
        m.SetDescription("This tutorial explains how to define and use another type of discontinuous equations: " 
                         "non-symmetric state transition networks (daeSTN). \n" 
                         "A piece of copper (a plate) is at one side exposed to the source of heat and at the " 
                         "other to the surroundings. The process starts at the temperature of the metal of 283K. " 
                         "The metal is allowed to warm up, and then its temperature is kept in the interval " 
                         "[320 - 340] for at least 350 seconds. After 350s the heat source is removed and the metal" 
                         "cools down slowly again to the ambient temperature.");
    }

public:
    void SetUpParametersAndDomains(void)
    {
        m.c_p.SetValue(385);
        m.mass.SetValue(1);
        m.alpha.SetValue(200);
        m.A.SetValue(0.1);
        m.T_surr.SetValue(283);
    }

    void SetUpVariables(void)
    {
    // Set the state active at the beginning (the default is the first declared state; here 'Heating')
        m.stnRegulator->SetActiveState2("Heating");

        m.T.SetInitialCondition(283);
    }
};

void runTutorial15(void)
{ 
    boost::scoped_ptr<daeSimulation_t>      pSimulation(new simTutorial15);  
    boost::scoped_ptr<daeDataReporter_t>    pDataReporter(daeCreateTCPIPDataReporter());
    boost::scoped_ptr<daeIDASolver>         pDAESolver(new daeIDASolver());
    boost::scoped_ptr<daeLog_t>             pLog(daeCreateStdOutLog());
    
    if(!pSimulation)
        daeDeclareAndThrowException(exInvalidPointer); 
    if(!pDataReporter)
        daeDeclareAndThrowException(exInvalidPointer); 
    if(!pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer); 
    if(!pLog)
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

    pSimulation->SetReportingInterval(0.5);
    pSimulation->SetTimeHorizon(500);
    pSimulation->GetModel()->SetReportingOn(true);
    
    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
    pSimulation->SolveInitial();
    
    pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
    pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
  
    pSimulation->Run();
    pSimulation->Finalize();
} 
