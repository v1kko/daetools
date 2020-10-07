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
 - Discontinuous equations (symmetrical state transition networks: daeIF statements)

Here we have a very simple heat balance:
    ro * cp * dT/dt - Qin = h * A * (T - Tsurr)

A piece of copper (a plate) is at one side exposed to the source of heat and at the
other to the surroundings. The process starts at the temperature of the metal of 283K.
The metal is allowed to warm up for 200 seconds and then the heat source is
removed and the metal cools down slowly again to the ambient temperature.
*/

#include "../dae_develop.h"
#include "../variable_types.h"
namespace vt = variable_types;

using units_pool::m;
using units_pool::kg;
using units_pool::K;
using units_pool::J;
using units_pool::W;
using units_pool::s;

class modTutorial4 : public daeModel
{
daeDeclareDynamicClass(modTutorial4)
public:
    daeParameter mass;
    daeParameter c_p;
    daeParameter alpha;
    daeParameter A;
    daeParameter T_surr;
    daeVariable Q_in;
    daeVariable T;

    modTutorial4(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        mass("m", kg, this, "Mass of the copper plate, kg"),
        c_p("c_p", J/(kg*K), this, "Specific heat capacity of the plate"),
        alpha("&alpha;", W/((m^2) * K), this, "Heat transfer coefficient"),
        A("A", m ^ 2, this, "Area of the plate"),
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
        Symmetrical STNs in DAE Tools can be created by using IF/ELSE_IF/ELSE/END_IF statements.
        These statements are more or less used as normal if/else if/else blocks in all programming languages.
        An important rule is that all states MUST contain the SAME NUMBER OF EQUATIONS.
        First start with the call to IF( condition ) function from daeModel class.
        After that call, write equations that will be active if 'condition' is satisfied.
        If there are only two states call the function ELSE() and write equations that will be active
        if 'condition' is not satisfied.
        If there are more than two states, start a new state by calling the function ELSE_IF (condition2)
        and write the equations that will be active if 'condition2' is satisfied. And so on...
        Finally call the function END_IF() to finalize the state transition network.
        There is an optional argument EventTolerance of functions IF and ELSE_IF. It is used by the solver
        to control the process of discovering the discontinuities.
        Details about the EventTolerance purpose will be given for the condition time < 200, given below.
        Conditions like time < 200 will be internally transformed into the following equations:
               time - 200 - EventTolerance = 0
               time - 200 = 0
               time - 200 + EventTolerance = 0
        where EventTolerance is used to control how far will solver go after/before discovering a discontinuity.
        The default value is 1E-7. Therefore, the above expressions will transform into:
               time - 199.9999999 = 0
               time - 200         = 0
               time - 200.0000001 = 0
        For example, if the variable 'time' is increasing from 0 and is approaching the value of 200,
        the equation 'Q_on' will be active. As the simulation goes on, the variable 'time' will reach the value
        of 199.9999999 and the solver will discover that the expression 'time - 199.9999999' became equal to zero.
        Then it will check if the condition 'time < 200' is satisfied. It is, and no state change will occur.
        The solver will continue, the variable 'time' will increase to 200 and the solver will discover that
        the expression 'time - 200' became equal to zero. It will again check the condition 'time < 200' and
        find out that it is not satisfied. Now the state ELSE becomes active, and the solver will use equations
        from that state (in this example equation 'Q_off').
        But, if we have 'time > 200' condition instead, we can see that when the variable 'time' reaches 200
        the expression 'time - 200' becomes equal to zero. The solver will check the condition 'time > 200'
        and will find out that it is not satisfied and no state change will occur. However, once the variable
        'time' reaches the value of 200.0000001 the expression 'time - 200.0000001' becomes equal to zero.
        The solver will check the condition 'time > 200' and will find out that it is satisfied and it will
        go to the state ELSE.
        In this example, input power of the heater will be 1500 Watts if the time is less than 200.
        Once we reach 200 seconds the heater is switched off (power is 0 W) and the sytem starts to cool down.
     */
        IF(Time() < Constant(200 * s));
        {
            eq = CreateEquation("Q_on", "The heater is on");
            eq->SetResidual(Q_in() - Constant(1500 * W));
        }
        ELSE();
        {
            eq = CreateEquation("Q_off", "The heater is off");
            eq->SetResidual(Q_in());
        }
        END_IF();
    }
};

class simTutorial4 : public daeSimulation
{
public:
    modTutorial4 M;
    
public:
    simTutorial4(void) : M("cdaeTutorial4")
    {
        SetModel(&M);
        M.SetDescription("This tutorial explains how to define and use discontinuous equations: symmetric state transition networks (daeIF). \n"
                         "A piece of copper (a plate) is at one side exposed to the source of heat and at the "
                         "other to the surroundings. The process starts at the temperature of the metal of 283K. "
                         "The metal is allowed to warm up for 200 seconds and then the heat source is "
                         "removed and the metal cools down slowly again to the ambient temperature.");
    }

public:
    void SetUpParametersAndDomains(void)
    {
		M.c_p.SetValue(385 * J/(kg*K));
        M.mass.SetValue(1 * kg);
        M.alpha.SetValue(200 * W/((m^2)*K));
        M.A.SetValue(0.1 * (m^2));
        M.T_surr.SetValue(283 * K);
    }

    void SetUpVariables(void)
    {
        M.T.SetInitialCondition(283 * K);
    }
};

int main(int argc, char *argv[])
{ 
    boost::scoped_ptr<daeSimulation_t>      pSimulation(new simTutorial4);  
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

    pSimulation->SetReportingInterval(10);
    pSimulation->SetTimeHorizon(500);
    pSimulation->GetModel()->SetReportingOn(true);

    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());

    pSimulation->SolveInitial();
    
    pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
    pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
  
    pSimulation->Run();
    pSimulation->Finalize();
} 

