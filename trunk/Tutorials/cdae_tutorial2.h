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
 - Arrays (discrete distribution domains)
 - Distributed parameters
 - Number of degrees of freedom and how to fix it
 - Initial guess of the variables
*/

#include "variable_types.h"

class modTutorial2 : public daeModel
{
daeDeclareDynamicClass(modTutorial2)
public:
    daeDomain x;
    daeDomain y;
    daeDomain Nq;
    daeParameter a;
    daeParameter b;
    daeParameter Q;
    daeParameter k;
    daeVariable c_p;
    daeVariable rho;
    daeVariable T;

    modTutorial2(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        x("x", this, "X axis domain"),
        y("y", this, "Y axis domain"),
        Nq("Nq", this, "Number of heat fluxes"),
        a("a", eReal, this, "Coefficient for calculation of cp"),
        b("b", eReal, this, "Coefficient for calculation of cp"),
        Q("Q", eReal, this, "Heat flux array at the edges of the plate (bottom/top), W/m2", &Nq),
        k("&lambda;", eReal, this, "Thermal conductivity of the plate, W/mK", &x, &y),
        c_p("c_p", specific_heat_capacity_t, this, "Specific heat capacity of the plate, J/kgK", &x, &y),
        rho("&rho;", density_t, this, "Density of the plate, kg/m3"),
        T("T", temperature_t, this, "Temperature of the plate, K", &x, &y)
    {
    }

    void DeclareEquations(void)
    {
        daeEquation* eq;
		daeDEDI *dx, *dy;

		eq = CreateEquation("HeatBalance", "Heat balance equation. Valid on the open x and y domains");
		dx = eq->DistributeOnDomain(x, eOpenOpen);
		dy = eq->DistributeOnDomain(y, eOpenOpen);
		eq->SetResidual(((rho() * c_p(dx, dy)) * T.dt(dx, dy)) - (k(dx, dy) * (T.d2(x, dx, dy) + T.d2(y, dx, dy))));

		eq = CreateEquation("BC_bottom", "Boundary conditions for the bottom edge");
		dx = eq->DistributeOnDomain(x, eClosedClosed);
		dy = eq->DistributeOnDomain(y, eLowerBound);
		eq->SetResidual((((-k(dx, dy))) * T.d(y, dx, dy)) - Q(0));

		eq = CreateEquation("BC_top", "Boundary conditions for the top edge");
		dx = eq->DistributeOnDomain(x, eClosedClosed);
		dy = eq->DistributeOnDomain(y, eUpperBound);
		eq->SetResidual((((-k(dx, dy))) * T.d(y, dx, dy)) - Q(1));

		eq = CreateEquation("BC_left", "Boundary conditions at the left edge");
		dx = eq->DistributeOnDomain(x, eLowerBound);
		dy = eq->DistributeOnDomain(y, eOpenOpen);
		eq->SetResidual(T.d(x, dx, dy));

		eq = CreateEquation("BC_right", "Boundary conditions for the right edge");
		dx = eq->DistributeOnDomain(x, eUpperBound);
		dy = eq->DistributeOnDomain(y, eOpenOpen);
		eq->SetResidual(T.d(x, dx, dy));

		eq = CreateEquation("C_p", "Equation to calculate the specific heat capacity of the plate as a function of the temperature.");
		dx = eq->DistributeOnDomain(x, eClosedClosed);
		dy = eq->DistributeOnDomain(y, eClosedClosed);
		eq->SetResidual((c_p(dx, dy) - a()) - (b() * T(dx, dy)));
    }
};

class simTutorial2 : public daeSimulation
{
public:
    modTutorial2 m;
    
public:
    simTutorial2(void) : m("cdaeTutorial2")
    {
        SetModel(&m);
        m.SetDescription("This tutorial explains how to define Arrays (discrete distribution domains) and "
                         "distributed parameters, how to calculate the number of degrees of freedom (NDOF) "
                         "and how to fix it, and how to set initial guesses of the variables.");
    }

public:
    void SetUpParametersAndDomains(void)
    {
        const size_t n = 25;

        m.x.CreateDistributed(eCFDM, 2, n, 0, 0.1);
        m.y.CreateDistributed(eCFDM, 2, n, 0, 0.1);
        
        m.Nq.CreateArray(2);

        m.a.SetValue(367.0);
        m.b.SetValue(0.07);

        m.Q.SetValue(0, 1e6);
        m.Q.SetValue(1, 0.0);

        for(size_t x = 0; x < m.x.GetNumberOfPoints(); x++)
            for(size_t y = 0; y < m.y.GetNumberOfPoints(); y++)
                m.k.SetValue(x, y, 401);
    }

    void SetUpVariables(void)
    {
    /*
     In the above model we defined 2*N*N+1 variables and 2*N*N equations,
     meaning that the number of degrees of freedom (NDoF) is equal to: 2*N*N+1 - 2*N*N = 1
     Therefore, we have to assign a value of one of the variables.
     This variable cannot be chosen randomly, but must be chosen so that the combination
     of defined equations and assigned variables produce a well posed system (that is a set of 2*N*N independent equations).
     In our case the only candidate is ro. However, in more complex models there can be many independent combinations of variables.
     The degrees of freedom can be fixed by assigning the variable value by using a function AssignValue:
    */
        m.rho.AssignValue(8960);
    
    /*
     To help the DAE solver it is possible to set initial guesses of of the variables.
     Closer the initial guess is to the solution - faster the solver will converge to the solution
     Just for fun, here we will try to obstruct the solver by setting the initial guess which is rather far from the solution.
     Despite that, the solver will successfully initialize the system!
    */
        m.T.SetInitialGuesses(1000);
        for(size_t x = 0; x < m.x.GetNumberOfPoints(); x++)
            for(size_t y = 0; y < m.y.GetNumberOfPoints(); y++)
                m.c_p.SetInitialGuess(x, y, 1000);

        for(size_t x = 1; x < m.x.GetNumberOfPoints()-1; x++)
            for(size_t y = 1; y < m.y.GetNumberOfPoints()-1; y++)
                m.T.SetInitialCondition(x, y, 300);
    }
};

void simulateTutorial2(void)
{ 
    boost::scoped_ptr<daeSimulation_t>      pSimulation(new simTutorial2);  
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
    pSimulation->SetTimeHorizon(1000);
    pSimulation->GetModel()->SetReportingOn(true);
    
    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
    pSimulation->SolveInitial();
    
    pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
    pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
  
    pSimulation->Run();
    pSimulation->Finalize();
} 

