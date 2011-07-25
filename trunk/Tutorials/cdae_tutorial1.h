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
This tutorial introduces several new concepts:
 - Distribution domains
 - Distributed parameters, variables and equations
 - Boundary and initial conditions

In this example we model a simple heat conduction problem:
   - a conduction through a very thin, rectangular copper plate.
This example should be sufficiently complex to describe all basic DAE Tools features.
For this problem, we need a two-dimensional Cartesian grid in X and Y axis
(here, for simplicity, divided into 10 x 10 segments):

Y axis
    ^
    |
Ly -| T T T T T T T T T T T
    | L + + + + + + + + + R
    | L + + + + + + + + + R
    | L + + + + + + + + + R
    | L + + + + + + + + + R
    | L + + + + + + + + + R
    | L + + + + + + + + + R
    | L + + + + + + + + + R
    | L + + + + + + + + + R
    | L + + + + + + + + + R
 0 -| B B B B B B B B B B B
    --|-------------------|----> X axis
      0                   Lx

Points 'B' at the bottom edge of the plate (for y = 0), and the points 'T' at the top edge of the plate
(for y = Ly) represent the points where the heat is applied.
The plate is considered insulated at the left (x = 0) and the right edges (x = Lx) of the plate (points 'L' and 'R').
To model this type of problem, we have to write a heat balance equation for all interior points except the left, right,
top and bottom edges, where we need to define the Neumann type boundary conditions.

In this problem we have to define the following domains:
 - x: X axis domain, length Lx = 0.1 m
 - y: Y axis domain, length Ly = 0.1 m

the following parameters:
 - ro: copper density, 8960 kg/m3
 - cp: copper specific heat capacity, 385 J/(kgK)
 - k:  copper heat conductivity, 401 W/(mK)
 - Qb: heat flux at the bottom edge of the plate, 1E6 W/m2 (or 100 W/cm2)
 - Qt: heat flux at the top edge of the plate, here set to 0 W/m2

and the following variable:
 - T: the temperature of the plate, K (distributed on x and y domains)

Also, we need to write the following 5 equations:

1) Heat balance:
      ro * cp * dT(x,y) / dt = k * (d2T(x,y) / dx2 + d2T(x,y) / dy2);  for all x in: (0, Lx),
                                                                       for all y in: (0, Ly)

2) Boundary conditions for the bottom edge:
      -k * dT(x,y) / dy = Qin;  for all x in: [0, Lx],
                                and y = 0

3) Boundary conditions for the top edge:
      -k * dT(x,y) / dy = Qin;  for all x in: [0, Lx],
                                and y = Ly

4) Boundary conditions for the left edge:
      dT(x,y) / dx = 0;  for all y in: (0, Ly),
                         and x = 0

5) Boundary conditions for the right edge:
      dT(x,y) / dx = 0;  for all y in: (0, Ly),
                         and x = Ln
*/

#include "variable_types.h"


class modTutorial1 : public daeModel
{
daeDeclareDynamicClass(modTutorial1)
public:
    daeDomain x;
    daeDomain y;
    daeParameter Q_b;
    daeParameter Q_t;
    daeParameter rho;
    daeParameter c_p;
    daeParameter k;
    daeVariable T;

    modTutorial1(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        x("x", this, "X axis domain"),
        y("y", this, "Y axis domain"),
        Q_b("Q_b", eReal, this, "Heat flux at the bottom edge of the plate, W/m2"),
        Q_t("Q_t", eReal, this, "Heat flux at the top edge of the plate, W/m2"),
        rho("&rho;", eReal, this, "Density of the plate, kg/m3"),
        c_p("c_p", eReal, this, "Specific heat capacity of the plate, J/kgK"),
        k("&lambda;_p", eReal, this, "Thermal conductivity of the plate, W/mK"),
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
		eq->SetResidual( rho() * c_p() * T.dt(dx, dy) - k() * (T.d2(x, dx, dy) + T.d2(y, dx, dy)) );

		eq = CreateEquation("BC_bottom", "Boundary conditions for the bottom edge");
		dx = eq->DistributeOnDomain(x, eClosedClosed);
		dy = eq->DistributeOnDomain(y, eLowerBound);
		eq->SetResidual( -k() * T.d(y, dx, dy) - Q_b());

		eq = CreateEquation("BC_top", "Boundary conditions for the top edge");
		dx = eq->DistributeOnDomain(x, eClosedClosed);
		dy = eq->DistributeOnDomain(y, eUpperBound);
		eq->SetResidual( -k() * T.d(y, dx, dy) - Q_t());

		eq = CreateEquation("BC_left", "Boundary conditions at the left edge");
		dx = eq->DistributeOnDomain(x, eLowerBound);
		dy = eq->DistributeOnDomain(y, eOpenOpen);
		eq->SetResidual(T.d(x, dx, dy));

		eq = CreateEquation("BC_righ", "Boundary conditions for the right edge");
		dx = eq->DistributeOnDomain(x, eUpperBound);
		dy = eq->DistributeOnDomain(y, eOpenOpen);
		eq->SetResidual(T.d(x, dx, dy));
    }
};

class simTutorial1 : public daeSimulation
{
public:
	modTutorial1 m;
	
public:
	simTutorial1(void) : m("cdaeTutorial1")
	{
		SetModel(&m);
        m.SetDescription("This tutorial explains how to define and set up domains, ordinary and distributed parameters "
                         "and variables, how to define distributed domains, declare distributed equations and set "
                         "their boundary and initial conditions.");
	}

public:
	void SetUpParametersAndDomains(void)
	{
	/*
		In this example we use the center-finite difference method (CFDM) of 2nd order to discretize the domains x and y.
		The function CreateDistributed can be used to create a distributed domain. It accepts 5 arguments:
		 - DiscretizationMethod: can be eBFDM (backward-), BFDM (forward) and eCFDM (center) finite difference method
		 - Order: currently only 2nd order is implemented
		 - NoIntervals: 25
		 - LowerBound: 0
		 - UpperBound: 0.1
		Here we use 25 intervals. In general any number of intervals can be used. However, the computational costs become
		prohibitive at the very high number (especially if dense linear solvers are used).
	*/
		m.x.CreateDistributed(eCFDM, 2, 25, 0, 0.1);
		m.y.CreateDistributed(eCFDM, 2, 25, 0, 0.1);
		
	// Parameters' value can be set by using a function SetValue.
		m.k.SetValue(401);
		m.c_p.SetValue(385);
		m.rho.SetValue(8960);
		m.Q_b.SetValue(1e6);
		m.Q_t.SetValue(0);
	}

	void SetUpVariables(void)
	{
	/*
		SetInitialCondition function in the case of distributed variables can accept additional arguments
		specifying the indexes in the domains. In this example we loop over the open x and y domains,
		thus we start the loop with 1 and end with NumberOfPoints-1 (for both domains)
	*/
		for(size_t x = 1; x < m.x.GetNumberOfPoints()-1; x++)
			for(size_t y = 1; y < m.y.GetNumberOfPoints()-1; y++)
				m.T.SetInitialCondition(x, y, 300);
	}
};

void runTutorial1(void)
{ 
	boost::scoped_ptr<daeSimulation_t>		pSimulation(new simTutorial1);  
	boost::scoped_ptr<daeDataReporter_t>	pDataReporter(daeCreateTCPIPDataReporter());
	boost::scoped_ptr<daeIDASolver>			pDAESolver(new daeIDASolver());
	boost::scoped_ptr<daeLog_t>				pLog(daeCreateStdOutLog());
	
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
