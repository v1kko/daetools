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
 - Arrays of variable values
 - Functions that operate on arrays of values
 - Non-uniform domain grids
*/

#include "variable_types.h"


class modTutorial3 : public daeModel
{
daeDeclareDynamicClass(modTutorial3)
public:
    daeDomain x;
    daeDomain y;
    daeParameter Q_b;
    daeParameter Q_t;
    daeParameter rho;
    daeParameter c_p;
    daeParameter k;
    daeVariable T_ave;
    daeVariable T_sum;
    daeVariable T;

    modTutorial3(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        x("x", this, "X axis domain"),
        y("y", this, "Y axis domain"),
        Q_b("Q_b", eReal, this, "Heat flux at the bottom edge of the plate, W/m2"),
        Q_t("Q_t", eReal, this, "Heat flux at the top edge of the plate, W/m2"),
        rho("&rho;", eReal, this, "Density of the plate, kg/m3"),
        c_p("c_p", eReal, this, "Specific heat capacity of the plate, J/kgK"),
        k("&lambda;", eReal, this, "Thermal conductivity of the plate, W/mK"),
        T_ave("T_ave", temperature_t, this, "The average temperature, K"),
        T_sum("T_sum", temperature_t, this, "The sum of heat fluxes at the bottom edge of the plate, W/m2"),
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
        eq->SetResidual(((rho() * c_p()) * T.dt(dx, dy)) - (k() * (T.d2(x, dx, dy) + T.d2(y, dx, dy))));

        eq = CreateEquation("BC_bottom", "Boundary conditions for the bottom edge");
        dx = eq->DistributeOnDomain(x, eClosedClosed);
        dy = eq->DistributeOnDomain(y, eLowerBound);
        eq->SetResidual((((-k())) * T.d(y, dx, dy)) - Q_b());

        eq = CreateEquation("BC_top", "Boundary conditions for the top edge");
        dx = eq->DistributeOnDomain(x, eClosedClosed);
        dy = eq->DistributeOnDomain(y, eUpperBound);
        eq->SetResidual((((-k())) * T.d(y, dx, dy)) - Q_t());
    
        eq = CreateEquation("BC_left", "Boundary conditions at the left edge");
        dx = eq->DistributeOnDomain(x, eLowerBound);
        dy = eq->DistributeOnDomain(y, eOpenOpen);
        eq->SetResidual(T.d(x, dx, dy));

        eq = CreateEquation("BC_right", "Boundary conditions for the right edge");
        dx = eq->DistributeOnDomain(x, eUpperBound);
        dy = eq->DistributeOnDomain(y, eOpenOpen);
        eq->SetResidual(T.d(x, dx, dy));

    /*
         There are several function that return arrays of values (or time- or partial-derivatives)
         such as daeParameter and daeVariable functions array(), which return an array of parameter/variable values
         To obtain the array of values it is necessary to define points from all domains that the parameter
         or variable is distributed on. Functions that return array of values accept daeIndexRange objects as
         their arguments. daeIndexRange constructor has three variants:
          1. The first one accepts a single argument: Domain
             in that case the array will contain all points from the domains
          2. The second one accepts 2 arguments: Domain and Indexes
             the argument indexes is a list of indexes within the domain and the array will contain the values
             of the variable at those points
          3. The third one accepts 4 arguments: Domain, StartIndex, EndIndex, Step
             Basically this defines a slice on the array of points in the domain
             StartIndex is the starting index, EndIndex is the last index and Step is used to iterate over
             this sub-domain [StartIndex, EndIndex). For example if we want values at even indexes in the domain
             we can write: xr = daeDomainIndex(x, 0, -1, 2)
         In this example we want to calculate:
          a) the average temperature of the plate
          b) the sum of heat fluxes at the bottom edge of the plate (at y = 0)
         Thus we use the first version of the above daeIndexRange constructor.
         To calculate the average and the sum of heat fluxes we can use functions 'average' and 'sum' from daeModel class.
         For the list of all available functions please have a look on pyDAE API Reference, module Core.
    */
        daeIndexRange xr(&x);
        daeIndexRange yr(&y);

        eq = CreateEquation("T_ave", "The average temperature of the plate");
        eq->SetResidual(T_ave() - average(T.array(xr, yr)));

        eq = CreateEquation("T_sum", "The sum of the plate temperatures");
        eq->SetResidual(T_sum() + k() * sum(T.d_array(y, xr, 0)));
    }
};

class simTutorial3 : public daeSimulation
{
public:
    modTutorial3 m;
    
public:
    simTutorial3(void) : m("cdaeTutorial3")
    {
        SetModel(&m);
        m.SetDescription("This tutorial explains how to define arrays of variable values and "
                         "functions that operate on these arrays, and how to define a non-uniform domain grid.");
    }

public:
    void SetUpParametersAndDomains(void)
    {
        const size_t n = 10;

        m.x.CreateDistributed(eCFDM, 2, n, 0, 0.1);
        m.y.CreateDistributed(eCFDM, 2, n, 0, 0.1);

    /*
        Points in distributed domains can be changed after the domain is defined by the CreateDistributed function.
        In certain situations it is not desired to have a uniform distribution of the points within the given interval (LB, UB)
        In these cases, a non-uniform grid can be specified by using the Points property od daeDomain.
        A good candidates for the non-uniform grid are cases where we have a very stiff fronts at one side of the domain.
        In these cases it is desirable to place more points at that part od the domain.
        Here, we first print the points before changing them and then set the new values.
    */
		vector<real_t> darrPoints;
		double points[n+1] = {0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.070, 0.100};
		
		m.y.GetPoints(darrPoints);
        GetLog()->Message("  Before:" + toString(darrPoints), 0);
        
        for(size_t i = 0; i < m.y.GetNumberOfPoints(); i++)
			darrPoints[i] = points[i];
		m.y.SetPoints(darrPoints);
        GetLog()->Message("  After:" + toString(darrPoints), 0);

        m.rho.SetValue(8960);
        m.c_p.SetValue(385);
        m.k.SetValue(401);

        m.Q_b.SetValue(1e6);
        m.Q_t.SetValue(0);
    }

    void SetUpVariables(void)
    {
        for(size_t x = 1; x < m.x.GetNumberOfPoints()-1; x++)
            for(size_t y = 1; y < m.y.GetNumberOfPoints()-1; y++)
                m.T.SetInitialCondition(x, y, 300);
    }
};

void runTutorial3(void)
{ 
    boost::scoped_ptr<daeSimulation_t>      pSimulation(new simTutorial3);  
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

