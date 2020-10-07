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
 - Arrays (discrete distribution domains)
 - Distributed parameters
 - Number of degrees of freedom and how to fix it
 - Initial guess of the variables
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

class modTutorial2 : public daeModel
{
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

    modTutorial2(std::string strName, daeModel* pParent = NULL, std::string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
		x("x", this, m,        "X axis domain"),
		y("y", this, m,        "Y axis domain"),
        Nq("Nq", this, unit(), "Number of heat fluxes"),
        
	    a("a", J/(kg*K), this, "Coefficient for calculation of cp"),
        b("b", J/(kg*(K^2)), this, "Coefficient for calculation of cp"),
        Q("Q", W/(m^2), this, "Heat flux array at the edges of the plate (bottom/top)", &Nq),
        k("&lambda;", W/(m*K), this, "Thermal conductivity of the plate", &x, &y),
        c_p("c_p", vt::specific_heat_capacity_t, this, "Specific heat capacity of the plate", &x, &y),
        rho("&rho;", vt::density_t, this, "Density of the plate"),
        T("T", vt::temperature_t, this, "Temperature of the plate", &x, &y)
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
    modTutorial2 model;
    
public:
    simTutorial2(void) : model("cdaeTutorial2")
    {
        SetModel(&model);
    }

public:
    void SetUpParametersAndDomains(void)
    {
        const size_t n = 25;

        model.x.CreateStructuredGrid(n, 0, 0.1);
        model.y.CreateStructuredGrid(n, 0, 0.1);
        
        model.Nq.CreateArray(2);

        model.a.SetValue(367.0 * J/(kg*K));
        model.b.SetValue(0.07  * J/(kg*(K^2)));

        model.Q.SetValue(0, 1e6 * W/(m^2));
        model.Q.SetValue(1, 0.0 * W/(m^2));

        for(size_t x = 0; x < model.x.GetNumberOfPoints(); x++)
            for(size_t y = 0; y < model.y.GetNumberOfPoints(); y++)
                model.k.SetValue(x, y, 401 * W/(m*K));
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
        model.rho.AssignValue(8960 * kg/(m^3));
    
        /*
        To help the DAE solver it is possible to set initial guesses of of the variables.
        Closer the initial guess is to the solution - faster the solver will converge to the solution
        Just for fun, here we will try to obstruct the solver by setting the initial guess which is rather far from the solution.
        Despite that, the solver will successfully initialize the system!
        */
        model.T.SetInitialGuesses(1000 * K);
        for(size_t x = 0; x < model.x.GetNumberOfPoints(); x++)
            for(size_t y = 0; y < model.y.GetNumberOfPoints(); y++)
                model.c_p.SetInitialGuess(x, y, 1000 * J/(kg*K));

        for(size_t x = 1; x < model.x.GetNumberOfPoints()-1; x++)
            for(size_t y = 1; y < model.y.GetNumberOfPoints()-1; y++)
                model.T.SetInitialCondition(x, y, 300 * K);
    }
};

int main(int argc, char *argv[])
{   
    std::unique_ptr<daeSimulation_t>    pSimulation  (new simTutorial2);
    std::unique_ptr<daeDataReporter_t>  pDataReporter(daeCreateTCPIPDataReporter());
    std::unique_ptr<daeDAESolver_t>     pDAESolver   (daeCreateIDASolver());
    std::unique_ptr<daeLog_t>           pLog         (daeCreateStdOutLog());

    pDataReporter->Connect("", "cdae_tutorial2-" + daetools::getFormattedDateTime());

    pSimulation->SetReportingInterval(10);
    pSimulation->SetTimeHorizon(1000);
    pSimulation->GetModel()->SetReportingOn(true);
    
    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
	pSimulation->SolveInitial();
    pSimulation->Run();
    pSimulation->Finalize();
} 

