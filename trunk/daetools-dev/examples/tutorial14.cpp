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

using units_pool::m;
using units_pool::kg;
using units_pool::K;
using units_pool::J;
using units_pool::W;
using units_pool::s;

class extfnPower : public daeScalarExternalFunction
{
public:
    extfnPower(const std::string& strName, daeModel* pModel, const unit& units, adouble m, adouble cp, adouble dT) 
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
        /* Calculate function is used to calculate a value and a derivative (if requested)
         * of the external function per given argument. Here the simple function is given by:
         *    f(m, cp, dT/dt) = m * cp * dT/dt
         *
         * Procedure:
         * 1. Get the arguments from the dictionary values: {'arg-name' : adouble-object}.
         *    Every adouble object has two properties: Value and Derivative that can be
         *    used to evaluate function or its partial derivatives per its arguments
         *    (partial derivatives are used to fill in a Jacobian matrix necessary to solve
         *    a system of non-linear equations using the Newton method). */
        adouble m_     = boost::get<adouble>(mapValues["m"]);
        adouble cp_    = boost::get<adouble>(mapValues["cp"]);
        adouble dT_dt_ = boost::get<adouble>(mapValues["dT"]);
		
		double m     = m_.getValue();
		double cp    = cp_.getValue();
		double dT_dt = dT_dt_.getValue();

		double m_der     = m_.getDerivative();
		double cp_der    = cp_.getDerivative();
		double dT_dt_der = dT_dt_.getDerivative();

        /* 2. Always calculate the value of a function. */
        double value = m * cp * dT_dt;
        
        /* 3. If a function derivative per one of its arguments is requested,
         *    a derivative part of that argument will be non-zero.
         *    In that case, investigate which derivative is requested and calculate it
         *    using the chain rule: f'(x) = x' * df(x)/dx */
        double derivative = 0.0;
        
        if(m_der != 0)        // A derivative per 'm' was requested
            derivative = m_der * (cp * dT_dt);
        else if(cp_der != 0)  // A derivative per 'cp' was requested
            derivative = cp_der * (m * dT_dt);
        else if(dT_dt_der != 0) // A derivative per 'dT_dt' was requested
            derivative = dT_dt_der * (m * cp);
		
		//std::cout << (boost::format("m = %1%, cp = %2%, dT = %3%") % m % cp % dT).str() << std::endl;
		//std::cout << (boost::format("mcpdT = %1%") % (m*cp*dT)).str() << std::endl;
		
        // The result is 'adouble' object that contain both value and derivative
        return adouble(value, derivative);
	}
};

class modTutorial14 : public daeModel
{
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
    daeVariable Power_ext;
	std::shared_ptr<extfnPower> P;

    modTutorial14(std::string strName, daeModel* pParent = NULL, std::string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        mass("m", kg, this, "Mass of the copper plate"),
        c_p("c_p", J/(kg*K), this, "Specific heat capacity of the plate"),
        alpha("&alpha;", W/((m^2)*K), this, "Heat transfer coefficient"),
        A("A", m^2, this, "Area of the plate"),
        T_surr("T_surr", K, this, "Temperature of the surroundings"),
        Q_in("Q_in", vt::power_t, this, "Power of the heater"),
        T("T", vt::temperature_t, this, "Temperature of the plate"),
		
		Power("Power", vt::power_t, this, "External function power"),
		Power_ext("Power_ext", vt::power_t, this, "External function power")
    {
    }

    void DeclareEquations(void)
    {
        daeEquation* eq;

        eq = CreateEquation("HeatBalance", "Integral heat balance equation");
        eq->SetResidual( mass() * c_p() * T.dt() - Q_in() + alpha() * A() * (T() - T_surr()) );

		P = std::shared_ptr<extfnPower>(new extfnPower("Power", this, W, mass(), c_p(), T.dt()));
		
        eq = CreateEquation("Power", "");
        eq->SetResidual( Power() - mass() * c_p() * T.dt() );

        eq = CreateEquation("Power_ExternalFunction", "");
        eq->SetResidual( Power_ext() - (*P)() );
		
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

class simTutorial14 : public daeSimulation
{
public:
    modTutorial14 model;
    
public:
    simTutorial14(void) : model("cdaeTutorial14")
    {
        SetModel(&model);
        model.SetDescription("");
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
        model.stnRegulator->SetActiveState("Heating");

        model.T.SetInitialCondition(283 * K);
    }
};

int main(int argc, char *argv[])
{ 
    std::unique_ptr<daeSimulation_t>    pSimulation  (new simTutorial14);
    std::unique_ptr<daeDataReporter_t>  pDataReporter(daeCreateTCPIPDataReporter());
    std::unique_ptr<daeDAESolver_t>     pDAESolver   (daeCreateIDASolver());
    std::unique_ptr<daeLog_t>           pLog         (daeCreateStdOutLog());

    pDataReporter->Connect("", "cdae_tutorial14-" + daetools::getFormattedDateTime());

    pSimulation->SetReportingInterval(0.5);
    pSimulation->SetTimeHorizon(500);
    pSimulation->GetModel()->SetReportingOn(true);
    
    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
    pSimulation->SolveInitial();
    pSimulation->Run();
    pSimulation->Finalize();
} 
