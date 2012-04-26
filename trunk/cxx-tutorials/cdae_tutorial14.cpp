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
*/

#include "../dae_develop.h"
#include "../variable_types.h"

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
	// Arguments can be variables, parameters
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
		
	// If derivative part is non-zero then we have to calculate a derivative for that argument
		double m_der  = m.getDerivative();
		double cp_der = m.getDerivative();
		double dT_der = m.getDerivative();
		
		std::cout << (boost::format("m = %1%, cp = %2%, dT = %3%") % m % cp % dT).str() << std::endl;
		std::cout << (boost::format("mcpdT = %1%") % (m*cp*dT)).str() << std::endl;
		
	// The result is 'adouble' object that contain both value and derivative
        return m * cp * dT;
	}
};

class modTutorial14 : public daeModel
{
daeDeclareDynamicClass(modTutorial14)
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
	boost::shared_ptr<extfnPower> P;

    modTutorial14(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        mass("m", kg, this, "Mass of the copper plate"),
        c_p("c_p", J/(kg*K), this, "Specific heat capacity of the plate"),
        alpha("&alpha;", W/((m^2)*K), this, "Heat transfer coefficient"),
        A("A", m^2, this, "Area of the plate"),
        T_surr("T_surr", K, this, "Temperature of the surroundings"),
        Q_in("Q_in", power_t, this, "Power of the heater"),
        T("T", temperature_t, this, "Temperature of the plate"),
		
		Power("Power", power_t, this, "External function power"),
		Power_ext("Power_ext", power_t, this, "External function power")
    {
    }

    void DeclareEquations(void)
    {
        daeEquation* eq;

        eq = CreateEquation("HeatBalance", "Integral heat balance equation");
        eq->SetResidual( mass() * c_p() * T.dt() - Q_in() + alpha() * A() * (T() - T_surr()) );

		P = boost::shared_ptr<extfnPower>(new extfnPower("Power", this, W, mass(), c_p(), T.dt()));
		
        eq = CreateEquation("Power", "");
        eq->SetResidual( Power() - mass() * c_p() * T.dt() );

        eq = CreateEquation("Power_ExternalFunction", "");
        eq->SetResidual( Power_ext() - (*P)() );
		
		std::vector< std::pair<daeVariableWrapper, adouble> >	arrSetVariables;
	    std::vector< std::pair<daeEventPort*, adouble> >		arrTriggerEvents;
	    std::vector<daeAction*>									ptrarrUserDefinedOnEventActions;

		stnRegulator = STN("Regulator");
        STATE("Heating");
            eq = CreateEquation("Q_in", "The heater is on");
            eq->SetResidual(Q_in() - Constant(1500*W));

            ON_CONDITION(T()    > Constant(340*K), "Cooling",   arrSetVariables, arrTriggerEvents, ptrarrUserDefinedOnEventActions);
			ON_CONDITION(Time() > Constant(350*s), "HeaterOff", arrSetVariables, arrTriggerEvents, ptrarrUserDefinedOnEventActions);

        STATE("Cooling");
            eq = CreateEquation("Q_in", "The heater is off");
            eq->SetResidual(Q_in());

            SWITCH_TO("Heating",   Constant(320*K) > T());
            SWITCH_TO("HeaterOff", Constant(350*s) < Time());

        STATE("HeaterOff");
            eq = CreateEquation("Q_in", "The heater is off");
            eq->SetResidual(Q_in());

		END_STN();
    }
};

class simTutorial14 : public daeSimulation
{
public:
    modTutorial14 M;
    
public:
    simTutorial14(void) : M("cdaeTutorial14")
    {
        SetModel(&M);
        M.SetDescription("");
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
        M.stnRegulator->SetActiveState2("Heating");

        M.T.SetInitialCondition(283 * K);
    }
};

int main(int argc, char *argv[])
{ 
    boost::scoped_ptr<daeSimulation_t>      pSimulation(new simTutorial14);  
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
