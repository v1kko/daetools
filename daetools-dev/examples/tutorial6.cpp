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
This tutorial introduces the following concepts:

- Ports
- Port connections
- Units (instances of other models)

A simple port type 'portSimple' is defined which contains only one variable 't'.
Two models 'modPortIn' and 'modPortOut' are defined, each having one port of type 'portSimple'.
The wrapper model 'modTutorial' instantiate these two models as its units and connects them
by connecting their ports.
*/

#include <daetools.h>
using namespace daetools::logging;
using namespace daetools::core;
using namespace daetools::solver;
using namespace daetools::datareporting;
using namespace daetools::activity;
namespace vt = daetools::core::variable_types;

using units_pool::s;

/*
Ports, like models, consist of domains, parameters and variables. Parameters and variables
can be distributed as well. Here we define a very simple port, with only one variable.
The process of defining ports is analogous to defining models. Domains, parameters and
variables are declared in the constructor and their constructor accepts ports as
the 'Parent' argument.
*/
class portSimple : public daePort
{
public:
    daeVariable t;

    portSimple(std::string strName, daeePortType portType, daeModel* parent, std::string strDescription = "")
      : daePort(strName, portType, parent, strDescription),
        t("t", vt::no_t, this, "Time elapsed in the process, s")
    {
    }
};

// Here we define two models, 'modPortIn' and 'modPortOut' each having one port of type portSimple.
// The model 'modPortIn' contains inlet port Pin while the model 'modPortOut' contains outlet port Pout.
class modPortIn : public daeModel
{
public:
    portSimple P_in;

    modPortIn(std::string strName, daeModel* pParent = NULL, std::string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        P_in("P_in", eInletPort, this, "The simple port")
    {
    }

    void DeclareEquations(void)
    {
    }
};

class modPortOut : public daeModel
{
public:
    daeVariable Time;
    portSimple P_out;

    modPortOut(std::string strName, daeModel* pParent = NULL, std::string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        Time("Time", vt::no_t, this, "Time elapsed in the process, s"),
        P_out("P_out", eOutletPort, this, "The simple port")
    {
    }

    void DeclareEquations(void)
    {
        daeEquation* eq;

        eq = CreateEquation("time", "Differential equation to calculate the time elapsed in the process.");
        eq->SetResidual(Time.dt() - Constant(1 * (1/s)));

        eq = CreateEquation("Port_t", "");
        eq->SetResidual(P_out.t() - Time());
    }
};

// Model 'modTutorial' declares two units Port_In of type 'modPortIn' and 'Port_Out' of type 'modPortOut'.
class modTutorial6 : public daeModel
{
public:
    modPortIn Port_In;
    modPortOut Port_Out;

    modTutorial6(std::string strName, daeModel* pParent = NULL, std::string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        Port_In("Port_In", this, ""),
        Port_Out("Port_Out", this, "")
    {
    }

    void DeclareEquations(void)
    {
        daeModel::DeclareEquations();
        
        // Ports can be connected by using the function ConnectPorts from daeModel class. Apparently,
        // ports dont have to be of the same type but must contain the same number of parameters and variables.
        ConnectPorts(&Port_Out.P_out, &Port_In.P_in);
    }
};

class simTutorial6 : public daeSimulation
{
public:
    modTutorial6 model;
    
public:
    simTutorial6(void) : model("cdaeTutorial6")
    {
        SetModel(&model);
    }

public:
    void SetUpParametersAndDomains(void)
    {
    }

    void SetUpVariables(void)
    {
        model.Port_Out.Time.SetInitialCondition(0);
    }
};

int main(int argc, char *argv[])
{ 
    std::unique_ptr<daeSimulation_t>    pSimulation  (new simTutorial6);
    std::unique_ptr<daeDataReporter_t>  pDataReporter(daeCreateTCPIPDataReporter());
    std::unique_ptr<daeDAESolver_t>     pDAESolver   (daeCreateIDASolver());
    std::unique_ptr<daeLog_t>           pLog         (daeCreateStdOutLog());

    pDataReporter->Connect("", "cdae_tutorial6-" + daetools::getFormattedDateTime());

    pSimulation->SetReportingInterval(10);
    pSimulation->SetTimeHorizon(100);
    pSimulation->GetModel()->SetReportingOn(true);
    
    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
    pSimulation->SolveInitial();
    pSimulation->Run();
    pSimulation->Finalize();
} 
