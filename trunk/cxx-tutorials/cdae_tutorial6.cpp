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
This is the simple port demo.
Here we introduce:
 - Ports
 - Port connections
 - Units (instances of other models)

A simple port type 'portSimple' is defined which contains only one variable 't'.
Two models 'modPortIn' and 'modPortOut' are defined, each having one port of type 'portSimple'.
The wrapper model 'modTutorial' instantiate these two models as its units and connects them
by connecting their ports.
*/

#include "../dae_develop.h"
#include "../variable_types.h"

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
daeDeclareDynamicClass(portSimple)
public:
    daeVariable t;

    portSimple(string strName, daeePortType portType, daeModel* parent, string strDescription = "")
      : daePort(strName, portType, parent, strDescription),
        t("t", no_t, this, "Time elapsed in the process, s")
    {
    }
};

// Here we define two models, 'modPortIn' and 'modPortOut' each having one port of type portSimple.
// The model 'modPortIn' contains inlet port Pin while the model 'modPortOut' contains outlet port Pout.
class modPortIn : public daeModel
{
daeDeclareDynamicClass(modPortIn)
public:
    portSimple P_in;

    modPortIn(string strName, daeModel* pParent = NULL, string strDescription = "") 
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
daeDeclareDynamicClass(modPortOut)
public:
    daeVariable Time;
    portSimple P_out;

    modPortOut(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        Time("Time", no_t, this, "Time elapsed in the process, s"),
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
daeDeclareDynamicClass(modTutorial6)
public:
    modPortIn Port_In;
    modPortOut Port_Out;

    modTutorial6(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        Port_In("Port_In", this, ""),
        Port_Out("Port_Out", this, "")
    {
    }

    void DeclareEquations(void)
    {
       // Ports can be connected by using the function ConnectPorts from daeModel class. Apparently,
       // ports dont have to be of the same type but must contain the same number of parameters and variables.
       ConnectPorts(&Port_Out.P_out, &Port_In.P_in);
    }
};

class simTutorial6 : public daeSimulation
{
public:
    modTutorial6 M;
    
public:
    simTutorial6(void) : M("cdaeTutorial6")
    {
        SetModel(&M);
        M.SetDescription("This tutorial explains how to define and connect ports. \n"
                         "A simple port type 'portSimple' is defined which contains only one variable 't'. "
                         "Two models 'modPortIn' and 'modPortOut' are defined, each having one port of type 'portSimple'. "
                         "The wrapper model 'modTutorial' instantiate these two models as its units and connects them "
                         "by connecting their ports.");
    }

public:
    void SetUpParametersAndDomains(void)
    {
    }

    void SetUpVariables(void)
    {
        M.Port_Out.Time.SetInitialCondition(0);
    }
};

void Export(daeModel* model, std::vector<daeExportable_t*>& objects_to_export)
{
	string pydae_model = model->ExportObjects(objects_to_export, ePYDAE);
	string cdae_model  = model->ExportObjects(objects_to_export, eCDAE);
	
	string strFileName = model->GetName() + "_export.py";
	std::ofstream file_pydae(strFileName.c_str());
	strFileName = model->GetName() + "_export.h";
	std::ofstream file_cdae(strFileName.c_str());
	
	file_pydae << pydae_model;
	file_cdae << cdae_model;
	file_pydae.close();
	file_cdae.close();
}

int main(int argc, char *argv[])
{ 
    boost::scoped_ptr<daeSimulation_t>      pSimulation(new simTutorial6);  
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
    pSimulation->SetTimeHorizon(100);
    pSimulation->GetModel()->SetReportingOn(true);
    
    pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
    pSimulation->SolveInitial();
    
    pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
    pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
    
	// Export models and ports
	simTutorial6* simulation = (simTutorial6*)pSimulation.get();
	std::vector<daeExportable_t*> objects_to_export;
    objects_to_export.push_back(&simulation->M.Port_In.P_in);
	objects_to_export.push_back(&simulation->M.Port_In);
	objects_to_export.push_back(&simulation->M.Port_Out);
	objects_to_export.push_back(&simulation->M);
	Export(&simulation->M, objects_to_export);
  
    pSimulation->Run();
    pSimulation->Finalize();
} 
