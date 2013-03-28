#include "dealii_poisson.h"
#include "../variable_types.h"
namespace vt = variable_types;

void run_dealii_poisson_test();

class modTutorial1 : public daeModel
{
daeDeclareDynamicClass(modTutorial1)
public:
    daeDomain                       xyz;
    daeVariable                     T;
    dae::fe::dae_dealII_Poisson<2>  fem;

    modTutorial1(string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        xyz("fem", this, unit(),   "FEM domain"),
        T("T",   vt::no_t, this, "Temperature of the plate, -", &xyz),
        fem(*this, xyz, T, 1)
    {
        fem.Initialize();
    }

    void DeclareEquations(void)
    {
        fem.DeclareEquations();
    }
};

class simTutorial1 : public daeSimulation
{
public:
	modTutorial1 M;
	
public:
	simTutorial1() : M("dealii4")
	{
		SetModel(&M);
        M.SetDescription("dealii description");
	}

public:
	void SetUpParametersAndDomains(void)
	{
		M.fem.SetUpParametersAndDomains();
	}

	void SetUpVariables(void)
	{
	}
};

void run_dealii_poisson_test()
{
    try
    {
        boost::scoped_ptr<daeSimulation_t>  	pSimulation(new simTutorial1());  
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
    
        pSimulation->SetReportingInterval(0);
        pSimulation->SetTimeHorizon(0);
        pSimulation->GetModel()->SetReportingOn(true);
        
        pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
        
        pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
        //pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
      
        pSimulation->SolveInitial();
        //pSimulation->Run();
        pSimulation->Finalize();
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}
