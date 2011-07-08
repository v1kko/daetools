#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <ctime>
#include "../Examples/test_models.h"
#include "../DataReporting/datareporters.h"
#include "../IDAS_DAESolver/ida_solver.h"
#include "../Activity/simulation.h"
#define daeIPOPT
#include "../BONMIN_MINLPSolver/base_solvers.h"

int main(int argc, char *argv[])
{ 
	try
	{
		boost::scoped_ptr<daeSimulation_t>		pSimulation(new simHS71);  
		boost::scoped_ptr<daeDataReporter_t>	pDataReporter(daeCreateTCPIPDataReporter());
		boost::scoped_ptr<daeIDASolver>			pDAESolver(new daeIDASolver());
		boost::scoped_ptr<daeLog_t>				pLog(daeCreateStdOutLog());
		boost::scoped_ptr<daeOptimization_t>	pOptimization(new daeOptimization());
		boost::scoped_ptr<daeNLPSolver_t>	    pNLPSolver(daeCreateIPOPTSolver());
 
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
    
        pSimulation->SetReportingInterval(1);
        pSimulation->SetTimeHorizon(200);
		pSimulation->GetModel()->SetReportingOn(true);
		
		bool bRunOptimization = true;
		
		if(bRunOptimization)
		{	
			pOptimization->Initialize(pSimulation.get(), pNLPSolver.get(), pDAESolver.get(), pDataReporter.get(), pLog.get());
			
			pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
			pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
			
			pOptimization->Run();
			pOptimization->Finalize();
		}
		else
		{
			pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
			pSimulation->SolveInitial();
			
			pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
			pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
		  
			pSimulation->Run();
			pSimulation->Finalize();
		}
	}
	catch(std::exception& e)
	{ 
	 	std::cout << e.what() << std::endl;
		return -1;
	}
} 

