#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <ctime>
#include "../Examples/test_models.h"
#include "../DataReporters/datareporters.h"
#include "../Solver/ida_solver.h"
#include "../Simulation/dyn_simulation.h"
#include "../NLPSolver/nlpsolver.h"

int main(int argc, char *argv[])
{ 
	try
	{
		boost::scoped_ptr<daeSimulation_t>		pSimulation(new simToy);  
		boost::scoped_ptr<daeDataReporter_t>	pDataReporter(daeCreateTCPIPDataReporter());
		boost::scoped_ptr<daeIDASolver>			pDAESolver(new daeIDASolver()); //daeCreateIDASolver());
		boost::scoped_ptr<daeLog_t>				pLog(daeCreateStdOutLog());
		boost::scoped_ptr<daeOptimization_t>	pOptimization(new daeOptimization());
		boost::scoped_ptr<daeNLPSolver_t>	    pNLPSolver(new daeIPOPTSolver());
 
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
    
//		daeConfig& cfg = daeConfig::GetConfig();
//		std::cout << cfg.Get<string>("daetools.version") << std::endl;
//		std::cout << cfg.Get<real_t>("daetools.core.eventTolerance") << std::endl;
//		std::cout << cfg.Get<real_t>("daetools.activity.timeHorizon") << std::endl;
//		std::cout << cfg.Get<real_t>("daetools.activity.reportingInterval") << std::endl;
		
		pDAESolver->SetRelativeTolerance(1e-6);
        pSimulation->SetReportingInterval(1);
        pSimulation->SetTimeHorizon(2);
		pSimulation->GetModel()->SetReportingOn(true);
         
 		//pSimulation->InitOptimization(pDAESolver.get(), pDataReporter.get(), pLog.get());
		pSimulation->GetModel()->SaveModelReport("simTest.xml");
		pSimulation->GetModel()->SaveRuntimeModelReport("simTest-rt.xml");

 		pOptimization->Initialize(pSimulation.get(), pNLPSolver.get(), pDAESolver.get(), pDataReporter.get(), pLog.get());
		pSimulation->GetModel()->SaveModelReport("simTest.xml");
		pSimulation->GetModel()->SaveRuntimeModelReport("simTest-rt.xml");
		
 		pOptimization->Run();
		pOptimization->Finalize();

//		pSimulation->InitOptimization(pDAESolver.get(), pDataReporter.get(), pLog.get());
//		pSimulation->SolveInitial();
		
//		pSimulation->GetModel()->SaveModelReport("simTest.xml");
//		pSimulation->GetModel()->SaveRuntimeModelReport("simTest-rt.xml");
	  
//		pSimulation->Run();
//		pSimulation->Finalize();
	}
	catch(std::exception& e)
	{ 
	 	std::cout << e.what() << std::endl;
		return -1;
	}
} 

