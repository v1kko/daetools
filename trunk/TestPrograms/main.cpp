#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <ctime>
#include "../Examples/test_models.h"
#include "../DataReporters/datareporters.h"
#include "../Solver/ida_solver.h"
 
int main(int argc, char *argv[])
{ 
	try
	{
		boost::scoped_ptr<daeDynamicSimulation_t>		pSimulation(new simTutorial3);  
		boost::scoped_ptr<daeDataReporter_t>			pDataReporter(daeCreateTCPIPDataReporter());
		boost::scoped_ptr<daeIDASolver>					pDAESolver(new daeIDASolver()); //daeCreateIDASolver());
		boost::scoped_ptr<daeLog_t>						pLog(daeCreateStdOutLog());
 
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
    
		daeConfig& cfg = daeConfig::GetConfig();
		std::cout << cfg.Get<string>("daetools.version") << std::endl;
		std::cout << cfg.Get<real_t>("daetools.core.eventTolerance") << std::endl;
		std::cout << cfg.Get<real_t>("daetools.activity.timeHorizon") << std::endl;
		std::cout << cfg.Get<real_t>("daetools.activity.reportingInterval") << std::endl;
		
        pSimulation->SetReportingInterval(10);
        pSimulation->SetTimeHorizon(100);
		pSimulation->GetModel()->SetReportingOn(true);
         
 		pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
		pSimulation->SolveInitial();
 		pSimulation->GetModel()->SaveModelReport("simTest.xml");
 		pSimulation->GetModel()->SaveRuntimeModelReport("simTest-rt.xml");
      
 		pSimulation->Run();
	}
	catch(std::exception& e)
	{ 
	 	std::cout << e.what() << std::endl;
		return -1;
	}
} 

