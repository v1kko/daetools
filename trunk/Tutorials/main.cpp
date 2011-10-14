//#include "cdae_whats_the_time.h"
#include "cdae_tutorial1.h"
//#include "cdae_tutorial2.h"
//#include "cdae_tutorial3.h"
//#include "cdae_tutorial4.h"
//#include "cdae_tutorial5.h"
//#include "cdae_tutorial6.h"

int main(int argc, char *argv[])
{ 
	try
	{
//		runWhatsTheTime();
		runTutorial1();
//		runTutorial2();
//		runTutorial3();
//		runTutorial4();
//		runTutorial5();
//		runTutorial6();
	}
	catch(std::exception& e)
	{ 
	 	std::cout << e.what() << std::endl;
	}
} 

