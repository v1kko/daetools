#include "../simulation_loader/simulation_loader.h"
#include <string>
#include <fstream>
#include <streambuf>
#include <boost/python.hpp>

#define FMI2_Export
#include "../fmi/include/fmi2Functions.h"


int main(int argc, char *argv[])
{
    fmi2Component comp = fmi2Instantiate("tutorial20",
                                         fmi2CoSimulation,
                                         "6f6dd048-7eff-11e4-bf92-9cb70d5dfdfc",
                                         "/tmp/daetools-fmu-QOp8id",
                                         NULL,
                                         0,
                                         1);

    if(comp == NULL)
        std::cout << "comp is NULL" << std::endl;
}
/*
int main(int argc, char *argv[])
{
    try
    {
        daeSimulationLoader loader;
        loader.LoadSimulation("../daetools-package/daetools/examples/tutorial20.py", "simTutorial");

        if(true)
        {
            loader.Initialize("Sundials IDAS", "", "TCPIPDataReporter", "", "StdOutLog", false, "");
        }
        else
        {
            std::ifstream jsonFile("../daetools-package/daetools/examples/tutorial20.json");
            std::string jsonSettings((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
            loader.Initialize(jsonSettings);
        }

        // Has to go after the call to Initialize() for it sets a default time horizon and reporting interval
        loader.SetReportingInterval(5);
        loader.SetTimeHorizon(100);

        std::cout << std::endl << "NumberOfParameters = " << loader.GetNumberOfParameters() << std::endl;
        for(size_t i = 0; i < loader.GetNumberOfParameters(); i++)
        {
            std::string strName;
            unsigned int numberOfPoints;
            loader.GetParameterInfo(i, strName, &numberOfPoints);
            std::cout << "  parameter [" << i <<  "]: " << strName << " with " << numberOfPoints << " points" << std::endl;
        }

        std::cout << std::endl << "Number Of Inputs = " << loader.GetNumberOfInputs() << std::endl;
        for(size_t i = 0; i < loader.GetNumberOfInputs(); i++)
        {
            std::string strName;
            unsigned int numberOfPoints;
            loader.GetInputInfo(i, strName, &numberOfPoints);
            std::cout << "  input [" << i <<  "]: " << strName << " with " << numberOfPoints << " points" << std::endl;
        }

        std::cout << std::endl << "Number Of Outputs = " << loader.GetNumberOfOutputs() << std::endl;
        for(size_t i = 0; i < loader.GetNumberOfOutputs(); i++)
        {
            std::string strName;
            unsigned int numberOfPoints;
            loader.GetOutputInfo(i, strName, &numberOfPoints);
            std::cout << "  output [" << i <<  "]: " << strName << " with " << numberOfPoints << " points" << std::endl;
        }

        loader.ShowSimulationExplorer();
        loader.SolveInitial();
        loader.Run();
        loader.Finalize();
    }
    catch(boost::python::error_already_set const &)
    {
        PyErr_Print();
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}
*/
