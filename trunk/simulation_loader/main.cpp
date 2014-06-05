#include "simulation_loader.h"
#include <string>
#include <fstream>
#include <streambuf>
#include <boost/python.hpp>

int main(int argc, char *argv[])
{
    try
    {
        dae_simulation_loader::daeSimulationLoader loader;
        loader.LoadSimulation("../daetools-package/daetools/examples/tutorial2.py", "simTutorial");

        //loader.Initialize("IDAS", "Sundials LU", "TCPIPDataReporter", "", "StdOutLog", false, "");

        std::ifstream jsonFile("../daetools-package/daetools/examples/tutorial2.json");
        std::string jsonSettings((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
        loader.Initialize(jsonSettings);

        std::cout << std::endl << "NumberOfParameters = " << loader.GetNumberOfParameters() << std::endl;
        for(size_t i = 0; i < loader.GetNumberOfParameters(); i++)
        {
            std::string strName;
            size_t numberOfPoints;
            loader.GetParameterInfo(i, strName, numberOfPoints);
            std::cout << "  parameter [" << i <<  "]: " << strName << " with " << numberOfPoints << " points" << std::endl;
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
