#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>

#define FMI2_Export
#include "../fmi/include/fmi2Functions.h"

/* Test FMU */
void logger(fmi2ComponentEnvironment env, fmi2String instanceName, fmi2Status status, fmi2String category, fmi2String message, ...)
{
    std::string strStatus;
    if(status == fmi2OK)
        strStatus = "fmi2OK";
    else if(status == fmi2Warning)
        strStatus = "fmi2Warning";
    else if(status == fmi2Discard)
        strStatus = "fmi2Discard";
    else if(status == fmi2Error)
        strStatus = "fmi2Error";
    else if(status == fmi2Fatal)
        strStatus = "fmi2Fatal";
    else
       strStatus = "fmi2Pending";

    std::cout << "[" << (const char*)env << "] log message from \"" << instanceName << "\" "
              << "(status: \"" << strStatus << "\", logCategory: \"" << category << "\"):" << std::endl;
    std::cout << message << std::endl;
}

int main(int argc, char *argv[])
{
    fmi2Status status;

    fmi2CallbackFunctions functions = {logger, NULL, NULL, NULL, (const fmi2ComponentEnvironment)"testFMU"};

    fmi2Component comp = fmi2Instantiate("tutorial20",
                                         fmi2CoSimulation,
                                         "6f6dd048-7eff-11e4-bf92-9cb70d5dfdfc",
                                         "/tmp/daetools-fmu-0qoln2fk/resources",
                                         &functions,
                                         true, /* visible */
                                         true  /* logging */);

    if(comp == NULL)
    {
        std::cout << "Cannot load tutorial20.fmu" << std::endl;
        return 66;
    }

    /* tutorial20.py FMI references:
         0 - parameter p1
         1 - parameter p2
         2 - stn Multipliers ['variableMultipliers', 'constantMultipliers']
         3 - input in_1.y
         4 - input in_2.y
         5 - output out_1.y
         6 - output out_2.y
       There are variables m1 and m2[] but they are internal and therefore not exported.
    */

    /* Setup experiment */
    status = fmi2SetupExperiment(comp, false, 1e-5, 0.0, true, 100.0);
    if(status != fmi2OK)
        return 1;

    /* Initialization phase */
    status = fmi2EnterInitializationMode(comp);
    if(status != fmi2OK)
        return 2;

    /* Set Real parameters/inputs values */
    {
        unsigned int nvr  = 4;
        unsigned int vr[] = {0, 1, 3, 4};
        double value[]    = {10, 20, 1, 2};
        status = fmi2SetReal(comp, vr, nvr, value);
        if(status != fmi2OK)
            return 3;
    }
    /* Set String inputs (STNs active states) */
    {
        unsigned int nvr  = 1;
        unsigned int vr[] = {2};
        fmi2String state[1] = {"variableMultipliers"};

        status = fmi2SetString(comp, vr, nvr, state);
        if(status != fmi2OK)
            return 3;
    }

    status = fmi2ExitInitializationMode(comp);
    if(status != fmi2OK)
        return 4;

    /* Integration from 0.0 to 100.0 */
    double current_time = 0;
    double step = 10;
    while(current_time < 100)
    {
        std::cout << "Integrating from [" << current_time << "] to [" << current_time+step << "]..." << std::endl;
        status = fmi2DoStep(comp, current_time, step, true);
        if(status != fmi2OK)
            return 5;

        /* Get Real values */
        {
            unsigned int nvr  = 6;
            unsigned int vr[] = {0, 1, 3, 4, 5, 6};
            double value[]    = {0, 0, 0, 0, 0, 0};

            status = fmi2GetReal(comp, vr, nvr, value);
            if(status != fmi2OK)
                return 6;

            for(size_t i = 0; i < nvr; i++)
                std::cout << "  v[" << vr[i] << "] = " << value[i] << std::endl;
        }
        /* Get String values (STNs active states) */
        {
            unsigned int nvr  = 1;
            unsigned int vr[] = {2};
            fmi2String state[1]; // allocate an array of {const char*} pointers (all NULL, will be set by fmi2GetString)

            status = fmi2GetString(comp, vr, nvr, state);
            if(status != fmi2OK)
                return 6;

            for(size_t i = 0; i < nvr; i++)
                std::cout << "  stn[" << vr[i] << "] = " << state[i] << std::endl;
        }

        current_time += step;
    }

    /* Terminate simulation */
    status = fmi2Terminate(comp);
    if(status != fmi2OK)
        return 7;

    /* Free FMU object */
    fmi2FreeInstance(comp);

    std::cout << "Simulation has been completed successfuly" << std::endl;
}

/* Test simulation loader
#include "../simulation_loader/simulation_loader.h"

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
