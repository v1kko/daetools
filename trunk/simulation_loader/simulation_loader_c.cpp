#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <boost/python.hpp>

#include "simulation_loader_c.h"
#include "simulation_loader.h"
#include "../dae.h"

/* Common functions */
static std::string g_strLastError;

const char* GetLastPythonError()
{
    return g_strLastError.c_str();
}

bool GetStrippedName(const char strSource[512], char strDestination[512])
{
    try
    {
        std::string strStripped = daeSimulationLoader::GetStrippedName(strSource);
        strncpy(strDestination, strStripped.c_str(), 512);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

/* SimulationLoader c-interface */
void* LoadSimulation(const char*  strPythonFile, const char* strSimulationCallable, const char* strArguments)
{
    try
    {
        daeSimulationLoader* loader = new daeSimulationLoader();
        loader->LoadSimulation(strPythonFile, strSimulationCallable, strArguments);
        return loader;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return NULL;
}

bool Initialize(void* s,
                const char* strDAESolver,
                const char* strLASolver,
                const char* strDataReporter,
                const char* strDataReporterConnectionString,
                const char* strLog,
                bool bCalculateSensitivities)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->Initialize(std::string(strDAESolver),
                               std::string(strLASolver),
                               std::string(strDataReporter),
                               std::string(strDataReporterConnectionString),
                               std::string(strLog),
                               bCalculateSensitivities);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool InitializeJSON(void* s, const char* strJSONRuntimeSettings)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->Initialize(strJSONRuntimeSettings);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool SetTimeHorizon(void* s, double timeHorizon)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->SetTimeHorizon(timeHorizon);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool SetReportingInterval(void* s, double reportingInterval)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->SetReportingInterval(reportingInterval);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool SolveInitial(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->SolveInitial();
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool Run(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->Run();
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool Reinitialize(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->Reinitialize();
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool ResetToInitialSystem(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->ResetToInitialSystem();
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool Finalize(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->Finalize();
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

unsigned int GetNumberOfParameters(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return 0;
        }

        return ptr_loader->GetNumberOfParameters();
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return 0;
}

// unsigned int GetNumberOfDOFs(void* s)
// {
//     daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
//     if(ptr_loader)
//         return ptr_loader->GetNumberOfDOFs();
//     return 0;
// }

unsigned int GetNumberOfInputs(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return 0;
        }
        return ptr_loader->GetNumberOfInputs();
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return 0;
}

unsigned int GetNumberOfOutputs(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return 0;
        }
        return ptr_loader->GetNumberOfOutputs();
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return 0;
}

unsigned int GetNumberOfStateTransitionNetworks(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return 0;
        }
        return ptr_loader->GetNumberOfStateTransitionNetworks();
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return 0;
}

bool GetParameterInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        std::string name;
        ptr_loader->GetParameterInfo(index, name, numberOfPoints);
        strncpy(strName, name.c_str(), 512);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

// bool GetDOFInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints)
// {
//     daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
//     if(ptr_loader)
//     {
//         std::string name;
//         ptr_loader->GetDOFInfo(index, name, *numberOfPoints);
//         strncpy(strName, name.c_str(), 512);
//     }
// }

bool GetInputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        std::string name;
        ptr_loader->GetInputInfo(index, name, numberOfPoints);
        strncpy(strName, name.c_str(), 512);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool GetOutputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        std::string name;
        ptr_loader->GetOutputInfo(index, name, numberOfPoints);
        strncpy(strName, name.c_str(), 512);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool GetStateTransitionNetworkInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfStates)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        std::string name;
        ptr_loader->GetStateTransitionNetworkInfo(index, name, numberOfStates);
        strncpy(strName, name.c_str(), 512);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}


bool GetParameterValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->GetParameterValue(index, value, numberOfPoints);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

// bool GetDOFValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
// {
//     daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
//     if(ptr_loader)
//         ptr_loader->GetDOFValue(index, value, numberOfPoints);
// }

bool GetInputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->GetInputValue(index, value, numberOfPoints);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool GetOutputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->GetOutputValue(index, value, numberOfPoints);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool GetActiveState(void* s, unsigned int index, char strActiveState[64])
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        std::string state;
        ptr_loader->GetActiveState(index, state);
        strncpy(strActiveState, state.c_str(), 64);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool SetParameterValue(void* s, unsigned int index, const double* value, unsigned int numberOfPoints)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->SetParameterValue(index, value, numberOfPoints);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

// bool SetDOFValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
// {
//     daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
//     if(ptr_loader)
//         ptr_loader->SetDOFValue(index, value, numberOfPoints);
// }

bool SetInputValue(void* s, unsigned int index, const double* value, unsigned int numberOfPoints)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->SetInputValue(index, value, numberOfPoints);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool SetOutputValue(void* s, unsigned int index, const double* value, unsigned int numberOfPoints)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->SetOutputValue(index, value, numberOfPoints);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool GetFMIValue(void* s, unsigned int fmi_reference, double* value)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        *value = ptr_loader->GetFMIValue(fmi_reference);

        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool GetFMIActiveState(void* s, unsigned int fmi_reference, char* state)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        std::string str_state = ptr_loader->GetFMIActiveState(fmi_reference);
        strncpy(state, str_state.c_str(), 512);

        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool SetFMIValue(void* s, unsigned int fmi_reference, double value)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->SetFMIValue(fmi_reference, value);

        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool SetFMIActiveState(void* s, unsigned int fmi_reference, const char* state)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        std::string str_state = state;
        ptr_loader->SetFMIActiveState(fmi_reference, str_state);

        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool SetActiveState(void* s, unsigned int index, char strActiveState[64])
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        std::string state = strActiveState;
        ptr_loader->SetActiveState(index, state);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool Integrate(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->Integrate(false, true);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool IntegrateForTimeInterval(void* s, double timeInterval)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->IntegrateForTimeInterval(timeInterval, false, true);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool IntegrateUntilTime(void* s, double time)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->IntegrateUntilTime(time, false, true);
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool ReportData(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        ptr_loader->ReportData();
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool FreeSimulation(void* s)
{
    try
    {
        daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
        if(!ptr_loader)
        {
            g_strLastError = "Invalid simulation pointer (has the simulation been loaded?)";
            return false;
        }

        delete ptr_loader;
        ptr_loader = NULL;
        return true;
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}
