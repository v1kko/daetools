#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <boost/python.hpp>

#include "simulation_loader.h"
#include "simulation_loader_c.h"
#include "../dae.h"

static std::string g_strLastError;

static std::string getPythonTraceback()
{
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    boost::python::handle<> hType(ptype);
    boost::python::object extype(hType);
    boost::python::handle<> hTraceback(ptraceback);
    boost::python::object traceback(hTraceback);

    //Extract error message
    std::string strErrorMessage = boost::python::extract<std::string>(pvalue);
    std::string strTraceback    = boost::python::extract<std::string>(ptraceback);

    return strTraceback;
}

const char* GetLastError()
{
    return g_strLastError.c_str();
}

void* LoadSimulation(const char*  strPythonFile, const char* strSimulationClass)
{
    try
    {
        daeSimulationLoader* loader = new daeSimulationLoader();
        loader->LoadSimulation(strPythonFile, strSimulationClass);
        return loader;
    }
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}

bool GetStrippedName(const char strSource[512], char strDestination[512])
{
    try
    {
        std::string strStripped = daeSimulationLoader::GetStrippedName(strSource);
        strncpy(strDestination, strStripped.c_str(), 512);
        return true;
    }
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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

        ptr_loader->IntegrateForTimeInterval(timeInterval, true);
        return true;
    }
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
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
    catch(boost::python::error_already_set const &)
    {
        g_strLastError = getPythonTraceback();
        PyErr_Print();
    }
    catch(std::exception& e)
    {
        g_strLastError = e.what();
        std::cout << e.what() << std::endl;
    }
    return false;
}
