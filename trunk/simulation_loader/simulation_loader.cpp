#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#define BOOST_FILESYSTEM_NO_DEPRECATED
#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "simulation_loader.h"
#include "simulation_loader_c.h"
#include "../dae.h"

void Simulate(const char*  strPythonFile, const char* strSimulationClass, bool bShowSimulationExplorer)
{
    try
    {
        daeSimulationLoader loader;

        loader.LoadSimulation(strPythonFile, strSimulationClass);
        loader.Simulate(bShowSimulationExplorer);
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
        PyErr_Print();
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    return NULL;
}

void Initialize(void* s,
                const char* strDAESolver,
                const char* strLASolver,
                const char* strDataReporter,
                const char* strDataReporterConnectionString,
                const char* strLog,
                bool bCalculateSensitivities)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->Initialize(std::string(strDAESolver),
                               std::string(strLASolver),
                               std::string(strDataReporter),
                               std::string(strDataReporterConnectionString),
                               std::string(strLog),
                               bCalculateSensitivities);
    else
        daeDeclareAndThrowException(exInvalidCall);
}

void InitializeJSON(void* s, const char* strJSONRuntimeSettings)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->Initialize(strJSONRuntimeSettings);
}

void SetTimeHorizon(void* s, double timeHorizon)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->SetTimeHorizon(timeHorizon);
}

void SetReportingInterval(void* s, double reportingInterval)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->SetReportingInterval(reportingInterval);
}

void SolveInitial(void* s)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->SolveInitial();
}

void Run(void* s)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->Run();
}

void Reinitialize(void* s)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->Reinitialize();
}

void Finalize(void* s)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->Finalize();
}

unsigned int GetNumberOfParameters(void* s)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        return ptr_loader->GetNumberOfParameters();
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
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        return ptr_loader->GetNumberOfInputs();
    return -1;
}

unsigned int GetNumberOfOutputs(void* s)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        return ptr_loader->GetNumberOfOutputs();
    return -1;
}

void GetParameterInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
    {
        std::string name;
        ptr_loader->GetParameterInfo(index, name, *numberOfPoints);
        strncpy(strName, name.c_str(), 512);
    }
}

// void GetDOFInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints)
// {
//     daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
//     if(ptr_loader)
//     {
//         std::string name;
//         ptr_loader->GetDOFInfo(index, name, *numberOfPoints);
//         strncpy(strName, name.c_str(), 512);
//     }
// }

void GetInputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
    {
        std::string name;
        ptr_loader->GetInputInfo(index, name, *numberOfPoints);
        strncpy(strName, name.c_str(), 512);
    }
}

void GetOutputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
    {
        std::string name;
        ptr_loader->GetOutputInfo(index, name, *numberOfPoints);
        strncpy(strName, name.c_str(), 512);
    }
}

void GetParameterValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->GetParameterValue(index, value, numberOfPoints);
}

// void GetDOFValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
// {
//     daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
//     if(ptr_loader)
//         ptr_loader->GetDOFValue(index, value, numberOfPoints);
// }

void GetInputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->GetInputValue(index, value, numberOfPoints);
}

void GetOutputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->GetOutputValue(index, value, numberOfPoints);
}

void SetParameterValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->SetParameterValue(index, value, numberOfPoints);
}

// void SetDOFValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
// {
//     daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
//     if(ptr_loader)
//         ptr_loader->SetDOFValue(index, value, numberOfPoints);
// }

void SetInputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->SetInputValue(index, value, numberOfPoints);
}

void SetOutputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->SetOutputValue(index, value, numberOfPoints);
}

void IntegrateForTimeInterval(void* s, double timeInterval)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->IntegrateForTimeInterval(timeInterval, true);
}

void IntegrateUntilTime(void* s, double time)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->IntegrateUntilTime(time, false, true);
}

void ReportData(void* s)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        ptr_loader->ReportData();
}

void FreeSimulation(void* s)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)s;
    if(ptr_loader)
        delete ptr_loader;
}

class daeSimulationLoaderData
{
public:
    daeSimulationLoaderData()
    {
        m_pSimulation = NULL;
    }

public:
// Solver creation routines
    void SetupDAESolver(const std::string& strDAESolver);
    void SetupLASolver(const std::string& strLASolver);
    void SetupNLPSolver(const std::string& strNLPSolver);
    void SetupDataReporter(const std::string& strDataReporter, const std::string& strConnectionString);
    void SetupLog(const std::string& strLog);

public:
// Created and owned by Python, thus the raw pointers
    daeSimulation_t*     m_pSimulation;
    daeDataReporter_t*   m_pDataReporter;
    daeDAESolver_t*	     m_pDAESolver;
    daeLASolver_t*       m_pLASolver;
    daeLog_t*	         m_pLog;

// Parameters/Inputs/Outputs
    std::vector<daeParameter_t*> m_ptrarrParameters;
    std::vector<daeVariable_t*>  m_ptrarrDOFs;
    std::vector<daeVariable_t*>  m_ptrarrInputs;
    std::vector<daeVariable_t*>  m_ptrarrOutputs;

// Python related objects
    boost::python::object m_pyMainModule;
    boost::python::object m_pySimulation;
};

daeSimulationLoader::daeSimulationLoader()
{
    m_pData = new daeSimulationLoaderData;

    // Py_Initialize() call oved to dllmain.cpp

    if(!Py_IsInitialized())
        daeDeclareAndThrowException(exInvalidCall);
}

daeSimulationLoader::~daeSimulationLoader()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(pData)
    {
        delete pData;
        m_pData = NULL;
    }

    //Py_Finalize() call moved to dllmain.cpp
}

void daeSimulationLoader::Simulate(bool bShowSimulationExplorer)
{
//    try
//    {
        daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
        if(!pData)
            daeDeclareAndThrowException(exInvalidPointer);

        pData->SetupDAESolver("");
        //pData->SetupLASolver("");
        pData->SetupDataReporter("", "");
        pData->SetupLog("");

        if(!pData->m_pSimulation)
            daeDeclareAndThrowException(exInvalidPointer);
        if(!pData->m_pDataReporter)
            daeDeclareAndThrowException(exInvalidPointer);
        if(!pData->m_pDAESolver)
            daeDeclareAndThrowException(exInvalidPointer);
        if(!pData->m_pLog)
            daeDeclareAndThrowException(exInvalidPointer);

        time_t rawtime;
        struct tm* timeinfo;
        char buffer[80];
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime (buffer, 80, " [%d.%m.%Y %H:%M:%S]", timeinfo);
        string simName = pData->m_pSimulation->GetModel()->GetName() + buffer;
        if(!pData->m_pDataReporter->Connect(string(""), simName))
            daeDeclareAndThrowException(exInvalidCall);

        pData->m_pSimulation->SetReportingInterval(10);
        pData->m_pSimulation->SetTimeHorizon(1000);
        pData->m_pSimulation->GetModel()->SetReportingOn(true);

        pData->m_pSimulation->Initialize(pData->m_pDAESolver, pData->m_pDataReporter, pData->m_pLog);
        pData->m_pSimulation->SolveInitial();

        if(bShowSimulationExplorer)
            ShowSimulationExplorer();

        pData->m_pSimulation->Run();
        pData->m_pSimulation->Finalize();
//    }
//    catch(std::exception& e)
//    {
//        std::cout << e.what() << std::endl;
//    }
}

void daeSimulationLoader::Initialize(const std::string& strJSONRuntimeSettings)
{
//    try
//    {
        daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
        if(!pData)
            daeDeclareAndThrowException(exInvalidPointer);

        boost::python::dict locals;
        boost::python::object main_namespace = pData->m_pyMainModule.attr("__dict__");

        boost::python::exec("from daetools.dae_simulator.auxiliary import InitializeSimulation", main_namespace);
        locals["_json_runtime_settings_"] = strJSONRuntimeSettings;
        std::string command = "InitializeSimulation(__daetools_simulation__, _json_runtime_settings_)";
        boost::python::exec(command.c_str(), main_namespace, locals);

        SetupInputsAndOutputs();
//    }
//    catch(boost::python::error_already_set const &)
//    {
//        PyErr_Print();
//    }
}

void daeSimulationLoader::Initialize(const std::string& strDAESolver,
                                     const std::string& strLASolver,
                                     const std::string& strDataReporter,
                                     const std::string& strDataReporterConnectionString,
                                     const std::string& strLog,
                                     bool bCalculateSensitivities,
                                     const std::string& strJSONRuntimeSettings)
{
//    try
//    {
        daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
        if(!pData)
            daeDeclareAndThrowException(exInvalidPointer);

        pData->SetupDAESolver(strDAESolver);
        pData->SetupLASolver(strLASolver);
        pData->SetupDataReporter(strDataReporter, strDataReporterConnectionString);
        pData->SetupLog(strLog);

        if(!pData->m_pSimulation)
            daeDeclareAndThrowException(exInvalidPointer);
        if(!pData->m_pDataReporter)
            daeDeclareAndThrowException(exInvalidPointer);
        if(!pData->m_pDAESolver)
            daeDeclareAndThrowException(exInvalidPointer);
        if(!pData->m_pLog)
            daeDeclareAndThrowException(exInvalidPointer);

        if(!pData->m_pDataReporter->Connect(strDataReporterConnectionString, pData->m_pSimulation->GetModel()->GetName()))
            daeDeclareAndThrowException(exInvalidCall);

        pData->m_pSimulation->SetReportingInterval(1); // only provisional, must be set by SetReportingInterval
        pData->m_pSimulation->SetTimeHorizon(10);      // only provisional, must be set by SetTimeHorizon
        pData->m_pSimulation->GetModel()->SetReportingOn(true);

        pData->m_pSimulation->Initialize(pData->m_pDAESolver, pData->m_pDataReporter, pData->m_pLog, bCalculateSensitivities, strJSONRuntimeSettings);

        SetupInputsAndOutputs();
//    }
//    catch(std::exception& e)
//    {
//        std::cout << e.what() << std::endl;
//    }
}

void daeSimulationLoader::SetupInputsAndOutputs()
{
    // Simulation must be first initialized
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    // Collect all parameters and ports and initialize parameters/inputs/outputs arrays
    std::vector<daeVariable_t*> ptrarrVariables;
    std::map<std::string, daeParameter_t*> mapParameters;
    std::map<std::string, daeVariable_t*> mapVariables;
    std::vector<daePort_t*> arrPorts;
    daeModel_t* pTopLevelModel = pData->m_pSimulation->GetModel();

    pTopLevelModel->CollectAllParameters(mapParameters);
    pTopLevelModel->GetPorts(arrPorts);
    pTopLevelModel->CollectAllVariables(mapVariables);

    for(std::map<std::string, daeParameter_t*>::iterator iter = mapParameters.begin(); iter != mapParameters.end(); iter++)
        pData->m_ptrarrParameters.push_back(iter->second);

    // Only ports from the top-level model (not internal models!)
    std::ofstream out("daetools_s_fun.txt");
    if(!out.is_open())
        daeDeclareAndThrowException(exInvalidCall);

    for(size_t i = 0; i < arrPorts.size(); i++)
    {
        daePort_t* port = arrPorts[i];

        out << "Found port: " << port->GetCanonicalName() << std::endl;

        ptrarrVariables.clear();
        port->GetVariables(ptrarrVariables);

        if(port->GetType() == eInletPort)
        {
            for(size_t i = 0; i < ptrarrVariables.size(); i++)
            {
                daeVariable_t* pVariable = ptrarrVariables[i];
                if(pVariable->GetType() == cnAssigned || pVariable->GetType() == cnSomePointsAssigned)
                {
                    daeDeclareException(exInvalidCall);
                    e << "Inlet port variables [" << pVariable->GetCanonicalName() << "] cannot have assigned values (can't be DOFs)";
                    throw e;
                }

                pData->m_ptrarrInputs.push_back(pVariable);
            }
        }
        else if(port->GetType() == eOutletPort)
        {
            for(size_t i = 0; i < ptrarrVariables.size(); i++)
            {
                daeVariable_t* pVariable = ptrarrVariables[i];
                if(pVariable->GetType() == cnAssigned || pVariable->GetType() == cnSomePointsAssigned)
                {
                    daeDeclareException(exInvalidCall);
                    e << "Outlet port variables [" << pVariable->GetCanonicalName() << "] cannot have assigned values (can't be DOFs)";
                    throw e;
                }

                pData->m_ptrarrOutputs.push_back(pVariable);
            }
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Ports can be either inlet or outlet; inlet/outlet ports are not supported [" << port->GetCanonicalName() << "]";
            throw e;
        }
    }
    out.flush();
    out.close();
    
    // DOFs
    for(std::map<std::string, daeVariable_t*>::iterator iter = mapVariables.begin(); iter != mapVariables.end(); iter++)
    {
        daeVariable_t* pVariable = iter->second;
        if(pVariable->GetType() == cnAssigned)
        {
            pData->m_ptrarrDOFs.push_back(pVariable);
        }
        else if(pVariable->GetType() == cnSomePointsAssigned)
        {
            std::cout << "Variable: " << pVariable->GetCanonicalName() << " has only some points assigned" << std::endl;
        }
    }

}

void daeSimulationLoader::SetTimeHorizon(double timeHorizon)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->SetTimeHorizon(timeHorizon);
}

void daeSimulationLoader::SetReportingInterval(double reportingInterval)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->SetReportingInterval(reportingInterval);
}

void daeSimulationLoader::SolveInitial()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->SolveInitial();
}

void daeSimulationLoader::Run()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->Run();
}

void daeSimulationLoader::Pause()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->Pause();
}

void daeSimulationLoader::Finalize()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->Finalize();
}

void daeSimulationLoader::Reinitialize()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->Reinitialize();
}

void daeSimulationLoader::Reset()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->Reset();
}

void daeSimulationLoader::ReportData()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->ReportData(pData->m_pSimulation->GetCurrentTime_());
}

double daeSimulationLoader::Integrate(bool bStopAtDiscontinuity, bool bReportDataAroundDiscontinuities)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    if(bStopAtDiscontinuity)
        return pData->m_pSimulation->Integrate(eStopAtModelDiscontinuity, bReportDataAroundDiscontinuities);
    else
        return pData->m_pSimulation->Integrate(eDoNotStopAtDiscontinuity, bReportDataAroundDiscontinuities);
}

double daeSimulationLoader::IntegrateForTimeInterval(double timeInterval, bool bReportDataAroundDiscontinuities)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    return pData->m_pSimulation->IntegrateForTimeInterval(timeInterval, bReportDataAroundDiscontinuities);
}

double daeSimulationLoader::IntegrateUntilTime(double time, bool bStopAtDiscontinuity, bool bReportDataAroundDiscontinuities)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    if(bStopAtDiscontinuity)
        return pData->m_pSimulation->IntegrateUntilTime(time, eStopAtModelDiscontinuity, bReportDataAroundDiscontinuities);
    else
        return pData->m_pSimulation->IntegrateUntilTime(time, eDoNotStopAtDiscontinuity, bReportDataAroundDiscontinuities);
}

unsigned int daeSimulationLoader::GetNumberOfParameters() const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    return pData->m_ptrarrParameters.size();
}
/*
unsigned int daeSimulationLoader::GetNumberOfDOFs() const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    return pData->m_ptrarrDOFs.size();
}
*/
unsigned int daeSimulationLoader::GetNumberOfInputs() const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    // Inputs are input ports and DOFs
    return (pData->m_ptrarrInputs.size() + pData->m_ptrarrDOFs.size());
}

unsigned int daeSimulationLoader::GetNumberOfOutputs() const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    return pData->m_ptrarrOutputs.size();
}

void daeSimulationLoader::GetParameterInfo(unsigned int index, std::string& strName, unsigned int& numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeParameter_t* pParameter = pData->m_ptrarrParameters[index];

    numberOfPoints = pParameter->GetNumberOfPoints();
    strName        = pParameter->GetCanonicalName();
}
/*
void daeSimulationLoader::GetDOFInfo(unsigned int index, std::string& strName, unsigned int& numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeVariable_t* pDOF = pData->m_ptrarrDOFs[index];

    numberOfPoints = pDOF->GetNumberOfPoints();
    strName        = pDOF->GetCanonicalName();
}
*/
void daeSimulationLoader::GetInputInfo(unsigned int index, std::string& strName, unsigned int& numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    unsigned int nInputs = pData->m_ptrarrInputs.size();
    unsigned int nDOFs   = pData->m_ptrarrDOFs.size();
    
    if(index >= nInputs + nDOFs)
    {
        daeDeclareAndThrowException(exOutOfBounds);
    }
    else if(index < nInputs)
    {
        // Inputs indexes start at nInputs (not zero!)
        daeVariable_t* pVariable = pData->m_ptrarrInputs[index];

        numberOfPoints = pVariable->GetNumberOfPoints();
        strName        = pVariable->GetCanonicalName();
    }
    else
    {
        // Achtung, Achtung!!
        // DOFs indexes start at nInputs (not zero!)
        // Obviously, if a DOF is distributed variable all its points must be fixed, that is to be DOFs
        daeVariable_t* pVariable = pData->m_ptrarrDOFs[index - nInputs];

        numberOfPoints = pVariable->GetNumberOfPoints();
        strName        = pVariable->GetCanonicalName();
    }
}

void daeSimulationLoader::GetOutputInfo(unsigned int index, std::string& strName, unsigned int& numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeVariable_t* pVariable = pData->m_ptrarrOutputs[index];

    numberOfPoints = pVariable->GetNumberOfPoints();
    strName        = pVariable->GetCanonicalName();
}

void daeSimulationLoader::GetParameterValue(unsigned int index, double* value, unsigned int numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(index >= pData->m_ptrarrParameters.size())
        daeDeclareAndThrowException(exOutOfBounds);

    daeParameter_t* pParameter = pData->m_ptrarrParameters[index];
    if(pParameter->GetNumberOfPoints() != numberOfPoints)
        daeDeclareAndThrowException(exInvalidCall);

    real_t* s_values = pParameter->GetValuePointer();
    for(unsigned int i = 0; i < numberOfPoints; i++)
        value[i] = static_cast<double>(s_values[i]);
}
/*
void daeSimulationLoader::GetDOFValue(unsigned int index, double* value, unsigned int numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(index >= pData->m_ptrarrDOFs.size())
        daeDeclareAndThrowException(exOutOfBounds);

    daeVariable_t* pDOF = pData->m_ptrarrDOFs[index];
    if(pDOF->GetNumberOfPoints() != numberOfPoints)
        daeDeclareAndThrowException(exInvalidCall);

    std::vector<real_t> s_values;
    s_values.resize(numberOfPoints);
    pDOF->GetValues(s_values);

    for(unsigned int i = 0; i < numberOfPoints; i++)
        value[i] = static_cast<double>(s_values[i]);
}
*/
void daeSimulationLoader::GetInputValue(unsigned int index, double* value, unsigned int numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    unsigned int nInputs = pData->m_ptrarrInputs.size();
    unsigned int nDOFs   = pData->m_ptrarrDOFs.size();

    daeVariable_t* pVariable = NULL;
    if(index >= nInputs + nDOFs)
    {
        daeDeclareAndThrowException(exOutOfBounds);
    }
    else if(index < nInputs)
    {
        // Inputs indexes start at nInputs (not zero!)
        pVariable = pData->m_ptrarrInputs[index];
    }
    else
    {
        // Achtung, Achtung!!
        // DOFs indexes start at nInputs (not zero!)
        // Obviously, if a DOF is distributed variable all its points must be fixed, that is to be DOFs
        pVariable = pData->m_ptrarrDOFs[index - nInputs];
    }

    if(pVariable->GetNumberOfPoints() != numberOfPoints)
        daeDeclareAndThrowException(exInvalidCall);

    std::vector<real_t> s_values;
    s_values.resize(numberOfPoints);
    pVariable->GetValues(s_values);

    for(unsigned int i = 0; i < numberOfPoints; i++)
        value[i] = static_cast<double>(s_values[i]);
}

void daeSimulationLoader::GetOutputValue(unsigned int index, double* value, unsigned int numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(index >= pData->m_ptrarrOutputs.size())
        daeDeclareAndThrowException(exOutOfBounds);

    daeVariable_t* pVariable = pData->m_ptrarrOutputs[index];
    if(pVariable->GetNumberOfPoints() != numberOfPoints)
        daeDeclareAndThrowException(exInvalidCall);

    std::vector<real_t> s_values;
    s_values.resize(numberOfPoints);
    pVariable->GetValues(s_values);

    for(unsigned int i = 0; i < numberOfPoints; i++)
        value[i] = static_cast<double>(s_values[i]);
}

void daeSimulationLoader::SetParameterValue(unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(index >= pData->m_ptrarrParameters.size())
        daeDeclareAndThrowException(exOutOfBounds);

    daeParameter_t* pParameter = pData->m_ptrarrParameters[index];
    if(pParameter->GetNumberOfPoints() != numberOfPoints)
        daeDeclareAndThrowException(exInvalidCall);

    real_t* s_values = pParameter->GetValuePointer();
    for(unsigned int i = 0; i < numberOfPoints; i++)
        s_values[i] = static_cast<real_t>(value[i]);
}
/*
void daeSimulationLoader::SetDOFValue(unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(index >= pData->m_ptrarrInputs.size())
        daeDeclareAndThrowException(exOutOfBounds);

    daeVariable_t* pDOF = pData->m_ptrarrDOFs[index];
    if(pDOF->GetNumberOfPoints() != numberOfPoints)
        daeDeclareAndThrowException(exInvalidCall);

    std::vector<real_t> s_values;
    s_values.resize(numberOfPoints);

    for(unsigned int i = 0; i < numberOfPoints; i++)
        s_values[i] = static_cast<real_t>(value[i]);

    pDOF->SetValues(s_values);
}
*/

void daeSimulationLoader::SetInputValue(unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    unsigned int nInputs = pData->m_ptrarrInputs.size();
    unsigned int nDOFs   = pData->m_ptrarrDOFs.size();

    daeVariable_t* pVariable = NULL;
    if(index >= nInputs + nDOFs)
    {
        daeDeclareAndThrowException(exOutOfBounds);
    }
    else if(index < nInputs)
    {
        // Inputs indexes start at nInputs (not zero!)
        pVariable = pData->m_ptrarrInputs[index];

        if(pVariable->GetNumberOfPoints() != numberOfPoints)
            daeDeclareAndThrowException(exInvalidCall);

        std::vector<real_t> s_values;
        s_values.resize(numberOfPoints);

        for(unsigned int i = 0; i < numberOfPoints; i++)
            s_values[i] = static_cast<real_t>(value[i]);

        pVariable->SetValues(s_values);
    }
    else
    {
        // Achtung, Achtung!!
        // DOFs indexes start at nInputs (not zero!)
        // Obviously, if a DOF is distributed variable all its points must be fixed, that is to be DOFs
        pVariable = pData->m_ptrarrDOFs[index - nInputs];

        if(pVariable->GetNumberOfPoints() != numberOfPoints)
            daeDeclareAndThrowException(exInvalidCall);

        std::vector<real_t> s_values;
        s_values.resize(numberOfPoints);

        for(unsigned int i = 0; i < numberOfPoints; i++)
            s_values[i] = static_cast<real_t>(value[i]);

        pVariable->ReAssignValues(s_values);       
    }
}

void daeSimulationLoader::SetOutputValue(unsigned int index, double* value, unsigned int numberOfPoints)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(index >= pData->m_ptrarrOutputs.size())
        daeDeclareAndThrowException(exOutOfBounds);

    daeVariable_t* pVariable = pData->m_ptrarrOutputs[index];
    if(pVariable->GetNumberOfPoints() != numberOfPoints)
        daeDeclareAndThrowException(exInvalidCall);

    std::vector<real_t> s_values;
    s_values.resize(numberOfPoints);

    for(unsigned int i = 0; i < numberOfPoints; i++)
        s_values[i] = static_cast<real_t>(value[i]);

    pVariable->SetValues(s_values);
}

void daeSimulationLoader::ShowSimulationExplorer()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

//    try
//    {
        boost::python::object main_namespace = pData->m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);
        boost::python::exec("from daetools.dae_simulator.simulation_explorer import daeSimulationExplorer", main_namespace);
        boost::python::exec("from PyQt4 import QtCore, QtGui", main_namespace);
        boost::python::exec("__qt_app__ = ( QtCore.QCoreApplication.instance() if QtCore.QCoreApplication.instance() else QtGui.QApplication(['no_main']) )", main_namespace);

        // Get the Qt QApplication object and daeSimulationExplorer class object
        boost::python::object qt_app       = main_namespace["__qt_app__"];
        boost::python::object sim_expl_cls = main_namespace["daeSimulationExplorer"];

        // Create daeSimulationExplorer object
        boost::python::object se = sim_expl_cls(qt_app, pData->m_pySimulation);

        // Show the explorer dialog
        se.attr("exec_")();
//    }
//    catch(boost::python::error_already_set const &)
//    {
//        PyErr_Print();
//    }
}

void daeSimulationLoader::LoadSimulation(const std::string& strPythonFile, const std::string& strSimulationClass)
{
//    try
//    {
        std::string command;
        boost::python::object result;

        daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
        if(!pData)
            daeDeclareAndThrowException(exInvalidPointer);

        pData->m_pyMainModule = boost::python::import("__main__");
        if(!pData->m_pyMainModule)
            daeDeclareAndThrowException(exInvalidPointer);

        boost::python::object main_namespace = pData->m_pyMainModule.attr("__dict__");
        result = boost::python::exec("import os, sys", main_namespace);

        boost::filesystem::path py_file = strPythonFile.c_str();
        std::string strSimulationModule = py_file.stem().string().c_str();
        std::string strPath = py_file.parent_path().string().c_str();

        command = (boost::format("sys.path.insert(0, '%s')") % strPath).str();
        result  = boost::python::exec(command.c_str(), main_namespace);

        command = (boost::format("import %s") % strSimulationModule).str();
// Here it fails if I do call Py_Finalize()
        result  = boost::python::exec(command.c_str(), main_namespace);

        command = (boost::format("__daetools_simulation__ = %s.%s()") % strSimulationModule % strSimulationClass).str();
        result  = boost::python::exec(command.c_str(), main_namespace);

        // Set the boost::python simulation object
        pData->m_pySimulation = main_namespace["__daetools_simulation__"];
        if(!pData->m_pySimulation)
            daeDeclareAndThrowException(exInvalidPointer);

        // Set the daeSimulation* pointer
        pData->m_pSimulation = boost::python::extract<daeSimulation_t*>(main_namespace["__daetools_simulation__"]);
        if(!pData->m_pSimulation)
            daeDeclareAndThrowException(exInvalidPointer);

//    }
//    catch(boost::python::error_already_set const &)
//    {
//        if(PyErr_ExceptionMatches(PyExc_ImportError))
//        {
//            PyErr_Print();
//        }
//        else
//        {
//            PyErr_Print();
//        }
//    }
//    catch(std::exception& e)
//    {
//        std::cout << e.what() << std::endl;
//    }
}

void daeSimulationLoaderData::SetupLASolver(const std::string& strLASolver)
{
    try
    {
//        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
//        boost::python::exec("import sys", main_namespace);
//        boost::python::exec("import daetools", main_namespace);
//        boost::python::exec("import daetools.pyDAE", main_namespace);

//    // Achtung, Achtung!!
//    // LA solver is created and owned by the Python
//        daeLASolver_t* lasolver = NULL;
//        boost::python::exec("import daetools.dae_simulator.auxiliary", main_namespace);
//        std::string command = "__la_solver__ = daetools.dae_simulator.auxiliary.createLASolverByName('" + strLASolver + "')";
//        boost::python::exec(command.c_str(), main_namespace);
//        lasolver = boost::python::extract<daeLASolver_t*>(main_namespace["__la_solver__"]);
//        if(lasolver)
//            m_pDAESolver->SetLASolver(lasolver);

        /*
        if(strLASolver == "SuperLU")
        {
            boost::python::exec("from daetools.solvers.superlu import pySuperLU", main_namespace);
            boost::python::exec("__superlu__ = pySuperLU.daeCreateSuperLUSolver()", main_namespace);
            lasolver = boost::python::extract<daeIDALASolver_t*>(main_namespace["__superlu__"]);
            m_pDAESolver->SetLASolver(lasolver);
        }
        else if(strLASolver == "SuperLU_MT")
        {
            boost::python::exec("from daetools.solvers.superlu import pySuperLU", main_namespace);
            boost::python::exec("__superlu_mt__ = pySuperLU.daeCreateSuperLUSolver()", main_namespace);
            lasolver = boost::python::extract<daeIDALASolver_t*>(main_namespace["__superlu_mt__"]);
            m_pDAESolver->SetLASolver(lasolver);
        }
        else if(strLASolver == "Trilinos Amesos - KLU")
        {
        }
        */
    }
    catch(boost::python::error_already_set const &)
    {
        PyErr_Print();
    }
}

void daeSimulationLoaderData::SetupDAESolver(const std::string& strDAESolver)
{
//    try
//    {
        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);

    // Achtung, Achtung!!
    // DAE solver is created and owned by the Python
        m_pDAESolver = NULL;
        boost::python::exec("__dae_solver__ = daetools.pyDAE.daeIDAS()", main_namespace);
        m_pDAESolver = boost::python::extract<daeDAESolver_t*>(main_namespace["__dae_solver__"]);
//    }
//    catch(boost::python::error_already_set const &)
//    {
//        PyErr_Print();
//    }
}

void daeSimulationLoaderData::SetupNLPSolver(const std::string& strNLPSolver)
{
//    try
//    {
        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);

//    }
//    catch(boost::python::error_already_set const &)
//    {
//        PyErr_Print();
//    }
}

void daeSimulationLoaderData::SetupDataReporter(const std::string& strDataReporter, const std::string &strConnectionString)
{
//    try
//    {
        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);

        //if(strDataReporter == "TCP/IP DataReporter")
       // pData->m_pDataReporter = daeCreateTCPIPDataReporter();

        m_pDataReporter = NULL;
        boost::python::exec("__data_reporter__ = daetools.pyDAE.daeTCPIPDataReporter()", main_namespace);
        m_pDataReporter = boost::python::extract<daeDataReporter_t*>(main_namespace["__data_reporter__"]);
        std::string strProcessName = m_pSimulation->GetModel()->GetName() + "-" + boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time());
        m_pDataReporter->Connect(strConnectionString, strProcessName);
//    }
//    catch(boost::python::error_already_set const &)
//    {
//        PyErr_Print();
//    }
}

void daeSimulationLoaderData::SetupLog(const std::string& strLog)
{
//    try
//    {
        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);

        //if(strLog == "StdOut")
        //pData->m_pLog = daeCreateStdOutLog();

        m_pLog = NULL;
        boost::python::exec("__log__ = daetools.pyDAE.daeStdOutLog()", main_namespace);
        m_pLog = boost::python::extract<daeLog_t*>(main_namespace["__log__"]);
//    }
//    catch(boost::python::error_already_set const &)
//    {
//        PyErr_Print();
//    }
}

