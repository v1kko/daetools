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

/*
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

void Simulate(const char*  strPythonFile, const char* strSimulationClass, bool bShowSimulationExplorer)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)loader;
    if(ptr_loader)
        ptr_loader->Simulate(bShowSimulationExplorer);
}

void FreeLoader(void* loader)
{
    daeSimulationLoader* ptr_loader = (daeSimulationLoader*)loader;
    if(ptr_loader)
        delete ptr_loader;
}
*/

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

        pData->m_pSimulation->SetReportingInterval(10);
        pData->m_pSimulation->SetTimeHorizon(1000);
        pData->m_pSimulation->GetModel()->SetReportingOn(true);

        pData->m_pSimulation->Initialize(pData->m_pDAESolver, pData->m_pDataReporter, pData->m_pLog, bCalculateSensitivities, strJSONRuntimeSettings);
//    }
//    catch(std::exception& e)
//    {
//        std::cout << e.what() << std::endl;
//    }
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

size_t daeSimulationLoader::GetNumberOfParameters() const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    return pData->m_ptrarrParameters.size();
}

size_t daeSimulationLoader::GetNumberOfInputs() const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    return pData->m_ptrarrInputs.size();
}

size_t daeSimulationLoader::GetNumberOfOutputs() const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    return pData->m_ptrarrOutputs.size();
}

void daeSimulationLoader::GetParameterInfo(size_t index, std::string& strName, size_t& numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeParameter_t* pParameter = pData->m_ptrarrParameters[index];

    numberOfPoints = pParameter->GetNumberOfPoints();
    strName        = pParameter->GetCanonicalName();
}

void daeSimulationLoader::GetInputInfo(size_t index, std::string& strName, size_t& numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeVariable_t* pVariable = pData->m_ptrarrInputs[index];

    numberOfPoints = pVariable->GetNumberOfPoints();
    strName        = pVariable->GetCanonicalName();
}

void daeSimulationLoader::GetOutputInfo(size_t index, std::string& strName, size_t& numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeVariable_t* pVariable = pData->m_ptrarrOutputs[index];

    numberOfPoints = pVariable->GetNumberOfPoints();
    strName        = pVariable->GetCanonicalName();
}

void daeSimulationLoader::GetParameterValue(size_t index, double* value, size_t numberOfPoints) const
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
    for(size_t i = 0; i < numberOfPoints; i++)
        value[i] = static_cast<double>(s_values[i]);
}

void daeSimulationLoader::GetInputValue(size_t index, double* value, size_t numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(index >= pData->m_ptrarrInputs.size())
        daeDeclareAndThrowException(exOutOfBounds);

    daeVariable_t* pVariable = pData->m_ptrarrInputs[index];
    if(pVariable->GetNumberOfPoints() != numberOfPoints)
        daeDeclareAndThrowException(exInvalidCall);

    std::vector<real_t> s_values;
    s_values.resize(numberOfPoints);
    pVariable->GetValues(s_values);

    for(size_t i = 0; i < numberOfPoints; i++)
        value[i] = static_cast<double>(s_values[i]);
}

void daeSimulationLoader::GetOutputValue(size_t index, double* value, size_t numberOfPoints) const
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

    for(size_t i = 0; i < numberOfPoints; i++)
        value[i] = static_cast<double>(s_values[i]);
}

void daeSimulationLoader::SetParameterValue(size_t index, double* value, size_t numberOfPoints)
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
    for(size_t i = 0; i < numberOfPoints; i++)
        s_values[i] = static_cast<real_t>(value[i]);
}

void daeSimulationLoader::SetInputValue(size_t index, double* value, size_t numberOfPoints)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(index >= pData->m_ptrarrInputs.size())
        daeDeclareAndThrowException(exOutOfBounds);

    daeVariable_t* pVariable = pData->m_ptrarrInputs[index];
    if(pVariable->GetNumberOfPoints() != numberOfPoints)
        daeDeclareAndThrowException(exInvalidCall);

    std::vector<real_t> s_values;
    s_values.resize(numberOfPoints);

    for(size_t i = 0; i < numberOfPoints; i++)
        s_values[i] = static_cast<real_t>(value[i]);

    pVariable->SetValues(s_values);
}

void daeSimulationLoader::SetOutputValue(size_t index, double* value, size_t numberOfPoints)
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

    for(size_t i = 0; i < numberOfPoints; i++)
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

        // Collect all parameters and ports and initialize parameters/inputs/outputs arrays
        std::vector<daeVariable_t*> ptrarrVariables;
        std::map<std::string, daeParameter_t*> mapParameters;
        std::map<std::string, daePort_t*> mapPorts;
        daeModel_t* pTopLevelModel = pData->m_pSimulation->GetModel();

        pTopLevelModel->CollectAllParameters(mapParameters);
        pTopLevelModel->CollectAllPorts(mapPorts);

        for(std::map<std::string, daeParameter_t*>::iterator iter = mapParameters.begin(); iter != mapParameters.end(); iter++)
            pData->m_ptrarrParameters.push_back(iter->second);

        for(std::map<std::string, daePort_t*>::iterator iter = mapPorts.begin(); iter != mapPorts.end(); iter++)
        {
            ptrarrVariables.clear();
            iter->second->GetVariables(ptrarrVariables);

            if(iter->second->GetType() == eInletPort)
            {
                for(size_t i = 0; i < ptrarrVariables.size(); i++)
                    pData->m_ptrarrInputs.push_back(ptrarrVariables[i]);
            }
            else if(iter->second->GetType() == eOutletPort)
            {
                for(size_t i = 0; i < ptrarrVariables.size(); i++)
                    pData->m_ptrarrOutputs.push_back(ptrarrVariables[i]);
            }
        }
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

