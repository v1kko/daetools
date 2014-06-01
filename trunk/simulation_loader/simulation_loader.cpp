#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#define BOOST_FILESYSTEM_NO_DEPRECATED
#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem.hpp>

#include "simulation_loader.h"

namespace dae
{
namespace activity
{
namespace simulation_loader
{
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
// Created and owned by Python, thus the raw pointer
    daeSimulation_t*     m_pSimulation;
    daeDataReporter_t*   m_pDataReporter;
    daeDAESolver_t*	     m_pDAESolver;
    daeLASolver_t*       m_pLASolver;
    daeLog_t*	         m_pLog;

    boost::python::object m_pyMainModule;
    boost::python::object m_pySimulation;
};

daeSimulationLoader::daeSimulationLoader()
{
    m_pData = new daeSimulationLoaderData;

    char argv[] = "daeSimulationLoader";
    Py_SetProgramName(argv);
    Py_Initialize();
}

daeSimulationLoader::~daeSimulationLoader()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(pData)
    {
        delete pData;
        m_pData = NULL;
    }

    Py_Finalize();
}

void daeSimulationLoader::Simulate(bool bShowSimulationExplorer)
{
    try
    {
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
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}

void daeSimulationLoader::Initialize(const std::string& strDAESolver,
                                     const std::string& strLASolver,
                                     const std::string& strDataReporter,
                                     const std::string& strDataReporterConnectionString,
                                     const std::string& strLog,
                                     bool bCalculateSensitivities,
                                     const std::string& strJSONRuntimeSettings)
{
    try
    {
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
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
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
        return pData->m_pSimulation->Integrate(eStopAtModelDiscontinuity);
    else
        return pData->m_pSimulation->Integrate(eDoNotStopAtDiscontinuity);
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

void daeSimulationLoader::CollectAllDomains(std::map<std::string, daeDomain_t*>& mapDomains)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    daeModel_t* pTopLevelModel = pData->m_pSimulation->GetModel();
    pTopLevelModel->CollectAllDomains(mapDomains);
}

void daeSimulationLoader::CollectAllParameters(std::map<std::string, daeParameter_t*>& mapParameters)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    daeModel_t* pTopLevelModel = pData->m_pSimulation->GetModel();
    pTopLevelModel->CollectAllParameters(mapParameters);
}

void daeSimulationLoader::CollectAllVariables(std::map<std::string, daeVariable_t*>& mapVariables)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    daeModel_t* pTopLevelModel = pData->m_pSimulation->GetModel();
    pTopLevelModel->CollectAllVariables(mapVariables);
}

void daeSimulationLoader::CollectAllPorts(std::map<std::string, daePort_t*>& mapPorts)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    daeModel_t* pTopLevelModel = pData->m_pSimulation->GetModel();
    pTopLevelModel->CollectAllPorts(mapPorts);
}

void daeSimulationLoader::CollectAllSTNs(std::map<std::string, daeSTN_t*>& mapSTNs)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    if(!pData->m_pSimulation)
        daeDeclareAndThrowException(exInvalidPointer);

    daeModel_t* pTopLevelModel = pData->m_pSimulation->GetModel();
    pTopLevelModel->CollectAllSTNs(mapSTNs);
}

void daeSimulationLoader::ShowSimulationExplorer()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    try
    {
        boost::python::object main_namespace = pData->m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);
        boost::python::exec("from daetools.dae_simulator.simulation_explorer import daeSimulationExplorer", main_namespace);
        boost::python::exec("from PyQt4 import QtCore, QtGui", main_namespace);
        boost::python::exec("__qt_app__ = QtGui.QApplication(['no_main'])", main_namespace);

        // Get the Qt QApplication object and daeSimulationExplorer class object
        boost::python::object qt_app       = main_namespace["__qt_app__"];
        boost::python::object sim_expl_cls = main_namespace["daeSimulationExplorer"];

        // Create daeSimulationExplorer object
        boost::python::object se = sim_expl_cls(qt_app, pData->m_pySimulation);

        // Show the explorer dialog
        se.attr("exec_")();
    }
    catch(boost::python::error_already_set const &)
    {
        PyErr_Print();
    }
}

void daeSimulationLoader::LoadPythonSimulation(const std::string& strPythonFile, const std::string& strSimulationClass)
{
    try
    {
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
    }
    catch(boost::python::error_already_set const &)
    {
        if(PyErr_ExceptionMatches(PyExc_ImportError))
        {
            PyErr_Print();
        }
        else
        {
            PyErr_Print();
        }
    }
}

void daeSimulationLoaderData::SetupLASolver(const std::string& strLASolver)
{
    try
    {
        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);

    // Achtung, Achtung!!
    // LA solver is created and owned by the Python
        daeLASolver_t* lasolver = NULL;
        boost::python::exec("import daetools.dae_simulator.auxiliary", main_namespace);
        std::string command = "__la_solver__ = daetools.dae_simulator.auxiliary.createLASolverByName(" + strLASolver + ")";
        boost::python::exec(command.c_str(), main_namespace);
        lasolver = boost::python::extract<daeLASolver_t*>(main_namespace["__la_solver__"]);
        if(lasolver)
            m_pDAESolver->SetLASolver(lasolver);

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
    try
    {
        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);

    // Achtung, Achtung!!
    // DAE solver is created and owned by the Python
        m_pDAESolver = NULL;
        boost::python::exec("__dae_solver__ = daetools.pyDAE.daeIDAS()", main_namespace);
        m_pDAESolver = boost::python::extract<daeDAESolver_t*>(main_namespace["__dae_solver__"]);
    }
    catch(boost::python::error_already_set const &)
    {
        PyErr_Print();
    }
}

void daeSimulationLoaderData::SetupNLPSolver(const std::string& strNLPSolver)
{
    try
    {
        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);

    }
    catch(boost::python::error_already_set const &)
    {
        PyErr_Print();
    }
}

void daeSimulationLoaderData::SetupDataReporter(const std::string& strDataReporter, const std::string &strConnectionString)
{
    try
    {
        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);

        //if(strDataReporter == "TCP/IP DataReporter")
       // pData->m_pDataReporter = daeCreateTCPIPDataReporter();

        m_pDataReporter = NULL;
        boost::python::exec("__data_reporter__ = daetools.pyDAE.daeTCPIPDataReporter()", main_namespace);
        m_pDataReporter = boost::python::extract<daeDataReporter_t*>(main_namespace["__data_reporter__"]);

    }
    catch(boost::python::error_already_set const &)
    {
        PyErr_Print();
    }
}

void daeSimulationLoaderData::SetupLog(const std::string& strLog)
{
    try
    {
        boost::python::object main_namespace = m_pyMainModule.attr("__dict__");
        boost::python::exec("import sys", main_namespace);
        boost::python::exec("import daetools", main_namespace);
        boost::python::exec("import daetools.pyDAE", main_namespace);

        //if(strLog == "StdOut")
        //pData->m_pLog = daeCreateStdOutLog();

        m_pLog = NULL;
        boost::python::exec("__log__ = daetools.pyDAE.daeStdOutLog()", main_namespace);
        m_pLog = boost::python::extract<daeLog_t*>(main_namespace["__log__"]);
    }
    catch(boost::python::error_already_set const &)
    {
        PyErr_Print();
    }
}


}
}
}
