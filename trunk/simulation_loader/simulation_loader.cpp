#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#define BOOST_FILESYSTEM_NO_DEPRECATED
#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>

#include "simulation_loader.h"
#include "../dae.h"

class daeSimulationLoaderData
{
public:
    daeSimulationLoaderData()
    {
        m_pSimulation = NULL;
    }

public:
// Created and owned by Python, thus the raw pointers
    daeSimulation_t*     m_pSimulation;

// Parameters/Inputs/Outputs
    std::vector<daeParameter_t*> m_ptrarrParameters;
    std::vector<daeVariable_t*>  m_ptrarrInputs;
    std::vector<daeVariable_t*>  m_ptrarrOutputs;
    std::vector<daeSTN_t*>       m_ptrarrSTNs;
    std::vector<daeVariable_t*>  m_ptrarrDOFs;

// FMI references
    std::map<size_t, daeFMI2Object_t> m_mapFMIReferences;

// Python related objects
    boost::python::object m_pyMainModule;
    boost::python::object m_pySimulation;
};

daeSimulationLoader::daeSimulationLoader()
{
    m_pData = new daeSimulationLoaderData;

    // Achtung, Achtung!!
    // Py_Initialize() call moved to dllmain.cpp

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

    // Achtung, Achtung!!
    //Py_Finalize() call moved to dllmain.cpp
}

void daeSimulationLoader::Initialize(const std::string& strJSONRuntimeSettings)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    boost::python::dict locals;
    boost::python::object main_namespace = pData->m_pyMainModule.attr("__dict__");
    boost::python::exec("from daetools.dae_simulator.auxiliary import InitializeSimulationJSON", main_namespace);
    locals["_json_runtime_settings_"] = strJSONRuntimeSettings;
    std::string command = "__DAESolver__, __LASolver__, __DataReporter__, __Log__ = InitializeSimulationJSON(__daetools_simulation__, _json_runtime_settings_)";
    boost::python::exec(command.c_str(), main_namespace, locals);

    SetupInputsAndOutputs();
}

void daeSimulationLoader::Initialize(const std::string& strDAESolver,
                                     const std::string& strLASolver,
                                     const std::string& strDataReporter,
                                     const std::string& strDataReporterConnectionString,
                                     const std::string& strLog,
                                     bool bCalculateSensitivities,
                                     const std::string& strJSONRuntimeSettings)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_pSimulation->SetReportingInterval(1); // only provisional, must be set by SetReportingInterval
    pData->m_pSimulation->SetTimeHorizon(10);      // only provisional, must be set by SetTimeHorizon
    pData->m_pSimulation->GetModel()->SetReportingOn(true);

    boost::python::object main_namespace = pData->m_pyMainModule.attr("__dict__");
    boost::python::exec("from daetools.dae_simulator.auxiliary import InitializeSimulation", main_namespace);
    std::string msg     = "__DAESolver__, __LASolver__, __DataReporter__, __Log__ = InitializeSimulation(__daetools_simulation__, '%s', '%s', '%s', '%s', '%s', %s)";
    std::string command = (boost::format(msg) % strDAESolver % strLASolver % strDataReporter
                                              % strDataReporterConnectionString % strLog % (bCalculateSensitivities ? "True" : "False")).str();
    boost::python::exec(command.c_str(), main_namespace);

    SetupInputsAndOutputs();
}

void daeSimulationLoader::SetupInputsAndOutputs()
{
    // Simulation must be first initialized
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeModel_t* pTopLevelModel = pData->m_pSimulation->GetModel();
    pTopLevelModel->GetCoSimulationInterface(pData->m_ptrarrParameters,
                                             pData->m_ptrarrInputs,
                                             pData->m_ptrarrOutputs,
                                             pData->m_ptrarrSTNs);
}

std::string daeSimulationLoader::GetStrippedName(const std::string& strSource)
{
    std::string strStripped = strSource;
  
    std::replace(strStripped.begin(), strStripped.end(), '.', '_');
    std::replace(strStripped.begin(), strStripped.end(), '(', '_');
    std::replace(strStripped.begin(), strStripped.end(), ')', '_');
    std::replace(strStripped.begin(), strStripped.end(), '&', '_');
    std::replace(strStripped.begin(), strStripped.end(), ';', '_');
    
    return strStripped;
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
    return pData->m_ptrarrInputs.size(); // + pData->m_ptrarrDOFs.size();
}

unsigned int daeSimulationLoader::GetNumberOfOutputs() const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    return pData->m_ptrarrOutputs.size();
}

unsigned int daeSimulationLoader::GetNumberOfStateTransitionNetworks() const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    return pData->m_ptrarrSTNs.size();
}

void daeSimulationLoader::GetParameterInfo(unsigned int index, std::string& strName, unsigned int* numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeParameter_t* pParameter = pData->m_ptrarrParameters[index];

    *numberOfPoints = pParameter->GetNumberOfPoints();
    strName         = pParameter->GetCanonicalName();
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
void daeSimulationLoader::GetInputInfo(unsigned int index, std::string& strName, unsigned int* numberOfPoints) const
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

        *numberOfPoints = pVariable->GetNumberOfPoints();
        strName         = pVariable->GetCanonicalName();
    }
    else
    {
        // Achtung, Achtung!!
        // DOFs indexes start at nInputs (not zero!)
        // Obviously, if a DOF is distributed variable all its points must be fixed, that is to be DOFs
        daeVariable_t* pVariable = pData->m_ptrarrDOFs[index - nInputs];

        *numberOfPoints = pVariable->GetNumberOfPoints();
        strName         = pVariable->GetCanonicalName();
    }
}

void daeSimulationLoader::GetOutputInfo(unsigned int index, std::string& strName, unsigned int* numberOfPoints) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeVariable_t* pVariable = pData->m_ptrarrOutputs[index];

    *numberOfPoints = pVariable->GetNumberOfPoints();
    strName         = pVariable->GetCanonicalName();
}

void daeSimulationLoader::GetStateTransitionNetworkInfo(unsigned int index, std::string& strName, unsigned int* numberOfStates) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    std::vector<daeState_t*> ptrarrStates;

    daeSTN_t* pSTN = pData->m_ptrarrSTNs[index];
    if(!pSTN)
        daeDeclareAndThrowException(exInvalidPointer);

    pSTN->GetStates(ptrarrStates);

    *numberOfStates = ptrarrStates.size();
    strName         = pSTN->GetCanonicalName();
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

void daeSimulationLoader::GetActiveState(unsigned int index, std::string& strActiveState) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeSTN_t* pSTN = pData->m_ptrarrSTNs[index];
    if(!pSTN)
        daeDeclareAndThrowException(exInvalidPointer);
    strActiveState = pSTN->GetActiveState()->GetName();
}

void daeSimulationLoader::SetParameterValue(unsigned int index, const double* value, unsigned int numberOfPoints)
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
        s_values[i] = static_cast<const real_t>(value[i]);
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

void daeSimulationLoader::SetInputValue(unsigned int index, const double* value, unsigned int numberOfPoints)
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
            s_values[i] = static_cast<const real_t>(value[i]);

        pVariable->ReAssignValues(s_values);
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
            s_values[i] = static_cast<const real_t>(value[i]);

        pVariable->ReAssignValues(s_values);       
    }
}

void daeSimulationLoader::SetOutputValue(unsigned int index, const double* value, unsigned int numberOfPoints)
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
        s_values[i] = static_cast<const real_t>(value[i]);

    pVariable->SetValues(s_values);
}

void daeSimulationLoader::SetActiveState(unsigned int index, const std::string& strActiveState)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    daeSTN_t* pSTN = pData->m_ptrarrSTNs[index];
    if(!pSTN)
        daeDeclareAndThrowException(exInvalidPointer);
    pSTN->SetActiveState(strActiveState);
}

void daeSimulationLoader::ShowSimulationExplorer()
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

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
}

void daeSimulationLoader::LoadSimulation(const std::string& strPythonFile, const std::string& strSimulationClass)
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
    if(!pData->m_pSimulation->GetModel())
        daeDeclareAndThrowException(exInvalidPointer);

    pData->m_mapFMIReferences.clear();
    pData->m_pSimulation->GetModel()->GetFMIInterface(pData->m_mapFMIReferences);
}

double daeSimulationLoader::GetFMIValue(unsigned int fmi_reference) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    std::map<size_t, daeFMI2Object_t>::const_iterator citer;
    citer = pData->m_mapFMIReferences.find(fmi_reference);
    if(citer == pData->m_mapFMIReferences.end())
        daeDeclareAndThrowException(exInvalidCall);

    const daeFMI2Object_t& fmi = citer->second;
    if(fmi.type == "Parameter")
        return fmi.parameter->GetValue(fmi.indexes);
    else if(fmi.type == "Input" || fmi.type == "Output")
        return fmi.variable->GetValue(fmi.indexes);
    else if(fmi.type == "STN")
        return fmi.variable->GetValue(fmi.indexes);
    else
        daeDeclareAndThrowException(exInvalidCall);

    return 0;
}

std::string daeSimulationLoader::GetFMIActiveState(unsigned int fmi_reference) const
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    std::map<size_t, daeFMI2Object_t>::const_iterator citer;
    citer = pData->m_mapFMIReferences.find(fmi_reference);
    if(citer == pData->m_mapFMIReferences.end())
        daeDeclareAndThrowException(exInvalidCall);

    const daeFMI2Object_t& fmi = citer->second;
    if(fmi.type == "STN")
        return fmi.stn->GetActiveState()->GetName();
    else
        daeDeclareAndThrowException(exInvalidCall);
}

void daeSimulationLoader::SetFMIValue(unsigned int fmi_reference, double value)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    std::map<size_t, daeFMI2Object_t>::const_iterator citer;
    citer = pData->m_mapFMIReferences.find(fmi_reference);
    if(citer == pData->m_mapFMIReferences.end())
        daeDeclareAndThrowException(exInvalidCall);

    const daeFMI2Object_t& fmi = citer->second;
    if(fmi.type == "Parameter")
        fmi.parameter->SetValue(fmi.indexes, value);
    else if(fmi.type == "Input" || fmi.type == "Output")
        fmi.variable->SetValue(fmi.indexes, value);
    else if(fmi.type == "STN")
        fmi.variable->SetValue(fmi.indexes, value);
    else
        daeDeclareAndThrowException(exInvalidCall);
}

void daeSimulationLoader::SetFMIActiveState(unsigned int fmi_reference, const std::string& value)
{
    daeSimulationLoaderData* pData = static_cast<daeSimulationLoaderData*>(m_pData);
    if(!pData)
        daeDeclareAndThrowException(exInvalidPointer);

    std::map<size_t, daeFMI2Object_t>::const_iterator citer;
    citer = pData->m_mapFMIReferences.find(fmi_reference);
    if(citer == pData->m_mapFMIReferences.end())
        daeDeclareAndThrowException(exInvalidCall);

    const daeFMI2Object_t& fmi = citer->second;
    if(fmi.type == "STN")
        return fmi.stn->SetActiveState(value);
    else
        daeDeclareAndThrowException(exInvalidCall);
}
