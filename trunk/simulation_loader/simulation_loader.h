#ifndef DAE_SIMULATION_LOADER_H
#define DAE_SIMULATION_LOADER_H

#include <string>
#include <iostream>

class daeSimulationLoader
{
public:
    daeSimulationLoader();
    ~daeSimulationLoader();

public:
// Loading function
    void LoadSimulation(const std::string& strPythonFile, const std::string& strSimulationClass);

// Low-level simulation functions
    void Initialize(const std::string& strDAESolver,
                    const std::string& strLASolver,
                    const std::string& strDataReporter,
                    const std::string& strDataReporterConnectionString,
                    const std::string& strLog,
                    bool bCalculateSensitivities = false,
                    const std::string& strJSONRuntimeSettings = "");
    void Initialize(const std::string& strJSONRuntimeSettings);

    void SetTimeHorizon(double timeHorizon);
    void SetReportingInterval(double reportingInterval);
    void SolveInitial();
    void Run();
    void Pause();
    void Finalize();
    void Reinitialize();
    void Reset();
    void ShowSimulationExplorer();

// Data reporting
    void ReportData();

// Integration functions
    double Integrate(bool bStopAtDiscontinuity, bool bReportDataAroundDiscontinuities = true);
    double IntegrateForTimeInterval(double timeInterval, bool bReportDataAroundDiscontinuities = true);
    double IntegrateUntilTime(double time, bool bStopAtDiscontinuity, bool bReportDataAroundDiscontinuities = true);

// Info functions
    unsigned int GetNumberOfParameters() const;
    unsigned int GetNumberOfInputs() const;
    unsigned int GetNumberOfOutputs() const;
    unsigned int GetNumberOfStateTransitionNetworks() const;

    void GetParameterInfo(unsigned int index, std::string& strName, unsigned int* numberOfPoints) const;
    void GetInputInfo(unsigned int index, std::string& strName, unsigned int* numberOfPoints) const;
    void GetOutputInfo(unsigned int index, std::string& strName, unsigned int* numberOfPoints) const;
    void GetStateTransitionNetworkInfo(unsigned int index, std::string& strName, unsigned int* numberOfStates) const;

    void GetParameterValue(unsigned int index, double* value, unsigned int numberOfPoints) const;
    void GetInputValue(unsigned int index, double* value, unsigned int numberOfPoints) const;
    void GetOutputValue(unsigned int index, double* value, unsigned int numberOfPoints) const;
    void GetActiveState(unsigned int index, std::string& strActiveState) const;

    void SetParameterValue(unsigned int index, const double* value, unsigned int numberOfPoints);
    void SetInputValue(unsigned int index, const double* value, unsigned int numberOfPoints);
    void SetOutputValue(unsigned int index, const double* value, unsigned int numberOfPoints);
    void SetActiveState(unsigned int index, const std::string& strActiveState);

    static std::string GetStrippedName(const std::string& strSource);

protected:
    void SetupInputsAndOutputs();

protected:
// Internal data
    void* m_pData;
};

#endif
