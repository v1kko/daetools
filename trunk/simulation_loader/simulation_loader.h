#ifndef DAE_SIMULATION_LOADER_H
#define DAE_SIMULATION_LOADER_H

#include <string>
#include <iostream>

namespace dae_simulation_loader
{
class daeSimulationLoader
{
public:
    daeSimulationLoader();
    ~daeSimulationLoader();

public:
// Loading functions
    void LoadSimulation(const std::string& strPythonFile, const std::string& strSimulationClass);

// High-level simulation functions
    void Simulate(bool bShowSimulationExplorer = false);

// Low-level simulation functions
    void Initialize(const std::string& strDAESolver,
                    const std::string& strLASolver,
                    const std::string& strDataReporter,
                    const std::string& strDataReporterConnectionString,
                    const std::string& strLog,
                    bool bCalculateSensitivities = false,
                    const std::string& strJSONRuntimeSettings = "");
    void Initialize(const std::string& strJSONRuntimeSettings);
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
    size_t GetNumberOfParameters() const;
    size_t GetNumberOfInputs() const;
    size_t GetNumberOfOutputs() const;

    void GetParameterInfo(size_t index, std::string& strName, size_t& numberOfPoints) const;
    void GetInputInfo(size_t index, std::string& strName, size_t& numberOfPoints) const;
    void GetOutputInfo(size_t index, std::string& strName, size_t& numberOfPoints) const;

    void GetParameterValue(size_t index, double* value, size_t numberOfPoints) const;
    void GetInputValue(size_t index, double* value, size_t numberOfPoints) const;
    void GetOutputValue(size_t index, double* value, size_t numberOfPoints) const;

    void SetParameterValue(size_t index, double* value, size_t numberOfPoints);
    void SetInputValue(size_t index, double* value, size_t numberOfPoints);
    void SetOutputValue(size_t index, double* value, size_t numberOfPoints);

protected:
// Internal data
    void* m_pData;
};

}

#endif
