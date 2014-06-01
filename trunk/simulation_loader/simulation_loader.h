#ifndef DAE_SIMULATION_LOADER_H
#define DAE_SIMULATION_LOADER_H

#include <string>
#include <iostream>
#include "../dae.h"

namespace dae
{
namespace activity
{
namespace simulation_loader
{
class daeSimulationLoader
{
public:
    daeSimulationLoader();
    ~daeSimulationLoader();

public:
// Loading functions
    void LoadPythonSimulation(const std::string& strPythonFile, const std::string& strSimulationClass);

// High-level simulation functions
    void Simulate(bool bShowSimulationExplorer = false);

// Low-lever simulation functions
    void Initialize(const std::string& strDAESolver,
                    const std::string& strLASolver,
                    const std::string& strDataReporter,
                    const std::string& strDataReporterConnectionString,
                    const std::string& strLog,
                    bool bCalculateSensitivities = false,
                    const std::string& strJSONRuntimeSettings = "");
    void SolveInitial();
    void Run();
    void Pause();
    void Finalize();

    void Reinitialize();
    void Reset();

    void ReportData();
    void ShowSimulationExplorer();

// Integration functions
	double Integrate(bool bStopAtDiscontinuity, bool bReportDataAroundDiscontinuities = true);
	double IntegrateForTimeInterval(double timeInterval, bool bReportDataAroundDiscontinuities = true);
	double IntegrateUntilTime(double time, bool bStopAtDiscontinuity, bool bReportDataAroundDiscontinuities = true);

// Model information
    void CollectAllDomains(std::map<std::string, daeDomain_t*>& mapDomains);
    void CollectAllParameters(std::map<std::string, daeParameter_t*>& mapParameters);
    void CollectAllVariables(std::map<std::string, daeVariable_t*>& mapVariables);
    void CollectAllPorts(std::map<std::string, daePort_t*>& mapPorts);
    void CollectAllSTNs(std::map<std::string, daeSTN_t*>& mapSTNs);

protected:
// Internal data
    void* m_pData;
};

}
}
}

#endif
