#ifndef DAE_SIMULATION_LOADER_C_H
#define DAE_SIMULATION_LOADER_C_H

#ifdef __cplusplus
extern "C"
{
#endif

void Simulate(const char*  strPythonFile, const char* strSimulationClass, bool bShowSimulationExplorer);

void* LoadSimulation(const char*  strPythonFile, const char* strSimulationClass);
void Initialize(void* s,
                const char* strDAESolver,
                const char* strLASolver,
                const char* strDataReporter,
                const char* strDataReporterConnectionString,
                const char* strLog,
                bool bCalculateSensitivities);
void InitializeJSON(void* s, const char* strJSONRuntimeSettings);
void SetTimeHorizon(void* s, double timeHorizon);
void SetReportingInterval(void* s, double reportingInterval);
void SolveInitial(void* s);
void Run(void* s);
void Reinitialize(void* s);
void Finalize(void* s);
void FreeSimulation(void* s);

unsigned int GetNumberOfParameters(void* s);
unsigned int GetNumberOfInputs(void* s);
unsigned int GetNumberOfOutputs(void* s);

void GetParameterInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints);
void GetInputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints);
void GetOutputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints);

void GetParameterValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
void GetInputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
void GetOutputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);

void SetParameterValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
void SetInputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
void SetOutputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);

void  IntegrateForTimeInterval(void* s, double timeInterval);
void  IntegrateUntilTime(void* s, double time);
void  ReportData(void* s);

#ifdef __cplusplus
}
#endif


#endif
