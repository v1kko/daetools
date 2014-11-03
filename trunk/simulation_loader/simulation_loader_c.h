#ifndef DAE_SIMULATION_LOADER_C_H
#define DAE_SIMULATION_LOADER_C_H

#ifdef __cplusplus
extern "C"
{
#endif

const char* GetLastError();
bool GetStrippedName(const char strSource[512], char strDestination[512]);

void* LoadSimulation(const char*  strPythonFile, const char* strSimulationClass);
bool Initialize(void* s,
                const char* strDAESolver,
                const char* strLASolver,
                const char* strDataReporter,
                const char* strDataReporterConnectionString,
                const char* strLog,
                bool bCalculateSensitivities);
bool InitializeJSON(void* s, const char* strJSONRuntimeSettings);
bool SetTimeHorizon(void* s, double timeHorizon);
bool SetReportingInterval(void* s, double reportingInterval);
bool SolveInitial(void* s);
bool Run(void* s);
bool Reinitialize(void* s);
bool Finalize(void* s);
bool FreeSimulation(void* s);

unsigned int GetNumberOfParameters(void* s);
unsigned int GetNumberOfInputs(void* s);
unsigned int GetNumberOfOutputs(void* s);
unsigned int GetNumberOfStateTransitionNetworks(void* s);

bool GetParameterInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints);
bool GetInputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints);
bool GetOutputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints);
bool GetStateTransitionNetworkInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfStates);

bool GetParameterValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
bool GetInputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
bool GetOutputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
bool GetActiveState(void* s, unsigned int stnIndex, char strActiveState[64]);

bool SetParameterValue(void* s, unsigned int index, const double* value, unsigned int numberOfPoints);
bool SetInputValue(void* s, unsigned int index, const double* value, unsigned int numberOfPoints);
bool SetOutputValue(void* s, unsigned int index, const double* value, unsigned int numberOfPoints);
bool SetActiveState(void* s, unsigned int stnIndex, char strActiveState[64]);

bool IntegrateForTimeInterval(void* s, double timeInterval);
bool IntegrateUntilTime(void* s, double time);
bool ReportData(void* s);

#ifdef __cplusplus
}
#endif


#endif
