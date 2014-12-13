#ifndef DAE_SIMULATION_LOADER_C_H
#define DAE_SIMULATION_LOADER_C_H

#include "simulation_loader_common.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Loading function */
void* LoadSimulation(const char*  strPythonFile, const char* strSimulationCallable, const char* strArguments);

/* Nota bene:
 * Initialize() functions not needed anymore since the python callable object has to instantiate and initialize a simulation.
 * However, they are still available, just in case. */
bool Initialize(void* s,
                const char* strDAESolver,
                const char* strLASolver,
                const char* strDataReporter,
                const char* strDataReporterConnectionString,
                const char* strLog,
                bool bCalculateSensitivities);
bool InitializeJSON(void* s, const char* strJSONRuntimeSettings);

/* Low-level simulation functions */
bool SetTimeHorizon(void* s, double timeHorizon);
bool SetReportingInterval(void* s, double reportingInterval);
bool SolveInitial(void* s);
bool Run(void* s);
bool Reinitialize(void* s);
bool ResetToInitialSystem(void* s);
bool Finalize(void* s);
bool FreeSimulation(void* s);

/* Data reporting */
bool ReportData(void* s);

/* Integration functions */
bool IntegrateForTimeInterval(void* s, double timeInterval);
bool IntegrateUntilTime(void* s, double time);

/* Info functions */
unsigned int GetNumberOfParameters(void* s);
unsigned int GetNumberOfInputs(void* s);
unsigned int GetNumberOfOutputs(void* s);
unsigned int GetNumberOfStateTransitionNetworks(void* s);

bool GetParameterInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints);
bool GetInputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints);
bool GetOutputInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfPoints);
bool GetStateTransitionNetworkInfo(void* s, unsigned int index, char strName[512], unsigned int* numberOfStates);

/* Get/Set value functions */
bool GetParameterValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
bool GetInputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
bool GetOutputValue(void* s, unsigned int index, double* value, unsigned int numberOfPoints);
bool GetActiveState(void* s, unsigned int stnIndex, char strActiveState[64]);

bool SetParameterValue(void* s, unsigned int index, const double* value, unsigned int numberOfPoints);
bool SetInputValue(void* s, unsigned int index, const double* value, unsigned int numberOfPoints);
bool SetOutputValue(void* s, unsigned int index, const double* value, unsigned int numberOfPoints);
bool SetActiveState(void* s, unsigned int stnIndex, char strActiveState[64]);

/* FMI interface */
bool GetFMIValue(void* s, unsigned int fmi_reference, double* value);
bool GetFMIActiveState(void* s, unsigned int fmi_reference, char* state);

bool SetFMIValue(void* s, unsigned int fmi_reference, double value);
bool SetFMIActiveState(void* s, unsigned int fmi_reference, const char* state);

#ifdef __cplusplus
}
#endif


#endif
