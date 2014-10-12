#ifndef DAE_SIMULATION_LOADER_C_H
#define DAE_SIMULATION_LOADER_C_H

#ifdef __cplusplus
extern "C"
{
#endif

void  Simulate(const char*  strPythonFile, const char* strSimulationClass, bool bShowSimulationExplorer);

void* LoadSimulation(const char*  strPythonFile, const char* strSimulationClass);
void  Initialize(void* s);
void  ReInitialize(void* s);

void  IntegrateForTimeInterval(void* s, double timeInterval);
void  IntegrateUntilTime(void* s, double time);

void  FreeSimulation(void* s);

#ifdef __cplusplus
}
#endif


#endif
