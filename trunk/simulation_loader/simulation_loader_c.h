#ifndef DAE_SIMULATION_LOADER_C_H
#define DAE_SIMULATION_LOADER_C_H

#ifdef __cplusplus
extern "C"
{
#endif

void  Simulate(const char*  strPythonFile, const char* strSimulationClass, bool bShowSimulationExplorer);

/*
void* LoadSimulation(const char*  strPythonFile, const char* strSimulationClass);
void  Simulate(void* loader, bool bShowSimulationExplorer);
void  FreeLoader(void* loader);
*/

#ifdef __cplusplus
}
#endif


#endif
