#ifndef DAE_SIMULATION_LOADER_COMMON_H
#define DAE_SIMULATION_LOADER_COMMON_H

#ifdef __cplusplus
extern "C"
{
#endif

const char* GetLastPythonError();
bool GetStrippedName(const char strSource[512], char strDestination[512]);

#ifdef __cplusplus
}
#endif

#endif
