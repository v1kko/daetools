#ifndef DAE_SIMULATION_LOADER_COMMON_H
#define DAE_SIMULATION_LOADER_COMMON_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifdef DAE_DLL_INTERFACE
#ifdef SIMULATION_LOADER_EXPORTS
#define DAE_SIMULATION_LOADER_API __declspec(dllexport)
#else
#define DAE_SIMULATION_LOADER_API __declspec(dllimport)
#endif
#else // DAE_DLL_INTERFACE
#define DAE_SIMULATION_LOADER_API
#endif // DAE_DLL_INTERFACE

#else // WIN32
#define DAE_SIMULATION_LOADER_API
#endif // WIN32


#ifdef __cplusplus
extern "C"
{
#endif

DAE_SIMULATION_LOADER_API const char* GetLastPythonError();
DAE_SIMULATION_LOADER_API bool GetStrippedName(const char strSource[512], char strDestination[512]);

#ifdef __cplusplus
}
#endif

#endif
