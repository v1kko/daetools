#ifndef DAE_SIMULATION_LOADER_COMMON_H
#define DAE_SIMULATION_LOADER_COMMON_H

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#ifdef DAE_DLL_EXPORTS
#define DAE_API __declspec(dllexport)
#else // CONFIG_EXPORTS
#define DAE_API __declspec(dllimport)
#endif // CONFIG_EXPORTS
#else // WIN32
#define DAE_API
#endif // WIN32

#ifdef __cplusplus
extern "C"
{
#endif

DAE_API const char* GetLastPythonError();
DAE_API bool GetStrippedName(const char strSource[512], char strDestination[512]);

#ifdef __cplusplus
}
#endif

#endif
