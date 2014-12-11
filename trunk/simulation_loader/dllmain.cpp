#include "stdafx.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
#endif

// GNU/Linux and MacOS compiled with g++
#if defined(__linux__) || defined(__MACH__) || defined(__APPLE__)

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <boost/python.hpp>
#include <cstdio>
#include <dlfcn.h>

void __attribute__ ((constructor)) so_load(void);
void __attribute__ ((destructor))  so_unload(void);

// Called when the library is loaded and before dlopen() returns
void so_load(void)
{
    try
    {
        char python_so_name[20];
        ::sprintf(python_so_name, "libpython%d.%d.so", DAE_PYTHON_MAJOR, DAE_PYTHON_MINOR);
        //printf("dlopen(%s);\n", python_so_name);

        dlopen(python_so_name, RTLD_GLOBAL|RTLD_LAZY);

        //char argv[] = "daeSimulationLoader";
        //Py_SetProgramName(argv);
        Py_Initialize();
    }
    catch(...)
    {
    }
}

// Called when the library is unloaded and before dlclose() returns
void so_unload(void)
{
    try
    {
        Py_Finalize();
    }
    catch(...)
    {
    }
}

#endif

