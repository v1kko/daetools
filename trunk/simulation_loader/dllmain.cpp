#include "stdafx.h"
#define BOOST_FILESYSTEM_NO_DEPRECATED
#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

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
// Here, we have to load python shared library despite it was loaded by FMI_CS library.
// Otherwise it does not work: for instance, numpy cowardly refuses to import.
#if defined(__linux__) || defined(__MACH__) || defined(__APPLE__)

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <boost/python.hpp>
#include <cstdio>
#include <dlfcn.h>
#include <iostream>
#include <sstream>

void __attribute__ ((constructor)) so_load(void);
void __attribute__ ((destructor))  so_unload(void);

void* handle = NULL;

// Called when the library is loaded and before dlopen() returns
void so_load(void)
{
    handle = NULL;
    std::stringstream f;

    try
    {
        std::vector<std::string> paths;
        char* szpathEnv            = getenv("PATH");
        char* szpythonpathEnv      = getenv("PYTHONPATH");
        std::string pathEnv        = szpathEnv       ? szpathEnv       : "";
        std::string pythonpathEnv  = szpythonpathEnv ? szpythonpathEnv : "";
        std::string path_separator = ":";
        std::string python_exe     = "python";
#if defined(__MACH__) || defined(__APPLE__)
        std::string solib_ext = "dylib";
#else
        std::string solib_ext = "so";
#endif

        f << "SimulationLoader using python" << DAE_PYTHON_MAJOR << "." << DAE_PYTHON_MINOR << std::endl;
        f << "PATH=" << pathEnv << std::endl;
        f << "PYTHONPATH=" << pythonpathEnv << std::endl;

        boost::split(paths, pathEnv, boost::is_any_of(path_separator.c_str()));
        for(int i = 0; i < paths.size(); i++)
        {
              std::string path = paths[i];
              boost::filesystem::path pythonPath = boost::filesystem::path(path) / python_exe;
              if (boost::filesystem::exists(pythonPath))
              {
                    boost::filesystem::path pythonHome_bin  = pythonPath.parent_path();
                    boost::filesystem::path pythonHome      = pythonHome_bin.parent_path();
                    boost::filesystem::path pythonHome_lib  = pythonHome / std::string("lib");

                    std::string libpython   = (boost::format("libpython%d.%d.%s")   % DAE_PYTHON_MAJOR % DAE_PYTHON_MINOR % solib_ext).str();
                    std::string libpython_m = (boost::format("libpython%d.%dm.%s")  % DAE_PYTHON_MAJOR % DAE_PYTHON_MINOR % solib_ext).str();
                    std::string libpython_u = (boost::format("libpython%d.%du.%s")  % DAE_PYTHON_MAJOR % DAE_PYTHON_MINOR % solib_ext).str();
                    boost::filesystem::path libpython_path   = pythonHome_lib / libpython;
                    boost::filesystem::path libpython_m_path = pythonHome_lib / libpython_m;
                    boost::filesystem::path libpython_u_path = pythonHome_lib / libpython_u;

                    f << "Python binary found in " << pythonHome_bin.string() << std::endl;

                    // Try to load the libpython shared library from Python found in PATH.
                    if(boost::filesystem::exists(libpython_path))
                    {
                        handle = dlopen(libpython_path.string().c_str(), RTLD_GLOBAL|RTLD_LAZY);
                        f << "Python from PATH " << libpython_path.string() << " loaded" << std::endl;
                    }
                    else if(boost::filesystem::exists(libpython_m_path))
                    {
                        handle = dlopen(libpython_m_path.string().c_str(), RTLD_GLOBAL|RTLD_LAZY);
                        f << "Python from PATH " << libpython_m_path.string() << " loaded" << std::endl;
                    }
                    else if(boost::filesystem::exists(libpython_u_path))
                    {
                        handle = dlopen(libpython_u_path.string().c_str(), RTLD_GLOBAL|RTLD_LAZY);
                        f << "Python from PATH " << libpython_u_path.string() << " loaded" << std::endl;
                    }
                    else
                    {
                        // Try to load the libpython shared library from the system library locations.
                        // This is typically executed when running the system Python.
                        handle = dlopen(libpython.c_str(), RTLD_GLOBAL|RTLD_LAZY);
                        if(handle)
                        {
                            f << "The system python " << libpython.c_str() << " loaded" << std::endl;
                            break;
                        }

                        handle = dlopen(libpython_m.c_str(), RTLD_GLOBAL|RTLD_LAZY);
                        if(handle)
                        {
                            f << "The system python " << libpython_m.c_str() << " loaded" << std::endl;
                            break;
                        }

                        handle = dlopen(libpython_u.c_str(), RTLD_GLOBAL|RTLD_LAZY);
                        if(handle)
                        {
                            f << "The system python " << libpython_u.c_str() << " loaded" << std::endl;
                            break;
                        }
                    }
                    break;
              }
        }

        // Log the activity in this function:
        std::ofstream log("/tmp/daetools-simulation_loader.log");
        if(log.is_open())
            log << f.str() << std::endl;

        // While here, initialise Python interpreter.
        Py_Initialize();
    }
    catch(std::exception& e)
    {
        // Log the activity in this function:
        std::ofstream log("/tmp/daetools-simulation_loader.log");
        if(log.is_open())
            log << f.str() << std::endl;
    }
    catch(...)
    {
        // Log the activity in this function:
        std::ofstream log("/tmp/daetools-simulation_loader.log");
        if(log.is_open())
            log << f.str() << std::endl;
    }
}

// Called when the library is unloaded and before dlclose() returns
void so_unload(void)
{
    if(handle)
        dlclose(handle);
    handle = NULL;
}

#endif

