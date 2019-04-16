#ifndef BONMIN_CLASS_FACTORY_H
#define BONMIN_CLASS_FACTORY_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifdef DAE_DLL_INTERFACE
#ifdef BONMIN_EXPORTS
#define DAE_BONMIN_API __declspec(dllexport)
#else
#define DAE_BONMIN_API __declspec(dllimport)
#endif
#else // DAE_DLL_INTERFACE
#define DAE_BONMIN_API
#endif // DAE_DLL_INTERFACE

#else // WIN32
#define DAE_BONMIN_API
#endif // WIN32

#endif
