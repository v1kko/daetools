#ifndef NLPSOLVER_CLASS_FACTORY_H
#define NLPSOLVER_CLASS_FACTORY_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifdef DAEDLL
#ifdef NLPSOLVER_EXPORTS
#define DAE_NLPSOLVER_API __declspec(dllexport)
#else // NLPSOLVER_EXPORTS
#define DAE_NLPSOLVER_API __declspec(dllimport)
#endif // NLPSOLVER_EXPORTS
#else // DAEDLL
#define DAE_NLPSOLVER_API
#endif // DAEDLL

#else // WIN32
#define DAE_NLPSOLVER_API 
#endif // WIN32



#endif
