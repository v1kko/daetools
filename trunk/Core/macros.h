/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_MACROS_H
#define DAE_MACROS_H

#include "helpers.h"

#define daepython_try		try {
#define daepython_catch		} \
							catch(const boost::python::error_already_set&) \
							{ \
							} \
							catch(const std::exception& e) \
							{ \
								PyErr_SetString(PyExc_RuntimeError, e.what()); \
								boost::python::throw_error_already_set(); \
							} \
							catch(...) \
							{ \
								PyErr_SetString(PyExc_RuntimeError, "unidentifiable C++ exception"); \
								boost::python::throw_error_already_set(); \
							}

#define AddFunction(MODEL_CLASS, FUNCTION)	AddEquation<MODEL_CLASS>(#FUNCTION, &MODEL_CLASS::FUNCTION);

#define daeRegisterModel(CLASS)					static bool _bRetCreate##CLASS = g_ModelClassFactory.RegisterModel(               string(#CLASS), new dae::daeCreateObjectDelegateDerived<CLASS, dae::core::daeModel>());

#define daeRegisterDynamicSimulation(CLASS)		static bool _bRetCreate##CLASS = g_ActivityClassFactory.RegisterDynamicSimulation(string(#CLASS), new dae::daeCreateObjectDelegateDerived<CLASS, dae::activity::daeSimulation_t>());

#define daeRegisterDataReporter(CLASS)			static bool _bRetCreate##CLASS = g_DataReportingClassFactory.RegisterDataReporter(string(#CLASS), new dae::daeCreateObjectDelegateDerived<CLASS, dae::datareporting::daeDataReporter_t>());

#define daeRegisterDAESolver(CLASS)				static bool _bRetCreate##CLASS = g_SolverClassFactory.RegisterDAESolver(          string(#CLASS), new dae::daeCreateObjectDelegateDerived<CLASS, dae::solver::daeDAESolver_t>());

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#define DAE_SYMBOL_EXPORT __declspec(dllexport)
#else
#define DAE_SYMBOL_EXPORT
#endif

#define daeDeclareModelLibrary(Name, Description, AuthorInfo, LicenceInfo, Version)		\
		static daeCoreClassFactory g_ModelClassFactory(Name, Description, AuthorInfo, LicenceInfo, Version); \
		extern "C" \
		{ \
			DAE_SYMBOL_EXPORT daeCoreClassFactory_t* GetCoreClassFactory(void); \
			daeCoreClassFactory_t* GetCoreClassFactory(void) \
			{ \
				return &g_ModelClassFactory; \
			} \
		}

#define daeDeclareDataReportingLibrary(Name, Description, AuthorInfo, LicenceInfo, Version)		\
		static daeDataReportingClassFactory g_DataReportingClassFactory(Name, Description, AuthorInfo, LicenceInfo, Version); \
		extern "C" \
		{ \
			DAE_SYMBOL_EXPORT daeDataReportingClassFactory_t* GetDataReportingClassFactory(void); \
			daeDataReportingClassFactory_t* GetDataReportingClassFactory(void) \
			{ \
			  return &g_DataReportingClassFactory; \
			} \
		}
																						
#define daeDeclareActivityLibrary(Name, Description, AuthorInfo, LicenceInfo, Version)	\
		static daeActivityClassFactory g_ActivityClassFactory(Name, Description, AuthorInfo, LicenceInfo, Version); \
		extern "C" \
		{ \
			DAE_SYMBOL_EXPORT daeActivityClassFactory_t* GetActivityClassFactory(void); \
			daeActivityClassFactory_t* GetActivityClassFactory(void) \
			{ \
				return &g_ActivityClassFactory; \
			} \
		}
																						
#define daeDeclareSolverLibrary(Name, Description, AuthorInfo, LicenceInfo, Version)	\
		static daeSolverClassFactory g_SolverClassFactory(Name, Description, AuthorInfo, LicenceInfo, Version); \
		extern "C" \
		{ \
			DAE_SYMBOL_EXPORT daeSolverClassFactory_t* GetSolverClassFactory(void); \
			daeSolverClassFactory_t* GetSolverClassFactory(void) \
			{ \
				return &g_SolverClassFactory; \
			} \
		}

#endif
