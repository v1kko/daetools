#ifndef IDA_LA_SOLVER_INTERFACE_H
#define IDA_LA_SOLVER_INTERFACE_H

#include "../IDAS_DAESolver/solver_class_factory.h"
#include <string>

namespace dae
{
namespace solver
{
enum daeeIDALASolverType
{
	eSundialsLU = 0,
	eSundialsGMRES,
	eThirdParty
};
	
class daeIDALASolver_t
{
public:
	virtual ~daeIDALASolver_t(){}
	
public:
	virtual int Create(void* ida, 
					   size_t n, 
					   daeDAESolver_t* pDAESolver)			= 0;
	virtual int Reinitialize(void* ida)						= 0;
	virtual int Init(void* ida)								= 0;
	virtual int SaveAsXPM(const std::string& strFileName)	= 0;
};

}
}

#endif
