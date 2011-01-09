#include "stdafx.h"
#include "coreimpl.h"
#include "xmlfunctions.h"
#include <typeinfo>
using namespace dae;
using namespace dae::xml;
using namespace boost;

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	daeExternalFunctionArgument
**********************************************************************************************/
daeExternalFunctionArgument::daeExternalFunctionArgument(const string& strName, const vector<adouble>& adarrArguments)
{
	m_strName        = strName;
	m_adarrArguments = adarrArguments;
}

daeExternalFunctionArgument::~daeExternalFunctionArgument(void)
{
}

void daeExternalFunctionArgument::GetValues(real_t* values, size_t n)
{
}

void daeExternalFunctionArgument::SetValues(const real_t* values, size_t n)
{
}

daeExternalFunctionArgumentInfo_t daeExternalFunctionArgument::GetInfo(void) const
{
	return daeExternalFunctionArgumentInfo_t();
}

/*********************************************************************************************
	daeExternalFunction
**********************************************************************************************/
daeExternalFunction::daeExternalFunction(void)
{
}

daeExternalFunction::~daeExternalFunction(void)
{
}

void daeExternalFunction::GetArguments(std::vector<daeExternalFunctionArgument_t*>& ptrarrArguments) const
{
}

void daeExternalFunction::Calculate(real_t* results, size_t n)
{
}

void daeExternalFunction::CalculateDerivatives(daeMatrix<real_t>& derivatives)
{
}

daeExternalFunctionInfo_t daeExternalFunction::GetInfo(void) const
{
	return daeExternalFunctionInfo_t();
}

/*********************************************************************************************
	daeExternalObject
**********************************************************************************************/
daeExternalObject::daeExternalObject(void)
{
}

daeExternalObject::~daeExternalObject(void)
{
}

daeExternalFunction_t* daeExternalObject::CreateFunction(const std::string& strFunctionName)
{
	return NULL;
}

daeExternalObjectInfo_t daeExternalObject::GetInfo(void) const
{
	return daeExternalObjectInfo_t();
}


	
	
}
}

