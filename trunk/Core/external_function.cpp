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
//daeExternalFunctionArgument::daeExternalFunctionArgument(const string& strName, size_t n)
//{
//	if(n == 0)
//		daeDeclareAndThrowException(exInvalidCall);
	
//	N          = n;
//	m_strName  = strName;
//	m_pdValues = new real_t[N];
//}

//daeExternalFunctionArgument::~daeExternalFunctionArgument(void)
//{
//	if(m_pdValues)
//		delete[] m_pdValues;
//}

//void daeExternalFunctionArgument::GetValues(real_t* values, size_t n)
//{
//	if(n != N)
//		daeDeclareAndThrowException(exInvalidCall);
//	if(!m_pdValues)
//		daeDeclareAndThrowException(exInvalidPointer);
	
//	for(size_t i = 0; i < n; i++)
//		values[i] = m_pdValues[i];
//}

//void daeExternalFunctionArgument::SetValues(const real_t* values, size_t n)
//{
//	if(n != N)
//		daeDeclareAndThrowException(exInvalidCall);
//	if(!m_pdValues)
//		daeDeclareAndThrowException(exInvalidPointer);
	
//	for(size_t i = 0; i < n; i++)
//		m_pdValues[i] = values[i];
//}

//real_t daeExternalFunctionArgument::operator [](size_t i) const
//{
//	if(i >= N)
//		daeDeclareAndThrowException(exInvalidCall);
//	if(!m_pdValues)
//		daeDeclareAndThrowException(exInvalidPointer);
	
//	return m_pdValues[i];
//}

//daeExternalFunctionArgumentInfo_t daeExternalFunctionArgument::GetInfo(void) const
//{
//	daeExternalFunctionArgumentInfo_t info;
//	info.m_strName = m_strName;
//	info.m_nLength = N;
//	return info;
//}

/*********************************************************************************************
	daeExternalFunction
**********************************************************************************************/
//daeExternalFunction::daeExternalFunction(void)
//{
//}

//daeExternalFunction::~daeExternalFunction(void)
//{
//}

//void daeExternalFunction::GetArguments(std::vector<daeExternalFunctionArgument_t*>& ptrarrArguments) const
//{
//}

//void daeExternalFunction::Calculate(real_t* results, size_t n)
//{
//}

//void daeExternalFunction::CalculateDerivatives(daeMatrix<real_t>& derivatives)
//{
//}

//daeExternalFunctionInfo_t daeExternalFunction::GetInfo(void) const
//{
//	return daeExternalFunctionInfo_t();
//}

/*********************************************************************************************
	daeExternalObject
**********************************************************************************************/
//daeExternalObject::daeExternalObject(void)
//{
//}

//daeExternalObject::~daeExternalObject(void)
//{
//}

//daeExternalFunction_t* daeExternalObject::CreateFunction(const std::string& strFunctionName)
//{
//	return NULL;
//}

//daeExternalObjectInfo_t daeExternalObject::GetInfo(void) const
//{
//	return daeExternalObjectInfo_t();
//}


	
	
}
}

