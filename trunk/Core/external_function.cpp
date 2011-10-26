#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "nodes_array.h"
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
	daeScalarExternalFunction
**********************************************************************************************/
daeScalarExternalFunction::daeScalarExternalFunction(void)
{
}

daeScalarExternalFunction::~daeScalarExternalFunction(void)
{
}

void daeScalarExternalFunction::SetArguments(const daeExternalFunctionArgumentMap_t& mapArguments)
{
	std::string strName;
	daeExternalFunctionArgument_t argument;
	daeExternalFunctionArgumentMap_t::const_iterator iter;
	
	m_mapArgumentNodes.clear();
	for(iter = mapArguments.begin(); iter != mapArguments.end(); iter++)
	{
		strName  = iter->first;
		argument = iter->second;
		
		adouble*       ad    = boost::get<adouble>(&argument);
		adouble_array* adarr = boost::get<adouble_array>(&argument);
		
		if(ad)
		{
			m_mapArgumentNodes[strName] = (*ad).node;
		}
		else if(adarr)
		{
			m_mapArgumentNodes[strName] = (*adarr).node;
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}
}

const daeExternalFunctionNodeMap_t& daeScalarExternalFunction::GetArgumentNodes(void) const
{
	return m_mapArgumentNodes;
}

adouble daeScalarExternalFunction::Calculate(const daeExternalFunctionArgumentValueMap_t& mapValues) const
{
	daeDeclareAndThrowException(exNotImplemented);
	return adouble();
}

adouble daeScalarExternalFunction::operator() (void) const
{
	adouble tmp;
	tmp.node = boost::shared_ptr<adNode>(new adScalarExternalFunctionNode(*this));
	tmp.setGatherInfo(true);
	return tmp;
}

/*********************************************************************************************
	daeVectorExternalFunction
**********************************************************************************************/
daeVectorExternalFunction::daeVectorExternalFunction(void)
{
}

daeVectorExternalFunction::~daeVectorExternalFunction(void)
{
}

void daeVectorExternalFunction::SetArguments(const daeExternalFunctionArgumentMap_t& mapArguments)
{
	std::string strName;
	daeExternalFunctionArgument_t argument;
	daeExternalFunctionArgumentMap_t::const_iterator iter;
	
	m_mapArgumentNodes.clear();
	for(iter = mapArguments.begin(); iter != mapArguments.end(); iter++)
	{
		strName  = iter->first;
		argument = iter->second;
		
		adouble*       ad    = boost::get<adouble>(&argument);
		adouble_array* adarr = boost::get<adouble_array>(&argument);
		
		if(ad)
		{
			m_mapArgumentNodes[strName] = (*ad).node;
		}
		else if(adarr)
		{
			m_mapArgumentNodes[strName] = (*adarr).node;
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}
}

const daeExternalFunctionNodeMap_t& daeVectorExternalFunction::GetArgumentNodes(void) const
{
	return m_mapArgumentNodes;
}

std::vector<adouble> daeVectorExternalFunction::Calculate(const daeExternalFunctionArgumentValueMap_t& mapValues) const
{
	daeDeclareAndThrowException(exNotImplemented);
	return std::vector<adouble>();
}

adouble_array daeVectorExternalFunction::operator() (void) const
{
	adouble_array tmp;
	tmp.node = boost::shared_ptr<adNodeArray>(new adVectorExternalFunctionNode(*this));
	tmp.setGatherInfo(true);
	return tmp;
}

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

