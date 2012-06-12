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
	daeExternalFunction_t
**********************************************************************************************/
daeExternalFunction_t::daeExternalFunction_t(const string& strName, daeModel* pModel, const unit& units)
{
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);

    m_strShortName  = strName;
    m_Unit          = units;
    m_pModel        = pModel;
	m_pModel->AddExternalFunction(this);
}

daeExternalFunction_t::~daeExternalFunction_t(void)
{
}

void daeExternalFunction_t::SetArguments(const daeExternalFunctionArgumentMap_t& mapArguments)
{
	std::string strName;
	daeExternalFunctionArgument_t argument;
	daeExternalFunctionArgumentMap_t::const_iterator iter;
	
	m_mapSetupArgumentNodes.clear();
	for(iter = mapArguments.begin(); iter != mapArguments.end(); iter++)
	{
		strName  = iter->first;
		argument = iter->second;
		
		adouble*       ad    = boost::get<adouble>(&argument);
		adouble_array* adarr = boost::get<adouble_array>(&argument);
		
		if(ad)
		{
			m_mapSetupArgumentNodes[strName] = (*ad).node;
		}
		else if(adarr)
		{
			m_mapSetupArgumentNodes[strName] = (*adarr).node;
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}
}

const daeExternalFunctionNodeMap_t& daeExternalFunction_t::GetArgumentNodes(void) const
{
// Achtung, Achtung!!
// Returns Runtime nodes!!!
	return m_mapArgumentNodes;
}

const daeExternalFunctionNodeMap_t& daeExternalFunction_t::GetSetupArgumentNodes(void) const
{
// Achtung, Achtung!!
// Returns Setup nodes!!!
    return m_mapSetupArgumentNodes;
}

void daeExternalFunction_t::InitializeArguments(const daeExecutionContext* pExecutionContext)
{
	std::string strName;
	daeExternalFunctionNode_t setup_node;
	daeExternalFunctionNodeMap_t::iterator iter;
	
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_mapArgumentNodes.clear();
	for(iter = m_mapSetupArgumentNodes.begin(); iter != m_mapSetupArgumentNodes.end(); iter++)
	{
		strName    = iter->first;
		setup_node = iter->second;
		
		adNodePtr*      ad    = boost::get<adNodePtr >     (&setup_node);
		adNodeArrayPtr* adarr = boost::get<adNodeArrayPtr>(&setup_node);
		
		if(ad)
		{
			m_mapArgumentNodes[strName] = (*ad)->Evaluate(pExecutionContext).node;
		}
		else if(adarr)
		{
			m_mapArgumentNodes[strName] = (*adarr)->Evaluate(pExecutionContext).node;
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}
}

unit daeExternalFunction_t::GetUnits(void) const
{
	return m_Unit;
}

/*********************************************************************************************
	daeScalarExternalFunction
**********************************************************************************************/
daeScalarExternalFunction::daeScalarExternalFunction(const string& strName, daeModel* pModel, const unit& units)
                         : daeExternalFunction_t(strName, pModel, units)
{	
}

daeScalarExternalFunction::~daeScalarExternalFunction(void)
{
}

adouble daeScalarExternalFunction::Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const
{
	daeDeclareAndThrowException(exNotImplemented);
	return adouble();
}

adouble daeScalarExternalFunction::operator() (void)
{
	adouble tmp;
	tmp.node = adNodePtr(new adScalarExternalFunctionNode(this));
	tmp.setGatherInfo(true);
	return tmp;
}

/*********************************************************************************************
	daeVectorExternalFunction
**********************************************************************************************/
daeVectorExternalFunction::daeVectorExternalFunction(const string& strName, daeModel* pModel, const unit& units, size_t nNumberofArguments)
                         : daeExternalFunction_t(strName, pModel, units), m_nNumberofArguments(nNumberofArguments)
{	
}

daeVectorExternalFunction::~daeVectorExternalFunction(void)
{
}

std::vector<adouble> daeVectorExternalFunction::Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const
{
	daeDeclareAndThrowException(exNotImplemented);
	return std::vector<adouble>();
}

adouble_array daeVectorExternalFunction::operator() (void)
{
	adouble_array tmp;
	tmp.node = adNodeArrayPtr(new adVectorExternalFunctionNode(this));
	tmp.setGatherInfo(true);
	return tmp;
}

size_t daeVectorExternalFunction::GetNumberOfResults(void) const
{
	return m_nNumberofArguments;
}

	
}
}

