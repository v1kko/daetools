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

	m_strName = strName;
	m_Unit    = units;
	m_pModel  = pModel;
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

void daeExternalFunction_t::Initialize(void)
{
	std::string strName;
	daeExecutionContext EC;
	daeExternalFunctionNode_t setup_node;
	daeExternalFunctionNodeMap_t::iterator iter;
	
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_eEquationCalculationMode	= eCalculate;

	m_mapArgumentNodes.clear();
	for(iter = m_mapSetupArgumentNodes.begin(); iter != m_mapSetupArgumentNodes.end(); iter++)
	{
		strName    = iter->first;
		setup_node = iter->second;
		
		shared_ptr<adNode>*      ad    = boost::get<shared_ptr<adNode> >     (&setup_node);
		shared_ptr<adNodeArray>* adarr = boost::get<shared_ptr<adNodeArray> >(&setup_node);
		
		if(ad)
		{
			m_mapArgumentNodes[strName] = (*ad)->Evaluate(&EC).node;
		}
		else if(adarr)
		{
			m_mapArgumentNodes[strName] = (*adarr)->Evaluate(&EC).node;
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

adouble_array daeVectorExternalFunction::operator() (void) const
{
	adouble_array tmp;
	tmp.node = boost::shared_ptr<adNodeArray>(new adVectorExternalFunctionNode(*this));
	tmp.setGatherInfo(true);
	return tmp;
}

size_t daeVectorExternalFunction::GetNumberOfResults(void) const
{
	return m_nNumberofArguments;
}

	
}
}

