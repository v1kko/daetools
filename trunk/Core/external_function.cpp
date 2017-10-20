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
            if(!(*ad).node)
                daeDeclareAndThrowException(exInvalidPointer);
            m_mapSetupArgumentNodes[strName] = (*ad).node;
        }
        else if(adarr)
        {
            if(!(*adarr).node)
                daeDeclareAndThrowException(exInvalidPointer);
            m_mapSetupArgumentNodes[strName] = (*adarr).node;
        }
        else
        {
            daeDeclareAndThrowException(exInvalidCall);
        }
    }
}

const daeExternalFunctionArgumentMap_t& daeExternalFunction_t::GetArgumentNodes(void) const
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

        adNodePtr*      ad    = boost::get<adNodePtr >    (&setup_node);
        adNodeArrayPtr* adarr = boost::get<adNodeArrayPtr>(&setup_node);

        if(ad)
        {
            adouble val = (*ad)->Evaluate(pExecutionContext);
            if(!val.node)
                daeDeclareAndThrowException(exInvalidPointer);
            m_mapArgumentNodes[strName] = val;
        }
        else if(adarr)
        {
            adouble_array val = (*adarr)->Evaluate(pExecutionContext);
            if(val.m_arrValues.empty())
                daeDeclareAndThrowException(exInvalidCall);
            m_mapArgumentNodes[strName] = val;
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
    daeLinearInterpolationFunction
**********************************************************************************************/
daeLinearInterpolationFunction::daeLinearInterpolationFunction(const string& strName,
                                                               daeModel* pModel,
                                                               const unit& units)
                         : daeScalarExternalFunction(strName, pModel, units)
{
}

daeLinearInterpolationFunction::~daeLinearInterpolationFunction(void)
{
}

void daeLinearInterpolationFunction::InitData(const std::vector<real_t>& x,
                                              const std::vector<real_t>& y,
                                              adouble& arg)
{
    if(x.size() != y.size())
    {
        daeDeclareException(exInvalidCall);
        e << "The size of x and y data arrays does not match";
        throw e;
    }

    x_arr = x;
    y_arr = y;

    daeExternalFunctionArgumentMap_t mapArguments;
    mapArguments["x"] = arg;
    this->daeExternalFunction_t::SetArguments(mapArguments);
}

adouble daeLinearInterpolationFunction::Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const
{
    daeExternalFunctionArgumentValue_t x_arg = mapValues["x"];
    adouble* x_ad = boost::get<adouble>(&x_arg);
    if(!x_ad)
        daeDeclareAndThrowException(exInvalidPointer);

    real_t x, y;

    x = x_ad->getValue();

    size_t n = x_arr.size();
    for(size_t i = 0; i < n-1; i++)
    {
        if(x >= x_arr[i] && x <= x_arr[i+1]) // Interval [x0,x1] found
        {
            real_t x0 = x_arr[i];
            real_t x1 = x_arr[i+1];
            real_t y0 = y_arr[i];
            real_t y1 = y_arr[i+1];
            y = y0 + (x - x0) * (y1 - y0) / (x1 - x0);

            return adouble(y);
        }
    }

    daeDeclareException(exInvalidCall);
    e << "daeLinearInterpolationFunction: the argument " << x << " out of bounds: [" << x_arr[0] << "," << x_arr[n-1] << "]";
    throw e;

    return adouble();
}

/*********************************************************************************************
    daeCTypesExternalFunction
**********************************************************************************************/
daeCTypesExternalFunction::daeCTypesExternalFunction(const string& strName,
                                     daeModel* pModel,
                                     const unit& units,
                                     external_lib_function fun_ptr)
                         : daeScalarExternalFunction(strName, pModel, units)
{
    m_external_lib_function = fun_ptr;
}

daeCTypesExternalFunction::~daeCTypesExternalFunction(void)
{
}

adouble daeCTypesExternalFunction::Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const
{
    if(!m_external_lib_function)
        daeDeclareAndThrowException(exInvalidPointer);

    std::vector<adouble_c>   values;
    std::vector<const char*> names;

    for(daeExternalFunctionArgumentValueMap_t::iterator iter = mapValues.begin(); iter != mapValues.end(); iter++)
    {
        adouble*              ad    = boost::get<adouble>              (&iter->second);
        std::vector<adouble>* adarr = boost::get<std::vector<adouble> >(&iter->second);

        names.push_back(iter->first.c_str());

        if(ad)
        {
            adouble_c a(ad->getValue(), ad->getDerivative());

            values.push_back(a);
        }
        else if(adarr)
        {
            daeDeclareAndThrowException(exNotImplemented);
        }
        else
        {
            daeDeclareAndThrowException(exInvalidCall);
        }
    }

    //printf("no_arguments = %d\n", values.size());
    //printf("names[0] = %s\n", names[0]);
    //printf("values[0] = %f\n", values[0]);

    adouble_c res = (*m_external_lib_function)(values.data(), names.data(), values.size());

    return adouble(res.value, res.derivative);
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

