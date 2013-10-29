#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#define NO_IMPORT_ARRAY
#include <noprefix.h>
using namespace std;
using namespace boost;
using namespace boost::python;
  
namespace daepython
{
/*******************************************************
	daeDataReporterVariable
*******************************************************/
boost::python::list GetDataReporterDomains(daeDataReporterVariable& Variable)
{
	boost::python::list l;

	for(size_t i = 0; i < Variable.m_strarrDomains.size(); i++)
		l.append(Variable.m_strarrDomains[i]);
	return l;
}

boost::python::list GetDataReporterDomainPoints(daeDataReporterDomain& Domain)
{
	boost::python::list l;

	for(size_t i = 0; i < Domain.m_nNumberOfPoints; i++)
        l.append(Domain.m_arrPoints[i]);
	return l;
}

python::numeric::array GetNumPyArrayDataReporterVariableValue(daeDataReporterVariableValue& var)
{
	size_t i, nType;
	npy_intp dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = var.m_nNumberOfPoints;
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	memcpy(values, var.m_pValues, sizeof(real_t)*var.m_nNumberOfPoints);

	return numpy_array;
}

/*******************************************************
	daeDataReceiverDomain
*******************************************************/
boost::python::list GetDataReceiverDomainPoints(daeDataReceiverDomain& domain)
{
	python::list l;
	for(size_t i = 0; i < domain.m_nNumberOfPoints; i++)
        l.append(domain.m_arrPoints[i]);
	return l;
}

boost::python::list GetDataReceiverDomainCoordinates(daeDataReceiverDomain& domain)
{
    python::list l;
    for(size_t i = 0; i < domain.m_nNumberOfPoints; i++)
    {
        python::list xyz;
        xyz.append(domain.m_arrCoordinates[i].x);
        xyz.append(domain.m_arrCoordinates[i].y);
        xyz.append(domain.m_arrCoordinates[i].z);
        l.append(xyz);
    }
    return l;
}

/*******************************************************
	daeDataReceiverVariable
*******************************************************/
python::numeric::array GetNumPyArrayDataReceiverVariable(daeDataReceiverVariable& var)
{
	size_t i, nType, nDomains, nTotalSize, nTimeSize;
	npy_intp* dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	nDomains = var.m_ptrarrDomains.size();
	dimensions = new npy_intp[nDomains + 1];
	nTimeSize = var.m_ptrarrValues.size();
	nTotalSize = nTimeSize;
	dimensions[0] = nTimeSize;
	for(i = 0; i < nDomains; i++)
	{
		dimensions[i+1] = var.m_ptrarrDomains[i]->m_nNumberOfPoints;
		nTotalSize *= dimensions[i+1];
	}
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(nDomains+1, dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	for(i = 0; i < nTimeSize; i++)
	{
		memcpy(values, var.m_ptrarrValues[i]->m_pValues, sizeof(real_t)*var.m_nNumberOfPoints);
		values += var.m_nNumberOfPoints;
	}

	delete[] dimensions;
	return numpy_array;
}

python::numeric::array GetTimeValuesDataReceiverVariable(daeDataReceiverVariable& var)
{
	size_t nType;
	npy_intp dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = var.m_ptrarrValues.size();
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	for(size_t k = 0; k < (size_t)dimensions; k++)
		values[k] = var.m_ptrarrValues[k]->m_dTime;

	return numpy_array;
}

python::list GetDomainsDataReceiverVariable(daeDataReceiverVariable& var)
{
	python::list l;
	daeDataReceiverDomain* obj;
	size_t i, nDomains;

	nDomains = var.m_ptrarrDomains.size();
	
	for(i = 0; i < nDomains; i++)
	{
		obj = var.m_ptrarrDomains[i];
		l.append(boost::ref(obj));
	}
	return l;
}

/*******************************************************
	daeDataReceiverProcess
*******************************************************/
python::list GetDomainsDataReporterProcess(daeDataReceiverProcess& process)
{
	python::list l;
	daeDataReceiverDomain* obj;

	for(size_t i = 0; i < process.m_ptrarrRegisteredDomains.size(); i++)
	{
		obj = process.m_ptrarrRegisteredDomains[i];
		l.append(boost::ref(obj));
	}
	return l;
}

python::list GetVariablesDataReporterProcess(daeDataReceiverProcess& process)
{
	python::list l;
	daeDataReceiverVariable* obj;

	map<string, daeDataReceiverVariable*>::const_iterator iter;
	for(iter = process.m_ptrmapRegisteredVariables.begin(); iter != process.m_ptrmapRegisteredVariables.end(); iter++)
	{
		obj = (*iter).second;
		l.append(boost::ref(obj));
	}

	return l;
}

boost::python::dict GetDomainsAsDictDataReporterProcess(daeDataReceiverProcess& process)
{
    python::dict d;
	daeDataReceiverDomain* obj;

	for(size_t i = 0; i < process.m_ptrarrRegisteredDomains.size(); i++)
	{
		obj = process.m_ptrarrRegisteredDomains[i];
		d[obj->m_strName] = boost::ref(obj);
	}
	return d;
}

boost::python::dict GetVariablesAsDictDataReporterProcess(daeDataReceiverProcess& process)
{
    boost::python::dict d;
    map<string, daeDataReceiverVariable*>::const_iterator iter;
    
    for(iter = process.m_ptrmapRegisteredVariables.begin(); iter != process.m_ptrmapRegisteredVariables.end(); iter++)
        d[ (*iter).first ] = boost::ref( (*iter).second );

    return d;    
}

}
