#include "python_wraps.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
//#define NO_IMPORT_ARRAY
//#include <noprefix.h>
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

boost::python::object GetNumPyArrayDataReporterVariableValue(daeDataReporterVariableValue& self)
{
/* NUMPY
 	size_t i, nType;
	npy_intp dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = var.m_nNumberOfPoints;
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	memcpy(values, var.m_pValues, sizeof(real_t)*var.m_nNumberOfPoints);

	return numpy_array;
*/
    // Import numpy
    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import numpy", main_namespace);
    boost::python::object numpy = main_namespace["numpy"];

    // Create shape
    boost::python::tuple shape = boost::python::tuple(self.m_nNumberOfPoints);

    // Create a flat list of values
    boost::python::list lvalues;
    for(size_t k = 0; k < self.m_nNumberOfPoints; k++)
        lvalues.append(self.m_pValues[k]);

    // Create a flat ndarray
    boost::python::dict kwargs;
    if(typeid(real_t) == typeid(double))
        kwargs["dtype"] = numpy.attr("float64");
    else
        kwargs["dtype"] = numpy.attr("float32");
    boost::python::tuple args = boost::python::make_tuple(lvalues);
    boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

    // Return a re-shaped ndarray (not really needed here)
    return ndarray.attr("reshape")(shape);
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
boost::python::object GetNumPyArrayDataReceiverVariable(daeDataReceiverVariable& self)
{
/* NUMPY
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
*/
    size_t nDomains  = self.m_ptrarrDomains.size();
	size_t nTimeSize = self.m_ptrarrValues.size();

    // Import numpy
    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import numpy", main_namespace);
    boost::python::object numpy = main_namespace["numpy"];

    // Create shape
    boost::python::list ldimensions;
    ldimensions.append(nTimeSize); // First add time
    for(size_t i = 0; i < nDomains; i++) // Then add all domains
        ldimensions.append(self.m_ptrarrDomains[i]->m_nNumberOfPoints);
    boost::python::tuple shape = boost::python::tuple(ldimensions);

    // Create a flat list of values [times * d_1 * d_2 * ... * d_n]
    boost::python::list lvalues;
    for(size_t i = 0; i < nTimeSize; i++) // x [times]
    {
        real_t* values = self.m_ptrarrValues[i]->m_pValues;
        for(size_t k = 0; k < self.m_nNumberOfPoints; k++) // x [d_1 * d_2 * ... * d_n]
            lvalues.append(values[k]);
    }

    // Create a flat ndarray
    boost::python::dict kwargs;
    if(typeid(real_t) == typeid(double))
        kwargs["dtype"] = numpy.attr("float64");
    else
        kwargs["dtype"] = numpy.attr("float32");
    boost::python::tuple args = boost::python::make_tuple(lvalues);
    boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

    // Return a re-shaped ndarray
    return ndarray.attr("reshape")(shape);
}

boost::python::object GetTimeValuesDataReceiverVariable(daeDataReceiverVariable& self)
{
/* NUMPY
	size_t nType;
	npy_intp dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = var.m_ptrarrValues.size();
	
	python::numeric::array numpy_array(static_cast<python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	for(size_t k = 0; k < (size_t)dimensions; k++)
		values[k] = var.m_ptrarrValues[k]->m_dTime;

	return numpy_array;
*/
    // Import numpy
    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import numpy", main_namespace);
    boost::python::object numpy = main_namespace["numpy"];

    // Create shape
    boost::python::tuple shape = boost::python::make_tuple(self.m_ptrarrValues.size());

    // Create a flat list of values
    boost::python::list lvalues;
    for(size_t k = 0; k < self.m_ptrarrValues.size(); k++)
        lvalues.append(self.m_ptrarrValues[k]->m_dTime);

    // Create a flat ndarray
    boost::python::dict kwargs;
    if(typeid(real_t) == typeid(double))
        kwargs["dtype"] = numpy.attr("float64");
    else
        kwargs["dtype"] = numpy.attr("float32");
    boost::python::tuple args = boost::python::make_tuple(lvalues);
    boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

    // Return a re-shaped ndarray (not really needed here)
    return ndarray.attr("reshape")(shape);
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
