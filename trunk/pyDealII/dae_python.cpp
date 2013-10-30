#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include "docstrings.h"
#include <noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyDealII)
{
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");  
    
    docstring_options doc_options(true, true, false);

    class_<daeDealIIDataReporter, boost::noncopyable>("daeDealIIDataReporter", no_init)
        .def("Connect",				&daeDealIIDataReporter::Connect)
        .def("Disconnect",			&daeDealIIDataReporter::Disconnect)
        .def("IsConnected",			&daeDealIIDataReporter::IsConnected)
        .def("StartRegistration",	&daeDealIIDataReporter::StartRegistration)
        .def("RegisterDomain",		&daeDealIIDataReporter::RegisterDomain)
        .def("RegisterVariable",	&daeDealIIDataReporter::RegisterVariable)
        .def("EndRegistration",		&daeDealIIDataReporter::EndRegistration)
        .def("StartNewResultSet",	&daeDealIIDataReporter::StartNewResultSet)
        .def("EndOfData",	    	&daeDealIIDataReporter::EndOfData)
        .def("SendVariable",	  	&daeDealIIDataReporter::SendVariable)
        ;

    class_<daeConvectionDiffusion_1D, bases<daeModel>, boost::noncopyable>("daeConvectionDiffusion_1D", no_init)
        .def(init<string, optional<daeModel*, string> >(( arg("self"),
                                                          arg("name"),
                                                          arg("parentModel") = NULL,
                                                          arg("description") = ""
                                                       )))
        .add_property("DataOut", make_function(&daeConvectionDiffusion_1D::GetDataOut, return_internal_reference<>()) )
    ;

    class_<daeConvectionDiffusion_2D, bases<daeModel>, boost::noncopyable>("daeConvectionDiffusion_2D", no_init)
        .def(init<string, optional<daeModel*, string> >(( arg("self"),
                                                          arg("name"),
                                                          arg("parentModel") = NULL,
                                                          arg("description") = ""
                                                       )))
        .add_property("DataOut", make_function(&daeConvectionDiffusion_2D::GetDataOut, return_internal_reference<>()) )
    ;
    
    class_<daeConvectionDiffusion_3D, bases<daeModel>, boost::noncopyable>("daeConvectionDiffusion_3D", no_init)
        .def(init<string, optional<daeModel*, string> >(( arg("self"),
                                                          arg("name"),
                                                          arg("parentModel") = NULL,
                                                          arg("description") = ""
                                                       )))
        .add_property("DataOut", make_function(&daeConvectionDiffusion_3D::GetDataOut, return_internal_reference<>()) )
    ;
}
