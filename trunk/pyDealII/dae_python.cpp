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

    class_<diffusion::daeDiffusion_1D, bases<daeModel>, boost::noncopyable>("daeDiffusion_1D", no_init)
        .def(init<string, optional<daeModel*, string> >(( arg("self"),
                                                          arg("name"),
                                                          arg("parentModel") = NULL,
                                                          arg("description") = ""
                                                       )))
        .def("Initialize", &diffusion::daeDiffusion_1D::Initialize,
                           ( arg("self"), arg("meshFilename"), arg("diffusivity"), arg("polynomialOrder")/*, arg("dirichletBC"), arg("neumanBC")*/ ) )
        ;

    class_<diffusion::daeDiffusion_2D, bases<daeModel>, boost::noncopyable>("daeDiffusion_2D", no_init)
        .def(init<string, optional<daeModel*, string> >(( arg("self"),
                                                          arg("name"),
                                                          arg("parentModel") = NULL,
                                                          arg("description") = ""
                                                       )))
        .def("Initialize", &diffusion::daeDiffusion_2D::Initialize,
                           ( arg("self"), arg("meshFilename"), arg("diffusivity"), arg("polynomialOrder")/*, arg("dirichletBC"), arg("neumanBC")*/ ) )
        ;
    
    class_<diffusion::daeDiffusion_3D, bases<daeModel>, boost::noncopyable>("daeDiffusion_3D", no_init)
        .def(init<string, optional<daeModel*, string> >(( arg("self"),
                                                          arg("name"),
                                                          arg("parentModel") = NULL,
                                                          arg("description") = ""
                                                       )))
        .def("Initialize", &diffusion::daeDiffusion_3D::Initialize,
                           ( arg("self"), arg("meshFilename"), arg("diffusivity"), arg("polynomialOrder")/*, arg("dirichletBC"), arg("neumanBC")*/ ) )
        ;
}

//int main(int argc, char *argv[])
//{
//    daeHelmholtzProblem_2D h(1, "fe");
//}
