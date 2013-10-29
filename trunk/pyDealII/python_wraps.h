#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif

#include <string>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/python/slice.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include "../FE_DealII/convection_diffusion.h"
using namespace dae::fe_solver::convection_diffusion_dealii;

namespace daepython
{
template<int dim>
void daeConvectionDiffusion_Initialize(daeConvectionDiffusion<dim>& self, string meshFilename,
                                                                          unsigned int polynomialOrder,
                                                                          double diffusivity,
                                                                          boost::python::list lVelocity,
                                                                          double generation,
                                                                          boost::python::dict dictDirichletBC,
                                                                          boost::python::dict dictNeumannBC)
{
    std::vector<double> velocity;
    std::map<unsigned int, double> dirichletBC;
    std::map<unsigned int, double> neumannBC;

    boost::python::list keys;

    for(int i = 0; i < len(lVelocity); ++i)
    {
        double vel = boost::python::extract<double>(lVelocity[i]);

        velocity.push_back(vel);
    }

    keys = dictDirichletBC.keys();
    for(int i = 0; i < len(keys); ++i)
    {
        boost::python::object key_ = keys[i];
        boost::python::object val_ = dictDirichletBC[key_];

        unsigned int key = boost::python::extract<unsigned int>(key_);
        double value     = boost::python::extract<double>(val_);

        dirichletBC[key] = value;
    }

    keys = dictNeumannBC.keys();
    for(int i = 0; i < len(keys); ++i)
    {
        boost::python::object key_ = keys[i];
        boost::python::object val_ = dictNeumannBC[key_];

        unsigned int key = boost::python::extract<unsigned int>(key_);
        double value     = boost::python::extract<double>(val_);

        neumannBC[key] = value;
    }

    self.Initialize(meshFilename, polynomialOrder, diffusivity, velocity, generation, dirichletBC, neumannBC);
}

}

#endif
