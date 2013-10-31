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
using namespace dae::fe_solver;
using namespace dae::fe_solver::convection_diffusion_dealii;

namespace daepython
{
unsigned int Tensor_1_1D_rank(Tensor_1_1D& self)
{
    return Tensor_1_1D::rank;
}
unsigned int Tensor_1_1D_dimension(Tensor_1_1D& self)
{
    return Tensor_1_1D::dimension;
}
unsigned int Tensor_1_1D_n_independent_components(Tensor_1_1D& self)
{
    return Tensor_1_1D::n_independent_components;
}

unsigned int Tensor_1_2D_rank(Tensor_1_2D& self)
{
    return Tensor_1_2D::rank;
}
unsigned int Tensor_1_2D_dimension(Tensor_1_2D& self)
{
    return Tensor_1_2D::dimension;
}
unsigned int Tensor_1_2D_n_independent_components(Tensor_1_2D& self)
{
    return Tensor_1_2D::n_independent_components;
}

unsigned int Tensor_1_3D_rank(Tensor_1_3D& self)
{
    return Tensor_1_3D::rank;
}
unsigned int Tensor_1_3D_dimension(Tensor_1_3D& self)
{
    return Tensor_1_3D::dimension;
}
unsigned int Tensor_1_3D_n_independent_components(Tensor_1_3D& self)
{
    return Tensor_1_3D::n_independent_components;
}

double Tensor_1_1D_getitem(Tensor_1_1D& self, size_t i)
{
    return self[i];
}
double Tensor_1_2D_getitem(Tensor_1_2D& self, size_t i)
{
    return self[i];
}
double Tensor_1_3D_getitem(Tensor_1_3D& self, size_t i)
{
    return self[i];
}

void Tensor_1_1D_setitem(Tensor_1_1D& self, size_t i, double value)
{
    self[i] = value;
}
void Tensor_1_2D_setitem(Tensor_1_2D& self, size_t i, double value)
{
    self[i] = value;
}
void Tensor_1_3D_setitem(Tensor_1_3D& self, size_t i, double value)
{
    self[i] = value;
}


string Tensor_1_1D_str(Tensor_1_1D& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "(" << self[0] << ")";
    return s.str();
}
string Tensor_1_2D_str(Tensor_1_2D& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "(" << self[0] << ", " << self[1]<< ")";
    return s.str();
}
string Tensor_1_3D_str(Tensor_1_3D& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "(" << self[0] << ", " << self[1] << ", " << self[2] << ")";
    return s.str();
}

string Tensor_1_1D_repr(Tensor_1_1D& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "Tensor<1,1,double>(" << self[0] << ")";
    return s.str();
}
string Tensor_1_2D_repr(Tensor_1_2D& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "Tensor<1,2,double>(" << self[0] << ", " << self[1] << ")";
    return s.str();
}
string Tensor_1_3D_repr(Tensor_1_3D& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "Tensor<1,3,double>(" << self[0] << ", " << self[1] << ", " << self[2] << ")";
    return s.str();
}

string Point_1D_repr(Point_1D& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "Point<1,double>(x=" << self[0] << ")";
    return s.str();
}
string Point_2D_repr(Point_2D& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "Point<2,double>(x=" << self[0] << ", y=" << self[1] << ")";
    return s.str();
}
string Point_3D_repr(Point_3D& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "Point<3,double>(x=" << self[0] << ", y=" << self[1] << ", z=" << self[2] << ")";
    return s.str();
}

double Point_1D_x(Point_1D& self)
{
    return self[0];
}

double Point_2D_x(Point_2D& self)
{
    return self[0];
}
double Point_2D_y(Point_2D& self)
{
    return self[1];
}

double Point_3D_x(Point_3D& self)
{
    return self[0];
}
double Point_3D_y(Point_3D& self)
{
    return self[1];
}
double Point_3D_z(Point_3D& self)
{
    return self[2];
}


// Indices must be int because python complains about wrong argument types:
// it cannot convert from int to unsigned int for some reasons
double Vector_getitem(Vector<double>& self, unsigned int i)
{
    return self[i];
}
void Vector_set(Vector<double>& self, unsigned int i, double value)
{
    self[i] = value;
}
void Vector_add(Vector<double>& self, unsigned int i, double value)
{
    self[i] += value;
}


double FullMatrix_getitem(FullMatrix<double>& self, unsigned int i, unsigned int j)
{
    return self(i,j);
}
void FullMatrix_set(FullMatrix<double>& self, unsigned int i, unsigned int j, double value)
{
    self(i,j) = value;
}
void FullMatrix_add(FullMatrix<double>& self, unsigned int i, unsigned int j, double value)
{
    self(i,j) += value;
}

double SparseMatrix_getitem(SparseMatrix<double>& self, unsigned int i, unsigned int j)
{
    return self(i,j);
}
double SparseMatrix_el(SparseMatrix<double>& self, unsigned int i, unsigned int j)
{
    return self.el(i,j);
}
void SparseMatrix_set(SparseMatrix<double>& self, unsigned int i, unsigned int j, double value)
{
    self.set(i, j, value);
}
void SparseMatrix_add(SparseMatrix<double>& self, unsigned int i, unsigned int j, double value)
{
    self.add(i, j, value);
}


template<int dim>
class Function_wrapper : public dealiiFunction<dim>,
                         public boost::python::wrapper< dealiiFunction<dim> >
{
public:
    Function_wrapper(unsigned int n_components = 1) : dealiiFunction<dim>(n_components)
    {

    }

    virtual ~Function_wrapper()
    {

    }

    unsigned int dimension() const
    {
        return dealiiFunction<dim>::dimension;
    }

    unsigned int n_components() const
    {
        return dealiiFunction<dim>::n_components;
    }

    double value(const Point<dim> &p, const unsigned int component = 0) const
    {
        boost::python::override f = this->get_override("value");
        return f(p, component);
    }

    void vector_value(const Point<dim> &p, Vector<double>& values) const
    {
        boost::python::override f = this->get_override("vector_value");
        boost::python::list lvalues = f(p);

        boost::python::ssize_t i, n;
        n = boost::python::len(lvalues);
        if(n != Function<dim>::n_components)
        {
            daeDeclareException(exInvalidCall);
            e << "The number of items (" << n << ") returned from the Function<" << Function<dim>::dimension
              << ">::vector_value call must be " << Function<dim>::n_components;
            throw e;
        }
        for(i = 0; i < n; i++)
            values[i] = boost::python::extract<double>(lvalues[i]);
    }

    Tensor<1,dim> gradient(const Point<dim> &p, const unsigned int component = 0) const
    {
        boost::python::override f = this->get_override("gradient");
        return f(p, component);
    }

    void vector_gradient(const Point<dim> &p, std::vector<Tensor<1,dim> > &gradients) const
    {
        boost::python::override f = this->get_override("vector_gradient");
        boost::python::list lgradients = f(p);

        boost::python::ssize_t i, n;
        n = boost::python::len(lgradients);
        if(n != Function<dim>::n_components)
        {
            daeDeclareException(exInvalidCall);
            e << "The number of items (" << n << ") returned from the Function<" << Function<dim>::dimension
              << ">::vector_gradient call must be " << Function<dim>::n_components;
            throw e;
        }
        for(i = 0; i < n; i++)
            gradients[i] = boost::python::extract< Tensor<1,dim> >(lgradients[i]);
    }
};
typedef Function_wrapper<1> Function_wrapper_1D;
typedef Function_wrapper<2> Function_wrapper_2D;
typedef Function_wrapper<3> Function_wrapper_3D;


template<int dim>
daeConvectionDiffusion<dim>* daeConvectionDiffusion__init__(std::string                 strName,
                                                            daeModel*                   pModel,
                                                            std::string                 strDescription,
                                                            std::string                 meshFilename,
                                                            string                      quadratureFormula,
                                                            unsigned int                polynomialOrder,
                                                            string                      outputDirectory,
                                                            boost::python::dict         dictFunctions,
                                                            boost::python::dict         dictDirichletBC,
                                                            boost::python::dict         dictNeumannBC)
{
    boost::python::list keys;
    std::map<unsigned int, const dealiiFunction<dim>*> mapDirichletBC;
    std::map<unsigned int, const dealiiFunction<dim>*> mapNeumannBC;
    std::map<std::string,  const dealiiFunction<dim>*> mapFunctions;

    keys = dictFunctions.keys();
    for(int i = 0; i < len(keys); ++i)
    {
        boost::python::object key_ = keys[i];
        boost::python::object val_ = dictFunctions[key_];

        std::string                key = boost::python::extract<std::string>(key_);
        const dealiiFunction<dim>* fn  = boost::python::extract<const dealiiFunction<dim>*>(val_);

        mapFunctions[key] = fn;
    }

    keys = dictDirichletBC.keys();
    for(int i = 0; i < len(keys); ++i)
    {
        boost::python::object key_ = keys[i];
        boost::python::object val_ = dictDirichletBC[key_];

        unsigned int               key = boost::python::extract<unsigned int>(key_);
        const dealiiFunction<dim>* fn  = boost::python::extract<const dealiiFunction<dim>*>(val_);

        mapDirichletBC[key] = fn;
    }

    keys = dictNeumannBC.keys();
    for(int i = 0; i < len(keys); ++i)
    {
        boost::python::object key_ = keys[i];
        boost::python::object val_ = dictNeumannBC[key_];

        unsigned int               key = boost::python::extract<unsigned int>(key_);
        const dealiiFunction<dim>* fn  = boost::python::extract<const dealiiFunction<dim>*>(val_);

        mapNeumannBC[key] = fn;
    }

    return new daeConvectionDiffusion<dim>(strName, pModel, strDescription, meshFilename, quadratureFormula, polynomialOrder,
                                           outputDirectory, mapFunctions, mapDirichletBC, mapNeumannBC);
}

}

#endif
