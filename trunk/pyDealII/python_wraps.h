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

#include "../FE_DealII/dealii_fe_object.h"
#include "../FE_DealII/dealii_fe_system.h"
#include "../FE_DealII/dealii_iterators.h"
#include "../dae_develop.h"
using namespace dae::fe_solver;

namespace daepython
{
template<typename ITEM>
boost::python::list getListFromVectorByValue(const std::vector<ITEM>& arrItems)
{
    boost::python::list l;

    for(size_t i = 0; i < arrItems.size(); i++)
        l.append(arrItems[i]);

    return l;
}

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






unsigned int Tensor_2_1D_rank(Tensor_2_1D& self)
{
    return Tensor_2_1D::rank;
}
unsigned int Tensor_2_1D_dimension(Tensor_2_1D& self)
{
    return Tensor_2_1D::dimension;
}
unsigned int Tensor_2_1D_n_independent_components(Tensor_2_1D& self)
{
    return Tensor_2_1D::n_independent_components;
}

unsigned int Tensor_2_2D_rank(Tensor_2_2D& self)
{
    return Tensor_2_2D::rank;
}
unsigned int Tensor_2_2D_dimension(Tensor_2_2D& self)
{
    return Tensor_2_2D::dimension;
}
unsigned int Tensor_2_2D_n_independent_components(Tensor_2_2D& self)
{
    return Tensor_2_2D::n_independent_components;
}

unsigned int Tensor_2_3D_rank(Tensor_2_3D& self)
{
    return Tensor_2_3D::rank;
}
unsigned int Tensor_2_3D_dimension(Tensor_2_3D& self)
{
    return Tensor_2_3D::dimension;
}
unsigned int Tensor_2_3D_n_independent_components(Tensor_2_3D& self)
{
    return Tensor_2_3D::n_independent_components;
}

Tensor_1_1D Tensor_2_1D_getitem(Tensor_2_1D& self, size_t i)
{
    return self[i];
}
Tensor_1_2D Tensor_2_2D_getitem(Tensor_2_2D& self, size_t i)
{
    return self[i];
}
Tensor_1_3D Tensor_2_3D_getitem(Tensor_2_3D& self, size_t i)
{
    return self[i];
}

void Tensor_2_1D_setitem(Tensor_2_1D& self, size_t i, const Tensor_1_1D& value)
{
    self[i] = value;
}
void Tensor_2_2D_setitem(Tensor_2_2D& self, size_t i, const Tensor_1_2D& value)
{
    self[i] = value;
}
void Tensor_2_3D_setitem(Tensor_2_3D& self, size_t i, const Tensor_1_3D& value)
{
    self[i] = value;
}


string Tensor_2_1D_str(Tensor_2_1D& self)
{
    return (boost::format("[[%f], [%f]]") % self[0][0] % self[0][1]).str();
}
string Tensor_2_2D_str(Tensor_2_2D& self)
{
    return (boost::format("[[%f, %f], [%f, %f]]")  % self[0][0] % self[0][1]
                                                   % self[1][0] % self[1][1]).str();
}
string Tensor_2_3D_str(Tensor_2_3D& self)
{
    return (boost::format("[[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]") % self[0][0] % self[0][1] % self[0][2]
                                                                        % self[1][0] % self[1][1] % self[1][2]
                                                                        % self[1][0] % self[1][1] % self[1][2]).str();
}

string Tensor_2_1D_repr(Tensor_2_1D& self)
{
    return (boost::format("Tensor<2,1,double>([[%f], [%f]])") % self[0][0] % self[0][1]).str();
}
string Tensor_2_2D_repr(Tensor_2_2D& self)
{
    return (boost::format("Tensor<2,2,double>([[%f, %f], [%f, %f]])")  % self[0][0] % self[0][1]
                                                                       % self[1][0] % self[1][1]).str();
}
string Tensor_2_3D_repr(Tensor_2_3D& self)
{
    return (boost::format("Tensor<2,3,double>([[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]])") % self[0][0] % self[0][1] % self[0][2]
                                                                                            % self[1][0] % self[1][1] % self[1][2]
                                                                                            % self[2][0] % self[2][1] % self[2][2]).str();
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
class Function_wrapper : public Function<dim>,
                         public boost::python::wrapper< Function<dim> >
{
public:
    Function_wrapper(unsigned int n_components = 1) : Function<dim>(n_components)
    {

    }

    virtual ~Function_wrapper()
    {

    }

    unsigned int dimension() const
    {
        return Function<dim>::dimension;
    }

    unsigned int n_components() const
    {
        return Function<dim>::n_components;
    }

    double value(const Point<dim> &p, const unsigned int component = 0) const
    {
        boost::python::override f = this->get_override("value");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'value' must be implemented in the python Function_nD-derived class";
            throw e;
        }
        return f(p, component);
    }

    void vector_value(const Point<dim> &p, Vector<double>& values) const
    {
        boost::python::override f = this->get_override("vector_value");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'vector_value' must be implemented in the python Function_nD-derived class";
            throw e;
        }

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
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'gradient' must be implemented in the python class";
            throw e;
        }

        return f(p, component);
    }

    void vector_gradient(const Point<dim> &p, std::vector<Tensor<1,dim> > &gradients) const
    {
        boost::python::override f = this->get_override("vector_gradient");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'vector_gradient' must be implemented in the python Function_nD-derived class";
            throw e;
        }

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
class dealiiFiniteElementEquationWrapper : public dealiiFiniteElementEquation<dim>,
                                           public boost::python::wrapper< dealiiFiniteElementEquation<dim> >
{
public:
    dealiiFiniteElementEquationWrapper(const std::string&       variableName,
                                       const std::string&       variableDescription,
                                       unsigned int             multiplicity,
                                       const feExpression<dim>& elementMatrix,
                                       const feExpression<dim>& elementMatrix_dt,
                                       const feExpression<dim>& elementRHS,
                                       boost::python::dict      dictDirichletBC,
                                       boost::python::dict      dictNeumannBC)
    {
        boost::python::list keys;

        this->m_elementMatrix             = elementMatrix;
        this->m_elementMatrix_dt          = elementMatrix_dt;
        this->m_elementRHS                = elementRHS;
        //this->m_elementBoundary           = ;
        this->m_strVariableName           = variableName;
        this->m_strVariableDescription    = variableDescription;
        this->m_nMultiplicity             = multiplicity;

        keys = dictDirichletBC.keys();
        for(int i = 0; i < len(keys); ++i)
        {
            boost::python::object key_ = keys[i];
            boost::python::object val_ = dictDirichletBC[key_];

            unsigned int         key = boost::python::extract<unsigned int>(key_);
            const Function<dim>* fn  = boost::python::extract<const Function<dim>*>(val_);

            this->m_dirichletBC[key] = fn;
        }

        keys = dictNeumannBC.keys();
        for(int i = 0; i < len(keys); ++i)
        {
            boost::python::object key_ = keys[i];
            boost::python::object val_ = dictNeumannBC[key_];

            unsigned int         key = boost::python::extract<unsigned int>(key_);
            boost::python::tuple t   = boost::python::extract<boost::python::tuple>(val_);

            const Function<dim>* fn  = boost::python::extract<const Function<dim>*>(t[0]);
            dealiiFluxType       ft  = boost::python::extract<dealiiFluxType>(t[1]);

            this->m_neumannBC[key] = std::pair<const Function<dim>*, dealiiFluxType>(fn, ft);
        }
    }

    ~dealiiFiniteElementEquationWrapper()
    {
    }

    static
    dealiiFiniteElementEquationWrapper<dim>* ConvectionDiffusionEquation(const std::string&  variableName,
                                                                         const std::string&  variableDescription,
                                                                         boost::python::dict dictDirichletBC,
                                                                         boost::python::dict dictNeumannBC)
    {
        /* Available functions:
         * - feExpression
         * - constant
         * - phi, dphi, d2phi
         * - phi_vec, dphi_vec, d2phi_vec
         * - JxW, xyz, normal
         * - fvalue, fgrad
         * - feExpression.sqrt, feExpression.exp, feExpression.log, feExpression.log10, feExpression.abs
         *   feExpression.sin, feExpression.cos, feExpression.tan, feExpression.asin, feExpression.acos, feExpression.atan */

        feExpression<dim> matrix    = (dphi<dim>(variableName, fe_i, fe_q) * dphi<dim>(variableName, fe_j, fe_q)) * function_value<dim>("Diffusivity", xyz<dim>(fe_q)) * JxW<dim>(fe_q);
        feExpression<dim> matrix_dt = phi<dim>(variableName, fe_i, fe_q) * phi<dim>(variableName, fe_j, fe_q) * JxW<dim>(fe_q);
        feExpression<dim> rhs       = phi<dim>(variableName, fe_i, fe_q) * function_value<dim>("Generation", xyz<dim>(fe_q)) * JxW<dim>(fe_q);

        return new dealiiFiniteElementEquationWrapper<dim>(variableName, variableDescription, 1, matrix, matrix_dt, rhs, dictDirichletBC, dictNeumannBC);
    }

};

template<int dim>
class dealiiFiniteElementSystemWrapper : public dealiiFiniteElementSystem<dim>,
                                         public boost::python::wrapper< dealiiFiniteElementSystem<dim> >
{
public:
    dealiiFiniteElementSystemWrapper(std::string                                meshFilename,
                                     unsigned int                               polynomialOrder,
                                     const Quadrature<dim>&                     quadrature,
                                     const Quadrature<dim-1>&                   faceQuadrature,
                                     boost::python::dict                        dictFunctions,
                                     boost::python::list                        listEquations)
    {
        boost::python::list keys;
        std::map<std::string,  const Function<dim>*> mapFunctions;
        std::vector< dealiiFiniteElementEquation<dim>* > arrEquations;

        keys = dictFunctions.keys();
        for(int i = 0; i < len(keys); ++i)
        {
            boost::python::object key_ = keys[i];
            boost::python::object val_ = dictFunctions[key_];

            std::string          key = boost::python::extract<std::string>(key_);
            const Function<dim>* fn  = boost::python::extract<const Function<dim>*>(val_);

            mapFunctions[key] = fn;
        }

        for(int i = 0; i < len(listEquations); ++i)
        {
            dealiiFiniteElementEquation<dim>* eq = boost::python::extract< dealiiFiniteElementEquation<dim>* >(listEquations[i]);
            arrEquations.push_back(eq);
        }

        this->Initialize(meshFilename, polynomialOrder, quadrature, faceQuadrature, mapFunctions, arrEquations);
    }

    ~dealiiFiniteElementSystemWrapper()
    {
    }

public:
    void AssembleSystem()
    {
        if(boost::python::override f = this->get_override("AssembleSystem"))
            f();
        else
            this->dealiiFiniteElementSystem<dim>::AssembleSystem();
    }
    void def_AssembleSystem()
    {
        this->dealiiFiniteElementSystem<dim>::AssembleSystem();
    }

    void ReAssembleSystem()
    {
        if(boost::python::override f = this->get_override("ReAssembleSystem"))
            f();
        else
            this->dealiiFiniteElementSystem<dim>::ReAssembleSystem();
    }
    void def_ReAssembleSystem()
    {
        this->dealiiFiniteElementSystem<dim>::ReAssembleSystem();
    }

    bool NeedsReAssembling()
    {
        if(boost::python::override f = this->get_override("NeedsReAssembling"))
            return f();
        else
            return this->dealiiFiniteElementSystem<dim>::NeedsReAssembling();
    }
    bool def_NeedsReAssembling()
    {
        return this->dealiiFiniteElementSystem<dim>::NeedsReAssembling();
    }

    boost::python::list GetRowIndices(unsigned int row)
    {
        std::vector<unsigned int> narrIndices;
        this->RowIndices(row, narrIndices);
        return getListFromVectorByValue(narrIndices);
    }
};


}

#endif
