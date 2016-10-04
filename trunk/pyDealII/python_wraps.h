#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif

#include <string>
#include <boost/python.hpp>
//#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/python/slice.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include "../FE_DealII/dealii_fe_system.h"
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

template<class Number>
Number Tensor_1_1D_getitem(Tensor<1,1,Number>& self, size_t i)
{
    return self[i];
}
template<class Number>
Number Tensor_1_2D_getitem(Tensor<1,2,Number>& self, size_t i)
{
    return self[i];
}
template<class Number>
Number Tensor_1_3D_getitem(Tensor<1,3,Number>& self, size_t i)
{
    return self[i];
}

template<class Number>
void Tensor_1_1D_setitem(Tensor<1,1,Number>& self, size_t i, Number value)
{
    self[i] = value;
}
template<class Number>
void Tensor_1_2D_setitem(Tensor<1,2,Number>& self, size_t i, Number value)
{
    self[i] = value;
}
template<class Number>
void Tensor_1_3D_setitem(Tensor<1,3,Number>& self, size_t i, Number value)
{
    self[i] = value;
}


template<class Number>
string Tensor_1_1D_str(Tensor<1,1,Number>& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "(" << self[0] << ")";
    return s.str();
}
template<class Number>
string Tensor_1_2D_str(Tensor<1,2,Number>& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "(" << self[0] << ", " << self[1]<< ")";
    return s.str();
}
template<class Number>
string Tensor_1_3D_str(Tensor<1,3,Number>& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "(" << self[0] << ", " << self[1] << ", " << self[2] << ")";
    return s.str();
}

template<class Number>
string Tensor_1_1D_repr(Tensor<1,1,Number>& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "Tensor<1,1>(" << self[0] << ")";
    return s.str();
}
template<class Number>
string Tensor_1_2D_repr(Tensor<1,2,Number>& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "Tensor<1,2>(" << self[0] << ", " << self[1] << ")";
    return s.str();
}
template<class Number>
string Tensor_1_3D_repr(Tensor<1,3,Number>& self)
{
    std::stringstream s(std::ios_base::out|std::ios_base::in);
    s << "Tensor<1,3>(" << self[0] << ", " << self[1] << ", " << self[2] << ")";
    return s.str();
}

template<class Number>
Tensor<1,1,Number>* Tensor_2_1D_getitem(Tensor<2,1,Number>& self, size_t i)
{
    return &self[i];
}
template<class Number>
Tensor<1,2,Number>* Tensor_2_2D_getitem(Tensor<2,2,Number>& self, size_t i)
{
    return &self[i];
}
template<class Number>
Tensor<1,3,Number>* Tensor_2_3D_getitem(Tensor<2,3,Number>& self, size_t i)
{
    return &self[i];
}

template<class Number>
void Tensor_2_1D_setitem(Tensor<2,1,Number>& self, size_t i, const Tensor<1,1,Number>& value)
{
    self[i] = value;
}
template<class Number>
void Tensor_2_2D_setitem(Tensor<2,2,Number>& self, size_t i, const Tensor<1,2,Number>& value)
{
    self[i] = value;
}
template<class Number>
void Tensor_2_3D_setitem(Tensor<2,3,Number>& self, size_t i, const Tensor<1,3,Number>& value)
{
    self[i] = value;
}

template<int rank, int dim, typename Number>
std::string tensor__str__(const Tensor<rank,dim,Number>& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

template<int rank, int dim, typename Number>
std::string tensor__repr__(const Tensor<rank,dim,Number>& t)
{
    std::stringstream ss;
    ss << "Tensor<" << rank << "," << dim << ">(" << t << ")";
    return ss.str();
}

/*
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
*/

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
class Function_wrapper : public Function<dim,double>,
                         public boost::python::wrapper< Function<dim,double> >
{
public:
    Function_wrapper(unsigned int n_components = 1) : Function<dim,double>(n_components)
    {
    }

    virtual ~Function_wrapper()
    {
    }

    double value(const Point<dim> &p, const unsigned int component = 0) const
    {
        boost::python::override f = this->get_override("value");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'value' not implemented in the python Function_" << dim << "D-derived class";
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
            e << "The function 'vector_value' not implemented in the python Function_" << dim << "D-derived class";
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
            e << "The function 'gradient' not implemented in the python Function_" << dim << "D-derived class";
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
            e << "The function 'vector_gradient' not implemented in the python Function_" << dim << "D-derived class";
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
ConstantFunction<dim>* ConstantFunction_init(boost::python::list l_values)
{
    std::vector<double> values;

    for(boost::python::ssize_t i = 0; i < boost::python::len(l_values); i++)
    {
        double item_value = boost::python::extract<double>(l_values[i]);
        values.push_back(item_value);
    }

    return new ConstantFunction<dim>(values);
}

template<int rank, int dim>
class TensorFunction_wrapper : public TensorFunction<rank,dim,double>,
                               public boost::python::wrapper< TensorFunction<rank,dim,double> >
{
public:
    TensorFunction_wrapper() : TensorFunction<rank,dim,double>()
    {
    }

    virtual ~TensorFunction_wrapper()
    {
    }

    Tensor<rank,dim,double> value(const Point<dim> &p) const
    {
        boost::python::override f = this->get_override("value");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'value' not implemented in the python TensorFunction_" << dim << "D-derived class";
            throw e;
        }
        return f(p);
    }

    Tensor<rank+1,dim,double> gradient(const Point<dim> &p) const
    {
        boost::python::override f = this->get_override("gradient");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'gradient' not implemented in the python TensorFunction_" << dim << "D-derived class";
            throw e;
        }
        return f(p);
    }
};


template<int dim>
class adoubleFunction_wrapper : public Function<dim,adouble>,
                                public boost::python::wrapper< Function<dim,adouble> >
{
public:
    adoubleFunction_wrapper(unsigned int n_components = 1) : Function<dim,adouble>(n_components)
    {
    }

    virtual ~adoubleFunction_wrapper()
    {
    }

    adouble value(const Point<dim> &p, const unsigned int component = 0) const
    {
        boost::python::override f = this->get_override("value");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'value' not implemented in the python adoubleFunction_" << dim << "D-derived class";
            throw e;
        }
        return f(p, component);
    }

    void vector_value(const Point<dim> &p, Vector<adouble>& values) const
    {
        boost::python::override f = this->get_override("vector_value");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'vector_value' not implemented in the python adoubleFunction_" << dim << "D-derived class";
            throw e;
        }

        boost::python::list lvalues = f(p);

        boost::python::ssize_t i, n;
        n = boost::python::len(lvalues);
        if(n != Function<dim,adouble>::n_components)
        {
            daeDeclareException(exInvalidCall);
            e << "The number of items (" << n << ") returned from the adoubleFunction<" << dim
              << ">::vector_value call must be " << Function<dim,adouble>::n_components;
            throw e;
        }
        for(i = 0; i < n; i++)
            values[i] = boost::python::extract<adouble>(lvalues[i]);
    }

    Tensor<1,dim,adouble> gradient(const Point<dim> &p, const unsigned int component = 0) const
    {
        boost::python::override f = this->get_override("gradient");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'gradient' not implemented in the python adoubleFunction_" << dim << "D-derived class";
            throw e;
        }

        return f(p, component);
    }

    void vector_gradient(const Point<dim> &p, std::vector<Tensor<1,dim,adouble> > &gradients) const
    {
        boost::python::override f = this->get_override("vector_gradient");
        if(!f)
        {
            daeDeclareException(exInvalidCall);
            e << "The function 'vector_gradient' not implemented in the python adoubleFunction_" << dim << "D-derived class";
            throw e;
        }

        boost::python::list lgradients = f(p);

        boost::python::ssize_t i, n;
        n = boost::python::len(lgradients);
        if(n != Function<dim,adouble>::n_components)
        {
            daeDeclareException(exInvalidCall);
            e << "The number of items (" << n << ") returned from the adoubleFunction<" << dim
              << ">::vector_gradient call must be " << Function<dim,adouble>::n_components;
            throw e;
        }
        for(i = 0; i < n; i++)
            gradients[i] = boost::python::extract< Tensor<1,dim,adouble> >(lgradients[i]);
    }
};
typedef adoubleFunction_wrapper<1> adoubleFunction_wrapper_1D;
typedef adoubleFunction_wrapper<2> adoubleFunction_wrapper_2D;
typedef adoubleFunction_wrapper<3> adoubleFunction_wrapper_3D;

template<int dim>
ConstantFunction<dim,adouble>* adoubleConstantFunction_init(boost::python::list l_values)
{
    std::vector<adouble> values;

    for(boost::python::ssize_t i = 0; i < boost::python::len(l_values); i++)
    {
        adouble item_value = boost::python::extract<adouble>(l_values[i]);
        values.push_back(item_value);
    }

    return new ConstantFunction<dim,adouble>(values);
}


template<int dim>
class dealiiFiniteElementWeakFormWrapper : public dealiiFiniteElementWeakForm<dim>,
                                           public boost::python::wrapper< dealiiFiniteElementWeakForm<dim> >
{
public:
    dealiiFiniteElementWeakFormWrapper(const feExpression<dim>& Aij, // Local contribution to the stiffness matrix
                                       const feExpression<dim>& Mij, // Local contribution to the mass matrix
                                       const feExpression<dim>& Fi,  // Local contribution to the load vector
                                       boost::python::dict      dictFaceAij              = boost::python::dict(),
                                       boost::python::dict      dictFaceFi               = boost::python::dict(),
                                       boost::python::dict      dictFunctions            = boost::python::dict(),
                                       boost::python::dict      dictFunctionsDirichletBC = boost::python::dict(),
                                       boost::python::dict      dictBoundaryIntegrals    = boost::python::dict())
    {
        boost::python::list keys;

	// These terms get copied using the copy constructor and do need to be stored internally
        this->m_Aij = Aij;
        this->m_Mij = Mij;
        this->m_Fi  = Fi;

        // Keep these objects so they do not go out of scope and get destroyed while still in use by daetools
        this->m_dictFaceAij              = dictFaceAij;
        this->m_dictFaceFi               = dictFaceFi;
        this->m_dictFunctions            = dictFunctions;
        this->m_dictFunctionsDirichletBC = dictFunctionsDirichletBC;
        this->m_dictBoundaryIntegrals    = dictBoundaryIntegrals;

        keys = dictBoundaryIntegrals.keys();
        for(int i = 0; i < len(keys); i++)
        {
            boost::python::object key_  = keys[i];
            boost::python::list   vals_ = boost::python::extract<boost::python::list>(dictBoundaryIntegrals[key_]);

            unsigned int key = boost::python::extract<unsigned int>(key_);

            std::vector< std::pair<adouble, feExpression<dim> > >  pairs;
            for(int k = 0; k < len(vals_); k++)
            {
                boost::python::tuple  t_var_expr = boost::python::extract<boost::python::tuple>(vals_[k]);
                boost::python::object o_variable   = t_var_expr[0];
                boost::python::object o_expression = t_var_expr[1];

                adouble           advar  = boost::python::extract<adouble>(o_variable);
                feExpression<dim> expr  = boost::python::extract< feExpression<dim> >(o_expression);
                
                if(!advar.node || !dynamic_cast<adSetupVariableNode*>(advar.node.get()))
                    throw std::runtime_error(std::string("Invalid variable node in the boundary integrals dictionary (must be adSetupVariableNode)"));

                pairs.push_back(std::pair< adouble, feExpression<dim> >(advar, expr));
            }

            if(pairs.size() > 0)
                this->m_mapBoundaryIntegrals[key] = pairs;
        }

        keys = dictFunctionsDirichletBC.keys();
        for(int i = 0; i < len(keys); i++)
        {
            boost::python::object key_  = keys[i];
            boost::python::list   vals_ = boost::python::extract<boost::python::list>(dictFunctionsDirichletBC[key_]);

            unsigned int key = boost::python::extract<unsigned int>(key_);

            //std::vector< std::pair<std::string, const Function<dim,double>*> >  bcs_double;
            std::vector< std::pair<std::string, const Function<dim,adouble>*> > bcs_adouble;
            for(int k = 0; k < len(vals_); k++)
            {
                boost::python::tuple  t_var_fn = boost::python::extract<boost::python::tuple>(vals_[k]);
                boost::python::object o_var = t_var_fn[0];
                boost::python::object o_fn  = t_var_fn[1];

                std::string var = boost::python::extract<std::string>(o_var);

                boost::python::extract<const Function<dim,double >*> get_double_fn (o_fn);
                boost::python::extract<const Function<dim,adouble>*> get_adouble_fn(o_fn);

                if(get_double_fn.check())
                {
                    daeDeclareException(exInvalidCall);
                    e << "Only the adouble version of the Function class (that is Function<dim,adouble>) objects are allowed for setting the Dirichlet BCs";
                    throw e;
                    //const Function<dim,double>* fn = get_double_fn();
                    //bcs_double.push_back( std::pair<std::string, const Function<dim,double>*>(var,fn) );
                }
                else if(get_adouble_fn.check())
                {
                    const Function<dim,adouble>* fn = get_adouble_fn();
                    bcs_adouble.push_back( std::pair<std::string, const Function<dim,adouble>*>(var,fn) );
                }
                else
                {
                    daeDeclareException(exInvalidCall);
                    e << "Invalid function " << key << " in the functions dictionary";
                    throw e;
                }

            }

            //if(bcs_double.size() > 0)
            //    this->m_functionsDirichletBC[key] = bcs_double;
            if(bcs_adouble.size() > 0)
                this->m_adoubleFunctionsDirichletBC[key] = bcs_adouble;
        }

        keys = dictFunctions.keys();
        for(int i = 0; i < len(keys); ++i)
        {
            boost::python::object key_ = keys[i];
            boost::python::object val_ = dictFunctions[key_];

            std::string key = boost::python::extract<std::string>(key_);

            boost::python::extract<const Function<dim,double >*> get_double_fn (val_);
            boost::python::extract<const Function<dim,adouble>*> get_adouble_fn(val_);

            if(get_double_fn.check())
            {
                this->m_functions[key] = get_double_fn();
            }
            else if(get_adouble_fn.check())
            {
                this->m_adouble_functions[key] = get_adouble_fn();
            }
            else
            {
                daeDeclareException(exInvalidCall);
                e << "Invalid function " << key << " in the functions dictionary";
                throw e;
            }
        }

        keys = dictFaceAij.keys();
        for(int i = 0; i < len(keys); ++i)
        {
            boost::python::object key_ = keys[i];
            boost::python::object val_ = dictFaceAij[key_];

            unsigned int      key  = boost::python::extract<unsigned int>(key_);
            feExpression<dim> expr = boost::python::extract< feExpression<dim> >(val_);

            this->m_faceAij[key] = expr;
        }

        keys = dictFaceFi.keys();
        for(int i = 0; i < len(keys); ++i)
        {
            boost::python::object key_ = keys[i];
            boost::python::object val_ = dictFaceFi[key_];

            unsigned int      key  = boost::python::extract<unsigned int>(key_);
            feExpression<dim> expr = boost::python::extract< feExpression<dim> >(val_);

            this->m_faceFi[key] = expr;
        }
    }

    ~dealiiFiniteElementWeakFormWrapper()
    {
    }

public:
    // Keep these objects so they do not go out of scope and get destroyed while still in use by daetools
    boost::python::object m_dictFaceAij;
    boost::python::object m_dictFaceFi;
    boost::python::object m_dictFunctions;
    boost::python::object m_dictFunctionsDirichletBC;
    boost::python::object m_dictBoundaryIntegrals;
};

template<int dim>
class dealiiFiniteElementSystemWrapper : public dealiiFiniteElementSystem<dim>,
                                         public boost::python::wrapper< dealiiFiniteElementSystem<dim> >
{
public:
    dealiiFiniteElementSystemWrapper(std::string               meshFilename,
                                     const Quadrature<dim>&    quadrature,
                                     const Quadrature<dim-1>&  faceQuadrature,
                                     boost::python::list       listDOFs)
    {
        std::vector< dealiiFiniteElementDOF<dim>* > arrDOFs;
        for(int i = 0; i < len(listDOFs); ++i)
        {
            dealiiFiniteElementDOF<dim>* dof = boost::python::extract< dealiiFiniteElementDOF<dim>* >(listDOFs[i]);
            arrDOFs.push_back(dof);
        }
        
        // Keep these objects so they do not go out of scope and get destroyed while still in use by daetools
        this->m_oDOFs = listDOFs;

        this->Initialize(meshFilename, quadrature, faceQuadrature, arrDOFs);
    }

    ~dealiiFiniteElementSystemWrapper()
    {
    }

public:
    dealiiFiniteElementWeakForm<dim>* GetWeakForm() const
    {
        return this->dealiiFiniteElementSystem<dim>::GetWeakForm();
    }

    void SetWeakForm(boost::python::object weakForm)
    {
        // Keep these objects so they do not go out of scope and get destroyed while still in use by daetools
        this->m_oWeakForm = weakForm;

        dealiiFiniteElementWeakForm<dim>* cweakForm  = boost::python::extract<dealiiFiniteElementWeakForm<dim>*>(weakForm);

        this->dealiiFiniteElementSystem<dim>::SetWeakForm(cweakForm);
    }

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
    
public:
    boost::python::object m_oWeakForm;
    boost::python::object m_oDOFs;
};


}

#endif
