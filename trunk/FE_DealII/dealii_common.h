#ifndef DEAL_II_COMMON_H
#define DEAL_II_COMMON_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include "../Core/coreimpl.h"
#include <typeinfo>

using namespace dealii;
/*********************************************************
 * deal.II related classes and typedefs
 *********************************************************/
typedef Function<1> Function_1D;
typedef Function<2> Function_2D;
typedef Function<3> Function_3D;

// Tensors of rank=1
typedef Tensor<1, 1, double> Tensor_1_1D;
typedef Tensor<1, 2, double> Tensor_1_2D;
typedef Tensor<1, 3, double> Tensor_1_3D;

// Tensors of rank=2
typedef Tensor<2, 1, double> Tensor_2_1D;
typedef Tensor<2, 2, double> Tensor_2_2D;
typedef Tensor<2, 3, double> Tensor_2_3D;

// Tensors of rank=3
typedef Tensor<3, 1, double> Tensor_3_1D;
typedef Tensor<3, 2, double> Tensor_3_2D;
typedef Tensor<3, 3, double> Tensor_3_3D;

// Tensors of rank=1
typedef Tensor<1, 1, adouble> Tensor_1_adouble_1D;
typedef Tensor<1, 2, adouble> Tensor_1_adouble_2D;
typedef Tensor<1, 3, adouble> Tensor_1_adouble_3D;

// Tensors of rank=2
typedef Tensor<2, 1, adouble> Tensor_2_adouble_1D;
typedef Tensor<2, 2, adouble> Tensor_2_adouble_2D;
typedef Tensor<2, 3, adouble> Tensor_2_adouble_3D;

// Tensors of rank=3
typedef Tensor<3, 1, adouble> Tensor_3_adouble_1D;
typedef Tensor<3, 2, adouble> Tensor_3_adouble_2D;
typedef Tensor<3, 3, adouble> Tensor_3_adouble_3D;

// Points are in fact tensors with rank=1 just their coordinates mean length
// and have some additional functions
typedef Point<1, double> Point_1D;
typedef Point<2, double> Point_2D;
typedef Point<3, double> Point_3D;

namespace dealii
{
template <> struct EnableIfScalar<adouble>
{
  typedef adouble type;
};
}

namespace dae
{
namespace fe_solver
{
/*********************************************************
 * daeFEMatrix
 * A wrapper around deal.II SparseMatrix<double>
 *********************************************************/
template<typename REAL = double>
class daeFEMatrix : public daeMatrix<REAL>
{
public:
    daeFEMatrix(const SparseMatrix<REAL>& matrix) : deal_ii_matrix(matrix)
    {
    }

    virtual ~daeFEMatrix(void)
    {
    }

public:
    virtual REAL GetItem(size_t row, size_t col) const
    {
        return deal_ii_matrix(row, col);
    }

    virtual void SetItem(size_t row, size_t col, REAL value)
    {
        // ACHTUNG, ACHTUNG!! Setting a new value is NOT permitted!
        daeDeclareAndThrowException(exInvalidCall);
    }

    virtual size_t GetNrows(void) const
    {
        return deal_ii_matrix.n();
    }

    virtual size_t GetNcols(void) const
    {
        return deal_ii_matrix.m();
    }

protected:
    const SparseMatrix<REAL>& deal_ii_matrix;
};

/*********************************************************
 * daeFEBlockMatrix
 * A wrapper around deal.II BlockSparseMatrix<double>
 *********************************************************/
template<typename REAL = double>
class daeFEBlockMatrix : public daeMatrix<REAL>
{
public:
    daeFEBlockMatrix(const BlockSparseMatrix<REAL>& matrix) : deal_ii_matrix(matrix)
    {
    }

    virtual ~daeFEBlockMatrix(void)
    {
    }

public:
    virtual REAL GetItem(size_t row, size_t col) const
    {
        return deal_ii_matrix(row, col);
    }

    virtual void SetItem(size_t row, size_t col, REAL value)
    {
        // ACHTUNG, ACHTUNG!! Setting a new value is NOT permitted!
        daeDeclareAndThrowException(exInvalidCall);
    }

    virtual size_t GetNrows(void) const
    {
        return deal_ii_matrix.n();
    }

    virtual size_t GetNcols(void) const
    {
        return deal_ii_matrix.m();
    }

protected:
    const BlockSparseMatrix<REAL>& deal_ii_matrix;
};

/*********************************************************
 * daeFEArray
 * A wrapper around deal.II Vector<REAL>
 *********************************************************/
template<typename REAL = double>
class daeFEArray : public daeArray<REAL>
{
public:
    daeFEArray(const Vector<REAL>& vect) : deal_ii_vector(vect)
    {
    }

    virtual ~daeFEArray(void)
    {
    }

public:
    REAL operator [](size_t i) const
    {
        return deal_ii_vector[i];
    }

    REAL GetItem(size_t i) const
    {
        return deal_ii_vector[i];
    }

    void SetItem(size_t i, REAL value)
    {
        // ACHTUNG, ACHTUNG!! Setting a new value is NOT permitted!
        daeDeclareAndThrowException(exInvalidCall);
    }

    size_t GetSize(void) const
    {
        return deal_ii_vector.size();
    }

protected:
    const Vector<REAL>& deal_ii_vector;
};

/*********************************************************
 * daeFEBlockArray
 * A wrapper around deal.II Vector<REAL>
 *********************************************************/
template<typename REAL = double>
class daeFEBlockArray : public daeArray<REAL>
{
public:
    daeFEBlockArray(const BlockVector<REAL>& vect) : deal_ii_vector(vect)
    {
    }

    virtual ~daeFEBlockArray(void)
    {
    }

public:
    REAL operator [](size_t i) const
    {
        return deal_ii_vector[i];
    }

    REAL GetItem(size_t i) const
    {
        return deal_ii_vector[i];
    }

    void SetItem(size_t i, REAL value)
    {
        // ACHTUNG, ACHTUNG!! Setting a new value is NOT permitted!
        daeDeclareAndThrowException(exInvalidCall);
    }

    size_t GetSize(void) const
    {
        return deal_ii_vector.size();
    }

protected:
    const BlockVector<REAL>& deal_ii_vector;
};

/*********************************************************
 * dealiiSparsityPatternIterator
 * A wrapper around dealSparsityPattern::iterator
 *********************************************************/
class dealiiSparsityPatternIterator : public daeSparseMatrixRowIterator
{
public:
    dealiiSparsityPatternIterator(const SparsityPattern::iterator& start, const SparsityPattern::iterator& end):
        m_iterator(start),
        m_end(end)
    {
    }

    void first()
    {
    }

    void next()
    {
        ++m_iterator;
    }

    bool isDone()
    {
        return m_iterator == m_end;
    }

    unsigned int currentItem()
    {
        return m_iterator->column();
    }

public:
    SparsityPattern::iterator m_iterator;
    SparsityPattern::iterator m_end;
};


/*********************************************************
 * feExpression
 *********************************************************/
enum efeNumberType
{
    eFEScalar = 0,
    eFECurl2D,
    eFETensor1,
    eFETensor2,
    eFETensor3,
    eFEPoint,
    eFEScalar_adouble,
    eFETensor1_adouble,
    eFETensor2_adouble,
    eFETensor3_adouble,
    eFESymmetricTensor2,
    eFEInvalid
};

enum efeFunctionCall
{
    eFunctionValue,
    eFunctionGradient
};

const int fe_q = -1000;
const int fe_i = -1001;
const int fe_j = -1002;


template<int dim>
class feNode;

template<int dim>
class feExpression
{
public:
    typedef typename boost::shared_ptr< feNode<dim> > feNodePtr;

    feExpression()
    {
    }

    feExpression(feNodePtr node) : m_node(node)
    {
    }

    bool operator ==(const feExpression<dim>& other) const
    {
        throw std::runtime_error(std::string("not implemented"));

        if(!m_node && !other.m_node)
            return true;
        if((m_node && !other.m_node) || (!m_node && other.m_node))
            return false;

        return false;
    }

    std::string ToString() const
    {
        return m_node->ToString();
    }

public:
    feNodePtr m_node;
};

inline std::string type(efeNumberType eType)
{
    if(eType == eFEScalar)
        return "Scalar";
    else if(eType == eFEScalar_adouble)
        return "Scalar_adouble";
    else if(eType == eFETensor1)
        return "Tensor1";
    else if(eType == eFETensor2)
        return "Tensor2";
    else if(eType == eFETensor3)
        return "Tensor3";
    else if(eType == eFETensor1_adouble)
        return "Tensor1_adouble";
    else if(eType == eFETensor2_adouble)
        return "Tensor2_adouble";
    else if(eType == eFETensor3_adouble)
        return "Tensor3_adouble";
    else if(eType == eFEPoint)
        return "Point";
    else if(eType == eFESymmetricTensor2)
        return "SymmetricTensor2";
    else if(eType == eFECurl2D)
        return "Curl2D";
    else if(eType == eFEInvalid)
        return "Invalid";
    else
        return "unknown";
}

template<int dim>
class feRuntimeNumber
{
public:
    feRuntimeNumber()
    {
        m_eType = eFEInvalid;
    }

    feRuntimeNumber(double v)
    {
        m_eType = eFEScalar;
        m_value = v;
    }

    feRuntimeNumber(const adouble& a)
    {
        m_eType = eFEScalar_adouble;
        m_adouble_value = a;
    }

    /* CURL
    feRuntimeNumber(const Tensor<1, 1, double>& t)
    {
        m_eType   = eFECurl2D;
        m_curl2D  = t;
    }
    */

    feRuntimeNumber(const Tensor<1, dim, double>& t)
    {
        m_eType   = eFETensor1;
        m_tensor1 = t;
    }

    feRuntimeNumber(const Tensor<2, dim, double>& t)
    {
        m_eType   = eFETensor2;
        m_tensor2 = t;
    }

    feRuntimeNumber(const Tensor<3, dim, double>& t)
    {
        m_eType   = eFETensor3;
        m_tensor3 = t;
    }

    feRuntimeNumber(const Tensor<1, dim, adouble>& t)
    {
        m_eType           = eFETensor1_adouble;
        m_tensor1_adouble = t;
    }

    feRuntimeNumber(const Tensor<2, dim, adouble>& t)
    {
        m_eType           = eFETensor2_adouble;
        m_tensor2_adouble = t;
    }

    feRuntimeNumber(const Tensor<3, dim, adouble>& t)
    {
        m_eType           = eFETensor3_adouble;
        m_tensor3_adouble = t;
    }

    feRuntimeNumber(const Point<dim, double>& t)
    {
        m_eType = eFEPoint;
        m_point = t;
    }

    feRuntimeNumber(const SymmetricTensor<2, dim, double>& st)
    {
        m_eType             = eFESymmetricTensor2;
        m_symmetric_tensor2 = st;
    }

    std::string ToString() const
    {
        if(m_eType == eFEScalar)
        {
            return (boost::format("%f") % m_value).str();
        }
        else if(m_eType == eFEScalar_adouble)
        {
            return (boost::format("(%f,%f,%p)") % m_adouble_value.getValue() % m_adouble_value.getDerivative() % (void*)m_adouble_value.node.get()).str();
        }
        /* CURL
        else if(m_eType == eFECurl2D)
        {
            if(dim != 2)
                throw std::runtime_error("Invalid number of dimensions for curl() result; it must be 2D");

            return (boost::format("[%f]") % m_curl2D[0]).str();
        }
        */
        else if(m_eType == eFETensor1)
        {
            if(dim == 1)
                return (boost::format("[%f]") % m_tensor1[0]).str();
            else if(dim == 2)
                return (boost::format("[%f, %f]") % m_tensor1[0] % m_tensor1[1]).str();
            else if(dim == 3)
                return (boost::format("[%f, %f, %f]") % m_tensor1[0] % m_tensor1[1] % m_tensor1[2]).str();
        }
        else if(m_eType == eFETensor2)
        {
            if(dim == 1)
                return (boost::format("[[%f], [%f]]") % m_tensor2[0][0] % m_tensor2[0][1]).str();
            else if(dim == 2)
                return (boost::format("[[%f, %f], [%f, %f]]")  % m_tensor2[0][0] % m_tensor2[0][1]
                                                               % m_tensor2[1][0] % m_tensor2[1][1]).str();
            else if(dim == 3)
                return (boost::format("[[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]") % m_tensor2[0][0] % m_tensor2[0][1] % m_tensor2[0][2]
                                                                                    % m_tensor2[1][0] % m_tensor2[1][1] % m_tensor2[1][2]
                                                                                    % m_tensor2[2][0] % m_tensor2[2][1] % m_tensor2[2][2]).str();
        }
        else if(m_eType == eFETensor3)
        {
            return "";
        }
        else if(m_eType == eFETensor1_adouble)
        {
            if(dim == 1)
                return (boost::format("[%s]") % m_tensor1_adouble[0].NodeAsPlainText()).str();
            else if(dim == 2)
                return (boost::format("[%s, %s]") % m_tensor1_adouble[0].NodeAsPlainText() % m_tensor1_adouble[1].NodeAsPlainText()).str();
            else if(dim == 3)
                return (boost::format("[%s, %s, %s]") % m_tensor1_adouble[0].NodeAsPlainText() % m_tensor1_adouble[1].NodeAsPlainText() % m_tensor1_adouble[2].NodeAsPlainText()).str();
        }
        else if(m_eType == eFETensor2_adouble)
        {
            if(dim == 1)
                return (boost::format("[[%s], [%s]]") % m_tensor2_adouble[0][0].NodeAsPlainText() % m_tensor2_adouble[0][1].NodeAsPlainText()).str();
            else if(dim == 2)
                return (boost::format("[[%s, %s], [%s, %s]]")  % m_tensor2_adouble[0][0].NodeAsPlainText() % m_tensor2_adouble[0][1].NodeAsPlainText()
                                                               % m_tensor2_adouble[1][0].NodeAsPlainText() % m_tensor2_adouble[1][1].NodeAsPlainText()).str();
            else if(dim == 3)
                return (boost::format("[[%s, %s, %s], [%s, %s, %s], [%s, %s, %s]]") % m_tensor2_adouble[0][0].NodeAsPlainText() % m_tensor2_adouble[0][1].NodeAsPlainText() % m_tensor2_adouble[0][2].NodeAsPlainText()
                                                                                    % m_tensor2_adouble[1][0].NodeAsPlainText() % m_tensor2_adouble[1][1].NodeAsPlainText() % m_tensor2_adouble[1][2].NodeAsPlainText()
                                                                                    % m_tensor2_adouble[2][0].NodeAsPlainText() % m_tensor2_adouble[2][1].NodeAsPlainText() % m_tensor2_adouble[2][2].NodeAsPlainText()).str();
        }
        else if(m_eType == eFETensor3_adouble)
        {
            return "";
        }
        else if(m_eType == eFESymmetricTensor2)
        {
            if(dim == 1)
                return (boost::format("[[%f], [%f]]") % m_symmetric_tensor2[0][0] % m_symmetric_tensor2[0][1]).str();
            else if(dim == 2)
                return (boost::format("[[%f, %f], [%f, %f]]")  % m_symmetric_tensor2[0][0] % m_symmetric_tensor2[0][1]
                                                               % m_symmetric_tensor2[1][0] % m_symmetric_tensor2[1][1]).str();
            else if(dim == 3)
                return (boost::format("[[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]") % m_symmetric_tensor2[0][0] % m_symmetric_tensor2[0][1] % m_symmetric_tensor2[0][2]
                                                                                    % m_symmetric_tensor2[1][0] % m_symmetric_tensor2[1][1] % m_symmetric_tensor2[1][2]
                                                                                    % m_symmetric_tensor2[2][0] % m_symmetric_tensor2[2][1] % m_symmetric_tensor2[2][2]).str();
        }
        else if(m_eType == eFEPoint)
        {
            if(dim == 1)
                return (boost::format("(%f)") % m_point[0]).str();
            else if(dim == 2)
                return (boost::format("(%f, %f)") % m_point[0] % m_point[1]).str();
            else if(dim == 3)
                return (boost::format("(%f, %f, %f)") % m_point[0] % m_point[1] % m_point[2]).str();
        }
        else
            return "(unknown)";
    }

public:
    efeNumberType                       m_eType;
    Point<dim, double>                  m_point;
    /* CURL
    Tensor<1, 1, double>                m_curl2D;
    */
    Tensor<1, dim, double>              m_tensor1;
    Tensor<2, dim, double>              m_tensor2;
    Tensor<3, dim, double>              m_tensor3;
    Tensor<1, dim, adouble>             m_tensor1_adouble;
    Tensor<2, dim, adouble>             m_tensor2_adouble;
    Tensor<3, dim, adouble>             m_tensor3_adouble;
    SymmetricTensor<2, dim, double>     m_symmetric_tensor2;
    double                              m_value;
    adouble                             m_adouble_value;
};

template<int dim>
feRuntimeNumber<dim> operator -(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = -fe.m_value;
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = -fe.m_adouble_value;
    }
    /* CURL
    else if(fe.m_eType == eFECurl2D)
    {
        tmp.m_eType  = eFECurl2D;
        tmp.m_curl2D = -fe.m_curl2D;
    }
    */
    else if(fe.m_eType == eFETensor1)
    {
        tmp.m_eType = eFETensor1;
        tmp.m_tensor1 = -fe.m_tensor1;
    }
    else if(fe.m_eType == eFETensor2)
    {
        tmp.m_eType = eFETensor2;
        tmp.m_tensor2 = -fe.m_tensor2;
    }
    else if(fe.m_eType == eFETensor3)
    {
        tmp.m_eType = eFETensor3;
        tmp.m_tensor3 = -fe.m_tensor3;
    }

    else if(fe.m_eType == eFETensor1_adouble)
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = -fe.m_tensor1_adouble;
    }
    else if(fe.m_eType == eFETensor2_adouble)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = -fe.m_tensor2_adouble;
    }
    else if(fe.m_eType == eFETensor3_adouble)
    {
        tmp.m_eType = eFETensor3_adouble;
        tmp.m_tensor3_adouble = -fe.m_tensor3_adouble;
    }

    else if(fe.m_eType == eFEPoint)
    {
        tmp.m_eType = eFEPoint;
        tmp.m_point = -fe.m_point;
    }
    else
        throw std::runtime_error(std::string("Invalid operation - for the type: ") + type(fe.m_eType));
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> operator +(const feRuntimeNumber<dim>& l, const feRuntimeNumber<dim>& r)
{
    feRuntimeNumber<dim> tmp;
    if(l.m_eType == eFEScalar && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = l.m_value + r.m_value;
    }
    else if(l.m_eType == eFEScalar_adouble || r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        
        adouble lad, rad;

        if(l.m_eType == eFEScalar)
            lad = l.m_value;
        else if(l.m_eType == eFEScalar_adouble)
            lad = l.m_adouble_value;
        else
            throw std::runtime_error(std::string("Invalid operation ") + type(l.m_eType) + " + " + type(r.m_eType));

        if(r.m_eType == eFEScalar)
            rad = r.m_value;
        else if(r.m_eType == eFEScalar_adouble)
            rad = r.m_adouble_value;
        else
            throw std::runtime_error(std::string("Invalid operation ") + type(l.m_eType) + " + " + type(r.m_eType));

        tmp.m_adouble_value = lad + rad;
    }
    /* CURL
    else if(l.m_eType == eFECurl2D && r.m_eType == eFECurl2D)
    {
        tmp.m_eType  = eFECurl2D;
        tmp.m_curl2D = l.m_curl2D + r.m_curl2D;
    }
    */
    else if(l.m_eType == eFETensor1 && r.m_eType == eFETensor1)
    {
        tmp.m_eType = eFETensor1;
        tmp.m_tensor1 = l.m_tensor1 + r.m_tensor1;
    }
    else if(l.m_eType == eFETensor2 && r.m_eType == eFETensor2)
    {
        tmp.m_eType = eFETensor2;
        tmp.m_tensor2 = l.m_tensor2 + r.m_tensor2;
    }
    else if(l.m_eType == eFETensor3 && r.m_eType == eFETensor3)
    {
        tmp.m_eType = eFETensor3;
        tmp.m_tensor3 = l.m_tensor3 + r.m_tensor3;
    }

    else if(l.m_eType == eFETensor1_adouble && r.m_eType == eFETensor1_adouble)
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_tensor1_adouble + r.m_tensor1_adouble;
    }
    else if(l.m_eType == eFETensor2_adouble && r.m_eType == eFETensor2_adouble)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = l.m_tensor2_adouble + r.m_tensor2_adouble;
    }
    else if(l.m_eType == eFETensor3_adouble && r.m_eType == eFETensor3_adouble)
    {
        tmp.m_eType = eFETensor3_adouble;
        tmp.m_tensor3_adouble = l.m_tensor3_adouble + r.m_tensor3_adouble;
    }

    else if(l.m_eType == eFEPoint && r.m_eType == eFEPoint)
    {
        tmp.m_eType = eFEPoint;
        tmp.m_tensor1 = l.m_point + r.m_point;
    }
    else
    {
        std::string error = std::string("Invalid operation ") + type(l.m_eType) + " + " + type(r.m_eType) + ":\n";
        error += l.ToString() + " + " + r.ToString(); 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> operator -(const feRuntimeNumber<dim>& l, const feRuntimeNumber<dim>& r)
{
    feRuntimeNumber<dim> tmp;
    if(l.m_eType == eFEScalar && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = l.m_value - r.m_value;
    }
    else if(l.m_eType == eFEScalar_adouble || r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        
        adouble lad, rad;

        if(l.m_eType == eFEScalar)
            lad = l.m_value;
        else if(l.m_eType == eFEScalar_adouble)
            lad = l.m_adouble_value;
        else
            throw std::runtime_error(std::string("Invalid operation ") + type(l.m_eType) + " - " + type(r.m_eType));

        if(r.m_eType == eFEScalar)
            rad = r.m_value;
        else if(r.m_eType == eFEScalar_adouble)
            rad = r.m_adouble_value;
        else
            throw std::runtime_error(std::string("Invalid operation ") + type(l.m_eType) + " - " + type(r.m_eType));

        tmp.m_adouble_value = lad - rad;
    }
    /* CURL
    else if(l.m_eType == eFECurl2D && r.m_eType == eFECurl2D)
    {
        tmp.m_eType  = eFECurl2D;
        tmp.m_curl2D = l.m_curl2D - r.m_curl2D;
    }
    */
    else if(l.m_eType == eFETensor1 && r.m_eType == eFETensor1)
    {
        tmp.m_eType = eFETensor1;
        tmp.m_tensor1 = l.m_tensor1 - r.m_tensor1;
    }
    else if(l.m_eType == eFETensor2 && r.m_eType == eFETensor2)
    {
        tmp.m_eType = eFETensor2;
        tmp.m_tensor2 = l.m_tensor2 - r.m_tensor2;
    }
    else if(l.m_eType == eFETensor3 && r.m_eType == eFETensor3)
    {
        tmp.m_eType = eFETensor3;
        tmp.m_tensor3 = l.m_tensor3 - r.m_tensor3;
    }

    else if(l.m_eType == eFETensor1_adouble && r.m_eType == eFETensor1_adouble)
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_tensor1_adouble - r.m_tensor1_adouble;
    }
    else if(l.m_eType == eFETensor2_adouble && r.m_eType == eFETensor2_adouble)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = l.m_tensor2_adouble - r.m_tensor2_adouble;
    }
    else if(l.m_eType == eFETensor3_adouble && r.m_eType == eFETensor3_adouble)
    {
        tmp.m_eType = eFETensor3_adouble;
        tmp.m_tensor3_adouble = l.m_tensor3_adouble - r.m_tensor3_adouble;
    }

    else if(l.m_eType == eFEPoint && r.m_eType == eFEPoint)
    {
        tmp.m_eType = eFEPoint;
        tmp.m_tensor1 = l.m_point - r.m_point;
    }
    else
    {
        std::string error = std::string("Invalid operation ") + type(l.m_eType) + " - " + type(r.m_eType) + ":\n";
        error += l.ToString() + " - " + r.ToString(); 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> operator *(const feRuntimeNumber<dim>& l, const feRuntimeNumber<dim>& r)
{
    feRuntimeNumber<dim> tmp;
    if(l.m_eType == eFEScalar && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = l.m_value * r.m_value;
    }
    else if(l.m_eType == eFEScalar_adouble || r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        
        adouble lad, rad;
        if(l.m_eType == eFEScalar)
            lad = l.m_value;
        else if(l.m_eType == eFEScalar_adouble)
            lad = l.m_adouble_value;
        else
            throw std::runtime_error(std::string("Invalid operation ") + type(l.m_eType) + " * " + type(r.m_eType));

        if(r.m_eType == eFEScalar)
            rad = r.m_value;
        else if(r.m_eType == eFEScalar_adouble)
            rad = r.m_adouble_value;
        else
            throw std::runtime_error(std::string("Invalid operation ") + type(l.m_eType) + " * " + type(r.m_eType));

        tmp.m_adouble_value = lad * rad;
    }
    /* CURL
    else if(l.m_eType == eFECurl2D && r.m_eType == eFECurl2D)
    {
        tmp.m_eType  = eFEScalar;
        tmp.m_value = l.m_curl2D * r.m_curl2D;
    }
    */
    else if(l.m_eType == eFETensor1 && r.m_eType == eFETensor1)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = l.m_tensor1 * r.m_tensor1;
    }
    else if(l.m_eType == eFETensor2 && r.m_eType == eFETensor2)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = scalar_product(l.m_tensor2, r.m_tensor2);
    }
    /*
    else if(l.m_eType == eFETensor3 && r.m_eType == eFETensor3)
    {
        tmp.m_eType = eFETensor2;
        tmp.m_tensor2 = l.m_tensor3 * r.m_tensor3;
    }
    */
    else if(l.m_eType == eFETensor1_adouble && r.m_eType == eFETensor1_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = l.m_tensor1_adouble * r.m_tensor1_adouble;
    }
    else if(l.m_eType == eFETensor2_adouble && r.m_eType == eFETensor2_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = scalar_product(l.m_tensor2_adouble, r.m_tensor2_adouble);
    }
    /*
    else if(l.m_eType == eFETensor3_adouble && r.m_eType == eFETensor3_adouble)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = l.m_tensor3_adouble * r.m_tensor3_adouble;
    }
    */

    else if(l.m_eType == eFETensor2 && r.m_eType == eFETensor1) // Tensor<2> * Tensor<1> = Tensor<2+1-2> => Tensor<1>
    {
        tmp.m_eType = eFETensor1;
        tmp.m_tensor1 = l.m_tensor2 * r.m_tensor1;
    }
    else if(l.m_eType == eFETensor1 && r.m_eType == eFETensor2) // Tensor<2> * Tensor<1> = Tensor<1+2-2> => Tensor<1>
    {
        tmp.m_eType = eFETensor1;
        tmp.m_tensor1 = l.m_tensor1 * r.m_tensor2;
    }

    else if(l.m_eType == eFETensor2 && r.m_eType == eFETensor1_adouble) // Tensor<2> * Tensor<1> = Tensor<2+1-2> => Tensor<1>
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_tensor2 * r.m_tensor1_adouble;
    }
    else if(l.m_eType == eFETensor1_adouble && r.m_eType == eFETensor2) // Tensor<2> * Tensor<1> = Tensor<1+2-2> => Tensor<1>
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_tensor1_adouble * r.m_tensor2;
    }

    
    else if(l.m_eType == eFETensor1 && r.m_eType == eFETensor1_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = l.m_tensor1 * r.m_tensor1_adouble;
    }
    else if(l.m_eType == eFETensor1_adouble && r.m_eType == eFETensor1)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = l.m_tensor1_adouble * r.m_tensor1;
    }
    
    
    else if(l.m_eType == eFESymmetricTensor2 && r.m_eType == eFESymmetricTensor2)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = scalar_product(l.m_symmetric_tensor2, r.m_symmetric_tensor2);
    }
    else if(l.m_eType == eFESymmetricTensor2 && r.m_eType == eFETensor2)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = scalar_product(l.m_symmetric_tensor2, r.m_tensor2);
    }
    else if(l.m_eType == eFETensor2 && r.m_eType == eFESymmetricTensor2)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = scalar_product(l.m_tensor2, r.m_symmetric_tensor2);
    }
    else if(l.m_eType == eFEPoint && r.m_eType == eFEPoint)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = l.m_point * r.m_point;
    }

    /* CURL
    else if(l.m_eType == eFEScalar && r.m_eType == eFECurl2D)
    {
        tmp.m_eType  = eFECurl2D;
        tmp.m_curl2D = l.m_value * r.m_curl2D;
    }
    else if(l.m_eType == eFECurl2D && r.m_eType == eFEScalar)
    {
        tmp.m_eType  = eFECurl2D;
        tmp.m_curl2D = l.m_curl2D * r.m_value;
    }
    */

    else if(l.m_eType == eFEScalar && r.m_eType == eFETensor1)
    {
        tmp.m_eType = eFETensor1;
        tmp.m_tensor1 = l.m_value * r.m_tensor1;
    }
    else if(l.m_eType == eFETensor1 && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor1;
        tmp.m_tensor1 = l.m_tensor1 * r.m_value;
    }


    else if(l.m_eType == eFEScalar && r.m_eType == eFETensor1_adouble)
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_value * r.m_tensor1_adouble;
    }
    else if(l.m_eType == eFETensor1_adouble && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_tensor1_adouble * r.m_value;
    }
    else if(l.m_eType == eFEScalar_adouble && r.m_eType == eFETensor1_adouble)
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_adouble_value * r.m_tensor1_adouble;
    }
    else if(l.m_eType == eFETensor1_adouble && r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_tensor1_adouble * r.m_adouble_value;
    }


    else if(l.m_eType == eFEScalar && r.m_eType == eFETensor2)
    {
        tmp.m_eType = eFETensor2;
        tmp.m_tensor2 = l.m_value * r.m_tensor2;
    }
    else if(l.m_eType == eFETensor2 && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor2;
        tmp.m_tensor2 = l.m_tensor2 * r.m_value;
    }


    else if(l.m_eType == eFEScalar && r.m_eType == eFETensor2_adouble)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = l.m_value * r.m_tensor2_adouble;
    }
    else if(l.m_eType == eFETensor2_adouble && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = l.m_tensor2_adouble * r.m_value;
    }
    else if(l.m_eType == eFEScalar_adouble && r.m_eType == eFETensor2_adouble)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = l.m_adouble_value * r.m_tensor2_adouble;
    }
    else if(l.m_eType == eFETensor2_adouble && r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = l.m_tensor2_adouble * r.m_adouble_value;
    }


    else if(l.m_eType == eFEScalar && r.m_eType == eFETensor3)
    {
        tmp.m_eType = eFETensor3;
        tmp.m_tensor3 = l.m_value * r.m_tensor3;
    }
    else if(l.m_eType == eFETensor3 && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor3;
        tmp.m_tensor3 = l.m_tensor3 * r.m_value;
    }


    else if(l.m_eType == eFEScalar && r.m_eType == eFETensor3_adouble)
    {
        tmp.m_eType = eFETensor3_adouble;
        tmp.m_tensor3_adouble = l.m_value * r.m_tensor3_adouble;
    }
    else if(l.m_eType == eFETensor3_adouble && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor3_adouble;
        tmp.m_tensor3_adouble = l.m_tensor3_adouble * r.m_value;
    }
    else if(l.m_eType == eFEScalar_adouble && r.m_eType == eFETensor3_adouble)
    {
        tmp.m_eType = eFETensor3_adouble;
        tmp.m_tensor3_adouble = l.m_adouble_value * r.m_tensor3_adouble;
    }
    else if(l.m_eType == eFETensor3_adouble && r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFETensor3_adouble;
        tmp.m_tensor3_adouble = l.m_tensor3_adouble * r.m_adouble_value;
    }


    else if(l.m_eType == eFEScalar && r.m_eType == eFEPoint)
    {
        tmp.m_eType = eFEPoint;
        tmp.m_point = l.m_value * r.m_point;
    }
    else if(l.m_eType == eFEPoint && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEPoint;
        tmp.m_point = l.m_point * r.m_value;
    }

    else
    {
        std::string error = std::string("Invalid operation ") + type(l.m_eType) + " * " + type(r.m_eType) + ":\n";
        error += l.ToString() + " * " + r.ToString(); 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> operator /(const feRuntimeNumber<dim>& l, const feRuntimeNumber<dim>& r)
{
    feRuntimeNumber<dim> tmp;
    if(l.m_eType == eFEScalar && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = l.m_value / r.m_value;
    }
    else if(l.m_eType == eFEScalar_adouble || r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        
        adouble lad, rad;
        if(l.m_eType == eFEScalar)
            lad = l.m_value;
        else if(l.m_eType == eFEScalar_adouble)
            lad = l.m_adouble_value;
        else
            throw std::runtime_error(std::string("Invalid operation ") + type(l.m_eType) + " / " + type(r.m_eType));

        if(r.m_eType == eFEScalar)
            rad = r.m_value;
        else if(r.m_eType == eFEScalar_adouble)
            rad = r.m_adouble_value;
        else
            throw std::runtime_error(std::string("Invalid operation ") + type(l.m_eType) + " / " + type(r.m_eType));

        tmp.m_adouble_value = lad / rad;
    }
    /* CURL
    else if(l.m_eType == eFECurl2D && r.m_eType == eFEScalar)
    {
        tmp.m_eType  = eFECurl2D;
        tmp.m_curl2D = l.m_curl2D / r.m_value;
    }
    */
    else if(l.m_eType == eFETensor1 && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor1;
        tmp.m_tensor1 = l.m_tensor1 / r.m_value;
    }
    else if(l.m_eType == eFETensor2 && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor2;
        tmp.m_tensor2 = l.m_tensor2 / r.m_value;
    }
    else if(l.m_eType == eFETensor3 && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor3;
        tmp.m_tensor3 = l.m_tensor3 / r.m_value;
    }


    else if(l.m_eType == eFETensor1_adouble && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_tensor1_adouble / r.m_value;
    }
    else if(l.m_eType == eFETensor2_adouble && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = l.m_tensor2_adouble / r.m_value;
    }
    else if(l.m_eType == eFETensor3_adouble && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFETensor3_adouble;
        tmp.m_tensor3_adouble = l.m_tensor3_adouble / r.m_value;
    }
    else if(l.m_eType == eFETensor1_adouble && r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFETensor1_adouble;
        tmp.m_tensor1_adouble = l.m_tensor1_adouble / r.m_adouble_value;
    }
    else if(l.m_eType == eFETensor2_adouble && r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFETensor2_adouble;
        tmp.m_tensor2_adouble = l.m_tensor2_adouble / r.m_adouble_value;
    }
    else if(l.m_eType == eFETensor3_adouble && r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFETensor3_adouble;
        tmp.m_tensor3_adouble = l.m_tensor3_adouble / r.m_adouble_value;
    }


    else if(l.m_eType == eFEPoint && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEPoint;
        tmp.m_point = l.m_point / r.m_value;
    }
    else
    {
        std::string error = std::string("Invalid operation ") + type(l.m_eType) + " / " + type(r.m_eType) + ":\n";
        error += l.ToString() + " / " + r.ToString(); 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> operator ^(const feRuntimeNumber<dim>& l, const feRuntimeNumber<dim>& r)
{
    feRuntimeNumber<dim> tmp;
    if(l.m_eType == eFEScalar && r.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = pow(l.m_value, r.m_value);
    }
    else if(l.m_eType == eFEScalar_adouble && r.m_eType == eFEScalar)
    {
        tmp.m_eType         = eFEScalar_adouble;
        tmp.m_adouble_value = pow(l.m_adouble_value, r.m_value);
    }
    else if(l.m_eType == eFEScalar_adouble && r.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType         = eFEScalar_adouble;
        tmp.m_adouble_value = pow(l.m_adouble_value, r.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation ") + type(l.m_eType) + " ** " + type(r.m_eType) + ":\n";
        error += l.ToString() + " ** " + r.ToString(); 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> pow(const feRuntimeNumber<dim>& l, const feRuntimeNumber<dim>& r)
{
    return l ^ r;
}

template<int dim>
feRuntimeNumber<dim> sin_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = sin(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = sin(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation sin(") + type(fe.m_eType) + "):\n";
        error += "sin(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> cos_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = cos(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = cos(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation cos(") + type(fe.m_eType) + "):\n";
        error += "cos(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> tan_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = tan(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = tan(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation tan(") + type(fe.m_eType) + "):\n";
        error += "tan(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> asin_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = asin(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = asin(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation asin(") + type(fe.m_eType) + "):\n";
        error += "asin(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> acos_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = acos(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = acos(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation acos(") + type(fe.m_eType) + "):\n";
        error += "acos(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> atan_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = atan(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = atan(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation atan(") + type(fe.m_eType) + "):\n";
        error += "atan(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> sqrt_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = sqrt(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = sqrt(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation sqrt(") + type(fe.m_eType) + "):\n";
        error += "sqrt(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> log_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = log(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = log(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation log(") + type(fe.m_eType) + "):\n";
        error += "log(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> log10_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = log10(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = log10(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation log10(") + type(fe.m_eType) + "):\n";
        error += "log10(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> exp_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = exp(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = exp(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation exp(") + type(fe.m_eType) + "):\n";
        error += "exp(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> abs_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = abs(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = abs(fe.m_adouble_value);
    }
    else
    {
        std::string error = std::string("Invalid operation abs(") + type(fe.m_eType) + "):\n";
        error += "abs(" + fe.ToString() + ")"; 
        throw std::runtime_error(error);
    }
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> sinh_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = sinh(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = sinh(fe.m_adouble_value);
    }
    else
        throw std::runtime_error(std::string("Invalid operation sinh(") + typeid(fe).name() + ")");
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> cosh_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = cosh(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = cosh(fe.m_adouble_value);
    }
    else
        throw std::runtime_error(std::string("Invalid operation cosh(") + typeid(fe).name() + ")");
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> tanh_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = tanh(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = tanh(fe.m_adouble_value);
    }
    else
        throw std::runtime_error(std::string("Invalid operation tanh(") + typeid(fe).name() + ")");
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> asinh_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = asinh(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = asinh(fe.m_adouble_value);
    }
    else
        throw std::runtime_error(std::string("Invalid operation asinh(") + typeid(fe).name() + ")");
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> acosh_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = acosh(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = acosh(fe.m_adouble_value);
    }
    else
        throw std::runtime_error(std::string("Invalid operation acosh(") + typeid(fe).name() + ")");
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> atanh_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = atanh(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = atanh(fe.m_adouble_value);
    }
    else
        throw std::runtime_error(std::string("Invalid operation atanh(") + typeid(fe).name() + ")");
    return tmp;
}

template<int dim>
feRuntimeNumber<dim> erf_(const feRuntimeNumber<dim>& fe)
{
    feRuntimeNumber<dim> tmp;
    if(fe.m_eType == eFEScalar)
    {
        tmp.m_eType = eFEScalar;
        tmp.m_value = erf(fe.m_value);
    }
    else if(fe.m_eType == eFEScalar_adouble)
    {
        tmp.m_eType = eFEScalar_adouble;
        tmp.m_adouble_value = erf(fe.m_adouble_value);
    }
    else
        throw std::runtime_error(std::string("Invalid operation erf(") + typeid(fe).name() + ")");
    return tmp;
}


template<int dim>
class feCellContext
{
public:
    ~feCellContext(){}

public:
    virtual double value(const std::string& variableName,
                         const unsigned int i,
                         const unsigned int q) const = 0;
    virtual Tensor<1,dim> gradient (const std::string& variableName,
                                    const unsigned int i,
                                    const unsigned int q) const = 0;
    virtual Tensor<2,dim> hessian (const std::string& variableName,
                                   const unsigned int i,
                                   const unsigned int q) const = 0;

    virtual Tensor<1,dim> vector_value(const std::string& variableName,
                                       const unsigned int i,
                                       const unsigned int q) const = 0;
    virtual Tensor<2,dim> vector_gradient (const std::string& variableName,
                                           const unsigned int i,
                                           const unsigned int q) const = 0;
    virtual Tensor<3,dim> vector_hessian (const std::string& variableName,
                                          const unsigned int i,
                                          const unsigned int q) const = 0;
    virtual double divergence(const std::string& variableName,
                              const unsigned int i,
                              const unsigned int q) const = 0;
    virtual SymmetricTensor<2,dim > symmetric_gradient(const std::string& variableName,
                                                       const unsigned int i,
                                                       const unsigned int q) const = 0;

    /* CURL
    virtual Tensor<1,1> curl_2D(const std::string& variableName,
                                const unsigned int i,
                                const unsigned int q) const = 0;

    virtual Tensor<1,3> curl_3D(const std::string& variableName,
                                const unsigned int i,
                                const unsigned int q) const = 0;
    */

    virtual const Point<dim>& quadrature_point (const unsigned int q) const = 0;
    virtual double JxW (const unsigned int q) const = 0;
    virtual const Tensor<1,dim>& normal_vector (const unsigned int q) const = 0;

    //virtual const Function<dim, double>&  function(const std::string& functionName) const = 0;
    //virtual const Function<dim, adouble>& adouble_function(const std::string& functionName) const = 0;

    virtual adouble dof(const std::string& variableName, const unsigned int i) const = 0;

    virtual adouble               dof_approximation(const std::string& variableName, const unsigned int q) const = 0;
    virtual Tensor<1,dim,adouble> dof_gradient_approximation(const std::string& variableName, const unsigned int q) const = 0;
    virtual Tensor<2,dim,adouble> dof_hessian_approximation(const std::string& variableName, const unsigned int q) const = 0;

    virtual Tensor<1,dim,adouble> vector_dof_approximation(const std::string& variableName, const unsigned int q) const = 0;
    virtual Tensor<2,dim,adouble> vector_dof_gradient_approximation(const std::string& variableName, const unsigned int q) const = 0;

    virtual unsigned int q() const = 0;
    virtual unsigned int i() const = 0;
    virtual unsigned int j() const = 0;
    virtual unsigned int component(unsigned int index) const = 0;
};

template<int dim>
class feNode
{
public:
    virtual ~feNode(void){}

public:
    virtual feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const = 0;
    virtual std::string          ToString() const                                       = 0;
};

template<int dim>
class feNode_constant : public feNode<dim>
{
public:
    feNode_constant(double value)
    {
        m_value = value;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        return feRuntimeNumber<dim>(m_value);
    }

    std::string ToString() const
    {
        if(m_value < 0)
            return (boost::format("(%f)") % m_value).str();
        else
            return (boost::format("%f") % m_value).str();
    }

public:
    double m_value;
};

template<int dim>
unsigned int getComponentIndex(int index, const feCellContext<dim>* pCellContext)
{
    unsigned int c_index;
    if(index == fe_i)
        c_index = pCellContext->i();
    else if(index == fe_j)
        c_index = pCellContext->j();
    else
        c_index = index;

    return pCellContext->component(c_index);
}

inline std::string getComponentIndex(int index)
{
    if(index == fe_i)
        return "i";
    else if(index == fe_j)
        return "j";
    else
        return (boost::format("%d") % index).str();
}

template<int dim>
unsigned int getIndex(int index, const feCellContext<dim>* pCellContext)
{
    if(index == fe_i)
        return pCellContext->i();
    else if(index == fe_j)
        return pCellContext->j();
    else if(index == fe_q)
        return pCellContext->q();
    else
        return index;
}

inline std::string getIndex(int index)
{
    if(index == fe_i)
        return "i";
    else if(index == fe_j)
        return "j";
    else if(index == fe_q)
        return "q";
    else
        return (boost::format("%d") % index).str();
}

template<int dim>
class feNode_phi : public feNode<dim>
{
public:
    feNode_phi(const std::string& variableName, int i, int q, unsigned int component = 0)
    {
        m_variableName = variableName;
        m_i = i;
        m_q = q;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int i = getIndex<dim>(m_i, pCellContext);
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->value(m_variableName, i, q) );
    }

    std::string ToString() const
    {
        return (boost::format("phi('%s', %d, %d)") % m_variableName % getIndex(m_i) % getIndex(m_q)).str();
    }

public:
    std::string    m_variableName;
    int            m_i;
    int            m_q;
    unsigned int   m_component;
};

template<int dim>
class feNode_dphi : public feNode<dim>
{
public:
    feNode_dphi(const std::string& variableName, int i, int q, unsigned int component = 0)
    {
        m_variableName = variableName;
        m_i = i;
        m_q = q;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int i = getIndex<dim>(m_i, pCellContext);
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->gradient(m_variableName, i, q) );
    }

    std::string ToString() const
    {
        return (boost::format("dphi('%s', %d, %d)") % m_variableName % getIndex(m_i) % getIndex(m_q)).str();
    }

public:
    std::string    m_variableName;
    int            m_i;
    int            m_q;
    unsigned int   m_component;
};

template<int dim>
class feNode_d2phi : public feNode<dim>
{
public:
    feNode_d2phi(const std::string& variableName, int i, int q, unsigned int component = 0)
    {
        m_variableName = variableName;
        m_i = i;
        m_q = q;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int i = getIndex<dim>(m_i, pCellContext);
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->hessian(m_variableName, i, q) );
    }

    std::string ToString() const
    {
        return (boost::format("d2phi('%s', %d, %d)") % m_variableName % getIndex(m_i) % getIndex(m_q)).str();
    }

public:
    std::string    m_variableName;
    int            m_i;
    int            m_q;
    unsigned int   m_component;
};



// Vector-data nodes
template<int dim>
class feNode_phi_vector : public feNode<dim>
{
public:
    feNode_phi_vector(const std::string& variableName, int i, int q, unsigned int component = 0)
    {
        m_variableName = variableName;
        m_i = i;
        m_q = q;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int i = getIndex<dim>(m_i, pCellContext);
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->vector_value(m_variableName, i, q) );
    }

    std::string ToString() const
    {
        return (boost::format("phi_vector('%s', %d, %d)") % m_variableName % getIndex(m_i) % getIndex(m_q)).str();
    }

public:
    std::string    m_variableName;
    int            m_i;
    int            m_q;
    unsigned int   m_component;
};

template<int dim>
class feNode_dphi_vector : public feNode<dim>
{
public:
    feNode_dphi_vector(const std::string& variableName, int i, int q, unsigned int component = 0)
    {
        m_variableName = variableName;
        m_i = i;
        m_q = q;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int i = getIndex<dim>(m_i, pCellContext);
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->vector_gradient(m_variableName, i, q) );
    }

    std::string ToString() const
    {
        return (boost::format("dphi_vector('%s', %d, %d)") % m_variableName % getIndex(m_i) % getIndex(m_q)).str();
    }

public:
    std::string    m_variableName;
    int            m_i;
    int            m_q;
    unsigned int   m_component;
};

template<int dim>
class feNode_d2phi_vector : public feNode<dim>
{
public:
    feNode_d2phi_vector(const std::string& variableName, int i, int q, unsigned int component = 0)
    {
        m_variableName = variableName;
        m_i = i;
        m_q = q;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int i = getIndex<dim>(m_i, pCellContext);
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->vector_hessian(m_variableName, i, q) );
    }

    std::string ToString() const
    {
        return (boost::format("d2phi_vector('%s', %d, %d)") % m_variableName % getIndex(m_i) % getIndex(m_q)).str();
    }

public:
    std::string    m_variableName;
    int            m_i;
    int            m_q;
    unsigned int   m_component;
};

template<int dim>
class feNode_div_phi : public feNode<dim>
{
public:
    feNode_div_phi(const std::string& variableName, int i, int q, unsigned int component = 0)
    {
        m_variableName = variableName;
        m_i = i;
        m_q = q;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int i = getIndex<dim>(m_i, pCellContext);
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->divergence(m_variableName, i, q) );
    }

    std::string ToString() const
    {
        return (boost::format("div_phi('%s', %d, %d)") % m_variableName % getIndex(m_i) % getIndex(m_q)).str();
    }

public:
    std::string    m_variableName;
    int            m_i;
    int            m_q;
    unsigned int   m_component;
};

template<int dim>
class feNode_symmetric_gradient : public feNode<dim>
{
public:
    feNode_symmetric_gradient(const std::string& variableName, int i, int q)
    {
        m_variableName = variableName;
        m_i = i;
        m_q = q;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int i = getIndex<dim>(m_i, pCellContext);
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->symmetric_gradient(m_variableName, i, q) );
    }

    std::string ToString() const
    {
        return (boost::format("symmetric_gradient('%s', %d, %d)") % m_variableName % getIndex(m_i) % getIndex(m_q)).str();
    }

public:
    std::string    m_variableName;
    int            m_i;
    int            m_q;
};

template<int dim>
class feNode_curl : public feNode<dim>
{
public:
    feNode_curl(const std::string& variableName, int i, int q, unsigned int component = 0)
    {
        if(dim == 1)
            throw std::runtime_error("Invalid call to curl() for the 1D system");

        m_variableName = variableName;
        m_i = i;
        m_q = q;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int i = getIndex<dim>(m_i, pCellContext);
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        if(dim == 2)
            return feRuntimeNumber<dim>( pCellContext->curl_2D(m_variableName, i, q) );
        else if(dim == 3)
            return feRuntimeNumber<dim>( pCellContext->curl_3D(m_variableName, i, q) );
        else
            throw std::runtime_error("Invalid call to curl() for the 1D system");
    }

    std::string ToString() const
    {
        return (boost::format("curl('%s', %d, %d)") % m_variableName % getIndex(m_i) % getIndex(m_q)).str();
    }

public:
    std::string    m_variableName;
    int            m_i;
    int            m_q;
    unsigned int   m_component;
};

template<int dim>
class feNode_JxW : public feNode<dim>
{
public:
    feNode_JxW(int q)
    {
        m_q = q;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->JxW(q) );
    }

    std::string ToString() const
    {
        return (boost::format("JxW(%d)") % getIndex(m_q)).str();
    }

public:
    int m_q;
};

template<int dim>
class feNode_xyz : public feNode<dim>
{
public:
    feNode_xyz(int q)
    {
        m_q = q;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->quadrature_point(q) );
    }

    std::string ToString() const
    {
        return (boost::format("xyz(%d)") % getIndex(m_q)).str();
    }

public:
    int m_q;
};


template<int dim>
class feNode_normal : public feNode<dim>
{
public:
    feNode_normal(int q)
    {
        m_q = q;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int q = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->normal_vector(q) );
    }

    std::string ToString() const
    {
        return (boost::format("normal(%d)") % getIndex(m_q)).str();
    }

public:
    int m_q;
};

/*
template<int dim>
class feNode_function : public feNode<dim>
{
public:
    typedef typename boost::shared_ptr< feNode<dim> > feNodePtr;

    feNode_function(const std::string& name, efeFunctionCall call, feNodePtr xyz_node, unsigned int component = 0)
    {
        if(!dynamic_cast<feNode_xyz<dim>*>(xyz_node.get()))
            throw std::runtime_error(std::string("An argument to the Function must be a point"));

        m_xyz_node = xyz_node;
        m_name = name;
        m_call = call;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        // According to the documentation for FiniteElement<dim,spacedim>::system_to_component_index(index):
        //  - for scalar elements, this returns always zero
        //  - for vector elements, this returns the corresponding vector component
        unsigned int component_index = getComponentIndex<dim>(m_component, pCellContext);

        feRuntimeNumber<dim> node = m_xyz_node->Evaluate(pCellContext);

        if(node.m_eType != eFEPoint)
            throw std::runtime_error(std::string("An argument to the Function must be a point"));

        if(m_call == eFunctionValue)
        {
            return feRuntimeNumber<dim>( pCellContext->function(m_name).value(node.m_point, component_index) );
        }
        else if(m_call == eFunctionGradient)
        {
            return feRuntimeNumber<dim>( pCellContext->function(m_name).gradient(node.m_point, component_index) );
        }
        else
        {
            throw std::runtime_error(std::string("Invalid Function call type"));
        }
    }

    std::string ToString() const
    {
        if(m_call == eFunctionValue)
        {
            return (boost::format("fvalue('%s', %s, %d)") % m_name % m_xyz_node->ToString() % getComponentIndex(m_component)).str();
        }
        else if(m_call == eFunctionGradient)
        {
            return (boost::format("fgrad('%s', %s, %d)") % m_name % m_xyz_node->ToString() % getComponentIndex(m_component)).str();
        }
        else
            throw std::runtime_error(std::string("Invalid Function call type"));
    }

public:
    feNodePtr       m_xyz_node;
    std::string     m_name;
    efeFunctionCall m_call;
    unsigned int    m_component;
};

template<int dim>
class feNode_adouble_function : public feNode<dim>
{
public:
    typedef typename boost::shared_ptr< feNode<dim> > feNodePtr;

    feNode_adouble_function(const std::string& name, efeFunctionCall call, feNodePtr xyz_node, unsigned int component = 0)
    {
        if(!dynamic_cast<feNode_xyz<dim>*>(xyz_node.get()))
            throw std::runtime_error(std::string("An argument to the Function must be a point"));

        m_xyz_node = xyz_node;
        m_name = name;
        m_call = call;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        // According to the documentation for FiniteElement<dim,spacedim>::system_to_component_index(index):
        //  - for scalar elements, this returns always zero
        //  - for vector elements, this returns the corresponding vector component
        unsigned int component_index = getComponentIndex<dim>(m_component, pCellContext);

        feRuntimeNumber<dim> node = m_xyz_node->Evaluate(pCellContext);

        if(node.m_eType != eFEPoint)
            throw std::runtime_error(std::string("An argument to the Function must be a point"));

        if(m_call == eFunctionValue)
        {
            return feRuntimeNumber<dim>( pCellContext->adouble_function(m_name).value(node.m_point, component_index) );
        }
        else if(m_call == eFunctionGradient)
        {
            return feRuntimeNumber<dim>( pCellContext->adouble_function(m_name).gradient(node.m_point, component_index) );
        }
        else
        {
            throw std::runtime_error(std::string("Invalid Function call type"));
        }

        return feRuntimeNumber<dim>();
    }

    std::string ToString() const
    {
        if(m_call == eFunctionValue)
        {
            return (boost::format("fvalue_adouble('%s', %s, %d)") % m_name % m_xyz_node->ToString() % getComponentIndex(m_component)).str();
        }
        else if(m_call == eFunctionGradient)
        {
            return (boost::format("fgrad_adouble('%s', %s, %d)") % m_name % m_xyz_node->ToString() % getComponentIndex(m_component)).str();
        }
        else
            throw std::runtime_error(std::string("Invalid Function call type"));
    }

public:
    feNodePtr       m_xyz_node;
    std::string     m_name;
    efeFunctionCall m_call;
    unsigned int    m_component;
};
*/


template<int dim, typename Number>
class feNode_function : public feNode<dim>
{
public:
    typedef typename boost::shared_ptr< feNode<dim> > feNodePtr;

    feNode_function(const std::string& name, const Function<dim,Number>& fun, efeFunctionCall call, feNodePtr xyz_node, unsigned int component = 0)
        : m_function(fun)
    {
        if(!dynamic_cast<feNode_xyz<dim>*>(xyz_node.get()))
            throw std::runtime_error(std::string("An argument of the function_value|gradient() must be a point"));

        m_name      = name;
        m_xyz_node  = xyz_node;
        m_call      = call;
        m_component = component;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        // According to the documentation for FiniteElement<dim,spacedim>::system_to_component_index(index):
        //  - for scalar elements, this returns always zero
        //  - for vector elements, this returns the corresponding vector component
        unsigned int component_index = getComponentIndex<dim>(m_component, pCellContext);

        feRuntimeNumber<dim> node = m_xyz_node->Evaluate(pCellContext);

        if(node.m_eType != eFEPoint)
            throw std::runtime_error(std::string("An argument to the Function must be a point"));

        if(m_call == eFunctionValue)
        {
            return feRuntimeNumber<dim>( m_function.value(node.m_point, component_index) );
        }
        else if(m_call == eFunctionGradient)
        {
            return feRuntimeNumber<dim>( m_function.gradient(node.m_point, component_index) );
        }
        else
        {
            throw std::runtime_error(std::string("Invalid Function call type"));
        }
    }

    std::string ToString() const
    {
        if(m_call == eFunctionValue)
        {
            return (boost::format("function_value('%s', %s)") % m_name % m_xyz_node->ToString()).str();
        }
        else if(m_call == eFunctionGradient)
        {
            return (boost::format("function_gradient('%s', %s)") % m_name % m_xyz_node->ToString()).str();
        }
        else
            throw std::runtime_error(std::string("Invalid Function call type"));
    }

public:
    feNodePtr                   m_xyz_node;
    std::string                 m_name;
    const Function<dim,Number>& m_function;
    efeFunctionCall             m_call;
    unsigned int                m_component;
};

template<int rank, int dim, typename Number>
class feNode_tensor_function : public feNode<dim>
{
public:
    typedef typename boost::shared_ptr< feNode<dim> > feNodePtr;

    feNode_tensor_function(const std::string& name, const TensorFunction<rank,dim,Number>& tfun, efeFunctionCall call, feNodePtr xyz_node)
        : m_tensor_function(tfun)
    {
        if(!dynamic_cast<feNode_xyz<dim>*>(xyz_node.get()))
            throw std::runtime_error(std::string("An argument to the TensorFunction must be a point"));

        m_name      = name;
        m_xyz_node  = xyz_node;
        m_call      = call;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        feRuntimeNumber<dim> node = m_xyz_node->Evaluate(pCellContext);

        if(node.m_eType != eFEPoint)
            throw std::runtime_error(std::string("An argument to the TensorFunction must be a point"));

        if(m_call == eFunctionValue)
        {
            return feRuntimeNumber<dim>( m_tensor_function.value(node.m_point) );
        }
        else if(m_call == eFunctionGradient)
        {
            return feRuntimeNumber<dim>( m_tensor_function.gradient(node.m_point) );
        }
        else
        {
            throw std::runtime_error(std::string("Invalid TensorFunction call type"));
        }
    }

    std::string ToString() const
    {
        if(m_call == eFunctionValue)
        {
            return (boost::format("tensor%d_function_value('%s', %s)") % rank % m_name % m_xyz_node->ToString()).str();
        }
        else if(m_call == eFunctionGradient)
        {
            return (boost::format("tensor%d_function_gradient('%s', %s)") % rank % m_name % m_xyz_node->ToString()).str();
        }
        else
            throw std::runtime_error(std::string("Invalid Function call type"));
    }

public:
    feNodePtr                              m_xyz_node;
    std::string                            m_name;
    const TensorFunction<rank,dim,Number>& m_tensor_function;
    efeFunctionCall                        m_call;
};

template<int dim>
class feNode_dof : public feNode<dim>
{
public:
    feNode_dof(const std::string& dofName, int i)
    {
        m_dofName  = dofName;
        m_i        = i;
    }
    
public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int index = getIndex<dim>(m_i, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->dof(m_dofName, index) );
    }
    
    std::string ToString() const
    {
        return (boost::format("dof('%s', %s)") % m_dofName % getIndex(m_i)).str();
    }
    
public:
    std::string  m_dofName;
    int          m_i;
};

template<int dim>
class feNode_vector_dof_approximation : public feNode<dim>
{
public:
    feNode_vector_dof_approximation(const std::string& dofName, int q)
    {
        m_dofName  = dofName;
        m_q        = q;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int index = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->vector_dof_approximation(m_dofName, index) );
    }

    std::string ToString() const
    {
        return (boost::format("dof_vector_approximation('%s', %s)") % m_dofName % getIndex(m_q)).str();
    }

public:
    std::string  m_dofName;
    int          m_q;
};

template<int dim>
class feNode_vector_dof_gradient_approximation : public feNode<dim>
{
public:
    feNode_vector_dof_gradient_approximation(const std::string& dofName, int q)
    {
        m_dofName  = dofName;
        m_q        = q;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int index = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->vector_dof_gradient_approximation(m_dofName, index) );
    }

    std::string ToString() const
    {
        return (boost::format("dof_vector_approximation('%s', %s)") % m_dofName % getIndex(m_q)).str();
    }

public:
    std::string  m_dofName;
    int          m_q;
};

template<int dim>
class feNode_dof_approximation : public feNode<dim>
{
public:
    feNode_dof_approximation(const std::string& dofName, int q)
    {
        m_dofName  = dofName;
        m_q        = q;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int index = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->dof_approximation(m_dofName, index) );
    }

    std::string ToString() const
    {
        return (boost::format("dof_approximation('%s', %s)") % m_dofName % getIndex(m_q)).str();
    }

public:
    std::string  m_dofName;
    int          m_q;
};

template<int dim>
class feNode_dof_gradient_approximation : public feNode<dim>
{
public:
    feNode_dof_gradient_approximation(const std::string& dofName, int q)
    {
        m_dofName  = dofName;
        m_q        = q;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int index = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->dof_gradient_approximation(m_dofName, index) );
    }

    std::string ToString() const
    {
        return (boost::format("dof_gradient_approximation('%s', %s)") % m_dofName % getIndex(m_q)).str();
    }

public:
    std::string  m_dofName;
    int          m_q;
};

template<int dim>
class feNode_dof_hessian_approximation : public feNode<dim>
{
public:
    feNode_dof_hessian_approximation(const std::string& dofName, int q)
    {
        m_dofName  = dofName;
        m_q        = q;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        unsigned int index = getIndex<dim>(m_q, pCellContext);
        return feRuntimeNumber<dim>( pCellContext->dof_hessian_approximation(m_dofName, index) );
    }

    std::string ToString() const
    {
        return (boost::format("dof_laplacian_approximation('%s', %s)") % m_dofName % getIndex(m_q)).str();
    }

public:
    std::string  m_dofName;
    int          m_q;
};

template<int dim>
class feNode_adouble : public feNode<dim>
{
public:
    feNode_adouble(const adouble& ad)
    {
        m_ad = ad;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        return feRuntimeNumber<dim>( m_ad );
    }

    std::string ToString() const
    {
        std::string res = m_ad.NodeAsPlainText();
        return (boost::format("adouble(%s)") % res).str();
    }

public:
    adouble m_ad;
};

template<int rank, int dim>
class feNode_tensor : public feNode<dim>
{
public:
    feNode_tensor(const Tensor<rank,dim,double>& t)
    {
        m_tensor = t;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        return feRuntimeNumber<dim>( m_tensor );
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << m_tensor;
        std::string res = ss.str();
        return (boost::format("tensor%d(%s)") % rank % res).str();
    }

public:
    Tensor<rank,dim,double> m_tensor;
};

template<int rank, int dim>
class feNode_adouble_tensor : public feNode<dim>
{
public:
    feNode_adouble_tensor(const Tensor<rank,dim,adouble>& t)
    {
        m_tensor = t;
    }

public:
    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        return feRuntimeNumber<dim>( m_tensor );
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << m_tensor;
        std::string res = ss.str();
        return (boost::format("adouble_tensor%d(%s)") % rank % res).str();
    }

public:
    Tensor<rank,dim,adouble> m_tensor;
};

enum efeUnaryFunction
{
    eSign,
    eSqrt,
    eExp,
    eLog,
    eLog10,
    eAbs,
    eSin,
    eCos,
    eTan,
    eArcSin,
    eArcCos,
    eArcTan,
    eSinh,
    eCosh,
    eTanh,
    eArcSinh,
    eArcCosh,
    eArcTanh,
    eErf
};

enum efeBinaryFunction
{
    ePlus,
    eMinus,
    eMulti,
    eDivide,
    ePower
};

template<int dim>
class feNode_unary : public feNode<dim>
{
public:
    typedef typename boost::shared_ptr< feNode<dim> > feNodePtr;

    feNode_unary(efeUnaryFunction function, feNodePtr node)
    {
        m_function = function;
        m_node     = node;
    }

    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        feRuntimeNumber<dim> node = m_node->Evaluate(pCellContext);

        if(m_function == eSign)
            return -node;
        else if(m_function == eSqrt)
            return sqrt_(node);
        else if(m_function == eExp)
            return exp_(node);
        else if(m_function == eLog)
            return log_(node);
        else if(m_function == eLog10)
            return log10_(node);
        else if(m_function == eAbs)
            return abs_(node);
        else if(m_function == eSin)
            return sin_(node);
        else if(m_function == eCos)
            return cos_(node);
        else if(m_function == eTan)
            return tan_(node);
        else if(m_function == eArcSin)
            return asin_(node);
        else if(m_function == eArcCos)
            return acos_(node);
        else if(m_function == eArcTan)
            return atan_(node);
        else if(m_function == eSinh)
            return sinh_(node);
        else if(m_function == eCosh)
            return cosh_(node);
        else if(m_function == eTanh)
            return tanh_(node);
        else if(m_function == eArcSinh)
            return asinh_(node);
        else if(m_function == eArcCosh)
            return acosh_(node);
        else if(m_function == eArcTanh)
            return atanh_(node);
        else if(m_function == eErf)
            return erf_(node);
        else
            throw std::runtime_error(std::string("Invalid unary function"));
    }

    std::string ToString() const
    {
        if(m_function == eSign)
            return (boost::format("-(%s)") % m_node->ToString()).str();
        else if(m_function == eSqrt)
            return (boost::format("sqrt(%s)") % m_node->ToString()).str();
        else if(m_function == eExp)
            return (boost::format("exp(%s)") % m_node->ToString()).str();
        else if(m_function == eLog)
            return (boost::format("log(%s)") % m_node->ToString()).str();
        else if(m_function == eLog10)
            return (boost::format("log10(%s)") % m_node->ToString()).str();
        else if(m_function == eAbs)
            return (boost::format("abs(%s)") % m_node->ToString()).str();
        else if(m_function == eSin)
            return (boost::format("sin(%s)") % m_node->ToString()).str();
        else if(m_function == eCos)
            return (boost::format("cos(%s)") % m_node->ToString()).str();
        else if(m_function == eTan)
            return (boost::format("tan(%s)") % m_node->ToString()).str();
        else if(m_function == eArcSin)
            return (boost::format("asin(%s)") % m_node->ToString()).str();
        else if(m_function == eArcCos)
            return (boost::format("acos(%s)") % m_node->ToString()).str();
        else if(m_function == eArcTan)
            return (boost::format("atan(%s)") % m_node->ToString()).str();
        else if(m_function == eSinh)
            return (boost::format("sinh(%s)") % m_node->ToString()).str();
        else if(m_function == eCosh)
            return (boost::format("cosh(%s)") % m_node->ToString()).str();
        else if(m_function == eTanh)
            return (boost::format("tanh(%s)") % m_node->ToString()).str();
        else if(m_function == eArcSinh)
            return (boost::format("asinh(%s)") % m_node->ToString()).str();
        else if(m_function == eArcCosh)
            return (boost::format("acosh(%s)") % m_node->ToString()).str();
        else if(m_function == eArcTanh)
            return (boost::format("atanh(%s)") % m_node->ToString()).str();
        else if(m_function == eErf)
            return (boost::format("erf(%s)") % m_node->ToString()).str();
        else
            throw std::runtime_error(std::string("Invalid unary function"));
    }

public:
    efeUnaryFunction m_function;
    feNodePtr        m_node;
};

template<int dim>
class feNode_binary : public feNode<dim>
{
public:
    typedef typename boost::shared_ptr< feNode<dim> > feNodePtr;

    feNode_binary(efeBinaryFunction function, feNodePtr left, feNodePtr right)
    {
        m_function = function;
        m_left     = left;
        m_right    = right;
    }

    feRuntimeNumber<dim> Evaluate(const feCellContext<dim>* pCellContext) const
    {
        feRuntimeNumber<dim> left  = m_left->Evaluate(pCellContext);
        feRuntimeNumber<dim> right = m_right->Evaluate(pCellContext);

        if(m_function == ePlus)
            return left + right;
        else if(m_function == eMinus)
            return left - right;
        else if(m_function == eMulti)
            return left * right;
        else if(m_function == eDivide)
            return left / right;
        else if(m_function == ePower)
            return pow(left, right);

        throw std::runtime_error(std::string("Invalid binary function"));
    }

    std::string ToString() const
    {
        if(m_function == ePlus)
            return (boost::format("(%s + %s)") % m_left->ToString() % m_right->ToString()).str();
        else if(m_function == eMinus)
            return (boost::format("(%s - %s)") % m_left->ToString() % m_right->ToString()).str();
        else if(m_function == eMulti)
            return (boost::format("(%s * %s)") % m_left->ToString() % m_right->ToString()).str();
        else if(m_function == eDivide)
            return (boost::format("(%s / %s)") % m_left->ToString() % m_right->ToString()).str();
        else if(m_function == ePower)
            return (boost::format("(%s ** %s)") % m_left->ToString() % m_right->ToString()).str();

        throw std::runtime_error(std::string("Invalid unary function"));
    }

public:
    efeBinaryFunction m_function;
    feNodePtr         m_left;
    feNodePtr         m_right;
};

template<int dim>
feExpression<dim> operator +(const feExpression<dim>& l, const feExpression<dim>& r)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(ePlus, l.m_node, r.m_node) ));
}

template<int dim>
feExpression<dim> operator -(const feExpression<dim>& l, const feExpression<dim>& r)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(eMinus, l.m_node, r.m_node) ));
}

template<int dim>
feExpression<dim> operator *(const feExpression<dim>& l, const feExpression<dim>& r)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(eMulti, l.m_node, r.m_node) ));
}

template<int dim>
feExpression<dim> operator /(const feExpression<dim>& l, const feExpression<dim>& r)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(eDivide, l.m_node, r.m_node) ));
}

template<int dim>
feExpression<dim> operator -(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eSign, fe.m_node) ));
}

template<int dim>
feExpression<dim> operator +(const feExpression<dim>& fe)
{
    return fe;
}

template<int dim>
feExpression<dim> operator +(const feExpression<dim>& l, double r)
{
    typename feExpression<dim>::feNodePtr rnode( new feNode_constant<dim>(r) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(ePlus, l.m_node, rnode) ));
}

template<int dim>
feExpression<dim> operator -(const feExpression<dim>& l, double r)
{
    typename feExpression<dim>::feNodePtr rnode( new feNode_constant<dim>(r) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(eMinus, l.m_node, rnode) ));
}

template<int dim>
feExpression<dim> operator *(const feExpression<dim>& l, double r)
{
    typename feExpression<dim>::feNodePtr rnode( new feNode_constant<dim>(r) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(eMulti, l.m_node, rnode) ));
}

template<int dim>
feExpression<dim> operator /(const feExpression<dim>& l, double r)
{
    typename feExpression<dim>::feNodePtr rnode( new feNode_constant<dim>(r) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(eDivide, l.m_node, rnode) ));
}

template<int dim>
feExpression<dim> operator ^(const feExpression<dim>& l, double r)
{
    typename feExpression<dim>::feNodePtr rnode( new feNode_constant<dim>(r) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(ePower, l.m_node, rnode) ));
}

template<int dim>
feExpression<dim> pow(const feExpression<dim>& l, double r)
{
    typename feExpression<dim>::feNodePtr rnode( new feNode_constant<dim>(r) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(ePower, l.m_node, rnode) ));
}

template<int dim>
feExpression<dim> pow(const feExpression<dim>& l, const feExpression<dim>& r)
{
    if( !dynamic_cast<feNode_constant<dim>*>(r.m_node.get()) )
        throw std::runtime_error(std::string("pow function accepts only float powers"));

    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(ePower, l.m_node, r.m_node) ));
}


template<int dim>
feExpression<dim> operator +(double l, const feExpression<dim>& r)
{
    typename feExpression<dim>::feNodePtr lnode( new feNode_constant<dim>(l) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(ePlus, lnode, r.m_node) ));
}

template<int dim>
feExpression<dim> operator -(double l, const feExpression<dim>& r)
{
    typename feExpression<dim>::feNodePtr lnode( new feNode_constant<dim>(l) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(eMinus, lnode, r.m_node) ));
}

template<int dim>
feExpression<dim> operator *(double l, const feExpression<dim>& r)
{
    typename feExpression<dim>::feNodePtr lnode( new feNode_constant<dim>(l) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(eMulti, lnode, r.m_node) ));
}

template<int dim>
feExpression<dim> operator /(double l, const feExpression<dim>& r)
{
    typename feExpression<dim>::feNodePtr lnode( new feNode_constant<dim>(l) );
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_binary<dim>(eDivide, lnode, r.m_node) ));
}


template<int dim>
feExpression<dim> sqrt(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eSqrt, fe.m_node) ));
}

template<int dim>
feExpression<dim> exp(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eExp, fe.m_node) ));
}

template<int dim>
feExpression<dim> log(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eLog, fe.m_node) ));
}

template<int dim>
feExpression<dim> log10(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eLog10, fe.m_node) ));
}

template<int dim>
feExpression<dim> abs(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eAbs, fe.m_node) ));
}

template<int dim>
feExpression<dim> sin(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eSin, fe.m_node) ));
}

template<int dim>
feExpression<dim> cos(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eCos, fe.m_node) ));
}

template<int dim>
feExpression<dim> tan(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eTan, fe.m_node) ));
}

template<int dim>
feExpression<dim> asin(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eArcSin, fe.m_node) ));
}

template<int dim>
feExpression<dim> acos(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eArcCos, fe.m_node) ));
}

template<int dim>
feExpression<dim> atan(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eArcTan, fe.m_node) ));
}



template<int dim>
feExpression<dim> sinh(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eSinh, fe.m_node) ));
}

template<int dim>
feExpression<dim> cosh(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eCosh, fe.m_node) ));
}

template<int dim>
feExpression<dim> tanh(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eTanh, fe.m_node) ));
}

template<int dim>
feExpression<dim> asinh(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eArcSinh, fe.m_node) ));
}

template<int dim>
feExpression<dim> acosh(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eArcCosh, fe.m_node) ));
}

template<int dim>
feExpression<dim> atanh(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eArcTanh, fe.m_node) ));
}



template<int dim>
feExpression<dim> erf(const feExpression<dim>& fe)
{
    return feExpression<dim>(typename feExpression<dim>::feNodePtr( new feNode_unary<dim>(eErf, fe.m_node) ));
}



template<int dim>
feExpression<dim> constant(double value)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_constant<dim>(value) ) );
}

// Scalar-data functions
template<int dim>
feExpression<dim> phi(const std::string& variableName, int i, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_phi<dim>(variableName, i, q) ) );
}

template<int dim>
feExpression<dim> dphi(const std::string& variableName, int i, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_dphi<dim>(variableName, i, q) ) );
}

template<int dim>
feExpression<dim> d2phi(const std::string& variableName, int i, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_d2phi<dim>(variableName, i, q) ) );
}


// Vector-data functions
template<int dim>
feExpression<dim> phi_vector(const std::string& variableName, int i, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_phi_vector<dim>(variableName, i, q) ) );
}

template<int dim>
feExpression<dim> dphi_vector(const std::string& variableName, int i, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_dphi_vector<dim>(variableName, i, q) ) );
}

template<int dim>
feExpression<dim> d2phi_vector(const std::string& variableName, int i, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_d2phi_vector<dim>(variableName, i, q) ) );
}

template<int dim>
feExpression<dim> div_phi(const std::string& variableName, int i, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_div_phi<dim>(variableName, i, q) ) );
}

template<int dim>
feExpression<dim> symmetric_gradient(const std::string& variableName, int i, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_symmetric_gradient<dim>(variableName, i, q) ) );
}

/* CURL
template<int dim>
feExpression<dim> curl(const std::string& variableName, int i, int q)
{
    if(dim == 1)
        throw std::runtime_error("Invalid call to curl() when the system is 1D");

    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_curl<dim>(variableName, i, q) ) );
}
*/


template<int dim>
feExpression<dim> xyz(int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_xyz<dim>(q) ) );
}

template<int dim>
feExpression<dim> JxW(int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_JxW<dim>(q) ) );
}

template<int dim>
feExpression<dim> normal(int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_normal<dim>(q) ) );
}

/*
 template<int dim>
feExpression<dim> function_value(const std::string& name, const feExpression<dim>& xyz, unsigned int component = 0)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_function<dim>(name, eFunctionValue, xyz.m_node, component) ) );
}

template<int dim>
feExpression<dim> function_gradient(const std::string& name, const feExpression<dim>& xyz, unsigned int component = 0)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_function<dim>(name, eFunctionGradient, xyz.m_node, component) ) );
}

template<int dim>
feExpression<dim> function_adouble_value(const std::string& name, const feExpression<dim>& xyz, unsigned int component = 0)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_adouble_function<dim>(name, eFunctionValue, xyz.m_node, component) ) );
}

template<int dim>
feExpression<dim> function_adouble_gradient(const std::string& name, const feExpression<dim>& xyz, unsigned int component = 0)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_adouble_function<dim>(name, eFunctionGradient, xyz.m_node, component) ) );
}
*/

template<int dim, typename Number>
feExpression<dim> function_value(const std::string& name, const Function<dim,Number>& fun, const feExpression<dim>& xyz, unsigned int component = 0)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_function<dim,Number>(name, fun, eFunctionValue, xyz.m_node, component) ) );
}
template<int dim, typename Number>
feExpression<dim> function_gradient(const std::string& name, const Function<dim,Number>& fun, const feExpression<dim>& xyz, unsigned int component = 0)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_function<dim,Number>(name, fun, eFunctionGradient, xyz.m_node, component) ) );
}


template<int rank, int dim, typename Number>
feExpression<dim> tensor_function_value(const std::string& name, const TensorFunction<rank,dim,Number>& tfun, const feExpression<dim>& xyz)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_tensor_function<rank,dim,Number>(name, tfun, eFunctionValue, xyz.m_node) ) );
}
template<int rank, int dim, typename Number>
feExpression<dim> tensor_function_gradient(const std::string& name, const TensorFunction<rank,dim,Number>& tfun, const feExpression<dim>& xyz)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_tensor_function<rank,dim,Number>(name, tfun, eFunctionGradient, xyz.m_node) ) );
}

template<int dim>
feExpression<dim> dof(const std::string& variableName, int i)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_dof<dim>(variableName, i) ) );
}

template<int dim>
feExpression<dim> dof_approximation(const std::string& variableName, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_dof_approximation<dim>(variableName, q) ) );
}

template<int dim>
feExpression<dim> dof_gradient_approximation(const std::string& variableName, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_dof_gradient_approximation<dim>(variableName, q) ) );
}

template<int dim>
feExpression<dim> dof_hessian_approximation(const std::string& variableName, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_dof_hessian_approximation<dim>(variableName, q) ) );
}

template<int dim>
feExpression<dim> adouble_(const adouble& ad)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_adouble<dim>(ad) ) );
}

template<int dim>
feExpression<dim> tensor1(const Tensor<1,dim,double>& t1)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_tensor<1,dim>(t1) ) );
}

template<int dim>
feExpression<dim> tensor2(const Tensor<2,dim,double>& t2)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_tensor<2,dim>(t2) ) );
}

template<int dim>
feExpression<dim> tensor3(const Tensor<3,dim,double>& t3)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_tensor<3,dim>(t3) ) );
}

template<int dim>
feExpression<dim> adouble_tensor1(const Tensor<1,dim,adouble>& t1)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_adouble_tensor<1,dim>(t1) ) );
}

template<int dim>
feExpression<dim> adouble_tensor2(const Tensor<2,dim,adouble>& t2)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_adouble_tensor<2,dim>(t2) ) );
}

template<int dim>
feExpression<dim> adouble_tensor3(const Tensor<3,dim,adouble>& t3)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_adouble_tensor<3,dim>(t3) ) );
}

template<int dim>
feExpression<dim> vector_dof_approximation(const std::string& variableName, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_vector_dof_approximation<dim>(variableName, q) ) );
}

template<int dim>
feExpression<dim> vector_dof_gradient_approximation(const std::string& variableName, int q)
{
    return feExpression<dim>( typename feExpression<dim>::feNodePtr( new feNode_vector_dof_gradient_approximation<dim>(variableName, q) ) );
}


}
}

#endif
