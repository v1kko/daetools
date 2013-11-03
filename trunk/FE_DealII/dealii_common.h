#ifndef DAE_DEALII_COMMON_H
#define DAE_DEALII_COMMON_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include "../Core/definitions.h"

namespace dae
{
namespace fe_solver
{
using namespace dealii;
/*********************************************************
 * deal.II related classes and typedefs
 *********************************************************/
typedef Function<1> Function_1D;
typedef Function<2> Function_2D;
typedef Function<3> Function_3D;

// Tensors of rank=1 are used for a gradient of scalar functions
typedef Tensor<1, 1, double> Tensor_1_1D;
typedef Tensor<1, 2, double> Tensor_1_2D;
typedef Tensor<1, 3, double> Tensor_1_3D;

// Points are in fact tensors with rank=1 just their coordinates mean length
// and have some additional functions
typedef Point<1, double> Point_1D;
typedef Point<2, double> Point_2D;
typedef Point<3, double> Point_3D;

///******************************************************************
//    dealiiFunction
//*******************************************************************/
//template <int dim>
//class dealiiFunction : public Function<dim>
//{
//public:
//    dealiiFunction(const unsigned int n_components = 1) : Function<dim>(n_components)
//    {
//    }
//};

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

}
}

#endif
