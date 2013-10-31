#ifndef DAE_DEALII_COMMON_H
#define DAE_DEALII_COMMON_H

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/data_out.h>

#include "../dae_develop.h"
#include "../variable_types.h"
#include "../Core/nodes.h"

namespace dae
{
namespace fe_solver
{
using namespace dae::core;
namespace vt = variable_types;
using namespace dealii;

/*********************************************************
 * daeFEMatrix
 * A wrapper around deal.II SparseMatrix<double>
 *********************************************************/
template<typename REAL = double>
class daeFEMatrix : public daeMatrix<REAL>
{
public:
    daeFEMatrix(SparseMatrix<REAL>& matrix) : deal_ii_matrix(matrix)
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
    SparseMatrix<REAL>& deal_ii_matrix;
};

/*********************************************************
 * daeFEArray
 * A wrapper around deal.II Vector<REAL>
 *********************************************************/
template<typename REAL = double>
class daeFEArray : public daeArray<REAL>
{
public:
    daeFEArray(Vector<REAL>& vect) : deal_ii_vector(vect)
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
        deal_ii_vector[i] = value;
    }

    size_t GetSize(void) const
    {
        return deal_ii_vector.size();
    }

protected:
    Vector<REAL>& deal_ii_vector;
};

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

// This type will be held in daeModel and sent to deal.II
// It cannot be Function<dim> for we cannot use it with map_indexing_suite
// because it is an abstract class (the destructor is abstract function).
template <int dim>
class dealiiFunction : public ZeroFunction<dim>
{
public:
    dealiiFunction(const unsigned int n_components = 1) : ZeroFunction<dim>(n_components)
    {
    }
};

inline adouble create_adouble(adNode* n)
{
    return adouble(0.0, 0.0, true, n);
}

/*
const unsigned int dofs_per_cell = pDealII->fe->dofs_per_cell;
std::vector<unsigned int> local_dof_indices (dofs_per_cell);
typename DoFHandler<dim>::active_cell_iterator cell = pDealII->dof_handler.begin_active(), endc = pDealII->dof_handler.end();
for (; cell!=endc; ++cell)
{
    cell->get_dof_indices (local_dof_indices);

    for(unsigned int v = 0; v < GeometryInfo<dim >::vertices_per_cell; ++v)
    {
        Point<dim> p = cell->vertex(v);
        unsigned int dof = local_dof_indices[v];
        //string msg = "dof[%d] - vertex[%d]: (%f, %f, %f)";
        //std::cout << (boost::format(msg) % local_dof_indices[v] % v % p(0) % p(1) % p(2)).str() << std::endl;

        if(dim == 1)
            coords[dof] = daePoint(p(0), 0.0, 0.0);
        else if(dim == 2)
            coords[dof] = daePoint(p(0), p(1), 0.0);
        else if(dim == 3)
            coords[dof] = daePoint(p(0), p(1), p(2));
    }
}
*/

//std::cout << "coords = " << std::endl;
//for(unsigned int i = 0; i < coords.size(); i++)
//    std::cout << boost::get<0>(coords[i]) << ", " << boost::get<1>(coords[i]) << ", " << boost::get<2>(coords[i]) << std::endl;
//std::cout << std::endl;


}
}

#endif
