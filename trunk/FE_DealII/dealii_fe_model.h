#ifndef DEAL_II_MODEL_BASE_H
#define DEAL_II_MODEL_BASE_H

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/base/function.h>

namespace dae
{
namespace fe_solver
{
using namespace dealii;

// This type will be held in daeModel and sent to deal.II
// It cannot be Function<dim> for we cannot use it with map_indexing_suite
// because it is an abstract class (the destructor is abstract function).
template <int dim>
class dealiiFunction : public Function<dim>
{
public:
    dealiiFunction(const unsigned int n_components = 1) : Function<dim>(n_components)
    {
    }
};

/******************************************************************
    daeFiniteElementsModel_dealII
*******************************************************************/
class daeFiniteElementsModel_dealII
{
public:
    virtual ~daeFiniteElementsModel_dealII() {}

    virtual void assemble_system() = 0;
    virtual void finalize_solution_and_save(const std::string& strOutputDirectory,
                                            const std::string& strFormat,
                                            const std::string& strVariableName,
                                            const double* values,
                                            size_t n,
                                            double time) = 0;

public:
    SparsityPattern        sparsity_pattern;
    SparseMatrix<double>   system_matrix;
    SparseMatrix<double>   system_matrix_dt;
    Vector<double>         system_rhs;
    Vector<double>         solution;
};

}
}

#endif
