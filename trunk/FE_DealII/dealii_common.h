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
 * SingleValue_Function
 * Necessary for Diffusion/Generation/BC terms
 *********************************************************/
template <int dim>
class SingleValue_Function : public Function<dim>
{
public:
    SingleValue_Function(double value = 0.0) : Function<dim>()
    {
        m_value = value;
    }

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const
    {
        return m_value;
    }

public:
    double m_value;
};

inline adouble create_adouble(adNode* n)
{
    return adouble(0.0, 0.0, true, n);
}

/*********************************************************************
   daeDealIIDataReporter
*********************************************************************/
typedef boost::function<void (const daeDataReporterVariableValue*, double, const std::string&)> fnDataOut;
class daeDealIIDataReporter : public daeDataReporterLocal
{
public:
    daeDealIIDataReporter(fnDataOut f, const std::string& strDirectory)
        : fn(f), m_strDirectory(strDirectory)
    {
    }

    virtual ~daeDealIIDataReporter(void)
    {
    }

public:
    virtual std::string GetName() const {return "DealIIDataReporter";}
    virtual bool Connect(const string& strConnectString, const string& strProcessName){return true;}
    virtual bool Disconnect(void){return true;}
    virtual bool IsConnected(void){return true;}
    virtual bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
    {
        fn(pVariableValue, m_dCurrentTime, m_strDirectory);
        return true;
    }

public:
    std::string m_strDirectory;
    fnDataOut   fn;
};

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
