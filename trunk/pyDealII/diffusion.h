#ifndef DAE_TRANSITIONAL_DIFFUSION_H
#define DAE_TRANSITIONAL_DIFFUSION_H

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

#include "../dae_develop.h"
#include "../variable_types.h"
#include "../Core/nodes.h"
namespace vt = variable_types;

using units_pool::m;
using units_pool::kg;
using units_pool::K;
using units_pool::J;
using units_pool::W;
using units_pool::s;

#include "dealii_diffusion.h"


namespace diffusion
{
/*********************************************************
 * daeFEMatrix
 * A wrapper around deal.II SparseMatrix<double>
 *********************************************************/
class daeFEMatrix : public daeMatrix<double>
{
public:
    daeFEMatrix(SparseMatrix<double>& matrix) : deal_ii_matrix(matrix)
    {
    }

    virtual ~daeFEMatrix(void)
    {
    }

public:
    virtual double GetItem(size_t row, size_t col) const
    {
        return deal_ii_matrix(row, col);
    }

    virtual void SetItem(size_t row, size_t col, double value)
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
    SparseMatrix<double>& deal_ii_matrix;
};

/*********************************************************
 * daeFEArray
 * A wrapper around deal.II Vector<double>
 *********************************************************/
class daeFEArray : public daeArray<double>
{
public:
    daeFEArray(Vector<double>& vect) : deal_ii_vector(vect)
    {
    }

    virtual ~daeFEArray(void)
    {
    }

public:
    double operator [](size_t i) const
    {
        return deal_ii_vector[i];
    }

    double GetItem(size_t i) const
    {
        return deal_ii_vector[i];
    }

    void SetItem(size_t i, double value)
    {
        deal_ii_vector[i] = value;
    }

    size_t GetSize(void) const
    {
        return deal_ii_vector.size();
    }

protected:
    Vector<double>& deal_ii_vector;
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

/*********************************************************
 *     daeDiffusion
 *********************************************************/
template <int dim>
class daeDiffusion : public daeModel
{
typedef typename boost::shared_ptr< Function<dim> > FunctionPtr;

public:
    // D is diffusivity
    daeDiffusion(string strName, daeModel* pModel = NULL, string strDescription = "")
        : daeModel(strName, pModel, strDescription),

          omega("&Omega;", this, unit(), "Omega domain"),
          T("T", vt::temperature_t, this, "Temperature, K", &omega)

    {
    }

    void Initialize(string meshFilename,
                    unsigned int polynomialOrder,
                    double diffusivity,
                    const std::vector<double>& velocity,
                    double generation,
                    const std::map<unsigned int, double>& dirichletBC,
                    const std::map<unsigned int, double>& neumannBC)
    {
        //dirichletBC[0] = 1.0;
        //neumannBC[1]   = 0.5;

        funDiffusivity.reset(new SingleValue_Function<dim>(diffusivity));
        funVelocity.reset(new SingleValue_Function<dim>(0.0));
        funGeneration.reset(new SingleValue_Function<dim>(generation));

        for(std::map<unsigned int, double>::const_iterator it = dirichletBC.begin(); it != dirichletBC.end(); it++)
        {
            const unsigned int id    = it->first;
            const double       value = it->second;

            funsDirichletBC[id] = FunctionPtr(new SingleValue_Function<dim>(value));
        }

        for(std::map<unsigned int, double>::const_iterator it = neumannBC.begin(); it != neumannBC.end(); it++)
        {
            const unsigned int id    = it->first;
            const double       value = it->second;

            funsNeumannBC[id] = FunctionPtr(new SingleValue_Function<dim>(value));
        }

        // 1. Create deal.II solver
        pDealII.reset(new dealiiDiffusion<dim>(polynomialOrder, funDiffusivity, funVelocity, funGeneration, funsDirichletBC, funsNeumannBC));

        // 2. Initialize daetools wrapper matrices and arrays that will be used by adFEMatrixItem/VectorItem nodes
        matK.reset(new daeFEMatrix(pDealII->system_matrix));
        matKdt.reset(new daeFEMatrix(pDealII->system_matrix_dt));
        vecf.reset(new daeFEArray(pDealII->system_rhs));

        // 3. Load mesh from the supplied file
        GridIn<dim> gridin;
        gridin.attach_triangulation(pDealII->triangulation);
        std::ifstream f(meshFilename);
        gridin.read_msh(f);

        // 4. Setup deal.II system
        pDealII->setup_system();

        // 5. Assemble deal.II system
        pDealII->assemble_system();

        // 6. Initialize domain Omega
        std::vector<daePoint> coords;
        size_t n = pDealII->dof_handler.n_dofs();
        coords.resize(n);

        const unsigned int dofs_per_cell = pDealII->fe->dofs_per_cell;
        std::vector<unsigned int> local_dof_indices (dofs_per_cell);

        std::cout << "dofs_per_cell = " << dofs_per_cell << std::endl;

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

        //std::cout << "coords = " << std::endl;
        //for(unsigned int i = 0; i < coords.size(); i++)
        //    std::cout << boost::get<0>(coords[i]) << ", " << boost::get<1>(coords[i]) << ", " << boost::get<2>(coords[i]) << std::endl;
        //std::cout << std::endl;

        omega.CreateUnstructuredGrid(coords);
    }

    adouble create_adouble(adNode* n)
    {
        return adouble(0.0, 0.0, true, n);
    }

    void DeclareEquations(void)
    {
        daeEquation* eq;
        adouble a_K, a_Kdt, a_f;

        size_t nrows = pDealII->system_matrix.n();

        std::cout << "system_matrix" << std::endl;
        pDealII->system_matrix.print(std::cout);
        std::cout << "system_matrix_dt" << std::endl;
        pDealII->system_matrix_dt.print(std::cout);
        std::cout << "system_rhs" << std::endl;
        pDealII->system_rhs.print(std::cout);

        for(size_t row = 0; row < nrows; row++)
        {
            eq = this->CreateEquation("eq_" + toString(row));

            // Reset equation's contributions
            a_K   = 0;
            a_Kdt = 0;

            // RHS
            a_f.node = adNodePtr(new adFEVectorItemNode("f", *vecf, row, unit()));

            // K and Kdt matrices
            for(SparsityPattern::iterator iter = pDealII->sparsity_pattern.begin(row); iter != pDealII->sparsity_pattern.end(row); iter++)
            {
                const size_t col = (*iter).column();

                if(!a_K.node)
                    a_K =       create_adouble(new adFEMatrixItemNode("K", *matK, row, col, unit())) * this->T(col);
                else
                    a_K = a_K + create_adouble(new adFEMatrixItemNode("K", *matK, row, col, unit())) * this->T(col);

                //if(!a_Kdt.node)
                //    a_Kdt =         create_adouble(new adFEMatrixItemNode("", *matKdt, row, col, unit())) * this->T.dt(col);
                //else
                //    a_Kdt = a_Kdt + create_adouble(new adFEMatrixItemNode("", *matKdt, row, col, unit())) * this->T.dt(col);
            }

            eq->SetResidual(/*a_Kdt +*/ a_K - a_f);
            eq->SetCheckUnitsConsistency(false);
        }

    }

    void UpdateEquations(const daeExecutionContext* pExecutionContext)
    {
        daeModel::UpdateEquations(pExecutionContext);

        // Here we have to call pDealII->assemble_system() to update the system matrix and the system RHS vector
    }

public:
    daeDomain   omega;
    daeVariable T;

    boost::shared_ptr< dealiiDiffusion<dim> > pDealII;

    boost::shared_ptr<daeFEMatrix> matK;
    boost::shared_ptr<daeFEMatrix> matKdt;
    boost::shared_ptr<daeFEArray>  vecf;

    FunctionPtr                          funDiffusivity;
    FunctionPtr                          funVelocity;
    FunctionPtr                          funGeneration;
    std::map<unsigned int, FunctionPtr>  funsDirichletBC;
    std::map<unsigned int, FunctionPtr>  funsNeumannBC;
};

typedef daeDiffusion<1> daeDiffusion_1D;
typedef daeDiffusion<2> daeDiffusion_2D;
typedef daeDiffusion<3> daeDiffusion_3D;

}

#endif
