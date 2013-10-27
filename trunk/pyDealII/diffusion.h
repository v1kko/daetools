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
template <int dim>
class daeDiffusion : public daeModel
{
public:
    // D is diffusivity
    daeDiffusion(string strName, daeModel* pModel = NULL, string strDescription = "")
        : daeModel(strName, pModel, strDescription),

          omega("&Omega;", this, unit(), "Omega domain"),
          T("T", vt::temperature_t, this, "Temperature, K", &omega)

    {
    }

    void Initialize(string meshFilename,
                    double diffusivity,
                    unsigned int polynomialOrder/*,
                    const std::map<unsigned int, double>& dirichletBC,
                    const std::map<unsigned int, double>& neumanBC*/)
    {
        std::map<unsigned int, double> dirichletBC;
        std::map<unsigned int, double> neumanBC;

        dirichletBC[0] = 0.0;
        neumanBC[1] = 0.5;
        //neumanBC[0] = -0.1E6;
        //neumanBC[1] =  1.0E6;
        //neumanBC[2] =  0.5E6;
        //dirichletBC[0] = 310;
        //dirichletBC[1] = 310;
        //dirichletBC[2] = 310;

        // 1. Create deal.II solver
        pDealII.reset(new dealiiDiffusion<dim>(diffusivity, polynomialOrder, dirichletBC, neumanBC));

        // 2. Load mesh from the supplied file
        GridIn<dim> gridin;
        gridin.attach_triangulation(pDealII->triangulation);
        std::ifstream f(meshFilename);
        gridin.read_msh(f);

        // 3. Setup deal.II system
        pDealII->setup_system();

        // 4. Assemble deal.II system
        pDealII->assemble_system();

        // 5. Initialize domain Omega
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

    void DeclareEquations(void)
    {
        daeEquation* eq;
        adouble a_diffusivity, a_accumulation, a_neumann, a_generation, a_dirichlet_f, a_dirichlet_K;

        size_t nrows = pDealII->K_diffusion.n();

        std::cout << "K_diffusion" << std::endl;
        pDealII->K_diffusion.print(std::cout);
        std::cout << "K_accumulation" << std::endl;
        pDealII->K_accumulation.print(std::cout);
        std::cout << "f_neuman" << std::endl;
        pDealII->f_neuman.print(std::cout);
        //std::cout << "f_dirichlet" << std::endl;
        //pDealII->f_dirichlet.print(std::cout);
        std::cout << "f_generation" << std::endl;
        pDealII->f_generation.print(std::cout);

        for(size_t row = 0; row < nrows; row++)
        {
            eq = this->CreateEquation("fe_" + toString(row));

            a_diffusivity = 0;
            a_neumann     = 0;
            a_dirichlet_f = 0;
            a_dirichlet_K = 0;
            a_generation  = 0;

            // Get data from the vectors
            const double neumann     = pDealII->f_neuman(row);
            const double dirichlet_f = pDealII->f_dirichlet(row);
            const double generation  = pDealII->f_generation(row);

            // Neuman BC (RHS)
            a_neumann     = adouble(neumann,     0.0, true);
            a_dirichlet_f = adouble(dirichlet_f, 0.0, true);
            a_generation  = adouble(generation,  0.0, true);

            for(SparsityPattern::iterator iter = pDealII->sparsity_pattern.begin(row); iter != pDealII->sparsity_pattern.end(row); iter++)
            {
                const size_t col = (*iter).column();

                // Get data from matrices
                const double diffusivity  = pDealII->K_diffusion   (row, col);
                const double dirichlet_K  = pDealII->K_dirichlet   (row, col);
                const double accumulation = pDealII->K_accumulation(row, col);

                // Diffusion term (LHS)
                if(diffusivity != 0)
                    a_diffusivity = a_diffusivity + diffusivity * this->T(col);

                if(accumulation != 0)
                    a_accumulation = a_accumulation +  accumulation * this->T.dt(col);

                if(dirichlet_K != 0)
                    a_dirichlet_K = a_dirichlet_K + dirichlet_K * this->T(col);
            }

            eq->SetResidual(a_accumulation + a_diffusivity + a_dirichlet_K - a_neumann - a_dirichlet_f - a_generation);
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
};

typedef daeDiffusion<1> daeDiffusion_1D;
typedef daeDiffusion<2> daeDiffusion_2D;
typedef daeDiffusion<3> daeDiffusion_3D;

}

#endif
