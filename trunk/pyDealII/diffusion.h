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
class daeDiffusion : public daeFiniteElementModel
{
public:
    // D is diffusivity
    daeDiffusion(string strName, daeModel* pModel = NULL, string strDescription = "")
        : daeFiniteElementModel(strName, pModel, strDescription),

          omega("&Omega;", this, unit(), "Omega domain"),
          T("T", vt::temperature_t, this, "Temperature, K", &omega)

    {
    }

    void Initialize(string meshFilename,
                    double diffusivity,
                    unsigned int polynomialOrder/*,
                    const std::map<int, double>& dirichletBC,
                    const std::map<int, double>& neumanBC*/)
    {
        std::map<int, double> dirichletBC;
        std::map<int, double> neumanBC;

        neumanBC[0] = -0.1E6;
        neumanBC[1] =  1.0E6;
        neumanBC[2] =  0.5E6;

        pDealII.reset(new dealiiDiffusion<dim>(diffusivity, polynomialOrder, dirichletBC, neumanBC));

        GridIn<dim> gridin;
        gridin.attach_triangulation(pDealII->triangulation);
        std::ifstream f(meshFilename);
        gridin.read_msh(f);

        pDealII->setup_system();

        std::vector<daePoint> coords;
        size_t n = pDealII->dof_handler.n_dofs();
        coords.resize(n);

        const unsigned int dofs_per_cell = pDealII->fe->dofs_per_cell;
        std::vector<unsigned int> local_dof_indices (dofs_per_cell);

        std::cout << "dofs_per_cell = " << dofs_per_cell << std::endl;

        typename DoFHandler<dim>::active_cell_iterator cell = pDealII->dof_handler.begin(), endc = pDealII->dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            cell->get_dof_indices (local_dof_indices);

            for(unsigned int v = 0; v < 4; ++v)
            {
                Point<dim> p = cell->vertex(v);
                unsigned int dof = local_dof_indices[v];
                string msg = "dof[%d] - vertex[%d]: (%f, %f, %f)";
                std::cout << (boost::format(msg) % local_dof_indices[v] % v % p(0) % p(1) % p(2)).str() << std::endl;

                coords[dof] = daePoint(p(0), p(1), p(2));
            }
        }

        std::cout << "coords = " << std::endl;
        for(unsigned int i = 0; i < coords.size(); i++)
            std::cout << boost::get<0>(coords[i]) << ", " << boost::get<1>(coords[i]) << ", " << boost::get<2>(coords[i]) << std::endl;
        std::cout << std::endl;

        omega.CreateUnstructuredGrid(coords);

        pDealII->assemble_system();
    }

    void DeclareEquations(void)
    {
        daeFiniteElementEquationExecutionInfo* pEEI;
        std::vector<daeEquationExecutionInfo*> ptrarrEquationExecutionInfos;
        std::vector<adouble> arrT;

        eq = CreateFiniteElementEquation("fe", &omega, "description");

        GetVariableRuntimeNodes(T, arrT);
        eq->GetEquationExecutionInfos(ptrarrEquationExecutionInfos);
        for(size_t i = 0; i < ptrarrEquationExecutionInfos.size(); i++)
        {
            pEEI = dynamic_cast<daeFiniteElementEquationExecutionInfo*>(ptrarrEquationExecutionInfos[i]);
            adouble a = arrT[i];
            pEEI->SetEvaluationNode(a);
        }

    }

    void AssembleEquation(daeFiniteElementEquation* pEquation)
    {
    }

public:
    daeDomain   omega;
    daeVariable T;
    daeFiniteElementEquation* eq;

    boost::shared_ptr< dealiiDiffusion<dim> > pDealII;
};

typedef daeDiffusion<1> daeDiffusion_1D;
typedef daeDiffusion<2> daeDiffusion_2D;
typedef daeDiffusion<3> daeDiffusion_3D;

}

#endif
