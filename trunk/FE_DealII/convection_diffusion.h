#ifndef DAE_CONVECTION_DIFFUSION_DEALII_H
#define DAE_CONVECTION_DIFFUSION_DEALII_H

#include "dealii_common.h"
#include "convection_diffusion_dealii.h"

namespace dae
{
namespace fe_solver
{
namespace convection_diffusion_dealii
{
using namespace dae::fe_solver;

/*********************************************************
 *     daeConvectionDiffusion
 *********************************************************/
template <int dim>
class daeConvectionDiffusion : public daeModel
{
typedef typename boost::shared_ptr< Function<dim> > FunctionPtr;
public:
    // D is diffusivity
    daeConvectionDiffusion(string strName, daeModel* pModel = NULL, string strDescription = "")
        : daeModel(strName, pModel, strDescription),
          m_fnDataOut(*this),

          omega("&Omega;", this, unit(), "Omega domain"),
          T("T", vt::temperature_t, this, "Temperature, K", &omega)
    {
    }

//    void Initialize(string meshFilename,
//                    unsigned int polynomialOrder,
//                    double diffusivity,
//                    const std::vector<double>& velocity,
//                    double generation,
//                    const std::map<unsigned int, double>& dirichletBC,
//                    const std::map<unsigned int, double>& neumannBC)
//    {

    void InitializeModel(const std::string& jsonInit)
    {
        /* Here we have to process input arguments, create Function objects,
         * create deal.II model and finally call daeModel_dealII_Base::InitializeModel
         * to finalize initialization. */
        string meshFilename;
        unsigned int polynomialOrder;
        double diffusivity;
        std::vector<double> velocity;
        double generation;
        std::map<unsigned int, double> dirichletBC;
        std::map<unsigned int, double> neumannBC;

        meshFilename    = "step-7.msh";
        polynomialOrder = 1;
        diffusivity     = 1.0;
        //velocity;
        generation      = 1.0;
        dirichletBC[0]  = 1.0;
        neumannBC[1]    = 0.5;

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
        pDealII.reset(new dealiiConvectionDiffusion<dim>(meshFilename, polynomialOrder, funDiffusivity, funVelocity, funGeneration, funsDirichletBC, funsNeumannBC));

        // Initialize daetools wrapper matrices and arrays that will be used by adFEMatrixItem/VectorItem nodes
        matK.reset  (new daeFEMatrix<double> (pDealII->system_matrix));
        matKdt.reset(new daeFEMatrix<double> (pDealII->system_matrix_dt));
        vecf.reset  (new daeFEArray<double>  (pDealII->system_rhs));

        // Setup deal.II system
        pDealII->setup_system();

        // Assemble deal.II system
        pDealII->assemble_system();

        // Prepare DataOut object
        output.reinit(pDealII->dof_handler.n_dofs());
        data_out.attach_dof_handler(pDealII->dof_handler);
        data_out.add_data_vector(output, "solution");
        data_out.build_patches(pDealII->fe->degree);

        // Initialize domain Omega
        std::vector<daePoint> coords;
        const std::vector< Point<dim> >& vertices = pDealII->triangulation.get_vertices();
        coords.resize(vertices.size());
        for(size_t i = 0; i < vertices.size(); i++)
        {
            if(dim == 1)
                coords[i] = daePoint(vertices[i](0), 0.0, 0.0);
            else if(dim == 2)
                coords[i] = daePoint(vertices[i](0), vertices[i](1), 0.0);
            else if(dim == 3)
                coords[i] = daePoint(vertices[i](0), vertices[i](1), vertices[i](2));
        }

        omega.CreateUnstructuredGrid(coords);
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
            a_f = create_adouble(new adFEVectorItemNode("f", *vecf, row, unit()));

            // K and Kdt matrices
            for(SparsityPattern::iterator iter = pDealII->sparsity_pattern.begin(row); iter != pDealII->sparsity_pattern.end(row); iter++)
            {
                const size_t col = (*iter).column();

                if(!a_K.node)
                    a_K =       create_adouble(new adFEMatrixItemNode("K", *matK, row, col, unit())) * T(col);
                else
                    a_K = a_K + create_adouble(new adFEMatrixItemNode("K", *matK, row, col, unit())) * T(col);

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
        std::cout << "daeConvectionDiffusion::UpdateEquations 1/dt = " << pExecutionContext->m_dInverseTimeStep << std::endl;
        daeModel::UpdateEquations(pExecutionContext);

        // Here we have to call pDealII->assemble_system() to update the system matrix and the system RHS vector
        // In the Helmholtz case we should not do anything
    }

    daeDealIIDataReporter* CreateDataReporter(const std::string& strDirectory)
    {
        return new daeDealIIDataReporter(m_fnDataOut, strDirectory);
    }

    void operator() (const daeDataReporterVariableValue* value, double time, const std::string& directory)
    {
        if(value->m_nNumberOfPoints != output.size())
            return;

        for(size_t i = 0; i < value->m_nNumberOfPoints; i++)
            output[i] = value->m_pValues[i];

        std::cout << "Report: " << value->m_strName << " at " << time  << std::endl;
        std::ofstream output("/home/ciroki/Data/daetools/trunk/daetools-package/daetools/examples/solution.vtk");
        data_out.write_vtk (output);
    }

public:
    daeDomain                                           omega;
    daeVariable                                         T;
    boost::shared_ptr< dealiiConvectionDiffusion<dim> > pDealII;
    boost::shared_ptr< daeFEMatrix<double> >            matK;
    boost::shared_ptr< daeFEMatrix<double> >            matKdt;
    boost::shared_ptr< daeFEArray<double>  >            vecf;
    std::map<unsigned int, FunctionPtr>                 funsDirichletBC;
    std::map<unsigned int, FunctionPtr>                 funsNeumannBC;
    FunctionPtr                                         funDiffusivity;
    FunctionPtr                                         funVelocity;
    FunctionPtr                                         funGeneration;

    DataOut<dim>                                        data_out;
    Vector<double>                                      output;
    fnDataOut                                           m_fnDataOut;
};

typedef daeConvectionDiffusion<1> daeConvectionDiffusion_1D;
typedef daeConvectionDiffusion<2> daeConvectionDiffusion_2D;
typedef daeConvectionDiffusion<3> daeConvectionDiffusion_3D;

}
}
}

#endif
