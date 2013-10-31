#ifndef DAE_CONVECTION_DIFFUSION_DEALII_H
#define DAE_CONVECTION_DIFFUSION_DEALII_H

#include "dealii_common.h"
#include "convection_diffusion_dealii.h"
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

namespace dae
{
namespace fe_solver
{
namespace convection_diffusion_dealii
{
using namespace dae::fe_solver;

/*********************************************************
 *     daeConvectionDiffusion_(1D, 2D, 3D)
 *********************************************************/
template <int dim>
class daeConvectionDiffusion : public daeModel,
                               public daeDataOut
{
public:
    typedef typename std::map<unsigned int, const dealiiFunction<dim>*> map_Uint_FunctionPtr;
    typedef typename dealiiCell<dim>::iterator iterator;

    daeConvectionDiffusion(std::string                  strName,
                           daeModel*                    pModel,
                           std::string                  strDescription,
                           std::string                  meshFilename,
                           string                       quadratureFormula,
                           unsigned int                 polynomialOrder,
                           string                       outputDirectory,
                           const dealiiFunction<dim>&   diffusivity,
                           const dealiiFunction<dim>&   velocity,
                           const dealiiFunction<dim>&   generation,
                           const map_Uint_FunctionPtr&  dirichletBC,
                           const map_Uint_FunctionPtr&  neumannBC)
        : daeModel(strName, pModel, strDescription),
          // Functions:
          m_Diffusivity(diffusivity),
          m_Velocity(velocity),
          m_Generation(generation),
          m_mapDirichletBC(dirichletBC),
          m_mapNeumannBC(neumannBC),
          // daetools objects:
          m_dimension("dimesnion", this,              unit(), "Number of spatial dimensions"),
          m_omega    ("&Omega;",   this,              unit(), "Omega domain"),
          m_T        ("T",         vt::temperature_t, this,   "Temperature", &m_omega)
    {
        m_outputCounter      = 0;
        m_strOutputDirectory = outputDirectory;
        boost::filesystem::path folder(m_strOutputDirectory);
        if(!boost::filesystem::exists(folder))
        {
            if(!boost::filesystem::create_directories(folder))
            {
                daeDeclareException(exIOError);
                e << "Invalid directory specified: " << m_strOutputDirectory;
                throw e;
            }
        }

        // Create deal.II solver
        pDealII.reset(new dealiiConvectionDiffusion<dim>(meshFilename,
                                                         quadratureFormula,
                                                         polynomialOrder,
                                                         m_Diffusivity,
                                                         m_Velocity,
                                                         m_Generation,
                                                         m_mapDirichletBC,
                                                         m_mapNeumannBC));

        pdealIICell.reset(new dealiiCell<dim>((FiniteElement<dim>*)pDealII->fe, pDealII->dof_handler));

        // Initialize daetools wrapper matrices and arrays that will be used by adFEMatrixItem/VectorItem nodes
        matK.reset  (new daeFEMatrix<double> (pDealII->system_matrix));
        matKdt.reset(new daeFEMatrix<double> (pDealII->system_matrix_dt));
        vecf.reset  (new daeFEArray<double>  (pDealII->system_rhs));

        // Setup deal.II system
        pDealII->setup_system();

        // Initialize domains and parameters
        size_t n = pDealII->system_matrix.n();
        std::vector<daePoint> coords(n);
        /*
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
        */

        m_omega.CreateUnstructuredGrid(coords);
        m_dimension.CreateArray(dim);
    }

    void DeclareEquations(void)
    {
        daeModel::DeclareEquations();
    }

    void AssembleSystem()
    {
        // Assemble deal.II system
        pDealII->assemble_system();
    }

    void GenerateEquations()
    {
        daeEquation* eq;
        adouble a_K, a_Kdt, a_f;

        /*
        std::cout << "system_matrix" << std::endl;
        pDealII->system_matrix.print(std::cout);
        std::cout << "system_matrix_dt" << std::endl;
        pDealII->system_matrix_dt.print(std::cout);
        std::cout << "system_rhs" << std::endl;
        pDealII->system_rhs.print(std::cout);
        */

        size_t nrows = pDealII->system_matrix.n();
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
                    a_K =       create_adouble(new adFEMatrixItemNode("K", *matK, row, col, unit())) * m_T(col);
                else
                    a_K = a_K + create_adouble(new adFEMatrixItemNode("K", *matK, row, col, unit())) * m_T(col);

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

        /* Here we should to call pDealII->assemble_system() to update the system matrix and rhs.
         * In the Helmholtz case we should not do anything. */
        // pDealII->assemble_system();
    }

    const daeDataOut* GetDataOut() const
    {
        return this;
    }

    void SendVariable(const daeDataReporterVariableValue* value, double time) const
    {
        if(value->m_strName != m_T.GetCanonicalName())
            return;

        // Prepare DataOut object
        std::string strDataSourceName = m_T.GetStrippedName();
        boost::filesystem::path vtkFilename((boost::format("%4d. %s(t=%f).vtk") % m_outputCounter
                                                                                % daeGetStrippedRelativeName(dynamic_cast<daeObject*>(GetModel()), &m_T)
                                                                                % time).str());
        boost::filesystem::path vtkPath(m_strOutputDirectory);
        vtkPath /= vtkFilename;

        for(size_t i = 0; i < value->m_nNumberOfPoints; i++)
            pDealII->solution[i] = value->m_pValues[i];

        // We may call distribute() on solution to fix hanging nodes
        std::cout << "solution after solve:" << pDealII->solution << std::endl;
        pDealII->hanging_node_constraints.distribute(pDealII->solution);
        std::cout << "solution after distribute:" << pDealII->solution << std::endl;

        //std::cout << "strVariableName = " << strVariableName << std::endl;
        //std::cout << "strFileName = "     << strFileName << std::endl;
        //std::cout << "solution:" << std::endl;
        //for(size_t i = 0; i < value->m_nNumberOfPoints; i++)
        //    std::cout << solution[i] << " ";
        //std::cout << std::endl;
        //std::cout << "Report: " << value->m_strName << " at " << time  << std::endl;

        DataOut<dim> data_out;
        data_out.attach_dof_handler(pDealII->dof_handler);
        data_out.add_data_vector(pDealII->solution, strDataSourceName);
        data_out.build_patches(pDealII->fe->degree);
        std::ofstream output(vtkPath.c_str());
        data_out.write_vtk (output);
    }

    static iterator begin(daeConvectionDiffusion<dim>& x) { return x.begin(); }
    static iterator end(daeConvectionDiffusion<dim>& x) { return x.end(); }

    iterator begin()
    {
        std::cout << "begin" << std::endl;
        return pdealIICell->begin();
    }

    iterator end()
    {
        std::cout << "end" << std::endl;
        return pdealIICell->end();
    }

public:
    const dealiiFunction<dim>&                          m_Diffusivity;
    const dealiiFunction<dim>&                          m_Velocity;
    const dealiiFunction<dim>&                          m_Generation;
    map_Uint_FunctionPtr                                m_mapDirichletBC;
    map_Uint_FunctionPtr                                m_mapNeumannBC;
    daeDomain                                           m_dimension;
    daeDomain                                           m_omega;
    daeVariable                                         m_T;
    boost::shared_ptr< dealiiConvectionDiffusion<dim> > pDealII;
    boost::shared_ptr< dealiiCell<dim> >                pdealIICell;

    boost::shared_ptr< daeFEMatrix<double> >            matK;
    boost::shared_ptr< daeFEMatrix<double> >            matKdt;
    boost::shared_ptr< daeFEArray<double>  >            vecf;
    std::string                                         m_strOutputDirectory;
    int                                                 m_outputCounter;
};

typedef daeConvectionDiffusion<1> daeConvectionDiffusion_1D;
typedef daeConvectionDiffusion<2> daeConvectionDiffusion_2D;
typedef daeConvectionDiffusion<3> daeConvectionDiffusion_3D;
}
}
}

#endif
