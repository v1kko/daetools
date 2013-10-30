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
 *     daeConvectionDiffusion
 *********************************************************/
template <int dim>
class daeConvectionDiffusion : public daeModel,
                               public daeDataOut
{
typedef typename boost::shared_ptr< Function<dim> > FunctionPtr;
public:
    // D is diffusivity
    daeConvectionDiffusion(string strName, daeModel* pModel = NULL, string strDescription = "")
        : daeModel(strName, pModel, strDescription),

          m_dim("dim", this, unit(), "Number of spatial dimensions"),
          m_omega("&Omega;", this, unit(), "Omega domain"),
          m_diffusivity("Diffusivity", unit(), this, ""),
          m_velocity("Velocity", unit(), this, "", &m_dim),
          m_generation("Generation", unit(), this, ""),
          m_T("T", vt::temperature_t, this, "Temperature, K", &m_omega)
    {
        m_outputCounter = 0;
    }

    void InitializeModel(const std::string& jsonInit)
    {
        /* Here we have to process input arguments, create Function objects,
         * create deal.II model and finally call daeModel_dealII_Base::InitializeModel
         * to finalize initialization. */
        boost::property_tree::ptree ptInit;
        std::stringstream ss(jsonInit);
        boost::property_tree::json_parser::read_json(ss, ptInit);

        std::string meshFilename;
        unsigned int polynomialOrder;
        std::map<unsigned int, double> dirichletBC;
        std::map<unsigned int, double> neumannBC;

        meshFilename         = ptInit.get<string>      ("meshFilename");
        m_strOutputDirectory = ptInit.get<string>      ("outputDirectory");
        polynomialOrder      = ptInit.get<unsigned int>("polynomialOrder");

        /*
        diffusivity          = ptInit.get<double>      ("diffusivity", 0.0);
        generation           = ptInit.get<double>      ("generation", 0.0);
        BOOST_FOREACH(boost::property_tree::ptree::value_type& v, ptInit.get_child("dirichletBC"))
        {
            double vel  = boost::lexical_cast<double>(v.first);

            velocity.push_back(vel);
            std::cout << "velocity = " << vel << std::endl;
        }
        */
        BOOST_FOREACH(boost::property_tree::ptree::value_type& bc, ptInit.get_child("dirichletBC"))
        {
            unsigned int id = boost::lexical_cast<unsigned int>(bc.first);
            double      val = boost::lexical_cast<double>(bc.second.data());

            dirichletBC[id] = val;
            std::cout << "dirichletBC[" << id << "] = " << val << std::endl;
        }
        BOOST_FOREACH(boost::property_tree::ptree::value_type& bc, ptInit.get_child("neumannBC"))
        {
            unsigned int id = boost::lexical_cast<unsigned int>(bc.first);
            double      val = boost::lexical_cast<double>(bc.second.data());

            neumannBC[id] = val;
            std::cout << "neumannBC[" << id << "] = " << val << std::endl;
        }

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

        // The values here are provisional and can (and should) be changed before assembling
        funDiffusivity.reset(new SingleValue_Function<dim>(0.0));
        funVelocity.reset(new SingleValue_Function<dim>(0.0));
        funGeneration.reset(new SingleValue_Function<dim>(0.0));

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

        // Create deal.II solver
        pDealII.reset(new dealiiConvectionDiffusion<dim>(meshFilename, polynomialOrder, funDiffusivity, funVelocity, funGeneration, funsDirichletBC, funsNeumannBC));

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
        m_dim.CreateArray(dim);
        //m_diffusivity.SetValue(0.0);
        //m_velocity.SetValues(0.0);
        //m_generation.SetValue(0.0);
    }

    void DeclareEquations(void)
    {
        daeEquation* eq;
        adouble a_K, a_Kdt, a_f;

        // Update values with the values from parameters
        dynamic_cast<SingleValue_Function<dim>*>(funDiffusivity.get())->m_value = m_diffusivity.GetValue();
        dynamic_cast<SingleValue_Function<dim>*>(funVelocity.get())->m_value    = 0.0;
        dynamic_cast<SingleValue_Function<dim>*>(funGeneration.get())->m_value  = m_generation.GetValue();

        // Assemble deal.II system
        pDealII->assemble_system();

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
        boost::filesystem::path vtkFilename((boost::format("%d)%s(t=%f).vtk") % m_outputCounter
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

public:
    daeDomain                                           m_dim;
    daeDomain                                           m_omega;
    daeParameter                                        m_diffusivity;
    daeParameter                                        m_velocity;
    daeParameter                                        m_generation;
    daeVariable                                         m_T;
    boost::shared_ptr< dealiiConvectionDiffusion<dim> > pDealII;
    boost::shared_ptr< daeFEMatrix<double> >            matK;
    boost::shared_ptr< daeFEMatrix<double> >            matKdt;
    boost::shared_ptr< daeFEArray<double>  >            vecf;
    std::map<unsigned int, FunctionPtr>                 funsDirichletBC;
    std::map<unsigned int, FunctionPtr>                 funsNeumannBC;
    FunctionPtr                                         funDiffusivity;
    FunctionPtr                                         funVelocity;
    FunctionPtr                                         funGeneration;

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
