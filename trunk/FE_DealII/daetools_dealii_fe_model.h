#ifndef DAETOOLS_DEALII_MODEL_H
#define DAETOOLS_DEALII_MODEL_H

#include "dealii_common.h"
#include "dealii_fe_model.h"
#include <boost/filesystem.hpp>

namespace dae
{
namespace fe_solver
{
/*********************************************************
 *     daeModel_dealII
 *********************************************************/
class daeModel_dealII : public daeModel,
                        public daeDataOut
{
public:
    daeModel_dealII(std::string                    strName,
                    daeModel*                      pModel,
                    std::string                    strDescription,
                    daeFiniteElementsModel_dealII* fe,
                    std::string                    strOutputDirectory):
          daeModel(strName, pModel, strDescription),
          m_fe(fe),
          m_strOutputDirectory(strOutputDirectory),
          // daetools objects:
          //m_dimension("dimension", this,              unit(), "Number of spatial dimensions"),
          m_omega    ("&Omega;",   this,              unit(), "Omega domain"),
          m_T        ("T",         vt::temperature_t, this,   "Temperature", &m_omega)
    {
        if(!m_fe)
            daeDeclareAndThrowException(exInvalidPointer);

        // Check if output directory exists
        // If not create the whole hierarchy, up to the top directory
        boost::filesystem::path folder(m_strOutputDirectory);
        if(!boost::filesystem::exists(folder))
        {
            if(!boost::filesystem::create_directories(folder))
            {
                daeDeclareException(exInvalidCall);
                e << "Invalid output directory name specified: " << m_strOutputDirectory;
                throw e;
            }
        }

        // Initialize daetools wrapper matrices and arrays that will be used by adFEMatrixItem/VectorItem nodes
        matK.reset  (new daeFEMatrix<double> (m_fe->system_matrix));
        matKdt.reset(new daeFEMatrix<double> (m_fe->system_matrix_dt));
        vecf.reset  (new daeFEArray<double>  (m_fe->system_rhs));

        // Setup deal.II system
        //m_fe->setup_system();

        // Initialize domains and parameters
        size_t n = m_fe->system_matrix.n();
        std::vector<daePoint> coords(n);
        m_omega.CreateUnstructuredGrid(coords);
        //m_dimension.CreateArray(dim);
    }

    void DeclareEquations(void)
    {
        daeEquation* eq;
        adouble a_K, a_Kdt, a_f;

        daeModel::DeclareEquations();

        m_fe->assemble_system();

        //std::cout << "system_matrix" << std::endl;
        //m_fe->system_matrix.print(std::cout);
        //std::cout << "system_matrix_dt" << std::endl;
        //pDealII->system_matrix_dt.print(std::cout);
        //std::cout << "system_rhs" << std::endl;
        //m_fe->system_rhs.print(std::cout);

        size_t nrows = m_fe->system_matrix.n();
        for(size_t row = 0; row < nrows; row++)
        {
            eq = this->CreateEquation("eq_" + toString(row));

            // Reset equation's contributions
            a_K   = 0;
            a_Kdt = 0;

            // RHS
            a_f = create_adouble(new adFEVectorItemNode("f", *vecf, row, unit()));

            // K and Kdt matrices
            for(SparsityPattern::iterator iter = m_fe->sparsity_pattern.begin(row); iter != m_fe->sparsity_pattern.end(row); iter++)
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
        // m_fe->AssembleSystem();
    }

    const daeDataOut* GetDataOut() const
    {
        return this;
    }

    void SendVariable(const daeDataReporterVariableValue* value, double time) const
    {
        if(value->m_strName != m_T.GetCanonicalName())
            return;

        m_fe->finalize_solution_and_save(m_strOutputDirectory,
                                         "vtk",
                                         daeGetStrippedRelativeName(dynamic_cast<daeObject*>(GetModel()), &m_T),
                                         value->m_pValues,
                                         value->m_nNumberOfPoints,
                                         time);
    }

public:
    //daeDomain                                           m_dimension;
    daeDomain                                           m_omega;
    daeVariable                                         m_T;
    daeFiniteElementsModel_dealII*                      m_fe;
    boost::shared_ptr< daeFEMatrix<double> >            matK;
    boost::shared_ptr< daeFEMatrix<double> >            matKdt;
    boost::shared_ptr< daeFEArray<double>  >            vecf;
    std::string                                         m_strOutputDirectory;
};
}
}


#endif
