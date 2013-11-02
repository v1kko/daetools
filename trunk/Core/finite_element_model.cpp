#include "coreimpl.h"
#include "nodes.h"
#include "../variable_types.h"

namespace dae 
{
namespace core 
{

/*********************************************************
    daeFiniteElementModel
*********************************************************/
daeFiniteElementModel::daeFiniteElementModel(std::string strName, daeModel* pModel, std::string strDescription, daeFiniteElementObject* fe):
      daeModel(strName, pModel, strDescription),
      m_fe(fe),
      //m_dimension("dimension", this,               unit(), "Number of spatial dimensions"),
      m_omega    ("&Omega;",   this,                 unit(), "Omega domain"),
      m_T        ("T",         variable_types::no_t, this,   "Temperature", &m_omega)
{
    if(!m_fe)
        daeDeclareAndThrowException(exInvalidPointer);

    // Initialize daetools wrapper matrices and arrays that will be used by adFEMatrixItem/VectorItem nodes
    matK.reset  (m_fe->SystemMatrix());
    matKdt.reset(m_fe->SystemMatrix_dt());
    vecf.reset  (m_fe->SystemRHS());

    // Initialize domains and parameters
    size_t n = matK->GetNrows();
    std::vector<daePoint> coords(n);
    m_omega.CreateUnstructuredGrid(coords);
    std::cout << "n = " << n << std::endl;
    //m_dimension.CreateArray(dim);
}

inline adouble create_adouble(adNode* n)
{
    return adouble(0.0, 0.0, true, n);
}

void daeFiniteElementModel::DeclareEquations(void)
{
    daeEquation* eq;
    adouble a_K, a_Kdt, a_f;

    daeModel::DeclareEquations();

    m_fe->AssembleSystem();

    //std::cout << "system_matrix" << std::endl;
    //m_fe->system_matrix.print(std::cout);
    //std::cout << "system_matrix_dt" << std::endl;
    //pDealII->system_matrix_dt.print(std::cout);
    //std::cout << "system_rhs" << std::endl;
    //m_fe->system_rhs.print(std::cout);

    size_t nrows = matK->GetNrows();
    std::cout << "nrows = " << nrows << std::endl;
    for(size_t row = 0; row < nrows; row++)
    {
        eq = this->CreateEquation("eq_" + toString(row));

        // Reset equation's contributions
        a_K   = 0;
        a_Kdt = 0;

        // RHS
        a_f = create_adouble(new adFEVectorItemNode("f", *vecf, row, unit()));

        // begin_row() and end_row() return pointers created with the new operator
        boost::scoped_ptr<daeSparseMatrixRowIterator> iter(m_fe->RowIterator(row));
        for(iter->first(); !iter->isDone(); iter->next())
        {
            const size_t col = iter->currentItem();

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

void daeFiniteElementModel::UpdateEquations(const daeExecutionContext* pExecutionContext)
{
    std::cout << "daeFiniteElementModel::UpdateEquations 1/dt = " << pExecutionContext->m_dInverseTimeStep << std::endl;
    daeModel::UpdateEquations(pExecutionContext);

    /* Here we should to call pDealII->assemble_system() to update the system matrix and rhs.
     * In the Helmholtz case we should not do anything. */
    // m_fe->AssembleSystem();
}


}
}
