#ifndef DEAL_II_SYSTEM_H
#define DEAL_II_SYSTEM_H

#include "dealii_cell_context.h"
#include "dealii_datareporter.h"
#include "dealii_template_inst.h"
#include "../Core/nodes.h"

namespace dae
{
namespace fe_solver
{
using namespace dealii;

template<int dim>
adouble getValueFromNumber(const feRuntimeNumber<dim>& fe_number)
{
    if(fe_number.m_eType == eFEScalar)
        return adouble(fe_number.m_value);
    else if(fe_number.m_eType == eFEScalar_adouble)
        return fe_number.m_adouble_value;
    else
        throw std::runtime_error(std::string("Invalid expression return type"));

    return adouble();
}

/******************************************************************
    dealiiFiniteElementDOF<dim>
*******************************************************************/
template <int dim>
class dealiiFiniteElementDOF
{
public:
    dealiiFiniteElementDOF(const std::string&                      strName,
                           const std::string&                      strDescription,
                           boost::shared_ptr< FiniteElement<dim> > fe,
                           unsigned int                            nMultiplicity)
    {
        m_strName        = strName;
        m_strDescription = strDescription;
        m_fe             = fe;
        m_nMultiplicity  = nMultiplicity;

        // For vector-valued FE spaces use 1, and for the others the normal multiplicity
        m_nMultiplicityForFESystem = m_nMultiplicity;
        if(dynamic_cast< FE_RaviartThomas<dim>*   >(m_fe.get()) ||
           dynamic_cast< FE_DGRaviartThomas<dim>* >(m_fe.get()) ||
           dynamic_cast< FE_ABF<dim>*             >(m_fe.get()) ||
           dynamic_cast< FE_Nedelec<dim>*         >(m_fe.get()) ||
           dynamic_cast< FE_DGNedelec<dim>*       >(m_fe.get()) ||
           dynamic_cast< FE_BDM<dim>*             >(m_fe.get()) ||
           dynamic_cast< FE_DGBDM<dim>*           >(m_fe.get()))
            m_nMultiplicityForFESystem = 1;
    }

public:
    std::string  m_strName;
    std::string  m_strDescription;
    unsigned int m_nMultiplicity;
    unsigned int m_nMultiplicityForFESystem;

// Internal data
    boost::shared_ptr< FiniteElement<dim> > m_fe;
};

/******************************************************************
    dealiiFiniteElementWeakForm<dim>
*******************************************************************/
template <int dim>
class dealiiFiniteElementWeakForm
{
public:
    typedef typename std::map<unsigned int, const Function<dim,double>*>                    map_Uint_FunctionPtr;
    typedef typename std::pair<std::string, const Function<dim,double>*>                    pair_String_FunctionPtr;
    typedef typename std::pair<std::string, const Function<dim,adouble>*>                   pair_String_adoubleFunctionPtr;
    typedef typename std::map<unsigned int, std::vector<pair_String_FunctionPtr> >          map_Uint_vector_pair_String_FunctionPtr;
    typedef typename std::map<unsigned int, std::vector<pair_String_adoubleFunctionPtr> >   map_Uint_vector_pair_String_adoubleFunctionPtr;
    typedef typename std::map<unsigned int, feExpression<dim> >                             map_Uint_Expression;
    typedef typename std::map<std::string,  const Function<dim,double>*>                    map_String_FunctionPtr;
    typedef typename std::map<std::string,  const Function<dim,adouble>*>                   map_String_adoubleFunctionPtr;
    typedef typename std::pair< adouble,feExpression<dim> >                                 pair_Variable_Expression;
    typedef typename std::vector<pair_Variable_Expression>                                  vector_pair_Variable_Expression;
    typedef typename std::map<unsigned int, vector_pair_Variable_Expression>                map_Uint_vector_pair_Variable_Expression;

    dealiiFiniteElementWeakForm()
    {
    }

public:
    bool                                            m_bNeedsUpdate;     // Not used at the moment
    feExpression<dim>                               m_Aij;              // Stiffness matrix
    feExpression<dim>                               m_Mij;              // Mass matrix (dt)
    feExpression<dim>                               m_Fi;               // Load vector (rhs)
    feExpression<dim>                               m_innerCellFaceAij;
    feExpression<dim>                               m_innerCellFaceFi;
    map_Uint_Expression                             m_boundaryFaceAij;
    map_Uint_Expression                             m_boundaryFaceFi;
    map_String_FunctionPtr                          m_functions;
    map_String_adoubleFunctionPtr                   m_adouble_functions;
    //map_Uint_vector_pair_String_FunctionPtr       m_functionsDirichletBC;
    map_Uint_vector_pair_String_adoubleFunctionPtr  m_adoubleFunctionsDirichletBC;
    map_Uint_vector_pair_Variable_Expression        m_mapSurfaceIntegrals;
    vector_pair_Variable_Expression                 m_arrVolumeIntegrals;
};

/******************************************************************
    dealiiFiniteElementSystem<dim>
*******************************************************************/
template <int dim>
class dealiiFiniteElementSystem : public daeFiniteElementObject_t
{
typedef typename std::map<unsigned int, const Function<dim>*> map_Uint_FunctionPtr;
typedef typename std::pair<std::string, const Function<dim>*> pair_String_FunctionPtr;
typedef typename std::map<unsigned int, std::vector<pair_String_FunctionPtr> > map_Uint_vector_pair_String_FunctionPtr;
typedef typename std::map<unsigned int, feExpression<dim> >   map_Uint_Expression;
typedef typename std::map<std::string,  const Function<dim,double>*>  map_String_FunctionPtr;
typedef typename std::map<std::string,  const Function<dim,adouble>*> map_String_adoubleFunctionPtr;
typedef typename std::vector< dealiiFiniteElementDOF<dim>* > vector_DOFs;
typedef typename std::map< std::string, boost::variant<FEValuesExtractors::Scalar, FEValuesExtractors::Vector> > map_String_FEValuesExtractor;
typedef typename std::map<std::string, ComponentMask> map_string_ComponentMask;
typedef typename std::pair< adouble,feExpression<dim> >  pair_Variable_Expression;
typedef typename std::vector<pair_Variable_Expression>   vector_pair_Variable_Expression;
typedef typename std::map<unsigned int, vector_pair_Variable_Expression> map_Uint_vector_pair_Variable_Expression;
typedef typename std::pair<adouble,adouble> pair_Variable_adouble;
typedef typename std::vector<pair_Variable_adouble> vector_pair_Variable_adouble;
typedef typename std::map<unsigned int, vector_pair_Variable_adouble> map_Uint_vector_pair_Variable_adouble;

public:
    dealiiFiniteElementSystem();
    virtual ~dealiiFiniteElementSystem();

public:
    virtual void SetModel(daeFiniteElementModel* pFEModel);
    virtual void AssembleSystem();
    virtual bool NeedsReAssembling();
    virtual void ReAssembleSystem();

    virtual dae::daeMatrix<adouble>*                                                    Asystem() const; // Stiffness matrix
    virtual dae::daeMatrix<adouble>*                                                    Msystem() const; // Mass matrix (dt)
    virtual dae::daeArray<adouble>*                                                     Fload() const;   // Load vector
    virtual const std::map< unsigned int, std::vector< std::pair<adouble,adouble> > >*  SurfaceIntegrals() const;
    virtual const std::vector< std::pair<adouble,adouble> >*                            VolumeIntegrals() const;

    virtual void                        RowIndices(unsigned int row, std::vector<unsigned int>& narrIndices) const;
    virtual daeFiniteElementObjectInfo  GetObjectInfo() const;
    virtual std::vector<unsigned int>   GetDOFtoBoundaryMap();
    virtual dealIIDataReporter*         CreateDataReporter();

    void Initialize(const std::string&       meshFilename,
                    const Quadrature<dim>&   quadrature,
                    const Quadrature<dim-1>& faceQuadrature,
                    vector_DOFs&             DOFs);

    dealiiFiniteElementWeakForm<dim>* GetWeakForm() const;
    void SetWeakForm(dealiiFiniteElementWeakForm<dim>* wf);

protected:
    void setup_system();
    void assemble_system();
    void update_block(unsigned int block_index, double* values, unsigned int n);
    void write_solution(const std::string& strFilename);

    void assemble_cell(const unsigned int dofs_per_cell,
                       const unsigned int n_q_points,
                       feCellContextImpl< dim,FEValues<dim> >& cellContext,
                       boost::numeric::ublas::matrix<adouble>& cell_matrix_dt,
                       boost::numeric::ublas::matrix<adouble>& cell_matrix,
                       std::vector<adouble>& cell_rhs);
    void assemble_boundary_face(const unsigned int face,
                                const unsigned int boundary_id,
                                FEFaceValues<dim>& fe_face_values,
                                const unsigned int dofs_per_cell,
                                const unsigned int n_face_q_points,
                                feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                typename DoFHandler<dim>::active_cell_iterator& cell,
                                boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                std::vector<adouble>& cell_rhs);
    void assemble_inner_cell_face(const unsigned int face,
                                  FEFaceValues<dim>& fe_face_values,
                                  const unsigned int dofs_per_cell,
                                  const unsigned int n_face_q_points,
                                  feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                  typename DoFHandler<dim>::active_cell_iterator& cell,
                                  boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                  std::vector<adouble>& cell_rhs);
    void integrate_surface_integrals(const unsigned int face,
                                     const unsigned int boundary_id,
                                     FEFaceValues<dim>& fe_face_values,
                                     const unsigned int dofs_per_cell,
                                     const unsigned int n_face_q_points,
                                     feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                     typename DoFHandler<dim>::active_cell_iterator& cell);
    void integrate_volume_integrals(const unsigned int dofs_per_cell,
                                    const unsigned int n_q_points,
                                    feCellContextImpl< dim,FEValues<dim> >& cellContext);

public:
    // Additional deal.II specific data
    Triangulation<dim>              triangulation;
    DoFHandler<dim>                 dof_handler;
    SmartPointer< FESystem<dim> >   fe;
    ConstraintMatrix                hanging_node_constraints;

    BlockSparsityPattern        sparsity_pattern;
    BlockSparseMatrix<adouble>  system_matrix;
    BlockSparseMatrix<adouble>  system_matrix_dt;
    BlockVector<adouble>        system_rhs;
    BlockVector<adouble>        solution;
    BlockVector<double>         datareporter_solution;

    std::map< unsigned int, std::vector< std::pair<adouble,adouble> > > m_mapSurfaceIntegrals;
    std::vector< std::pair<adouble,adouble> >                           m_arrVolumeIntegrals;

    SmartPointer< Quadrature<dim>   >  m_quadrature_formula;
    SmartPointer< Quadrature<dim-1> >  m_face_quadrature_formula;

    // Model-specific data
    daeFiniteElementModel*                    m_pfeModel;
    dealiiFiniteElementWeakForm<dim>*         m_weakForm;
    std::vector<dealiiFiniteElementDOF<dim>*> m_DOFs;
    unsigned int                              m_no_components;
    std::vector<unsigned int>                 m_block_component;
    std::vector<types::global_dof_index>      m_dofs_per_block;
};

template <int dim>
dealiiFiniteElementSystem<dim>::dealiiFiniteElementSystem():
    dof_handler (triangulation)
{
    m_pfeModel = NULL;
    m_weakForm = NULL;
}

template <int dim>
void dealiiFiniteElementSystem<dim>::Initialize(const std::string&                meshFilename,
                                                const Quadrature<dim>&            quadrature,
                                                const Quadrature<dim-1>&          faceQuadrature,
                                                vector_DOFs&                      DOFs)
{
    m_DOFs = DOFs;

    // Create FESystem using one FE<dim> per dof (for multicomponent dofs too)
    std::vector<const FiniteElement<dim>*> arrFEs;
    std::vector<unsigned int> arrMultiplicities;
    for(unsigned int i = 0; i < m_DOFs.size(); i++)
    {
        arrFEs.push_back(m_DOFs[i]->m_fe.get());
        arrMultiplicities.push_back(m_DOFs[i]->m_nMultiplicityForFESystem);
    }

    fe = SmartPointer< FESystem<dim> >(new FESystem<dim>(arrFEs, arrMultiplicities));

    // Read the mesh
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);

    std::ifstream f(meshFilename.c_str());
    std::string extension = boost::filesystem::path(meshFilename).extension().string().c_str();
    size_t iFound = extension.find('.');
    if(iFound != std::string::npos)
        extension.erase(iFound, 1);

    typename GridIn<dim>::Format format = gridin.parse_format(extension);
    gridin.read(f, format);

    m_quadrature_formula      = SmartPointer< Quadrature<dim>   >(new Quadrature<dim>  (quadrature));
    m_face_quadrature_formula = SmartPointer< Quadrature<dim-1> >(new Quadrature<dim-1>(faceQuadrature));

    // Setup the system
    dealiiFiniteElementSystem<dim>::setup_system();
}

template <int dim>
void dealiiFiniteElementSystem<dim>::SetModel(daeFiniteElementModel* pFEModel)
{
    if(!pFEModel)
        throw std::runtime_error(std::string("The finite element model is null pointer"));

    m_pfeModel = pFEModel;
}

template <int dim>
dealiiFiniteElementWeakForm<dim>* dealiiFiniteElementSystem<dim>::GetWeakForm() const
{
    return m_weakForm;
}

template <int dim>
void dealiiFiniteElementSystem<dim>::SetWeakForm(dealiiFiniteElementWeakForm<dim>* weakForm)
{
    if(!weakForm)
        throw std::runtime_error(std::string("The weak form is null pointer"));

    m_weakForm = weakForm;

    // The weak form is typically set in daeModel::DeclareEquations(), therefore
    // immeadiately call DeclareEquationsForWeakForm() to declare equations for
    // the system [Mij]{dx/dt} + [Aij]{x} = {Fi} and boundary integrals.
    m_pfeModel->DeclareEquationsForWeakForm();
}

template <int dim>
dealiiFiniteElementSystem<dim>::~dealiiFiniteElementSystem ()
{
    dof_handler.clear ();
}

template <int dim>
void dealiiFiniteElementSystem<dim>::setup_system()
{
    dof_handler.distribute_dofs (*fe);

    size_t n_dofs = m_DOFs.size();

    m_no_components = 0;
    for(unsigned int i = 0; i < m_DOFs.size(); i++)
        m_no_components += m_DOFs[i]->m_nMultiplicity;

    //m_block_component.clear();
    //for(unsigned int i = 0; i < n_dofs; i++)
    //    m_block_component.insert(m_block_component.end(), m_DOFs[i]->m_nMultiplicity, i);

    DoFRenumbering::component_wise(dof_handler);//, m_block_component);

    m_dofs_per_block.resize(n_dofs);
    DoFTools::count_dofs_per_block (dof_handler, m_dofs_per_block);//, m_block_component);

    //for(unsigned int i = 0; i < m_no_components; i++)
    //    printf("m_block_component[%d] = %d\n", i, m_block_component[i]);
    for(unsigned int i = 0; i < n_dofs; i++)
        printf("m_dofs_per_block[%d] = %d\n", i, m_dofs_per_block[i]);

    const unsigned int n_couplings = dof_handler.max_couplings_between_dofs();
    //printf("n_couplings = %d\n", n_couplings);

    sparsity_pattern.reinit (n_dofs, n_dofs);
    for(unsigned int i = 0; i < n_dofs; i++)
        for(unsigned int j = 0; j < n_dofs; j++)
            sparsity_pattern.block(i, j).reinit (m_dofs_per_block[i], m_dofs_per_block[j], n_couplings);

    sparsity_pattern.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    sparsity_pattern.compress();

    system_matrix.reinit (sparsity_pattern);
    system_matrix_dt.reinit(sparsity_pattern);

    solution.reinit (n_dofs);
    for(unsigned int i = 0; i < n_dofs; i++)
        solution.block(i).reinit(m_dofs_per_block[i]);
    solution.collect_sizes();

    datareporter_solution.reinit (n_dofs);
    for(unsigned int i = 0; i < n_dofs; i++)
        datareporter_solution.block(i).reinit(m_dofs_per_block[i]);
    datareporter_solution.collect_sizes();

    system_rhs.reinit (n_dofs);
    for(unsigned int i = 0; i < n_dofs; i++)
        system_rhs.block(i).reinit(m_dofs_per_block[i]);
    system_rhs.collect_sizes();
}

template <int dim>
bool dealiiFiniteElementSystem<dim>::NeedsReAssembling()
{
    return false;
}

template <int dim>
void dealiiFiniteElementSystem<dim>::ReAssembleSystem()
{
    this->assemble_system();
}

template <int dim>
void dealiiFiniteElementSystem<dim>::AssembleSystem()
{
    this->assemble_system();
}

template <int dim>
void dealiiFiniteElementSystem<dim>::assemble_cell(const unsigned int dofs_per_cell,
                                                   const unsigned int n_q_points,
                                                   feCellContextImpl< dim,FEValues<dim> >& cellContext,
                                                   boost::numeric::ublas::matrix<adouble>& cell_matrix_dt,
                                                   boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                                   std::vector<adouble>& cell_rhs)
{
    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
        cellContext.m_q = q_point;

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            cellContext.m_i = i;

            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                cellContext.m_j = j;

                /* Stifness matrix (Aij) */
                if(m_weakForm->m_Aij.m_node)
                {
                    feRuntimeNumber<dim> result = m_weakForm->m_Aij.m_node->Evaluate(&cellContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid Aij expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    cell_matrix(i,j) += res;
                }

                /* Mass matrix (Mij) */
                if(m_weakForm->m_Mij.m_node)
                {
                    feRuntimeNumber<dim> result = m_weakForm->m_Mij.m_node->Evaluate(&cellContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid Mij expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    cell_matrix_dt(i,j) += res;
                }
            }

            /* Load vector (Fi) */
            if(m_weakForm->m_Fi.m_node)
            {
                cellContext.m_j = -1; // Set the unphysical value since it must not be used in Fi contributions

                feRuntimeNumber<dim> result = m_weakForm->m_Fi.m_node->Evaluate(&cellContext);
                if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                    throw std::runtime_error(std::string("Invalid Fi expression specified: (it must be a scalar value or adouble)"));

                adouble res = getValueFromNumber<dim>(result);
                if(res.node)
                    res.node = adNode::SimplifyNode(res.node);

                cell_rhs[i] += res;
            }
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::assemble_inner_cell_face(const unsigned int face,
                                                              FEFaceValues<dim>& fe_face_values,
                                                              const unsigned int dofs_per_cell,
                                                              const unsigned int n_face_q_points,
                                                              feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                                              typename DoFHandler<dim>::active_cell_iterator& cell,
                                                              boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                                              std::vector<adouble>& cell_rhs)
{
    fe_face_values.reinit (cell, face);

    for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
    {
        cellFaceContext.m_q = q_point;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            cellFaceContext.m_i = i;

            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                cellFaceContext.m_j = j;

                if(m_weakForm->m_innerCellFaceAij.m_node)
                {
                    const feExpression<dim>& faceAij = m_weakForm->m_innerCellFaceAij;
                    if(!faceAij.m_node)
                        throw std::runtime_error(std::string("Empty innerCellFaceAij expression specified"));

                    feRuntimeNumber<dim> result = faceAij.m_node->Evaluate(&cellFaceContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid innerCellFaceAij expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    cell_matrix(i,j) += res;
                }
            }

            if(m_weakForm->m_innerCellFaceFi.m_node)
            {
                const feExpression<dim>& faceFi = m_weakForm->m_innerCellFaceFi;
                if(!faceFi.m_node)
                    throw std::runtime_error(std::string("Empty innerCellFaceFi expression specified"));

                cellFaceContext.m_j = -1; // Set the unphysical value since it must not be used in faceFi contributions

                feRuntimeNumber<dim> result = faceFi.m_node->Evaluate(&cellFaceContext);
                if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                    throw std::runtime_error(std::string("Invalid innerCellFaceFi expression specified (it must be a scalar value or adouble)"));

                adouble res = getValueFromNumber<dim>(result);
                if(res.node)
                    res.node = adNode::SimplifyNode(res.node);

                cell_rhs[i] += res;
            }
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::assemble_boundary_face(const unsigned int face,
                                                            const unsigned int boundary_id,
                                                            FEFaceValues<dim>& fe_face_values,
                                                            const unsigned int dofs_per_cell,
                                                            const unsigned int n_face_q_points,
                                                            feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                                            typename DoFHandler<dim>::active_cell_iterator& cell,
                                                            boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                                            std::vector<adouble>& cell_rhs)
{
    typename map_Uint_Expression::const_iterator itAij = m_weakForm->m_boundaryFaceAij.find(boundary_id);
    typename map_Uint_Expression::const_iterator itFi  = m_weakForm->m_boundaryFaceFi.find(boundary_id);

    // If there is face Aij or Fi (or both)
    if(itAij != m_weakForm->m_boundaryFaceAij.end() || itFi != m_weakForm->m_boundaryFaceFi.end())
    {
        fe_face_values.reinit (cell, face);

        for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
        {
            cellFaceContext.m_q = q_point;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                cellFaceContext.m_i = i;

                for(unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cellFaceContext.m_j = j;

                    if(itAij != m_weakForm->m_boundaryFaceAij.end())
                    {
                        const feExpression<dim>& faceAij = itAij->second;
                        if(!faceAij.m_node)
                            throw std::runtime_error(std::string("Empty boundaryFaceAij expression specified"));

                        feRuntimeNumber<dim> result = faceAij.m_node->Evaluate(&cellFaceContext);
                        if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                            throw std::runtime_error(std::string("Invalid boundaryFaceAij expression specified (it must be a scalar value or adouble)"));

                        adouble res = getValueFromNumber<dim>(result);
                        if(res.node)
                            res.node = adNode::SimplifyNode(res.node);

                        cell_matrix(i,j) += res;
                    }
                }

                if(itFi != m_weakForm->m_boundaryFaceFi.end())
                {
                    const feExpression<dim>& faceFi = itFi->second;
                    if(!faceFi.m_node)
                        throw std::runtime_error(std::string("Empty boundaryFaceFi expression specified"));

                    cellFaceContext.m_j = -1; // Set the unphysical value since it must not be used in faceFi contributions

                    feRuntimeNumber<dim> result = faceFi.m_node->Evaluate(&cellFaceContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid boundaryFaceFi expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    cell_rhs[i] += res;
                }
            }
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::integrate_volume_integrals(const unsigned int dofs_per_cell,
                                                                const unsigned int n_q_points,
                                                                feCellContextImpl< dim,FEValues<dim> >& cellContext)
{
    // Nota bene:
    //   fe_values is already reinitialised

    for(int v = 0; v < m_weakForm->m_arrVolumeIntegrals.size(); v++)
    {
        const std::pair< adouble, feExpression<dim> >& pve = m_weakForm->m_arrVolumeIntegrals[v];

        const adouble&           ad_variable  = pve.first; // not used here
        const feExpression<dim>& viExpression = pve.second;

        if(!viExpression.m_node)
            continue;

        adouble adIntegral;
        for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            cellContext.m_q = q_point;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                cellContext.m_i = i;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cellContext.m_j = j;

                    feRuntimeNumber<dim> result = viExpression.m_node->Evaluate(&cellContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid boundaryIntegral expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    adIntegral += res;
                }
            }
        }

        // Finally, add the sum to the vector's item v
        std::pair<adouble,adouble>& pad = m_arrVolumeIntegrals[v];
        adouble& pad_integral = pad.second;
        pad_integral += adIntegral;
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::integrate_surface_integrals(const unsigned int face,
                                                                 const unsigned int boundary_id,
                                                                 FEFaceValues<dim>& fe_face_values,
                                                                 const unsigned int dofs_per_cell,
                                                                 const unsigned int n_face_q_points,
                                                                 feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                                                 typename DoFHandler<dim>::active_cell_iterator& cell)
{
    typename map_Uint_vector_pair_Variable_Expression::const_iterator itboundaryIntegral = m_weakForm->m_mapSurfaceIntegrals.find(boundary_id);
    if(itboundaryIntegral != m_weakForm->m_mapSurfaceIntegrals.end())
    {
        // Nota bene:
        //   This can be evaluated twice if there are FaceFi or FaceAij contributions
        fe_face_values.reinit (cell, face);

        const std::vector<pair_Variable_Expression>& arrExpressions = itboundaryIntegral->second;

        // Get the vector of pairs <variable,integral_adouble_expression> where the integral expressions wil be stored
        std::vector< std::pair<adouble,adouble> >& vpaa = m_mapSurfaceIntegrals[boundary_id];

        for(int v = 0; v < arrExpressions.size(); v++)
        {
            const std::pair< adouble, feExpression<dim> >& pve = arrExpressions[v];

            const adouble&           ad_variable  = pve.first; // not used here
            const feExpression<dim>& siExpression = pve.second;

            if(!siExpression.m_node)
                continue;

            adouble adIntegral;
            for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
            {
                cellFaceContext.m_q = q_point;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    cellFaceContext.m_i = i;
                    cellFaceContext.m_j = -1;

                    feRuntimeNumber<dim> result = siExpression.m_node->Evaluate(&cellFaceContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid boundaryIntegral expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    adIntegral += res;
                }
            }

            // Finally, add the sum to the vpaa vector's item v (which is the vector at boundary=ID)
            std::pair<adouble,adouble>& pad = vpaa[v];
            adouble& pad_integral = pad.second;
            pad_integral += adIntegral;
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::assemble_system()
{
    if(!m_weakForm)
        throw std::runtime_error(std::string("The weak form has not been set"));
    if(!m_pfeModel)
        throw std::runtime_error(std::string("The finite element model has not been set"));

    Quadrature<dim>&   quadrature_formula      = *m_quadrature_formula;
    Quadrature<dim-1>& face_quadrature_formula = *m_face_quadrature_formula;

    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    boost::numeric::ublas::matrix<adouble> cell_matrix   (dofs_per_cell, dofs_per_cell);
    boost::numeric::ublas::matrix<adouble> cell_matrix_dt(dofs_per_cell, dofs_per_cell);
    std::vector<adouble>                   cell_rhs      (dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    FEValues<dim>  fe_values (*fe, quadrature_formula,
                              update_values   | update_gradients |
                              update_quadrature_points | update_normal_vectors | update_JxW_values);

    FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
                                      update_values | update_gradients | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);
    
    std::vector<types::global_dof_index> mapGlobalDOFtoBoundary;
    DoFTools::map_dof_to_boundary_indices(dof_handler, mapGlobalDOFtoBoundary);

    // Used to identify FEValues/FEFaceValues that belong to a particular equation
    map_String_FEValuesExtractor mapExtractors;

    // Used to identify DOFs that belong to a particular equation
    map_string_ComponentMask mapComponentMasks;
    int currentIndex = 0;
    for(unsigned int k = 0; k < m_DOFs.size(); k++)
    {
        const dealiiFiniteElementDOF<dim>& dof = *m_DOFs[k];

        if(dof.m_nMultiplicity == 1)
        {
            std::cout << (boost::format("VariableName = %s, FEValuesExtractors::Scalar(index = %d)") % dof.m_strName % currentIndex).str() << std::endl;
            mapExtractors[dof.m_strName] = FEValuesExtractors::Scalar(currentIndex);
            mapComponentMasks[dof.m_strName] = fe->component_mask(FEValuesExtractors::Scalar(currentIndex));
        }
        else
        {
            std::cout << (boost::format("VariableName = %s, FEValuesExtractors::Vector(index = %d)") % dof.m_strName % currentIndex).str() << std::endl;
            mapExtractors[dof.m_strName] = FEValuesExtractors::Vector(currentIndex);
            mapComponentMasks[dof.m_strName] = fe->component_mask(FEValuesExtractors::Vector(currentIndex));
        }

        currentIndex += dof.m_nMultiplicity;
    }

    feCellContextImpl< dim, FEValues<dim> >      cellContext    (fe_values,      m_pfeModel, sparsity_pattern, local_dof_indices, m_weakForm->m_functions, m_weakForm->m_adouble_functions, mapExtractors);
    feCellContextImpl< dim, FEFaceValues<dim> >  cellFaceContext(fe_face_values, m_pfeModel, sparsity_pattern, local_dof_indices, m_weakForm->m_functions, m_weakForm->m_adouble_functions, mapExtractors);

    // Interpolate Dirichlet boundary conditions on the system matrix and rhs
    //std::map<types::global_dof_index,  double> boundary_values_map_double;
    std::map<types::global_dof_index, adouble> boundary_values_map_adouble;

    // Create a map with values for Dirichlet BCs using the adouble version of interpolate_boundary_values() function
    {
        typedef typename std::pair<std::string, const Function<dim,adouble>*>          pair_String_FunctionPtr;
        typedef typename std::map<unsigned int, std::vector<pair_String_FunctionPtr> > map_Uint_vector_pair_String_FunctionPtr;

        for(typename map_Uint_vector_pair_String_FunctionPtr::const_iterator it = m_weakForm->m_adoubleFunctionsDirichletBC.begin(); it != m_weakForm->m_adoubleFunctionsDirichletBC.end(); it++)
        {
            const unsigned int   id                         = it->first;
            const std::vector<pair_String_FunctionPtr>& bcs = it->second;

            for(int k = 0; k < bcs.size(); k++)
            {
                pair_String_FunctionPtr p = bcs[k];
                const std::string           variableName =  p.first;
                const Function<dim,adouble>& fun          = *p.second;

                typename map_string_ComponentMask::iterator iter = mapComponentMasks.find(variableName);
                if(iter == mapComponentMasks.end())
                    throw std::runtime_error("Cannot find variable: " + variableName + " in the DirichletBC dictionary");

                std::cout << "Interpolate DirichletBC at id: " << id << " for variable " << variableName << " with sample value for component = 0 and at point (0,0,0): " << fun.value(Point<dim>(0,0,0), 0) << std::endl;

                daeVectorTools::interpolate_boundary_values (dof_handler,
                                                             id,
                                                             fun,
                                                             boundary_values_map_adouble,
                                                             iter->second);

                //printf("bc[%d] = [", id);
                //for(std::map<types::global_dof_index, double>::const_iterator it = boundary_values.begin(); it != boundary_values.end(); it++)
                //    printf("(%d,%f) ", it->first, it->second);
                //printf("]\n");
            }
        }
    }
    
    //for(std::map<types::global_dof_index, adouble>::const_iterator it = boundary_values_map_adouble.begin(); it != boundary_values_map_adouble.end(); it++)
    //   printf("(%d,%f) ", it->first, it->second.getValue());
    //printf("\n");
    
    // Populate the map std:map< std::vector< std::pair<adouble,adouble> > > with variable adouble objects.
    // The integral expressions will be built and added later.
    for(typename map_Uint_vector_pair_Variable_Expression::const_iterator it = m_weakForm->m_mapSurfaceIntegrals.begin(); it != m_weakForm->m_mapSurfaceIntegrals.end(); it++)
    {
        const unsigned int                           id   = it->first;
        const std::vector<pair_Variable_Expression>& vpve = it->second;

        vector_pair_Variable_adouble vpaa;
        vpaa.reserve(vpve.size());
        for(size_t i = 0; i < vpve.size(); i++)
        {
            const pair_Variable_Expression& pve = vpve[i];
            const adouble& ad_variable = pve.first;
            vpaa.push_back( std::pair<adouble,adouble>(ad_variable, adouble()) );
        }
        this->m_mapSurfaceIntegrals[id] = vpaa;
    }

    // Populate the vector std:vector< std::pair<adouble,adouble> > with variable adouble objects.
    // The integral expressions will be built and added later.
    m_arrVolumeIntegrals.reserve(m_weakForm->m_arrVolumeIntegrals.size());
    for(size_t i = 0; i < m_weakForm->m_arrVolumeIntegrals.size(); i++)
    {
        const pair_Variable_Expression& pve = m_weakForm->m_arrVolumeIntegrals[i];
        const adouble& ad_variable = pve.first;
        m_arrVolumeIntegrals.push_back( std::pair<adouble,adouble>(ad_variable, adouble()) );
    }

    int n_active_cells = triangulation.n_active_cells();
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();    
    for(int cellCounter = 0; cell != endc; ++cell, cellCounter++)
    {
        cell_matrix.clear();
        cell_matrix_dt.clear();
        std::fill(cell_rhs.begin(), cell_rhs.end(), adouble(0.0));

        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        assemble_cell(dofs_per_cell,
                      n_q_points,
                      cellContext,
                      cell_matrix_dt,
                      cell_matrix,
                      cell_rhs);

        integrate_volume_integrals(dofs_per_cell,
                                   n_q_points,
                                   cellContext);

        /* Typically boundary conditions of the Neumann or Robin type. */
        for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
            // Only boundary faces
            if(cell->face(face)->at_boundary())
            {
                const unsigned int id = cell->face(face)->boundary_id();

                assemble_boundary_face(face,
                                       id,
                                       fe_face_values,
                                       dofs_per_cell,
                                       n_face_q_points,
                                       cellFaceContext,
                                       cell,
                                       cell_matrix,
                                       cell_rhs);

                integrate_surface_integrals(face,
                                            id,
                                            fe_face_values,
                                            dofs_per_cell,
                                            n_face_q_points,
                                            cellFaceContext,
                                            cell);

            }

            // All faces
            assemble_inner_cell_face(face,
                                     fe_face_values,
                                     dofs_per_cell,
                                     n_face_q_points,
                                     cellFaceContext,
                                     cell,
                                     cell_matrix,
                                     cell_rhs);
        }

   /*
        printf("cell_matrix before bcs:\n");
        for(int x = 0; x < cell_matrix.size1(); x++)
        {
            printf("[");
            for(int y = 0; y < cell_matrix.size2(); y++)
                printf("%+f ", cell_matrix(x,y).getValue());
            printf("]\n");
        }
        printf("\n");

        printf("cell_rhs before bcs:\n");
        printf("[");
        for(int x = 0; x < cell_rhs.size(); x++)
            printf("%+f ", cell_rhs[x].getValue());
        printf("]\n");
        printf("\n");
    */
        // double version, not used anymore
/*      {
            // Apply Dirichlet boundary conditions on the stiffness matrix and rhs
            daeMatrixTools::local_apply_boundary_values(boundary_values_map_double,
                                                        local_dof_indices,
                                                        cell_matrix,
                                                        cell_rhs);

            // Modify the local mass atrix for those nodes that have Dirichlet boundary conditions set
            daeMatrixTools::local_process_mass_matrix(boundary_values_map_double,
                                                      local_dof_indices,
                                                      cell_matrix_dt);
        }
*/
        
        // Apply a map with values of Dirichlet BCs using the adouble version of local_apply_boundary_values()
        // and local_process_mass_matrix() functions.
        {
            // Apply Dirichlet boundary conditions on the stiffness matrix and rhs
            daeMatrixTools::local_apply_boundary_values(boundary_values_map_adouble,
                                                        local_dof_indices,
                                                        cell_matrix,
                                                        cell_rhs);

            // Modify the local mass atrix for those nodes that have Dirichlet boundary conditions set
            daeMatrixTools::local_process_mass_matrix(boundary_values_map_adouble,
                                                      local_dof_indices,
                                                      cell_matrix_dt);
        }
    /*
        printf("cell_matrix after bcs:\n");
        for(int x = 0; x < cell_matrix.size1(); x++)
        {
            printf("[");
            for(int y = 0; y < cell_matrix.size2(); y++)
                printf("%+f ", cell_matrix(x,y).getValue());
            printf("]\n");
        }
        printf("\n");

        printf("cell_rhs after bcs:\n");
        printf("[");
        for(int x = 0; x < cell_rhs.size(); x++)
            printf("%+f ", cell_rhs[x].getValue());
        printf("]\n");
        printf("\n");
    */

        // Add local contributions Aij, Mij, Fi to the system matrices/vector
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));
                system_matrix_dt.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_dt(i,j));
            }
            if(cell_rhs[i].node || cell_rhs[i].getValue() != 0.0)
                system_rhs(local_dof_indices[i]) += cell_rhs[i];
        }
        
        printf("\rAssembling the system... %5.1f%%", float((100.0*(cellCounter+1))/n_active_cells));
        fflush(stdout);
        
    } // End cell iteration
    printf("\rAssembling the system... done.  \n");
    fflush(stdout);
    
    // Achtung, Achtung!!!
    //   Hanging nodes are NOT supported at th moment!! 
    //   There must be a way to stop simulation if hanging nodes are detected.

    // If using refined grids condense hanging nodes
    //hanging_node_constraints.condense(system_matrix);
    //hanging_node_constraints.condense(system_rhs);

    // What about this matrix? Should it also be condensed?
    //hanging_node_constraints.condense(system_matrix_dt);
}

template <int dim>
void dealiiFiniteElementSystem<dim>::update_block(unsigned int block_index, double* values, unsigned int n)
{
    Vector<double>& block_i = datareporter_solution.block(block_index);
    if(block_i.size() != n)
        throw std::runtime_error("The size of the variable value does not match the size of the solution");

    for(size_t k = 0; k < n; k++)
        block_i[k] = values[k];
}

// solution vector must be updated with values of all blocks/variables
template <int dim>
void dealiiFiniteElementSystem<dim>::write_solution(const std::string& strFilename)
{
    std::vector<std::string> solution_names;
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation;
    for(unsigned int i = 0; i < m_DOFs.size(); i++)
    {
        if(m_DOFs[i]->m_nMultiplicity == 1)
        {
            solution_names.push_back(m_DOFs[i]->m_strName);
            data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        }
        else
        {
            solution_names.insert(solution_names.end(), m_DOFs[i]->m_nMultiplicity, m_DOFs[i]->m_strName);
            data_component_interpretation.insert(data_component_interpretation.end(), m_DOFs[i]->m_nMultiplicity, DataComponentInterpretation::component_is_part_of_vector);
        }
    }

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector (datareporter_solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
    data_out.build_patches(fe->degree);
    std::ofstream output(strFilename.c_str());
    data_out.write_vtk(output);
}

template <int dim>
daeFiniteElementObjectInfo dealiiFiniteElementSystem<dim>::GetObjectInfo() const
{
    daeFiniteElementObjectInfo feObjectInfo;

    feObjectInfo.m_VariableInfos.resize(m_DOFs.size());

    feObjectInfo.m_nTotalNumberDOFs         = dof_handler.n_dofs();
    feObjectInfo.m_nNumberOfDOFsPerVariable = dof_handler.n_dofs() / m_no_components;

    for(unsigned int i = 0; i < m_DOFs.size(); i++)
    {
        if(m_dofs_per_block[i] / m_DOFs[i]->m_nMultiplicity != feObjectInfo.m_nNumberOfDOFsPerVariable)
            std::runtime_error("Number of DOFs per each component must be equal (for dof " + std::to_string(i) + ")");
    }

    for(unsigned int i = 0; i < m_DOFs.size(); i++)
    {
        feObjectInfo.m_VariableInfos[i].m_strName        = m_DOFs[i]->m_strName;
        feObjectInfo.m_VariableInfos[i].m_strDescription = m_DOFs[i]->m_strDescription;
        feObjectInfo.m_VariableInfos[i].m_nMultiplicity  = m_DOFs[i]->m_nMultiplicity;
        feObjectInfo.m_VariableInfos[i].m_nNumberOfDOFs  = m_dofs_per_block[i];
    }

    return feObjectInfo;
}

template <int dim>
void dealiiFiniteElementSystem<dim>::RowIndices(unsigned int row, std::vector<unsigned int>& narrIndices) const
{
    /* Block sparse matrix:

             |block|block|block|
             |col_0|col_1|col_2|
             +-----+-----+-----+ ..
      block  |x x x|     |     |
      row_0  |x x x|     |     |
             +-----+-----+-----+ ..
      block  |     |     |     |
      row_1  |     |     |     |
             +-----+-----+-----+ ..
      block  |     |     |     |
      row_2  |     |     |     |
             +-----+-----+-----+ ..
             :                 :
    */

    // First locate the block where row belongs: block_index and the local row index within that block: local_row
    // This can be done by using sparsity_pattern.row_indices().global_to_local(row)!!

    //    std::pair<block_row_index, local_row_index_in_block>
    const std::pair<size_t,          BlockIndices::size_type> row_index = sparsity_pattern.get_row_indices().global_to_local(row);
    unsigned int block_row_index = row_index.first;  // block index
    unsigned int local_row       = row_index.second; // local row index in the above block

    //std::cout << "block_row_index = " << block_row_index << std::endl;
    //std::cout << "local_row = " << local_row << std::endl;
    //std::cout << "row = " << row << std::endl;
    //std::cout << "sparsity_pattern.row_length(row) = " << sparsity_pattern.row_length(row) << std::endl;

    const BlockIndices& column_indices = sparsity_pattern.get_column_indices();
    narrIndices.reserve(sparsity_pattern.row_length(row));

    for(unsigned int block_column_index = 0; block_column_index < sparsity_pattern.n_block_cols(); block_column_index++)
    {
        const SparsityPattern& block = sparsity_pattern.block(block_row_index, block_column_index);
        SparsityPattern::iterator it     = block.begin(local_row);
        SparsityPattern::iterator it_end = block.end(local_row);
        for(; it != it_end; it++)
            narrIndices.push_back(column_indices.local_to_global(block_column_index, it->column()));
    }
}

template <int dim>
dae::daeMatrix<adouble>* dealiiFiniteElementSystem<dim>::Asystem() const
{
    dae::daeMatrix<adouble>* p = new daeFEBlockMatrix<adouble>(system_matrix);
    return p;
}

template <int dim>
dae::daeMatrix<adouble>* dealiiFiniteElementSystem<dim>::Msystem() const
{
    return new daeFEBlockMatrix<adouble>(system_matrix_dt);
}

template <int dim>
dae::daeArray<adouble>* dealiiFiniteElementSystem<dim>::Fload() const
{
    return new daeFEBlockArray<adouble>(system_rhs);
}

template <int dim>
const std::map< unsigned int, std::vector< std::pair<adouble,adouble> > >*  dealiiFiniteElementSystem<dim>::SurfaceIntegrals() const
{
    return &m_mapSurfaceIntegrals;
}

template <int dim>
const std::vector< std::pair<adouble,adouble> >*  dealiiFiniteElementSystem<dim>::VolumeIntegrals() const
{
    return &m_arrVolumeIntegrals;
}


template <int dim>
std::vector<unsigned int> dealiiFiniteElementSystem<dim>::GetDOFtoBoundaryMap()
{
    std::vector<types::global_dof_index> mapGlobalDOFtoBoundary;
    DoFTools::map_dof_to_boundary_indices(dof_handler, mapGlobalDOFtoBoundary);
    return mapGlobalDOFtoBoundary;
}

template <int dim>
dealIIDataReporter* dealiiFiniteElementSystem<dim>::CreateDataReporter()
{
    // map<name:block_index>
    std::map<std::string, size_t> mapVariables;
    for(unsigned int i = 0; i < m_DOFs.size(); i++)
        mapVariables[m_DOFs[i]->m_strName] = i; // here i represents the block_index

    fnUpdateBlock   update_block_callback(boost::bind(&dealiiFiniteElementSystem<dim>::update_block, this, _1, _2, _3));
    fnWriteSolution write_solution_callback(boost::bind(&dealiiFiniteElementSystem<dim>::write_solution, this, _1));
    return new dealIIDataReporter(update_block_callback, write_solution_callback, mapVariables);
}


}
}

#endif
