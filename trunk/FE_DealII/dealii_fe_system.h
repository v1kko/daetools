#ifndef DEAL_II_SYSTEM_H
#define DEAL_II_SYSTEM_H

#include "dealii_cell_context.h"
#include "dealii_datareporter.h"
#include "dealii_template_inst.h"
#include "../Core/nodes.h"
#include "../variable_types.h"
#include <deal.II/base/work_stream.h>
#include "dealii_omp_work_stream.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <thread>
#include <chrono>
#include <atomic>

namespace daetools
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

inline bool hasNonzeroValue(const adouble& a)
{
    if(a.node)
    {
        const adConstantNode* cn = dynamic_cast<const adConstantNode*>(a.node.get());
        if(cn)
        {
            if(cn->m_quantity.getValue() == 0)
                return false;
             else
                return true;
        }
        else
            return true;
    }
    else if(a.getValue() != 0.0)
    {
        return true;
    }
    else
    {
        return false;
    }
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
                           std::shared_ptr< FiniteElement<dim> > fe,
                           unsigned int                            nMultiplicity,
                           daeVariableType                         variableType = variable_types::no_t)
    {
        m_strName        = strName;
        m_strDescription = strDescription;
        m_fe             = fe;
        m_nMultiplicity  = nMultiplicity;
        m_VariableType   = variableType;

        // For vector-valued FE spaces use 1, and for the others the normal multiplicity
        m_nMultiplicityForFESystem = m_nMultiplicity;
        if(dynamic_cast< FE_RaviartThomas<dim>*   >(m_fe.get()) ||
           dynamic_cast< FE_DGRaviartThomas<dim>* >(m_fe.get()) ||
           dynamic_cast< FE_ABF<dim>*             >(m_fe.get()) ||
           dynamic_cast< FE_Nedelec<dim>*         >(m_fe.get()) ||
           dynamic_cast< FE_DGNedelec<dim>*       >(m_fe.get()) ||
           dynamic_cast< FE_BDM<dim>*             >(m_fe.get()) ||
           dynamic_cast< FE_DGBDM<dim>*           >(m_fe.get()))
        {
            m_nMultiplicityForFESystem = 1;
        }
    }

public:
    std::string     m_strName;
    std::string     m_strDescription;
    unsigned int    m_nMultiplicity;
    unsigned int    m_nMultiplicityForFESystem;
    daeVariableType m_VariableType;

// Internal data
    std::shared_ptr< FiniteElement<dim> > m_fe;
};

enum dealiiContributionType
{
    eExpression_invalid,
    eExpression_single,
    eExpression_qi,
    eExpression_qij
};

template <int dim>
class dealiiCellContribution
{
public:
    dealiiCellContribution()
    {
        m_eContributionType = eExpression_invalid;
    }

    dealiiCellContribution(const feExpression<dim>& single)
    {
        m_eContributionType = eExpression_single;
        m_single = single;
    }

    dealiiCellContribution(const feExpression<dim>& q_loop, const feExpression<dim>& i_loop)
    {
        m_eContributionType = eExpression_qi;
        m_q_loop = q_loop;
        m_i_loop = i_loop;
    }

    dealiiCellContribution(const feExpression<dim>& q_loop, const feExpression<dim>& i_loop, const feExpression<dim>& j_loop)
    {
        m_eContributionType = eExpression_qij;
        m_q_loop = q_loop;
        m_i_loop = i_loop;
        m_j_loop = j_loop;
    }

    bool operator ==(const dealiiCellContribution& other)
    {
        return (this->m_eContributionType == other.m_eContributionType) &&
               (this->m_single == other.m_single) &&
               (this->m_q_loop == other.m_q_loop) &&
               (this->m_i_loop == other.m_i_loop) &&
               (this->m_j_loop == other.m_j_loop);
    }

public:
    dealiiContributionType m_eContributionType;
    feExpression<dim> m_q_loop;
    feExpression<dim> m_i_loop;
    feExpression<dim> m_j_loop;
    feExpression<dim> m_single;
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
    std::vector< dealiiCellContribution<dim> >      m_Aij;              // Stiffness matrix contributions
    std::vector< dealiiCellContribution<dim> >      m_Mij;              // Mass matrix (dt) contributions
    std::vector< dealiiCellContribution<dim> >      m_Fi;               // Load vector (rhs) contributions
    feExpression<dim>                               m_innerCellFaceAij;
    feExpression<dim>                               m_innerCellFaceFi;
    map_Uint_Expression                             m_boundaryFaceAij;
    map_Uint_Expression                             m_boundaryFaceFi;
    map_String_FunctionPtr                          m_functions;
    map_String_adoubleFunctionPtr                   m_adouble_functions;
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
    virtual void ClearAssembledSystem();
    virtual bool NeedsReAssembling();
    virtual void ReAssembleSystem();

    virtual daetools::daeMatrix<adouble>*                                                    Asystem() const; // Stiffness matrix
    virtual daetools::daeMatrix<adouble>*                                                    Msystem() const; // Mass matrix (dt)
    virtual daetools::daeArray<adouble>*                                                     Fload() const;   // Load vector
    virtual const std::map< unsigned int, std::vector< std::pair<adouble,adouble> > >*  SurfaceIntegrals() const;
    virtual const std::vector< std::pair<adouble,adouble> >*                            VolumeIntegrals() const;

    virtual void                        RowIndices(unsigned int row, std::vector<unsigned int>& narrIndices) const;
    virtual daeFiniteElementObjectInfo  GetObjectInfo() const;
    virtual dealIIDataReporter*         CreateDataReporter();

public:
    std::vector<bool>           GetBoundaryDOFs(const std::string& variableName, const std::set<types::boundary_id>& setBoundaryIDs);
    std::vector< Point<dim> >   GetDOFSupportPoints() const;
    std::vector<unsigned int>   GetDOFtoBoundaryMap();

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

    void cell_matrix_contribution(const unsigned int dofs_per_cell,
                               const unsigned int n_q_points,
                               feCellContextImpl< dim,FEValues<dim> >& cellContext,
                               boost::numeric::ublas::matrix<adouble>& cell_matrix,
                               feExpression<dim>& contribution);

    void cell_matrix_contribution(const unsigned int dofs_per_cell,
                                  const unsigned int n_q_points,
                                  feCellContextImpl< dim,FEValues<dim> >& cellContext,
                                  boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                  feExpression<dim>& q_term,
                                  feExpression<dim>& i_term,
                                  feExpression<dim>& j_term);

    void cell_rhs_contribution(const unsigned int dofs_per_cell,
                               const unsigned int n_q_points,
                               feCellContextImpl< dim,FEValues<dim> >& cellContext,
                               std::vector<adouble>& cell_rhs,
                               feExpression<dim>& contribution);

    void cell_rhs_contribution(const unsigned int dofs_per_cell,
                               const unsigned int n_q_points,
                               feCellContextImpl< dim,FEValues<dim> >& cellContext,
                               std::vector<adouble>& cell_rhs,
                               feExpression<dim>& q_term,
                               feExpression<dim>& i_term);

    void assemble_boundary_face(const unsigned int face,
                                const unsigned int boundary_id,
                                FEFaceValues<dim>& fe_face_values,
                                const unsigned int dofs_per_cell,
                                const unsigned int n_face_q_points,
                                feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                const typename DoFHandler<dim>::active_cell_iterator& cell,
                                boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                std::vector<adouble>& cell_rhs);
    void assemble_inner_cell_face(const unsigned int face,
                                  FEFaceValues<dim>& fe_face_values,
                                  const unsigned int dofs_per_cell,
                                  const unsigned int n_face_q_points,
                                  feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                  const typename DoFHandler<dim>::active_cell_iterator& cell,
                                  boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                  std::vector<adouble>& cell_rhs);
    void integrate_surface_integrals(const unsigned int face,
                                     const unsigned int boundary_id,
                                     FEFaceValues<dim>& fe_face_values,
                                     const unsigned int dofs_per_cell,
                                     const unsigned int n_face_q_points,
                                     feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                     const typename DoFHandler<dim>::active_cell_iterator& cell);
    void integrate_volume_integrals(const unsigned int dofs_per_cell,
                                    const unsigned int n_q_points,
                                    feCellContextImpl< dim,FEValues<dim> >& cellContext);

    /******************************************************************
        Assemble classes
    *******************************************************************/
    struct PerTaskData {
        PerTaskData(const FiniteElement<dim> &fe) :
            cell_matrix   (fe.dofs_per_cell, fe.dofs_per_cell),
            cell_matrix_dt(fe.dofs_per_cell, fe.dofs_per_cell),
            cell_rhs      (fe.dofs_per_cell),
            local_dof_indices(fe.dofs_per_cell),
            dofs_per_cell(fe.dofs_per_cell)
        {
            PerTaskData::count++;
            //printf("Constructor PerTaskData (count = %d)\n", (int)PerTaskData::count);
        }

        PerTaskData(const PerTaskData &data) :
            cell_matrix   (data.dofs_per_cell, data.dofs_per_cell),
            cell_matrix_dt(data.dofs_per_cell, data.dofs_per_cell),
            cell_rhs      (data.dofs_per_cell),
            local_dof_indices(data.dofs_per_cell),
            dofs_per_cell(data.dofs_per_cell)
        {
            PerTaskData::count++;
            //printf("Copy constructor PerTaskData (count = %d)\n", (int)PerTaskData::count);
        }

        ~PerTaskData()
        {
            PerTaskData::count--;
            //printf("Deleted PerTaskData (count = %d)\n", (int)PerTaskData::count);
        }

        static std::atomic_int count;

        boost::numeric::ublas::matrix<adouble> cell_matrix;
        boost::numeric::ublas::matrix<adouble> cell_matrix_dt;
        std::vector<adouble>                   cell_rhs;
        std::vector<unsigned int>              local_dof_indices;
        unsigned int                           dofs_per_cell;
        CellId id;
    };

    struct ScratchData {
        FEValues<dim>               fe_values;
        FEFaceValues<dim>           fe_face_values;
        const unsigned int          n_q_points;
        const unsigned int          n_face_q_points;
        const unsigned int          dofs_per_cell;
        //std::vector<unsigned int>   local_dof_indices;
        map_String_FEValuesExtractor& mapExtractors;
        map_string_ComponentMask&     mapComponentMasks;
        std::map<types::global_dof_index, adouble>& boundary_values_map_adouble;

      ScratchData (FiniteElement<dim>& fe,
                   Quadrature<dim>&    quadrature_formula,
                   Quadrature<dim-1>&  face_quadrature_formula,
                   map_String_FEValuesExtractor& mapExtractors_,
                   map_string_ComponentMask& mapComponentMasks_,
                   std::map<types::global_dof_index, adouble>& boundary_values_map_adouble_) :
            fe_values (fe,
                       quadrature_formula,
                       update_values   | update_gradients |
                       update_quadrature_points | update_normal_vectors | update_JxW_values),
            fe_face_values(fe,
                            face_quadrature_formula,
                            update_values | update_gradients | update_quadrature_points  |
                            update_normal_vectors | update_JxW_values),
            n_q_points(quadrature_formula.size()),
            n_face_q_points(face_quadrature_formula.size()),
            dofs_per_cell(fe.dofs_per_cell),
            //local_dof_indices(fe->dofs_per_cell),
            mapExtractors(mapExtractors_),
            mapComponentMasks(mapComponentMasks_),
            boundary_values_map_adouble(boundary_values_map_adouble_)
      {}

      ScratchData (const ScratchData &scratch) :
                fe_values (scratch.fe_values.get_fe(),
                           scratch.fe_values.get_quadrature(),
                           scratch.fe_values.get_update_flags()),
                fe_face_values (scratch.fe_face_values.get_fe(),
                                scratch.fe_face_values.get_quadrature(),
                                scratch.fe_face_values.get_update_flags()),
                n_q_points(scratch.n_q_points),
                n_face_q_points(scratch.n_face_q_points),
                dofs_per_cell(scratch.dofs_per_cell),
                //local_dof_indices(scratch.local_dof_indices),
                mapExtractors(scratch.mapExtractors),
                mapComponentMasks(scratch.mapComponentMasks),
                boundary_values_map_adouble(scratch.boundary_values_map_adouble)
        {}
    };

    void copy_local_to_global(const PerTaskData &data);
    void assemble_one_cell(const typename DoFHandler<dim>::active_cell_iterator& cell,
                                                           ScratchData& scratch,
                                                           PerTaskData& data);

public:
    // Additional deal.II specific data
    Triangulation<dim>              triangulation;
    DoFHandler<dim>                 dof_handler;
    SmartPointer< FESystem<dim> >   fe;
    ConstraintMatrix                hanging_node_constraints;

    BlockSparsityPattern            sparsity_pattern;

    std::shared_ptr< BlockSparseMatrix<adouble> > system_matrix;
    std::shared_ptr< BlockSparseMatrix<adouble> > system_matrix_dt;
    std::shared_ptr< BlockVector<adouble> >       system_rhs;

    //BlockVector<adouble>        solution;
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
    std::vector<unsigned int>                 m_sub_blocks;
    std::vector<types::global_dof_index>      m_dofs_per_block;

    // Used to identify FEValues/FEFaceValues that belong to a particular equation
    map_String_FEValuesExtractor m_mapExtractors;
    // Used to identify DOFs that belong to a particular equation
    map_string_ComponentMask m_mapComponentMasks;

    bool         m_bPrintInfo;
    std::string  m_paralell_assembly_scheme;
    int          m_num_threads;
    unsigned int m_queueSize;
    int          numberOfAssembled;
    int          n_active_cells;
};

template <int dim>
std::atomic_int dealiiFiniteElementSystem<dim>::PerTaskData::count(0);

template <int dim>
dealiiFiniteElementSystem<dim>::dealiiFiniteElementSystem():
    dof_handler (triangulation)
{
    m_pfeModel = NULL;
    m_weakForm = NULL;

    daeConfig& cfg = daeConfig::GetConfig();
    m_bPrintInfo = cfg.GetBoolean  ("daetools.deal_II.printInfo", false);

    m_paralell_assembly_scheme = cfg.GetString ("daetools.deal_II.assembly.parallelAssembly", "Sequential");
    m_num_threads              = cfg.GetInteger("daetools.deal_II.assembly.numThreads", 1);
    m_queueSize                = cfg.GetInteger("daetools.deal_II.assembly.queueSize", 32);
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

static void LogMemoryUsage()
{
    int currRealMem = 0;
    int peakRealMem = 0;
    int currVirtMem = 0;
    int peakVirtMem = 0;

    GetProcessMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);

    // Transform kB into MB
    currRealMem /= 1000;
    peakRealMem /= 1000;
    currVirtMem /= 1000;
    peakVirtMem /= 1000;
    printf("       RAM = %5d MB (peak RAM = %5d MB, virtRAM = %5d MB, peak virtRAM = %5d MB)\n", currRealMem, peakRealMem, currVirtMem, peakVirtMem);
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
}

template <int dim>
void dealiiFiniteElementSystem<dim>::setup_system()
{
    dof_handler.distribute_dofs (*fe);

    m_no_components = 0;
    for(unsigned int i = 0; i < m_DOFs.size(); i++)
        m_no_components += m_DOFs[i]->m_nMultiplicity;
    if(m_bPrintInfo)
        printf("m_no_components = %d\n", m_no_components);

    size_t n_blocks = m_DOFs.size();

    if(m_bPrintInfo)
    {
        printf("n_blocks fe     = %d\n", fe->n_blocks());
        printf("n_components fe = %d\n", fe->n_components());
        printf("is_primitive fe = %d\n", fe->is_primitive());
    }

    // Important!
    // If we use primitive FE we need to mark components that belong to the same vector dof,
    //   i.e. m_sub_blocks = [0, 0, 1, 2]
    //   (the number of components is equal to the number of blocks).
    // If some of the FE spaces in the FESystem are a non-primitive FE we *can not* use m_sub_blocks
    //   and it must be left empty
    //   (the number of components is not equal to the number of blocks).
    m_sub_blocks.clear();
    if(fe->is_primitive()) // || (fe->n_blocks() == fe->n_components()))
    {
        for(unsigned int dof = 0; dof < m_DOFs.size(); dof++)
            for(unsigned int i = 0; i < m_DOFs[dof]->m_nMultiplicityForFESystem; i++)
                m_sub_blocks.push_back(dof);

        if(m_bPrintInfo)
            for(unsigned int i = 0; i < m_sub_blocks.size(); i++)
                printf("m_sub_blocks[%d] = %d\n", i, m_sub_blocks[i]);
    }

    DoFRenumbering::component_wise(dof_handler, m_sub_blocks);

    m_dofs_per_block.resize(n_blocks);
    DoFTools::count_dofs_per_block (dof_handler, m_dofs_per_block, m_sub_blocks);
    if(m_bPrintInfo)
        for(unsigned int i = 0; i < m_dofs_per_block.size(); i++)
            printf("m_dofs_per_block[%d] = %d\n", i, m_dofs_per_block[i]);

    std::vector<unsigned int> dofs_per_component(fe->n_components());
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    if(m_bPrintInfo)
        for(unsigned int i = 0; i < dofs_per_component.size(); i++)
            printf("dofs_per_component[%d] = %d\n", i, dofs_per_component[i]);

    const unsigned int n_couplings = dof_handler.max_couplings_between_dofs();
    if(m_bPrintInfo)
        printf("n_couplings = %d\n", n_couplings);

    sparsity_pattern.reinit (n_blocks, n_blocks);
    for(unsigned int i = 0; i < n_blocks; i++)
        for(unsigned int j = 0; j < n_blocks; j++)
            sparsity_pattern.block(i, j).reinit (m_dofs_per_block[i], m_dofs_per_block[j], n_couplings);

    sparsity_pattern.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    sparsity_pattern.compress();

    system_matrix.reset(new BlockSparseMatrix<adouble>());
    system_matrix->reinit (sparsity_pattern);

    system_matrix_dt.reset(new BlockSparseMatrix<adouble>());
    system_matrix_dt->reinit(sparsity_pattern);

    system_rhs.reset(new BlockVector<adouble>());
    system_rhs->reinit (n_blocks);
    for(unsigned int i = 0; i < n_blocks; i++)
        system_rhs->block(i).reinit(m_dofs_per_block[i]);
    system_rhs->collect_sizes();

    datareporter_solution.reinit (n_blocks);
    for(unsigned int i = 0; i < n_blocks; i++)
        datareporter_solution.block(i).reinit(m_dofs_per_block[i]);
    datareporter_solution.collect_sizes();

    int currentIndex = 0;
    for(unsigned int k = 0; k < m_DOFs.size(); k++)
    {
        const dealiiFiniteElementDOF<dim>& dof = *m_DOFs[k];

        if(dof.m_nMultiplicity == 1)
        {
            if(m_bPrintInfo)
                std::cout << (boost::format("VariableName = %s, FEValuesExtractors::Scalar(index = %d)") % dof.m_strName % currentIndex).str() << std::endl;
            m_mapExtractors[dof.m_strName]     = FEValuesExtractors::Scalar(currentIndex);
            m_mapComponentMasks[dof.m_strName] = fe->component_mask(FEValuesExtractors::Scalar(currentIndex));
        }
        else
        {
            if(m_bPrintInfo)
                std::cout << (boost::format("VariableName = %s, FEValuesExtractors::Vector(index = %d)") % dof.m_strName % currentIndex).str() << std::endl;
            m_mapExtractors[dof.m_strName]     = FEValuesExtractors::Vector(currentIndex);
            m_mapComponentMasks[dof.m_strName] = fe->component_mask(FEValuesExtractors::Vector(currentIndex));
        }

        currentIndex += dof.m_nMultiplicity;
    }
}

template <int dim>
std::vector< Point<dim> > dealiiFiniteElementSystem<dim>::GetDOFSupportPoints() const
{
    std::vector< Point<dim> > support_points;
    unsigned int n_dofs = dof_handler.n_dofs();
    support_points.resize(n_dofs);

    // We need FEValues to get the mapping object
    FEValues<dim>  fe_values (*fe, *m_quadrature_formula,
                              update_values   | update_gradients |
                              update_quadrature_points | update_normal_vectors | update_JxW_values);
    const Mapping<dim>& mapping = fe_values.get_mapping();

    DoFTools::map_dofs_to_support_points (mapping, dof_handler, support_points);
    return support_points;
}

template <int dim>
bool dealiiFiniteElementSystem<dim>::NeedsReAssembling()
{
    return false;
}

template <int dim>
void dealiiFiniteElementSystem<dim>::ReAssembleSystem()
{
    if(!system_matrix || !system_matrix_dt || !system_rhs)
        throw std::runtime_error(std::string("The Finite Element systems has been cleared"));

    this->assemble_system();
}

template <int dim>
void dealiiFiniteElementSystem<dim>::AssembleSystem()
{
    if(!system_matrix || !system_matrix_dt || !system_rhs)
        throw std::runtime_error(std::string("The Finite Element systems has been cleared"));

    this->assemble_system();
}

//#define  SIMPLIFY(res_node) { \
//                                adNodePtr tmp = adNode::SimplifyNode(res_node); \
//                                res_node.reset(); \
//                                res_node = tmp; \
//                            }

template <int dim>
void dealiiFiniteElementSystem<dim>::cell_matrix_contribution(const unsigned int dofs_per_cell,
                                                              const unsigned int n_q_points,
                                                              feCellContextImpl< dim,FEValues<dim> >& cellContext,
                                                              boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                                              feExpression<dim>& contribution)
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

                if(contribution.m_node)
                {
                    feRuntimeNumber<dim> result = contribution.m_node->Evaluate_with_GIL(&cellContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid expression specified:") +
                                                 contribution.ToString() +
                                                 std::string(" (its result must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    if(hasNonzeroValue(res))
                        cell_matrix(i,j) += res;
                }
            }
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::cell_rhs_contribution(const unsigned int dofs_per_cell,
                                                           const unsigned int n_q_points,
                                                           feCellContextImpl< dim,FEValues<dim> >& cellContext,
                                                           std::vector<adouble>& cell_rhs,
                                                           feExpression<dim>& contribution)
{
    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
        cellContext.m_q = q_point;

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            cellContext.m_i = i;

            if(contribution.m_node)
            {
                cellContext.m_j = -1; // Set the unphysical value since it must not be used in Fi contributions

                feRuntimeNumber<dim> result = contribution.m_node->Evaluate_with_GIL(&cellContext);
                if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                    throw std::runtime_error(std::string("Invalid expression specified:") +
                                             contribution.ToString() +
                                             std::string(" (its result must be a scalar value or adouble)"));

                adouble res = getValueFromNumber<dim>(result);
                if(res.node)
                    res.node = adNode::SimplifyNode(res.node);

                if(hasNonzeroValue(res))
                    cell_rhs[i] += res;
            }
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::cell_matrix_contribution(const unsigned int dofs_per_cell,
                                                              const unsigned int n_q_points,
                                                              feCellContextImpl< dim,FEValues<dim> >& cellContext,
                                                              boost::numeric::ublas::matrix<adouble>& cell_matrix,
                                                              feExpression<dim>& q_term,
                                                              feExpression<dim>& i_term,
                                                              feExpression<dim>& j_term)
{
    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
        // Matrix will be initialised with feRuntimeNumber<dim> objects with eFEInvalid type
        boost::numeric::ublas::matrix< feRuntimeNumber<dim> > cell_matrix_temp(dofs_per_cell, dofs_per_cell);

        cellContext.m_q = q_point;

        feRuntimeNumber<dim> q_rt(0.0);
        if(q_term.m_node)
            q_rt = q_term.m_node->Evaluate_with_GIL(&cellContext);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            cellContext.m_i = i;

            feRuntimeNumber<dim> i_rt(0.0);
            if(i_term.m_node)
                i_rt = i_term.m_node->Evaluate_with_GIL(&cellContext);

            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                cellContext.m_j = j;

                feRuntimeNumber<dim> j_rt(0.0);
                if(j_term.m_node)
                    j_rt = j_term.m_node->Evaluate_with_GIL(&cellContext);

                // If it is a first addition do an assignment; otherwise add the new value to the existing item
                if(cell_matrix_temp(i,j).m_eType == eFEInvalid)
                    cell_matrix_temp(i,j) = i_rt * j_rt;
                else
                    cell_matrix_temp(i,j) = cell_matrix_temp(i,j) + i_rt * j_rt;
            }
        }

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix_temp(i,j) = q_rt * cell_matrix_temp(i,j);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                feRuntimeNumber<dim>& result = cell_matrix_temp(i,j);

                if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                    throw std::runtime_error(std::string("Invalid expression specified: ") +
                                             (q_term*i_term*j_term).ToString() +
                                             std::string(" (its result must be a scalar value or adouble)"));

                adouble res = getValueFromNumber<dim>(result);
                if(res.node)
                    res.node = adNode::SimplifyNode(res.node);

                if(hasNonzeroValue(res))
                    cell_matrix(i,j) += res;
            }
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::cell_rhs_contribution(const unsigned int dofs_per_cell,
                                                           const unsigned int n_q_points,
                                                           feCellContextImpl< dim,FEValues<dim> >& cellContext,
                                                           std::vector<adouble>& cell_rhs,
                                                           feExpression<dim>& q_term,
                                                           feExpression<dim>& i_term)
{
    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
        // Vector will be initialised with feRuntimeNumber<dim> objects with eFEInvalid type
        std::vector< feRuntimeNumber<dim> > cell_rhs_temp(dofs_per_cell);

        cellContext.m_q = q_point;

        feRuntimeNumber<dim> q_rt(0.0);
        if(q_term.m_node)
            q_rt = q_term.m_node->Evaluate_with_GIL(&cellContext);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            cellContext.m_i = i;

            feRuntimeNumber<dim> i_rt(0.0);
            if(i_term.m_node)
                i_rt = i_term.m_node->Evaluate_with_GIL(&cellContext);

            cell_rhs_temp[i] = i_rt;
        }

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
            cell_rhs_temp[i] = q_rt * cell_rhs_temp[i];

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            feRuntimeNumber<dim>& result = cell_rhs_temp[i];

            if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                throw std::runtime_error(std::string("Invalid expression specified: ") +
                                         (q_term*i_term).ToString() +
                                         std::string(" (its result must be a scalar value or adouble)"));

            adouble res = getValueFromNumber<dim>(result);
            if(res.node)
                res.node = adNode::SimplifyNode(res.node);

            if(hasNonzeroValue(res))
                cell_rhs[i] += res;
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::assemble_inner_cell_face(const unsigned int face,
                                                              FEFaceValues<dim>& fe_face_values,
                                                              const unsigned int dofs_per_cell,
                                                              const unsigned int n_face_q_points,
                                                              feCellContextImpl< dim,FEFaceValues<dim> >& cellFaceContext,
                                                              const typename DoFHandler<dim>::active_cell_iterator& cell,
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

                    feRuntimeNumber<dim> result = faceAij.m_node->Evaluate_with_GIL(&cellFaceContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid innerCellFaceAij expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    if(hasNonzeroValue(res))
                        cell_matrix(i,j) += res;
                }
            }

            if(m_weakForm->m_innerCellFaceFi.m_node)
            {
                const feExpression<dim>& faceFi = m_weakForm->m_innerCellFaceFi;
                if(!faceFi.m_node)
                    throw std::runtime_error(std::string("Empty innerCellFaceFi expression specified"));

                cellFaceContext.m_j = -1; // Set the unphysical value since it must not be used in faceFi contributions

                feRuntimeNumber<dim> result = faceFi.m_node->Evaluate_with_GIL(&cellFaceContext);
                if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                    throw std::runtime_error(std::string("Invalid innerCellFaceFi expression specified (it must be a scalar value or adouble)"));

                adouble res = getValueFromNumber<dim>(result);
                if(res.node)
                    res.node = adNode::SimplifyNode(res.node);

                if(hasNonzeroValue(res))
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
                                                            const typename DoFHandler<dim>::active_cell_iterator& cell,
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

                        feRuntimeNumber<dim> result = faceAij.m_node->Evaluate_with_GIL(&cellFaceContext);
                        if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                            throw std::runtime_error(std::string("Invalid boundaryFaceAij expression specified (it must be a scalar value or adouble)"));

                        adouble res = getValueFromNumber<dim>(result);
                        if(res.node)
                            res.node = adNode::SimplifyNode(res.node);

                        if(hasNonzeroValue(res))
                            cell_matrix(i,j) += res;
                    }
                }

                if(itFi != m_weakForm->m_boundaryFaceFi.end())
                {
                    const feExpression<dim>& faceFi = itFi->second;
                    if(!faceFi.m_node)
                        throw std::runtime_error(std::string("Empty boundaryFaceFi expression specified"));

                    cellFaceContext.m_j = -1; // Set the unphysical value since it must not be used in faceFi contributions

                    feRuntimeNumber<dim> result = faceFi.m_node->Evaluate_with_GIL(&cellFaceContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid boundaryFaceFi expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    if(hasNonzeroValue(res))
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

                    feRuntimeNumber<dim> result = viExpression.m_node->Evaluate_with_GIL(&cellContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid boundaryIntegral expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    if(hasNonzeroValue(res))
                        adIntegral += res;
                }
            }
        }

        // Finally, add the sum to the vector's item v
        std::pair<adouble,adouble>& pad = m_arrVolumeIntegrals[v];
        adouble& pad_integral = pad.second;
        if(adIntegral.node)
            adIntegral.node = adNode::SimplifyNode(adIntegral.node);
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
                                                                 const typename DoFHandler<dim>::active_cell_iterator& cell)
{
    typename map_Uint_vector_pair_Variable_Expression::const_iterator itboundaryIntegral = m_weakForm->m_mapSurfaceIntegrals.find(boundary_id);
    if(itboundaryIntegral != m_weakForm->m_mapSurfaceIntegrals.end())
    {
        // Nota bene:
        //   This can be evaluated twice if there are FaceFi or FaceAij contributions
        fe_face_values.reinit (cell, face);

        const std::vector<pair_Variable_Expression>& arrExpressions = itboundaryIntegral->second;

        // Get the vector of pairs <variable,integral_adouble_expression> where the integral expressions will be stored
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

                    feRuntimeNumber<dim> result = siExpression.m_node->Evaluate_with_GIL(&cellFaceContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid boundaryIntegral expression specified (it must be a scalar value or adouble)"));

                    adouble res = getValueFromNumber<dim>(result);
                    if(res.node)
                        res.node = adNode::SimplifyNode(res.node);

                    if(hasNonzeroValue(res))
                        adIntegral += res;
                }
            }

            // Finally, add the sum to the vpaa vector's item v (which is the vector at boundary=ID)
            std::pair<adouble,adouble>& pad = vpaa[v];
            adouble& pad_integral = pad.second;
            if(adIntegral.node)
                adIntegral.node = adNode::SimplifyNode(adIntegral.node);
            pad_integral += adIntegral;
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::assemble_one_cell(const typename DoFHandler<dim>::active_cell_iterator& cell,
                                                       ScratchData& scratch,
                                                       PerTaskData& data)
{
    CellId id = cell->id();
    data.id = id;

    FEValues<dim>&                              fe_values                   = scratch.fe_values;
    FEFaceValues<dim>&                          fe_face_values              = scratch.fe_face_values;
    const unsigned int                          n_q_points                  = scratch.n_q_points;
    const unsigned int                          n_face_q_points             = scratch.n_face_q_points;
    const unsigned int                          dofs_per_cell               = scratch.dofs_per_cell;
    map_String_FEValuesExtractor&               mapExtractors               = scratch.mapExtractors;
    map_string_ComponentMask&                   mapComponentMasks           = scratch.mapComponentMasks;
    std::map<types::global_dof_index, adouble>& boundary_values_map_adouble = scratch.boundary_values_map_adouble;

    boost::numeric::ublas::matrix<adouble>& cell_matrix       = data.cell_matrix;
    boost::numeric::ublas::matrix<adouble>& cell_matrix_dt    = data.cell_matrix_dt;
    std::vector<adouble>&                   cell_rhs          = data.cell_rhs;
    std::vector<unsigned int>&              local_dof_indices = data.local_dof_indices;

    feCellContextImpl< dim, FEValues<dim> >      cellContext    (fe_values,      m_pfeModel, sparsity_pattern, local_dof_indices, m_weakForm->m_functions, m_weakForm->m_adouble_functions, mapExtractors);
    feCellContextImpl< dim, FEFaceValues<dim> >  cellFaceContext(fe_face_values, m_pfeModel, sparsity_pattern, local_dof_indices, m_weakForm->m_functions, m_weakForm->m_adouble_functions, mapExtractors);

    cell_matrix.clear();
    cell_matrix_dt.clear();
    std::fill(cell_rhs.begin(), cell_rhs.end(), adouble(0.0));

    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    //printf("local_dof_indices = [");
    //for(unsigned int i = 0; i < local_dof_indices.size(); i++)
    //    printf("%d, ", local_dof_indices[i]);
    //printf("]\n");

    for(size_t k = 0; k < m_weakForm->m_Mij.size(); k++)
    {
        dealiiCellContribution<dim>& contribution = m_weakForm->m_Mij[k];
        //printf("Mij = %s\n", contribution.m_single.ToString().c_str());

        if(contribution.m_eContributionType == eExpression_single)
            cell_matrix_contribution(dofs_per_cell,
                                     n_q_points,
                                     cellContext,
                                     cell_matrix_dt,
                                     contribution.m_single);

        else if(contribution.m_eContributionType == eExpression_qij)
            cell_matrix_contribution(dofs_per_cell,
                                     n_q_points,
                                     cellContext,
                                     cell_matrix_dt,
                                     contribution.m_q_loop, contribution.m_i_loop, contribution.m_j_loop);
    }

    for(size_t k = 0; k < m_weakForm->m_Aij.size(); k++)
    {
        dealiiCellContribution<dim>& contribution = m_weakForm->m_Aij[k];
        //printf("Aij = %s\n", contribution.m_single.ToString().c_str());

        if(contribution.m_eContributionType == eExpression_single)
            cell_matrix_contribution(dofs_per_cell,
                                     n_q_points,
                                     cellContext,
                                     cell_matrix,
                                     contribution.m_single);

        else if(contribution.m_eContributionType == eExpression_qij)
            cell_matrix_contribution(dofs_per_cell,
                                     n_q_points,
                                     cellContext,
                                     cell_matrix,
                                     contribution.m_q_loop, contribution.m_i_loop, contribution.m_j_loop);
    }

    for(size_t k = 0; k < m_weakForm->m_Fi.size(); k++)
    {
        dealiiCellContribution<dim>& contribution = m_weakForm->m_Fi[k];
        //printf("Fi = %s\n", contribution.m_single.ToString().c_str());

        if(contribution.m_eContributionType == eExpression_single)
            cell_rhs_contribution(dofs_per_cell,
                                  n_q_points,
                                  cellContext,
                                  cell_rhs,
                                  contribution.m_single);

        else if(contribution.m_eContributionType == eExpression_qi)
            cell_rhs_contribution(dofs_per_cell,
                                  n_q_points,
                                  cellContext,
                                  cell_rhs,
                                  contribution.m_q_loop, contribution.m_i_loop);
    }

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

    // Apply a map with values of Dirichlet BCs using the adouble version of local_apply_boundary_values()
    // and local_process_mass_matrix() functions.
    {
        // Apply Dirichlet boundary conditions on the stiffness matrix and rhs
        daeMatrixTools::local_apply_boundary_values(boundary_values_map_adouble,
                                                    local_dof_indices,
                                                    cell_matrix,
                                                    cell_rhs);

        // Modify the local mass matrix for those nodes that have Dirichlet boundary conditions set
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

    // Do the final simplification before adding the local contributions to the system matrices
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
    {
        if(cell_rhs[i].node)
            cell_rhs[i].node = adNode::SimplifyNode(cell_rhs[i].node);

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
            if(cell_matrix(i,j).node)
                cell_matrix(i,j).node = adNode::SimplifyNode(cell_matrix(i,j).node);
            if(cell_matrix_dt(i,j).node)
                cell_matrix_dt(i,j).node = adNode::SimplifyNode(cell_matrix_dt(i,j).node);
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::copy_local_to_global (const PerTaskData &data)
{

    // Add local contributions Aij, Mij, Fi to the system matrices/vector
    for(unsigned int i = 0; i < data.dofs_per_cell; ++i)
    {
        for(unsigned int j = 0; j < data.dofs_per_cell; ++j)
        {
            if(hasNonzeroValue( data.cell_matrix(i,j) ))
                system_matrix->add(data.local_dof_indices[i], data.local_dof_indices[j], data.cell_matrix(i,j));

            if(hasNonzeroValue( data.cell_matrix_dt(i,j) ))
                system_matrix_dt->add(data.local_dof_indices[i], data.local_dof_indices[j], data.cell_matrix_dt(i,j));
        }
        if(hasNonzeroValue( data.cell_rhs[i] ))
            (*system_rhs)(data.local_dof_indices[i]) += data.cell_rhs[i];
    }

    float percent = 100.0*(numberOfAssembled+1)/n_active_cells;
    printf("\rAssembling the cell %d / %d (%5.1f%%)...", numberOfAssembled+1, n_active_cells, percent);
    fflush(stdout);
    numberOfAssembled++;
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
    const unsigned int dofs_per_cell   = fe->dofs_per_cell;

    std::map<types::global_dof_index, adouble> boundary_values_map_adouble;

    // Interpolate Dirichlet boundary conditions on the system matrix and rhs
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

                typename map_string_ComponentMask::iterator iter = m_mapComponentMasks.find(variableName);
                if(iter == m_mapComponentMasks.end())
                    throw std::runtime_error("Cannot find variable: " + variableName + " in the DirichletBC dictionary");

                if(m_bPrintInfo)
                    std::cout << "Interpolate DirichletBC at id: " << id << " for variable " << variableName << std::endl;

                daeVectorTools::interpolate_boundary_values (dof_handler,
                                                             id,
                                                             fun,
                                                             boundary_values_map_adouble,
                                                             iter->second);
                /*
                if(m_bPrintInfo)
                {
                    printf("bc[%d] = [", id);
                    for(std::map<types::global_dof_index, adouble>::const_iterator it = boundary_values_map_adouble.begin(); it != boundary_values_map_adouble.end(); it++)
                        printf("(%d,%f) ", it->first, it->second);
                    printf("]\n");
                }
                */
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
        m_mapSurfaceIntegrals[id] = vpaa;
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

    n_active_cells = triangulation.n_active_cells();
    numberOfAssembled = 0;

    typedef typename DoFHandler<dim>::active_cell_iterator cell_iterator;

    PerTaskData copy_data_s(*fe);
    ScratchData scratch_s(*fe,
                          quadrature_formula,
                          face_quadrature_formula,
                          m_mapExtractors,
                          m_mapComponentMasks,
                          boundary_values_map_adouble);

    // Very important!!
    //   Acquire thread state and release the GIL so that the spawned worker-threads can acquire it when required.
    INIT_THREAD_STATE_AND_RELEASE_GIL;

    if(m_paralell_assembly_scheme == "TBB")
    {
        if(m_num_threads > 0)
            MultithreadInfo::set_thread_limit(m_num_threads);

        if(m_bPrintInfo)
        {
            printf("Number of threads = %d\n", MultithreadInfo::n_threads());
            //printf("Queue length    = %d\n", queue_length);
            //printf("Chunk size      = %d\n", chunk_size);
        }

        WorkStream::run(dof_handler.begin_active(),
                        dof_handler.end(),
                        *this,
                        &dealiiFiniteElementSystem<dim>::assemble_one_cell,
                        &dealiiFiniteElementSystem<dim>::copy_local_to_global,
                        scratch_s,
                        copy_data_s);
    }
    else if(m_paralell_assembly_scheme == "OpenMP")
    {
#ifdef _OPENMP
        if(m_num_threads > 0)
            omp_set_num_threads(m_num_threads);
#endif
        OpenMP_WorkStream::run(dof_handler.begin_active(),
                               dof_handler.end(),
                               *this,
                               &dealiiFiniteElementSystem<dim>::assemble_one_cell,
                               &dealiiFiniteElementSystem<dim>::copy_local_to_global,
                               scratch_s,
                               copy_data_s,
                               m_queueSize,
                               m_bPrintInfo);
    /* The code below is refactored into the OpenMP_WorkStream::run function.

        omp_lock_t lock;
        omp_init_lock(&lock);

        if(m_num_threads > 0)
            omp_set_num_threads(m_num_threads);

        cell_iterator cell_i = dof_handler.begin_active();
        cell_iterator endc   = dof_handler.end();

        // OpenMP does not work with iterators so populate std::vector with all cells.
        // The std::vector supports the random access and can be used with OpenMP.
        std::vector<cell_iterator> all_iterators;
        all_iterators.reserve(n_active_cells);
        for(; cell_i != endc; ++cell_i)
            all_iterators.push_back(cell_i);

        std::queue< std::shared_ptr<PerTaskData> > copy_data_queue;
        std::queue< std::shared_ptr<PerTaskData> > copy_data_queue_swap;

        int n_cells = all_iterators.size();
        #pragma omp parallel
        {
            if(m_bPrintInfo && omp_get_thread_num() == 0)
            {
                printf("Number of threads    = %d\n", omp_get_num_threads());
                printf("Queue size           = %d\n", m_queueSize);
            }

            #pragma omp for schedule(static, 1)
            for(int cellCounter = 0; cellCounter < n_cells; cellCounter++)
            {
                int tid = omp_get_thread_num();

                // Get the cell
                cell_iterator cell = all_iterators[cellCounter];
                //printf("Thread %d assembling cell %s\n", tid, cell->id().to_string().c_str());

                // Create the scratch and the copy_data objects
                std::shared_ptr<PerTaskData> copy_data(new PerTaskData(copy_data_s));
                ScratchData scratch(scratch_s);

                // Assemble cell
                assemble_one_cell(cell, scratch, *copy_data);

                // Add the data to the queue
                omp_set_lock(&lock);
                    copy_data_queue.push(copy_data);
                omp_unset_lock(&lock);

                // When the queue size reaches the specified m_queueSize
                // the master thread takes all copy_data objects from the queue
                // and copies the data to the global matrices/array.
                if(tid == 0)
                {
                    //printf("copy_data_queue.size = %d\n", copy_data_queue.size());
                    if(copy_data_queue.size() >= m_queueSize)
                    {
                        // Take all objects from the queue and copy them to the global structures.
                        // This way, the other threads do not wait to acquire the omp lock.
                        // The std::queue::swap() function should be fast since it only swaps a couple of pointers.
                        // Anyhow, even copying the shared_ptr objects is cheap.
                        omp_set_lock(&lock);
                            copy_data_queue_swap.swap(copy_data_queue);
                        omp_unset_lock(&lock);

                        //printf("copy_data_queue_swap.size = %d\n", copy_data_queue_swap.size());
                        while(!copy_data_queue_swap.empty())
                        {
                            std::shared_ptr<PerTaskData> cd = copy_data_queue_swap.front();
                            copy_data_queue_swap.pop();

                            copy_local_to_global(*cd);
                        }
                    }
                }
            }
        }

        // If something is left in the queue process it
        while(!copy_data_queue.empty())
        {
            std::shared_ptr<PerTaskData> cd = copy_data_queue.front();
            copy_data_queue.pop();
            copy_local_to_global(*cd);
            //printf("copy_data_queue.size = %d\n", copy_data_queue.size());
        }
    */
    }
    else // Sequential
    {
        if(m_bPrintInfo)
        {
            printf("Number of threads = 1 (sequential)\n");
        }

        cell_iterator cell = dof_handler.begin_active();
        cell_iterator endc = dof_handler.end();
        for(; cell != endc; ++cell)
        {
            PerTaskData copy_data(copy_data_s);
            ScratchData scratch(scratch_s);

            // Assemble cell
            assemble_one_cell(cell, scratch, copy_data);

            // Copies the data to the global matrices/array.
            copy_local_to_global(copy_data);
        }
    }

    printf("\rAssembling the system... done.                      \n");
    fflush(stdout);
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
    feObjectInfo.m_nNumberOfDOFsPerVariable = 0;
    //feObjectInfo.m_nNumberOfDOFsPerVariable = dof_handler.n_dofs() / m_no_components;
    //
    //for(unsigned int i = 0; i < m_DOFs.size(); i++)
    //{
    //    if(m_dofs_per_block[i] / m_DOFs[i]->m_nMultiplicity != feObjectInfo.m_nNumberOfDOFsPerVariable)
    //        std::runtime_error("Number of DOFs per each component must be equal (for dof " + m_DOFs[i]->m_strName + ")");
    //}

    for(unsigned int i = 0; i < m_DOFs.size(); i++)
    {
        feObjectInfo.m_VariableInfos[i].m_strName        = m_DOFs[i]->m_strName;
        feObjectInfo.m_VariableInfos[i].m_strDescription = m_DOFs[i]->m_strDescription;
        feObjectInfo.m_VariableInfos[i].m_nMultiplicity  = m_DOFs[i]->m_nMultiplicity;
        feObjectInfo.m_VariableInfos[i].m_nNumberOfDOFs  = m_dofs_per_block[i];
        feObjectInfo.m_VariableInfos[i].m_VariableType   = m_DOFs[i]->m_VariableType;
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
/*
static void printNodeInfo()
{
    size_t constant_ti  = typeid(adConstantNode).hash_code();
    size_t fcvs_ti      = typeid(adFloatCoefficientVariableSumNode).hash_code();
    size_t variable_ti  = typeid(adSetupVariableNode).hash_code();
    size_t timederiv_ti = typeid(adSetupTimeDerivativeNode).hash_code();
    size_t unary_ti     = typeid(adUnaryNode).hash_code();
    size_t binary_ti    = typeid(adBinaryNode).hash_code();

    std::map<size_t,adNode*>::iterator it;
    std::map<size_t,adNode*>& constant_map  = adNodeImpl::g_allNodes[constant_ti];
    std::map<size_t,adNode*>& fcvs_map      = adNodeImpl::g_allNodes[fcvs_ti];
    std::map<size_t,adNode*>& variable_map  = adNodeImpl::g_allNodes[variable_ti];
    std::map<size_t,adNode*>& timederiv_map = adNodeImpl::g_allNodes[timederiv_ti];
    std::map<size_t,adNode*>& unary_map     = adNodeImpl::g_allNodes[unary_ti];
    std::map<size_t,adNode*>& binary_map    = adNodeImpl::g_allNodes[binary_ti];

    int mem = 0;
    for(it = constant_map.begin(); it != constant_map.end(); it++)
        mem += ( sizeof(adConstantNode) );
    printf("adConstantNode count = %d, memory = %d MB\n", (int)constant_map.size(), mem / (1024*1024));

    printf("adFloatCoefficientVariableSumNode count = %d\n", (int)fcvs_map.size());
    mem = 0;
    for(it = fcvs_map.begin(); it != fcvs_map.end(); it++)
    {
        adFloatCoefficientVariableSumNode* pn = dynamic_cast<adFloatCoefficientVariableSumNode*>(it->second);
        if(pn)
            mem += ( sizeof(adFloatCoefficientVariableSumNode) + pn->m_sum.size() * (sizeof(size_t) + sizeof(daeFloatCoefficientVariableProduct)) );
    }
    printf("adFloatCoefficientVariableSumNode count = %d, memory = %d MB\n", (int)fcvs_map.size(), mem / (1024*1024));

    mem = 0;
    for(it = unary_map.begin(); it != unary_map.end(); it++)
        mem += ( sizeof(adUnaryNode) );
    printf("adUnaryNode count = %d, memory = %d MB\n", (int)unary_map.size(), mem / (1024*1024));

    mem = 0;
    for(it = binary_map.begin(); it != binary_map.end(); it++)
        mem += ( sizeof(adBinaryNode) );
    printf("adBinaryNode count = %d, memory = %d MB\n", (int)binary_map.size(), mem / (1024*1024));

    mem = 0;
    for(it = variable_map.begin(); it != variable_map.end(); it++)
    {
        adSetupVariableNode* pn = dynamic_cast<adSetupVariableNode*>(it->second);
        if(pn)
            mem += ( sizeof(adSetupVariableNode) + pn->m_arrDomains.capacity() * sizeof(daeDomainIndex) );
    }
    printf("adSetupVariableNode count = %d, memory = %d MB\n", (int)variable_map.size(), mem / (1024*1024));

    mem = 0;
    for(it = timederiv_map.begin(); it != timederiv_map.end(); it++)
    {
        adSetupTimeDerivativeNode* pn = dynamic_cast<adSetupTimeDerivativeNode*>(it->second);
        if(pn)
            mem += ( sizeof(adSetupTimeDerivativeNode) + pn->m_arrDomains.capacity() * sizeof(daeDomainIndex) );
    }
    printf("adSetupTimeDerivativeNode count = %d, memory = %d MB\n", (int)timederiv_map.size(), mem / (1024*1024));
}
*/
template <int dim>
void dealiiFiniteElementSystem<dim>::ClearAssembledSystem()
{
    system_matrix.reset();
    system_matrix_dt.reset();
    system_rhs.reset();
}


template <int dim>
daetools::daeMatrix<adouble>* dealiiFiniteElementSystem<dim>::Asystem() const
{
    BlockSparseMatrix<adouble>* Aij = const_cast<BlockSparseMatrix<adouble>*>(system_matrix.get());
    return new daeFEBlockMatrix<adouble>(*Aij);
}

template <int dim>
daetools::daeMatrix<adouble>* dealiiFiniteElementSystem<dim>::Msystem() const
{
    BlockSparseMatrix<adouble>* Mij = const_cast<BlockSparseMatrix<adouble>*>(system_matrix_dt.get());
    return new daeFEBlockMatrix<adouble>(*Mij);
}

template <int dim>
daetools::daeArray<adouble>* dealiiFiniteElementSystem<dim>::Fload() const
{
    BlockVector<adouble>* Fi = const_cast<BlockVector<adouble>*>(system_rhs.get());
    return new daeFEBlockArray<adouble>(*Fi);
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
std::vector<bool> dealiiFiniteElementSystem<dim>::GetBoundaryDOFs(const std::string& variableName, const std::set<types::boundary_id>& setBoundaryIDs)
{
    std::vector<bool> selectedDOFS;

    selectedDOFS.resize(dof_handler.n_dofs());

    typename map_string_ComponentMask::iterator iter = m_mapComponentMasks.find(variableName);
    if(iter == m_mapComponentMasks.end())
        throw std::runtime_error(std::string("Invalid DOF name in function GetBoundaryDOFs"));

    const ComponentMask& component_mask = iter->second;

    DoFTools::extract_boundary_dofs(dof_handler, component_mask, selectedDOFS, setBoundaryIDs);

    return selectedDOFS;
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
