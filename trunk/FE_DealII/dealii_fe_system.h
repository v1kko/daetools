#ifndef DEAL_II_SYSTEM_H
#define DEAL_II_SYSTEM_H

#include <typeinfo>
#include <fstream>
#include <iostream>
#include <iterator>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "dealii_common.h"
#include "dealii_datareporter.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_abf.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_bernstein.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "adouble_template_inst.h"

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
    feCellContextImpl<dim>
*******************************************************************/
template<int dim, typename FE_VALUES>
class feCellContextImpl : public feCellContext<dim>
{
public:
    typedef typename std::map<std::string,  const Function<dim,double>*>  map_String_FunctionPtr;
    typedef typename std::map<std::string,  const Function<dim,adouble>*> map_String_adoubleFunctionPtr;
    typedef typename std::map< std::string, boost::variant<FEValuesExtractors::Scalar, FEValuesExtractors::Vector> > map_String_FEValuesExtractor;

    feCellContextImpl(FE_VALUES&                      fe_values,
                      map_String_FunctionPtr&         mapFunctions,
                      map_String_adoubleFunctionPtr&  mapAdoubleFunctions,
                      map_String_FEValuesExtractor&   mapExtractors):
        m_fe_values(fe_values),
        m_mapFunctions(mapFunctions),
        m_mapAdoubleFunctions(mapAdoubleFunctions),
        m_mapExtractors(mapExtractors),
        m_i(-1),
        m_j(-1),
        m_q(-1)
    {
    }

public:
    /*
     * Scalar-data functions
     */
    virtual double value(const std::string& variableName,
                         const unsigned int i,
                         const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call to phi() for the non-scalar variable: " + variableName);

        return m_fe_values[*extractorScalar].value(i, q);
    }

    Tensor<1,dim> gradient (const std::string& variableName,
                            const unsigned int i,
                            const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call to dphi() for the non-scalar variable: " + variableName);

        return m_fe_values[*extractorScalar].gradient(i, q);
    }

    Tensor<2,dim> hessian (const std::string& variableName,
                           const unsigned int i,
                           const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call to d2phi() for the non-scalar variable: " + variableName);

        return m_fe_values[*extractorScalar].hessian(i, q);
    }

    /*
     * Vector-data functions
     */
    Tensor<1,dim> vector_value(const std::string& variableName,
                               const unsigned int i,
                               const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to phi_vec() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].value(i, q);
    }

    Tensor<2,dim> vector_gradient (const std::string& variableName,
                                   const unsigned int i,
                                   const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to dphi_vec() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].gradient(i, q);
    }

    Tensor<3,dim> vector_hessian (const std::string& variableName,
                                  const unsigned int i,
                                  const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to d2phi_vec() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].hessian(i, q);
    }

    double divergence(const std::string& variableName,
                      const unsigned int i,
                      const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to div() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].divergence(i, q);
    }

    /*
    Tensor<1,1> curl_2D(const std::string& variableName,
                        const unsigned int i,
                        const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to curl() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].curl<2,2>(i, q);
    }

    Tensor<1,3> curl_3D(const std::string& variableName,
                        const unsigned int i,
                        const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to curl() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].curl<3,3>(i, q);
    }
    */

    virtual const Point<dim>& quadrature_point(const unsigned int q) const
    {
        return m_fe_values.quadrature_point(q);
    }

    virtual double JxW(const unsigned int q) const
    {
        return m_fe_values.JxW(q);
    }

    virtual const Tensor<1,dim>& normal_vector(const unsigned int q) const
    {
        return m_fe_values.normal_vector(q);
    }

    virtual const Function<dim,double>& function(const std::string& functionName) const
    {
        typename map_String_FunctionPtr::iterator iter = m_mapFunctions.find(functionName);
        if(iter == m_mapFunctions.end())
            throw std::runtime_error(std::string("Cannot find Function<dim> with the name ") + functionName);

        return *(iter->second);
    }

    virtual const Function<dim, adouble>& adouble_function(const std::string& functionName) const
    {
        typename map_String_adoubleFunctionPtr::iterator iter = m_mapAdoubleFunctions.find(functionName);
        if(iter == m_mapAdoubleFunctions.end())
            throw std::runtime_error(std::string("Cannot find Function<dim,adouble> with the name ") + functionName);

        return *(iter->second);
    }

    virtual unsigned int q() const
    {
        return m_q;
    }

    virtual unsigned int i() const
    {
        return m_i;
    }

    virtual unsigned int j() const
    {
        return m_j;
    }

public:
    FE_VALUES&                      m_fe_values;
    map_String_FunctionPtr&         m_mapFunctions;
    map_String_adoubleFunctionPtr&  m_mapAdoubleFunctions;
    map_String_FEValuesExtractor&   m_mapExtractors;
    unsigned int                    m_i;
    int                             m_j;
    int                             m_q;
};

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
    typedef typename std::map<unsigned int, const Function<dim>*>                  map_Uint_FunctionPtr;
    typedef typename std::pair<std::string, const Function<dim>*>                  pair_String_FunctionPtr;
    typedef typename std::map<unsigned int, std::vector<pair_String_FunctionPtr> > map_Uint_vector_pair_String_FunctionPtr;
    typedef typename std::map<unsigned int, feExpression<dim> >                    map_Uint_Expression;
    typedef typename std::map<std::string,  const Function<dim,double>*>           map_String_FunctionPtr;
    typedef typename std::map<std::string,  const Function<dim,adouble>*>          map_String_adoubleFunctionPtr;

    dealiiFiniteElementWeakForm()
    {
    }

public:
    bool                                    m_bNeedsUpdate;
    feExpression<dim>                       m_Aij;  // Stiffness matrix
    feExpression<dim>                       m_Mij;  // Mass matrix (dt)
    feExpression<dim>                       m_Fi;   // Load vector (rhs)
    map_Uint_Expression                     m_faceAij;
    map_Uint_Expression                     m_faceFi;
    map_String_FunctionPtr                  m_functions;
    map_String_adoubleFunctionPtr           m_adouble_functions;
    map_Uint_vector_pair_String_FunctionPtr m_functionsDirichletBC;
};

/******************************************************************
    dealiiFiniteElementSystem<dim>
*******************************************************************/
template <int dim>
class dealiiFiniteElementSystem : public daeFiniteElementObject
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

public:
    dealiiFiniteElementSystem();
    virtual ~dealiiFiniteElementSystem();

public:
    virtual void AssembleSystem();
    virtual bool NeedsReAssembling();
    virtual void ReAssembleSystem();

    virtual void                        RowIndices(unsigned int row, std::vector<unsigned int>& narrIndices) const;
    virtual dae::daeMatrix<adouble>*     Asystem() const; // Stiffness matrix
    virtual dae::daeMatrix<adouble>*     Msystem() const; // Mass matrix (dt)
    virtual dae::daeArray<adouble>*      Fload() const;   // Load vector
    virtual daeFiniteElementObjectInfo  GetObjectInfo() const;
    virtual std::vector<unsigned int>   GetDOFtoBoundaryMap();
    virtual dealIIDataReporter*         CreateDataReporter();

    void Initialize(const std::string&                meshFilename,
                    const Quadrature<dim>&            quadrature,
                    const Quadrature<dim-1>&          faceQuadrature,
                    vector_DOFs&                      DOFs,
                    dealiiFiniteElementWeakForm<dim>& weakForm);

protected:
    void setup_system();
    void assemble_system();
    void update_block(unsigned int block_index, double* values, unsigned int n);
    void write_solution(const std::string& strFilename);

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

    SmartPointer< Quadrature<dim>   >  m_quadrature_formula;
    SmartPointer< Quadrature<dim-1> >  m_face_quadrature_formula;

    // Model-specific data
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
}

template <int dim>
void dealiiFiniteElementSystem<dim>::Initialize(const std::string&                meshFilename,
                                                const Quadrature<dim>&            quadrature,
                                                const Quadrature<dim-1>&          faceQuadrature,
                                                vector_DOFs&                      DOFs,
                                                dealiiFiniteElementWeakForm<dim>& weakForm)
{
    m_weakForm = &weakForm;
    m_DOFs     = DOFs;

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

    m_block_component.clear();
    for(unsigned int i = 0; i < n_dofs; i++)
        m_block_component.insert(m_block_component.end(), m_DOFs[i]->m_nMultiplicity, i);

    DoFRenumbering::component_wise(dof_handler, m_block_component);

    m_dofs_per_block.resize(n_dofs);
    DoFTools::count_dofs_per_block (dof_handler, m_dofs_per_block, m_block_component);

    for(unsigned int i = 0; i < m_no_components; i++)
        printf("m_block_component[%d] = %d\n", i, m_block_component[i]);
    for(unsigned int i = 0; i < n_dofs; i++)
        printf("m_dofs_per_block[%d] = %d\n", i, m_dofs_per_block[i]);

    const unsigned int n_couplings = dof_handler.max_couplings_between_dofs();

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
void dealiiFiniteElementSystem<dim>::assemble_system()
{
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
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
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

    feCellContextImpl< dim, FEValues<dim> >      cellContext    (fe_values,      m_weakForm->m_functions, m_weakForm->m_adouble_functions, mapExtractors);
    feCellContextImpl< dim, FEFaceValues<dim> >  cellFaceContext(fe_face_values, m_weakForm->m_functions, m_weakForm->m_adouble_functions, mapExtractors);

    // Interpolate Dirichlet boundary conditions on the system matrix and rhs
    std::vector< std::map<types::global_dof_index, double> > arr_boundary_values_map;

    typedef typename std::pair<std::string, const Function<dim>*>                  pair_String_FunctionPtr;
    typedef typename std::map<unsigned int, std::vector<pair_String_FunctionPtr> > map_Uint_vector_pair_String_FunctionPtr;
    for(typename map_Uint_vector_pair_String_FunctionPtr::const_iterator it = m_weakForm->m_functionsDirichletBC.begin(); it != m_weakForm->m_functionsDirichletBC.end(); it++)
    {
        const unsigned int   id                         = it->first;
        const std::vector<pair_String_FunctionPtr>& bcs = it->second;

        for(int k = 0; k < bcs.size(); k++)
        {
            pair_String_FunctionPtr p = bcs[k];
            const std::string    variableName =  p.first;
            const Function<dim>& fun          = *p.second;

            typename map_string_ComponentMask::iterator iter = mapComponentMasks.find(variableName);
            if(iter == mapComponentMasks.end())
                throw std::runtime_error("Cannot find variable: " + variableName + " in the DirichletBC dictionary");

            std::cout << "Interpolate DirichletBC at id: " << id << " for variable " << variableName << " with sample value for component = 0 and at point (0,0,0): " << fun.value(Point<dim>(0,0,0), 0) << std::endl;

            std::map<types::global_dof_index, double> boundary_values;
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      id,
                                                      fun,
                                                      boundary_values,
                                                      iter->second);
            arr_boundary_values_map.push_back(boundary_values);

            //printf("bc[%d] = [", id);
            //for(std::map<types::global_dof_index, double>::const_iterator it = boundary_values.begin(); it != boundary_values.end(); it++)
            //    printf("(%d,%f) ", it->first, it->second);
            //printf("]\n");
        }
    }

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for(int cellCounter = 0; cell != endc; ++cell, ++cellCounter)
    {
        cell_matrix.clear();
        cell_matrix_dt.clear();
        std::fill(cell_rhs.begin(), cell_rhs.end(), adouble(0.0));

        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

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

                        //std::cout << "  cell_matrix 1 = " << cell_matrix(i,j).getValue() << ", +value " << result.m_value << std::endl;
                        cell_matrix(i,j) += getValueFromNumber<dim>(result);
                        //std::cout << "  cell_matrix 2 = " << cell_matrix(i,j).getValue() << std::endl;

                        //std::cout << (boost::format("cell_matrix[%s](q=%d, i=%d, j=%d) = %f") % m_weakForm.m_strVariableName % q_point % i % j % result.m_value).str() << std::endl;
                    }

                    /* Mass matrix (Mij) */
                    if(m_weakForm->m_Mij.m_node)
                    {
                        feRuntimeNumber<dim> result = m_weakForm->m_Mij.m_node->Evaluate(&cellContext);
                        if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                            throw std::runtime_error(std::string("Invalid Mij expression specified (it must be a scalar value or adouble)"));

                        cell_matrix_dt(i,j) += getValueFromNumber<dim>(result);
                    }
                }

                /* Load vector (Fi) */
                if(m_weakForm->m_Fi.m_node)
                {
                    cellContext.m_j = -1; // Set the unphysical value since it must not be used in Fi contributions

                    feRuntimeNumber<dim> result = m_weakForm->m_Fi.m_node->Evaluate(&cellContext);
                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                        throw std::runtime_error(std::string("Invalid Fi expression specified: (it must be a scalar value or adouble)"));

                    //std::cout << "cell rhs 1 = " << cell_rhs[i] << ", +value " << result.m_value << std::endl;
                    cell_rhs[i] += getValueFromNumber<dim>(result);
                    //std::cout << "cell rhs 2 = " << cell_rhs[i] << std::endl;
                }
            }
        }

        /* Boundary conditions of the Neumann or Robin type. */
        for(unsigned int face = 0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if(cell->face(face)->at_boundary())
            {
                fe_face_values.reinit (cell, face);

                const unsigned int id = cell->face(face)->boundary_id();

                typename map_Uint_Expression::const_iterator itAij = m_weakForm->m_faceAij.find(id);
                typename map_Uint_Expression::const_iterator itFi  = m_weakForm->m_faceFi.find(id);

                // If there is face Aij or Fi (or both)
                if(itAij != m_weakForm->m_faceAij.end() || itFi != m_weakForm->m_faceFi.end())
                {
                    for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                    {
                        cellFaceContext.m_q = q_point;

                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                            cellFaceContext.m_i = i;

                            for(unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                cellFaceContext.m_j = j;

                                if(itAij != m_weakForm->m_faceAij.end())
                                {
                                    const feExpression<dim>& faceAij = itAij->second;
                                    if(!faceAij.m_node)
                                        throw std::runtime_error(std::string("Empty faceAij expression specified"));

                                    feRuntimeNumber<dim> result = faceAij.m_node->Evaluate(&cellFaceContext);
                                    if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                                        throw std::runtime_error(std::string("Invalid faceAij expression specified (it must be a scalar value or adouble)"));

                                    cell_matrix(i,j) += getValueFromNumber<dim>(result);
                                }
                            }

                            if(itFi != m_weakForm->m_faceFi.end())
                            {
                                const feExpression<dim>& faceFi = itFi->second;
                                if(!faceFi.m_node)
                                    throw std::runtime_error(std::string("Empty faceFi expression specified"));

                                cellFaceContext.m_j = -1; // Set the unphysical value since it must not be used in faceFi contributions

                                feRuntimeNumber<dim> result = faceFi.m_node->Evaluate(&cellFaceContext);
                                if(result.m_eType != eFEScalar && result.m_eType != eFEScalar_adouble)
                                    throw std::runtime_error(std::string("Invalid faceFi expression specified (it must be a scalar value or adouble)"));

                                cell_rhs[i] += getValueFromNumber<dim>(result);
                            }
                        }
                    }
                }
            }
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
        // Apply Dirichlet boundary conditions on the stiffness matrix and rhs
        for(size_t i = 0; i < arr_boundary_values_map.size(); i++)
        {
            std::map<types::global_dof_index, double>& boundary_values = arr_boundary_values_map[i];
            MatrixTools::local_apply_boundary_values(boundary_values,
                                                     local_dof_indices,
                                                     cell_matrix,
                                                     cell_rhs);
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

        /* This is some voodoo-mojo mumbo-jumbo uncomprehensible crap... What to do with it? */
        types::global_dof_index id;
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            id = mapGlobalDOFtoBoundary[ local_dof_indices[i] ];
            if(id != numbers::invalid_dof_index)
            {
                /* Achtung, Achtung!!
                 * We assume here that if a DOF is on a boundary that it has no system_matrix__dt contribution.
                 * What if someone wants to have it on the boundary? */

                // This dof IS on one of boundaries; therefore, remove its contributions to the cell_matrix_dt

                // 1. Reset the whole row 'i'
                for(unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix_dt(i, j) = 0;

                // 2. Reset the whole column 'i'
                for(unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix_dt(j, i) = 0;
            }
        }

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
    } // End cell iteration

    // Hanging nodes are NOT supported!! There must be a way to stop simulation if hanging nodes are detected

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
