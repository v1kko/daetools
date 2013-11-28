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

namespace dae
{
namespace fe_solver
{
using namespace dealii;

/******************************************************************
    feCellContextImpl<dim>
*******************************************************************/
template<int dim, typename FE_VALUES>
class feCellContextImpl : public feCellContext<dim>
{
public:
    typedef typename std::map<std::string,  const Function<dim>*> map_String_FunctionPtr;
    typedef typename std::map< std::string, boost::variant<FEValuesExtractors::Scalar, FEValuesExtractors::Vector> > map_String_FEValuesExtractor;

    feCellContextImpl(FE_VALUES&                      fe_values,
                      map_String_FunctionPtr&         mapFunctions,
                      map_String_FEValuesExtractor&   mapExtractors):
        m_fe_values(fe_values),
        m_mapFunctions(mapFunctions),
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
        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&m_mapExtractors[variableName]);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call to scalar phi() for the non-scalar variable: " + variableName);

        return m_fe_values[*extractorScalar].value(i, q);
    }

    Tensor<1,dim> gradient (const std::string& variableName,
                            const unsigned int i,
                            const unsigned int q) const
    {
        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&m_mapExtractors[variableName]);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call to scalar dphi() for the non-scalar variable: " + variableName);

        return m_fe_values[*extractorScalar].gradient(i, q);
    }

    Tensor<2,dim> hessian (const std::string& variableName,
                           const unsigned int i,
                           const unsigned int q) const
    {
        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&m_mapExtractors[variableName]);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call to scalar d2phi() for the non-scalar variable: " + variableName);

        return m_fe_values[*extractorScalar].hessian(i, q);
    }

    /*
     * Vector-data functions
     */
    Tensor<1,dim> vector_value(const std::string& variableName,
                               const unsigned int i,
                               const unsigned int q) const
    {
        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&m_mapExtractors[variableName]);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to scalar phi_vec() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].value(i, q);
    }

    Tensor<2,dim> vector_gradient (const std::string& variableName,
                                   const unsigned int i,
                                   const unsigned int q) const
    {
        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&m_mapExtractors[variableName]);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to scalar dphi_vec() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].gradient(i, q);
    }

    Tensor<3,dim> vector_hessian (const std::string& variableName,
                                  const unsigned int i,
                                  const unsigned int q) const
    {
        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&m_mapExtractors[variableName]);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to scalar d2phi_vec() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].hessian(i, q);
    }

    double divergence(const std::string& variableName,
                      const unsigned int i,
                      const unsigned int q) const
    {
        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&m_mapExtractors[variableName]);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to scalar div() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].divergence(i, q);
    }

    /*
    virtual ??? curl(const std::string& variableName,
                     const unsigned int i,
                     const unsigned int j) const = 0;
    */

    virtual const Point<dim>& quadrature_point(const unsigned int q) const
    {
        return m_fe_values.quadrature_point(q);
    }

    virtual double JxW(const unsigned int q) const
    {
        return m_fe_values.JxW(q);
    }

    virtual const Point<dim>& normal_vector(const unsigned int q) const
    {
        return m_fe_values.normal_vector(q);
    }

    virtual const Function<dim>& function(const std::string& functionName) const
    {
        typename map_String_FunctionPtr::iterator iter = m_mapFunctions.find(functionName);
        if(iter == m_mapFunctions.end())
            throw std::runtime_error(std::string("Cannot find Function<dim> with the name ") + functionName);

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
    map_String_FEValuesExtractor&   m_mapExtractors;
    unsigned int                    m_i;
    int                             m_j;
    int                             m_q;
};

/******************************************************************
    dealiiFiniteElementEquation<dim>
*******************************************************************/
template <int dim>
class dealiiFiniteElementEquation
{
public:
    typedef typename std::map<unsigned int, const Function<dim>*> map_Uint_FunctionPtr;
    typedef typename std::map<unsigned int, std::pair<const Function<dim>*, dealiiFluxType> > map_Uint_FunctionPtr_FunctionCall;
    typedef typename std::map<unsigned int, feExpression<dim> >   map_Uint_Expression;
    typedef typename std::map<std::string,  const Function<dim>*> map_String_FunctionPtr;

    dealiiFiniteElementEquation()
    {
    }

public:
    bool                                m_bNeedsUpdate;
    feExpression<dim>                   m_matAlocal; // Stiffness matrix
    feExpression<dim>                   m_matMlocal; // Mass matrix (dt)
    feExpression<dim>                   m_vecFlocal; // Load vector (rhs)
    map_Uint_Expression                 m_elementBoundary;
    map_Uint_Expression                 m_elementNeumann;
    map_Uint_FunctionPtr_FunctionCall   m_functionsNeumannBC;
    map_Uint_FunctionPtr                m_functionsDirichletBC;
    std::string                         m_strVariableName;
    std::string                         m_strVariableDescription;
    unsigned int                        m_nMultiplicity;

// Internal data
    SmartPointer< FE_Q<dim> >   m_fe_q;
};

/******************************************************************
    dealiiFiniteElementSystem<dim>
*******************************************************************/
template <int dim>
class dealiiFiniteElementSystem : public daeFiniteElementObject
{
typedef typename std::map<unsigned int, const Function<dim>*> map_Uint_FunctionPtr;
typedef typename std::map<unsigned int, std::pair<const Function<dim>*, dealiiFluxType> > map_Uint_FunctionPtr_FunctionCall;
typedef typename std::map<unsigned int, feExpression<dim> >   map_Uint_Expression;
typedef typename std::map<std::string,  const Function<dim>*> map_String_FunctionPtr;
typedef typename std::vector< dealiiFiniteElementEquation<dim>* > vector_Equations;
typedef typename std::map< std::string, boost::variant<FEValuesExtractors::Scalar, FEValuesExtractors::Vector> > map_String_FEValuesExtractor;

public:
    dealiiFiniteElementSystem();
    virtual ~dealiiFiniteElementSystem();

public:
    virtual void AssembleSystem();
    virtual bool NeedsReAssembling();
    virtual void ReAssembleSystem();

    virtual void                        RowIndices(unsigned int row, std::vector<unsigned int>& narrIndices) const;
    virtual dae::daeMatrix<real_t>*     Asystem() const; // Stiffness matrix
    virtual dae::daeMatrix<real_t>*     Msystem() const; // Mass matrix (dt)
    virtual dae::daeArray<real_t>*      Fload() const;   // Load vector
    virtual daeFiniteElementObjectInfo  GetObjectInfo() const;
    virtual std::vector<unsigned int>   GetDOFtoBoundaryMap();
    virtual dealIIDataReporter*         CreateDataReporter();

    void Initialize(const std::string&            meshFilename,
                    unsigned int                  polynomialOrder,
                    const Quadrature<dim>&        quadrature,
                    const Quadrature<dim-1>&      faceQuadrature,
                    const map_String_FunctionPtr& functions,
                    const vector_Equations&       equations);

protected:
    void setup_system();
    void assemble_system();
    void process_solution(const std::string& strFilename, const std::string& strVariableName, double* values, unsigned int n);

public:
    // Additional deal.II specific data
    Triangulation<dim>              triangulation;
    DoFHandler<dim>                 dof_handler;
    SmartPointer< FESystem<dim> >   fe;
    ConstraintMatrix                hanging_node_constraints;

    BlockSparsityPattern        sparsity_pattern;
    BlockSparseMatrix<double>   system_matrix;
    BlockSparseMatrix<double>   system_matrix_dt;
    BlockVector<double>         system_rhs;
    BlockVector<double>         solution;

    SmartPointer< Quadrature<dim>   >  m_quadrature_formula;
    SmartPointer< Quadrature<dim-1> >  m_face_quadrature_formula;

    // Model-specific data
    map_String_FunctionPtr                           m_functions;
    std::vector< dealiiFiniteElementEquation<dim>* > m_equations;
    dealiiFiniteElementEquation<dim> cdr;
    dealiiFiniteElementEquation<dim> cdr2;
    dealiiFiniteElementEquation<dim> cdr3;
};

template <int dim>
dealiiFiniteElementSystem<dim>::dealiiFiniteElementSystem():
    dof_handler (triangulation)
{
}

template <int dim>
void dealiiFiniteElementSystem<dim>::Initialize(const std::string&                      meshFilename,
                                                unsigned int                            polynomialOrder,
                                                const Quadrature<dim>&                  quadrature,
                                                const Quadrature<dim-1>&                faceQuadrature,
                                                const map_String_FunctionPtr&           functions,
                                                const vector_Equations&                 equations)
{
    if(equations.size() == 0)
        throw std::runtime_error("At least one equation must be specified");

    m_functions = functions;
    m_equations = equations;

    // Create FESystem
    std::vector<const FiniteElement<dim>*> arrFEs;
    std::vector<unsigned int> arrMultiplicities;
    for(unsigned int i = 0; i < m_equations.size(); i++)
    {
        m_equations[i]->m_fe_q = SmartPointer< FE_Q<dim> >( new FE_Q<dim>(polynomialOrder) );

        arrFEs.push_back(m_equations[i]->m_fe_q);
        arrMultiplicities.push_back(m_equations[i]->m_nMultiplicity);
    }

    fe = SmartPointer< FESystem<dim> >(new FESystem<dim>(arrFEs, arrMultiplicities));

    // Read the mesh
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);

    std::ifstream f(meshFilename);
    std::string extension = boost::filesystem::path(meshFilename).extension().c_str();
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
    DoFRenumbering::component_wise (dof_handler);

    unsigned int no_equations = m_equations.size();

    std::vector<types::global_dof_index> dofs_per_component (no_equations);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);

    const unsigned int n_couplings = dof_handler.max_couplings_between_dofs();

    sparsity_pattern.reinit (no_equations, no_equations);
    for(unsigned int i = 0; i < no_equations; i++)
        for(unsigned int j = 0; j < no_equations; j++)
            sparsity_pattern.block(i, j).reinit (dofs_per_component[i], dofs_per_component[j], n_couplings);

    sparsity_pattern.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    sparsity_pattern.compress();

    system_matrix.reinit (sparsity_pattern);
    system_matrix_dt.reinit (sparsity_pattern);

    solution.reinit (no_equations);
    for(unsigned int i = 0; i < no_equations; i++)
        solution.block(i).reinit (dofs_per_component[i]);
    solution.collect_sizes ();

    system_rhs.reinit (no_equations);
    for(unsigned int i = 0; i < no_equations; i++)
        system_rhs.block(i).reinit (dofs_per_component[i]);
    system_rhs.collect_sizes ();
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

    FullMatrix<double>  cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>  cell_matrix_dt(dofs_per_cell, dofs_per_cell);
    Vector<double>      cell_rhs(dofs_per_cell);

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
    std::vector<ComponentMask> componentMasks;

    unsigned int no_equations = m_equations.size();
    componentMasks.resize(no_equations);

    int currentIndex = 0;
    for(unsigned int eq = 0; eq < no_equations; eq++)
    {
        const dealiiFiniteElementEquation<dim>& equation = *m_equations[eq];

        if(equation.m_nMultiplicity == 1)
        {
            std::cout << (boost::format("VariableName = %s, FEValuesExtractors::Scalar(index = %d)") % equation.m_strVariableName % currentIndex).str() << std::endl;
            mapExtractors[equation.m_strVariableName] = FEValuesExtractors::Scalar(currentIndex);
            componentMasks[eq] = fe->component_mask(FEValuesExtractors::Scalar(currentIndex));
        }
        else
        {
            mapExtractors[equation.m_strVariableName] = FEValuesExtractors::Vector(currentIndex);
            componentMasks[eq] = fe->component_mask(FEValuesExtractors::Vector(currentIndex));
        }

        currentIndex += equation.m_nMultiplicity;
    }

    feCellContextImpl< dim, FEValues<dim> >      cellContext    (fe_values,      m_functions, mapExtractors);
    feCellContextImpl< dim, FEFaceValues<dim> >  cellFaceContext(fe_face_values, m_functions, mapExtractors);

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for(int cellCounter = 0; cell != endc; ++cell, ++cellCounter)
    {
        cell_matrix    = 0;
        cell_matrix_dt = 0;
        cell_rhs       = 0;

        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        //std::cout << (boost::format("local_dof_indices = %s") % dae::toString(local_dof_indices)).str() << std::endl;

        for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            cellContext.m_q = q_point;

            for(unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                cellContext.m_i = i;

                for(unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cellContext.m_j = j;

                    for(unsigned int eq = 0; eq < m_equations.size(); eq++)
                    {
                        const dealiiFiniteElementEquation<dim>& equation = *m_equations[eq];

                        if(equation.m_matAlocal.m_node)
                        {
                            feRuntimeNumber<dim> result = equation.m_matAlocal.m_node->Evaluate(&cellContext);
                            if(result.m_eType != eFEScalar)
                                throw std::runtime_error(std::string("Invalid element matrix contribution expression specified for equation: ") +
                                                         equation.m_strVariableName + std::string(" (it must be a scalar value)"));

                            cell_matrix(i,j) += result.m_value;

                            //std::cout << (boost::format("cell_matrix[%s](q=%d, i=%d, j=%d) = %f") % equation.m_strVariableName % q_point % i % j % result.m_value).str() << std::endl;
                        }

                        /* Accumulation term (in a separate matrix) */
                        if(equation.m_matMlocal.m_node)
                        {
                            feRuntimeNumber<dim> result = equation.m_matMlocal.m_node->Evaluate(&cellContext);
                            if(result.m_eType != eFEScalar)
                                throw std::runtime_error(std::string("Invalid element matrix_dt contribution expression specified for equation: ") +
                                                         equation.m_strVariableName + std::string(" (it must be a scalar value)"));

                            cell_matrix_dt(i,j) += result.m_value;
                        }
                    }
                }

                /* Generation */
                for(unsigned int eq = 0; eq < m_equations.size(); eq++)
                {
                    const dealiiFiniteElementEquation<dim>& equation = *m_equations[eq];
                    if(equation.m_vecFlocal.m_node)
                    {
                        feRuntimeNumber<dim> result = equation.m_vecFlocal.m_node->Evaluate(&cellContext);
                        if(result.m_eType != eFEScalar)
                            throw std::runtime_error(std::string("Invalid element rhs contribution expression specified for equation: ") +
                                                     equation.m_strVariableName + std::string(" (it must be a scalar value)"));
                        cell_rhs(i) += result.m_value;
                    }
                }
            }
        }

        for(unsigned int face = 0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if(cell->face(face)->at_boundary())
            {
                fe_face_values.reinit (cell, face);

                const unsigned int id = cell->face(face)->boundary_indicator();

                // Neumann BC
                for(unsigned int eq = 0; eq < m_equations.size(); eq++)
                {
                    const dealiiFiniteElementEquation<dim>& equation = *m_equations[eq];
                    typename map_Uint_FunctionPtr_FunctionCall::const_iterator itFunction   = equation.m_functionsNeumannBC.find(id);
                    typename map_Uint_Expression::const_iterator               itExpression = equation.m_elementNeumann.find(id);

                    if(itExpression != equation.m_elementNeumann.end())
                    {
                        const feExpression<dim>& elementNeumann = itExpression->second;

                        for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                        {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                                feRuntimeNumber<dim> result = elementNeumann.m_node->Evaluate(&cellFaceContext);
                                if(result.m_eType != eFEScalar)
                                    throw std::runtime_error(std::string("Invalid element neumann boundary condition expression specified for equation: ") +
                                                             equation.m_strVariableName + std::string(" (it must be a scalar value)"));

                                cell_rhs(i) += result.m_value;
                            }
                        }
                    }
                    else if(itFunction != equation.m_functionsNeumannBC.end())
                    {
                        double Neumann_value;
                        std::pair<const Function<dim>*, dealiiFluxType> pairNeumann = itFunction->second;
                        const Function<dim>& Fneumann = *pairNeumann.first;
                        const dealiiFluxType fluxType =  pairNeumann.second;

                        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&mapExtractors[equation.m_strVariableName]);
                        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&mapExtractors[equation.m_strVariableName]);

                        for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                        {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                                if(extractorScalar)
                                {
                                    if(fluxType == eConstantFlux)
                                        Neumann_value = Fneumann.value(fe_face_values.quadrature_point(q_point));
                                    else if(fluxType == eGradientFlux)
                                        Neumann_value = Fneumann.gradient(fe_face_values.quadrature_point(q_point)) *
                                                        fe_face_values.normal_vector(q_point);
                                    else
                                        throw std::runtime_error("Invalid NeumannBC function call type");

                                    /*
                                    std::cout << (boost::format("  Setting NeumanBC (cell=%d, point=(%d, %d), id=%d)[%d] = %f")  % cellCounter
                                                                                                                                 % fe_face_values.quadrature_point(q_point)(0)
                                                                                                                                 % fe_face_values.quadrature_point(q_point)(1)
                                                                                                                                 % id
                                                                                                                                 % i
                                                                                                                                 % (Neumann_value
                                                                                                                                    * fe_face_values[*extractorScalar].value(i, q_point)
                                                                                                                                    * fe_face_values.JxW(q_point))
                                                 ).str() << std::endl;
                                    */

                                    cell_rhs(i) += Neumann_value
                                                   *
                                                   fe_face_values[*extractorScalar].value(i, q_point)
                                                   *
                                                   fe_face_values.JxW(q_point);
                                }
                                else if(extractorVector)
                                {
                                    throw std::runtime_error("Not implemented");
                                    /*
                                    cell_rhs(i) += -Fneumann.value(fe_face_values.quadrature_point(q_point))
                                                   *
                                                   (
                                                      fe_face_values[*extractorVector].value(i, q_point)
                                                      *
                                                      fe_face_values.normal_vector(q_point)
                                                   )
                                                   *
                                                   fe_face_values.JxW(q_point);
                                    */
                                }
                                else
                                    throw std::runtime_error("Invalid FEValuesExtractor type");
                            }
                        }
                    }
                    else
                    {
                        // Not found
                        // Do nothing or do some default action (perhaps zero gradient?)
                    }
                }
            }
        }

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

        // Add to the system matrices/vector
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                system_matrix.add   (local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));
                system_matrix_dt.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_dt(i,j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    // If using refined grids condense hanging nodes
    hanging_node_constraints.condense(system_matrix);
    hanging_node_constraints.condense(system_rhs);

    // What about this matrix? Should it also be condensed?
    //hanging_node_constraints.condense(system_matrix_dt);

    // Apply Dirichlet boundary conditions on the system matrix and rhs
    for(unsigned int eq = 0; eq < m_equations.size(); eq++)
    {
        const dealiiFiniteElementEquation<dim>& equation = *m_equations[eq];

        for(typename map_Uint_FunctionPtr::const_iterator it = equation.m_functionsDirichletBC.begin(); it != equation.m_functionsDirichletBC.end(); it++)
        {
            const unsigned int    id =  it->first;
            const Function<dim>& fun = *it->second;

            std::cout << "Setting DirichletBC at id " << id << " with sample value " << fun.value(Point<dim>(0,0,0)) << std::endl;

            std::map<types::global_dof_index, double> boundary_values;
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      id,
                                                      fun,
                                                      boundary_values,
                                                      componentMasks[eq]);
            MatrixTools::apply_boundary_values (boundary_values,
                                                system_matrix,
                                                solution,
                                                system_rhs);
        }
    }
}

template <int dim>
void dealiiFiniteElementSystem<dim>::process_solution(const std::string& strFilename, const std::string& strVariableName, double* values, unsigned int n)
{
    for(unsigned int i = 0; i < m_equations.size(); i++)
    {
        if(m_equations[i]->m_strVariableName == strVariableName)
        {
            Vector<double>& block_i = solution.block(i);
            if(block_i.size() != n)
                throw std::runtime_error("The size of the variable value does not match the size of the solution");

            for(size_t k = 0; k < n; k++)
                block_i[k] = values[k];

            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(block_i, strVariableName);
            data_out.build_patches(fe->degree);
            std::ofstream output(strFilename);
            data_out.write_vtk(output);
        }
    }
/*
    //std::cout << "solution after solve:" << solution << std::endl;

    // We may call distribute() on solution to fix hanging nodes
    hanging_node_constraints.distribute(solution);

    //std::cout << "solution after distribute:" << solution << std::endl;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, strVariableName);
    data_out.build_patches(fe->degree);
    std::ofstream output(strFilename);
    data_out.write_vtk(output);
*/
}

template <int dim>
daeFiniteElementObjectInfo dealiiFiniteElementSystem<dim>::GetObjectInfo() const
{
    daeFiniteElementObjectInfo feObjectInfo;

    unsigned int no_equations = m_equations.size();
    feObjectInfo.m_VariableInfos.resize(no_equations);

    std::vector<types::global_dof_index> dofs_per_component (no_equations);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);

    if(std::min(dofs_per_component.begin(), dofs_per_component.end()) != std::max(dofs_per_component.begin(), dofs_per_component.end()))
        std::runtime_error("Number of DOFs pre each component must be equal");

    feObjectInfo.m_nNumberOfDOFsPerVariable = dofs_per_component[0];
    feObjectInfo.m_nTotalNumberDOFs         = dof_handler.n_dofs();

    unsigned int component_counter = 0;
    for(unsigned int i = 0; i < no_equations; i++)
    {
        feObjectInfo.m_VariableInfos[i].m_strName              = m_equations[i]->m_strVariableName;
        feObjectInfo.m_VariableInfos[i].m_strDescription       = m_equations[i]->m_strVariableDescription;
        feObjectInfo.m_VariableInfos[i].m_nMultiplicity        = m_equations[i]->m_nMultiplicity;
        for(unsigned int j = 0; j < m_equations[i]->m_nMultiplicity; j++)
            feObjectInfo.m_VariableInfos[i].m_narrDOFsPerComponent.push_back(dofs_per_component[component_counter + j]);

        component_counter += m_equations[i]->m_nMultiplicity;
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

    unsigned int block_row = -1;
    unsigned int local_row = -1;
    unsigned int crow = 0;
    for(unsigned int i = 0; i < sparsity_pattern.n_block_rows(); i++)
    {
        // If row is less than of the rows in the blocks looped over so far then it belongs to that block_row
        if(row < crow + sparsity_pattern.block(i, 0).n_rows())
        {
            block_row = i;
            local_row = row - crow;
            break;
        }
        crow += sparsity_pattern.block(i, 0).n_rows();
    }
    if(block_row == -1)
        throw std::runtime_error((boost::format("Cannot locate a block-row for the row %d") % row).str());

    //std::cout << "block_row = " << block_row << std::endl;
    //std::cout << "local_row = " << local_row << std::endl;
    //std::cout << "row = " << row << std::endl;
    //std::cout << "sparsity_pattern.row_length(row) = " << sparsity_pattern.row_length(row) << std::endl;

    const BlockIndices& column_indices = sparsity_pattern.get_column_indices();
    narrIndices.reserve(sparsity_pattern.row_length(row));
    for(unsigned int block_column = 0; block_column < sparsity_pattern.n_block_cols(); block_column++)
    {
        const SparsityPattern& block = sparsity_pattern.block(block_row, block_column);
        SparsityPattern::iterator it     = block.begin(local_row);
        SparsityPattern::iterator it_end = block.end(local_row);
        for(; it != it_end; it++)
            narrIndices.push_back(column_indices.local_to_global(block_column, it->column()));
    }

    //std::cout << "narrIndices = " << toString(narrIndices) << std::endl;
}

template <int dim>
dae::daeMatrix<double>* dealiiFiniteElementSystem<dim>::Asystem() const
{
    return new daeFEBlockMatrix<double>(system_matrix);
}

template <int dim>
dae::daeMatrix<double>* dealiiFiniteElementSystem<dim>::Msystem() const
{
    return new daeFEBlockMatrix<double>(system_matrix_dt);
}

template <int dim>
dae::daeArray<double>* dealiiFiniteElementSystem<dim>::Fload() const
{
    return new daeFEBlockArray<double>(system_rhs);
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
    std::map<std::string, size_t> mapVariables;
    unsigned int no_equations = m_equations.size();
    for(unsigned int i = 0; i < no_equations; i++)
        mapVariables[m_equations[i]->m_strVariableName] = i;

    fnProcessSolution callback(boost::bind(&dealiiFiniteElementSystem<dim>::process_solution, this, _1, _2, _3, _4));
    return new dealIIDataReporter(callback, mapVariables);
}


}
}

#endif
