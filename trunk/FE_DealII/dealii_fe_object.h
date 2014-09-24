#ifndef DEAL_II_MODEL_IMPLEMENTATION_H
#define DEAL_II_MODEL_IMPLEMENTATION_H

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
//#include <deal.II/lac/vector.templates.h>
#include <deal.II/lac/full_matrix.h>
//#include <deal.II/lac/full_matrix.templates.h>
#include <deal.II/lac/sparse_matrix.h>
//#include <deal.II/lac/sparse_matrix.templates.h>
//#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
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
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

namespace dae
{
namespace fe_solver
{
using namespace dealii;

///******************************************************************
//    feCellContextImpl<dim>
//*******************************************************************/
//template<int dim>
//class feCellContextImpl : public feCellContext<dim>
//{
//public:
//    typedef typename std::map<std::string,  const Function<dim>*> map_String_FunctionPtr;

//    feCellContextImpl(  FiniteElement<dim>*         fe,
//                        DoFHandler<dim>&            dof_handler,
//                        Quadrature<dim>&            quadrature_formula,
//                        Quadrature<dim-1>&          face_quadrature_formula,
//                        const unsigned int          dofs_per_cell,
//                        const unsigned int          n_q_points,
//                        const unsigned int          faces_per_cell,
//                        const unsigned int          n_face_q_points,
//                        std::vector<unsigned int>&  local_dof_indices,
//                        FEValues<dim>&              fe_values,
//                        FEFaceValues<dim>&          fe_face_values,
//                        map_String_FunctionPtr&     mapFunctions):
//        m_fe(fe),
//        m_dof_handler(dof_handler),
//        m_quadrature_formula(quadrature_formula),
//        m_face_quadrature_formula(face_quadrature_formula),
//        m_dofs_per_cell(dofs_per_cell),
//        m_n_q_points(n_q_points),
//        m_faces_per_cell(faces_per_cell),
//        m_n_face_q_points(n_face_q_points),
//        m_local_dof_indices(local_dof_indices),
//        m_fe_values(fe_values),
//        m_fe_face_values(fe_face_values),
//        m_mapFunctions(mapFunctions),
//        m_i(-1),
//        m_j(-1),
//        m_q(-1)
//    {
//    }

//public:
//    virtual double shape_value(const unsigned int i,
//                               const unsigned int q) const
//    {
//        return m_fe_values.shape_value(i, q);
//    }

//    virtual double shape_value_component (const unsigned int i,
//                                          const unsigned int q,
//                                          const unsigned int component) const
//    {
//        return m_fe_values.shape_value_component(i, q, component);
//    }

//    virtual const Tensor<1,dim>& shape_grad (const unsigned int i,
//                                             const unsigned int q) const
//    {
//        return m_fe_values.shape_grad(i, q);
//    }

//    virtual Tensor<1,dim> shape_grad_component (const unsigned int i,
//                                                const unsigned int q,
//                                                const unsigned int component) const
//    {
//        return m_fe_values.shape_grad_component(i, q, component);
//    }

//    virtual const Tensor<2,dim>& shape_hessian (const unsigned int i,
//                                                const unsigned int q) const
//    {
//        return m_fe_values.shape_hessian(i, q);
//    }

//    virtual Tensor<2,dim> shape_hessian_component (const unsigned int i,
//                                                   const unsigned int q,
//                                                   const unsigned int component) const
//    {
//        return m_fe_values.shape_hessian_component(i, q, component);
//    }

//    virtual const Point<dim>& quadrature_point (const unsigned int q) const
//    {
//        return m_fe_values.quadrature_point(q);
//    }

//    virtual double JxW (const unsigned int q) const
//    {
//        return m_fe_values.JxW(q);
//    }

//    virtual const Point<dim>& normal_vector (const unsigned int q) const
//    {
//        return m_fe_values.normal_vector(q);
//    }

//    virtual const Function<dim>& function (const std::string& name) const
//    {
//        typename map_String_FunctionPtr::iterator iter = m_mapFunctions.find(name);
//        if(iter == m_mapFunctions.end())
//            throw std::runtime_error(std::string("Cannot find Function<dim> with the name ") + name);

//        return *(iter->second);
//    }

//    virtual unsigned int q() const
//    {
//        return m_q;
//    }

//    virtual unsigned int i() const
//    {
//        return m_i;
//    }

//    virtual unsigned int j() const
//    {
//        return m_j;
//    }

//public:
//    FiniteElement<dim>*         m_fe;
//    DoFHandler<dim>&            m_dof_handler;
//    Quadrature<dim>&            m_quadrature_formula;
//    Quadrature<dim-1>&          m_face_quadrature_formula;
//    const unsigned int          m_dofs_per_cell;
//    const unsigned int          m_n_q_points;
//    const unsigned int          m_faces_per_cell;
//    const unsigned int          m_n_face_q_points;
//    std::vector<unsigned int>&  m_local_dof_indices;
//    FEValues<dim>&              m_fe_values;
//    FEFaceValues<dim>&          m_fe_face_values;
//    map_String_FunctionPtr&     m_mapFunctions;
//    unsigned int                m_i;
//    int                         m_j;
//    int                         m_q;
//};

///******************************************************************
//    dealiiFiniteElementEquation<dim>
//*******************************************************************/
//template <int dim>
//class dealiiFiniteElementEquation
//{
//public:
//    dealiiFiniteElementEquation()
//    {
//    }

//public:
//    bool                m_bNeedsUpdate;
//    feExpression<dim>   m_matrix;
//    feExpression<dim>   m_matrix_dt;
//    feExpression<dim>   m_rhs;
//};

///******************************************************************
//    dealiiFiniteElementObject<dim>
//*******************************************************************/
//template <int dim>
//class dealiiFiniteElementObject : public daeFiniteElementObject
//{
//typedef typename std::map<unsigned int, const Function<dim>*> map_Uint_FunctionPtr;
//typedef typename std::map<std::string,  const Function<dim>*> map_String_FunctionPtr;

//public:
//    dealiiFiniteElementObject();
//    virtual ~dealiiFiniteElementObject();
    
//public:
//    virtual void AssembleSystem();
//    virtual bool NeedsReAssembling();
//    virtual void ReAssembleSystem();

//    virtual daeSparseMatrixRowIterator*  RowIterator(unsigned int row) const;
//    virtual dae::daeMatrix<real_t>*      SystemMatrix() const;
//    virtual dae::daeMatrix<real_t>*      SystemMatrix_dt() const;
//    virtual dae::daeArray<real_t>*       SystemRHS() const;
//    virtual std::vector<std::string>     GetVariableNames() const;
//    virtual unsigned int                 GetNumberOfPointsInDomainOmega() const;
//    std::vector<unsigned int>            GetDOFtoBoundaryMap();
//    virtual daeDealIIDataReporter*       CreateDataReporter();

//    void Initialize(const std::string&                      meshFilename,
//                    unsigned int                            polynomialOrder,
//                    const Quadrature<dim>&                  quadrature,
//                    const Quadrature<dim-1>&                faceQuadrature,
//                    const map_String_FunctionPtr&           functions,
//                    const map_Uint_FunctionPtr&             dirichletBC,
//                    const map_Uint_FunctionPtr&             neumannBC,
//                    const dealiiFiniteElementEquation<dim>& equation);

//protected:
//    void setup_system();
//    void assemble_system();
//    void process_solution(const std::string& strFilename, const std::string& strVariableName, double* values, unsigned int n);

//public:
//    // Additional deal.II specific data
//    Triangulation<dim>                   triangulation;
//    DoFHandler<dim>                      dof_handler;
//    SmartPointer< FiniteElement<dim> >   fe;
//    ConstraintMatrix                     hanging_node_constraints;

//    SparsityPattern        sparsity_pattern;
//    SparseMatrix<double>   system_matrix;
//    SparseMatrix<double>   system_matrix_dt;
//    Vector<double>         system_rhs;
//    Vector<double>         solution;

//    SmartPointer< Quadrature<dim>   >  m_quadrature_formula;
//    SmartPointer< Quadrature<dim-1> >  m_face_quadrature_formula;

//    // Model-specific data
//    map_String_FunctionPtr  funsFunctions;
//    map_Uint_FunctionPtr    funsDirichletBC;
//    map_Uint_FunctionPtr    funsNeumannBC;

//    dealiiFiniteElementEquation<dim> m_equation;
//};

//template <int dim>
//dealiiFiniteElementObject<dim>::dealiiFiniteElementObject():
//    dof_handler (triangulation)
//{
//}

//template <int dim>
//void dealiiFiniteElementObject<dim>::Initialize(const std::string&                      meshFilename,
//                                                unsigned int                            polynomialOrder,
//                                                const Quadrature<dim>&                  quadrature,
//                                                const Quadrature<dim-1>&                faceQuadrature,
//                                                const map_String_FunctionPtr&           functions,
//                                                const map_Uint_FunctionPtr&             dirichletBC,
//                                                const map_Uint_FunctionPtr&             neumannBC,
//                                                const dealiiFiniteElementEquation<dim>& equation)
//{
//    fe = SmartPointer< FiniteElement<dim> >(new FE_Q<dim>(polynomialOrder));

//    funsFunctions   = functions;
//    funsDirichletBC = dirichletBC;
//    funsNeumannBC   = neumannBC;
//    m_equation      = equation;

//    GridIn<dim> gridin;
//    gridin.attach_triangulation(triangulation);

//    std::ifstream f(meshFilename);
//    std::string extension = boost::filesystem::path(meshFilename).extension().c_str();
//    size_t iFound = extension.find('.');
//    if(iFound != std::string::npos)
//        extension.erase(iFound, 1);

//    typename GridIn<dim>::Format format = gridin.parse_format(extension);
//    gridin.read(f, format);

//    m_quadrature_formula      = SmartPointer< Quadrature<dim>   >(new Quadrature<dim>  (quadrature));
//    m_face_quadrature_formula = SmartPointer< Quadrature<dim-1> >(new Quadrature<dim-1>(faceQuadrature));

//    dealiiFiniteElementObject<dim>::setup_system();
//}

//template <int dim>
//dealiiFiniteElementObject<dim>::~dealiiFiniteElementObject ()
//{
//    dof_handler.clear ();
//}

//template <int dim>
//void dealiiFiniteElementObject<dim>::setup_system()
//{
//    dof_handler.distribute_dofs (*fe);
//    DoFRenumbering::Cuthill_McKee (dof_handler);

//    hanging_node_constraints.clear ();
//    DoFTools::make_hanging_node_constraints (dof_handler,
//                                             hanging_node_constraints);
//    hanging_node_constraints.close ();

//    sparsity_pattern.reinit (dof_handler.n_dofs(),
//                             dof_handler.n_dofs(),
//                             dof_handler.max_couplings_between_dofs());
//    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
//    hanging_node_constraints.condense (sparsity_pattern);
//    sparsity_pattern.compress();

//    system_matrix.reinit (sparsity_pattern);
//    system_matrix_dt.reinit (sparsity_pattern);
//    solution.reinit (dof_handler.n_dofs());
//    system_rhs.reinit (dof_handler.n_dofs());

///* Old code
//    dof_handler.distribute_dofs (*fe);
    
//    SparsityPattern sparsity_pattern_pre;
//    sparsity_pattern_pre.reinit (dof_handler.n_dofs(),
//                                 dof_handler.n_dofs(),
//                                 dof_handler.max_couplings_between_dofs());
//    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern_pre);
//    hanging_node_constraints.condense (sparsity_pattern_pre);
//    sparsity_pattern_pre.compress();

//    std::ofstream out1 ("sparsity_pattern_pre");
//    sparsity_pattern_pre.print_gnuplot (out1);

//    DoFRenumbering::Cuthill_McKee (dof_handler);
    
//    hanging_node_constraints.clear ();
//    DoFTools::make_hanging_node_constraints (dof_handler,
//                                             hanging_node_constraints);
//    hanging_node_constraints.close ();
    
//    sparsity_pattern.reinit (dof_handler.n_dofs(),
//                             dof_handler.n_dofs(),
//                             dof_handler.max_couplings_between_dofs());
//    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
//    hanging_node_constraints.condense (sparsity_pattern);
//    sparsity_pattern.compress();

//    // Global matrices
//    system_matrix.reinit (sparsity_pattern);
//    system_matrix_dt.reinit (sparsity_pattern);
//    system_rhs.reinit (dof_handler.n_dofs());
//    solution.reinit(dof_handler.n_dofs());

//    std::ofstream out ("sparsity_pattern_after");
//    sparsity_pattern.print_gnuplot (out);
//*/
//}

//template <int dim>
//bool dealiiFiniteElementObject<dim>::NeedsReAssembling()
//{
//    return false;
//}

//template <int dim>
//void dealiiFiniteElementObject<dim>::ReAssembleSystem()
//{
//    this->assemble_system();
//}

//template <int dim>
//void dealiiFiniteElementObject<dim>::AssembleSystem()
//{
//    this->assemble_system();
//}

//template <int dim>
//void dealiiFiniteElementObject<dim>::assemble_system()
//{
//    Quadrature<dim>&   quadrature_formula      = *m_quadrature_formula;
//    Quadrature<dim-1>& face_quadrature_formula = *m_face_quadrature_formula;

//    const unsigned int n_q_points      = quadrature_formula.size();
//    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
//    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    
//    FullMatrix<double>  cell_matrix(dofs_per_cell, dofs_per_cell);
//    FullMatrix<double>  cell_matrix_dt(dofs_per_cell, dofs_per_cell);
//    Vector<double>      cell_rhs(dofs_per_cell);

//    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    
//    FEValues<dim>  fe_values (*fe, quadrature_formula,
//                              update_values   | update_gradients |
//                              update_quadrature_points | update_JxW_values);
    
//    FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
//                                      update_values         | update_quadrature_points  |
//                                      update_normal_vectors | update_JxW_values);

//    std::vector<types::global_dof_index> mapGlobalDOFtoBoundary;
//    DoFTools::map_dof_to_boundary_indices(dof_handler, mapGlobalDOFtoBoundary);

//    if(funsFunctions.find("Diffusivity") == funsFunctions.end() || !funsFunctions.find("Diffusivity")->second)
//        throw std::runtime_error("Invalid function: Diffusivity");
//    if(funsFunctions.find("Generation") == funsFunctions.end() || !funsFunctions.find("Generation")->second)
//        throw std::runtime_error("Invalid function: Generation");
//    if(funsFunctions.find("Velocity") == funsFunctions.end() || !funsFunctions.find("Velocity")->second)
//        throw std::runtime_error("Invalid function: Velocity");

//    const Function<dim>* funDiffusivity = funsFunctions.find("Diffusivity")->second;
//    const Function<dim>* funGeneration  = funsFunctions.find("Generation")->second;
//    const Function<dim>* funVelocity    = funsFunctions.find("Velocity")->second;

//    feCellContextImpl<dim> cellContext(fe,
//                                       dof_handler,
//                                       quadrature_formula,
//                                       face_quadrature_formula,
//                                       dofs_per_cell,
//                                       n_q_points,
//                                       GeometryInfo<dim>::faces_per_cell,
//                                       n_face_q_points,
//                                       local_dof_indices,
//                                       fe_values,
//                                       fe_face_values,
//                                       funsFunctions);

//    std::cout << "1" << std::endl;

//    // All DOFs at the boundary ID that have Dirichlet BCs imposed.
//    // mapDirichlets: map< boundary_id, map<dof, value> > will be used to apply boundary conditions locally.
//    // We build this map here since it is common for all cells.
//    /*
//    std::map< unsigned int, std::map<unsigned int, double> > mapDirichlets;
//    for(typename std::map< unsigned int, dealiiFunction<dim> >::const_iterator it = funsDirichletBC.begin(); it != funsDirichletBC.end(); it++)
//    {
//        const unsigned int id          = it->first;
//        const dealiiFunction<dim>& fun = it->second;

//        std::map<unsigned int,double> boundary_values;
//        VectorTools::interpolate_boundary_values (dof_handler,
//                                                  id,
//                                                  fun,
//                                                  boundary_values);

//        mapDirichlets[id] = boundary_values;
//    }
//    */

//    /*
//    // All DOFs at the boundary ID that have Neumann BCs imposed
//    // mapNeumanns: map< boundary_id, map<global_dof_index, value> >
//    std::map< unsigned int, std::map<unsigned int,double> > mapNeumanns;
//    for(std::map<unsigned int, double>::iterator it = NeumanBC.begin(); it != NeumanBC.end(); it++)
//    {
//        const unsigned int id      = it->first;
//        const double neumann_value = it->second;

//        std::map<unsigned int,double> boundary_values;
//        VectorTools::interpolate_boundary_values (dof_handler,
//                                                  id,
//                                                  SingleValue_Function<dim>(neumann_value),
//                                                  boundary_values);

//        mapNeumanns[id] = boundary_values;
//    }
//    */

//    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
//                                                   endc = dof_handler.end();
//    for(int cellCounter = 0; cell != endc; ++cell, ++cellCounter)
//    {
//        cell_matrix    = 0;
//        cell_matrix_dt = 0;
//        cell_rhs       = 0;
        
//        fe_values.reinit(cell);
//        cell->get_dof_indices(local_dof_indices);

//        for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
//        {
//            cellContext.m_q = q_point;

//            for(unsigned int i = 0; i < dofs_per_cell; ++i)
//            {
//                cellContext.m_i = i;

//                for(unsigned int j = 0; j < dofs_per_cell; ++j)
//                {
//                    cellContext.m_j = j;

//                    std::cout << "1a" << std::endl;
//                    if(m_equation.m_matrix.m_node)
//                    {
//                        feRuntimeNumber<dim> result = m_equation.m_matrix.m_node->Evaluate(&cellContext);
//                        if(result.m_eType != eFEScalar)
//                            throw std::runtime_error("Invalid local matrix contribution expresion specified: it must be a scalar value");

//                        cell_matrix(i,j) += result.m_value;
//                        std::cout << "1aa" << std::endl;
//                    }
//                    else
//                    {
//                        cell_matrix(i,j)    += (
//                                                 /* Diffusion term */
//                                                    (
//                                                       fe_values.shape_grad(i, q_point) *
//                                                       fe_values.shape_grad(j, q_point)
//                                                    )
//                                                    *
//                                                    funDiffusivity->value(fe_values.quadrature_point(q_point))

//                                                 /* Convection term */
//                                                    // +
//                                                    //fe_values.shape_value(i, q_point) *
//                                                    //fe_values.shape_value(j, q_point) *
//                                                    //funVelocity.value(fe_values.quadrature_point(q_point))
//                                               )
//                                               *
//                                               fe_values.JxW(q_point);
//                    }

//                    std::cout << "1b" << std::endl;
//                    /* Accumulation term (in a separate matrix) */
//                    if(m_equation.m_matrix_dt.m_node)
//                    {
//                        feRuntimeNumber<dim> result = m_equation.m_matrix_dt.m_node->Evaluate(&cellContext);
//                        if(result.m_eType != eFEScalar)
//                            throw std::runtime_error("Invalid local matrix_dt contribution expresion specified: it must be a scalar value");

//                        cell_matrix_dt(i,j) += result.m_value;
//                    }
//                    else
//                    {
//                        cell_matrix_dt(i,j) += (
//                                                  fe_values.shape_value(i, q_point) *
//                                                  fe_values.shape_value(j, q_point)
//                                               )
//                                               *
//                                               fe_values.JxW(q_point);
//                    }
//                }
                
//                std::cout << "1c" << std::endl;
//                /* Generation */
//                if(m_equation.m_rhs.m_node)
//                {
//                    feRuntimeNumber<dim> result = m_equation.m_rhs.m_node->Evaluate(&cellContext);
//                    if(result.m_eType != eFEScalar)
//                        throw std::runtime_error("Invalid local RHS contribution expresion specified: it must be a scalar value");
//                    cell_rhs(i) += result.m_value;
//                }
//                else
//                {
//                    cell_rhs(i) +=  fe_values.shape_value(i,q_point) *
//                                    funGeneration->value(fe_values.quadrature_point(q_point), 0) *
//                                    fe_values.JxW(q_point);
//                }
//            }
//        }

//        for(unsigned int face = 0; face<GeometryInfo<dim>::faces_per_cell; ++face)
//        {
//            if(cell->face(face)->at_boundary())
//            {
//                fe_face_values.reinit (cell, face);

//                const unsigned int id = cell->face(face)->boundary_indicator();

//                if(funsDirichletBC.find(id) != funsDirichletBC.end())
//                {
//                    // Dirichlet BC
//                    // Do nothing now; apply boundary conditions on the final system_matrix and system_rhs only at the end of the assembling
//                }
//                else if(funsNeumannBC.find(id) != funsNeumannBC.end())
//                {
//                    // Neumann BC
//                    const Function<dim>& neumann = *funsNeumannBC.find(id)->second;

//                    //std::cout << (boost::format("  Setting NeumanBC (cell=%d, face=%d, id= %d)[q0] = %f") % cellCounter % face % id % neumann.value(fe_face_values.quadrature_point(0))).str() << std::endl;

//                    for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
//                    {
//                        // Achtung, Achtung! For the Convection-Diffusion-Reaction system only:
//                        //                   the sign '-neumann' since we have the term: -integral(q * φ(i) * dΓq)
//                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
//                            cell_rhs(i) += -neumann.value(fe_face_values.quadrature_point(q_point))
//                                           *
//                                           fe_face_values.shape_value(i, q_point)
//                                           *
//                                           fe_face_values.JxW(q_point);
//                    }
//                }
//                else
//                {
//                    // Not found
//                    // Do nothing or do some default action (perhaps zero gradient?)
//                }
//            }
//        }

//        types::global_dof_index id;
//        for(unsigned int i = 0; i < dofs_per_cell; ++i)
//        {
//            id = mapGlobalDOFtoBoundary[ local_dof_indices[i] ];
//            if(id != numbers::invalid_dof_index)
//            {
//                // This dof IS on one of boundaries; therefore, remove its contributions to the cell_matrix_dt

//                // 1. Reset the whole row 'i'
//                for(unsigned int j = 0; j < dofs_per_cell; ++j)
//                    cell_matrix_dt(i, j) = 0;

//                // 2. Reset the whole column 'i'
//                for(unsigned int j = 0; j < dofs_per_cell; ++j)
//                    cell_matrix_dt(j, i) = 0;
//            }
//        }

//        /* ACHTUNG, ACHTUNG!!
//           Apply Dirichlet boundary conditions locally (conflicts with refined grids with hanging nodes!!)

//        // We already have a pre-calculated map<global_dof_index, bc_value> for every ID marked as having Dirichlet BCs imposed
//        for(std::map< unsigned int, std::map<unsigned int, double> >::iterator it = mapDirichlets.begin(); it != mapDirichlets.end(); it++)
//        {
//            unsigned int id                                 = it->first;
//            std::map<unsigned int, double>& boundary_values = it->second;

//            // Print some mumbo-jumbo voodoo-mojo stuf related to cell_matrix and cell_rhs...
//            std::cout << "boundary_values" << std::endl;
//            for(std::map<unsigned int,double>::iterator bviter = boundary_values.begin(); bviter != boundary_values.end(); bviter++)
//                std::cout << "(" << bviter->first << ", " << bviter->second << ") ";
//            std::cout << std::endl;

//            std::cout << "local_dof_indices" << std::endl;
//            for(std::vector<unsigned int>::iterator ldiiter = local_dof_indices.begin(); ldiiter != local_dof_indices.end(); ldiiter++)
//                std::cout << *ldiiter << " ";
//            std::cout << std::endl;

//            std::cout << "cell_matrix before bc:" << std::endl;
//            cell_matrix.print_formatted(std::cout);
//            std::cout << "cell_rhs before bc:" << std::endl;
//            cell_rhs.print(std::cout);

//            // Apply values to the cell_matrix and cell_rhs
//            MatrixTools::local_apply_boundary_values(boundary_values,
//                                                     local_dof_indices,
//                                                     cell_matrix,
//                                                     cell_rhs,
//                                                     true);

//            std::cout << "cell_matrix after bc:" << std::endl;
//            cell_matrix.print_formatted(std::cout);
//            std::cout << "cell_rhs after bc:" << std::endl;
//            cell_rhs.print(std::cout);
//        }
//        */

//        // Add to the system matrices/vector
//        for(unsigned int i = 0; i < dofs_per_cell; ++i)
//        {
//            for(unsigned int j = 0; j < dofs_per_cell; ++j)
//            {
//                system_matrix.add   (local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));
//                system_matrix_dt.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_dt(i,j));
//            }
//            system_rhs(local_dof_indices[i]) += cell_rhs(i);
//        }
//    }

//    // If using refined grids condense hanging nodes
//    hanging_node_constraints.condense(system_matrix);
//    hanging_node_constraints.condense(system_rhs);

//    // What about this matrix? Should it also be condensed?
//    //hanging_node_constraints.condense(system_matrix_dt);

//    // Apply Dirichlet boundary conditions on the system matrix and rhs
//    for(typename map_Uint_FunctionPtr::const_iterator it = funsDirichletBC.begin(); it != funsDirichletBC.end(); it++)
//    {
//        const unsigned int id    =  it->first;
//        const Function<dim>& fun = *it->second;

//        //std::cout << "Setting DirichletBC at id " << id << " with sample value " << fun.value(Point<dim>(0,0,0)) << std::endl;

//        std::map<types::global_dof_index, double> boundary_values;
//        VectorTools::interpolate_boundary_values (dof_handler,
//                                                  id,
//                                                  fun,
//                                                  boundary_values);
//        MatrixTools::apply_boundary_values (boundary_values,
//                                            system_matrix,
//                                            solution,
//                                            system_rhs);
//    }
//}

//template <int dim>
//void dealiiFiniteElementObject<dim>::process_solution(const std::string& strFilename, const std::string& strVariableName, double* values, unsigned int n)
//{
//    for(size_t i = 0; i < n; i++)
//        solution[i] = values[i];

//    //std::cout << "solution after solve:" << solution << std::endl;

//    // We may call distribute() on solution to fix hanging nodes
//    hanging_node_constraints.distribute(solution);

//    //std::cout << "solution after distribute:" << solution << std::endl;

//    DataOut<dim> data_out;
//    data_out.attach_dof_handler(dof_handler);
//    data_out.add_data_vector(solution, strVariableName);
//    data_out.build_patches(fe->degree);
//    std::ofstream output(strFilename);
//    data_out.write_vtk(output);

//    /*
//        Refinement:

//    if(true)
//    {
//        Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

//        typename FunctionMap<dim>::type neumann_boundary;
//        KellyErrorEstimator<dim>::estimate (dof_handler,
//                                            QGauss<dim-1>(3),
//                                            neumann_boundary,
//                                            solution,
//                                            estimated_error_per_cell);

//        GridRefinement::refine_and_coarsen_fixed_number (triangulation,
//                                                         estimated_error_per_cell,
//                                                         0.4, 0.1);

//        triangulation.execute_coarsening_and_refinement ();

//        GridOut grid;
//        std::ofstream fo("refined.msh");
//        GridOutFlags::Msh flags(true, true);
//        grid.set_flags(flags);
//        grid.write_msh(triangulation, fo);
//    }
//    */
//}

//template <int dim>
//std::vector<std::string> dealiiFiniteElementObject<dim>::GetVariableNames() const
//{
//    std::vector<std::string> strarrNames;
//    strarrNames.push_back("T");
//    return strarrNames;
//}

//template <int dim>
//unsigned int dealiiFiniteElementObject<dim>::GetNumberOfPointsInDomainOmega() const
//{
//    // For a scalar variable it is ok
//    return dof_handler.n_dofs();
//}

//template <int dim>
//daeSparseMatrixRowIterator* dealiiFiniteElementObject<dim>::RowIterator(unsigned int row) const
//{
//    return new dealiiSparsityPatternIterator(sparsity_pattern.begin(row), sparsity_pattern.end(row));
//}

//template <int dim>
//dae::daeMatrix<double>* dealiiFiniteElementObject<dim>::SystemMatrix() const
//{
//    return new daeFEMatrix<double>(system_matrix);
//}

//template <int dim>
//dae::daeMatrix<double>* dealiiFiniteElementObject<dim>::SystemMatrix_dt() const
//{
//    return new daeFEMatrix<double>(system_matrix_dt);
//}

//template <int dim>
//dae::daeArray<double>* dealiiFiniteElementObject<dim>::SystemRHS() const
//{
//    return new daeFEArray<double>(system_rhs);
//}

//template <int dim>
//std::vector<unsigned int> dealiiFiniteElementObject<dim>::GetDOFtoBoundaryMap()
//{
//    std::vector<types::global_dof_index> mapGlobalDOFtoBoundary;
//    DoFTools::map_dof_to_boundary_indices(dof_handler, mapGlobalDOFtoBoundary);
//    return mapGlobalDOFtoBoundary;
//}

//template <int dim>
//daeDealIIDataReporter* dealiiFiniteElementObject<dim>::CreateDataReporter()
//{
//    fnProcessSolution callback(boost::bind(&dealiiFiniteElementObject<dim>::process_solution, this, _1, _2, _3, _4));
//    return new daeDealIIDataReporter(callback);
//}

////    constant   = constant_2D
////    phi        = phi_2D
////    dphi       = dphi_2D
////    JxW        = JxW_2D
////    xyz        = xyz_2D
////    normal     = normal_2D
////    fvalue     = function_value_2D
////    grad       = function_gradient_2D
//template <int dim>
//dealiiFiniteElementEquation<dim> CreateEquation_ConvectionDiffusion()
//{
//    dealiiFiniteElementEquation<dim> equation;
//    equation.m_matrix    = (dphi<dim>(fe_i, fe_q) * dphi<dim>(fe_j, fe_q)) * function_value<dim>("Diffusivity", xyz<dim>(fe_q)) * JxW<dim>(fe_q);
//    equation.m_matrix_dt = phi<dim>(fe_i, fe_q) * phi<dim>(fe_j, fe_q) * JxW<dim>(fe_q);
//    equation.m_rhs       = phi<dim>(fe_i, fe_q) * function_value<dim>("Generation", xyz<dim>(fe_q)) * JxW<dim>(fe_q);
//    return equation;
//}

//template <int dim>
//dealiiFiniteElementEquation<dim> CreateEquation_Poisson()
//{
//    dealiiFiniteElementEquation<dim> equation;
//    return equation;
//}

//template <int dim>
//dealiiFiniteElementEquation<dim> CreateEquation_Laplace()
//{
//    dealiiFiniteElementEquation<dim> equation;
//    return equation;
//}



}
}

#endif
