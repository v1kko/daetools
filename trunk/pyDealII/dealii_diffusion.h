#include <typeinfo>
#include <fstream>
#include <iostream>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector.templates.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/full_matrix.templates.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/solver_cg.h>
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
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

namespace diffusion
{
using namespace dealii;

template <int dim>
class SolutionBase
{
protected:
    static const unsigned int n_source_centers = 3;
    static const Point<dim>   source_centers[n_source_centers];
    static const double       width;
};


template <int dim>
const Point<dim> SolutionBase<dim>::source_centers[SolutionBase<dim>::n_source_centers]
= { Point<dim>(-0.5, +0.5),
    Point<dim>(-0.5, -0.5),
    Point<dim>(+0.5, -0.5)   };

template <int dim>
const double SolutionBase<dim>::width = 1./3.;



template <int dim>
class Solution : public Function<dim>,
        protected SolutionBase<dim>
{
public:
    Solution () : Function<dim>() {}
    
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    
    virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                    const unsigned int  component = 0) const;
};


template <int dim>
double Solution<dim>::value (const Point<dim>   &p,
                             const unsigned int) const
{
    double return_value = 0;
    for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
        const Point<dim> x_minus_xi = p - this->source_centers[i];
        return_value += std::exp(-x_minus_xi.square() /
                                 (this->width * this->width));
    }
    
    return return_value;
}


template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
    Tensor<1,dim> return_value;
    
    for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
        const Point<dim> x_minus_xi = p - this->source_centers[i];
        
        return_value += (-2 / (this->width * this->width) *
                         std::exp(-x_minus_xi.square() /
                                  (this->width * this->width)) *
                         x_minus_xi);
    }
    
    return return_value;
}



template <int dim>
class RightHandSide : public Function<dim>,
        protected SolutionBase<dim>
{
public:
    RightHandSide () : Function<dim>() {}
    
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};


template <int dim>
double RightHandSide<dim>::value (const Point<dim>   &p,
                                  const unsigned int) const
{
    double return_value = 0;
    for (unsigned int i=0; i<this->n_source_centers; ++i)
    {
        const Point<dim> x_minus_xi = p - this->source_centers[i];
        
        return_value += ((2*dim - 4*x_minus_xi.square()/
                          (this->width * this->width)) /
                         (this->width * this->width) *
                         std::exp(-x_minus_xi.square() /
                                  (this->width * this->width)));
        return_value += std::exp(-x_minus_xi.square() /
                                 (this->width * this->width));
    }
    
    return return_value;
}


enum RefinementMode 
{
    global_refinement, 
    adaptive_refinement
};


template <int dim>
class dealiiDiffusion
{
public:
    
    dealiiDiffusion (double diffusivity,
                                 unsigned int polynomialOrder,
                                 const std::map<int, double>& dirichletBC,
                                 const std::map<int, double>& neumanBC);
    
    virtual ~dealiiDiffusion ();
    
    void run ();
    
public:
    virtual void setup_system ();
    virtual void assemble_system ();
    virtual void solve ();
    virtual void refine_grid ();
    virtual void process_solution (const unsigned int cycle);
    
    Triangulation<dim>                      triangulation;
    DoFHandler<dim>                         dof_handler;
    
    SmartPointer<FiniteElement<dim> >       fe;
    
    ConstraintMatrix                        hanging_node_constraints;
    
    SparsityPattern                         sparsity_pattern;
  
    SparseMatrix<double>                    system_matrix;
    SparseMatrix<double>                    system_matrix_dt;
    Vector<double>                          solution;
    Vector<double>                          system_rhs;
    
    const RefinementMode                    refinement_mode;
    
    ConvergenceTable                        convergence_table;

    double                                  Diffusivity;
    std::map<int, double>                   DirichletBC;
    std::map<int, double>                   NeumanBC;
};




template <int dim>
dealiiDiffusion<dim>::dealiiDiffusion (double diffusivity,
                                                               unsigned int polynomialOrder,
                                                               const std::map<int, double>& dirichletBC,
                                                               const std::map<int, double>& neumanBC):
    dof_handler (triangulation),
    fe (new FE_Q<dim>(polynomialOrder)),
    refinement_mode (global_refinement)
{
    Diffusivity = diffusivity;
    DirichletBC = dirichletBC;
    NeumanBC    = neumanBC;
}



template <int dim>
dealiiDiffusion<dim>::~dealiiDiffusion ()
{
    dof_handler.clear ();
}



template <int dim>
void dealiiDiffusion<dim>::setup_system ()
{
    dof_handler.distribute_dofs (*fe);
    
    SparsityPattern sparsity_pattern_pre;
    sparsity_pattern_pre.reinit (dof_handler.n_dofs(),
                             dof_handler.n_dofs(),
                             dof_handler.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern_pre);
    hanging_node_constraints.condense (sparsity_pattern_pre);
    sparsity_pattern_pre.compress();

    std::ofstream out1 ("sparsity_pattern_pre");
    sparsity_pattern_pre.print_gnuplot (out1);

    DoFRenumbering::Cuthill_McKee (dof_handler);
    
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             hanging_node_constraints);
    hanging_node_constraints.close ();
    
    sparsity_pattern.reinit (dof_handler.n_dofs(),
                             dof_handler.n_dofs(),
                             dof_handler.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    hanging_node_constraints.condense (sparsity_pattern);
    sparsity_pattern.compress();
    
    system_matrix.reinit (sparsity_pattern);
    system_matrix_dt.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    
    std::ofstream out ("sparsity_pattern_after");
    sparsity_pattern.print_gnuplot (out);
}

template <int dim>
void dealiiDiffusion<dim>::assemble_system ()
{
    QGauss<dim>   quadrature_formula(3);
    QGauss<dim-1> face_quadrature_formula(3);
    
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    
    FullMatrix<double>  cell_matrix    (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>  cell_matrix_dt (dofs_per_cell, dofs_per_cell);
    Vector<double>      cell_rhs       (dofs_per_cell);
    
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    
    FEValues<dim>  fe_values (*fe, quadrature_formula,
                              update_values   | update_gradients |
                              update_quadrature_points | update_JxW_values);
    
    FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);
    
    const RightHandSide<dim> right_hand_side;
    std::vector<double>  rhs_values (n_q_points);
    
    const Solution<dim> exact_solution;
    
    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        cell_matrix    = 0;
        cell_matrix_dt = 0;
        cell_rhs       = 0;
        
        fe_values.reinit (cell);
        
        right_hand_side.value_list (fe_values.get_quadrature_points(),
                                    rhs_values);
        
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    cell_matrix(i,j)    += (
                                             fe_values.shape_grad(i,q_point) *
                                             fe_values.shape_grad(j,q_point)
                                           )
                                           * Diffusivity
                                           * fe_values.JxW(q_point);

                    cell_matrix_dt(i,j) += (
                                             fe_values.shape_value(i,q_point) *
                                             fe_values.shape_value(j,q_point)
                                           )
                                           * fe_values.JxW(q_point);
                }
                
                // No generation in the interior elements
                //cell_rhs(i) +=  fe_values.shape_value(i,q_point) *
                //                rhs_values [q_point] *
                //                fe_values.JxW(q_point);
            }
        }

        for (unsigned int face = 0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if(cell->face(face)->at_boundary())
            {
                for(std::map<int, double>::iterator it = NeumanBC.begin(); it != NeumanBC.end(); it++)
                {
                    fe_face_values.reinit (cell, face);

                    if(cell->face(face)->boundary_indicator() == it->first)
                    {
                        std::cout << "    face " << face << ", id = " << it->first << ", NeumanBC = " << it->second << std::endl;

                        for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                        {
                            const double neumann_value = it->first;
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                cell_rhs(i) += neumann_value *
                                               fe_face_values.shape_value(i, q_point) *
                                               fe_face_values.JxW(q_point);
                        }
                    }
                }
            }
        }
        
        cell->get_dof_indices (local_dof_indices);
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            { 
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i,j));

                system_matrix_dt.add(local_dof_indices[i],
                                     local_dof_indices[j],
                                     cell_matrix_dt(i,j));
            }
            
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
    
    hanging_node_constraints.condense(system_matrix);
    hanging_node_constraints.condense(system_matrix_dt);
    hanging_node_constraints.condense(system_rhs);
    
//    std::map<unsigned int,double> boundary_values;
//    VectorTools::interpolate_boundary_values (dof_handler,
//                                              0,
//                                              Solution<dim>(),
//                                              boundary_values);
//    MatrixTools::apply_boundary_values (boundary_values,
//                                        system_matrix,
//                                        solution,
//                                        system_rhs);
}



template <int dim>
void dealiiDiffusion<dim>::solve ()
{
    SolverControl           solver_control (1000, 1e-12);
    SolverCG<>              cg (solver_control);
    
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    
    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);
    
    hanging_node_constraints.distribute (solution);
}



template <int dim>
void dealiiDiffusion<dim>::refine_grid ()
{
    switch (refinement_mode)
    {
    case global_refinement:
    {
        triangulation.refine_global (1);
        break;
    }
        
    case adaptive_refinement:
    {
        Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
        
        typename FunctionMap<dim>::type neumann_boundary;
        KellyErrorEstimator<dim>::estimate (dof_handler,
                                            QGauss<dim-1>(3),
                                            neumann_boundary,
                                            solution,
                                            estimated_error_per_cell);
        
        GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                         estimated_error_per_cell,
                                                         0.3, 0.03);
        
        triangulation.execute_coarsening_and_refinement ();
        
        break;
    }
        
    default:
    {
        Assert (false, ExcNotImplemented());
    }
    }
}



template <int dim>
void dealiiDiffusion<dim>::process_solution (const unsigned int cycle)
{
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(3),
                                       VectorTools::L2_norm);
    const double L2_error = difference_per_cell.l2_norm();
    
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(3),
                                       VectorTools::H1_seminorm);
    const double H1_error = difference_per_cell.l2_norm();
    
    const QTrapez<1>     q_trapez;
    const QIterated<dim> q_iterated (q_trapez, 5);
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       q_iterated,
                                       VectorTools::Linfty_norm);
    const double Linfty_error = difference_per_cell.linfty_norm();
    
    const unsigned int n_active_cells=triangulation.n_active_cells();
    const unsigned int n_dofs=dof_handler.n_dofs();
    
    std::cout << "Cycle " << cycle << ':'
              << std::endl
              << "   Number of active cells:       "
              << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: "
              << n_dofs
              << std::endl;
    
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("Linfty", Linfty_error);
    
    for(size_t i = 0; i < solution.size(); i++)
        std::cout << (boost::format("T[%1%] = %2%") % i % solution[i]).str() << std::endl;
}


template <int dim>
void dealiiDiffusion<dim>::run ()
{
    unsigned int cycle;
    for (cycle=0; cycle<5; ++cycle)
    {
        if (cycle == 0)
        {
            GridGenerator::hyper_cube (triangulation, -1, 1);
            triangulation.refine_global (1);

            typename Triangulation<dim>::cell_iterator
                    cell = triangulation.begin (),
                    endc = triangulation.end();
            for (; cell!=endc; ++cell)
                for (unsigned int face=0;
                     face<GeometryInfo<dim>::faces_per_cell;
                     ++face)
                    if ((std::fabs(cell->face(face)->center()(0) - (-1)) < 1e-12)
                            ||
                            (std::fabs(cell->face(face)->center()(1) - (-1)) < 1e-12))
                        cell->face(face)->set_boundary_indicator (1);
        }
        else
            refine_grid ();
    }

    setup_system ();

    assemble_system ();
    solve ();

    process_solution (cycle);

    std::string gmv_filename;
    switch (refinement_mode)
    {
    case global_refinement:
        gmv_filename = "solution-global";
        break;
    case adaptive_refinement:
        gmv_filename = "solution-adaptive";
        break;
    default:
        Assert (false, ExcNotImplemented());
    }

    switch (fe->degree)
    {
    case 1:
        gmv_filename += "-q1";
        break;
    case 2:
        gmv_filename += "-q2";
        break;

    default:
        Assert (false, ExcNotImplemented());
    }

    gmv_filename += ".vtk";
    std::ofstream output (gmv_filename.c_str());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");

    data_out.build_patches (fe->degree);
    data_out.write_vtk(output);



    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_precision("Linfty", 3);

    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("Linfty", true);

    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
    convergence_table.set_tex_caption("H1", "@f$H^1@f$-error");
    convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");

    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");

    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    std::string error_filename = "error";
    switch (refinement_mode)
    {
    case global_refinement:
        error_filename += "-global";
        break;
    case adaptive_refinement:
        error_filename += "-adaptive";
        break;
    default:
        Assert (false, ExcNotImplemented());
    }

    switch (fe->degree)
    {
    case 1:
        error_filename += "-q1";
        break;
    case 2:
        error_filename += "-q2";
        break;
    default:
        Assert (false, ExcNotImplemented());
    }

    error_filename += ".tex";
    std::ofstream error_table_file(error_filename.c_str());

    convergence_table.write_tex(error_table_file);



    if (refinement_mode==global_refinement)
    {
        convergence_table.add_column_to_supercolumn("cycle", "n cells");
        convergence_table.add_column_to_supercolumn("cells", "n cells");

        std::vector<std::string> new_order;
        new_order.push_back("n cells");
        new_order.push_back("H1");
        new_order.push_back("L2");
        convergence_table.set_column_order (new_order);

        convergence_table
                .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
        convergence_table
                .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
        convergence_table
                .evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);

        std::cout << std::endl;
        convergence_table.write_text(std::cout);

        std::string conv_filename = "convergence";
        switch (refinement_mode)
        {
        case global_refinement:
            conv_filename += "-global";
            break;
        case adaptive_refinement:
            conv_filename += "-adaptive";
            break;
        default:
            Assert (false, ExcNotImplemented());
        }
        switch (fe->degree)
        {
        case 1:
            conv_filename += "-q1";
            break;
        case 2:
            conv_filename += "-q2";
            break;
        default:
            Assert (false, ExcNotImplemented());
        }
        conv_filename += ".tex";

        std::ofstream table_file(conv_filename.c_str());
        convergence_table.write_tex(table_file);
    }
}
}

namespace diffusion
{
template const double SolutionBase<2>::width;
}
