#include <typeinfo>
#include <fstream>
#include <iostream>

#include <boost/format.hpp>

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
class SingleValue_Function : public Function<dim>
{
public:
    SingleValue_Function(double value = 0.0) : Function<dim>()
    {
        m_value = value;
    }

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const
    {
        return m_value;
    }

public:
    double m_value;
};


template <int dim>
class dealiiDiffusion
{
public:
    
    dealiiDiffusion (double diffusivity,
                     unsigned int polynomialOrder,
                     const std::map<unsigned int, double>& dirichletBC,
                     const std::map<unsigned int, double>& neumannBC);
    
    virtual ~dealiiDiffusion ();
    
public:
    virtual void setup_system ();
    virtual void assemble_system ();

public:
    Triangulation<dim>                      triangulation;
    DoFHandler<dim>                         dof_handler;
    
    SmartPointer<FiniteElement<dim> >       fe;
    
    ConstraintMatrix                        hanging_node_constraints;
    
    SparsityPattern                         sparsity_pattern;
  
    SparseMatrix<double>                    K_diffusion;
    //SparseMatrix<double>                    K_convection;
    //SparseMatrix<double>                    K_generation;
    SparseMatrix<double>                    K_dirichlet;
    SparseMatrix<double>                    K_accumulation;
    Vector<double>                          f_generation;
    Vector<double>                          f_dirichlet;
    Vector<double>                          f_neuman;

    Vector<double>                          solution;

    ConvergenceTable                        convergence_table;

    double                                  Diffusivity;
    std::map<unsigned int, double>          DirichletBC;
    std::map<unsigned int, double>          NeumanBC;

    SingleValue_Function<dim> funDiffusivity;
    SingleValue_Function<dim> funVelocity;
    SingleValue_Function<dim> funGeneration;

    std::map<unsigned int, SingleValue_Function<dim> > funsDirichletBC;
    std::map<unsigned int, SingleValue_Function<dim> > funsNeumannBC;

//    adoubleCSRMatrix     Kdiff;
//    adoubleCSRMatrix     Kdt;
//    std::vector<adouble> Kdt;
};

template <int dim>
dealiiDiffusion<dim>::dealiiDiffusion (double diffusivity,
                                       unsigned int polynomialOrder,
                                       const std::map<unsigned int, double>& dirichletBC,
                                       const std::map<unsigned int, double>& neumannBC):
    dof_handler (triangulation),
    fe (new FE_Q<dim>(polynomialOrder))
{
    Diffusivity        = diffusivity;
    DirichletBC        = dirichletBC;
    NeumanBC           = neumannBC;


    // New style
    funDiffusivity.m_value = diffusivity;
    funVelocity.m_value    = 0.0;
    funGeneration.m_value  = 0.0;

    for(std::map<unsigned int, double>::const_iterator it = dirichletBC.begin(); it != dirichletBC.end(); it++)
    {
        const unsigned int id    = it->first;
        const double       value = it->second;

        funsDirichletBC[id] = SingleValue_Function<dim>(value);
    }

    for(std::map<unsigned int, double>::const_iterator it = neumannBC.begin(); it != neumannBC.end(); it++)
    {
        const unsigned int id    = it->first;
        const double       value = it->second;

        funsNeumannBC[id] = SingleValue_Function<dim>(value);
    }
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

    // Global matrices
    K_diffusion.reinit (sparsity_pattern);
    K_accumulation.reinit (sparsity_pattern);
    K_dirichlet.reinit (sparsity_pattern);
    f_dirichlet.reinit (dof_handler.n_dofs());
    f_neuman.reinit (dof_handler.n_dofs());
    f_generation.reinit (dof_handler.n_dofs());

    // Global rhs arrays
    solution.reinit (dof_handler.n_dofs());
    
    std::ofstream out ("sparsity_pattern_after");
    sparsity_pattern.print_gnuplot (out);

/*
    std::map<size_t, size_t> mapIndexes;
    size_t nrows = system_matrix.m();
    size_t nnz = sparsity_pattern.n_nonzero_elements();

    b.resize(nrows);
    A.Reset(nrows, nnz, CSR_C_STYLE);
    A.ResetCounters();

    for(size_t row = 0; row < nrows; row++)
    {
        mapIndexes.clear();
        for(SparsityPattern::row_iterator iter = sparsity_pattern.row_begin(row); iter != sparsity_pattern.row_end(row); iter++)
            mapIndexes.insert(std::make_pair(mapIndexes.size(), *iter));

        A.AddRow(mapIndexes);
    }
    A.Sort();
    //A.Print(true);
*/
}

template <int dim>
void dealiiDiffusion<dim>::assemble_system ()
{
    QGauss<dim>   quadrature_formula(3);
    QGauss<dim-1> face_quadrature_formula(3);
    
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    
    FullMatrix<double>  Kel_diffusion(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>  Kel_accumulation(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>  Kel_dirichlet(dofs_per_cell, dofs_per_cell);
    Vector<double>      fel_dirichlet(dofs_per_cell);
    Vector<double>      fel_neuman(dofs_per_cell);
    Vector<double>      fel_generation(dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    
    FEValues<dim>  fe_values (*fe, quadrature_formula,
                              update_values   | update_gradients |
                              update_quadrature_points | update_JxW_values);
    
    FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);
    
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();

    // All DOFs at the boundary ID that have Dirichlet BCs imposed
    // mapDirichlets: map< boundary_id, map<dof, value> >
    std::map< unsigned int, std::map<unsigned int,double> > mapDirichlets;
    for(std::map<unsigned int, double>::iterator it = DirichletBC.begin(); it != DirichletBC.end(); it++)
    {
        const unsigned int id        = it->first;
        const double dirichlet_value = it->second;

        std::map<unsigned int,double> boundary_values;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  id,
                                                  SingleValue_Function<dim>(dirichlet_value),
                                                  boundary_values);

        mapDirichlets[id] = boundary_values;
    }

    // All DOFs at the boundary ID that have Neumann BCs imposed
    // mapNeumanns: map< boundary_id, map<global_dof_index, value> >
    std::map< unsigned int, std::map<unsigned int,double> > mapNeumanns;
    for(std::map<unsigned int, double>::iterator it = NeumanBC.begin(); it != NeumanBC.end(); it++)
    {
        const unsigned int id      = it->first;
        const double neumann_value = it->second;

        std::map<unsigned int,double> boundary_values;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  id,
                                                  SingleValue_Function<dim>(neumann_value),
                                                  boundary_values);

        mapNeumanns[id] = boundary_values;
    }

    int cellCounter = 0;
    for (; cell!=endc; ++cell, ++cellCounter)
    {
        Kel_diffusion    = 0;
        Kel_accumulation = 0;
        Kel_dirichlet    = 0;
        fel_neuman       = 0;
        fel_generation   = 0;
        
        fe_values.reinit (cell);

        cell->get_dof_indices (local_dof_indices);

        for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for(unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    Kel_diffusion(i,j)    += (
                                                fe_values.shape_grad(i, q_point) *
                                                fe_values.shape_grad(j, q_point)
                                                +
                                                fe_values.shape_value(i, q_point) *
                                                fe_values.shape_value(j, q_point)
                                             )
                                             /* * Diffusivity */
                                             * fe_values.JxW(q_point);

                    Kel_accumulation(i,j) += (
                                               fe_values.shape_value(i, q_point) *
                                               fe_values.shape_value(j, q_point)
                                             )
                                             * fe_values.JxW(q_point);
                }
                
                // No generation in the interior elements - we have only diffusion
                fel_generation(i) +=  fe_values.shape_value(i,q_point) *
                                      /*rhs_values [q_point] */  1 *
                                      fe_values.JxW(q_point);
            }
        }

        for(unsigned int face = 0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if(cell->face(face)->at_boundary())
            {
                fe_face_values.reinit (cell, face);

                const unsigned int face_dofs_per_cell = fe_face_values.dofs_per_cell;
                std::cout << (boost::format("     cell=%d, face=%d, dofs_per_cell=%d, n_face_q_points=%d") % cellCounter % face % dofs_per_cell % n_face_q_points).str() << std::endl;

                for(std::map<unsigned int, double>::iterator it = NeumanBC.begin(); it != NeumanBC.end(); it++)
                {
                    if(cell->face(face)->boundary_indicator() == it->first)
                    {
                        std::cout << (boost::format("  NeumanBC(cell=%d, face=%d, id= %d) = %f") % cellCounter % face % it->first % it->second).str() << std::endl;

                        for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                        {
                            // Achtung, Achtung!
                            // Note the sign '-' since we have the term: -integral(q * φ(i) * dΓq)
                            const double neumann_value = (it->second);

                            for (unsigned int i = 0; i < face_dofs_per_cell; ++i)
                                fel_neuman(i) += neumann_value *
                                                 fe_face_values.shape_value(i, q_point) *
                                                 fe_face_values.JxW(q_point);
                        }
                    }
                }

                /*
                for(std::map<int, double>::iterator it = DirichletBC.begin(); it != DirichletBC.end(); it++)
                {
                    if(cell->face(face)->boundary_indicator() == it->first)
                    {
                        std::cout << (boost::format("  DirichletBC(cell=%d, face=%d, id= %d) = %f") % cellCounter % face % it->first % it->second).str() << std::endl;

                        for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                        {
                            const double dirichlet_value = it->second;

                            for(unsigned int i = 0; i < face_dofs_per_cell; ++i)
                            {
                                // RHS Dirichlet contribution
                                fel_dirichlet(i) += DirichletBCPenalty * dirichlet_value *
                                                    fe_face_values.shape_value(i, q_point) *
                                                    fe_face_values.JxW(q_point);

                                // LHS Dirichlet contribution
                                for(unsigned int j = 0; j < face_dofs_per_cell; ++j)
                                    Kel_dirichlet(i,j) += DirichletBCPenalty *
                                                          fe_face_values.shape_value(i, q_point) *
                                                          fe_face_values.shape_value(j, q_point) *
                                                          fe_face_values.JxW(q_point);
                            }
                        }
                    }
                }
                */
            }
        }

        bool bCellHaveDirichletBC = false;
        bool bCellHaveNeumannBC   = false;
        for(std::map< unsigned int, std::map<unsigned int, double> >::iterator it = mapDirichlets.begin(); it != mapDirichlets.end(); it++)
        {
            unsigned int id                                 = it->first;
            std::map<unsigned int, double>& boundary_values = it->second;

            for(size_t m = 0; m < local_dof_indices.size(); m++)
            {
                unsigned int dof = local_dof_indices[m];
                if(boundary_values.find(dof) != boundary_values.end())
                {
                    // Cell have one or more nodes with Dirichlet BCs
                    bCellHaveDirichletBC = true;
                    break;
                }
            }
        }
        for(std::map< unsigned int, std::map<unsigned int, double> >::iterator it = mapNeumanns.begin(); it != mapNeumanns.end(); it++)
        {
            unsigned int id                                 = it->first;
            std::map<unsigned int, double>& boundary_values = it->second;

            for(size_t m = 0; m < local_dof_indices.size(); m++)
            {
                unsigned int dof = local_dof_indices[m];
                if(boundary_values.find(dof) != boundary_values.end())
                {
                    // Cell have one or more nodes with Dirichlet BCs
                    bCellHaveNeumannBC = true;
                    break;
                }
            }
        }

        if(bCellHaveDirichletBC && bCellHaveNeumannBC)
        {
            std::cout << "*******************************************" << std::endl;
            std::cout << "bCellHaveDirichletBC && bCellHaveNeumannBC" << std::endl;
            std::cout << "*******************************************" << std::endl;
        }


        // We already have a pre-calculated map<global_dof_index, bc_value> for every ID marked as having Dirichlet BCs imposed
        for(std::map< unsigned int, std::map<unsigned int, double> >::iterator it = mapDirichlets.begin(); it != mapDirichlets.end(); it++)
        {
            unsigned int id                                 = it->first;
            std::map<unsigned int, double>& boundary_values = it->second;

            std::cout << "boundary_values" << std::endl;
            for(std::map<unsigned int,double>::iterator bviter = boundary_values.begin(); bviter != boundary_values.end(); bviter++)
                std::cout << "(" << bviter->first << ", " << bviter->second << ") ";
            std::cout << std::endl;

            std::cout << "local_dof_indices" << std::endl;
            for(std::vector<unsigned int>::iterator ldiiter = local_dof_indices.begin(); ldiiter != local_dof_indices.end(); ldiiter++)
                std::cout << *ldiiter << " ";
            std::cout << std::endl;

            std::cout << "fel_neuman" << std::endl;
            fel_neuman.print(std::cout);

            std::cout << "Kel_diffusion pre" << std::endl;
            Kel_diffusion.print_formatted(std::cout);
            std::cout << "fel_generation pre" << std::endl;
            fel_generation.print(std::cout);
            MatrixTools::local_apply_boundary_values(boundary_values,
                                                     local_dof_indices,
                                                     Kel_diffusion,
                                                     fel_generation,
                                                     true);
            std::cout << "Kel_diffusion posle" << std::endl;
            Kel_diffusion.print_formatted(std::cout);
            std::cout << "fel_generation posle" << std::endl;
            fel_generation.print(std::cout);
        }


        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for(unsigned int j = 0; j < dofs_per_cell; ++j)
            { 
                K_diffusion.add   (local_dof_indices[i], local_dof_indices[j], Kel_diffusion(i,j));
                K_accumulation.add(local_dof_indices[i], local_dof_indices[j], Kel_accumulation(i,j));
                K_dirichlet.add   (local_dof_indices[i], local_dof_indices[j], Kel_dirichlet(i,j));
            }
            
            f_neuman    (local_dof_indices[i]) += fel_neuman(i);
            f_dirichlet (local_dof_indices[i]) += fel_dirichlet(i);
            f_generation(local_dof_indices[i]) += fel_generation(i);
        }
    }
    
//  A part we do not need for we use the way of imposing a penalty for DirichletBC
//    hanging_node_constraints.condense(system_matrix);
//    hanging_node_constraints.condense(system_matrix_dt);
//    hanging_node_constraints.condense(system_rhs);
    
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

}
