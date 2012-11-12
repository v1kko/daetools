#include "dealii_equation_generator.h"
using dae::core::adouble;

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <deal.II/base/logstream.h>
using namespace dealii;

template <int dim>
class Step4
{
public:
    Step4 ();
    void run ();
    
private:
    void make_grid ();
    void setup_system();
    void assemble_system ();
    void solve ();
    void output_results () const;
    
    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;
    
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    
    Vector<double>       solution;
    Vector<double>       system_rhs;
    
    
    adoubleCSRMatrix     A;
    std::vector<adouble> x;
    std::vector<adouble> b;
};

template <int dim>
class RightHandSide : public Function<dim>
{
  public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};



template <int dim>
class BoundaryValues : public Function<dim>
{
  public:
    BoundaryValues () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
  double return_value = 0;
  for (unsigned int i=0; i<dim; ++i)
    return_value += 4*std::pow(p(i), 4);

  return return_value;
}

template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
  return p.square();
}

template <int dim>
Step4<dim>::Step4 ()
    :
      fe (1),
      dof_handler (triangulation)
{}

template <int dim>
void Step4<dim>::make_grid ()
{
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (4);
    
    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Total number of cells: "
              << triangulation.n_cells()
              << std::endl;
}

template <int dim>
void Step4<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
  
    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;
  
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);
  
    system_matrix.reinit (sparsity_pattern);
  
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    
//    std::ofstream out ("sparsity_pattern.1");
//    sparsity_pattern.print_gnuplot (out);
    
    
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
}

template <int dim>
void Step4<dim>::assemble_system ()
{
    QGauss<dim>  quadrature_formula(2);
    
    const RightHandSide<dim> right_hand_side;
    
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);
    
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    
    FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs(dofs_per_cell);
    
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    
    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    
    for (; cell!=endc; ++cell)
    {
        fe_values.reinit (cell);
        cell_matrix = 0;
        cell_rhs = 0;
        
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (fe_values.shape_grad (i, q_point) *
                                     fe_values.shape_grad (j, q_point) *
                                     fe_values.JxW (q_point));
  
              cell_rhs(i) += (fe_values.shape_value (i, q_point) *
                              right_hand_side.value (fe_values.quadrature_point (q_point)) *
                              fe_values.JxW (q_point));
            }
        
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
                system_matrix.add (local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i,j));
                if(!A(local_dof_indices[i], local_dof_indices[j]).node)
                    A(local_dof_indices[i], local_dof_indices[j]) = adouble_(cell_matrix(i,j));
                else
                    A(local_dof_indices[i], local_dof_indices[j]) += adouble_(cell_matrix(i,j));
                
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
            if(!b[local_dof_indices[i]].node)
                b[local_dof_indices[i]] = adouble_(cell_rhs(i));
            else
                b[local_dof_indices[i]] += adouble_(cell_rhs(i));
        }
    }
    
    //daeNodeSaveAsContext c;
    //std::cout << A(0, 0).node->SaveAsLatex(&c) << std::endl;    
    //A.Print();

    std::map<unsigned int,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              BoundaryValues<dim>(),
                                              boundary_values);
    for(std::map<unsigned int,double>::const_iterator it = boundary_values.begin(); it != boundary_values.end(); it++)
    {
        //std::cout << (boost::format("dof[%1%] = %2%") % it->first % it->second).str() << std::endl;
       
        size_t index = it->first;
        double value = it->second;
        // 1. Set all elements of the row 'index' to zero
        for(int k = A.IA[index]; k < A.IA[index+1]; k++)
            A.A[k] = 0; // To avoid adding the item to the equation adouble must not have a node!!! 
        
        // 2. Set the diagonal matrix item to one 
        A(index, index) = adouble_(1);
        
        // 3. Set the right hand side to 'value'
        b[index] = adouble_(value);
    }
    
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
}

template <int dim>
void Step4<dim>::solve ()
{
    SolverControl solver_control (1000, 1e-12);
    SolverCG<>    solver (solver_control);
    solver.solve (system_matrix, solution, system_rhs,
                  PreconditionIdentity());
    
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence."
              << std::endl;
}

template <int dim>
void Step4<dim>::output_results () const
{
    DataOut<dim> data_out;
    
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    
    data_out.build_patches ();
    
    std::ofstream output (dim == 2 ?
                              "solution-2d.vtk" :
                              "solution-3d.vtk");
    data_out.write_vtk (output);
    
    for(size_t i = 0; i < solution.size(); i++)
        std::cout << (boost::format("T[%1%] = %2%") % i % solution[i]).str() << std::endl;
}


template <int dim>
void Step4<dim>::run ()
{
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
    
    try
    {
        make_grid();
        setup_system ();
        assemble_system ();
        solve ();
        output_results ();

        boost::scoped_ptr<daeSimulation_t>		pSimulation(new simTutorial1(A.N, A, x, b));  
        boost::scoped_ptr<daeDataReporter_t>	pDataReporter(daeCreateTCPIPDataReporter());
        boost::scoped_ptr<daeIDASolver>			pDAESolver(new daeIDASolver());
        boost::scoped_ptr<daeLog_t>				pLog(daeCreateStdOutLog());
        
        if(!pSimulation)
            daeDeclareAndThrowException(exInvalidPointer); 
        if(!pDataReporter)
            daeDeclareAndThrowException(exInvalidPointer); 
        if(!pDAESolver)
            daeDeclareAndThrowException(exInvalidPointer); 
        if(!pLog)
            daeDeclareAndThrowException(exInvalidPointer); 
    
        time_t rawtime;
        struct tm* timeinfo;
        char buffer[80];
        time(&rawtime);
        timeinfo = localtime(&rawtime);	  
        strftime (buffer, 80, " [%d.%m.%Y %H:%M:%S]", timeinfo);
        string simName = pSimulation->GetModel()->GetName() + buffer;
        if(!pDataReporter->Connect(string(""), simName))
            daeDeclareAndThrowException(exInvalidCall); 
    
        pSimulation->SetReportingInterval(10);
        pSimulation->SetTimeHorizon(1000);
        pSimulation->GetModel()->SetReportingOn(true);
        
        pSimulation->Initialize(pDAESolver.get(), pDataReporter.get(), pLog.get());
        
        pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
        //pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
      
        pSimulation->SolveInitial();
        //pSimulation->Run();
        pSimulation->Finalize();
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}

int main ()
{
    deallog.depth_console (0);
    {
        Step4<2> laplace_problem_2d;
        laplace_problem_2d.run ();
    }
    
//    {
//        Step4<3> laplace_problem_3d;
//        laplace_problem_3d.run ();
//    }
    
    return 0;
}
