/* Author: Wolfgang Bangerth and Ralf Hartmann, University of Heidelberg, 2000 */

/*    @f$Id: @ref step_7 "step-7".cc 25840 2012-08-09 20:22:00Z bangerth @f$       */
/*                                                                */
/*    Copyright (C) 2000, 2001, 2002, 2003, 2004, 2006, 2007, 2008, 2009, 2011, 2012 by the deal.II authors */
/*                                                                */
/*    This file is subject to QPL and may not be  distributed     */
/*    without copyright and license information. Please refer     */
/*    to the file deal.II/doc/license.html for the  text  and     */
/*    further information on this license.                        */

#include "fem_common.h"
using dae::core::adouble;

#include <typeinfo>
#include <fstream>
#include <iostream>

class adNumber 
{
public:
    adNumber(void) {}
    adNumber(double v) {ad.setValue(v);}
    adNumber(const adNumber& a) {ad = a.ad;}
    virtual ~adNumber(){}

public:
    adNumber& operator =(const adNumber& a) {ad = a.ad; return *this;}
    adNumber& operator =(double v) {ad.setValue(v); return *this;}

    adNumber operator -(void) const {adNumber tmp; tmp.ad = -ad; return tmp;}
    adNumber operator +(void) const {return *this;}

    adNumber& operator +=(const adNumber& a) {ad = ad + a.ad; return *this;}
    adNumber& operator -=(const adNumber& a) {ad = ad - a.ad; return *this;}
    adNumber& operator *=(const adNumber& a) {ad = ad * a.ad; return *this;}
    adNumber& operator *=(double v) {ad *= v; return *this;}
    adNumber& operator /=(const adNumber& a) {ad = ad / a.ad; return *this;}
    adNumber& operator /=(double v) {ad /= v; return *this;}

    bool operator !=(const adNumber& a) const {return (ad.getValue() != a.ad.getValue());}

    bool operator ==(const adNumber& a) const {return (ad.getValue() == a.ad.getValue());}

    bool operator <=(const adNumber& a) const {return (ad.getValue() <= a.ad.getValue());}

    bool operator >=(const adNumber& a) const {return (ad.getValue() >= a.ad.getValue());}

    bool operator >(const adNumber& a) const {return (ad.getValue() > a.ad.getValue());}

    bool operator <(const adNumber& a) const {return (ad.getValue() < a.ad.getValue());}
	
public:
    adouble ad;
};

adNumber operator +(const adNumber& l, const adNumber& r);
adNumber operator -(const adNumber& l, const adNumber& r);
adNumber operator *(const adNumber& l, const adNumber& r);
adNumber operator *(const adNumber& l, double r);
adNumber operator /(const adNumber& l, const adNumber& r);

namespace std
{
adNumber exp(const adNumber &a);
adNumber log(const adNumber &a);
adNumber sqrt(const adNumber &a);
adNumber sin(const adNumber &a);
adNumber cos(const adNumber &a);
adNumber tan(const adNumber &a);
adNumber asin(const adNumber &a);
adNumber acos(const adNumber &a);
adNumber atan(const adNumber &a);

adNumber pow(const adNumber &a, double v);
adNumber pow(const adNumber &a, const adNumber &b);

adNumber log10(const adNumber &a);

adNumber ceil(const adNumber &a);
adNumber floor(const adNumber &a);

adNumber abs(const adNumber &a);
adNumber fabs(const adNumber &a);
adNumber max(const adNumber &a, const adNumber &b);
adNumber min(const adNumber &a, const adNumber &b);


adNumber operator +(const adNumber& l, const adNumber& r) {adNumber tmp; tmp.ad = l.ad + r.ad; return tmp;}
adNumber operator -(const adNumber& l, const adNumber& r) {adNumber tmp; tmp.ad = l.ad - r.ad; return tmp;}
adNumber operator *(const adNumber& l, const adNumber& r) {adNumber tmp; tmp.ad = l.ad * r.ad; return tmp;}
adNumber operator *(const adNumber& l, double r) {adNumber tmp; tmp.ad = l.ad * r; return tmp;}
adNumber operator *(double l, const adNumber& r) {adNumber tmp; tmp.ad = l * r.ad; return tmp;}
adNumber operator /(const adNumber& l, const adNumber& r) {adNumber tmp; tmp.ad = l.ad / r.ad; return tmp;}
adNumber operator /(const adNumber& l, double r) {adNumber tmp; tmp.ad = l.ad / r; return tmp;}

adNumber exp(const adNumber &a) {adNumber tmp; tmp.ad = exp(a.ad); return tmp;}
adNumber log(const adNumber &a) {adNumber tmp; tmp.ad = log(a.ad); return tmp;}
adNumber sqrt(const adNumber &a) {adNumber tmp; tmp.ad = sqrt(a.ad); return tmp;}
adNumber sin(const adNumber &a) {adNumber tmp; tmp.ad = sin(a.ad); return tmp;}
adNumber cos(const adNumber &a) {adNumber tmp; tmp.ad = cos(a.ad); return tmp;}
adNumber tan(const adNumber &a) {adNumber tmp; tmp.ad = tan(a.ad); return tmp;}
adNumber asin(const adNumber &a) {adNumber tmp; tmp.ad = asin(a.ad); return tmp;}
adNumber acos(const adNumber &a) {adNumber tmp; tmp.ad = acos(a.ad); return tmp;}
adNumber atan(const adNumber &a) {adNumber tmp; tmp.ad = atan(a.ad); return tmp;}

adNumber pow(const adNumber &a, double v) {adNumber tmp; tmp.ad = pow(a.ad, v); return tmp;}
adNumber pow(const adNumber &a, const adNumber &b) {adNumber tmp; tmp.ad = pow(a.ad, b.ad); return tmp;}

adNumber log10(const adNumber &a) {adNumber tmp; tmp.ad = exp(a.ad); return tmp;}

adNumber ceil(const adNumber &a) {adNumber tmp; tmp.ad = ceil(a.ad); return tmp;}
adNumber floor(const adNumber &a) {adNumber tmp; tmp.ad = floor(a.ad); return tmp;}

adNumber abs(const adNumber &a) {adNumber tmp; tmp.ad = abs(a.ad); return tmp;}
adNumber fabs(const adNumber &a) {adNumber tmp; tmp.ad = abs(a.ad); return tmp;}
adNumber max(const adNumber &a, const adNumber &b) {adNumber tmp; tmp.ad = max(a.ad, b.ad); return tmp;}
adNumber min(const adNumber &a, const adNumber &b) {adNumber tmp; tmp.ad = min(a.ad, b.ad); return tmp;}
}
using namespace std;

std::ostream& operator<<(std::ostream& out, const adNumber& a);
std::size_t memory_consumption (const adNumber& a);

std::ostream& operator<<(std::ostream& out, const adNumber& a)
{
    return out << a.ad;
}

std::size_t memory_consumption (const adNumber& a)
{
    return 0;
}

namespace dealii
{
namespace internal
{
void print (const adNumber &t, const char *format);
void copy (const adNumber *begin,
           const adNumber *end,
           adNumber       *dest);

void print (const adNumber &t, const char *format)
{       
}

void copy (const adNumber *begin,
           const adNumber *end,
           adNumber       *dest)
{
}

}
}


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

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>

//extern void run_dealii_poisson_test();

#include <boost/math/complex.hpp>

namespace dealii
{
namespace numbers
{
template <>
struct NumberTraits<adNumber>
{
                                     /**
                                      * A flag that specifies whether the
                                      * template type given to this class is
                                      * complex or real. Since this
                                      * specialization of the general
                                      * template is selected for complex
                                      * types, the answer is
                                      * <code>true</code>.
                                      */
    static const bool is_complex = true;
    
    typedef adNumber real_type;

                                     /**
                                      * Return the complex-conjugate of the
                                      * given number.
                                      */
    static
    adNumber conjugate (const adNumber &x);

                                     /**
                                      * Return the square of the absolute
                                      * value of the given number. Since
                                      * this specialization of the general
                                      * template is chosen for types equal
                                      * to std::complex, this function
                                      * returns the product of a number and
                                      * its complex conjugate.
                                      */
    static
    real_type abs_square (const adNumber &x);


                                     /**
                                      * Return the absolute value of a
                                      * complex number.
                                      */
    static
    real_type abs (const adNumber &x);
};
}

namespace numbers
{
template <>
adNumber
NumberTraits<adNumber >::conjugate (const adNumber &x)
{
  return x;
}



template <>
typename NumberTraits<adNumber >::real_type
NumberTraits<adNumber >::abs (const adNumber &x)
{
  return std::abs(x);
}



template <>
typename NumberTraits<adNumber >::real_type
NumberTraits<adNumber >::abs_square (const adNumber &x)
{
  return x * x;
}

}

template
class SparseMatrix<adNumber>;

template
class FullMatrix<adNumber>;

template
class Vector<adNumber>;
}

namespace Step7
{
using namespace dealii;

typedef SparseMatrix<adNumber> adSparseMatrix;
typedef FullMatrix<adNumber>   adFullMatrix;
typedef Vector<adNumber>       adVector;


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
class HelmholtzProblem
{
public:
    
    HelmholtzProblem (unsigned int p, // *********
                      const RefinementMode      refinement_mode);
    
    virtual ~HelmholtzProblem ();
    
    void run ();
    
public:
    virtual void setup_system ();
    virtual void assemble_system ();
    virtual void solve ();
    virtual void refine_grid ();
    virtual void process_solution (const unsigned int cycle);
    
    Triangulation<dim>                      triangulation;
    DoFHandler<dim>                         dof_handler;
    
    SmartPointer<FiniteElement<dim> >       fe; // *******************
    
    ConstraintMatrix                        hanging_node_constraints;
    
    SparsityPattern                         sparsity_pattern;
  
    adSparseMatrix                          system_matrix;
    adVector                                solution;
    adVector                                system_rhs;
    
//    SparseMatrix<double>                    system_matrix;
//    Vector<double>                          solution;
//    Vector<double>                          system_rhs;
    
    const RefinementMode                    refinement_mode;
    
    ConvergenceTable                        convergence_table;
    
    adoubleCSRMatrix     A;
    std::vector<adouble> x;
    std::vector<adouble> b;
};




template <int dim>
HelmholtzProblem<dim>::HelmholtzProblem (unsigned int p,
                                         const RefinementMode refinement_mode) :
    dof_handler (triangulation),
    fe (new FE_Q<dim>(p)),
    refinement_mode (refinement_mode)
{}



template <int dim>
HelmholtzProblem<dim>::~HelmholtzProblem ()
{
    dof_handler.clear ();
}



template <int dim>
void HelmholtzProblem<dim>::setup_system ()
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
    
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    
    std::ofstream out ("sparsity_pattern_after");
    sparsity_pattern.print_gnuplot (out);

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
void HelmholtzProblem<dim>::assemble_system ()
{
    QGauss<dim>   quadrature_formula(3);
    QGauss<dim-1> face_quadrature_formula(3);
    
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    
    adFullMatrix  cell_matrix (dofs_per_cell, dofs_per_cell);
    adVector      cell_rhs (dofs_per_cell);
    
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
        cell_matrix = adNumber(0);
        cell_rhs = adNumber(0);
        
        fe_values.reinit (cell);
        
        right_hand_side.value_list (fe_values.get_quadrature_points(),
                                    rhs_values);
        
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    cell_matrix(i,j) += adNumber((fe_values.shape_grad(i,q_point) *
                                          fe_values.shape_grad(j,q_point)
                                          +
                                          fe_values.shape_value(i,q_point) *
                                          fe_values.shape_value(j,q_point)) *
                                          fe_values.JxW(q_point));
                }
                
                cell_rhs(i) += adNumber(fe_values.shape_value(i,q_point) *
                                rhs_values [q_point] *
                                fe_values.JxW(q_point));
            }
        
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary()
                    &&
                    (cell->face(face)->boundary_indicator() == 1))
            {
                fe_face_values.reinit (cell, face);
                
                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                {
                    const double neumann_value
                            = (exact_solution.gradient (fe_face_values.quadrature_point(q_point)) *
                               fe_face_values.normal_vector(q_point));
                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                        cell_rhs(i) += adNumber(neumann_value *
                                        fe_face_values.shape_value(i,q_point) *
                                        fe_face_values.JxW(q_point));
                }
            }
        
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            { 
                system_matrix.add (local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i,j));

            //    if(!A(local_dof_indices[i], local_dof_indices[j]).node)
            //        A(local_dof_indices[i], local_dof_indices[j]) = adouble_(cell_matrix(i,j));
            //    else
            //        A(local_dof_indices[i], local_dof_indices[j]) += adouble_(cell_matrix(i,j));
            }
            
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
            //if(!b[local_dof_indices[i]].node)
            //    b[local_dof_indices[i]] = adouble_(cell_rhs(i));
            //else
            //    b[local_dof_indices[i]] += adouble_(cell_rhs(i));
        }
    }
    
    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);
    
    std::map<unsigned int,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              Solution<dim>(),
                                              boundary_values);
/*    
    std::cout << (boost::format("N = %1%, NNZ = %2%") % A.N % A.NNZ).str() << std::endl;
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
    //A.Print(true);
*/    
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
}



template <int dim>
void HelmholtzProblem<dim>::solve ()
{
//    SolverControl           solver_control (1000, 1e-12);
//    SolverCG<>              cg (solver_control);
    
//    PreconditionSSOR<> preconditioner;
//    preconditioner.initialize(system_matrix, 1.2);
    
//    cg.solve (system_matrix, solution, system_rhs,
//              preconditioner);
    
//    hanging_node_constraints.distribute (solution);
}



template <int dim>
void HelmholtzProblem<dim>::refine_grid ()
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
void HelmholtzProblem<dim>::process_solution (const unsigned int cycle)
{
/*
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
*/
}


template <int dim>
void HelmholtzProblem<dim>::run ()
{
    try
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
        
    /*    
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
    */    
        //run_dealii_poisson_test();
        
/*        
        boost::scoped_ptr<daeSimulation_t>		pSimulation(new simTutorial1(A.N, A, x, b));  
        boost::scoped_ptr<daeDataReporter_t>	pDataReporter(daeCreateTCPIPDataReporter());
        boost::scoped_ptr<daeIDASolver>			pDAESolver(new daeIDASolver());
        boost::scoped_ptr<daeLog_t>				pLog(daeCreateStdOutLog());
        boost::scoped_ptr<daeIDALASolver_t>		pLASolver(daeCreateSuperLUSolver());
        
        if(!pSimulation)
            daeDeclareAndThrowException(exInvalidPointer); 
        if(!pDataReporter)
            daeDeclareAndThrowException(exInvalidPointer); 
        if(!pDAESolver)
            daeDeclareAndThrowException(exInvalidPointer); 
        if(!pLog)
            daeDeclareAndThrowException(exInvalidPointer); 
    
        pDAESolver->SetLASolver(pLASolver.get());

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
        
        //pSimulation->GetModel()->SaveModelReport(pSimulation->GetModel()->GetName() + ".xml");
        //pSimulation->GetModel()->SaveRuntimeModelReport(pSimulation->GetModel()->GetName() + "-rt.xml");
      
        pSimulation->SolveInitial();
        
        //pLASolver->SaveAsXPM("daetools_pattern");
        
        //pSimulation->Run();
        pSimulation->Finalize();
*/
     }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}

}


namespace Step7
{
template const double SolutionBase<2>::width;
}
