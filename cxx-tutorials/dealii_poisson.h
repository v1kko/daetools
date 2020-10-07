#ifndef DAE_DEALII_POISSON_H
#define DAE_DEALII_POISSON_H

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

#include <deal.II/grid/grid_in.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <deal.II/base/logstream.h>

#include "fem_common.h"

namespace dae
{
namespace fe
{
using namespace dealii;

// Poisson:                        ΔΨ = f
// Transient diffusion:            a(∂Ψ/∂t) = ∇D∇Ψ + g(Ψ)
// Transient convection-diffusion: a(∂Ψ/∂t) = ∇D∇Ψ - ∇.(uΨ) + g(Ψ)

/**************************************************************
  daeFEM: base Finite Elements class
***************************************************************/
class daeFEM
{
public:
    virtual ~daeFEM() {}
    
    virtual void Initialize() = 0;
    virtual void SetUpParametersAndDomains() = 0;
    virtual void DeclareEquations(void) = 0;
};

class bcDirichlet
{
public:
    bcDirichlet(int _id, adouble _value)
    {
        id    = _id;
        value = _value;
    }
    
public:
    int     id;
    adouble value;
};

class bcNeumann
{
public:
    bcNeumann(int _id, adouble _value)
    {
        id    = _id;
        value = _value;
    }
    
public:
    int     id;
    adouble value;
};

/**************************************************************
  dae_dealII_Poisson
  ΔΨ = f
***************************************************************/
template <int dim>
class dae_dealII_Poisson : public daeFEM
{
public:
    dae_dealII_Poisson(daeModel& _parent, daeDomain&  _xyz, daeVariable& _u);
    
    void Initialize();
    void SetUpParametersAndDomains();
    void DeclareEquations(void);

public:
// daetools equations related data
    daeDomain&           xyz;
    daeVariable&         u;
    daeModel&            parent;
    
// daetools matrices and vectors
    size_t               Np;
    adoubleCSRMatrix     cij;
    adoubleCSRMatrix     kij;
    std::vector<adouble> fi;
    
// deal.II related data
    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;    
    SparsityPattern      sparsity_pattern;
};

/**************************************************************
 Implementation
***************************************************************/
template <int dim>
class RightHandSide : public Function<dim>
{
  public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const
    {
      double return_value = 0;
      for (unsigned int i=0; i<dim; ++i)
        return_value += 4*std::pow(p(i), 4);
    
      return return_value;
    }
};

template <int dim>
class BoundaryValues : public Function<dim>
{
  public:
    BoundaryValues () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const
    {
      return p.square();
    }
};

template<int dim>
dae_dealII_Poisson<dim>::dae_dealII_Poisson(daeModel& _parent, daeDomain&  _xyz, daeVariable& _u) 
                  : xyz(_xyz), u(_u), parent(_parent),
                    fe(1), dof_handler(triangulation)
{
}
    
template <int dim>
void dae_dealII_Poisson<dim>::Initialize()
{
// 1. Create grid
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (4);
    
    {
        typename Triangulation<dim>::cell_iterator cell = triangulation.begin (),
                                                   endc = triangulation.end();
        for(; cell!=endc; ++cell)
        {
            for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            {
                if(cell->face(face)->center()(0) == -1)
                {
                    cell->face(face)->set_boundary_indicator (1);
                }
            }
        }
    }
    /* Reading from a file:
     GridIn<dim> grid;
     grid.attach_triangulation(triangulation);
     std::fstream mesh('filename', ios_base::in);
     grid.read(mesh);
    */
    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Total number of cells: "
              << triangulation.n_cells()
              << std::endl;
    
    dof_handler.distribute_dofs (fe);
    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;
  
// 2. Setup system      
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);
  
    std::ofstream out ("sparsity_pattern.1.daetools");
    sparsity_pattern.print_gnuplot (out);
    
// 3. Initialize daetools matrices and vectors
    std::map<size_t, size_t> mapIndexes;
    SparsityPattern::row_iterator iter;
    
    size_t ndofs = dof_handler.n_dofs();
    size_t nnz   = sparsity_pattern.n_nonzero_elements();
    
    Np = ndofs;
    fi.resize(ndofs);
    kij.Reset(ndofs, nnz, CSR_C_STYLE);
    kij.ResetCounters();
    
    for(size_t row = 0; row < ndofs; row++)
    {
        mapIndexes.clear();
        for(iter = sparsity_pattern.row_begin(row); iter != sparsity_pattern.row_end(row); iter++)
            mapIndexes.insert(std::make_pair(mapIndexes.size(), *iter));
        
        kij.AddRow(mapIndexes);
    }
    kij.Sort();
    //kij.Print(true);

// 4. Assemble system matrix and rhs
    QGauss<dim>  quadrature_formula(2);
    QGauss<dim-1> face_quadrature_formula(2);
    
    const RightHandSide<dim> right_hand_side;
    
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);
    
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);
    
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
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
        
        for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_matrix(i,j) += (fe_values.shape_grad (i, q_point) *
                                         fe_values.shape_grad (j, q_point) *
                                         fe_values.JxW (q_point));
                }
                cell_rhs(i) += (fe_values.shape_value (i, q_point) *
                                right_hand_side.value (fe_values.quadrature_point (q_point)) *
                                fe_values.JxW (q_point));
            }
        }
        
        for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if(cell->face(face)->at_boundary() && cell->face(face)->boundary_indicator() == 1)
            {
                fe_face_values.reinit (cell, face);
                
                for(unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                {
                    const double neumann_value = 1;
                    
                    for(unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        cell_rhs(i) += (neumann_value *
                                        fe_face_values.shape_value(i,q_point) *
                                        fe_face_values.JxW(q_point));
                    }
                }
            }
        }
        
        cell->get_dof_indices (local_dof_indices);
        for(unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for(unsigned int j=0; j<dofs_per_cell; ++j)
            {
            /* Start daetools specific */                
                if(!kij(local_dof_indices[i], local_dof_indices[j]).node)
                    kij(local_dof_indices[i], local_dof_indices[j]) = adouble_(cell_matrix(i,j));
                else
                    kij(local_dof_indices[i], local_dof_indices[j]) += adouble_(cell_matrix(i,j));
            /* End daetools specific */                 
            }
            
            /* Start daetools specific */                
            if(!fi[local_dof_indices[i]].node)
                fi[local_dof_indices[i]] = adouble_(cell_rhs(i));
            else
                fi[local_dof_indices[i]] += adouble_(cell_rhs(i));
            /* End daetools specific */                
        }
    }

    // Dirichlett's BC                
    std::map<unsigned int,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              BoundaryValues<dim>(),
                                              boundary_values);

    std::map<unsigned int, double>::const_iterator it;
    for(it = boundary_values.begin(); it != boundary_values.end(); it++)
    {
        size_t index = it->first;
        double value = it->second;
        
        // 1. Set all elements of the row 'index' to zero
        //    Why? If node is zero it won't be added to the equation!!!
        for(int k = kij.IA[index]; k < kij.IA[index+1]; k++)
            kij.A[k] = 0;  
        
        // 2. Set the diagonal matrix item to one 
        kij(index, index) = adouble_(1);
        
        // 3. Set the right hand side to 'value'
        fi[index] = adouble_(value);
    }
}

template<int dim>
void dae_dealII_Poisson<dim>::SetUpParametersAndDomains()
{
    xyz.CreateArray(Np);
}

template<int dim>
void dae_dealII_Poisson<dim>::DeclareEquations(void)
{
    int counter;
    daeEquation* eq;
    adouble res;
    
    for(size_t i = 0; i < Np; i++)
    {
        res = 0;
        //std::cout << (boost::format("k0= %1% to %2%") % A.IA[i] % A.IA[i+1]).str() << std::endl;

        counter = 0;
        for(int k = kij.IA[i]; k < kij.IA[i+1]; k++)
        {
            //std::cout << (boost::format("JA[%1%] = %2%") % k % A.JA[k]).str() << std::endl;
            
        // If node is NULL the values is zero; therefore, skip the item
            if(!kij.A[k].node)
                continue;
            
            if(counter == 0)
                res = kij.A[k] * u(kij.JA[k]);
            else
                res = res + kij.A[k] * u(kij.JA[k]);
            counter++;
        }
        eq = parent.CreateEquation("Elem_" + u.GetName() + "_" + toString(i), "");
        eq->SetResidual( res - fi[i] );
    }
}

}
}

#endif
