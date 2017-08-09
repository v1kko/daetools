#ifndef DEAL_II_TEMPLATE_INSTANTIATIONS_H
#define DEAL_II_TEMPLATE_INSTANTIATIONS_H

#include "../Core/coreimpl.h"

//#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <algorithm>
#include <numeric>
#include <set>
#include <ostream>

namespace std
{
adouble abs(const adouble& a);
adouble sqrt(const adouble& a);

adouble abs(const adouble& a)
{
    throw std::runtime_error("function std::abs(adouble) called");
    return a;
}
adouble sqrt(const adouble& a)
{
    throw std::runtime_error("function std::sqrt(adouble) called");
    return a;
}
}

#include <deal.II/lac/vector.templates.h>
#include <deal.II/lac/block_vector.templates.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/full_matrix.templates.h>
#include <deal.II/lac/block_sparse_matrix.templates.h>
#include <deal.II/base/function_time.templates.h>
#include <deal.II/base/function.templates.h>
#include <deal.II/base/tensor_function.templates.h>

namespace dealii
{
// Many template functions need to be specialized for Number=adouble
namespace numbers
{
    template <>
    typename NumberTraits<adouble>::real_type
    NumberTraits<adouble>::abs (const adouble &x)
    {
        throw std::runtime_error("function NumberTraits<adouble>::abs(adouble) called");
        return x;
    }
}

namespace internal
{
namespace VectorOperations
{
    template <>
    void print (const adouble    &t,
                const char *format)
    {
        printf("%f", t.getValue());
    }
}
}

namespace MemoryConsumption
{
    template <>
    inline
    typename std_cxx11::enable_if<!(std_cxx11::is_fundamental<adouble>::value || std_cxx11::is_pointer<adouble>::value), std::size_t>::type
    memory_consumption (const adouble &t)
    { return 0; }

}

template <>
typename Vector<adouble>::real_type
Vector<adouble>::lp_norm (const adouble p) const
{ return adouble(); }

template <>
typename Vector<adouble>::real_type
Vector<adouble>::l2_norm () const
{ return adouble(); }

template <>
typename FullMatrix<adouble>::real_type
FullMatrix<adouble>::l1_norm () const
{ return adouble(); }

template <>
typename FullMatrix<adouble>::real_type
FullMatrix<adouble>::linfty_norm () const
{ return adouble(); }

template <>
typename FullMatrix<adouble>::real_type
FullMatrix<adouble>::frobenius_norm () const
{ return adouble(); }

template <>
typename FullMatrix<adouble>::real_type
FullMatrix<adouble>::relative_symmetry_norm2 () const
{ return adouble(); }

template <>
void FullMatrix<adouble>::print_formatted (
    std::ostream       &out,
    const unsigned int  precision,
    const bool          scientific,
    const unsigned int  width_,
    const char         *zero_string,
    const double        denominator,
    const double        threshold) const
    {
    }

    template <>
    void FullMatrix<adouble>::gauss_jordan ()
    {}

template <>
inline void
SparseMatrix<adouble>::add (const size_type i,
                            const size_type j,
                            const adouble value)
{
    // Do not add if there is nothing in the adouble object, that is
    // both the value and the node are zero/NULL.
    if(!value.node && value.getValue() == 0.0)
        return;

    const size_type index = cols->operator()(i, j);

    // it is allowed to add elements to the matrix that are not part of the
    // sparsity pattern, if the value to which we set it is zero
    if (index == SparsityPattern::invalid_entry)
    {
        //printf("SparseMatrix<adouble>::add invalid item (%d,%j)", i, j);
        return;
    }

    val[index] += value;
}

template <>
void
SparseMatrix<adouble>::reinit (const SparsityPattern &sparsity)
{
    cols = &sparsity;

    if (cols->empty())
    {
        if (val != 0)
            delete[] val;
        val = 0;
        max_len = 0;
        return;
    }

    const std::size_t N = cols->n_nonzero_elements();
    if (N > max_len || max_len == 0)
    {
        if (val != 0)
            delete[] val;
        val = new adouble[N];
        max_len = N;
    }

    // Achtung, Achtung!!
    // This causes the seg. fault when std::memset is called on an aray of non-PODs
    // It is not needed for adouble, since it gets initialised to zeros in the constructor, anyway.
    //*this = 0.;
}

template <>
inline void
BlockMatrixBase< SparseMatrix<adouble> >::add (const size_type  i,
                                               const size_type  j,
                                               const adouble value)
{
    // Do not add if there is nothing in the adouble object, that is
    // both the value and the node are zero/NULL.
    if(!value.node && value.getValue() == 0.0)
        return;

    const std::pair<unsigned int,size_type>
    row_index = row_block_indices.global_to_local (i),
    col_index = column_block_indices.global_to_local (j);
    block(row_index.first,col_index.first).add (row_index.second,
                                                col_index.second,
                                                value);
}

template <>
void
Vector<adouble>::allocate()
{
  // make sure that we don't create a memory leak
  Assert (val == 0, ExcInternalError());

  // then allocate memory with the proper alignment requirements of 64 bytes
  val = new adouble[max_vec_size];
}

template <>
void
Vector<adouble>::deallocate()
{
  delete[] val;
  val = 0;
}

template <>
inline void
Vector<adouble>::reinit (const size_type n,
                         const bool omit_zeroing_entries)
{
  if (n==0)
    {
      if (val) deallocate();
      val = 0;
      max_vec_size = vec_size = 0;
      return;
    };

  if (n>max_vec_size)
    {
      if (val) deallocate();
      max_vec_size = n;

      allocate();

    };
  vec_size = n;
  if (omit_zeroing_entries == false)
      for(size_t i = 0; i < vec_size; i++)
          val[i] = adouble(0);

  // Achtung, Achtung!!
  // This causes the seg. fault when std::memset is called on an aray of non-PODs
  // It is not needed for adouble, since it gets initialised to zeros in the constructor, anyway.
  //*this = 0;
}

// Explicit instantiation of template classes for Number=adouble
template class Vector<adouble>;
template class FullMatrix<adouble>;
template class SparseMatrix<adouble>;
template class BlockVector<adouble>;
template class BlockSparseMatrix<adouble>;

template class FunctionTime<adouble>;
template class Function<1, adouble>;
template class Function<2, adouble>;
template class Function<3, adouble>;
template class ZeroFunction<1, adouble>;
template class ZeroFunction<2, adouble>;
template class ZeroFunction<3, adouble>;
template class ConstantFunction<1, adouble>;
template class ConstantFunction<2, adouble>;
template class ConstantFunction<3, adouble>;

template class TensorFunction<1,1,adouble>;
template class TensorFunction<1,2,adouble>;
template class TensorFunction<1,3,adouble>;

template class TensorFunction<2,1,adouble>;
template class TensorFunction<2,2,adouble>;
template class TensorFunction<2,3,adouble>;

template class TensorFunction<3,1,adouble>;
template class TensorFunction<3,2,adouble>;
template class TensorFunction<3,3,adouble>;

namespace daeMatrixTools
{
namespace
{
  template <typename Iterator>
  bool column_less_than(const typename Iterator::value_type p,
                        const unsigned int column)
  {
    return (p.column() < column);
  }
}

// This is my version of the local_apply_boundary_values function (NOT a specialization)!
template<class Number>
inline void local_apply_boundary_values (const typename std::map<types::global_dof_index,Number> &boundary_values,
                                         const std::vector<types::global_dof_index> &local_dof_indices,
                                         boost::numeric::ublas::matrix<adouble>& local_matrix,
                                         std::vector<adouble>& local_rhs,
                                         const bool eliminate_columns = true)
{
  Assert (local_dof_indices.size() == local_matrix.m(),
          ExcDimensionMismatch(local_dof_indices.size(),
                               local_matrix.m()));
  Assert (local_dof_indices.size() == local_matrix.n(),
          ExcDimensionMismatch(local_dof_indices.size(),
                               local_matrix.n()));
  Assert (local_dof_indices.size() == local_rhs.size(),
          ExcDimensionMismatch(local_dof_indices.size(),
                               local_rhs.size()));

  // if there is nothing to do, then exit
  // right away
  if (boundary_values.size() == 0)
    return;

  // otherwise traverse all the dofs used in
  // the local matrices and vectors and see
  // what's there to do

  // if we need to treat an entry, then we
  // set the diagonal entry to its absolute
  // value. if it is zero, we used to set it
  // to one, which is a really terrible
  // choice that can lead to hours of
  // searching for bugs in programs (I
  // experienced this :-( ) if the matrix
  // entries are otherwise very large. this
  // is so since iterative solvers would
  // simply not correct boundary nodes for
  // their correct values since the residual
  // contributions of their rows of the
  // linear system is almost zero if the
  // diagonal entry is one. thus, set it to
  // the average absolute value of the
  // nonzero diagonal elements.
  //
  // we only compute this value lazily the
  // first time we need it.
  double average_diagonal = 0;
  const unsigned int n_local_dofs = local_dof_indices.size();
  for (unsigned int i=0; i<n_local_dofs; ++i)
    {
      const typename std::map<types::global_dof_index, Number>::const_iterator boundary_value = boundary_values.find (local_dof_indices[i]);
      if (boundary_value != boundary_values.end())
        {
          // remove this row, except for the
          // diagonal element
          for (unsigned int j=0; j<n_local_dofs; ++j)
            if (i != j)
              local_matrix(i,j) = 0;

          // replace diagonal entry by its
          // absolute value to make sure that
          // everything remains positive, or
          // by the average diagonal value if zero

          // Nota bene:
          //   In my version the average_diagonal is always double (no matter what Number type is)
          //
          if (!local_matrix(i,i).node && local_matrix(i,i).getValue() == 0.0) // the item doesn't have a node and the value is 0
          {
          /* We have some adoubles with a value and some adoubles with a node.
           * We can't tell what is the average value taking into account all of them,
           * but we can work with those items that have float value. */

              // if average diagonal hasn't
              // yet been computed, do so now
              if (average_diagonal == 0.0)
                {
                  unsigned int nonzero_diagonals = 0;
                  for (unsigned int k=0; k<n_local_dofs; ++k)
                    if (!local_matrix(k,k).node && local_matrix(k,k).getValue() != 0.0) // only if it doesn't have a node and has a value!
                      {
                        average_diagonal += std::fabs(local_matrix(k,k).getValue());
                        ++nonzero_diagonals;
                      }
                  if (nonzero_diagonals != 0)
                    average_diagonal /= nonzero_diagonals;
                  else
                    average_diagonal = 0;
                }

              // only if all diagonal entries
              // are zero, then resort to the
              // last measure: choose one
              if (average_diagonal == 0.0)
                average_diagonal = 1.0;

              local_matrix(i,i) = average_diagonal;
          }
          else // the item either has a value or a node
          {
              local_matrix(i,i) = abs(local_matrix(i,i));
          }

          // and replace rhs entry by correct value
          Number ad = local_matrix(i,i) * boundary_value->second;
          ad.node = adNode::SimplifyNode(ad.node);
          local_rhs[i] = ad;

          // finally do the elimination step
          // if requested
          if (eliminate_columns == true)
            {
              for (unsigned int row=0; row<n_local_dofs; ++row)
                if (row != i)
                  {
                    ad = local_matrix(row,i) * boundary_value->second;
                    ad.node = adNode::SimplifyNode(ad.node);
                    if(ad.node || ad.getValue() != 0)
                        local_rhs[row] -= ad;
                    local_matrix(row,i) = 0;
                  }
            }
        }
    }

    for(unsigned int row=0; row<n_local_dofs; ++row)
    {
        local_rhs[row].node = adNode::SimplifyNode(local_rhs[row].node);
        for(unsigned int col=0; col<n_local_dofs; ++col)
            local_matrix(row,col).node = adNode::SimplifyNode(local_matrix(row,col).node);
    }
}

// This is my version of the local_apply_boundary_values function (NOT a specialization)!
template<class Number>
inline void local_process_mass_matrix(const std::map<types::global_dof_index,Number> &boundary_values,
                                      const std::vector<types::global_dof_index> &local_dof_indices,
                                      boost::numeric::ublas::matrix<adouble>& local_mass_matrix,
                                      const bool eliminate_columns = true)
{
  Assert (local_dof_indices.size() == local_mass_matrix.m(),
          ExcDimensionMismatch(local_dof_indices.size(),
                               local_mass_matrix.m()));
  Assert (local_dof_indices.size() == local_mass_matrix.n(),
          ExcDimensionMismatch(local_dof_indices.size(),
                               local_mass_matrix.n()));

  // if there is nothing to do, then exit
  // right away
  if (boundary_values.size() == 0)
    return;

  const unsigned int n_local_dofs = local_dof_indices.size();
  for (unsigned int i=0; i<n_local_dofs; ++i)
    {
      const typename std::map<types::global_dof_index, Number>::const_iterator boundary_value = boundary_values.find (local_dof_indices[i]);
      if (boundary_value != boundary_values.end())
        {
          // remove this row, and the diagonal element
          for (unsigned int j=0; j<n_local_dofs; ++j)
              local_mass_matrix(i,j) = 0;

          // finally do the elimination step
          // if requested
          if (eliminate_columns == true)
          {
              for (unsigned int row=0; row<n_local_dofs; ++row)
                  local_mass_matrix(row,i) = 0;
          }
        }
    }
}
}

namespace daeVectorTools
{
// interpolate boundary values in 1D.
//
// in higher dimensions, we
// use FEValues to figure out
// what to do on faces, but in 1d
// faces are points and it is far
// easier to simply work on
// individual vertices
template <typename DoFHandlerType, template <int,int> class M_or_MC, class Number>
static inline void do_interpolate_boundary_values(const M_or_MC<DoFHandlerType::dimension, DoFHandlerType::space_dimension>     &,
                                                  const DoFHandlerType                                                          &dof,
                                                  const typename FunctionMap<DoFHandlerType::space_dimension,adouble>::type     &function_map,
                                                  std::map<types::global_dof_index,Number>                                      &boundary_values,
                                                  const ComponentMask                                                           &component_mask,
                                                  const dealii::internal::int2type<1>)
{
  const unsigned int dim = DoFHandlerType::dimension;
  const unsigned int spacedim = DoFHandlerType::space_dimension;

  Assert (component_mask.represents_n_components(dof.get_fe().n_components()),
          ExcMessage ("The number of components in the mask has to be either "
                      "zero or equal to the number of components in the finite "
                      "element."));

  // if for whatever reason we were
  // passed an empty map, return
  // immediately
  if (function_map.size() == 0)
    return;

  for (typename DoFHandlerType::active_cell_iterator cell = dof.begin_active(); cell != dof.end(); ++cell)
    for (unsigned int direction=0;
         direction<GeometryInfo<dim>::faces_per_cell; ++direction)
      if (cell->at_boundary(direction)
          &&
          (function_map.find(cell->face(direction)->boundary_id()) != function_map.end()))
        {
          const Function<DoFHandlerType::space_dimension,adouble> &boundary_function = *function_map.find(cell->face(direction)->boundary_id())->second;

          // get the FE corresponding to this
          // cell
          const FiniteElement<dim,spacedim> &fe = cell->get_fe();
          Assert (fe.n_components() == boundary_function.n_components,
                  ExcDimensionMismatch(fe.n_components(),
                                       boundary_function.n_components));

          Assert (component_mask.n_selected_components(fe.n_components()) > 0,
                  ComponentMask::ExcNoComponentSelected());

          // now set the value of
          // the vertex degree of
          // freedom. setting
          // also creates the
          // entry in the map if
          // it did not exist
          // beforehand
          //
          // save some time by
          // requesting values
          // only once for each
          // point, irrespective
          // of the number of
          // components of the
          // function
          Vector<Number> function_values (fe.n_components());
          if (fe.n_components() == 1)
            function_values(0) = boundary_function.value (cell->vertex(direction));
          else
            boundary_function.vector_value (cell->vertex(direction), function_values);

          for (unsigned int i=0; i<fe.dofs_per_vertex; ++i)
            if (component_mask[fe.face_system_to_component_index(i).first])
              boundary_values[cell->vertex_dof_index(direction,i,cell->active_fe_index())] = function_values(fe.face_system_to_component_index(i).first);
        }
}


// template for the case dim > 1.
//
// Since the function has a template argument
// dim_, it is clearly less specialized than the 1D function above and
// whenever possible (i.e., if dim==1), the function template above will be used
template <typename DoFHandlerType, template <int,int> class M_or_MC, int dim_, class Number>
static inline void do_interpolate_boundary_values(const M_or_MC<DoFHandlerType::dimension, DoFHandlerType::space_dimension> &mapping,
                                                  const DoFHandlerType                                                      &dof,
                                                  const typename FunctionMap<DoFHandlerType::space_dimension,adouble>::type &function_map,
                                                  std::map<types::global_dof_index,Number>                                  &boundary_values,
                                                  const ComponentMask                                                       &component_mask,
                                                  const dealii::internal::int2type<dim_>)
{
  const unsigned int dim = DoFHandlerType::dimension;
  const unsigned int spacedim=DoFHandlerType::space_dimension;

  Assert (component_mask.represents_n_components(dof.get_fe().n_components()),
          ExcMessage ("The number of components in the mask has to be either "
                      "zero or equal to the number of components in the finite "
                      "element."));


  // if for whatever reason we were passed an empty map, return
  // immediately
  if (function_map.size() == 0)
    return;

  Assert (function_map.find(numbers::internal_face_boundary_id) == function_map.end(),
          ExcMessage("You cannot specify the special boundary indicator "
                     "for interior faces in your function map."));

  const unsigned int        n_components = DoFTools::n_components(dof);
  const bool                fe_is_system = (n_components != 1);

  for (typename FunctionMap<spacedim,adouble>::type::const_iterator i=function_map.begin(); i!=function_map.end(); ++i)
    Assert (n_components == i->second->n_components,
            ExcDimensionMismatch(n_components, i->second->n_components));

  // field to store the indices
  std::vector<types::global_dof_index> face_dofs;
  face_dofs.reserve (DoFTools::max_dofs_per_face(dof));

  std::vector<Point<spacedim> >  dof_locations;
  dof_locations.reserve (DoFTools::max_dofs_per_face(dof));

  // array to store the values of the boundary function at the boundary
  // points. have two arrays for scalar and vector functions to use the
  // more efficient one respectively
  std::vector<Number>          dof_values_scalar;
  std::vector<Vector<Number> > dof_values_system;
  dof_values_scalar.reserve (DoFTools::max_dofs_per_face (dof));
  dof_values_system.reserve (DoFTools::max_dofs_per_face (dof));

  // before we start with the loop over all cells create an hp::FEValues
  // object that holds the interpolation points of all finite elements
  // that may ever be in use
  dealii::hp::FECollection<dim,spacedim> finite_elements (dof.get_fe());
  dealii::hp::QCollection<dim-1>  q_collection;
  for (unsigned int f=0; f<finite_elements.size(); ++f)
    {
      const FiniteElement<dim,spacedim> &fe = finite_elements[f];

      // generate a quadrature rule on the face from the unit support
      // points. this will be used to obtain the quadrature points on the
      // real cell's face
      //
      // to do this, we check whether the FE has support points on the
      // face at all:
      if (fe.has_face_support_points())
        q_collection.push_back (Quadrature<dim-1>(fe.get_unit_face_support_points()));
      else
        {
          // if not, then we should try a more clever way. the idea is
          // that a finite element may not offer support points for all
          // its shape functions, but maybe only some. if it offers
          // support points for the components we are interested in in
          // this function, then that's fine. if not, the function we call
          // in the finite element will raise an exception. the support
          // points for the other shape functions are left uninitialized
          // (well, initialized by the default constructor), since we
          // don't need them anyway.
          //
          // As a detour, we must make sure we only query
          // face_system_to_component_index if the index corresponds to a
          // primitive shape function. since we know that all the
          // components we are interested in are primitive (by the above
          // check), we can safely put such a check in front
          std::vector<Point<dim-1> > unit_support_points (fe.dofs_per_face);

          for (unsigned int i=0; i<fe.dofs_per_face; ++i)
            if (fe.is_primitive (fe.face_to_cell_index(i,0)))
              if (component_mask[fe.face_system_to_component_index(i).first] == true)
                unit_support_points[i] = fe.unit_face_support_point(i);

          q_collection.push_back (Quadrature<dim-1>(unit_support_points));
        }
    }
  // now that we have a q_collection object with all the right quadrature
  // points, create an hp::FEFaceValues object that we can use to evaluate
  // the boundary values at
  dealii::hp::MappingCollection<dim,spacedim> mapping_collection (mapping);
  dealii::hp::FEFaceValues<dim,spacedim> x_fe_values (mapping_collection, finite_elements, q_collection, update_quadrature_points);

  typename DoFHandlerType::active_cell_iterator cell = dof.begin_active(), endc = dof.end();
  for (; cell!=endc; ++cell)
    if (!cell->is_artificial())
      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell;
           ++face_no)
        {
          const FiniteElement<dim,spacedim> &fe = cell->get_fe();

          // we can presently deal only with primitive elements for
          // boundary values. this does not preclude us using
          // non-primitive elements in components that we aren't
          // interested in, however. make sure that all shape functions
          // that are non-zero for the components we are interested in,
          // are in fact primitive
          for (unsigned int i=0; i<cell->get_fe().dofs_per_cell; ++i)
            {
              const ComponentMask &nonzero_component_array = cell->get_fe().get_nonzero_components (i);
              for (unsigned int c=0; c<n_components; ++c)
                if ((nonzero_component_array[c] == true) && (component_mask[c] == true))
                  Assert (cell->get_fe().is_primitive (i),
                          ExcMessage ("This function can only deal with requested boundary "
                                      "values that correspond to primitive (scalar) base "
                                      "elements"));
            }

          const typename DoFHandlerType::face_iterator face = cell->face(face_no);
          const types::boundary_id boundary_component = face->boundary_id();

          // see if this face is part of the boundaries for which we are
          // supposed to do something, and also see if the finite element
          // in use here has DoFs on the face at all
          if ((function_map.find(boundary_component) != function_map.end())
              &&
              (cell->get_fe().dofs_per_face > 0))
            {
              // face is of the right component
              x_fe_values.reinit(cell, face_no);
              const dealii::FEFaceValues<dim,spacedim> &fe_values = x_fe_values.get_present_fe_values();

              // get indices, physical location and boundary values of
              // dofs on this face
              face_dofs.resize (fe.dofs_per_face);
              face->get_dof_indices (face_dofs, cell->active_fe_index());
              const std::vector<Point<spacedim> > &dof_locations = fe_values.get_quadrature_points ();

              if (fe_is_system)
                {
                  // resize array. avoid construction of a memory
                  // allocating temporary if possible
                  if (dof_values_system.size() < fe.dofs_per_face)
                    dof_values_system.resize (fe.dofs_per_face, Vector<Number>(fe.n_components()));
                  else
                    dof_values_system.resize (fe.dofs_per_face);

                  function_map.find(boundary_component)->second->vector_value_list (dof_locations, dof_values_system);

                  // enter those dofs into the list that match the
                  // component signature. avoid the usual complication
                  // that we can't just use *_system_to_component_index
                  // for non-primitive FEs
                  for (unsigned int i=0; i<face_dofs.size(); ++i)
                    {
                      unsigned int component;
                      if (fe.is_primitive())
                        component = fe.face_system_to_component_index(i).first;
                      else
                        {
                          // non-primitive case. make sure that this
                          // particular shape function _is_ primitive, and
                          // get at it's component. use usual trick to
                          // transfer face dof index to cell dof index
                          const unsigned int cell_i
                            = (dim == 1 ?
                               i
                               :
                               (dim == 2 ?
                                (i<2*fe.dofs_per_vertex ? i : i+2*fe.dofs_per_vertex)
                                :
                                (dim == 3 ?
                                 (i<4*fe.dofs_per_vertex ?
                                  i
                                  :
                                  (i<4*fe.dofs_per_vertex+4*fe.dofs_per_line ?
                                   i+4*fe.dofs_per_vertex
                                   :
                                   i+4*fe.dofs_per_vertex+8*fe.dofs_per_line))
                                 :
                                 numbers::invalid_unsigned_int)));
                          Assert (cell_i < fe.dofs_per_cell, ExcInternalError());

                          // make sure that if this is not a primitive
                          // shape function, then all the corresponding
                          // components in the mask are not set
                          if (!fe.is_primitive(cell_i))
                            for (unsigned int c=0; c<n_components; ++c)
                              if (fe.get_nonzero_components(cell_i)[c])
                                Assert (component_mask[c] == false, FETools::ExcFENotPrimitive());

                          // let's pick the first of possibly more than
                          // one non-zero components. if shape function is
                          // non-primitive, then we will ignore the result
                          // in the following anyway, otherwise there's
                          // only one non-zero component which we will use
                          component = fe.get_nonzero_components(cell_i).first_selected_component();
                        }

                      if (component_mask[component] == true)
                        boundary_values[face_dofs[i]] = dof_values_system[i](component);
                    }
                }
              else
                // fe has only one component, so save some computations
                {
                  // get only the one component that this function has
                  dof_values_scalar.resize (fe.dofs_per_face);
                  function_map.find(boundary_component)->second->value_list (dof_locations, dof_values_scalar, 0);

                  // enter into list

                  for (unsigned int i=0; i<face_dofs.size(); ++i)
                    boundary_values[face_dofs[i]] = dof_values_scalar[i];
                }
            }
        }
} // end of interpolate_boundary_values


template <typename DoFHandlerType>
void interpolate_boundary_values(const Mapping<DoFHandlerType::dimension, DoFHandlerType::space_dimension>      &mapping,
                                 const DoFHandlerType                                                           &dof,
                                 const typename FunctionMap<DoFHandlerType::space_dimension,adouble>::type      &function_map,
                                 std::map<types::global_dof_index,adouble>                                      &boundary_values,
                                 const ComponentMask                                                            &component_mask_)
{
  do_interpolate_boundary_values (mapping, dof, function_map, boundary_values,
                                  component_mask_,
                                  dealii::internal::int2type<DoFHandlerType::dimension>());
}


template <typename DoFHandlerType>
void interpolate_boundary_values(const Mapping<DoFHandlerType::dimension, DoFHandlerType::space_dimension> &mapping,
                                 const DoFHandlerType                                                      &dof,
                                 const types::boundary_id                                                   boundary_component,
                                 const Function<DoFHandlerType::space_dimension,adouble>                   &boundary_function,
                                 std::map<types::global_dof_index,adouble>                                 &boundary_values,
                                 const ComponentMask                                                       &component_mask)
{
  typename FunctionMap<DoFHandlerType::space_dimension,adouble>::type function_map;
  function_map[boundary_component] = &boundary_function;
  interpolate_boundary_values (mapping, dof, function_map, boundary_values,
                               component_mask);
}



template <typename DoFHandlerType>
void interpolate_boundary_values(const DoFHandlerType                                       &dof,
                                 const types::boundary_id                                    boundary_component,
                                 const Function<DoFHandlerType::space_dimension,adouble>    &boundary_function,
                                 std::map<types::global_dof_index,adouble>                  &boundary_values,
                                 const ComponentMask                                        &component_mask)
{
  interpolate_boundary_values(StaticMappingQ1<DoFHandlerType::dimension,DoFHandlerType::space_dimension>::mapping,
                              dof, boundary_component,
                              boundary_function, boundary_values, component_mask);
}


}
}

#endif
