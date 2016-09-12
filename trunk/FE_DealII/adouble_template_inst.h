#ifndef DAE_ADOUBLE_TEMPLATE_INSTANTIATIONS_H
#define DAE_ADOUBLE_TEMPLATE_INSTANTIATIONS_H

#include "../Core/coreimpl.h"

//#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
//#include <deal.II/lac/sparse_matrix_ez.h>
//#include <deal.II/lac/chunk_sparse_matrix.h>
//#include <deal.II/lac/block_sparse_matrix_ez.h>
//#include <deal.II/lac/matrix_block.h>
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
    template <>
    void print (const adouble    &t,
                const char *format)
    {
        printf("%f", t.getValue());
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

   //*this = adouble(0); // this causes a segmentation fault!
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

namespace MatrixTools
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
inline void local_apply_boundary_values (const std::map<types::global_dof_index,double> &boundary_values,
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
      const std::map<types::global_dof_index, double>::const_iterator
      boundary_value = boundary_values.find (local_dof_indices[i]);
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
          // by the average diagonal value if
          // zero
          if (!local_matrix(i,i).node && local_matrix(i,i).getValue() == 0.0) // the item doesn't have a node and the value is 0
          {
          /* We have some adoubles with a value and some adoubles with a node.
           * We can;t tell what is the average value, but we can work with
           * those items that have value. */

              // if average diagonal hasn't
              // yet been computed, do so now
              if (average_diagonal == 0.)
                {
                  unsigned int nonzero_diagonals = 0;
                  for (unsigned int k=0; k<n_local_dofs; ++k)
                    if (local_matrix(k,k).getValue() != 0.0)
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

          // and replace rhs entry by correct
          // value
          local_rhs[i] = local_matrix(i,i) * boundary_value->second;

          // finally do the elimination step
          // if requested
          if (eliminate_columns == true)
            {
              for (unsigned int row=0; row<n_local_dofs; ++row)
                if (row != i)
                  {
                    local_rhs[row] -= local_matrix(row,i) * boundary_value->second;
                    local_matrix(row,i) = 0;
                  }
            }
        }
    }
}

/*
 * Specialization for Number=adouble (not used anymore - the local_apply_boundary_values is used now)!
 * Not sure if it works at all!
 */
//template<>
//void apply_boundary_values (const std::map<types::global_dof_index,double> &boundary_values,
//                            BlockSparseMatrix< adouble > &matrix,
//                            BlockVector< adouble > &solution,
//                            BlockVector< adouble > &right_hand_side,
//                            const bool eliminate_columns)
//{
//  const unsigned int blocks = matrix.n_block_rows();

//  Assert (matrix.n() == right_hand_side.size(),
//          ExcDimensionMismatch(matrix.n(), right_hand_side.size()));
//  Assert (matrix.n() == solution.size(),
//          ExcDimensionMismatch(matrix.n(), solution.size()));
//  Assert (matrix.n_block_rows() == matrix.n_block_cols(),
//          ExcNotQuadratic());
//  Assert (matrix.get_sparsity_pattern().get_row_indices() == matrix.get_sparsity_pattern().get_column_indices(),
//          ExcNotQuadratic());
//  Assert (matrix.get_sparsity_pattern().get_column_indices() == solution.get_block_indices (),
//          ExcBlocksDontMatch ());
//  Assert (matrix.get_sparsity_pattern().get_row_indices() == right_hand_side.get_block_indices (),
//          ExcBlocksDontMatch ());

//  // if no boundary values are to be applied
//  // simply return
//  if (boundary_values.size() == 0)
//    return;


//  const types::global_dof_index n_dofs = matrix.m();

//  // if a diagonal entry is zero
//  // later, then we use another
//  // number instead. take it to be
//  // the first nonzero diagonal
//  // element of the matrix, or 1 if
//  // there is no such thing
//  adouble first_nonzero_diagonal_entry = 0;
//  for (unsigned int diag_block=0; diag_block<blocks; ++diag_block)
//    {
//      for (unsigned int i=0; i<matrix.block(diag_block,diag_block).n(); ++i)
//        if (matrix.block(diag_block,diag_block).diag_element(i) != 0)
//          {
//            first_nonzero_diagonal_entry = matrix.block(diag_block,diag_block).diag_element(i);
//            break;
//          }
//      // check whether we have found
//      // something in the present
//      // block
//      if (first_nonzero_diagonal_entry.node || first_nonzero_diagonal_entry.getValue() != 0)
//        break;
//    }
//  // nothing found on all diagonal
//  // blocks? if so, use 1.0 instead
//  if (!first_nonzero_diagonal_entry.node && first_nonzero_diagonal_entry.getValue() == 0)
//    first_nonzero_diagonal_entry = 1;


//  std::map<types::global_dof_index,double>::const_iterator dof  = boundary_values.begin(),
//                                                           endd = boundary_values.end();
//  const BlockSparsityPattern & sparsity_pattern = matrix.get_sparsity_pattern();

//  // pointer to the mapping between
//  // global and block indices. since
//  // the row and column mappings are
//  // equal, store a pointer on only
//  // one of them
//  const BlockIndices & index_mapping = sparsity_pattern.get_column_indices();

//  // now loop over all boundary dofs
//  for (; dof != endd; ++dof)
//    {
//      Assert (dof->first < n_dofs, ExcInternalError());
//      (void)n_dofs;

//      // get global index and index
//      // in the block in which this
//      // dof is located
//      const types::global_dof_index dof_number = dof->first;
//      const std::pair<unsigned int,types::global_dof_index> block_index = index_mapping.global_to_local (dof_number);

//      // for each boundary dof:

//      // set entries of this line
//      // to zero except for the diagonal
//      // entry. Note that the diagonal
//      // entry is always the first one
//      // in a row for square matrices
//      for (unsigned int block_col=0; block_col<blocks; ++block_col)
//        for (typename SparseMatrix<adouble>::iterator
//             p = (block_col == block_index.first ?
//                  matrix.block(block_index.first,block_col).begin(block_index.second) + 1 :
//                  matrix.block(block_index.first,block_col).begin(block_index.second));
//             p != matrix.block(block_index.first,block_col).end(block_index.second);
//             ++p)
//          p->value() = 0;

//      // set right hand side to
//      // wanted value: if main diagonal
//      // entry nonzero, don't touch it
//      // and scale rhs accordingly. If
//      // zero, take the first main
//      // diagonal entry we can find, or
//      // one if no nonzero main diagonal
//      // element exists. Normally, however,
//      // the main diagonal entry should
//      // not be zero.
//      //
//      // store the new rhs entry to make
//      // the gauss step more efficient
//      adouble new_rhs;
//      if (matrix.block(block_index.first, block_index.first).diag_element(block_index.second).node ||
//          matrix.block(block_index.first, block_index.first).diag_element(block_index.second).getValue() != 0.0)
//        new_rhs = dof->second * matrix.block(block_index.first, block_index.first).diag_element(block_index.second);
//      else
//        {
//          matrix.block(block_index.first, block_index.first).diag_element(block_index.second) = first_nonzero_diagonal_entry;
//          new_rhs = dof->second * first_nonzero_diagonal_entry;
//        }
//      right_hand_side.block(block_index.first)(block_index.second) = new_rhs;


//      // if the user wants to have
//      // the symmetry of the matrix
//      // preserved, and if the
//      // sparsity pattern is
//      // symmetric, then do a Gauss
//      // elimination step with the
//      // present row. this is a
//      // little more complicated for
//      // block matrices.
//      if (eliminate_columns)
//        {
//          // store the only nonzero entry
//          // of this line for the Gauss
//          // elimination step
//          const adouble diagonal_entry = matrix.block(block_index.first,block_index.first).diag_element(block_index.second);

//          // we have to loop over all
//          // rows of the matrix which
//          // have a nonzero entry in
//          // the column which we work
//          // in presently. if the
//          // sparsity pattern is
//          // symmetric, then we can
//          // get the positions of
//          // these rows cheaply by
//          // looking at the nonzero
//          // column numbers of the
//          // present row.
//          //
//          // note that if we check
//          // whether row @p{row} in
//          // block (r,c) is non-zero,
//          // then we have to check
//          // for the existence of
//          // column @p{row} in block
//          // (c,r), i.e. of the
//          // transpose block
//          for (unsigned int block_row=0; block_row<blocks; ++block_row)
//            {
//              // get pointers to the sparsity patterns of this block and of
//              // the transpose one
//              const SparsityPattern &this_sparsity = sparsity_pattern.block (block_row, block_index.first);

//              SparseMatrix<adouble> &this_matrix = matrix.block(block_row, block_index.first);
//              SparseMatrix<adouble> &transpose_matrix = matrix.block(block_index.first, block_row);

//              // traverse the row of the transpose block to find the
//              // interesting rows in the present block.  don't use the
//              // diagonal element of the diagonal block
//              for (typename SparseMatrix<adouble>::iterator
//                   q = (block_index.first == block_row ?
//                        transpose_matrix.begin(block_index.second)+1 :
//                        transpose_matrix.begin(block_index.second));
//                   q != transpose_matrix.end(block_index.second);
//                   ++q)
//                {
//                  // get the number of the column in this row in which a
//                  // nonzero entry is. this is also the row of the transpose
//                  // block which has an entry in the interesting row
//                  const types::global_dof_index row = q->column();

//                  // find the position of element (row,dof_number) in this
//                  // block (not in the transpose one). note that we have to
//                  // take care of special cases with square sub-matrices
//                  bool (*comp)(typename SparseMatrix<adouble>::iterator::value_type p, const unsigned int column)
//                    = &column_less_than<typename SparseMatrix<adouble>::iterator>;

//                  typename SparseMatrix<adouble>::iterator p = this_matrix.end();

//                  if (this_sparsity.n_rows() == this_sparsity.n_cols())
//                    {
//                      if (this_matrix.begin(row)->column() == block_index.second)
//                        p = this_matrix.begin(row);
//                      else
//                        p = Utilities::lower_bound(this_matrix.begin(row)+1,
//                                                   this_matrix.end(row),
//                                                   block_index.second,
//                                                   comp);
//                    }
//                  else
//                    p = Utilities::lower_bound(this_matrix.begin(row),
//                                               this_matrix.end(row),
//                                               block_index.second,
//                                               comp);

//                  // check whether this line has an entry in the
//                  // regarding column (check for ==dof_number and !=
//                  // next_row, since if row==dof_number-1, *p is a
//                  // past-the-end pointer but points to dof_number
//                  // anyway...)
//                  //
//                  // there should be such an entry! we know this because
//                  // we have assumed that the sparsity pattern is
//                  // symmetric and we only walk over those rows for
//                  // which the current row has a column entry
//                  Assert ((p->column() == block_index.second) &&
//                          (p != this_matrix.end(row)),
//                          ExcInternalError());

//                  // correct right hand side
//                  right_hand_side.block(block_row)(row) -= adouble(p->value()) / diagonal_entry * new_rhs;

//                  // set matrix entry to zero
//                  p->value() = 0.0;
//                }
//            }
//        }

//      // preset solution vector
//      solution.block(block_index.first)(block_index.second) = dof->second;
//    }
//}

}

}

#endif
