#ifndef DEAL_II_ITERATORS_H
#define DEAL_II_ITERATORS_H

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

template<int dim>
class dealiiCell;

template<int dim>
class dealiiFace
{
public:
    dealiiFace():
        m_cell(NULL), m_face_no(-1)
    {

    }

    dealiiFace(dealiiCell<dim>* cell, unsigned int face_no):
        m_cell(cell), m_face_no(face_no)
    {

    }

    FEFaceValues<dim>& get_fe_values()
    {
        if(!m_cell)
            daeDeclareAndThrowException(exInvalidPointer);

        // Always reinit fe_face_values with the current iterator before returning
        m_cell->fe_face_values.reinit (m_cell->current_active_cell_iterator, m_face_no);

        return m_cell->fe_face_values;
    }

    unsigned int get_n_q_points()
    {
        if(!m_cell)
            daeDeclareAndThrowException(exInvalidPointer);
        return m_cell->n_face_q_points;
    }

    bool get_at_boundary()
    {
        if(!m_cell)
            daeDeclareAndThrowException(exInvalidPointer);
        return m_cell->current_active_cell_iterator->face(m_face_no)->at_boundary();
    }

    unsigned int get_boundary_id()
    {
        if(!m_cell)
            daeDeclareAndThrowException(exInvalidPointer);
        return m_cell->current_active_cell_iterator->face(m_face_no)->boundary_indicator();
    }

public:
    dealiiCell<dim>*  m_cell;
    unsigned int      m_face_no;
};
typedef dealiiFace<1> dealiiFace_1D;
typedef dealiiFace<2> dealiiFace_2D;
typedef dealiiFace<3> dealiiFace_3D;

template<int dim>
class dealiiCellIterator : public std::iterator<std::forward_iterator_tag, dealiiCell<dim> >
{
public:
    typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;

    dealiiCellIterator(dealiiCell<dim>& cell, active_cell_iterator iter):
        m_cell(cell),
        current_active_cell_iterator(iter)
    {
    }

    // Prefix ++operator
    dealiiCellIterator& operator++()
    {
        std::cout << "++operator" << std::endl;
        ++current_active_cell_iterator;

        m_cell.cell_matrix    = 0;
        m_cell.cell_matrix_dt = 0;
        m_cell.cell_rhs       = 0;

        m_cell.fe_values.reinit(current_active_cell_iterator);
        current_active_cell_iterator->get_dof_indices(m_cell.local_dof_indices);

        // Save current_active_cell_iterator (to be used in dealiiFace to reinit fe_face_values)
        m_cell.current_active_cell_iterator = current_active_cell_iterator;

        return *this;
    }

    // Postfix iterator++
    dealiiCellIterator operator++(int)
    {
        std::cout << "operator++" << std::endl;

        dealiiCellIterator tmp(*this);

        m_cell.cell_matrix    = 0;
        m_cell.cell_matrix_dt = 0;
        m_cell.cell_rhs       = 0;

        m_cell.fe_values.reinit(current_active_cell_iterator);
        current_active_cell_iterator->get_dof_indices(m_cell.local_dof_indices);

        // Save current_active_cell_iterator (to be used in dealiiFace to reinit fe_face_values)
        m_cell.current_active_cell_iterator = current_active_cell_iterator;

        ++current_active_cell_iterator;

        return tmp;
    }

    dealiiCell<dim>& operator*()
    {
        return m_cell;
    }

    bool operator==(const dealiiCellIterator& other) const
    {
        return current_active_cell_iterator == other.current_active_cell_iterator;
    }

    bool operator!=(const dealiiCellIterator& other) const
    {
        return current_active_cell_iterator != other.current_active_cell_iterator;
    }

    size_t distance(const dealiiCellIterator& first, const dealiiCellIterator& last)
    {
        return last.current_active_cell_iterator - first.current_active_cell_iterator;
    }

public:
    active_cell_iterator  current_active_cell_iterator;
    dealiiCell<dim>&      m_cell;
};
typedef dealiiCellIterator<1> dealiiCellIterator_1D;
typedef dealiiCellIterator<2> dealiiCellIterator_2D;
typedef dealiiCellIterator<3> dealiiCellIterator_3D;


template<int dim>
class dealiiCell
{
public:
    typedef dealiiCellIterator<dim> iterator;
    typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
    typedef typename std::vector< dealiiFace<dim> >::iterator face_iterator;

    dealiiCell(FiniteElement<dim>*      fe,
               DoFHandler<dim>&         dof_handler_,
               SparseMatrix<double>&    sys_matrix,
               SparseMatrix<double>&    sys_matrix_dt,
               Vector<double>&          sys_rhs,
               Vector<double>&          sol):
        m_fe(fe),
        dof_handler(dof_handler_),
        quadrature_formula(3),
        face_quadrature_formula(3),
        dofs_per_cell(fe->dofs_per_cell),
        n_q_points(quadrature_formula.size()),
        faces_per_cell(GeometryInfo<dim>::faces_per_cell),
        n_face_q_points(face_quadrature_formula.size()),
        cell_matrix(dofs_per_cell, dofs_per_cell),
        cell_matrix_dt(dofs_per_cell, dofs_per_cell),
        cell_rhs(dofs_per_cell),
        local_dof_indices (dofs_per_cell),
        fe_values(*fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values),
        fe_face_values(*fe, face_quadrature_formula, update_values | update_quadrature_points | update_normal_vectors | update_JxW_values),
        system_matrix(sys_matrix),
        system_matrix_dt(sys_matrix_dt),
        system_rhs(sys_rhs),
        solution(sol)
    {
        current_active_cell_iterator = dof_handler.begin_active();
        end_active_cell_iterator     = dof_handler.end();

        m_faces.resize(faces_per_cell);
        for(unsigned int face_no = 0; face_no < faces_per_cell; face_no++)
            m_faces[face_no] = dealiiFace<dim>(this, face_no);
    }

    iterator begin()
    {
        std::cout << "dealiiCell::begin" << std::endl;
        current_active_cell_iterator = dof_handler.begin_active();

        // Initialize the data for the first iteration
        // They will be re-initialized again after every call to dealiiCellIterator::operator++
//        cell_matrix    = 0;
//        cell_matrix_dt = 0;
//        cell_rhs       = 0;

//        fe_values.reinit(current_active_cell_iterator);
//        current_active_cell_iterator->get_dof_indices(local_dof_indices);

        return dealiiCellIterator<dim>(*this, current_active_cell_iterator);
    }

    iterator end()
    {
        std::cout << "dealiiCell::end" << std::endl;
        return dealiiCellIterator<dim>(*this, end_active_cell_iterator);
    }

    SparseMatrix<double>& get_system_matrix()
    {
        return system_matrix;
    }

    SparseMatrix<double>& get_system_matrix_dt()
    {
        return system_matrix_dt;
    }

    Vector<double>& get_system_rhs()
    {
        return system_rhs;
    }

    Vector<double>& get_solution()
    {
        return solution;
    }

    face_iterator begin_faces()
    {
        std::cout << "dealiiCell::begin_faces" << std::endl;
        return m_faces.begin();
    }

    face_iterator end_faces()
    {
        std::cout << "dealiiCell::end_faces" << std::endl;
        return m_faces.end();
    }

public:
    FiniteElement<dim>*         m_fe;
    DoFHandler<dim>&            dof_handler;
    QGauss<dim>                 quadrature_formula;
    QGauss<dim-1>               face_quadrature_formula;
    const unsigned int          dofs_per_cell;
    const unsigned int          n_q_points;
    const unsigned int          faces_per_cell;
    const unsigned int          n_face_q_points;
    FullMatrix<double>          cell_matrix;
    FullMatrix<double>          cell_matrix_dt;
    Vector<double>              cell_rhs;
    std::vector<unsigned int>   local_dof_indices;
    FEValues<dim>               fe_values;
    FEFaceValues<dim>           fe_face_values;

    SparseMatrix<double>&       system_matrix;
    SparseMatrix<double>&       system_matrix_dt;
    Vector<double>&             system_rhs;
    Vector<double>&             solution;

    std::vector< dealiiFace<dim> > m_faces;
    active_cell_iterator        current_active_cell_iterator;
    active_cell_iterator        end_active_cell_iterator;
};
typedef dealiiCell<1> dealiiCell_1D;
typedef dealiiCell<2> dealiiCell_2D;
typedef dealiiCell<3> dealiiCell_3D;

}
}

#endif
