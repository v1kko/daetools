#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <string>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "../dae_develop.h"

#include "deal_ii.h"
using namespace Step7;

// Sets all elements to zero
void clearMatrix(FullMatrix<double>& m);


typedef Triangulation<1> Triangulation_1D;
typedef Triangulation<2> Triangulation_2D;
typedef Triangulation<3> Triangulation_3D;

typedef DoFHandler<1> dof_handler_1D;
typedef DoFHandler<2> dof_handler_2D;
typedef DoFHandler<3> dof_handler_3D;

typedef FiniteElement<1> FiniteElement_1D;
typedef FiniteElement<2> FiniteElement_2D;
typedef FiniteElement<3> FiniteElement_3D;

//ConstraintMatrix                        hanging_node_constraints;
//SparsityPattern                         sparsity_pattern;

//SparseMatrix<double>                    system_matrix;
//Vector<double>                          solution;
//Vector<double>                          system_rhs;

//class daeDEAL_II_t 
//{
//public:
//    virtual ~daeDEAL_II_t (){}

//    virtual void setup_system () {}
//    virtual void assemble_system () {}
//    virtual void solve () {}
//    virtual void refine_grid () {}
//    virtual void process_solution() {}
//};

template <int dim>
class daeDEAL_II : public HelmholtzProblem<dim>,
                   public boost::python::wrapper< HelmholtzProblem<dim> >
{    
public:
    daeDEAL_II(unsigned int p, RefinementMode ref_mode) : HelmholtzProblem<dim>(p, ref_mode)
    {
    }

    ~daeDEAL_II() {}
    
    void setup_system(void)
	{
        if(boost::python::override f = this->get_override("setup_system"))
            f();
		else
			this->HelmholtzProblem<dim>::setup_system();
	}    
    void def_setup_system(void)
    {
        this->HelmholtzProblem<dim>::setup_system();
    }
    
    void assemble_system(void)
	{
        if(boost::python::override f = this->get_override("assemble_system"))
            f();
		else
			this->HelmholtzProblem<dim>::assemble_system();
	}    
    void def_assemble_system(void)
    {
        this->HelmholtzProblem<dim>::assemble_system();
    }
    
    void solve(void)
	{
        if(boost::python::override f = this->get_override("solve"))
            f();
		else
			this->HelmholtzProblem<dim>::solve();
	}    
    void def_solve(void)
    {
        this->HelmholtzProblem<dim>::solve();
    }
    
    void refine_grid(void)
	{
        if(boost::python::override f = this->get_override("refine_grid"))
            f();
		else
			this->HelmholtzProblem<dim>::refine_grid();
	}    
    void def_refine_grid(void)
    {
        this->HelmholtzProblem<dim>::refine_grid();
    }
    
    void process_solution(const unsigned int cycle)
	{
        if(boost::python::override f = this->get_override("process_solution"))
            f(cycle);
		else
			this->HelmholtzProblem<dim>::process_solution(cycle);
	}    
    void def_process_solution(const unsigned int cycle)
    {
        this->HelmholtzProblem<dim>::process_solution(cycle);
    }
};

typedef daeDEAL_II<1> daeDEAL_II_1D;
typedef daeDEAL_II<2> daeDEAL_II_2D;
typedef daeDEAL_II<3> daeDEAL_II_3D;

#endif
