#ifndef DEAL_II_CELL_CONTEXT_H
#define DEAL_II_CELL_CONTEXT_H

#include <typeinfo>
#include <fstream>
#include <iostream>
#include <iterator>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "dealii_common.h"
#include "../Core/coreimpl.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
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
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_abf.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_bernstein.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace dae
{
namespace fe_solver
{
using namespace dealii;

/******************************************************************
    feCellContextImpl<dim>
*******************************************************************/
template<int dim, typename FE_VALUES>
class feCellContextImpl : public feCellContext<dim>
{
public:
    typedef typename std::map<std::string,  const Function<dim,double>*>  map_String_FunctionPtr;
    typedef typename std::map<std::string,  const Function<dim,adouble>*> map_String_adoubleFunctionPtr;
    typedef typename std::map< std::string, boost::variant<FEValuesExtractors::Scalar, FEValuesExtractors::Vector> > map_String_FEValuesExtractor;

    feCellContextImpl(FE_VALUES&                        fe_values,
                      daeModel*                         model,
                      BlockSparsityPattern&             sparsity_pattern,
                      std::vector<unsigned int>&        local_dof_indices,
                      map_String_FunctionPtr&           mapFunctions,
                      map_String_adoubleFunctionPtr&    mapAdoubleFunctions,
                      map_String_FEValuesExtractor&     mapExtractors):
        m_fe_values(fe_values),
        m_model(model),
        m_sparsity_pattern(sparsity_pattern),
        m_local_dof_indices(local_dof_indices),
        m_mapFunctions(mapFunctions),
        m_mapAdoubleFunctions(mapAdoubleFunctions),
        m_mapExtractors(mapExtractors),
        m_i(-1),
        m_j(-1),
        m_q(-1)
    {
        if(!m_model)
            throw std::runtime_error(std::string("Model not set in feCellContext<dim>"));
    }

public:
    /*
     * Scalar-data functions
     */
    virtual double value(const std::string& variableName,
                         const unsigned int i,
                         const unsigned int q) const
    {
        if(i == -1 || q == -1)
            throw std::runtime_error((boost::format("Invalid index in function phi('%s', %d, %d)") % variableName % i % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function phi('%s', %d, %d)") % variableName % i % q).str());

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call of the function phi() for the non-scalar variable: " + variableName);

        return m_fe_values[*extractorScalar].value(i, q);
    }

    Tensor<1,dim> gradient (const std::string& variableName,
                            const unsigned int i,
                            const unsigned int q) const
    {
        if(i == -1 || q == -1)
            throw std::runtime_error((boost::format("Invalid index in function dphi('%s', %d, %d)") % variableName % i % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function dphi('%s', %d, %d)") % variableName % i % q).str());

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call of the function dphi() for the non-scalar variable: " + variableName);

        return m_fe_values[*extractorScalar].gradient(i, q);
    }

    Tensor<2,dim> hessian (const std::string& variableName,
                           const unsigned int i,
                           const unsigned int q) const
    {
        if(i == -1 || q == -1)
            throw std::runtime_error((boost::format("Invalid index in function d2phi('%s', %d, %d)") % variableName % i % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function d2phi('%s', %d, %d)") % variableName % i % q).str());

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call of the function d2phi() for the non-scalar variable: " + variableName);

        return m_fe_values[*extractorScalar].hessian(i, q);
    }

    /*
     * Vector-data functions
     */
    Tensor<1,dim> vector_value(const std::string& variableName,
                               const unsigned int i,
                               const unsigned int q) const
    {
        if(i == -1 || q == -1)
            throw std::runtime_error((boost::format("Invalid index in function phi_vector('%s', %d, %d)") % variableName % i % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function phi_vector('%s', %d, %d)") % variableName % i % q).str());

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call of the function phi_vector() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].value(i, q);
    }

    Tensor<2,dim> vector_gradient (const std::string& variableName,
                                   const unsigned int i,
                                   const unsigned int q) const
    {
        if(i == -1 || q == -1)
            throw std::runtime_error((boost::format("Invalid index in function dphi_vector('%s', %d, %d)") % variableName % i % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function dphi_vector('%s', %d, %d)") % variableName % i % q).str());

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call of the function dphi_vector() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].gradient(i, q);
    }

    Tensor<3,dim> vector_hessian (const std::string& variableName,
                                  const unsigned int i,
                                  const unsigned int q) const
    {
        if(i == -1 || q == -1)
            throw std::runtime_error((boost::format("Invalid index in function d2phi_vector('%s', %d, %d)") % variableName % i % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function d2phi_vector('%s', %d, %d)") % variableName % i % q).str());

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call of the function d2phi_vector() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].hessian(i, q);
    }

    double divergence(const std::string& variableName,
                      const unsigned int i,
                      const unsigned int q) const
    {
        if(i == -1 || q == -1)
            throw std::runtime_error((boost::format("Invalid index in function div_phi('%s', %d, %d)") % variableName % i % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function div_phi('%s', %d, %d)") % variableName % i % q).str());

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call of the function div_phi() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].divergence(i, q);
    }

    SymmetricTensor<2,dim,double> symmetric_gradient(const std::string& variableName,
                                                     const unsigned int i,
                                                     const unsigned int q) const
    {
        if(i == -1 || q == -1)
            throw std::runtime_error((boost::format("Invalid index in function symmetric_gradient('%s', %d, %d)") % variableName % i % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function symmetric_gradient('%s', %d, %d)") % variableName % i % q).str());

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call of the function symmetric_gradient() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].symmetric_gradient(i, q);
    }

    /*
    Tensor<1,1> curl_2D(const std::string& variableName,
                        const unsigned int i,
                        const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to curl() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].curl<2,2>(i, q);
    }

    Tensor<1,3> curl_3D(const std::string& variableName,
                        const unsigned int i,
                        const unsigned int q) const
    {
        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error("Cannot find variable " + variableName);

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call to curl() for the non-vector variable: " + variableName);

        return m_fe_values[*extractorVector].curl<3,3>(i, q);
    }
    */

    virtual const Point<dim>& quadrature_point(const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function xyz(%d)") % q).str());

        return m_fe_values.quadrature_point(q);
    }

    virtual double JxW(const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function JxW(%d)") % q).str());

        return m_fe_values.JxW(q);
    }

    virtual const Tensor<1,dim>& normal_vector(const unsigned int q) const
    {
        return m_fe_values.normal_vector(q);
    }
    /*
    virtual const Function<dim,double>& function(const std::string& functionName) const
    {
        typename map_String_FunctionPtr::iterator iter = m_mapFunctions.find(functionName);
        if(iter == m_mapFunctions.end())
            throw std::runtime_error(std::string("Cannot find Function<dim> with the name ") + functionName);

        return *(iter->second);
    }

    virtual const Function<dim, adouble>& adouble_function(const std::string& functionName) const
    {
        typename map_String_adoubleFunctionPtr::iterator iter = m_mapAdoubleFunctions.find(functionName);
        if(iter == m_mapAdoubleFunctions.end())
            throw std::runtime_error(std::string("Cannot find Function<dim,adouble> with the name ") + functionName);

        return *(iter->second);
    }
    */

    daeVariable* findVariable(const std::string& variableName, unsigned int li, unsigned int& index_in_block) const
    {
        //pyGILState GIL;

        // a) Find the index within block using the local cell dof index
        int global_dof_index = m_local_dof_indices[li];
        const BlockIndices & index_mapping = m_sparsity_pattern.get_column_indices();
        //             <block_index,  index_in_block>
        std::pair<unsigned int, BlockIndices::size_type> block_index = index_mapping.global_to_local(global_dof_index);
        unsigned int block = block_index.first;
        index_in_block     = block_index.second;

        //printf("dof_number = %d -> (block=%d, block_index = %d) \n", global_dof_index, block_index.first, block_index.second);

        // b) Find variable in the parent model using its name
        daeVariable* variable = NULL;
        unsigned int variable_block = -1;

        std::vector<daeVariable_t*> arrVariables;
        m_model->GetVariables(arrVariables);

        for(int i = 0; i < arrVariables.size(); i++)
        {
            daeVariable* pVariable = dynamic_cast<daeVariable*>(arrVariables[i]);
            if(pVariable && pVariable->GetName() == variableName)
            {
                variable_block = i; //  counter i is also the block counter: each variable forms one block (including the vector dofs)
                variable       = pVariable;
                break;
            }
        }
        if(!variable)
            throw std::runtime_error((boost::format("Invalid DOF name in function dof('%s', %d)") % variableName % li).str());

        if(variable_block != block)
        {
            std::string err = (boost::format("The block index found for the specified index [%d] does not match the block where variable belongs [%d]. "
                                             "Returning an empty adouble object.\n") % block % variable_block).str();
            throw std::runtime_error(err);
        }

        size_t n = variable->GetNumberOfPoints();
        if(index_in_block >= n)
        {
            printf("dof_number = %d -> (block=%d, block_index = %d) \n", global_dof_index, block_index.first, block_index.second);
            throw std::runtime_error((boost::format("DOF '%s' index in function dof('%s', %d) is out of bounds (global[%d] = %d) >= %d")
                                      % variableName % variableName % li % li % index_in_block % n).str());
        }

        return variable;
    }

    virtual adouble dof(const std::string& variableName, const unsigned int li) const
    {
        /* Achtung, Achtung!!
         * This function can get called with the argument li (index) that belongs to some other variable in the FE system.
         * What to do then?
         *  a) Throw an exception (bad idea for dof() can be called within the loop)
         *  b) Return adouble(0,0,false,NULL) that is an empty adouble object.
         *     Here, block_index.first can help detect the variable in that block.
         *     If its name is equal to the requested variableName return variable(index);
         *     otherwise, return an empt adouble().
         */
        if(li == -1)
            throw std::runtime_error((boost::format("Invalid index in function dof('%s', %d)") % variableName % li).str());

        unsigned int index_in_block;
        daeVariable* variable = findVariable(variableName, li, index_in_block);

        /*
        // a) Find the index within block using the local cell dof index
        int global_dof_index = m_local_dof_indices[li];
        const BlockIndices & index_mapping = m_sparsity_pattern.get_column_indices();
        //             <block_index,  index_in_block>
        std::pair<unsigned int, BlockIndices::size_type> block_index = index_mapping.global_to_local(global_dof_index);
        unsigned int block          = block_index.first;
        unsigned int index_in_block = block_index.second;

        //printf("dof_number = %d -> (block=%d, block_index = %d) \n", global_dof_index, block_index.first, block_index.second);

        // b) Find variable in the parent model using its name
        daeVariable* variable = NULL;
        unsigned int variable_block = -1;

        std::vector<daeVariable_t*> arrVariables;
        m_model->GetVariables(arrVariables);

        for(int i = 0; i < arrVariables.size(); i++)
        {
            daeVariable* pVariable = dynamic_cast<daeVariable*>(arrVariables[i]);
            if(pVariable && pVariable->GetName() == variableName)
            {
                variable_block = i; //  counter i is also the block counter: each variable forms one block (including the vector dofs)
                variable       = pVariable;
                break;
            }
        }
        if(!variable)
            throw std::runtime_error((boost::format("Invalid DOF name in function dof('%s', %d)") % variableName % li).str());

        if(variable_block != block)
        {
            printf("The block index found for the specified index [%d] does not match the block where variable belongs [%d]. "
                   "Returning an empty adouble object.\n", block, variable_block);
            return adouble();
        }
        */

        // c) Create adouble

        return (*variable)(index_in_block);
    }

    /*
    virtual Tensor<1,dim,adouble> vector_dof_approximation_old(const std::string& variableName, const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function vector_dof_approximation('%s', %d)") % variableName % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function vector_dof_approximation('%s', %d)") % variableName % q).str());

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call of the function vector_dof_approximation() for the non-vector variable: " + variableName);

        const unsigned int dofs_per_cell = m_fe_values.get_fe().dofs_per_cell;

        Tensor<1,dim,adouble> ad_vector_approximation;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
            Tensor<1,dim,double>  phi_vector_j = m_fe_values[*extractorVector].value(j, q);
            bool non_zero = false;
            for(int x = 0; x < dim; x++)
                if(phi_vector_j[x] != 0)
                {
                    non_zero = true;
                    break;
                }
            // If all items in the tensor are zero skip the addition
            // This will also skip creating adouble for indexes that represent variables
            // other than the requested one (variableName).
            // That creates a wrong variable and could throw an out-of-bounds exception in some cases.
            if(non_zero == false)
                continue;

            adouble dof = this->dof(variableName, j);

            // Bypass Tensor<1,dim,adouble> operator += and do additions only if phi_vector[i] != 0.0
            for(unsigned int i = 0; i < dim; i++)
                if(phi_vector_j[i] != 0.0)
                    ad_vector_approximation[i] += dof * phi_vector_j[i];

            //ad_vector_approximation += (dof * phi_vector_j);
        }

        return ad_vector_approximation;
    }
    */

    virtual daeTensor1DoFSum<dim> vector_dof_approximation(const std::string& variableName, const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function vector_dof_approximation('%s', %d)") % variableName % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function vector_dof_approximation('%s', %d)") % variableName % q).str());

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call of the function vector_dof_approximation() for the non-vector variable: " + variableName);

        const unsigned int dofs_per_cell = m_fe_values.get_fe().dofs_per_cell;

        daeTensor1DoFSum<dim> vectorDoFApproximation;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
            Tensor<1,dim,double>  phi_vector_j = m_fe_values[*extractorVector].value(j, q);
            bool non_zero = false;
            for(int x = 0; x < dim; x++)
                if(phi_vector_j[x] != 0)
                {
                    non_zero = true;
                    break;
                }
            // If all items in the tensor are zero skip the addition
            // This will also skip creating adouble for indexes that represent variables
            // other than the requested one (variableName).
            // That creates a wrong variable and could throw an out-of-bounds exception in some cases.
            if(non_zero == false)
                continue;

            unsigned int index_in_block;
            daeVariable* variable = findVariable(variableName, j, index_in_block);

            vectorDoFApproximation.items.push_back( daeTensorDoFItem<1,dim>(phi_vector_j, variable, index_in_block) );
        }
        return vectorDoFApproximation;
    }

    virtual daeTensor2DoFSum<dim> vector_dof_gradient_approximation(const std::string& variableName, const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function vector_dof_gradient_approximation('%s', %d)") % variableName % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function vector_dof_gradient_approximation('%s', %d)") % variableName % q).str());

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call of the function vector_dof_gradient_approximation() for the non-vector variable: " + variableName);

        const unsigned int dofs_per_cell = m_fe_values.get_fe().dofs_per_cell;

        daeTensor2DoFSum<dim> vectorDoFGradientApproximation;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
            Tensor<2,dim,double>  dphi_vector_j = m_fe_values[*extractorVector].gradient(j, q);
            bool non_zero = false;
            for(int x = 0; x < dim; x++)
                for(int y = 0; y < dim; y++)
                    if(dphi_vector_j[x][y] != 0)
                    {
                        non_zero = true;
                        break;
                    }
            // If all items in the tensor are zero skip the addition
            // This will also skip creating adouble for indexes that represent variables
            // other than the requested one (variableName).
            // That creates a wrong variable and could throw an out-of-bounds exception in some cases.
            if(non_zero == false)
                continue;

            unsigned int index_in_block;
            daeVariable* variable = findVariable(variableName, j, index_in_block);

            vectorDoFGradientApproximation.items.push_back( daeTensorDoFItem<2,dim>(dphi_vector_j, variable, index_in_block) );
        }
        return vectorDoFGradientApproximation;
    }
/*
    virtual Tensor<2,dim,adouble> vector_dof_gradient_approximation_old(const std::string& variableName, const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function vector_dof_gradient_approximation('%s', %d)") % variableName % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function vector_dof_gradient_approximation('%s', %d)") % variableName % q).str());

        FEValuesExtractors::Vector* extractorVector = boost::get<FEValuesExtractors::Vector>(&iter->second);
        if(!extractorVector)
            throw std::runtime_error("Invalid call of the function vector_dof_gradient_approximation() for the non-vector variable: " + variableName);

        const unsigned int dofs_per_cell = m_fe_values.get_fe().dofs_per_cell;

        Tensor<2,dim,adouble> ad_vector_approximation;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
            Tensor<2,dim,double>  dphi_vector_j = m_fe_values[*extractorVector].gradient(j, q);
            bool non_zero = false;
            for(int x = 0; x < dim; x++)
                for(int y = 0; y < dim; y++)
                    if(dphi_vector_j[x][y] != 0)
                    {
                        non_zero = true;
                        break;
                    }
            // If all items in the tensor are zero skip the addition.
            // This will also skip creating adouble for indexes that represent variables
            // other than the requested one (variableName).
            // That creates a wrong variable and could throw an out-of-bounds exception in some cases.
            if(non_zero == false)
                continue;

            adouble dof = this->dof(variableName, j);

            ad_vector_approximation += (dof * dphi_vector_j);
        }
        return ad_vector_approximation;
    }

    virtual adouble dof_approximation_old(const std::string& variableName, const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function dof_approximation('%s', %d)") % variableName % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function dof_approximation('%s', %d)") % variableName % q).str());

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call of the function dof_approximation() for the non-scalar variable: " + variableName);

        const unsigned int dofs_per_cell = m_fe_values.get_fe().dofs_per_cell;

        adouble ad_approximation;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
            double phi_j = m_fe_values[*extractorScalar].value(j, q);
            // If it is equal to zero skip the addition.
            // This will also skip creating adouble for indexes that represent variables
            // other than the requested one (variableName).
            // That creates a wrong variable and could throw an out-of-bounds exception in some cases.
            if(phi_j == 0)
                continue;

            adouble dof = this->dof(variableName, j);

            ad_approximation += (dof * phi_j);
        }

        return ad_approximation;
    }
*/
    virtual adouble dof_approximation(const std::string& variableName, const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function dof_approximation('%s', %d)") % variableName % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function dof_approximation('%s', %d)") % variableName % q).str());

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call of the function dof_approximation() for the non-scalar variable: " + variableName);

        const unsigned int dofs_per_cell = m_fe_values.get_fe().dofs_per_cell;

        adouble dof_approximation;
        adFloatCoefficientVariableSumNode* node = new adFloatCoefficientVariableSumNode();
        dof_approximation.setGatherInfo(true);
        dof_approximation.node.reset(node);

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
            double phi_j = m_fe_values[*extractorScalar].value(j, q);
            // If it is equal to zero skip the addition.
            if(phi_j == 0)
                continue;

            unsigned int index_in_block;
            daeVariable* variable = findVariable(variableName, j, index_in_block);

            node->AddItem(phi_j, variable, index_in_block);
        }

        return dof_approximation;
    }

    virtual daeTensor1DoFSum<dim> dof_gradient_approximation(const std::string& variableName, const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function dof_gradient_approximation('%s', %d)") % variableName % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function dof_gradient_approximation('%s', %d)") % variableName % q).str());

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call of the function dof_gradient_approximation() for the non-scalar variable: " + variableName);

        const unsigned int dofs_per_cell = m_fe_values.get_fe().dofs_per_cell;

        daeTensor1DoFSum<dim> dofGradientApproximation;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
            Tensor<1,dim,double> dphi_j = m_fe_values[*extractorScalar].gradient(j, q);
            bool non_zero = false;
            for(int x = 0; x < dim; x++)
                if(dphi_j[x] != 0)
                {
                    non_zero = true;
                    break;
                }
            // If all items in the tensor are zero skip the addition
            // This will also skip creating adouble for indexes that represent variables
            // other than the requested one (variableName).
            // That creates a wrong variable and could throw an out-of-bounds exception in some cases.
            if(non_zero == false)
                continue;

            unsigned int index_in_block;
            daeVariable* variable = findVariable(variableName, j, index_in_block);

            dofGradientApproximation.items.push_back( daeTensorDoFItem<1,dim>(dphi_j, variable, index_in_block) );
        }

        return dofGradientApproximation;
    }

    virtual daeTensor2DoFSum<dim> dof_hessian_approximation(const std::string& variableName, const unsigned int q) const
    {
        if(q == -1)
            throw std::runtime_error((boost::format("Invalid index in function dof_hessian_approximation('%s', %d)") % variableName % q).str());

        typename map_String_FEValuesExtractor::iterator iter = m_mapExtractors.find(variableName);
        if(iter == m_mapExtractors.end())
            throw std::runtime_error((boost::format("Invalid DOF name in function dof_hessian_approximation('%s', %d)") % variableName % q).str());

        FEValuesExtractors::Scalar* extractorScalar = boost::get<FEValuesExtractors::Scalar>(&iter->second);
        if(!extractorScalar)
            throw std::runtime_error("Invalid call of the function dof_hessian_approximation() for the non-scalar variable: " + variableName);

        const unsigned int dofs_per_cell = m_fe_values.get_fe().dofs_per_cell;

        daeTensor2DoFSum<dim> dofHessianApproximation;
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
            Tensor<2,dim,double> d2phi_j = m_fe_values[*extractorScalar].hessian(j, q);
            bool non_zero = false;
            for(int x = 0; x < dim; x++)
                for(int y = 0; y < dim; y++)
                    if(d2phi_j[x][y] != 0)
                    {
                        non_zero = true;
                        break;
                    }
            // If all items in the tensor are zero skip the addition
            // This will also skip creating adouble for indexes that represent variables
            // other than the requested one (variableName).
            // That creates a wrong variable and could throw an out-of-bounds exception in some cases.
            if(non_zero == false)
                continue;

            unsigned int index_in_block;
            daeVariable* variable = findVariable(variableName, j, index_in_block);

            dofHessianApproximation.items.push_back( daeTensorDoFItem<2,dim>(d2phi_j, variable, index_in_block) );
        }

        return dofHessianApproximation;
    }

    virtual unsigned int q() const
    {
        return m_q;
    }

    virtual unsigned int i() const
    {
        return m_i;
    }

    virtual unsigned int j() const
    {
        return m_j;
    }

    virtual unsigned int component(unsigned int index) const
    {
        const FiniteElement<dim> &fe = m_fe_values.get_fe();
        return fe.system_to_component_index(index).first;
    }

    virtual double cell_diameter() const
    {
        return m_fe_values.get_cell()->diameter();
    }

public:
    FE_VALUES&                      m_fe_values;
    daeModel*                       m_model;
    BlockSparsityPattern&           m_sparsity_pattern;
    std::vector<unsigned int>&      m_local_dof_indices;
    map_String_FunctionPtr&         m_mapFunctions;
    map_String_adoubleFunctionPtr&  m_mapAdoubleFunctions;
    map_String_FEValuesExtractor&   m_mapExtractors;
    unsigned int                    m_i;
    int                             m_j;
    int                             m_q;
};


}
}

#endif
