/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "cs_simulator.h"
#include "auxiliary.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include "idas_la_functions.h"

#include <string>
#include <iostream>
#define DAE_MAJOR 1
#define DAE_MINOR 8
#define DAE_BUILD 0
#define daeSuperLU
#include "../LA_Trilinos_Amesos/base_solvers.h"

namespace cs_dae_simulator
{
class daeLASolverData : public dae::solver::daeBlockOfEquations_t,
                        public cs::csMatrixAccess_t
{
public:
    daeLASolverData(size_t numVars, cs::csDAEModel_t* mod) :
        numberOfVariables(numVars), model(mod)
    {
        dae::solver::daeLASolver_t* lasolver = NULL;

        daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();
        std::string solverName = cfg.GetString("LinearSolver.Name", "Not specified");

        if(solverName == "Amesos_Superlu" || solverName == "Amesos_Umfpack" || solverName == "Amesos_Klu")
        {
            lasolver = dae::solver::daeCreateTrilinosSolver(solverName, "");
            m_pLASolver.reset(lasolver);

            printf("Processing linear solver parameters from '%s' ...\n", cfg.configFile.c_str());
            SetParameters(lasolver, cfg.pt.get_child("LinearSolver.Parameters"));
        }
        else if(solverName == "AztecOO")
        {
            std::string preconditionerLibrary = cfg.GetString("LinearSolver.Preconditioner.Library", "Not specified");
            std::string preconditionerName    = cfg.GetString("LinearSolver.Preconditioner.Name", "Not specified");

            if(preconditionerLibrary == "AztecOO") // Native AztecOO preconditioners
            {
                lasolver = dae::solver::daeCreateTrilinosSolver("AztecOO", preconditionerName);
                m_pLASolver.reset(lasolver);
            }
            else if(preconditionerLibrary == "Ifpack")
            {
                lasolver = dae::solver::daeCreateTrilinosSolver("AztecOO_Ifpack", preconditionerName);
                m_pLASolver.reset(lasolver);
            }
            else if(preconditionerLibrary == "AztecOO_ML")
            {
                lasolver = dae::solver::daeCreateTrilinosSolver("AztecOO_ML", preconditionerName);
                m_pLASolver.reset(lasolver);
            }
            else
            {
                daeThrowException("Unsupported preconditioner specified: " + preconditionerName);
            }

            printf("Processing linear solver parameters from '%s' ...\n", cfg.configFile.c_str());
            SetParameters(lasolver, cfg.pt.get_child("LinearSolver.Parameters"));

            printf("Processing preconditioner parameters from '%s' ...\n", cfg.configFile.c_str());
            SetParameters(lasolver, cfg.pt.get_child("LinearSolver.Preconditioner.Parameters"));
        }
        else
        {
            daeThrowException("Unsupported linear solver specified: " + solverName);
        }

        model->GetDAESystemStructure(N, NNZ, IA, JA);
        lasolver->Create(N, NNZ, this);

        pmatJacobian = NULL;
    }

    virtual int CalcNonZeroElements()
    {
        return NNZ;
    }

    void SetParameters(dae::solver::daeLASolver_t* lasolver, boost::property_tree::ptree& pt)
    {
        BOOST_FOREACH(boost::property_tree::ptree::value_type& pt_child, pt)
        {
            if(pt_child.second.size() == 0) // It is a leaf
            {
                std::cout << "  Set parameter: " << pt_child.first << " = " << pt_child.second.data() << " (";

                bool bValue;
                int iValue;
                double dValue;
                std::string sValue;

                std::string data = pt_child.second.data();
                if(data == "true" || data == "True")
                {
                    std::cout << "bool)" << std::endl;
                    lasolver->SetOption_bool(pt_child.first, true);
                }
                else if(data == "false" || data == "False")
                {
                    std::cout << "bool)" << std::endl;
                    lasolver->SetOption_bool(pt_child.first, true);
                }
                else if(boost::conversion::try_lexical_convert<int>(data, iValue))
                {
                    std::cout << "integer)" << std::endl;
                    lasolver->SetOption_int(pt_child.first, iValue);
                }
                else if(boost::conversion::try_lexical_convert<double>(data, dValue))
                {
                    std::cout << "float)" << std::endl;
                    lasolver->SetOption_float(pt_child.first, dValue);
                }
                else if(boost::conversion::try_lexical_convert<bool>(data, bValue))
                {
                    std::cout << "bool)" << std::endl;
                    lasolver->SetOption_bool(pt_child.first, bValue);
                }
                else
                {
                    std::cout << "string)" << std::endl;
                    sValue = data;
                    lasolver->SetOption_string(pt_child.first, sValue);
                }

            }
        }
    }

    virtual void FillSparseMatrix(dae::daeSparseMatrix<real_t>* pmatrix)
    {
        if(numberOfVariables != N)
            throw std::runtime_error("");
        if(numberOfVariables+1 != IA.size())
            throw std::runtime_error("");
        if(JA.size() != NNZ)
            throw std::runtime_error("");

        std::map<size_t, size_t> mapIndexes;
        for(int row = 0; row < numberOfVariables; row++)
        {
            int colStart = IA[row];
            int colEnd   = IA[row+1];
            mapIndexes.clear();
            for(int col = colStart; col < colEnd; col++)
            {
                size_t bi = JA[col];
                mapIndexes[bi] = bi;
            }
            pmatrix->AddRow(mapIndexes);
            /*
            printf("row %d\n", row);
            for(std::map<size_t, size_t>::iterator it = mapIndexes.begin(); it != mapIndexes.end(); it++)
                printf("{%d,%d} ", it->first, it->second);
            printf("\n");
            */
        }
    }

    virtual void CalculateJacobian(real_t                   time,
                                   real_t                   inverseTimeStep,
                                   dae::daeArray<real_t>&	arrValues,
                                   dae::daeArray<real_t>&	arrResiduals,
                                   dae::daeArray<real_t>&	arrTimeDerivatives,
                                   dae::daeMatrix<real_t>&	matJacobian)
    {
        real_t* values          = arrValues.Data();
        real_t* timeDerivatives = arrTimeDerivatives.Data();
        real_t* residuals       = arrResiduals.Data();

        // Calling SetAndSynchroniseData is not required here since it has previously been called by residuals function.

        pmatJacobian = &matJacobian;

        model->EvaluateJacobian(time, inverseTimeStep, this);

        pmatJacobian = NULL;
    }

    void SetItem(size_t row, size_t col, real_t value)
    {
        if(!pmatJacobian)
            throw std::runtime_error("daeLASolverData: matrix pointer not set");
        pmatJacobian->SetItem(row, col, value);
    }

public:
    dae::daeMatrix<real_t>*                          pmatJacobian;
    boost::shared_ptr<dae::solver::daeLASolver_t>    m_pLASolver;
    int                                              N;
    int                                              NNZ;
    std::vector<int>                                 IA;
    std::vector<int>                                 JA;
    size_t                                           numberOfVariables;
    cs::csDAEModel_t*                                model;
};

daeLASolver_t::daeLASolver_t()
{
}

daeLASolver_t::~daeLASolver_t()
{
}

int daeLASolver_t::Initialize(cs::csDAEModel_t* model, size_t numberOfVariables)
{
    daeLASolverData* la_data = new daeLASolverData(numberOfVariables, model);
    data = la_data;
    la_data->m_pLASolver->Init();
    return 0;
}

int daeLASolver_t::Setup(real_t  time,
                         real_t  inverseTimeStep,
                         real_t* values,
                         real_t* timeDerivatives,
                         real_t* residuals)
{
    daeLASolverData* la_data = (daeLASolverData*)this->data;
    if(!la_data)
        return IDA_MEM_NULL;
    if(!la_data->m_pLASolver)
        return IDA_MEM_NULL;

    int ret = la_data->m_pLASolver->Setup(time,
                                          inverseTimeStep,
                                          values,
                                          timeDerivatives,
                                          residuals);
    return ret;
}

int daeLASolver_t::Solve(real_t  time,
                         real_t  inverseTimeStep,
                         real_t  cjratio,
                         real_t* b,
                         real_t* weight,
                         real_t* values,
                         real_t* timeDerivatives,
                         real_t* residuals)
{
    daeLASolverData* la_data = (daeLASolverData*)this->data;
    if(!la_data)
        return IDA_MEM_NULL;
    if(!la_data->m_pLASolver)
        return IDA_MEM_NULL;

    int ret = la_data->m_pLASolver->Solve(time,
                                          inverseTimeStep,
                                          cjratio,
                                          b,
                                          weight,
                                          values,
                                          timeDerivatives,
                                          residuals);
    return ret;
}

int daeLASolver_t::Free()
{
    daeLASolverData* la_data = (daeLASolverData*)this->data;
    if(!la_data)
        return IDA_MEM_NULL;
    if(!la_data->m_pLASolver)
        return IDA_MEM_NULL;

    int ret = la_data->m_pLASolver->Free();
    return ret;
}

}


int init_la(IDAMem ida_mem)
{
    cs_dae_simulator::daeLASolver_t* pLASolver = (cs_dae_simulator::daeLASolver_t*)ida_mem->ida_lmem;
    if(!pLASolver)
        return IDA_MEM_NULL;
    return IDA_SUCCESS;
}

int setup_la(IDAMem	    ida_mem,
             N_Vector	vectorVariables,
             N_Vector	vectorTimeDerivatives,
             N_Vector	vectorResiduals,
             N_Vector	vectorTemp1,
             N_Vector	vectorTemp2,
             N_Vector	vectorTemp3)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.LASetup);

    cs_dae_simulator::daeLASolver_t* pLASolver = (cs_dae_simulator::daeLASolver_t*)ida_mem->ida_lmem;
    if(!pLASolver)
        return IDA_MEM_NULL;

    realtype *pdValues, *pdTimeDerivatives, *pdResiduals;

    real_t time            = ida_mem->ida_tn;
    real_t inverseTimeStep = ida_mem->ida_cj;

    pdValues			= NV_DATA_P(vectorVariables);
    pdTimeDerivatives	= NV_DATA_P(vectorTimeDerivatives);
    pdResiduals			= NV_DATA_P(vectorResiduals);

    int ret = pLASolver->Setup(time,
                               inverseTimeStep,
                               pdValues,
                               pdTimeDerivatives,
                               pdResiduals);

    if(ret < 0)
        return IDA_LSETUP_FAIL;
    return IDA_SUCCESS;
}

int solve_la(IDAMem	  ida_mem,
             N_Vector vectorB,
             N_Vector vectorWeight,
             N_Vector vectorVariables,
             N_Vector vectorTimeDerivatives,
             N_Vector vectorResiduals)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.LASolve);

    cs_dae_simulator::daeLASolver_t* pLASolver = (cs_dae_simulator::daeLASolver_t*)ida_mem->ida_lmem;
    if(!pLASolver)
        return IDA_MEM_NULL;

    realtype *pdValues, *pdTimeDerivatives, *pdResiduals, *pdWeight, *pdB;

    real_t time            = ida_mem->ida_tn;
    real_t inverseTimeStep = ida_mem->ida_cj;
    real_t cjratio         = ida_mem->ida_cjratio;

    pdWeight			= NV_DATA_P(vectorWeight);
    pdB      			= NV_DATA_P(vectorB);
    pdValues			= NV_DATA_P(vectorVariables);
    pdTimeDerivatives	= NV_DATA_P(vectorTimeDerivatives);
    pdResiduals			= NV_DATA_P(vectorResiduals);

    int ret = pLASolver->Solve(time,
                               inverseTimeStep,
                               cjratio,
                               pdB,
                               pdWeight,
                               pdValues,
                               pdTimeDerivatives,
                               pdResiduals);
    if(ret < 0)
        return IDA_LSOLVE_FAIL;
    return IDA_SUCCESS;
}

int free_la(IDAMem ida_mem)
{
    cs_dae_simulator::daeLASolver_t* pLASolver = (cs_dae_simulator::daeLASolver_t*)ida_mem->ida_lmem;
    if(!pLASolver)
        return IDA_MEM_NULL;

    pLASolver->Free();

    // It is the responsibility of the user to delete LA solver pointer!!
    ida_mem->ida_lmem = NULL;

    return IDA_SUCCESS;
}

