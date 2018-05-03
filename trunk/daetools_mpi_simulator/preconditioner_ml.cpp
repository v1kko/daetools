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
#include "typedefs.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "idas_la_functions.h"

#include <string>
#include <iostream>
#include <vector>
#define DAE_MAJOR 1
#define DAE_MINOR 8
#define DAE_BUILD 0
#define daeSuperLU
#include "../LA_Trilinos_Amesos/trilinos_amesos_la_solver.h"

namespace daetools_mpi
{
class daePreconditionerData_ML : public daeMatrixAccess_t
{
public:
    daePreconditionerData_ML(size_t numVars, daetools_mpi::daeModel_t* mod) :
        numberOfVariables(numVars), model(mod)
    {
        m_map.reset(new Epetra_Map((int)numberOfVariables, 0, m_Comm));
        m_matEPETRA.reset(new Epetra_CrsMatrix(Copy, *m_map, 0));

        m_matJacobian.InitMatrix(numberOfVariables, m_matEPETRA.get());
        m_matJacobian.ResetCounters();
        model->GetDAESystemStructure(N, NNZ, IA, JA);
        FillSparseMatrix(&m_matJacobian);
        m_matJacobian.Sort(); // The function Sort will call FillComplete on Epetra matrix (required for Ifpack constructor).

        daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();
        std::string strPreconditionerName = cfg.GetString("LinearSolver.Preconditioner.Name", "SA");

        if(strPreconditionerName != "SA"       &&
           strPreconditionerName != "DD"       &&
           strPreconditionerName != "DD-ML"    &&
           strPreconditionerName != "DD-ML-LU" &&
           strPreconditionerName != "maxwell"  &&
           strPreconditionerName != "NSSA")
        {
            throw std::runtime_error("Invalid preconditioner: " + strPreconditionerName);
        }

        Teuchos::ParameterList parameters;
        ML_Epetra::SetDefaults(strPreconditionerName, parameters);
        parameters.set("reuse: enable", true);
        printf("Default ML parameters:\n");
        parameters.print(std::cout, 2, true, true);
        printf("\n");

        // Iterate over the default list of parameters and update them (if set in the simulation options).
        printf("Processing ML parameters from '%s' ...\n", cfg.configFile.c_str());
        for(Teuchos::ParameterList::ConstIterator it = parameters.begin(); it != parameters.end(); it++)
        {
            std::string             paramName = it->first;
            Teuchos::ParameterEntry pe        = it->second;
            Teuchos::any            anyValue  = pe.getAny();

            std::string paramPath = "LinearSolver.Preconditioner.Parameters." + paramName;
            if(cfg.HasKey(paramPath))
            {
                if(anyValue.type() == typeid(int))
                {
                    int value = cfg.GetInteger(paramPath);
                    parameters.set(paramName, value);
                    printf("  Found parameter: %s = %d\n", paramName.c_str(), value);
                }
                else if(anyValue.type() == typeid(double))
                {
                    double value = cfg.GetFloat(paramPath);
                    parameters.set(paramName, value);
                    printf("  Found parameter: %s = %f\n", paramName.c_str(), value);
                }
                else if(anyValue.type() == typeid(bool))
                {
                    bool value = cfg.GetBoolean(paramPath);
                    parameters.set(paramName, value);
                    printf("  Found parameter: %s = %d\n", paramName.c_str(), value);
                }
                else if(anyValue.type() == typeid(std::string))
                {
                    std::string value = cfg.GetString(paramPath);
                    parameters.set(paramName, value);
                    printf("  Found parameter: %s = %s\n", paramName.c_str(), value.c_str());
                }
            }
        }
        printf("\n");

        ML_Epetra::MultiLevelPreconditioner* preconditioner = new ML_Epetra::MultiLevelPreconditioner(*m_matEPETRA.get(), parameters, false);
        if(preconditioner == NULL)
            throw std::runtime_error("Failed to create ML preconditioner " + strPreconditionerName);

        m_pPreconditionerML.reset(preconditioner);

        printf("Proceed with ML parameters:\n");
        parameters.print(std::cout, 2, true, true);
        printf("\n");
    }

    void FillSparseMatrix(dae::daeSparseMatrix<real_t>* pmatrix)
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
        }
    }

    void SetItem(size_t row, size_t col, real_t value)
    {
        m_matJacobian.SetItem(row, col, value);
    }

public:
    int                                                 N;
    int                                                 NNZ;
    std::vector<int>                                    IA;
    std::vector<int>                                    JA;
    size_t                                              numberOfVariables;
    daetools_mpi::daeModel_t*                           model;

    boost::shared_ptr<Epetra_Map>                           m_map;
    boost::shared_ptr<Epetra_CrsMatrix>                     m_matEPETRA;
    Epetra_SerialComm                                       m_Comm;
    boost::shared_ptr<ML_Epetra::MultiLevelPreconditioner>	m_pPreconditionerML;
    dae::solver::daeEpetraCSRMatrix                         m_matJacobian;
};

daePreconditioner_ML::daePreconditioner_ML()
{
}

daePreconditioner_ML::~daePreconditioner_ML()
{
    Free();
}

int daePreconditioner_ML::Initialize(daetools_mpi::daeModel_t *model, size_t numberOfVariables)
{
    daePreconditionerData_ML* p_data = new daePreconditionerData_ML(numberOfVariables, model);
    this->data = p_data;

    return 0;
}

int daePreconditioner_ML::Setup(real_t  time,
                                real_t  inverseTimeStep,
                                real_t* values,
                                real_t* timeDerivatives,
                                real_t* residuals)
{
    daePreconditionerData_ML* p_data = (daePreconditionerData_ML*)this->data;

    p_data->m_matJacobian.ClearMatrix();
    daeMatrixAccess_t* ma = p_data;
    p_data->model->EvaluateJacobian(p_data->numberOfVariables, time,  inverseTimeStep, values, timeDerivatives, residuals, ma);

    if(p_data->m_pPreconditionerML->IsPreconditionerComputed())
        p_data->m_pPreconditionerML->DestroyPreconditioner();

    int ret = p_data->m_pPreconditionerML->ComputePreconditioner();

    //p_data->m_pPreconditionerML->AnalyzeCoarse();
    return ret;
}

int daePreconditioner_ML::Solve(real_t  time, real_t* r, real_t* z)
{
    daePreconditionerData_ML* p_data = (daePreconditionerData_ML*)this->data;

    Epetra_MultiVector vecR(View, *p_data->m_map.get(), &r, 1);
    Epetra_MultiVector vecZ(View, *p_data->m_map.get(), &z, 1);

    p_data->m_pPreconditionerML->ApplyInverse(vecR, vecZ);
    //p_data->m_pPreconditionerML->ReportTime();

    return 0;
}

int daePreconditioner_ML::JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv)
{
    daePreconditionerData_ML* p_data = (daePreconditionerData_ML*)this->data;

    Epetra_MultiVector vecV (View, *p_data->m_map.get(), &v,  1);
    Epetra_MultiVector vecJv(View, *p_data->m_map.get(), &Jv, 1);

    return p_data->m_matEPETRA->Multiply(false, vecV, vecJv);
}

int daePreconditioner_ML::Free()
{
    daePreconditionerData_ML* p_data = (daePreconditionerData_ML*)this->data;

    if(p_data)
    {
        delete p_data;
        data = NULL;
    }

    return 0;
}

}
