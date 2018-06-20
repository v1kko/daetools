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
#include <string>
#include <iostream>
#include <vector>
#include "trilinos_amesos_la_solver.h"

namespace dae
{
namespace solver
{
class daePreconditionerData_Ifpack
{
public:
    daePreconditionerData_Ifpack(const std::string& strPreconditionerName, size_t numVars, daeBlockOfEquations_t* block, Teuchos::ParameterList& parameters) :
        m_strPreconditionerName(strPreconditionerName), numberOfVariables(numVars), blockOfEquations(block), m_parameterList(parameters)
    {
        CreateStorage();

        printf("Proceed with Ifpack parameters:\n");
        m_parameterList.print(std::cout, 2, true, true);
        printf("\n");
    }

    // Create or re-create storage
    void CreateStorage()
    {
        m_map.reset(new Epetra_Map((int)numberOfVariables, 0, m_Comm));
        m_matEPETRA.reset(new Epetra_CrsMatrix(Copy, *m_map, 0));

        m_matJacobian.InitMatrix(numberOfVariables, m_matEPETRA.get());
        m_matJacobian.ResetCounters();
        blockOfEquations->FillSparseMatrix(&m_matJacobian);
        m_matJacobian.Sort(); // The function Sort will call FillComplete on Epetra matrix (required for Ifpack constructor).

        Ifpack factory;
        Ifpack_Preconditioner* preconditioner = factory.Create(m_strPreconditionerName, m_matEPETRA.get());
        if(!preconditioner)
            daeDeclareAndThrowException(exInvalidPointer);

        m_pPreconditionerIfpack.reset(preconditioner);
        printf("Instantiated %s preconditioner (requested: %s)\n", preconditioner->Label(), m_strPreconditionerName.c_str());

        // This will set the default parameters.
        m_pPreconditionerIfpack->SetParameters(m_parameterList);
        m_pPreconditionerIfpack->Initialize();
    }

public:
    std::string                              m_strPreconditionerName;
    size_t                                   numberOfVariables;
    daeBlockOfEquations_t*                   blockOfEquations;
    Teuchos::ParameterList&                  m_parameterList;
    daeRawDataArray<real_t>                  m_arrValues;
    daeRawDataArray<real_t>                  m_arrTimeDerivatives;
    daeRawDataArray<real_t>                  m_arrResiduals;
    boost::shared_ptr<Epetra_Map>            m_map;
    boost::shared_ptr<Epetra_CrsMatrix>      m_matEPETRA;
    Epetra_SerialComm                        m_Comm;
    boost::shared_ptr<Ifpack_Preconditioner> m_pPreconditionerIfpack;
    dae::solver::daeEpetraCSRMatrix          m_matJacobian;
};

daePreconditioner_Ifpack::daePreconditioner_Ifpack(const std::string& preconditionerName)
{
    data                  = NULL;
    strPreconditionerName = preconditionerName;
}

daePreconditioner_Ifpack::~daePreconditioner_Ifpack()
{
    Free();
}

std::string daePreconditioner_Ifpack::GetName() const
{
    return "Ifpack (" + strPreconditionerName + ")";
}

int daePreconditioner_Ifpack::Initialize(size_t numberOfVariables, daeBlockOfEquations_t* block)
{
    call_stats::TimerCounter tc(m_stats["Create"]);

    daePreconditionerData_Ifpack* p_data = new daePreconditionerData_Ifpack(strPreconditionerName, numberOfVariables, block, m_parameterList);
    this->data = p_data;

    return 0;
}

int daePreconditioner_Ifpack::Reinitialize()
{
    call_stats::TimerCounter tc(m_stats["Reinitialize"]);

    daePreconditionerData_Ifpack* p_data = (daePreconditionerData_Ifpack*)this->data;
    p_data->CreateStorage();
    return 0;
}

Teuchos::ParameterList& daePreconditioner_Ifpack::GetParameterList(void)
{
    return m_parameterList;
}

int daePreconditioner_Ifpack::Setup(real_t  time,
                                    real_t  inverseTimeStep,
                                    real_t* values,
                                    real_t* timeDerivatives,
                                    real_t* residuals)
{
    call_stats::TimerCounter tc(m_stats["Setup"]);

    daePreconditionerData_Ifpack* p_data = (daePreconditionerData_Ifpack*)this->data;

    {
        call_stats::TimerCounter tc(m_stats["Jacobian"]);

        size_t Neq = p_data->numberOfVariables;

        p_data->m_arrValues.InitArray(Neq, values);
        p_data->m_arrTimeDerivatives.InitArray(Neq, timeDerivatives);
        p_data->m_arrResiduals.InitArray(Neq, residuals);

        p_data->m_matJacobian.ClearMatrix();

        p_data->blockOfEquations->CalculateJacobian(time,
                                                    inverseTimeStep,
                                                    p_data->m_arrValues,
                                                    p_data->m_arrTimeDerivatives,
                                                    p_data->m_arrResiduals,
                                                    p_data->m_matJacobian);
    }

    int ret = p_data->m_pPreconditionerIfpack->Compute();
    printf("    t = %.15f compute preconditioner (condest = %.2e)\n", time, p_data->m_pPreconditionerIfpack->Condest());

    return ret;
}

int daePreconditioner_Ifpack::Solve(real_t  time, real_t* r, real_t* z)
{
    call_stats::TimerCounter tc(m_stats["Solve"]);

    daePreconditionerData_Ifpack* p_data = (daePreconditionerData_Ifpack*)this->data;

    Epetra_MultiVector vecR(View, *p_data->m_map.get(), &r, 1);
    Epetra_MultiVector vecZ(View, *p_data->m_map.get(), &z, 1);

    return p_data->m_pPreconditionerIfpack->ApplyInverse(vecR, vecZ);
}

int daePreconditioner_Ifpack::JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv)
{
    call_stats::TimerCounter tc(m_stats["JacobianVectorMultiply"]);

    daePreconditionerData_Ifpack* p_data = (daePreconditionerData_Ifpack*)this->data;

    Epetra_MultiVector vecV (View, *p_data->m_map.get(), &v,  1);
    Epetra_MultiVector vecJv(View, *p_data->m_map.get(), &Jv, 1);

    return p_data->m_matEPETRA->Multiply(false, vecV, vecJv);
}

int daePreconditioner_Ifpack::Free()
{
    daePreconditionerData_Ifpack* p_data = (daePreconditionerData_Ifpack*)this->data;

    if(p_data)
    {
        delete p_data;
        data = NULL;
    }

    return 0;
}

std::map<std::string, call_stats::TimeAndCount> daePreconditioner_Ifpack::GetCallStats() const
{
    return m_stats;
}

void daePreconditioner_Ifpack::SetOption_string(const std::string& optionName, const std::string& value)
{
    m_parameterList.set<std::string>(optionName, value);
}

void daePreconditioner_Ifpack::SetOption_float(const std::string& optionName, double value)
{
    m_parameterList.set<double>(optionName, value);
}

void daePreconditioner_Ifpack::SetOption_int(const std::string& optionName, int value)
{
    m_parameterList.set<int>(optionName, value);
}

void daePreconditioner_Ifpack::SetOption_bool(const std::string& optionName, bool value)
{
    m_parameterList.set<bool>(optionName, value);
}

std::string daePreconditioner_Ifpack::GetOption_string(const std::string& optionName)
{
    return m_parameterList.get<std::string>(optionName);
}

double daePreconditioner_Ifpack::GetOption_float(const std::string& optionName)
{
    return m_parameterList.get<double>(optionName);
}

int daePreconditioner_Ifpack::GetOption_int(const std::string& optionName)
{
    return m_parameterList.get<int>(optionName);
}

bool daePreconditioner_Ifpack::GetOption_bool(const std::string& optionName)
{
    return m_parameterList.get<bool>(optionName);
}

}
}
