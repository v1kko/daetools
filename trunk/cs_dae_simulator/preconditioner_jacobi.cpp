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
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <string>
#include <iostream>
#include <vector>

namespace cs_dae_simulator
{
class daePreconditionerData : public cs::csMatrixAccess_t
{
public:
    daePreconditionerData(size_t numVars, cs::csDAEModel_t* mod):
        numberOfVariables(numVars), model(mod)
    {
        pp.resize(numberOfVariables, false);
        jacob.resize(numberOfVariables, false);
    }

    void SetItem(size_t row, size_t col, real_t value)
    {
        // Accepts only diagonal items
        if(row == col)
            jacob(row) = value;
    }

public:
    size_t            numberOfVariables;
    cs::csDAEModel_t* model;

    boost::numeric::ublas::vector<real_t> pp;    // Diagonal preconditioner
    boost::numeric::ublas::vector<real_t> jacob; // Jacobian matrix (only diagonal items)
};


daePreconditioner_Jacobi::daePreconditioner_Jacobi()
{
    data = NULL;
    printf("Instantiated Jacobi preconditioner\n");
}

daePreconditioner_Jacobi::~daePreconditioner_Jacobi()
{
    Free();
}

int daePreconditioner_Jacobi::Initialize(cs::csDAEModel_t* model, size_t numberOfVariables)
{
    daePreconditionerData* p_data = new daePreconditionerData(numberOfVariables, model);
    this->data = p_data;

    return 0;
}

int daePreconditioner_Jacobi::Setup(real_t  time,
                                    real_t  inverseTimeStep,
                                    real_t* values,
                                    real_t* timeDerivatives,
                                    real_t* residuals)
{
    daePreconditionerData* p_data = (daePreconditionerData*)this->data;

    // Calling SetAndSynchroniseData is not required here since it has previously been called by residuals function.
    p_data->jacob.clear();
    p_data->model->EvaluateJacobian(time, inverseTimeStep, p_data);

    /* Simple Jacobi preconditioner. */
    for(size_t i = 0; i < p_data->numberOfVariables; i++)
        p_data->pp[i] = 1.0 / (p_data->jacob[i] + 1e-20);

    return 0;
}

int daePreconditioner_Jacobi::Solve(real_t  time, real_t* r, real_t* z)
{
    daePreconditionerData* p_data = (daePreconditionerData*)this->data;

    /* Simple solve using the Jacobi preconditioner. */
    for(size_t i = 0; i < p_data->numberOfVariables; i++)
        z[i] = p_data->pp[i] * r[i];

    return 0;
}

int daePreconditioner_Jacobi::JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv)
{
    daePreconditionerData* p_data = (daePreconditionerData*)this->data;

    /* Simple solve using the Jacobi preconditioner. */
    for(size_t i = 0; i < p_data->numberOfVariables; i++)
        Jv[i] = p_data->jacob[i] * v[i];

    return 0;
}

int daePreconditioner_Jacobi::Free()
{
    daePreconditionerData* p_data = (daePreconditionerData*)this->data;

    if(p_data)
    {
        delete p_data;
        data = NULL;
    }
    return 0;
}

}
