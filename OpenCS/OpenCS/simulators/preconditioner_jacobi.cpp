/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "daesimulator.h"
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
    daePreconditionerData(size_t numVars, cs::csDifferentialEquationModel_t* mod):
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
    size_t                             numberOfVariables;
    cs::csDifferentialEquationModel_t* model;

    boost::numeric::ublas::vector<real_t> pp;    // Diagonal preconditioner
    boost::numeric::ublas::vector<real_t> jacob; // Jacobian matrix (only diagonal items)
};


daePreconditioner_Jacobi::daePreconditioner_Jacobi()
{
    data = NULL;
    //printf("Instantiated Jacobi preconditioner\n");
}

daePreconditioner_Jacobi::~daePreconditioner_Jacobi()
{
    Free();
}

int daePreconditioner_Jacobi::Initialize(cs::csDifferentialEquationModel_t* model,
                                         size_t numberOfVariables,
                                         bool   isODESystem_)
{
    isODESystem = isODESystem_;
    daePreconditionerData* p_data = new daePreconditionerData(numberOfVariables, model);
    this->data = p_data;

    return 0;
}

int daePreconditioner_Jacobi::Setup(real_t  time,
                                    real_t  inverseTimeStep,
                                    real_t* values,
                                    real_t* timeDerivatives,
                                    bool    recomputeJacobian,
                                    real_t  jacobianScaleFactor)
{
    daePreconditionerData* p_data = (daePreconditionerData*)this->data;

    // Calling SetAndSynchroniseData is not required here since it has previously been called by residuals function.

    // Here, we do not use recomputeJacobian since if we update the Jacobian with [I] - gamma*[Jrhs]
    //   we cannot do it again with the different gamma.
    // Therefore, always compute the Jacobian.
    // Double check about the performance penalty it might cause!!
    p_data->jacob.clear();
    p_data->model->EvaluateJacobian(time, inverseTimeStep, p_data);

    // The Jacobian matrix we calculated is the Jacobian for the system of equations.
    // For ODE systems it is the RHS Jacobian (Jrhs) not the full system Jacobian.
    // The Jacobian matrix for ODE systems is in the form: [J] = [I] - gamma * [Jrhs],
    //   where gamma is a scaling factor sent by the ODE solver.
    // Here, the full Jacobian must be calculated.
    if(isODESystem)
    {
        for(size_t i = 0; i < p_data->numberOfVariables; i++)
            p_data->jacob[i] = 1.0 - jacobianScaleFactor * p_data->jacob[i];
    }

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
