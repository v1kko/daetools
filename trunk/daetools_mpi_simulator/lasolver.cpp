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
#include "lasolver.h"
#include "model.h"
#include <boost/numeric/ublas/matrix.hpp>

struct daeLASolverData
{
    size_t      numberOfVariables;
    daeModel_t* model;

    boost::numeric::ublas::vector<real_t> pp;    // Diagonal preconditioner
    boost::numeric::ublas::matrix<real_t> jacob; // Jacobian matrix
};

void laSetMatrixItem(void* matrix, size_t row, size_t col, real_t value)
{
    boost::numeric::ublas::matrix<real_t>* jacob = (boost::numeric::ublas::matrix<real_t>*)matrix;
    (*jacob)(row,col) = value;
}

int laInitialize(daeLASolver_t* lasolver,
                 void*          model,
                 size_t         numberOfVariables)
{
    daeLASolverData* data = new daeLASolverData;
    lasolver->data = data;

    data->numberOfVariables = numberOfVariables;
    data->model             = (daeModel_t*)model;
    data->pp.resize(numberOfVariables, false);
    data->jacob.resize(numberOfVariables, numberOfVariables, false);

    return 0;
}

int laSetup(daeLASolver_t*  lasolver,
            real_t          time,
            real_t          inverseTimeStep,
            real_t*         values,
            real_t*         timeDerivatives,
            real_t*         residuals)
{
    daeLASolverData* data = (daeLASolverData*)lasolver->data;

    /* The values and timeDerivatives have been copied in mpiSynchroniseData function. */
    int res = modJacobian(data->model, data->numberOfVariables, time,  inverseTimeStep, values, timeDerivatives, NULL, (void*)&data->jacob);

    /* Simple Jacobi preconditioner. */
    for(size_t i = 0; i < data->numberOfVariables; i++)
        data->pp[i] = 1.0 / (data->jacob(i,i) + 1e-20);

    /* Or:
    for(size_t i = 0; i < dae_solver->Nequations; i++)
    {
        if(dae_solver->jacob(i,i) == 0)
            dae_solver->pp[i] = 1.0;
        else
            dae_solver->pp[i] = 1.0 / dae_solver->jacob(i,i);
    }
    */

    return 0;
}

int laSolve(daeLASolver_t*  lasolver,
            real_t          time,
            real_t          inverseTimeStep,
            real_t          tolerance,
            real_t*         values,
            real_t*         timeDerivatives,
            real_t*         residuals,
            real_t*         r,
            real_t*         z)
{
    daeLASolverData* data = (daeLASolverData*)lasolver->data;

    /* Simple solve using the Jacobi preconditioner. */
    for(size_t i = 0; i < data->numberOfVariables; i++)
        z[i] = data->pp[i] * r[i];

    return 0;
}

int laFree(daeLASolver_t* lasolver)
{
    daeLASolverData* data = (daeLASolverData*)lasolver->data;

    data->pp.clear();
    data->jacob.clear();
    delete data;
    lasolver->data = NULL;

    return 0;
}

