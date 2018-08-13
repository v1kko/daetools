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
#include "auxiliary.h"
using namespace cs;

namespace cs_dae_simulator
{
daeModel_t::daeModel_t()
{
}

daeModel_t::~daeModel_t()
{
}

void daeModel_t::EvaluateEquations(real_t time, real_t* equations)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.EquationsEvaluation);

    cs::csDifferentialEquationModel::EvaluateEquations(time, equations);
}

void daeModel_t::EvaluateJacobian(real_t time, real_t inverseTimeStep, cs::csMatrixAccess_t* ma)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.JacobianEvaluation);

    cs::csDifferentialEquationModel::EvaluateJacobian(time, inverseTimeStep, ma);
}

void daeModel_t::SetAndSynchroniseData(real_t time, real_t* values, real_t* time_derivatives)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.IPCDataExchange);

    cs::csDifferentialEquationModel::SetAndSynchroniseData(time, values, time_derivatives);
}

}
