/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
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
using namespace cs;

namespace cs_dae_simulator
{
daeModel_t::daeModel_t()
{
}

daeModel_t::~daeModel_t()
{
}

void daeModel_t::EvaluateResiduals(real_t time, real_t* residuals)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.ResidualsEvaluation);

    cs::csDAEModelImplementation_t::EvaluateResiduals(time, residuals);
}

void daeModel_t::EvaluateJacobian(real_t time, real_t inverseTimeStep, cs::csMatrixAccess_t* ma)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.JacobianEvaluation);

    cs::csDAEModelImplementation_t::EvaluateJacobian(time, inverseTimeStep, ma);
}

void daeModel_t::SetAndSynchroniseData(real_t time, real_t* values, real_t* time_derivatives)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.IPCDataExchange);

    cs::csDAEModelImplementation_t::SetAndSynchroniseData(time, values, time_derivatives);
}

}
