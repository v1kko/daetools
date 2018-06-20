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
#ifndef CS_SIMULATOR_AUXILIARY_H
#define CS_SIMULATOR_AUXILIARY_H

#include <cs_call_stats.h>

namespace auxiliary
{
class daeTimesAndCounters
{
public:
    daeTimesAndCounters()
    {
    }

    virtual ~daeTimesAndCounters(void)
    {
    }

    static daeTimesAndCounters& GetTimesAndCounters()
    {
        static daeTimesAndCounters tcs;
        return tcs;
    }

public:
    call_stats::TimeAndCount SimulationInitialise;
    call_stats::TimeAndCount SimulationSolveInitial;
    call_stats::TimeAndCount SimulationIntegration;
    call_stats::TimeAndCount DAESolverSolveInitial;
    call_stats::TimeAndCount DAESolverIntegration;
    call_stats::TimeAndCount LASetup;
    call_stats::TimeAndCount LASolve;
    call_stats::TimeAndCount PSetup;
    call_stats::TimeAndCount PSolve;
    call_stats::TimeAndCount Jvtimes;
    call_stats::TimeAndCount JvtimesDQ;
    call_stats::TimeAndCount ResidualsEvaluation;
    call_stats::TimeAndCount JacobianEvaluation;
    call_stats::TimeAndCount IPCDataExchange;
};

}
#endif
