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
#ifndef CS_SIMULATOR_AUXILIARY_H
#define CS_SIMULATOR_AUXILIARY_H

#include <omp.h>
#include <vector>

namespace auxiliary
{
class TimeAndCount
{
public:
    TimeAndCount()
    {
        count    = 0;
        duration = 0.0;
    }
public:
    int    count;
    double duration;
#ifdef STEP_DURATIONS
    std::vector<double> durations;
#endif
};

class TimerCounter
{
public:
    TimerCounter(TimeAndCount& tc_) : tc(tc_)
    {
        tc.count += 1;
        startTime = this->GetTime();
    }

    // At the end, adds the duration since creation to the contained TimeAndCount object.
    virtual ~TimerCounter()
    {
        double duration = GetDuration();
        tc.duration += duration;
#ifdef STEP_DURATIONS
        tc.durations.push_back(duration);
#endif
    }

    // Returns time in seconds since creation.
    double GetDuration()
    {
        double currentTime = this->GetTime();
        return (currentTime - startTime);
    }

    // Returns time in seconds.
    double GetTime()
    {
        return omp_get_wtime();
    }

public:
    double startTime;
    TimeAndCount& tc;
};

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
    TimeAndCount SimulationInitialise;
    TimeAndCount SimulationSolveInitial;
    TimeAndCount SimulationIntegration;
    TimeAndCount DAESolverSolveInitial;
    TimeAndCount DAESolverIntegration;
    TimeAndCount LASetup;
    TimeAndCount LASolve;
    TimeAndCount PSetup;
    TimeAndCount PSolve;
    TimeAndCount Jvtimes;
    TimeAndCount JvtimesDQ;
    TimeAndCount EquationsEvaluation;
    TimeAndCount JacobianEvaluation;
    TimeAndCount IPCDataExchange;
};

}
#endif
