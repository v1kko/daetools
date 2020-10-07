/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2018
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CALL_STATS_H
#define CALL_STATS_H

#include <stdlib.h>
#include <time.h>
#include <omp.h>

namespace call_stats
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
        tc.duration += GetDuration();
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

}

#endif
