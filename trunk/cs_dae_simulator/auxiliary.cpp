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
#include "auxiliary.h"
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>

namespace auxiliary
{
bool compare_strings(const char* s1, const char* s2)
{
    return (strcmp(s1, s2) == 0 ? true : false);
}

void log_message(const char* msg)
{
    printf(msg);
}

double get_time_in_seconds()
{
    return omp_get_wtime();
/*
#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
    DWORD time = GetTickCount();
    return (double)(time / 1.0E3);

#elif defined(__MACH__) || defined(__APPLE__)
// This part needs to be checked thoroughly...
    uint64_t time;
    uint64_t timeNano;
    static mach_timebase_info_data_t sTimebaseInfo;

    // Start the clock.
    time = mach_absolute_time();

    // If this is the first time we've run, get the timebase.
    // We can use denom == 0 to indicate that sTimebaseInfo is
    // uninitialised because it makes no sense to have a zero
    // denominator is a fraction.
    if(sTimebaseInfo.denom == 0)
        mach_timebase_info(&sTimebaseInfo);

    timeNano = time * sTimebaseInfo.numer / sTimebaseInfo.denom;
    return (double)(timeNano / 1.0E9);

#elif defined(__MINGW32__)
    clock_t t1 = clock();
    return (double)(t1 / CLOCKS_PER_SEC);

#elif __linux__ == 1
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)(time.tv_sec + time.tv_nsec / 1.0E9);

#else
    #error Unknown Platform!!
    return 0.0;
#endif
*/
}

}
