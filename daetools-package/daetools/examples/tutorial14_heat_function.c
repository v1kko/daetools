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
/* Compilation:
    - GNU/Linux gcc:
        gcc -fPIC -shared -std=c99 -o libheat_function.so tutorial14_heat_function.c
    - macOS gcc:
       gcc -fPIC -dynamiclib -o libheat_function.dylib tutorial14_heat_function.c
    - Windows vc++:
        cl /LD tutorial14_heat_function.c /link /dll /out:heat_function.dll
*/
#include <stdio.h>
#include <float.h>
#include <string.h>

typedef struct
{
    double Value;
    double Derivative;
}
adouble_c;

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

DLLEXPORT adouble_c calculate(const adouble_c values[], const char* names[], int no_arguments);

adouble_c calculate(const adouble_c values[], const char* names[], int no_arguments)
{
    adouble_c result;
    memset(&result, 0, sizeof(adouble_c));
    
    /* Print the values of the received arguments.
    printf("no_arguments = %d\n", no_arguments);
    printf("argument names: ");
    for(int i = 0; i < no_arguments; i++)
        printf("%s ", names[i]);
    printf("\n");

    printf("argument values: ");
    for(int i = 0; i < no_arguments; i++)
        printf("%.14f ", values[i]);
    printf("\n");
    */
    
    if(no_arguments != 3)
    {
        printf("Invalid number of arguments received %d (expected 3)\n", no_arguments);
        return result;
    }
    
    /* Get the arguments' values. */
    adouble_c m, cp, dT_dt;
    for(int i = 0; i < no_arguments; i++)
    {
        if(strcmp(names[i], "m") == 0)
            m = values[i];
        else if(strcmp(names[i], "cp") == 0)
            cp = values[i];
        else if(strcmp(names[i], "dT/dt") == 0)
            dT_dt = values[i];
    }
    //printf("m = %.14f\n", m.Value);
    //printf("cp = %.14f\n", cp.Value);
    //printf("dT_dt = %.14f\n", dT_dt.Value);
    
    /* Calculate the value. */
    result.Value = m.Value * cp.Value * dT_dt.Value;
    
    /* Calculate the derivative. */
    if(m.Derivative != 0) /* A derivative per 'm' was requested */
        result.Derivative = m.Derivative * (cp.Value * dT_dt.Value);
    else if(cp.Derivative != 0) /* A derivative per 'cp' was requested */
        result.Derivative = cp.Derivative * (m.Value * dT_dt.Value);
    else if(dT_dt.Derivative != 0) /* A derivative per 'dT_dt' was requested */
        result.Derivative = dT_dt.Derivative * (m.Value * cp.Value);
    
    return result;
}
