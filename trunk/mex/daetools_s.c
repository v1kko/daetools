#define S_FUNCTION_LEVEL 2
#define S_FUNCTION_NAME  daetools_s

/*
 * Need to include simstruc.h for the definition of the SimStruct and
 * its associated macro definitions.
 */
#include "simstruc.h"
#include "simulation_loader_c.h"

#define IS_PARAM_DOUBLE(pVal) (mxIsNumeric(pVal) && !mxIsLogical(pVal) &&\
!mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal) && mxIsDouble(pVal))

#define IS_PARAM_STRING(pVal) (mxIsChar(pVal) && !mxIsLogical(pVal) &&\
!mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal) && !mxIsDouble(pVal))


#define MDL_CHECK_PARAMETERS
#if defined(MDL_CHECK_PARAMETERS)  && defined(MATLAB_MEX_FILE)
static void mdlCheckParameters(SimStruct *S)
{
    const mxArray *pPythonPath      = ssGetSFcnParam(S,0);
    const mxArray *pSimulationClass = ssGetSFcnParam(S,1);

    if ( !IS_PARAM_STRING(pPythonPath)) 
    {
        ssSetErrorStatus(S, "First parameter to daetools_s S-function must be a string (full path to the python file)");
        return;
    } 
    
    if ( !IS_PARAM_STRING(pSimulationClass)) 
    {
        ssSetErrorStatus(S, "Second parameter to daetools_s S-function must be a string (simulation class name)");
        return;
    } 
}
#endif


static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, 2);  /* Number of expected parameters */
    
#if defined(MATLAB_MEX_FILE)
    if (ssGetNumSFcnParams(S) == ssGetSFcnParamsCount(S)) {
        mdlCheckParameters(S);
        if (ssGetErrorStatus(S) != NULL) {
            return;
        }
    } else {
        return; /* Parameter mismatch will be reported by Simulink */
    }
#endif
    ssSetSFcnParamTunable(S, 0, 0);

    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);

    if (!ssSetNumInputPorts(S, 1)) return;
    
    if (!ssSetNumOutputPorts(S, 1)) return;
    
    ssSetInputPortWidth(S, 0, 1);  /* scalar double */
    ssSetOutputPortWidth(S, 0, 1); /* scalar double */

    ssSetNumSampleTimes(S, 1);
    ssSetNumRWork(S, 0);
    ssSetNumIWork(S, 0);
    ssSetNumPWork(S, 1);
    ssSetNumModes(S, 0);
    ssSetNumNonsampledZCs(S, 0);

    ssSetSimStateCompliance(S, USE_CUSTOM_SIM_STATE);

    ssSetOptions(S, 0);
}

static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, mxGetScalar(ssGetSFcnParam(S, 0)));
    ssSetOffsetTime(S, 0, 0.0);
    ssSetModelReferenceSampleTimeDefaultInheritance(S);
}

#define MDL_START

static void mdlStart(SimStruct *S)
{
    char* path      = mxArrayToString(ssGetSFcnParam(S, 0));
    char* className = mxArrayToString(ssGetSFcnParam(S, 1));
    
    mexPrintf("Python file: %s\n", path);
    mexPrintf("Simulation class: %s\n", className);

    Simulate(path, className, false);
}

static void mdlOutputs(SimStruct *S, int_T tid)
{
    real_T  *y = ssGetOutputPortRealSignal(S,0);
    y[0] = 5;
    UNUSED_ARG(tid);
}                                                

static void mdlTerminate(SimStruct *S)
{
}

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif

