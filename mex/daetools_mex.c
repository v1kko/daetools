#include "mex.h"
#include "daetools_matlab_common.h"

char usage[] = "daetools_mex function usage:\n"
               "Arguments:\n"
               "  [1] - Path to python file (char array)\n"
               "  [2] - Name of python callable object that returns initialized simulation object (char array)\n"
               "  [3] - Arguments for the callable object (char array)\n"
               "  [4] - TimeHorizon (double scalar),\n"
               "  [5] - ReportingInterval (double scalar),\n"
               " Outputs:\n"
               "  [1] - Cell array (pairs: {'variable_name', double_matrix})\n"
               "\n"
               "Example:\n"
               "  res = daetools_mex('.../tutorial20.py', 'create_simulation', '', 100.0, 5.0)";

void mexFunction(int n_outputs,         mxArray* outputs[],
                 int n_arguments, const mxArray* arguments[])
{   
    int i, j;
    unsigned int noPoints;
    char name[512];
    bool debugMode = false;

    /* Check the number of arguments. */
    if(n_arguments != 5)
        mexErrMsgTxt(usage);

    /* Check the number of outputs. */
    if(n_outputs != 1)
        mexErrMsgTxt(usage);

    const mxArray *pPythonPath         = arguments[0];
    const mxArray *pSimulationCallable = arguments[1];
    const mxArray *pCallableArguments  = arguments[2];
    const mxArray *pTimeHorizon        = arguments[3];
    const mxArray *pReportingInterval  = arguments[4];
    
    /* Validate arguments */
    if(!IS_PARAM_STRING(pPythonPath))
    {
        mexErrMsgTxt("First argument must be a string (full path to the python file)");
        return;
    }

    if(!IS_PARAM_STRING(pSimulationCallable))
    {
        mexErrMsgTxt("Second argument must be a string (name of python callable object)");
        return;
    }

    if(!IS_PARAM_STRING_EMPTY_ALLOWED(pCallableArguments))
    {
        mexErrMsgTxt("Third argument must be a string (callable object arguments)");
        return;
    }

    if(!IS_PARAM_DOUBLE(pTimeHorizon))
    {
        mexErrMsgTxt("Fourth argument must be float value (time horizon)");
        return;
    }

    if(!IS_PARAM_DOUBLE(pReportingInterval))
    {
        mexErrMsgTxt("Fifth argument must be float value (reporting interval)");
        return;
    }
    
    /* Get the strings from mxArrays. */
    char* path              = mxArrayToString(pPythonPath);
    char* callableName      = mxArrayToString(pSimulationCallable);
    char* callableArguments = mxArrayToString(pCallableArguments);
    
    /* Load the simulation object from the specified file using 
     * the callable object and its arguments. */
    void *simulation = LoadSimulation(path, callableName, callableArguments);
    if(!simulation)
        mexErrMsgTxt("Cannot load DAETools simulation");

    /* Free memory */
    mxFree(path);
    mxFree(callableName);
    mxFree(callableArguments);
    
    /* Set the time horizon and reporting interval. */
    double timeHorizon       = (mxGetPr(pTimeHorizon))[0];
    double reportingInterval = (mxGetPr(pReportingInterval))[0];
    SetTimeHorizon(simulation,       timeHorizon);
    SetReportingInterval(simulation, reportingInterval);

    /* Allocate mx arrays to hold the output data. */
    int numberOfSteps = (int)(timeHorizon / reportingInterval) + 1 +
                        (fmod(timeHorizon, reportingInterval) == 0 ? 0 : 1);
    
    int nOutletPorts = GetNumberOfOutputs(simulation);

    mwSize ndim = 1;
    mwSize dims[1] = {nOutletPorts + 1};
    mxArray* results = mxCreateCellArray(ndim, dims);

    mxArray *itemCell, *outName, *outMatrix;
    mwSize ndim_out = 1;
    mwSize dims_out[1] = {2};
    for(i = 0; i < nOutletPorts; i++)
    {
        GetOutputInfo(simulation, i, name, &noPoints);

        itemCell = mxCreateCellArray(ndim_out, dims_out);

        outName   = mxCreateString(name);
        outMatrix = mxCreateDoubleMatrix(numberOfSteps, noPoints, mxREAL);
        mxSetCell(itemCell, 0, outName);
        mxSetCell(itemCell, 1, outMatrix);

        mxSetCell(results, i, itemCell);
    }
    /* Create the time array (index n+1) */
    itemCell  = mxCreateCellArray(ndim_out, dims_out);
    outName   = mxCreateString("times");
    outMatrix = mxCreateDoubleMatrix(numberOfSteps, 1, mxREAL);
    mxSetCell(itemCell, 0, outName);
    mxSetCell(itemCell, 1, outMatrix);
    mxSetCell(results, nOutletPorts, itemCell);
    
    /* Assign the results to outputs. */
    outputs[0] = results;

    int step = 0;
    double currentTime = 0;
    double targetTime  = 0;
    
    /* Solve the system with the specified initial conditions and report data. */
    SolveInitial(simulation);
    
    ReportData(simulation);
    reportDataToMatrix(simulation, results, currentTime, step);
    step += 1;

    /* Run. */
    while(targetTime < timeHorizon)
    {
        targetTime += reportingInterval;
        if(targetTime > timeHorizon)
            targetTime = timeHorizon;

        if(debugMode)
            mexPrintf("  Integrating from %.6f to %.6f ...\n", currentTime, targetTime);

        IntegrateUntilTime(simulation, targetTime);
        ReportData(simulation);
        reportDataToMatrix(simulation, results, targetTime, step);

        currentTime = targetTime;
        step += 1;
    }

    /* Put all outputs and times to the caller workspace. */
    char workspace[] = "caller";
    for(i = 0; i < nOutletPorts; i++)
    {
        GetOutputInfo(simulation, i, name, &noPoints);
        GetStrippedName(name, name);
        if(debugMode)
            mexPrintf("  Putting output %s with %d points to the %s workspace...\n", name, noPoints, workspace);

        itemCell  = mxGetCell(results, i);
        outMatrix = mxGetCell(itemCell, 1);
        mexPutVariable(workspace, name, outMatrix);
    }
    itemCell  = mxGetCell(results, nOutletPorts);
    outMatrix = mxGetCell(itemCell, 1);
    mexPutVariable(workspace, "times__", outMatrix);
}

