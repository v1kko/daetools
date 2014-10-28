#include "mex.h"
#include "math.h"
#include "simulation_loader_c.h"

#define IS_PARAM_DOUBLE(pVal) (mxIsNumeric(pVal) && !mxIsLogical(pVal) &&\
!mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal) && mxIsDouble(pVal))

#define IS_PARAM_UINT(pVal) (mxIsNumeric(pVal) && !mxIsLogical(pVal) &&\
!mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal))

#define IS_PARAM_STRING(pVal) (mxIsChar(pVal) && !mxIsLogical(pVal) &&\
!mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal) && !mxIsDouble(pVal))

static double getItem(mxArray* mat, int i, int j) 
{
    mwSize mrows = mxGetM(mat);
    double* data = mxGetPr(mat);
    return data[i + j*mrows];
}

static void setItem(mxArray* mat, int i, int j, double value) 
{
    mwSize mrows = mxGetM(mat);
    double* data = mxGetPr(mat);
    data[i + j*mrows] = value;
}

static reportData(void* simulation, mxArray* results, double currentTime, int step)
{
    int i, j;
    unsigned int noPoints;
    char name[512];
    mxArray *itemCell, *outMatrix;

    int nOutletPorts = GetNumberOfOutputs(simulation);

    /* Set the outputs' values */
    for(i = 0; i < nOutletPorts; i++)
    {
        /* Get the result from daetools. */
        GetOutputInfo(simulation, i, name, &noPoints);
        double* data = (double*)malloc(noPoints*sizeof(double));
        GetOutputValue(simulation, i, data, noPoints);

        /* Get the i-th output matrix */
        itemCell  = mxGetCell(results, i);
        outMatrix = mxGetCell(itemCell, 1);
        for(j = 0; j < noPoints; j++)
            setItem(outMatrix, step, j, data[j]);

        free(data);
    }
    /* Set time */
    itemCell  = mxGetCell(results, nOutletPorts);
    outMatrix = mxGetCell(itemCell, 1);
    setItem(outMatrix, step, 0, currentTime);
}

char usage[] = "\n"
               "daetools mex function usage:\n"
               "Arguments:\n"
               "  [0] - Path to python file (char array)\n"
               "  [1] - Simulation class name (char array)\n"
               "  [2] - Inputs (cell array)\n"
               "  [3] - Simulation options (cell array)\n"
               " Outputs:\n"
               "  [0] - Cell array\n";

void mexFunction (int n_outputs,         mxArray* outputs[],
                  int n_arguments, const mxArray* arguments[])
{
    int i, j;
    unsigned int noPoints;
    char name[512];
    bool debugMode = true;
    
    /* Check the number of arguments. */
    if(n_arguments != 4)
        mexErrMsgTxt(usage);

    /* Check the number of outputs. */
    if(n_outputs != 1)
        mexErrMsgTxt(usage);
    
    const mxArray *pPythonPath      = arguments[0];
    const mxArray *pSimulationClass = arguments[1];
    const mxArray *pParameters      = arguments[2];
    const mxArray *pOptions         = arguments[3];

    /* Validate arguments */    
    /* First two arguments must be strings. */
    if(!IS_PARAM_STRING(pPythonPath)) 
    {
        mexErrMsgTxt("First parameter to daetools_mex function must be a string (full path to the python file)");
        return;
    } 
    
    if(!IS_PARAM_STRING(pSimulationClass)) 
    {
        mexErrMsgTxt("Second parameter to daetools_mex function must be a string (simulation class name)");
        return;
    } 

    /*if(mxIsCell(arguments[2]) != 1 || mxIsCell(arguments[3]) != 1)
        mexErrMsgTxt("Third and fourth argument must be cells"); */
    
    /* Get the length of the input string. */
    char* path      = mxArrayToString(pPythonPath);
    char* className = mxArrayToString(pSimulationClass);

    /* Load the simulation object with the specified name (simClassName) and 
     * from the specified file (path). */
    void *simulation = LoadSimulation(path, className);
    if(!simulation)
        mexErrMsgTxt("Cannot load DAETools simulation");

    /* Initialize the simulation with the default settings */
    Initialize(simulation, "daeIDAS", "", "TCPIPDataReporter", "", "daeStdOutLog", false);

    /* Set the time horizon (see if it can be done before the simulation start). */
    double timeHorizon       = 100;
    double reportingInterval = 5;
    if(timeHorizon <= reportingInterval)
        mexErrMsgTxt("Time horizon must be greater than the reporting interval");
 
    SetReportingInterval(simulation, reportingInterval);
    SetTimeHorizon(simulation,       timeHorizon);

    /* Allocate mx arrays to hold the output data. */
    int numberOfSteps = (int)(timeHorizon / reportingInterval) + 1 + 
                        (fmod(timeHorizon, reportingInterval) == 0 ? 0 : 1);
    if(debugMode)
        mexPrintf("Number of time steps = %d\n", numberOfSteps);
    
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
    
    int step = 0;
    double currentTime = 0;
    double targetTime  = 0;

    /* Solve the system with the specified initial conditions and report data. */
    SolveInitial(simulation);
    ReportData(simulation);
    reportData(simulation, results, currentTime, step);
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
        reportData(simulation, results, targetTime, step);
        
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

    /* Return the results. */
    outputs[0] = results;
    
    /* Free memory */
    mxFree(path);
    mxFree(className);
}

