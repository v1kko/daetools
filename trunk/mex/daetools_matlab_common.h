#include "math.h"
#include "simulation_loader_c.h"

#define IS_PARAM_DOUBLE(pVal) (mxIsNumeric(pVal) && !mxIsLogical(pVal) &&\
!mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal) && mxIsDouble(pVal))

#define IS_PARAM_UINT(pVal) (mxIsNumeric(pVal) && !mxIsLogical(pVal) &&\
!mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal))

#define IS_PARAM_STRING(pVal) (mxIsChar(pVal) && !mxIsLogical(pVal) &&\
!mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal) && !mxIsDouble(pVal))

#define IS_PARAM_CELL(pVal) (mxIsCell(pVal) && !mxIsEmpty(pVal))

/* A buffer for error/warning messages. */
char msg[1024];

/* Flag indicating the debug mode when diagnostic messages are printed to std-out */
bool debugMode = false;

/* Gets an item of the Matlab 2D mxArray (matrix) */
static double getItem(mxArray* mat, int i, int j) 
{
    mwSize mrows = mxGetM(mat);
    double* data = mxGetPr(mat);
    return data[i + j*mrows];
}

/* Sets an item of the Matlab 2D mxArray (matrix) */
static void setItem(mxArray* mat, int i, int j, double value) 
{
    mwSize mrows = mxGetM(mat);
    double* data = mxGetPr(mat);
    data[i + j*mrows] = value;
}

/* MEX- an S-function inputs (parameters, DOFs, initial conditions, active states) */
static void setSimulationInputs(void* simulation, const mxArray* inputs)
{
}

/* MEX- an S-function initialization options (DAE solver, LA solver, data reporter, log, ...) */
static void initializeSimulation(void* simulation, const mxArray* pOptions)
{
    if(!pOptions)
        return;
  
    /* Default settings. */
    char DAESolver[64]          = "IDAS";
    char LASolver[64]           = "";
    char DataReporter[64]       = "TCPIPDataReporter";
    char ConnectionString[64]   = "";
    char Log[64]                = "StdOutLog";
    bool ShowSimulationExplorer = false;
    
    /* Initialize the simulation. */
    Initialize(simulation, DAESolver, LASolver, DataReporter, ConnectionString, Log, ShowSimulationExplorer);
}

/* Sets the outputs of the MEX-function (Matlab mxArray 2D array) */
static reportDataToMatrix(void* simulation, mxArray* results, double currentTime, int step)
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

