#include "math.h"
#include "simulation_loader_c.h"

/* In c99 bool is not a native type, and true/false are not defined
#ifndef false
#define false 0
#define true  1
#endif
*/

#define IS_PARAM_DOUBLE(pVal) (mxIsNumeric(pVal) && !mxIsLogical(pVal) &&\
                               !mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal) && mxIsDouble(pVal))

#define IS_PARAM_UINT(pVal) (mxIsNumeric(pVal) && !mxIsLogical(pVal) &&\
                             !mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal))

#define IS_PARAM_STRING(pVal) (mxIsChar(pVal) && !mxIsLogical(pVal) &&\
                               !mxIsEmpty(pVal) && !mxIsSparse(pVal) && !mxIsComplex(pVal) && !mxIsDouble(pVal))

#define IS_PARAM_STRING_EMPTY_ALLOWED(pVal) (mxIsChar(pVal) && !mxIsLogical(pVal) &&\
                                             !mxIsSparse(pVal) && !mxIsComplex(pVal) && !mxIsDouble(pVal))

#define IS_PARAM_LOGICAL(pVal) (mxIsLogical(pVal) && mxIsLogicalScalar(pVal) && !mxIsEmpty(pVal))

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

/* MEX- and S-function initialization options (LA solver or json runtime settings) */
/*
static int initializeSimulation(void* simulation, const mxArray* options)
{
    if(IS_PARAM_STRING(options))
    {
        char* jsonSettings = mxArrayToString(options);
        
        InitializeJSON(simulation, jsonSettings);

        mxFree(jsonSettings);

        return 0;
    }
    else if(IS_PARAM_CELL(options))
    {
        if(mxGetNumberOfElements(options) != 2)
        {
            mexErrMsgTxt("The size of simulation options argument must be 2 (LASolver and ShowSimulationExplorer)");
            return;
        }

        int i;
        mxArray* item;
        bool inputsOK = true;
        for(i = 0; i < 2; i++)
        {
            item = mxGetCell(options, i);
            if(i == 0 && !IS_PARAM_STRING(item))
            {
                mexPrintf("Item[%d] of the inputs cell (LA Solver) must be a string", i);
                inputsOK = false;
            }
            if(i == 1 && !IS_PARAM_LOGICAL(item))
            {
                mexPrintf("Item[%d] of the inputs cell (ShowSimulationExplorer) must be logical scalar", i);
                inputsOK = false;
            }
        }

        if(!inputsOK)
            return -1;

        char DAESolver[64]     = "Sundials IDAS";
        char DataReporter[64]  = "BlackHoleDataReporter";
        char ConnectString[64] = "";
        char Log[64]           = "StdOutLog";

        item = mxGetCell(options, 0);
        char* LASolver = mxArrayToString(item);

        item = mxGetCell(options, 1);
        bool ShowSimulationExplorer = mxIsLogicalScalarTrue(item);

        Initialize(simulation, DAESolver, LASolver, DataReporter, ConnectString, Log, ShowSimulationExplorer);

        mxFree(LASolver);

        return 0;
    }
    
    return -1;
}
*/
        
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

