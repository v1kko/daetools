#include "mex.h"
#include <string>
#include <cstring>
#include <fstream>
#include <streambuf>
#include "simulation_loader_c.h"

void mexFunction (int n_outputs,         mxArray* outputs[],
                  int n_arguments, const mxArray* arguments[])
{
    /* Check for proper number of arguments. */
    if(n_arguments != 4)
    {
        std::string usage = "\n"
        "daetools mex function usage:\n"
        "Arguments:\n"
        "  [0] - Path to python file (char array)\n"
        "  [1] - Simulation class name (char array)\n"
        "  [2] - Inputs (cell array)\n"
        "  [3] - Simulation options (cell array)\n"
        " Outputs:\n"
        "  [0] - Cell array\n";
        mexErrMsgTxt(usage.c_str());
        return;
    }

    /* Validate arguments */    
    /* First two arguments must be strings. */
    if(mxIsChar(arguments[0]) != 1 || mxIsChar(arguments[1]) != 1)
        mexErrMsgTxt("The first two arguments (1: a path to python file and 2: simulation class name) must be strings.");

    /* First two arguments must be row vectors. */
    if(mxGetM(arguments[0]) != 1 || mxGetM(arguments[1]) != 1)
        mexErrMsgTxt("Input must be a row vector.");

    //if(mxIsCell(arguments[2]) != 1 || mxIsCell(arguments[3]) != 1)
    //    mexErrMsgTxt("Third and fourth argument must be cells");
    
    /* Get the length of the input string. */
    char* path      = mxArrayToString(arguments[0]);
    char* className = mxArrayToString(arguments[1]);

    outputs[0] = mxCreateString(path);
    outputs[1] = mxCreateString(className);

    //int nInputs = mxGetNumberOfElements(arguments[2]);
    //for(int i = 0; i < nInputs; i++)
    //    outputs[i+2] = mxDuplicateArray(mxGetCell(arguments[2], i));

    Simulate(path, className, true);
    mexPrintf("loaded and simulated\n");
    
    //outputs[1] = mxDuplicateArray(inputs[1]);
    //outputs[2] = mxDuplicateArray(inputs[2]);

    /* Free memory */
    mxFree(path);
    mxFree(className);

    return;
}

