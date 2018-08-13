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
#if !defined(CS_EVALUATOR_OPENCL_H)
#define CS_EVALUATOR_OPENCL_H

#include <stdlib.h>
#include <stdio.h>
#include <stdexcept>

#include "../cs_evaluator.h"
#include "cs_opencl_platforms.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace cs
{
class csComputeStackEvaluator_OpenCL : public csComputeStackEvaluator_t
{
public:
    csComputeStackEvaluator_OpenCL(int platformID, int deviceID, std::string buildProgramOptions = "");
    virtual ~csComputeStackEvaluator_OpenCL();

    void FreeResources();

    void Initialize(bool                     calculateSensitivities,
                    size_t                   numberOfVariables,
                    size_t                   numberOfEquationsToProcess,
                    size_t                   numberOfDOFs,
                    size_t                   numberOfComputeStackItems,
                    size_t                   numberOfIncidenceMatrixItems,
                    size_t                   numberOfIncidenceMatrixItemsToProcess,
                    csComputeStackItem_t*    computeStacks,
                    uint32_t*                activeEquationSetIndexes,
                    csIncidenceMatrixItem_t* incidenceMatrixItems);

    /* Evaluate equations. */
    void EvaluateEquations(csEvaluationContext_t EC,
                           real_t*               dofs,
                           real_t*               values,
                           real_t*               timeDerivatives,
                           real_t*               residuals);

    /* Evaluate derivatives (Jacobian matrix). */
    void EvaluateDerivatives(csEvaluationContext_t EC,
                             real_t*               dofs,
                             real_t*               values,
                             real_t*               timeDerivatives,
                             real_t*               jacobianItems);

    /* Evaluate sensitivity derivatives. */
    void EvaluateSensitivityDerivatives(csEvaluationContext_t EC,
                                        real_t*               dofs,
                                        real_t*               values,
                                        real_t*               timeDerivatives,
                                        real_t*               svalues,
                                        real_t*               sdvalues,
                                        real_t*               sresiduals);

public:
    cl_context       context;
    cl_command_queue command_queue;
    cl_program       program;
    cl_kernel        kernel_equation;
    cl_kernel        kernel_derivative;
    cl_kernel        kernel_reset_jacobian;
    cl_kernel        kernel_sens_derivative;

    std::string    selectedDeviceName;
    cl_device_type selectedDeviceType;

    // Input buffers
    cl_mem mem_computeStacks;
    cl_mem mem_activeEquationSetIndexes;
    cl_mem mem_computeStackJacobianItems;
    cl_mem mem_dofs;
    cl_mem mem_values;
    cl_mem mem_timeDerivatives;
    cl_mem mem_svalues;
    cl_mem mem_sdvalues;

    // Output buffers
    cl_mem mem_residuals;
    cl_mem mem_jacobianItems;
    cl_mem mem_sresiduals;
};

static inline const char* getErrorString(cl_int error)
{
    switch(error)
    {
        // run-time and JIT compiler errors
        case  0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

#define clDeclareAndThrowException(ret_code)	{ \
    std::stringstream ss; \
    ss << "OpenCL error" << std::endl \
       << "return code: " << getErrorString(ret_code) << std::endl \
       << "function: " << std::string(__FUNCTION__) << std::endl \
       << "file: " << std::string(__FILE__) << std::endl \
       << "line:" << __LINE__; \
     std::string what = ss.str(); \
     throw std::runtime_error(what); \
  }

#define clCheck(ret_code) if(ret_code != CL_SUCCESS) clDeclareAndThrowException(ret_code)

}
#endif
