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
#include "compute_stack_opencl.h"
#include <iostream>
#include <sstream>

const char* opencl_kernels_source =
#include "compute_stack_opencl_kernel_source.cl"
;

daeComputeStackEvaluator_OpenCL::daeComputeStackEvaluator_OpenCL(int platformID, int deviceID, std::string buildProgramOptions)
{
    cl_uint num_devices;
    cl_uint num_platforms;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int  ret;

    if(platformID < 0)
        platformID = 0;
    if(deviceID < 0)
        deviceID = 0;

    std::vector<openclPlatform_t> arrPlatforms = AvailableOpenCLPlatforms();
    std::vector<openclDevice_t>   arrDevices   = AvailableOpenCLDevices();

    num_platforms = arrPlatforms.size();

    if(num_platforms == 0)
         throw std::runtime_error("No OpenCL platforms found");
    if(platformID >= num_platforms)
         throw std::runtime_error("Invalid OpenCL PlatformsID");

    std::vector<cl_platform_id> platform_ids(num_platforms, NULL);
    cl_device_id device_id;
    cl_platform_id platform_id;

    mem_computeStacks               = NULL;
    mem_activeEquationSetIndexes    = NULL;
    mem_computeStackJacobianItems   = NULL;
    mem_dofs                        = NULL;
    mem_values                      = NULL;
    mem_timeDerivatives             = NULL;
    mem_residuals                   = NULL;
    mem_jacobianItems               = NULL;
    mem_sresiduals                  = NULL;
    mem_svalues                     = NULL;
    mem_sdvalues                    = NULL;

    /* Find the platform with the specified platformID. */
    ret = clGetPlatformIDs(num_platforms, &platform_ids[0], &ret_num_platforms);
    clCheck(ret);

    platform_id = platform_ids[platformID];

    /* Find the device for the specified platform with the specified deviceID. */
    /*   a) Get Number of devices for given platform. */
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    clCheck(ret);

    /*   b) Get device IDs and select one with the given deviceID. */
    std::vector<cl_device_id> device_ids(num_devices, NULL);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, &device_ids[0], NULL);
    clCheck(ret);

    if(deviceID >= num_devices)
         throw std::runtime_error("Invalid OpenCL DeviceID");

    device_id = device_ids[deviceID];

    /* Get the selected device name and type for the input arguments: platformID and deviceID. */
    char devName[512];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(devName),        devName, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &selectedDeviceType, NULL);
    selectedDeviceName = devName;

    /* Print the selected device name and type. */
    printf("Selected OpenCL device: %s (", selectedDeviceName.c_str());
    if(selectedDeviceType == CL_DEVICE_TYPE_DEFAULT)
        printf("CL_DEVICE_TYPE_DEFAULT)\n\n");
    else if(selectedDeviceType == CL_DEVICE_TYPE_CPU)
        printf("CL_DEVICE_TYPE_CPU)\n\n");
    else if(selectedDeviceType == CL_DEVICE_TYPE_GPU)
        printf("CL_DEVICE_TYPE_GPU)\n\n");
    else if(selectedDeviceType == CL_DEVICE_TYPE_ACCELERATOR)
        printf("CL_DEVICE_TYPE_ACCELERATOR)\n\n");
    else if(selectedDeviceType == CL_DEVICE_TYPE_CUSTOM)
        printf("CL_DEVICE_TYPE_CUSTOM)\n\n");

    /* Create OpenCL context. */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    clCheck(ret);

    /* Create OpenCL command queue. */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    clCheck(ret);

    /* Create OpenCL program from the source(s). */
    std::string program_source;
    program_source += opencl_kernels_source;
    const char* program_source_str   = program_source.c_str();
    size_t program_source_size = program_source.size();
    //printf("program_source_str = %s\n", program_source_str);

    program = clCreateProgramWithSource(context, 1, (const char **)&program_source_str, (const size_t *)&program_source_size, &ret);
    clCheck(ret);

    /* Build OpenCL program. */
    std::string cl_options;
    if(selectedDeviceType == CL_DEVICE_TYPE_CPU)
        cl_options = "-cl-opt-disable " + buildProgramOptions;
    else
        cl_options = buildProgramOptions;

    ret = clBuildProgram(program, 1, &device_id, cl_options.c_str(), NULL, NULL);
    {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if(log_size > 0)
        {
            char* log = new char[log_size];
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("%s\n", log);
            delete[] log;
        }
    }
    clCheck(ret);

    /* Create OpenCL kernels. */
    kernel_residual = clCreateKernel(program, "EvaluateResidual", &ret);
    clCheck(ret);

    kernel_jacobian = clCreateKernel(program, "EvaluateJacobian", &ret);
    clCheck(ret);

    kernel_reset_jacobian = clCreateKernel(program, "ResetJacobianData", &ret);
    clCheck(ret);

    kernel_sens_residual = clCreateKernel(program, "EvaluateSensitivityResidual", &ret);
    clCheck(ret);
}

daeComputeStackEvaluator_OpenCL::~daeComputeStackEvaluator_OpenCL()
{
    FreeResources();
}

void daeComputeStackEvaluator_OpenCL::FreeResources()
{
    /* Do not check the return codes for there is no much room for recovery. */
    cl_int ret;

    if(mem_computeStacks)
        ret = clReleaseMemObject(mem_computeStacks);
    if(mem_activeEquationSetIndexes)
        ret = clReleaseMemObject(mem_activeEquationSetIndexes);
    if(mem_computeStackJacobianItems)
        ret = clReleaseMemObject(mem_computeStackJacobianItems);
    if(mem_dofs)
        ret = clReleaseMemObject(mem_dofs);
    if(mem_values)
        ret = clReleaseMemObject(mem_values);
    if(mem_timeDerivatives)
        ret = clReleaseMemObject(mem_timeDerivatives);
    if(mem_residuals)
        ret = clReleaseMemObject(mem_residuals);
    if(mem_jacobianItems)
        ret = clReleaseMemObject(mem_jacobianItems);
    if(mem_sresiduals)
        ret = clReleaseMemObject(mem_sresiduals);
    if(mem_svalues)
        ret = clReleaseMemObject(mem_svalues);
    if(mem_sdvalues)
        ret = clReleaseMemObject(mem_sdvalues);

    if(kernel_residual)
        ret = clReleaseKernel(kernel_residual);
    if(kernel_jacobian)
        ret = clReleaseKernel(kernel_jacobian);
    if(kernel_reset_jacobian)
        ret = clReleaseKernel(kernel_reset_jacobian);
    if(kernel_sens_residual)
        ret = clReleaseKernel(kernel_sens_residual);
    if(program)
        ret = clReleaseProgram(program);
    if(command_queue)
        ret = clReleaseCommandQueue(command_queue);
    if(context)
        ret = clReleaseContext(context);

    kernel_residual                 = NULL;
    kernel_jacobian                 = NULL;
    kernel_reset_jacobian           = NULL;
    kernel_sens_residual            = NULL;
    program                         = NULL;
    command_queue                   = NULL;
    context                         = NULL;

    mem_computeStacks               = NULL;
    mem_activeEquationSetIndexes    = NULL;
    mem_computeStackJacobianItems   = NULL;
    mem_dofs                        = NULL;
    mem_values                      = NULL;
    mem_timeDerivatives             = NULL;
    mem_residuals                   = NULL;
    mem_jacobianItems               = NULL;
    mem_sresiduals                  = NULL;
    mem_svalues                     = NULL;
    mem_sdvalues                    = NULL;
}

void daeComputeStackEvaluator_OpenCL::Initialize(bool calculateSensitivities,
                                                 size_t numberOfVariables,
                                                 size_t numberOfEquationsToProcess,
                                                 size_t numberOfDOFs,
                                                 size_t numberOfComputeStackItems,
                                                 size_t numberOfJacobianItems,
                                                 size_t numberOfJacobianItemsToProcess,
                                                 adComputeStackItem_t*    computeStacks,
                                                 uint32_t*                activeEquationSetIndexes,
                                                 adJacobianMatrixItem_t*  computeStackJacobianItems)
{
    if(mem_computeStacks || mem_activeEquationSetIndexes || mem_computeStackJacobianItems ||
       mem_dofs || mem_values || mem_timeDerivatives ||
       mem_residuals || mem_jacobianItems || mem_sresiduals || mem_svalues || mem_sdvalues)
    {
        clReleaseMemObject(mem_computeStacks);
        clReleaseMemObject(mem_activeEquationSetIndexes);
        clReleaseMemObject(mem_computeStackJacobianItems);
        clReleaseMemObject(mem_dofs);
        clReleaseMemObject(mem_values);
        clReleaseMemObject(mem_timeDerivatives);
        clReleaseMemObject(mem_residuals);
        clReleaseMemObject(mem_computeStackJacobianItems);
        clReleaseMemObject(mem_jacobianItems);
        clReleaseMemObject(mem_sresiduals);
        clReleaseMemObject(mem_svalues);
        clReleaseMemObject(mem_sdvalues);
    }

    cl_int ret;
    size_t global_item_size;

    printf("numberOfVariables          = %d\n", numberOfVariables);
    //printf("numberOfEquationsToProcess = %d\n", numberOfEquationsToProcess);
    //printf("numberOfDOFs               = %d\n", numberOfDOFs);
    printf("numberOfComputeStackItems  = %d\n", numberOfComputeStackItems);
    printf("numberOfJacobianItems      = %d\n", numberOfJacobianItems);

    size_t bufferSize = numberOfComputeStackItems*sizeof(adComputeStackItem_t);
    mem_computeStacks = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &ret);
    clCheck(ret);
    ret = clEnqueueWriteBuffer(command_queue, mem_computeStacks, CL_TRUE, 0, bufferSize, computeStacks, 0, NULL, NULL);
    clCheck(ret);

    bufferSize = numberOfVariables*sizeof(uint32_t);
    mem_activeEquationSetIndexes = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &ret);
    clCheck(ret);
    ret = clEnqueueWriteBuffer(command_queue, mem_activeEquationSetIndexes, CL_TRUE, 0, bufferSize, activeEquationSetIndexes, 0, NULL, NULL);
    clCheck(ret);

    bufferSize = numberOfJacobianItems*sizeof(adJacobianMatrixItem_t);
    mem_computeStackJacobianItems = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &ret);
    clCheck(ret);
    ret = clEnqueueWriteBuffer(command_queue, mem_computeStackJacobianItems, CL_TRUE, 0, bufferSize, computeStackJacobianItems, 0, NULL, NULL);
    clCheck(ret);

    bufferSize = numberOfDOFs*sizeof(real_t);
    if(bufferSize == 0)
        bufferSize = 1;
    mem_dofs = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &ret);
    clCheck(ret);

    bufferSize = numberOfVariables*sizeof(real_t);
    mem_values = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &ret);
    clCheck(ret);

    bufferSize = numberOfVariables*sizeof(real_t);
    mem_timeDerivatives = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &ret);
    clCheck(ret);

    bufferSize = numberOfEquationsToProcess*sizeof(real_t);
    mem_residuals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, &ret);
    clCheck(ret);

    bufferSize = numberOfJacobianItemsToProcess*sizeof(real_t);
    mem_jacobianItems = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, &ret);
    clCheck(ret);

    ret = clSetKernelArg(kernel_reset_jacobian, 0, sizeof(cl_mem), (void *)&mem_jacobianItems);
    clCheck(ret);
    /* Initialise jacobianItems with zeroes. */
    global_item_size = numberOfJacobianItemsToProcess;
    ret = clEnqueueNDRangeKernel(command_queue, kernel_reset_jacobian, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    clCheck(ret);

    if(calculateSensitivities)
    {
        bufferSize = numberOfVariables*sizeof(real_t);
        mem_svalues = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &ret);
        clCheck(ret);

        bufferSize = numberOfVariables*sizeof(real_t);
        mem_sdvalues = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &ret);
        clCheck(ret);

        bufferSize = numberOfEquationsToProcess*sizeof(real_t);
        mem_sresiduals = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, &ret);
        clCheck(ret);
    }

    ret = clFinish(command_queue);
    clCheck(ret);
}

/* Residual kernel function. */
void daeComputeStackEvaluator_OpenCL::EvaluateResiduals(daeComputeStackEvaluationContext_t EC,
                                                        real_t*                            dofs,
                                                        real_t*                            values,
                                                        real_t*                            timeDerivatives,
                                                        real_t*                            residuals)
{
    size_t bufferSize;
    cl_int ret;

    /* Copy input arrays to cl_mem buffers. */
    if(EC.numberOfDOFs > 0)
    {
        bufferSize = EC.numberOfDOFs*sizeof(real_t);
        ret = clEnqueueWriteBuffer(command_queue, mem_dofs, CL_TRUE, 0, bufferSize, dofs, 0, NULL, NULL);
        clCheck(ret);
    }

    bufferSize = EC.numberOfVariables*sizeof(real_t);
    ret = clEnqueueWriteBuffer(command_queue, mem_values, CL_TRUE, 0, bufferSize, values, 0, NULL, NULL);
    clCheck(ret);

    bufferSize = EC.numberOfVariables*sizeof(real_t);
    ret = clEnqueueWriteBuffer(command_queue, mem_timeDerivatives, CL_TRUE, 0, bufferSize, timeDerivatives, 0, NULL, NULL);
    clCheck(ret);

    /* Set OpenCL kernel arguments. */
    ret = clSetKernelArg(kernel_residual, 0, sizeof(cl_mem),                             (void *)&mem_computeStacks);
    clCheck(ret);
    ret = clSetKernelArg(kernel_residual, 1, sizeof(cl_mem),                             (void *)&mem_activeEquationSetIndexes);
    clCheck(ret);
    ret = clSetKernelArg(kernel_residual, 2, sizeof(daeComputeStackEvaluationContext_t), (void *)&EC);
    clCheck(ret);
    ret = clSetKernelArg(kernel_residual, 3, sizeof(cl_mem),                             (void *)&mem_dofs);
    clCheck(ret);
    ret = clSetKernelArg(kernel_residual, 4, sizeof(cl_mem),                             (void *)&mem_values);
    clCheck(ret);
    ret = clSetKernelArg(kernel_residual, 5, sizeof(cl_mem),                             (void *)&mem_timeDerivatives);
    clCheck(ret);
    ret = clSetKernelArg(kernel_residual, 6, sizeof(cl_mem),                             (void *)&mem_residuals);
    clCheck(ret);

    /* Execute OpenCL kernel. */
    size_t global_item_size = EC.numberOfEquations;
    size_t* plocal_item_size = NULL;

    ret = clEnqueueNDRangeKernel(command_queue, kernel_residual, 1, NULL, &global_item_size, plocal_item_size, 0, NULL, NULL);
    clCheck(ret);

    /* Copy results from the memory buffer. */
    bufferSize = EC.numberOfEquations*sizeof(real_t);
    ret = clEnqueueReadBuffer(command_queue, mem_residuals, CL_TRUE, 0, bufferSize, residuals, 0, NULL, NULL);
    clCheck(ret);

    /* Wait for the commands to finish. */
    ret = clFinish(command_queue);
    clCheck(ret);
}

/* Jacobian kernel function (generic version). */
void daeComputeStackEvaluator_OpenCL::EvaluateJacobian(daeComputeStackEvaluationContext_t EC,
                                                       real_t*                            dofs,
                                                       real_t*                            values,
                                                       real_t*                            timeDerivatives,
                                                       real_t*                            jacobianItems)
{
    size_t bufferSize;
    cl_int ret;

    /* Copy input arrays to cl_mem buffers. */
    if(EC.numberOfDOFs > 0)
    {
        bufferSize = EC.numberOfDOFs*sizeof(real_t);
        ret = clEnqueueWriteBuffer(command_queue, mem_dofs, CL_TRUE, 0, bufferSize, dofs, 0, NULL, NULL);
        clCheck(ret);
    }

    bufferSize = EC.numberOfVariables*sizeof(real_t);
    ret = clEnqueueWriteBuffer(command_queue, mem_values, CL_TRUE, 0, bufferSize, values, 0, NULL, NULL);
    clCheck(ret);

    bufferSize = EC.numberOfVariables*sizeof(real_t);
    ret = clEnqueueWriteBuffer(command_queue, mem_timeDerivatives, CL_TRUE, 0, bufferSize, timeDerivatives, 0, NULL, NULL);
    clCheck(ret);

    /* Set OpenCL kernel arguments. */
    ret = clSetKernelArg(kernel_jacobian, 0, sizeof(cl_mem),                             (void *)&mem_computeStacks);
    clCheck(ret);
    ret = clSetKernelArg(kernel_jacobian, 1, sizeof(cl_mem),                             (void *)&mem_activeEquationSetIndexes);
    clCheck(ret);
    ret = clSetKernelArg(kernel_jacobian, 2, sizeof(cl_mem),                             (void *)&mem_computeStackJacobianItems);
    clCheck(ret);
    ret = clSetKernelArg(kernel_jacobian, 3, sizeof(daeComputeStackEvaluationContext_t), (void *)&EC);
    clCheck(ret);
    ret = clSetKernelArg(kernel_jacobian, 4, sizeof(cl_mem),                             (void *)&mem_dofs);
    clCheck(ret);
    ret = clSetKernelArg(kernel_jacobian, 5, sizeof(cl_mem),                             (void *)&mem_values);
    clCheck(ret);
    ret = clSetKernelArg(kernel_jacobian, 6, sizeof(cl_mem),                             (void *)&mem_timeDerivatives);
    clCheck(ret);
    ret = clSetKernelArg(kernel_jacobian, 7, sizeof(cl_mem),                             (void *)&mem_jacobianItems);
    clCheck(ret);

    /* Execute OpenCL kernel. */
    size_t global_item_size = EC.numberOfJacobianItems;
    size_t* plocal_item_size = NULL;

    ret = clEnqueueNDRangeKernel(command_queue, kernel_jacobian, 1, NULL, &global_item_size, plocal_item_size, 0, NULL, NULL);
    clCheck(ret);

    /* Copy results from the memory buffer. */
    bufferSize = EC.numberOfJacobianItems*sizeof(real_t);
    ret = clEnqueueReadBuffer(command_queue, mem_jacobianItems, CL_TRUE, 0, bufferSize, jacobianItems, 0, NULL, NULL);
    clCheck(ret);

    /* Wait for the commands to finish. */
    ret = clFinish(command_queue);
    clCheck(ret);
}

/* Sensitivity residual kernel function. */
void daeComputeStackEvaluator_OpenCL::EvaluateSensitivityResiduals(daeComputeStackEvaluationContext_t EC,
                                                                   real_t*                            dofs,
                                                                   real_t*                            values,
                                                                   real_t*                            timeDerivatives,
                                                                   real_t*                            svalues,
                                                                   real_t*                            sdvalues,
                                                                   real_t*                            sresiduals)
{
    size_t bufferSize;
    cl_int ret;

    /* Copy input arrays to cl_mem buffers. */
    if(EC.numberOfDOFs > 0)
    {
        bufferSize = EC.numberOfDOFs*sizeof(real_t);
        ret = clEnqueueWriteBuffer(command_queue, mem_dofs, CL_TRUE, 0, bufferSize, dofs, 0, NULL, NULL);
        clCheck(ret);
    }

    bufferSize = EC.numberOfVariables*sizeof(real_t);
    ret = clEnqueueWriteBuffer(command_queue, mem_values, CL_TRUE, 0, bufferSize, values, 0, NULL, NULL);
    clCheck(ret);

    bufferSize = EC.numberOfVariables*sizeof(real_t);
    ret = clEnqueueWriteBuffer(command_queue, mem_timeDerivatives, CL_TRUE, 0, bufferSize, timeDerivatives, 0, NULL, NULL);
    clCheck(ret);

    bufferSize = EC.numberOfVariables*sizeof(real_t);
    ret = clEnqueueWriteBuffer(command_queue, mem_svalues, CL_TRUE, 0, bufferSize, svalues, 0, NULL, NULL);
    clCheck(ret);

    bufferSize = EC.numberOfVariables*sizeof(real_t);
    ret = clEnqueueWriteBuffer(command_queue, mem_sdvalues, CL_TRUE, 0, bufferSize, sdvalues, 0, NULL, NULL);
    clCheck(ret);

    /* Set OpenCL kernel arguments. */
    ret = clSetKernelArg(kernel_sens_residual, 0, sizeof(cl_mem),                             (void *)&mem_computeStacks);
    clCheck(ret);
    ret = clSetKernelArg(kernel_sens_residual, 1, sizeof(cl_mem),                             (void *)&mem_activeEquationSetIndexes);
    clCheck(ret);
    ret = clSetKernelArg(kernel_sens_residual, 2, sizeof(daeComputeStackEvaluationContext_t), (void *)&EC);
    clCheck(ret);
    ret = clSetKernelArg(kernel_sens_residual, 3, sizeof(cl_mem),                             (void *)&mem_dofs);
    clCheck(ret);
    ret = clSetKernelArg(kernel_sens_residual, 4, sizeof(cl_mem),                             (void *)&mem_values);
    clCheck(ret);
    ret = clSetKernelArg(kernel_sens_residual, 5, sizeof(cl_mem),                             (void *)&mem_timeDerivatives);
    clCheck(ret);
    ret = clSetKernelArg(kernel_sens_residual, 6, sizeof(cl_mem),                             (void *)&mem_sdvalues);
    clCheck(ret);
    ret = clSetKernelArg(kernel_sens_residual, 7, sizeof(cl_mem),                             (void *)&mem_sdvalues);
    clCheck(ret);
    ret = clSetKernelArg(kernel_sens_residual, 8, sizeof(cl_mem),                             (void *)&mem_sresiduals);
    clCheck(ret);

    /* Execute OpenCL kernel. */
    size_t global_item_size = EC.numberOfEquations;
    size_t* plocal_item_size = NULL;

    ret = clEnqueueNDRangeKernel(command_queue, kernel_sens_residual, 1, NULL, &global_item_size, plocal_item_size, 0, NULL, NULL);
    clCheck(ret);

    /* Copy results from the memory buffer. */
    bufferSize = EC.numberOfEquations*sizeof(real_t);
    ret = clEnqueueReadBuffer(command_queue, mem_sresiduals, CL_TRUE, 0, bufferSize, sresiduals, 0, NULL, NULL);
    clCheck(ret);

    /* Wait for the commands to finish. */
    ret = clFinish(command_queue);
    clCheck(ret);
}

