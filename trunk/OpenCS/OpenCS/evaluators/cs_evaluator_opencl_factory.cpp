#include "cs_evaluator_opencl_multidevice.h"
#include "cs_opencl_platforms.h"
#include "cs_evaluator_opencl_factory.h"

namespace cs
{
csComputeStackEvaluator_t* csCreateOpenCLEvaluator(int platformID, int deviceID, std::string buildProgramOptions)
{
    return new csComputeStackEvaluator_OpenCL(platformID, deviceID, buildProgramOptions);
}

csComputeStackEvaluator_t* csCreateOpenCLEvaluator_MultiDevice(const std::vector<int>&    platforms,
                                                               const std::vector<int>&    devices,
                                                               const std::vector<double>& taskPortions,
                                                               std::string                buildProgramOptions)
{
    return new csComputeStackEvaluator_OpenCL_MultiDevice(platforms, devices, taskPortions, buildProgramOptions);
}

// std::map<std::string, call_stats::TimeAndCount> GetEvaluatorCallStats(csComputeStackEvaluator_t* cse)
// {
//     std::map<std::string, call_stats::TimeAndCount> stats;
//     //daeComputeStackEvaluator_OpenCL* cse_opencl = dynamic_cast<daeComputeStackEvaluator_OpenCL*>(cse);
//     //if(cse_opencl)
//     //    stats = cse_opencl->GetCallStats();
//     return stats;
// }

std::vector<openclDevice_t> csAvailableOpenCLDevices()
{
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;

    std::vector<openclDevice_t> arrDevices;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++)
    {
        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++)
        {
            openclDevice_t dae_device;
            dae_device.PlatformID = i;
            dae_device.DeviceID   = j;

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            //printf("Platform %d, device %d: %s\n", i+1, j+1, value);
            dae_device.Name = value;
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            //printf("  Hardware version: %s\n", value);
            dae_device.DeviceVersion = value;
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            //printf("  Software version: %s\n", value);
            dae_device.DriverVersion = value;
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            //printf("  OpenCL C version: %s\n", value);
            dae_device.OpenCLVersion = value;
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            //printf("  Parallel compute units: %d\n", maxComputeUnits);
            dae_device.MaxComputeUnits = maxComputeUnits;

            arrDevices.push_back(dae_device);
        }
        free(devices);
    }
    free(platforms);

    return arrDevices;
}

std::vector<openclPlatform_t> csAvailableOpenCLPlatforms()
{
    int i, j;
    char* info;
    size_t infoSize;
    cl_uint platformCount;
    cl_platform_id *platforms;
    const char* attributeNames[5] = { "Name", "Vendor", "Version", "Profile", "Extensions" };
    const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME,
                                                 CL_PLATFORM_VENDOR,
                                                 CL_PLATFORM_VERSION,
                                                 CL_PLATFORM_PROFILE,
                                                 CL_PLATFORM_EXTENSIONS };
    const int attributeCount = sizeof(attributeNames) / sizeof(char*);

    std::vector<openclPlatform_t> arrPlatforms;

    // get platform count
    clGetPlatformIDs(5, NULL, &platformCount);

    // get all platforms
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    // for each platform print all attributes
    for (i = 0; i < platformCount; i++)
    {
        openclPlatform_t dae_platform;
        dae_platform.PlatformID = i;
        //printf("\n %d. Platform \n", i+1);

        {
            j = 0; // CL_PLATFORM_NAME
            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);

            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);
            dae_platform.Name = info;

            //printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
            free(info);
        }
        {
            j = 1; // CL_PLATFORM_VENDOR
            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);

            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);
            dae_platform.Vendor = info;

            //printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
            free(info);
        }
        {
            j = 2; // CL_PLATFORM_VERSION
            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);

            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);
            dae_platform.Version = info;

            //printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
            free(info);
        }
        {
            j = 3; // CL_PLATFORM_PROFILE
            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);

            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);
            dae_platform.Profile = info;

            //printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
            free(info);
        }
        {
            j = 4; // CL_PLATFORM_EXTENSIONS
            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);

            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);
            dae_platform.Extensions = info;

            //printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
            free(info);
        }

        arrPlatforms.push_back(dae_platform);

    }

    free(platforms);

    return arrPlatforms;
}
}
