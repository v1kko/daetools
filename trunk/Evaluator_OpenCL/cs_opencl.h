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
#if !defined(CS_OPENCL_H)
#define CS_OPENCL_H

#include <string>
#include <vector>
#include "../Core/compute_stack.h"
using namespace computestack;

typedef struct openclPlatform
{
    int         PlatformID;
    std::string Name;
    std::string Vendor;
    std::string Version;
    std::string Profile;
    std::string Extensions;
}openclPlatform_t;

typedef struct openclDevice
{
    int         PlatformID;
    int         DeviceID;
    std::string Name;
    std::string DeviceVersion;
    std::string DriverVersion;
    std::string OpenCLVersion;
    int         MaxComputeUnits;
}openclDevice_t;

std::vector<openclDevice_t>   AvailableOpenCLDevices();
std::vector<openclPlatform_t> AvailableOpenCLPlatforms();
adComputeStackEvaluator_t*    CreateComputeStackEvaluator(int platformID, int deviceID, std::string buildProgramOptions = "");
adComputeStackEvaluator_t*    CreateComputeStackEvaluator_multi(const std::vector<int>&    platforms,
                                                                const std::vector<int>&    devices,
                                                                const std::vector<double>& taskPortions,
                                                                std::string                buildProgramOptions = "");

#endif
