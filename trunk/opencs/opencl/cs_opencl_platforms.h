/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#if !defined(CS_OPENCL_PLATFORMS_H)
#define CS_OPENCL_PLATFORMS_H

#include <string>
#include <vector>

namespace cs
{
struct openclPlatform_t
{
    int         PlatformID;
    std::string Name;
    std::string Vendor;
    std::string Version;
    std::string Profile;
    std::string Extensions;
};

struct openclDevice_t
{
    int         PlatformID;
    int         DeviceID;
    std::string Name;
    std::string DeviceVersion;
    std::string DriverVersion;
    std::string OpenCLVersion;
    int         MaxComputeUnits;
};

std::vector<openclDevice_t>   AvailableOpenCLDevices();
std::vector<openclPlatform_t> AvailableOpenCLPlatforms();
}
#endif
