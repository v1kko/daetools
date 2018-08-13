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
#if !defined(CS_OPENCL_PLATFORMS_H)
#define CS_OPENCL_PLATFORMS_H

#include <string>
#include <vector>
#include "../opencs.h"

namespace cs
{
struct OPENCS_EVALUATORS_API openclPlatform_t
{
    int         PlatformID;
    std::string Name;
    std::string Vendor;
    std::string Version;
    std::string Profile;
    std::string Extensions;
};

struct OPENCS_EVALUATORS_API openclDevice_t
{
    int         PlatformID;
    int         DeviceID;
    std::string Name;
    std::string DeviceVersion;
    std::string DriverVersion;
    std::string OpenCLVersion;
    int         MaxComputeUnits;
};

OPENCS_EVALUATORS_API std::vector<openclDevice_t>   csAvailableOpenCLDevices();
OPENCS_EVALUATORS_API std::vector<openclPlatform_t> csAvailableOpenCLPlatforms();
}
#endif
