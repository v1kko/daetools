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
#include "cs_simulator.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/detail/file_parser_error.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

namespace cs_simulator
{
daeSimulationOptions::daeSimulationOptions()
{
}

daeSimulationOptions::~daeSimulationOptions()
{
}

daeSimulationOptions& daeSimulationOptions::GetConfig()
{
    static daeSimulationOptions cfg;
    return cfg;
}

void daeSimulationOptions::Load(const std::string& jsonFilePath)
{
    pt.clear();
    boost::property_tree::json_parser::read_json(jsonFilePath, pt);
    configFile = jsonFilePath;
}

bool daeSimulationOptions::HasKey(const std::string& strPropertyPath) const
{
    try
    {
        pt.get_child(strPropertyPath);
        return true;
    }
    catch(boost::property_tree::ptree_bad_path& e)
    {
    }

    return false;
}

std::string daeSimulationOptions::ToString(void) const
{
    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, pt);
    return ss.str();
}

bool daeSimulationOptions::GetBoolean(const std::string& strPropertyPath)
{
    return pt.get<bool>(strPropertyPath);
}

double daeSimulationOptions::GetFloat(const std::string& strPropertyPath)
{
    return pt.get<double>(strPropertyPath);
}

int daeSimulationOptions::GetInteger(const std::string& strPropertyPath)
{
    return pt.get<int>(strPropertyPath);
}

std::string daeSimulationOptions::GetString(const std::string& strPropertyPath)
{
    return pt.get<std::string>(strPropertyPath);
}

bool daeSimulationOptions::GetBoolean(const std::string& strPropertyPath, const bool defValue)
{
    return pt.get<bool>(strPropertyPath, defValue);
}

double daeSimulationOptions::GetFloat(const std::string& strPropertyPath, const double defValue)
{
    return pt.get<double>(strPropertyPath, defValue);
}

int daeSimulationOptions::GetInteger(const std::string& strPropertyPath, const int defValue)
{
    return pt.get<int>(strPropertyPath, defValue);
}

std::string daeSimulationOptions::GetString(const std::string& strPropertyPath, const std::string& defValue)
{
    return pt.get<std::string>(strPropertyPath, defValue);
}

}
