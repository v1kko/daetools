/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2015
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include <fstream>
#include <sys/stat.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/detail/file_parser_error.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include "../Core/definitions.h"
#include "../config.h"
using namespace dae;

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#include <Windows.h>
#endif

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

static std::string g_configfile = "";
static boost::property_tree::ptree g_pt;

void daeSetConfigFile(const std::string& strConfigFile)
{
    g_configfile = strConfigFile;
}

daeConfig& daeGetConfig(void)
{
    return daeConfig::GetConfig();
}

daeConfig::daeConfig()
{
    // std::cout << "\n\nLoaded config " << rand() << std::endl << std::endl;
    if(g_configfile.empty())
    {
        boost::filesystem::path cfg_file = daeConfig::GetConfigFolder();
        cfg_file /= "daetools.cfg";
        configfile = cfg_file.string();
    }
    else
    {
        configfile = g_configfile;
    }

    Reload();
}

daeConfig::~daeConfig(void)
{
}

void daeConfig::Reload(void)
{
    try
    {
        // First try to read the config file in "json" format
        g_pt.clear();
        boost::property_tree::json_parser::read_json(configfile, g_pt);
    }
    catch(boost::property_tree::file_parser_error& jsone)
    {
        // If json fails, try to read the same file in "info" format (this is just a transitional feature)
        try
        {
            boost::property_tree::info_parser::read_info(configfile, g_pt);
        }
        catch(boost::property_tree::file_parser_error& infoe)
        {
            std::cout << "Cannot load daetools.cfg config file [" << configfile << "] in neither 'json' nor 'info' format. Error: " << jsone.message() << std::endl;
            std::cout << "Config files are located in: (1)current_exe_directory, (b).../daetools or (c)$HOME/.daetools directory" << std::endl;
            return;
        }

        std::cout << "Config file is in deprecated 'info' format - switch to the new 'json' format." << std::endl;
        std::cout << "Config files are located in /etc/daetools, c:/daetools or $HOME/.daetools directory" << std::endl;
    }
}

bool daeConfig::HasKey(const std::string& strPropertyPath) const
{
    try
    {
        g_pt.get_child(strPropertyPath);
        return true;
    }
    catch(boost::property_tree::ptree_bad_path& e)
    {
    }

    return false;
}

bool daeConfig::GetBoolean(const std::string& strPropertyPath)
{
    return g_pt.get<bool>(strPropertyPath);
}

double daeConfig::GetFloat(const std::string& strPropertyPath)
{
    return g_pt.get<double>(strPropertyPath);
}

int daeConfig::GetInteger(const std::string& strPropertyPath)
{
    return g_pt.get<int>(strPropertyPath);
}

std::string daeConfig::GetString(const std::string& strPropertyPath)
{
    return g_pt.get<std::string>(strPropertyPath);
}

bool daeConfig::GetBoolean(const std::string& strPropertyPath, const bool defValue)
{
    return g_pt.get<bool>(strPropertyPath, defValue);
}

double daeConfig::GetFloat(const std::string& strPropertyPath, const double defValue)
{
    return g_pt.get<double>(strPropertyPath, defValue);
}

int daeConfig::GetInteger(const std::string& strPropertyPath, const int defValue)
{
    return g_pt.get<int>(strPropertyPath, defValue);
}

std::string daeConfig::GetString(const std::string& strPropertyPath, const std::string& defValue)
{
    return g_pt.get<std::string>(strPropertyPath, defValue);
}

void daeConfig::SetBoolean(const std::string& strPropertyPath, const bool value)
{
    boost::optional<bool> v = g_pt.get_optional<bool>(strPropertyPath);
    if(!v.is_initialized())
    {
        daeDeclareException(exInvalidCall);
        e << "Failed to set the value of the key: " << strPropertyPath << " - the wrong data type";
        throw e;
    }
    g_pt.put<bool>(strPropertyPath, value);
}

void daeConfig::SetFloat(const std::string& strPropertyPath, const double value)
{
    boost::optional<double> v = g_pt.get_optional<double>(strPropertyPath);
    if(!v.is_initialized())
    {
        daeDeclareException(exInvalidCall);
        e << "Failed to set the value of the key: " << strPropertyPath << " - the wrong data type";
        throw e;
    }
    g_pt.put<double>(strPropertyPath, value);
}

void daeConfig::SetInteger(const std::string& strPropertyPath, const int value)
{
    boost::optional<int> v = g_pt.get_optional<int>(strPropertyPath);
    if(!v.is_initialized())
    {
        daeDeclareException(exInvalidCall);
        e << "Failed to set the value of the key: " << strPropertyPath << " - the wrong data type";
        throw e;
    }
    g_pt.put<int>(strPropertyPath, value);
}

void daeConfig::SetString(const std::string& strPropertyPath, const std::string& value)
{
    boost::optional<std::string> v = g_pt.get_optional<std::string>(strPropertyPath);
    if(!v.is_initialized())
    {
        daeDeclareException(exInvalidCall);
        e << "Failed to set the value of the key: " << strPropertyPath << " - the wrong data type";
        throw e;
    }
    g_pt.put<std::string>(strPropertyPath, value);
}

daeConfig& daeConfig::GetConfig(void)
{
    static daeConfig cfg;
    return cfg;
}

std::string daeConfig::GetBONMINOptionsFile()
{
    boost::filesystem::path cfg_file = daeConfig::GetConfigFolder();
    cfg_file /= "bonmin.cfg";
    return cfg_file.string();
}

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
extern HMODULE config_hModule;
#endif

std::string daeConfig::GetConfigFolder()
{
    boost::filesystem::path cfg_home_folder,
                            cfg_app_folder,
                            cfg_python_daetools_folder;

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
    char* szUserProfile = getenv("USERPROFILE");
    if(szUserProfile != NULL)
    {
        cfg_home_folder = std::string(szUserProfile);
        cfg_home_folder /= std::string(".daetools");
    }

    wchar_t szModuleFileName[MAX_PATH];
    int returned = GetModuleFileName(NULL, szModuleFileName, MAX_PATH);
    if(returned > 0)
    {
        std::wstring w_cfg_app_folder(szModuleFileName); // i.e. app_folder/some_file.exe

        cfg_app_folder = std::string(w_cfg_app_folder.begin(), w_cfg_app_folder.end());
        //printf("szModuleFileName folder %s\n", cfg_app_folder.string().c_str());
        cfg_app_folder = cfg_app_folder.parent_path();  // i.e. app_folder
    }

    // GetModuleFileName with this .dll handle returns a path of this dll (not the executable).
    returned = GetModuleFileName(config_hModule, szModuleFileName, MAX_PATH);
    if(returned > 0)
    {
        std::wstring w_szModuleFileName(szModuleFileName);
        boost::filesystem::path canonical_config_path = std::string(w_szModuleFileName.begin(), w_szModuleFileName.end());
        canonical_config_path = boost::filesystem::weakly_canonical(canonical_config_path);

        cfg_python_daetools_folder = canonical_config_path;                    // i.e. daetools/solibs/Windows_win32/libcdaeConfig-py27.so
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools/solibs/Windows_win32
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools/solibs
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools
    }
#else
    char* szUserProfile = getenv("HOME");
    if(szUserProfile != NULL)
    {
        cfg_home_folder = std::string(szUserProfile);
        cfg_home_folder /= std::string(".daetools");
    }

    Dl_info dl_info;
    dladdr((void *)daeConfig::GetConfigFolder, &dl_info);
    //printf("module %s loaded\n", dl_info.dli_fname);
    if(dl_info.dli_fname != NULL)
    {
        boost::filesystem::path canonical_config_path = std::string(dl_info.dli_fname); // i.e. app_folder/some_file
        canonical_config_path = boost::filesystem::weakly_canonical(canonical_config_path);

        cfg_app_folder = canonical_config_path;
        cfg_app_folder = cfg_app_folder.parent_path();   // i.e. app_folder

        cfg_python_daetools_folder = canonical_config_path;                    // i.e. daetools/solibs/Linux_x86_64/libcdaeConfig-py27.so
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools/solibs/Linux_x86_64_py27
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools/solibs
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools
    }
#endif
    //printf("HOME folder %s\n", cfg_home_folder.string().c_str());
    //printf("APP folder %s\n", cfg_app_folder.string().c_str());
    //printf("daetools folder %s\n", cfg_python_daetools_folder.string().c_str());

    if(boost::filesystem::exists(cfg_home_folder / std::string("daetools.cfg")))
        return cfg_home_folder.string();

    else if(boost::filesystem::exists(cfg_app_folder / std::string("daetools.cfg")))
        return cfg_app_folder.string();

    else if(boost::filesystem::exists(cfg_python_daetools_folder / std::string("daetools.cfg")))
        return cfg_python_daetools_folder.string();

    else
        return std::string();
}

std::string daeConfig::toString(void) const
{
    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, g_pt);
    return ss.str();
}

std::string daeConfig::GetConfigFileName(void) const
{
    return configfile;
}
