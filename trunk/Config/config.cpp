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
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include "../Core/definitions.h"
#include "../config.h"
using namespace dae;

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
            std::cout << "Cannot load daetools.cfg config file in neither 'json' nor 'info' format. Error: " << jsone.message() << std::endl;
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

template<class T>
T daeConfig::Get(const std::string& strPropertyPath)
{
    return g_pt.get<T>(strPropertyPath);
}

template<class T>
T daeConfig::Get(const std::string& strPropertyPath, const T defValue)
{
    return g_pt.get<T>(strPropertyPath, defValue);
}

template<class T>
void daeConfig::Set(const std::string& strPropertyPath, const T value)
{
    boost::optional<T> v = g_pt.get_optional<T>(strPropertyPath);
    if(!v.is_initialized())
    {
        daeDeclareException(exInvalidCall);
        e << "Failed to set the value of the key: " << strPropertyPath << " - the wrong data type";
        throw e;
    }

    g_pt.put<T>(strPropertyPath, value);
}

// Explicit instantiations
template bool        daeConfig::Get<bool>(const std::string& strPropertyPath);
template real_t      daeConfig::Get<real_t>(const std::string& strPropertyPath);
template int         daeConfig::Get<int>(const std::string& strPropertyPath);
template std::string daeConfig::Get<std::string>(const std::string& strPropertyPath);

template bool        daeConfig::Get<bool>(const std::string& strPropertyPath,        const bool defValue);
template real_t      daeConfig::Get<real_t>(const std::string& strPropertyPath,      const real_t defValue);
template int         daeConfig::Get<int>(const std::string& strPropertyPath,         const int defValue);
template std::string daeConfig::Get<std::string>(const std::string& strPropertyPath, const std::string defValue);

template void daeConfig::Set<bool>(const std::string& strPropertyPath, const bool value);
template void daeConfig::Set<real_t>(const std::string& strPropertyPath, const real_t value);
template void daeConfig::Set<int>(const std::string& strPropertyPath, const int value);
template void daeConfig::Set<std::string>(const std::string& strPropertyPath, const std::string value);

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
        cfg_app_folder = cfg_app_folder.parent_path();  // i.e. app_folder

        // If we are in daetools module, GetModuleFileName returns a path of the python exectable, i.e. c:\Python34\python.exe
        cfg_python_daetools_folder = std::string(w_cfg_app_folder.begin(), w_cfg_app_folder.end());
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // python root, i.e. c:\Python34
        // Now we need to append "/Lib/site-packages/daetools" to the python root firectory
        cfg_python_daetools_folder /= "Lib";
        cfg_python_daetools_folder /= "site-packages";
        cfg_python_daetools_folder /= "daetools";
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
        cfg_app_folder = std::string(dl_info.dli_fname); // i.e. app_folder/some_file
        cfg_app_folder = cfg_app_folder.parent_path();   // i.e. app_folder

        cfg_python_daetools_folder = std::string(dl_info.dli_fname);           // i.e. daetools/pyDAE/Linux_x86_64_py27/pyCore.so
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools/pyDAE/Linux_x86_64_py27
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools/pyDAE
        cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools
    }
#endif

    if(boost::filesystem::exists(cfg_home_folder / std::string("daetools.cfg")))
        return cfg_home_folder.string();

    else if(boost::filesystem::exists(cfg_app_folder / std::string("daetools.cfg")))
        return cfg_app_folder.string();

    else if(boost::filesystem::exists(cfg_python_daetools_folder / std::string("daetools.cfg")))
        return cfg_python_daetools_folder.string();

    else
        return std::string();
}

/*
#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
        char* szUserProfile = getenv("USERPROFILE");
        if(szUserProfile != NULL)
        {
            cfg_folder = std::string(szUserProfile);
            cfg_folder /= std::string(".daetools");
            cfg_file = cfg_folder / std::string("daetools.cfg");
            if(boost::filesystem::exists(cfg_file))
                return cfg_folder.string();
        }

        // If not found there, look in the module/exe folder
        wchar_t szModuleFileName[MAX_PATH];
        int returned = GetModuleFileName(NULL, szModuleFileName, MAX_PATH);
        if(returned > 0)
        {
            cfg_folder = std::string(szModuleFileName); // i.e. daetools/pyDAE/Windows_win32_py27/pyCore.pyd
            cfg_folder = cfg_folder.parent_path();      // i.e. daetools/pyDAE/Windows_win32_py27
            cfg_folder = cfg_folder.parent_path();      // i.e. daetools/pyDAE
            cfg_file = cfg_folder / std::string("daetools.cfg");
            if(boost::filesystem::exists(cfg_file))
                return cfg_folder.string();
        }
#else
        char* szUserProfile = getenv("HOME");
        if(szUserProfile != NULL)
        {
            cfg_folder = std::string(szUserProfile);
            cfg_folder /= std::string(".daetools");
            cfg_file = cfg_folder / std::string("daetools.cfg");
            if(boost::filesystem::exists(cfg_file))
                return cfg_folder.string();
        }

        Dl_info dl_info;
        dladdr((void *)daeConfig::GetConfigFolder, &dl_info);
        //printf("module %s loaded\n", dl_info.dli_fname);
        if(dl_info.dli_fname != NULL)
        {
            cfg_folder = std::string(dl_info.dli_fname); // i.e. daetools/pyDAE/Linux_x86_64_py27/pyCore.so
            cfg_folder = cfg_folder.parent_path();       // i.e. daetools/pyDAE/Linux_x86_64_py27
            cfg_folder = cfg_folder.parent_path();       // i.e. daetools/pyDAE
            cfg_file = cfg_folder / std::string("daetools.cfg");
            if(boost::filesystem::exists(cfg_file))
                return cfg_folder.string();
        }
#endif
*/
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

//const boost::property_tree::ptree& daeConfig::GetPropertyTree(void) const
//{
//    return g_pt;
//}

