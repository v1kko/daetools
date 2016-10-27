#ifndef DAETOOLS_CONFIG_H
#define DAETOOLS_CONFIG_H

#include <string>
#include <fstream>
#include <sys/stat.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/detail/file_parser_error.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include "Core/definitions.h"

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

class daeConfig
{
public:
    daeConfig(void)
    {
        boost::filesystem::path cfg_file = daeConfig::GetConfigFolder();
        cfg_file /= "daetools.cfg";
        configfile = cfg_file.string();
        //std::cout << "using config file: " << cfg_file.string() << std::endl;
        Reload();
    }

    virtual ~daeConfig(void)
    {
    }

public:
    void Reload(void)
    {
        try
        {
            // First try to read the config file in "json" format
            pt.clear();
            boost::property_tree::json_parser::read_json(configfile, pt);
        }
        catch(boost::property_tree::file_parser_error& jsone)
        {
            // If json fails, try to read the same file in "info" format (this is just a transitional feature)
            try
            {
                boost::property_tree::info_parser::read_info(configfile, pt);
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

    bool HasKey(const std::string& strPropertyPath) const
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

    template<class T>
    T Get(const std::string& strPropertyPath)
    {
        return pt.get<T>(strPropertyPath);
    }

    template<class T>
    T Get(const std::string& strPropertyPath, const T defValue)
    {
        return pt.get<T>(strPropertyPath, defValue);
    }

    template<class T>
    void Set(const std::string& strPropertyPath, const T value)
    {
        pt.put<T>(strPropertyPath, value);
    }

    static daeConfig& GetConfig(void)
    {
        static daeConfig config;
        return config;
    }

    static std::string GetBONMINOptionsFile()
    {
        boost::filesystem::path cfg_file = daeConfig::GetConfigFolder();
        cfg_file /= "bonmin.cfg";
        return cfg_file.string();
    }

    static std::string GetConfigFolder()
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

            cfg_python_daetools_folder = std::string(w_cfg_app_folder.begin(), w_cfg_app_folder.end()); // i.e. daetools/pyDAE/Windows_win32_py27/pyCore.pyd
            cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools/pyDAE/Windows_win32_py27
            cfg_python_daetools_folder = cfg_python_daetools_folder.parent_path(); // i.e. daetools/pyDAE
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
    std::string toString(void) const
    {
        std::stringstream ss;
        boost::property_tree::info_parser::write_info(ss, pt);
        return ss.str();
    }

    const boost::property_tree::ptree& GetPropertyTree(void) const
    {
        return pt;
    }

protected:
    boost::property_tree::ptree pt;
    std::string					configfile;
};


#endif
