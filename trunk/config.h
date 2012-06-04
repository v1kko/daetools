#ifndef DAETOOLS_CONFIG_H
#define DAETOOLS_CONFIG_H

#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include "Core/definitions.h"
#define BOOST_FILESYSTEM_VERSION 2

class daeConfig
{
public:
	daeConfig(void)
	{
		configfile = daeConfig::GetConfigFolder() + "daetools.cfg";
		Reload();
	}
	
	virtual ~daeConfig(void)
	{
	}
	
public:
	void Reload(void)
	{
		pt.clear();
		boost::property_tree::info_parser::read_info(configfile, pt);
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
		return daeConfig::GetConfigFolder() + "bonmin.cfg";
	}

	static std::string GetConfigFolder()
	{
        std::string strConfigFolder;
#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
        char* szSysDrive = getenv("SYSTEMDRIVE");
        if(szSysDrive != NULL)
            strConfigFolder = std::string(szSysDrive) + std::string("\\daetools\\");
        else
            strConfigFolder = "c:\\daetools\\";
#else
        strConfigFolder = "/etc/daetools/";
#endif
        return strConfigFolder;
	}

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
