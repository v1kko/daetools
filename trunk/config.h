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
/*
		std::string strHOME;
		char* pPath;
#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
		pPath = getenv("\\%HOMEDRIVE%\\%HOMEPATH\\%");
#else
		pPath = getenv("HOME");
#endif
		if(!pPath) 
		{
			daeDeclareException(dae::exUnknown);
			e << "Cannot get the $HOME folder path";
			throw e;
		}
		strHOME = pPath;
		configfile = strHOME + "/.daetools/daetools.cfg";
		Reload();
*/
	
#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
		return std::string("c:\\daetools\\");
#else
		return std::string("/etc/daetools/");
#endif
	}
	
protected:
	boost::property_tree::ptree pt;
	std::string					configfile;	
};


#endif
