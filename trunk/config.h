#ifndef DAETOOLS_CONFIG_H
#define DAETOOLS_CONFIG_H

#include <string>

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#ifdef DAE_DLL_EXPORTS
#define DAE_API __declspec(dllexport)
#else // CONFIG_EXPORTS
#define DAE_API __declspec(dllimport)
#endif // CONFIG_EXPORTS
#else // WIN32
#define DAE_API
#endif // WIN32

class DAE_API daeConfig
{
public:
    daeConfig();
    virtual ~daeConfig(void);

public:
    void Reload(void);
    bool HasKey(const std::string& strPropertyPath) const;

    bool        GetBoolean(const std::string& strPropertyPath);
    double      GetFloat(const std::string& strPropertyPath);
    int         GetInteger(const std::string& strPropertyPath);
    std::string GetString(const std::string& strPropertyPath);

    bool        GetBoolean(const std::string& strPropertyPath, const bool defValue);
    double      GetFloat(const std::string& strPropertyPath, const double defValue);
    int         GetInteger(const std::string& strPropertyPath, const int defValue);
    std::string GetString(const std::string& strPropertyPath, const std::string& defValue);

    void SetBoolean(const std::string& strPropertyPath, const bool value);
    void SetFloat(const std::string& strPropertyPath, const double value);
    void SetInteger(const std::string& strPropertyPath, const int value);
    void SetString(const std::string& strPropertyPath, const std::string& value);

    static daeConfig& GetConfig(void);
    static std::string GetBONMINOptionsFile();
    static std::string GetConfigFolder();

    std::string toString(void) const;
    std::string GetConfigFileName(void) const;

protected:
    std::string	configfile;
};

DAE_API void       daeSetConfigFile(const std::string& strConfigFile);
DAE_API daeConfig& daeGetConfig(void);

#endif
