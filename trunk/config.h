#ifndef DAETOOLS_CONFIG_H
#define DAETOOLS_CONFIG_H

#include <string>

class daeConfig
{
public:
    daeConfig();
    virtual ~daeConfig(void);

public:
    void Reload(void);
    bool HasKey(const std::string& strPropertyPath) const;

    template<class T>
    T Get(const std::string& strPropertyPath);

    template<class T>
    T Get(const std::string& strPropertyPath, const T defValue);

    template<class T>
    void Set(const std::string& strPropertyPath, const T value);

    static daeConfig& GetConfig(void);
    static std::string GetBONMINOptionsFile();
    static std::string GetConfigFolder();

    std::string toString(void) const;
    std::string GetConfigFileName(void) const;

protected:
    std::string	configfile;
};

void       daeSetConfigFile(const std::string& strConfigFile);
daeConfig& daeGetConfig(void);

#endif
