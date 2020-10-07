#ifndef DAE_FMI_AUXILIARY_H
#define DAE_FMI_AUXILIARY_H

#include <cstdio>
#include <boost/lexical_cast.hpp>
#include "include/encode.hpp"
#include "fmi_component.h"
/*
double hexfloat_to_double(const char* val_hex_str)
{
    double value = 0;
    int res = ::sscanf(val_hex_str, "%lA", &value);
    if(res <= 0)
        throw std::runtime_error((boost::format("Invalid hexfloat received: %s\n") % val_hex_str).str());
    return value;
}

std::string double_to_hexfloat(double value)
{
    char hex_value[25] = {0};
    int res = ::snprintf(hex_value, 25, "%A", value);
    if(res < 0)
        throw std::runtime_error((boost::format("Values %f cannot be converted to hexfloat string\n") % value).str());
    return hex_value;
}

double str_to_double(const char* val_hex_str)
{
    return std::atof(val_hex_str);
    //double value = 0;
    //int res = ::sscanf(val_hex_str, "%f", &value);
    //if(res <= 0)
    //    throw std::runtime_error((boost::format("Invalid hexfloat received: %s\n") % val_hex_str).str());
    //return value;
}
*/

std::string double_to_str(double value)
{
    char hex_value[64] = {0};
    int res = ::snprintf(hex_value, 64, "%.16f", value);
    if(res < 0)
        throw std::runtime_error((boost::format("Values %f cannot be converted to hexfloat string\n") % value).str());
    return hex_value;
}

std::string array_to_json_str(const fmi2ValueReference vr[], size_t nvr)
{
    std::string json_str = "[";
    for(size_t i = 0; i < nvr; i++)
    {
        json_str += (i == 0 ? "" : ",");
        json_str += boost::lexical_cast<std::string>(vr[i]);
    }
    json_str += "]";
    return boost::network::uri::encoded(json_str);
}

std::string array_to_json_str(const fmi2Real value[], size_t nvr)
{
    std::string json_str = "[";
    for(size_t i = 0; i < nvr; i++)
    {
        json_str += (i == 0 ? "" : ",");
        json_str +=  double_to_str(value[i]);
    }
    json_str += "]";
    return boost::network::uri::encoded(json_str);
}

std::string array_to_json_str(const fmi2String value[], size_t nvr)
{
    std::string json_str = "[";
    for(size_t i = 0; i < nvr; i++)
    {
        json_str += (i == 0 ? "" : ",");
        json_str +=  "\"" + std::string(value[i]) + "\"";
    }
    json_str += "]";
    return boost::network::uri::encoded(json_str);
}

#endif
