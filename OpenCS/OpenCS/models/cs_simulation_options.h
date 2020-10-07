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
#ifndef CS_SIMULATION_OPTIONS_H
#define CS_SIMULATION_OPTIONS_H

#include <string>
#include <sstream>
//#include <boost/property_tree/ptree.hpp>
//#include <boost/property_tree/json_parser.hpp>
//#include <boost/property_tree/detail/file_parser_error.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
#include "cs_model_builder.h"

namespace cs
{
class OPENCS_MODELS_API csSimulationOptions : public csSimulationOptions_t
{
public:
    typedef rapidjson::Value                                rapidjsonValue;
    typedef rapidjson::GenericDocument< rapidjson::UTF8<> > rapidjsonDocument;

    csSimulationOptions()
    {
    }

    void LoadString(const std::string& jsonOptions)
    {
        document.Parse(jsonOptions.c_str(), jsonOptions.size());
        if(document.IsNull())
            csThrowException("Invalid simulation options JSON");
    }

    std::string ToString() const
    {
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter< rapidjson::StringBuffer, rapidjson::ASCII<>, rapidjson::ASCII<> > writer(buffer);
        bool res = document.Accept(writer);
        if(!res)
            csThrowException("csSimulationOptions::ToString: document.Accept failed");
        return buffer.GetString();
    }

    bool GetBoolean(const std::string& strPropertyPath) const
    {
        const rapidjsonValue* obj = getRapidjsonValue(strPropertyPath);
        return obj->GetBool();
    }

    double GetDouble(const std::string& strPropertyPath) const
    {
        const rapidjsonValue* obj = getRapidjsonValue(strPropertyPath);
        return obj->GetDouble();
    }

    int GetInteger(const std::string& strPropertyPath) const
    {
        const rapidjsonValue* obj = getRapidjsonValue(strPropertyPath);
        return obj->GetInt();
    }

    std::string GetString(const std::string& strPropertyPath) const
    {
        const rapidjsonValue* obj = getRapidjsonValue(strPropertyPath);
        return obj->GetString();
    }

    void SetBoolean(const std::string& strPropertyPath, const bool value)
    {
        rapidjsonValue* obj = getRapidjsonValue(strPropertyPath);
        obj->SetBool(value);
    }

    void SetDouble(const std::string& strPropertyPath, const double value)
    {
        rapidjsonValue* obj = getRapidjsonValue(strPropertyPath);
        obj->SetDouble(value);
    }

    void SetInteger(const std::string& strPropertyPath, const int value)
    {
        rapidjsonValue* obj = getRapidjsonValue(strPropertyPath);
        obj->SetInt(value);
    }

    void SetString(const std::string& strPropertyPath, const std::string& value)
    {
        rapidjsonValue* obj = getRapidjsonValue(strPropertyPath);
        obj->SetString(value.c_str(), value.size());
    }

    bool HasKey(const std::string& strPropertyPath) const
    {
        return false;
    }

protected:
    rapidjsonValue* getRapidjsonValue(const std::string& strPropertyPath) const
    {
        typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

        boost::char_separator<char> sep(".");
        tokenizer tokens(strPropertyPath, sep);

        rapidjsonValue* obj = NULL;
        int i = 0;
        BOOST_FOREACH(std::string const& token, tokens)
        {
            if(i == 0)
            {
                if(!document.HasMember(token.c_str()))
                    csThrowException("Invalid option " + token + "in " + strPropertyPath);
                obj = const_cast<rapidjsonValue*>(&document[token.c_str()]);
            }
            else
            {
                if(!obj || !obj->HasMember(token.c_str()))
                    csThrowException("Invalid option " + token + "in " + strPropertyPath);
                obj = &(*obj)[token.c_str()];
            }
            i++;
        }
        if(!obj)
            csThrowException("Invalid option " + strPropertyPath);
        return obj;
    }

protected:
    rapidjsonDocument document;
};


/*
The problem here is that all values are written as quoted strings (even numbers, i.e. "2.0", "12").
Thus, boost property tree implementation is replaced with the rapidjson.
class csSimulationOptions : public csSimulationOptions_t
{
public:
    csSimulationOptions()
    {
    }

    void LoadString(const std::string& jsonOptions)
    {
        pt.clear();
        std::stringstream ss(jsonOptions);
        boost::property_tree::json_parser::read_json(ss, pt);
    }

    std::string ToString() const
    {
        std::stringstream ss;
        boost::property_tree::json_parser::write_json(ss, pt);
        return ss.str();
    }

    bool GetBoolean(const std::string& strPropertyPath) const
    {
        return pt.get<bool>(strPropertyPath);
    }
    double GetDouble(const std::string& strPropertyPath) const
    {
        return pt.get<double>(strPropertyPath);
    }
    int GetInteger(const std::string& strPropertyPath) const
    {
        return pt.get<int>(strPropertyPath);
    }
    std::string GetString(const std::string& strPropertyPath) const
    {
        return pt.get<std::string>(strPropertyPath);
    }

    void SetBoolean(const std::string& strPropertyPath, const bool value)
    {
        pt.put<bool>(strPropertyPath, value);
    }
    void SetDouble(const std::string& strPropertyPath, const double value)
    {
        pt.put<double>(strPropertyPath, value);
    }
    void SetInteger(const std::string& strPropertyPath, const int value)
    {
        pt.put<int>(strPropertyPath, value);
    }
    void SetString(const std::string& strPropertyPath, const std::string& value)
    {
        pt.put<std::string>(strPropertyPath, value);
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

protected:
    boost::property_tree::ptree pt;
};
*/

}

#endif
