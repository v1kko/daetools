#include "python_wraps.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
//#define NO_IMPORT_ARRAY
//#include <noprefix.h>
using namespace std;
using namespace boost::python;
#include <boost/property_tree/json_parser.hpp>
#include "../Core/units_io.h"
#include <limits>

namespace daepython
{
/*******************************************************
    Common
*******************************************************/
daeDomainIndex CreateDomainIndex(object& o)
{
    extract<size_t>          size(o);
    extract<daeDEDI*>        DEDI(o);
    extract<daeDomainIndex>  domainIndex(o);

    if(size.check())
    {
        size_t n = size();
        return daeDomainIndex(n);
    }
    else if(DEDI.check())
    {
        daeDEDI* pDEDI = DEDI();
        return daeDomainIndex(pDEDI);
    }
    else if(domainIndex.check())
    {
        return domainIndex();
    }
    else
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid argument (must be integer, daeDomainIndex or daeDEDI object)" ;
        throw e;
        return daeDomainIndex();
    }
}

#define parRANGE(I) CreateArrayRange(o##I, param.GetDomain(I-1))
#define varRANGE(I) CreateArrayRange(o##I, var.GetDomain(I-1))

daeArrayRange CreateArrayRange(object& o, daeDomain* pDomain)
{
    extract<int>				  get_int(o);
    extract<daeDEDI*>			  get_DEDI(o);
    extract<daeIndexRange>		  get_IndexRange(o);
    extract<boost::python::list>  get_list(o);
    extract<boost::python::slice> get_slice(o);
    extract<char>                 get_char(o);

    if(get_DEDI.check())
    {
        daeDEDI* pDEDI = get_DEDI();
        return daeArrayRange(pDEDI);
    }
    else if(get_int.check())
    {
        int iIndex = get_int();

    // If < 0 get ALL points
    // Otherwise take the point defined with iIndex
        if(iIndex < 0)
        {
            daeIndexRange ir(pDomain);
            return daeArrayRange(ir);
        }
        else
        {
            return daeArrayRange(size_t(iIndex));
        }
    }
    else if(get_IndexRange.check())
    {
        daeIndexRange ir = get_IndexRange();
        return daeArrayRange(ir);
    }
    else if(get_list.check())
    {
        std::vector<size_t> narrCustomPoints;
        boost::python::list l = get_list();
        boost::python::ssize_t n = boost::python::len(l);

        // If list is empty get ALL points
        if(n == 0)
        {
            daeIndexRange ir(pDomain);
            return daeArrayRange(ir);
        }
        else
        {
            narrCustomPoints.resize(n);
            for(boost::python::ssize_t i = 0; i < n; i++)
                narrCustomPoints[i] = boost::python::extract<size_t>( l[i] );

            daeIndexRange ir(pDomain, narrCustomPoints);
            return daeArrayRange(ir);
        }
    }
    else if(get_slice.check())
    {
        boost::python::slice s = get_slice();

        extract<int> get_start(s.start());
        extract<int> get_end(s.stop());
        extract<int> get_step(s.step());

        int start = get_start.check() ? get_start() : 0;
        int end   = get_end.check()   ? get_end()   : pDomain->GetNumberOfPoints()-1;
        int step  = get_step.check()  ? get_step()  : 1;

        //std::cout << (boost::format("slice(%1%, %2%, %3%)") % start % end % step).str() << std::endl;

        daeIndexRange ir(pDomain, start, end, step);
        return daeArrayRange(ir);
    }
    else if(get_char.check())
    {
        char cIndex = get_char();

    // If char is '* get ALL points
        if(cIndex == '*')
        {
            daeIndexRange ir(pDomain);
            return daeArrayRange(ir);
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid string argument " << cIndex << " in the array_xxx() function call (can only be '*')";
            throw e;
        }
    }
    else
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid argument of the array_xxx function call (must be integer, daeDEDI object, slice, list of integers, or '*' character)" ;
        throw e;
    }
    return daeArrayRange();
}

boost::python::list daeVariableWrapper_GetDomainIndexes(daeVariableWrapper &self)
{
    return getListFromVectorByValue(self.m_narrDomainIndexes);
}

daeVariable* daeVariableWrapper_GetVariable(daeVariableWrapper &self)
{
    return boost::ref(self.m_pVariable);
}

boost::python::object daeGetConfig(void)
{
    return boost::python::object( boost::ref(daeConfig::GetConfig()) );
}

bool GetBoolean(daeConfig& self, const std::string& strPropertyPath)
{
    return self.Get<bool>(strPropertyPath);
}

real_t GetFloat(daeConfig& self, const std::string& strPropertyPath)
{
    return self.Get<real_t>(strPropertyPath);
}

int GetInteger(daeConfig& self, const std::string& strPropertyPath)
{
    return self.Get<int>(strPropertyPath);
}

std::string GetString(daeConfig& self, const std::string& strPropertyPath)
{
    return self.Get<std::string>(strPropertyPath);
}

bool GetBoolean1(daeConfig& self, const std::string& strPropertyPath, const bool defValue)
{
    return self.Get<bool>(strPropertyPath, defValue);
}

real_t GetFloat1(daeConfig& self, const std::string& strPropertyPath, const real_t defValue)
{
    return self.Get<real_t>(strPropertyPath, defValue);
}

int GetInteger1(daeConfig& self, const std::string& strPropertyPath, const int defValue)
{
    return self.Get<int>(strPropertyPath, defValue);
}

std::string GetString1(daeConfig& self, const std::string& strPropertyPath, const std::string defValue)
{
    return self.Get<std::string>(strPropertyPath, defValue);
}

void SetBoolean(daeConfig& self, const std::string& strPropertyPath, bool value)
{
    self.Set<bool>(strPropertyPath, value);
}

void SetFloat(daeConfig& self, const std::string& strPropertyPath, real_t value)
{
    self.Set<real_t>(strPropertyPath, value);
}

void SetInteger(daeConfig& self, const std::string& strPropertyPath, int value)
{
    self.Set<int>(strPropertyPath, value);
}

void SetString(daeConfig& self, const std::string& strPropertyPath, std::string value)
{
    self.Set<std::string>(strPropertyPath, value);
}

std::string daeConfig__str__(daeConfig& self)
{
    return self.toString();
}

std::string daeConfig__repr__(daeConfig& self)
{
    string msg = "daeConfig(\"%s\")";
    return (boost::format(msg) % daeConfig::GetConfigFolder()).str();
}

boost::python::object daeConfig__contains__(daeConfig& self, boost::python::object key)
{
    extract<string>  str_key(key);
    if(!str_key.check())
    {
        daeDeclareException(exInvalidCall);
        e << "The key in daeConfig must be a string" ;
        throw e;
    }
    return boost::python::object(self.HasKey(str_key()));
}

boost::python::object daeConfig_has_key(daeConfig& self, boost::python::object key)
{
    extract<string>  str_key(key);
    if(!str_key.check())
    {
        daeDeclareException(exInvalidCall);
        e << "The key in daeConfig must be a string" ;
        throw e;
    }
    return boost::python::object(self.HasKey(str_key()));
}

boost::python::object daeConfig__getitem__(daeConfig& self, boost::python::object key)
{
    extract<string>  str_key(key);
    if(!str_key.check())
    {
        daeDeclareException(exInvalidCall);
        e << "The key in daeConfig must be a string" ;
        throw e;
    }
    if(!self.HasKey(str_key()))
    {
        daeDeclareException(exInvalidCall);
        e << "The key " << str_key() << " does not exist in daeConfig";
        throw e;
    }

    try
    {
        return boost::python::object(self.Get<bool>(str_key()));
    }
    catch(boost::property_tree::ptree_error& e)
    {
    }

    try
    {
        return boost::python::object(self.Get<int>(str_key()));
    }
    catch(boost::property_tree::ptree_error& e)
    {
    }

    try
    {
        return boost::python::object(self.Get<double>(str_key()));
    }
    catch(boost::property_tree::ptree_error& e)
    {
    }

    try
    {
        return boost::python::object(self.Get<string>(str_key()));
    }
    catch(boost::property_tree::ptree_error& e)
    {
    }

    daeDeclareException(exInvalidCall);
    e << "The value in daeConfig is none of: string | boolean | integer | float" ;
    throw e;

    return boost::python::object();
}

void daeConfig__setitem__(daeConfig& self, boost::python::object key, boost::python::object value)
{
    extract<string>  str_key(key);
    extract<string>  string_(value);
    extract<bool>    bool_(value);
    extract<int>     int_(value);
    extract<double>  float_(value);

    if(!str_key.check())
    {
        daeDeclareException(exInvalidCall);
        e << "The key in daeConfig must be a string" ;
        throw e;
    }
    if(!self.HasKey(str_key()))
    {
        daeDeclareException(exInvalidCall);
        e << "The key " << str_key() << " does not exist in daeConfig";
        throw e;
    }

    if(bool_.check())
    {
        boost::optional<bool> v = self.GetPropertyTree().get_optional<bool>(str_key());
        if(!v.is_initialized())
        {
            daeDeclareException(exInvalidCall);
            e << "Failed to set the value of the key: the wrong data type" << str_key();
            throw e;
        }

        self.Set<bool>(str_key(), bool_());
    }
    else if(int_.check())
    {
        boost::optional<int> v = self.GetPropertyTree().get_optional<int>(str_key());
        if(!v.is_initialized())
        {
            daeDeclareException(exInvalidCall);
            e << "Failed to set the value of the key: the wrong data type" << str_key();
            throw e;
        }

        self.Set<int>(str_key(), int_());
    }
    else if(float_.check())
    {
        boost::optional<double> v = self.GetPropertyTree().get_optional<double>(str_key());
        if(!v.is_initialized())
        {
            daeDeclareException(exInvalidCall);
            e << "Failed to set the value of the key: the wrong data type" << str_key();
            throw e;
        }

        self.Set<double>(str_key(), float_());
    }
    else if(string_.check())
    {
        boost::optional<string> v = self.GetPropertyTree().get_optional<string>(str_key());
        if(!v.is_initialized())
        {
            daeDeclareException(exInvalidCall);
            e << "Failed to set the value of the key: the wrong data type" << str_key();
            throw e;
        }

        self.Set<string>(str_key(), string_());
    }
    else
    {
        daeDeclareException(exInvalidCall);
        e << "The value in daeConfig can only be one of: string|boolean|integer|float" ;
        throw e;
    }
}

/*******************************************************
    __str__ and __repr__ funkcije
*******************************************************/
string daeVariableType__repr__(const daeVariableType& self)
{
    daeModelExportContext c;
    c.m_nPythonIndentLevel = 0;
    c.m_bExportDefinition  = true;
    c.m_pModel             = NULL;
    string strUnits = units::Export(ePYDAE, c, self.GetUnits());

    return (boost::format("daeVariableType(name=\"%1%\", units=%2%, lowerBound=%3%, upperBound=%4%, initialGuess=%5%, absoluteTolerance=%6%)") %
            self.GetName() % strUnits % self.GetLowerBound() %
            self.GetUpperBound() % self.GetInitialGuess() % self.GetAbsoluteTolerance()).str();
}

string daeDomainIndex__repr__(const daeDomainIndex& self)
{
    if(self.m_eType == eConstantIndex)
    {
        return (boost::format("daeDomainIndex(index=%1%)") % self.m_nIndex).str();
    }
    else if(self.m_eType == eDomainIterator)
    {
        if(!self.m_pDEDI)
            daeDeclareAndThrowException(exInvalidPointer);
        return (boost::format("daeDomainIndex(dedi=%1%)") % daeDEDI__repr__(*self.m_pDEDI)).str();
    }
    else if(self.m_eType == eIncrementedDomainIterator)
    {
        if(!self.m_pDEDI)
            daeDeclareAndThrowException(exInvalidPointer);
        return (boost::format("daeDomainIndex(dedi=%1%, increment=%2%)") % daeDEDI__repr__(*self.m_pDEDI) % self.m_iIncrement).str();
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }
    return string("");
}

string daeIndexRange__repr__(const daeIndexRange& self)
{
    if(self.m_eType == eAllPointsInDomain)
    {
        return (boost::format("daeIndexRange(domain=%1%)") % self.m_pDomain->GetStrippedName()).str();
    }
    else if(self.m_eType == eRangeOfIndexes)
    {
        return (boost::format("daeIndexRange(domain=%1%, startIndex=%2%, stopIndex=%3%, step=%4%)") % self.m_pDomain->GetStrippedName()
                                                                                                    % self.m_iStartIndex
                                                                                                    % self.m_iEndIndex
                                                                                                    % self.m_iStride).str();
    }
    else if(self.m_eType == eCustomRange)
    {
        string strItems = "[" + toString(self.m_narrCustomPoints) + "]";
        return (boost::format("daeIndexRange(domain=%1%, customPoints=%2%)") % self.m_pDomain->GetStrippedName()
                                                                             % strItems).str();
    }
    else
    {
        daeDeclareAndThrowException(exInvalidCall)
    }
    return string("");
}

string daeArrayRange__str__(const daeArrayRange& self)
{
    return self.GetRangeAsString();
}

string daeArrayRange__repr__(const daeArrayRange& self)
{
    if(self.m_eType == eRangeDomainIndex)
    {
        return (boost::format("daeArrayRange(domainIndex=%1%)") % daeDomainIndex__repr__(self.m_domainIndex)).str();
    }
    else if(self.m_eType == eRange)
    {
        return (boost::format("daeArrayRange(indexRange=%1%)") % daeIndexRange__repr__(self.m_Range)).str();
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }
    return string("");
}

string daeDomain__str__(const daeDomain& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeDomain__repr__(const daeDomain& self)
{
    daeModelExportContext c;
    c.m_nPythonIndentLevel = 0;
    c.m_bExportDefinition  = true;
    c.m_pModel             = NULL;
    string strUnits = units::Export(ePYDAE, c, self.GetUnits());

    if(self.GetParentPort())
        return (boost::format("daeDomain(name=\"%1%\", parentPort=%2%, units=%3%, description=\"%4%\")") %
                self.GetName() % daeGetStrippedRelativeName(NULL, self.GetParentPort()) % strUnits % self.GetDescription()).str();
    else
        return (boost::format("daeDomain(name=\"%1%\", parentModel=%2%, units=%3%, description=\"%4%\")") %
                self.GetName() % daeGetStrippedRelativeName(NULL, dynamic_cast<daeModel*>(self.GetModel())) % strUnits % self.GetDescription()).str();
}

string daeParameter__str__(const daeParameter& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeParameter__repr__(const daeParameter& self)
{
    daeModelExportContext c;
    c.m_nPythonIndentLevel = 0;
    c.m_bExportDefinition  = true;
    c.m_pModel             = NULL;

    string strUnits   = units::Export(ePYDAE, c, self.GetUnits());
    string strDomains = "[" + toString_StrippedRelativeNames<daeDomain*, daeModel*>(self.Domains(), NULL) + "]";

    if(self.GetParentPort())
        return (boost::format("daeParameter(name=\"%1%\", units=%2%, parentPort=%3%, description=\"%4%\", domains=%5%)") %
                self.GetName() % strUnits % daeGetStrippedRelativeName(NULL, self.GetParentPort()) %
                self.GetDescription() % strDomains).str();
    else
        return (boost::format("daeParameter(name=\"%1%\", units=%2%, parentModel=%3%, description=\"%4%\", domains=%5%)") %
                self.GetName() % strUnits % daeGetStrippedRelativeName(NULL, dynamic_cast<daeModel*>(self.GetModel())) %
                self.GetDescription() % strDomains).str();
}

string daeVariable__str__(const daeVariable& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeVariable__repr__(const daeVariable& self)
{
    string strDomains = "[" + toString_StrippedRelativeNames<daeDomain*, daeModel*>(self.Domains(), NULL) + "]";
    if(self.GetParentPort())
        return (boost::format("daeVariable(name=\"%1%\", variableType=%2%, parentPort=%3%, description=\"%4%\", domains=%5%)") %
                self.GetName() % self.GetVariableType()->GetName() % daeGetStrippedRelativeName(NULL, self.GetParentPort()) %
                self.GetDescription() % strDomains).str();
    else
        return (boost::format("daeVariable(name=\"%1%\", variableType=%2%, parentModel=%3%, description=\"%4%\", domains=%5%)") %
                self.GetName() % self.GetVariableType()->GetName() % daeGetStrippedRelativeName(NULL, dynamic_cast<daeModel*>(self.GetModel())) %
                self.GetDescription() % strDomains).str();
}

string daePort__str__(const daePort& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daePort__repr__(const daePort& self)
{
    string strDomains    = getVector__repr__(self.Domains(),    &daeDomain__str__);
    string strParameters = getVector__repr__(self.Parameters(), &daeParameter__str__);
    string strVariables  = getVector__repr__(self.Variables(),  &daeVariable__str__);
    string strPortType   = g_EnumTypesCollection->esmap_daeePortType.GetString(self.GetType());
    string strParent     = self.GetModel() ? daeGetStrippedRelativeName(NULL, dynamic_cast<daeModel*>(self.GetModel())) : "None";

    string fmt = "daePort(name=\"%1%\", type=%2%, parentModel=%3%, description=\"%4%\", "
                         "domains=%5%, parameters=%6%, variables=%7%)";
    return (boost::format(fmt) % self.GetName() % strPortType % strParent % self.GetDescription()
                               % strDomains % strParameters % strVariables).str();
}

string daeEventPort__str__(const daeEventPort& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeEventPort__repr__(const daeEventPort& self)
{
    string strPortType = g_EnumTypesCollection->esmap_daeePortType.GetString(self.GetType());
    string strParent   = self.GetModel() ? daeGetStrippedRelativeName(NULL, dynamic_cast<daeModel*>(self.GetModel())) : "None";

    return (boost::format("daeEventPort(name=\"%1%\", type=%2%, parentModel=%3%, description=\"%4%\")") %
                           self.GetName() % strPortType % strParent % self.GetDescription()).str();
}

string daeModel__str__(const daeModel& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeModel__repr__(const daeModel& self)
{
    string strDomains    = getVector__repr__(self.Domains(),    &daeDomain__str__);
    string strParameters = getVector__repr__(self.Parameters(), &daeParameter__str__);
    string strVariables  = getVector__repr__(self.Variables(),  &daeVariable__str__);
    string strEquations  = getVector__repr__(self.Equations(),  &daeEquation__str__);
    string strSTNs       = getVector__repr__(self.STNs(),       &daeSTN__str__);
    string strPorts      = getVector__repr__(self.Ports(),      &daePort__str__);
    string strEventPorts = getVector__repr__(self.EventPorts(), &daeEventPort__str__);
    string strUnits      = getVector__repr__(self.Models(),     &daeModel__str__);
    string strParent     = self.GetModel() ? daeGetStrippedRelativeName(NULL, dynamic_cast<daeModel*>(self.GetModel())) : "None";

    string fmt = "daeModel(name=\"%1%\", parentModel=%2%, description=\"%3%\", "
                          "domains=%4%, parameters=%5%, variables=%6%, equations=%7%, "
                          "stns=%8%, ports=%9%, eventPorts=%10%, units=%11%)";
    return (boost::format(fmt) % self.GetName() % strParent % self.GetDescription() %
            strDomains % strParameters % strVariables % strEquations %
            strSTNs % strPorts % strEventPorts % strUnits).str();
}

string daeEquation__str__(const daeEquation& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeEquation__repr__(const daeEquation& self)
{
    string strResidual = adouble__str__(self.GetResidual());
    string strDEDIs    = getVector__repr__(self.GetDEDIs(), &daeDEDI__repr__);

    if(self.GetParentState())
        return (boost::format("daeEquation(name=\"%1%\", parentState=%2%, description=\"%3%\", distributedDomains=%4%, residual=%5%)") %
                self.GetName() %
                daeGetStrippedRelativeName(NULL, self.GetParentState()) %
                self.GetDescription() % strDEDIs % strResidual).str();
    else
        return (boost::format("daeEquation(name=\"%1%\", parentModel=%2%, description=\"%3%\", distributedDomains=%4%, residual=%5%)") %
                self.GetName() % daeGetStrippedRelativeName(NULL, dynamic_cast<daeModel*>(self.GetModel())) %
                self.GetDescription() % strDEDIs % strResidual).str();
}

string daeSTN__str__(const daeSTN& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeSTN__repr__(const daeSTN& self)
{
    string strStates = getVector__repr__(self.States(), &daeState__str__);
    string strParent = self.GetModel() ? daeGetStrippedRelativeName(NULL, dynamic_cast<daeModel*>(self.GetModel())) : "None";

    if(self.GetParentState())
        return (boost::format("daeSTN(name=\"%1%\", parentState=%2%, states=%3%, description=\"%4%\")") %
                              self.GetName() % daeGetStrippedRelativeName(NULL, self.GetParentState()) % strStates % self.GetDescription()).str();
    else
        return (boost::format("daeSTN(name=\"%1%\", parentModel=%2%, states=%3%, description=\"%4%\")") %
                              self.GetName() % strParent % strStates % self.GetDescription()).str();
}

string daeIF__str__(const daeIF& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeIF__repr__(const daeIF& self)
{
    string strStates = getVector__repr__(self.States(), &daeState__str__);
    string strParent = self.GetModel() ? daeGetStrippedRelativeName(NULL, dynamic_cast<daeModel*>(self.GetModel())) : "None";

    if(self.GetParentState())
        return (boost::format("daeIF(name=\"%1%\", parentState=%2%, states=%3%, description=\"%4%\")") %
                              self.GetName() % daeGetStrippedRelativeName(NULL, self.GetParentState()) % strStates % self.GetDescription()).str();
    else
        return (boost::format("daeIF(name=\"%1%\", parentModel=%2%, states=%3%, description=\"%4%\")") %
                              self.GetName() % strParent % strStates % self.GetDescription()).str();
}

string daeState__str__(const daeState& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeState__repr__(const daeState& self)
{
    string strEquations          = getVector__repr__(self.Equations(),      &daeEquation__str__);
    string strOnEventActions     = getVector__repr__(self.OnEventActions(), &daeOnEventActions__str__);
    string strNestedSTNs         = getVector__repr__(self.NestedSTNs(),     &daeSTN__str__);
    string strOnConditionActions = "[]";

    return (boost::format("daeState(name=\"%1%\", equations=%2%, onEventActions=%3%, onConditionActions=%4%, nestedSTNs=%5%)") %
                          self.GetName() % strEquations % strOnEventActions % strOnConditionActions % strNestedSTNs).str();
}

string daeDEDI__str__(const daeDEDI& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeDEDI__repr__(const daeDEDI& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeOptimizationVariable__str__(const daeOptimizationVariable& self)
{
    return self.GetName();
}

string daeOptimizationVariable__repr__(const daeOptimizationVariable& self)
{
    return self.GetName();
}

string daeObjectiveFunction__str__(const daeObjectiveFunction& self)
{
    return self.GetName();
}

string daeObjectiveFunction__repr__(const daeObjectiveFunction& self)
{
    return self.GetName();
}

string daeOptimizationConstraint__str__(const daeOptimizationConstraint& self)
{
    return self.GetName();
}

string daeOptimizationConstraint__repr__(const daeOptimizationConstraint& self)
{
    return self.GetName();
}

string daeMeasuredVariable__str__(const daeMeasuredVariable& self)
{
    return self.GetName();
}

string daeMeasuredVariable__repr__(const daeMeasuredVariable& self)
{
    return self.GetName();
}

string daeEventPortConnection__str__(const daeEventPortConnection& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeEventPortConnection__repr__(const daeEventPortConnection& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daePortConnection__str__(const daePortConnection& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daePortConnection__repr__(const daePortConnection& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeOnEventActions__str__(const daeOnEventActions& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeOnEventActions__repr__(const daeOnEventActions& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeOnConditionActions__str__(const daeOnConditionActions& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeOnConditionActions__repr__(const daeOnConditionActions& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeCondition__str__(const daeCondition& self)
{
    return self.SetupNodeAsPlainText();
}

string daeCondition__repr__(const daeCondition& self)
{
    return (boost::format("daeCondition(node=%1%, eventTolerance=%2%)") %
                           self.SetupNodeAsPlainText() % self.GetEventTolerance()).str();
}

string daeAction__str__(daeAction& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeAction__repr__(daeAction& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string adNode__str__(const adNode& self)
{
    string str;
    daeModelExportContext c;

    c.m_nPythonIndentLevel = 0;
    c.m_bExportDefinition  = true;
    c.m_pModel             = NULL;
    self.Export(str, ePYDAE, c);

    return str;
}

string adNode__repr__(const adNode& self)
{
    return adNode__str__(self);
}

string adouble__str__(const adouble& self)
{
    string str;
    daeModelExportContext c;

    c.m_nPythonIndentLevel = 0;
    c.m_bExportDefinition  = true;
    c.m_pModel             = NULL;
    if(self.node)
        self.node->Export(str, ePYDAE, c);
    else
        str = toString(self.getValue());

    return str;
}

string adouble__repr__(const adouble& self)
{
    string strNode;
    daeModelExportContext c;

    c.m_nPythonIndentLevel = 0;
    c.m_bExportDefinition  = true;
    c.m_pModel             = NULL;
    if(self.node)
        self.node->Export(strNode, ePYDAE, c);
    else
        strNode = "None";

    return (boost::format("adouble(value=%1%, derivative=%2%, gatherInfo=%3%, node=%4%)") % self.getValue() % self.getDerivative() % self.getGatherInfo() % strNode).str();
}

string adouble_array__str__(const adouble_array& self)
{
    string str;
    if(self.node)
    {
        daeModelExportContext c;
        c.m_nPythonIndentLevel = 0;
        c.m_bExportDefinition  = true;
        c.m_pModel             = NULL;
        self.node->Export(str, ePYDAE, c);
    }
    else
    {
        str = "[";
        for(size_t i = 0; i < self.m_arrValues.size(); i++)
            str += (boost::format("%1%%2%") % (i==0 ? "" : ", ") % adouble__str__(self.m_arrValues[i])).str();
        str += "]";
    }
    return str;
}

string adouble_array__repr__(const adouble_array& self)
{
    string strNode;
    daeModelExportContext c;

    c.m_nPythonIndentLevel = 0;
    c.m_bExportDefinition  = true;
    c.m_pModel             = NULL;
    if(self.node)
        self.node->Export(strNode, ePYDAE, c);
    else
        strNode = "None";

    return (boost::format("adouble_array(gatherInfo=%1%, node=%2%)") %
            self.getGatherInfo() % strNode).str();
}

string daeScalarExternalFunction__str__(daeScalarExternalFunction& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeScalarExternalFunction__repr__(daeScalarExternalFunction& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeVectorExternalFunction__str__(daeVectorExternalFunction& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

string daeVectorExternalFunction__repr__(daeVectorExternalFunction& self)
{
    return daeGetStrippedRelativeName(NULL, &self);
}

/*******************************************************
    adNode...
*******************************************************/
daeParameter* adSetupParameterNode_Parameter(adSetupParameterNode& node)
{
    return node.m_pParameter;
}

boost::python::list adSetupParameterNode_Domains(adSetupParameterNode& node)
{
    return getListFromVector(node.m_arrDomains);
}

daeVariable* adSetupVariableNode_Variable(adSetupVariableNode& node)
{
    return node.m_pVariable;
}

boost::python::list adSetupVariableNode_Domains(adSetupVariableNode& node)
{
    return getListFromVector(node.m_arrDomains);
}

daeVariable* adSetupTimeDerivativeNode_Variable(adSetupTimeDerivativeNode& node)
{
    return node.m_pVariable;
}

boost::python::list adSetupTimeDerivativeNode_Domains(adSetupTimeDerivativeNode& node)
{
    return getListFromVector(node.m_arrDomains);
}

daeVariable* adSetupPartialDerivativeNode_Variable(adSetupPartialDerivativeNode& node)
{
    return node.m_pVariable;
}

daeDomain* adSetupPartialDerivativeNode_Domain(adSetupPartialDerivativeNode& node)
{
    return node.m_pDomain;
}

boost::python::list adSetupPartialDerivativeNode_Domains(adSetupPartialDerivativeNode& node)
{
    return getListFromVector(node.m_arrDomains);
}


daeParameter* adRuntimeParameterNode_Parameter(adRuntimeParameterNode& node)
{
    return node.m_pParameter;
}

real_t adRuntimeParameterNode_Value(adRuntimeParameterNode& node)
{
    return *node.m_pdValue;
}

boost::python::list adRuntimeParameterNode_Domains(adRuntimeParameterNode& node)
{
    return getListFromVectorByValue(node.m_narrDomains);
}

daeVariable* adRuntimeVariableNode_Variable(adRuntimeVariableNode& node)
{
    return node.m_pVariable;
}

boost::python::list adRuntimeVariableNode_Domains(adRuntimeVariableNode& node)
{
    return getListFromVectorByValue(node.m_narrDomains);
}

daeVariable* adRuntimeTimeDerivativeNode_Variable(adRuntimeTimeDerivativeNode& node)
{
    return node.m_pVariable;
}

boost::python::list adRuntimeTimeDerivativeNode_Domains(adRuntimeTimeDerivativeNode& node)
{
    return getListFromVectorByValue(node.m_narrDomains);
}

daeDomain* adDomainIndexNode_Domain(adDomainIndexNode& node)
{
    return node.m_pDomain;
}

real_t adDomainIndexNode_Value(adDomainIndexNode& node)
{
    return (*node.m_pdPointValue);
}

boost::python::list adRuntimeSpecialFunctionForLargeArraysNode_RuntimeNodes(adRuntimeSpecialFunctionForLargeArraysNode& node)
{
    return getListFromVector(node.m_ptrarrRuntimeNodes);
}

daeScalarExternalFunction* adScalarExternalFunctionNode_ExternalFunction(adScalarExternalFunctionNode& node)
{
    return node.m_pExternalFunction;
}

daeVectorExternalFunction* adVectorExternalFunctionNode_ExternalFunction(adVectorExternalFunctionNode& node)
{
    return node.m_pExternalFunction;
}

real_t adFEMatrixItemNode_Value(adFEMatrixItemNode& self)
{
    return self.m_matrix.GetItem(self.m_row, self.m_column);
}

real_t adFEVectorItemNode_Value(adFEVectorItemNode& self)
{
    return self.m_vector.GetItem(self.m_row);
}

/*******************************************************
    adouble
*******************************************************/
const adouble ad_Constant_q(const quantity& q)
{
    return Constant(q);
}

const adouble ad_Constant_c(real_t c)
{
    return Constant(c);
}

const adouble_array adarr_Array(boost::python::list Values)
{
    std::vector<quantity> qarrValues;
    boost::python::ssize_t i, n;

    n = boost::python::len(Values);
    if(n == 0)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid size of the list in Array function (it is empty)";
        throw e;
    }

    qarrValues.resize(n);
    for(i = 0; i < n; i++)
    {
        boost::python::extract<real_t>   get_real_t(Values[i]);
        boost::python::extract<quantity> get_quantity(Values[i]);

        if(get_real_t.check())
            qarrValues[i] = quantity(get_real_t(), unit());
        else if(get_quantity.check())
            qarrValues[i] = get_quantity();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid item for Array function (must be a floating point value or a quantity object)";
            throw e;
        }
    }

    return Array(qarrValues);
}

const adouble_array adarr_FromNumpyArray(boost::python::object ndValues)
{
    adouble ad;
    std::vector<adNodePtr> ptrarrValues;
    boost::python::ssize_t i, n;

    n = boost::python::extract<boost::python::ssize_t>(ndValues.attr("size"));
    if(n == 0)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid size of the list in CustomArray function (it is empty)";
        throw e;
    }

    ptrarrValues.resize(n);

    for(i = 0; i < n; i++)
    {
        boost::python::object value = ndValues.attr("__getitem__")(i);
        boost::python::extract<adouble> get_adouble(value);

        if(get_adouble.check())
        {
            ad = get_adouble();
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid item for CustomArray function (must be the adouble object)";
            throw e;
        }
        ptrarrValues[i] = ad.node; // Should we Clone it???
    }

    return Array(ptrarrValues);
}

const adouble_array adarr_FromList(boost::python::list lValues)
{
    adouble ad;
    std::vector<adNodePtr> ptrarrValues;
    boost::python::ssize_t i, n;

    n = boost::python::len(lValues);
    if(n == 0)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid size of the list in CustomArray function (it is empty)";
        throw e;
    }

    ptrarrValues.resize(n);

    for(i = 0; i < n; i++)
    {
        boost::python::extract<adouble> get_adouble(lValues[i]);

        if(get_adouble.check())
        {
            ad = get_adouble();
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid item for CustomArray function (must be the adouble object)";
            throw e;
        }
        ptrarrValues[i] = ad.node; // Should we Clone it???
    }

    return Array(ptrarrValues);
}

const adouble ad_exp(const adouble &a)
{
    return exp(a);
}
const adouble ad_log(const adouble &a)
{
    return log(a);
}
const adouble ad_sqrt(const adouble &a)
{
    return sqrt(a);
}
const adouble ad_sin(const adouble &a)
{
    return sin(a);
}
const adouble ad_cos(const adouble &a)
{
    return cos(a);
}
const adouble ad_tan(const adouble &a)
{
    return tan(a);
}
const adouble ad_asin(const adouble &a)
{
    return asin(a);
}
const adouble ad_acos(const adouble &a)
{
    return acos(a);
}
const adouble ad_atan(const adouble &a)
{
    return atan(a);
}

const adouble ad_sinh(const adouble &a)
{
    return sinh(a);
}
const adouble ad_cosh(const adouble &a)
{
    return cosh(a);
}
const adouble ad_tanh(const adouble &a)
{
    return tanh(a);
}
const adouble ad_asinh(const adouble &a)
{
    return asinh(a);
}
const adouble ad_acosh(const adouble &a)
{
    return acosh(a);
}
const adouble ad_atanh(const adouble &a)
{
    return atanh(a);
}
const adouble ad_atan2(const adouble &a, const adouble &b)
{
    return atan2(a, b);
}
const adouble ad_erf(const adouble &a)
{
    return erf(a);
}

const adouble ad_pow1(const adouble &a, real_t v)
{
    return pow(a,v);
}
const adouble ad_pow2(const adouble &a, const adouble &b)
{
    return pow(a,b);
}
const adouble ad_pow3(real_t v, const adouble &a)
{
    return pow(v,a);
}
const adouble ad_log10(const adouble &a)
{
    return log10(a);
}
const adouble ad_abs(const adouble &a)
{
    return abs(a);
}
const adouble ad_ceil(const adouble &a)
{
    return ceil(a);
}
const adouble ad_floor(const adouble &a)
{
    return floor(a);
}
const adouble ad_max1(const adouble &a, const adouble &b)
{
    return max(a,b);
}
const adouble ad_max2(real_t v, const adouble &a)
{
    return max(v,a);
}
const adouble ad_max3(const adouble &a, real_t v)
{
    return max(a,v);
}
const adouble ad_min1(const adouble &a, const adouble &b)
{
    return min(a,b);
}
const adouble ad_min2(real_t v, const adouble &a)
{
    return min(v,a);
}
const adouble ad_min3(const adouble &a, real_t v)
{
    return min(a,v);
}
const adouble ad_dt(const adouble& a)
{
    return dt(a);
}
const adouble ad_d(const adouble& a, daeDomain& domain)
{
    return d(a, domain);
}

/*******************************************************
    adouble_array
*******************************************************/
const adouble_array adarr_exp(const adouble_array& a)
{
    return exp(a);
}
const adouble_array adarr_sqrt(const adouble_array& a)
{
    return sqrt(a);
}
const adouble_array adarr_log(const adouble_array& a)
{
    return log(a);
}
const adouble_array adarr_log10(const adouble_array& a)
{
    return log10(a);
}
const adouble_array adarr_abs(const adouble_array& a)
{
    return abs(a);
}
const adouble_array adarr_floor(const adouble_array& a)
{
    return floor(a);
}
const adouble_array adarr_ceil(const adouble_array& a)
{
    return ceil(a);
}
const adouble_array adarr_sin(const adouble_array& a)
{
    return sin(a);
}
const adouble_array adarr_cos(const adouble_array& a)
{
    return cos(a);
}
const adouble_array adarr_tan(const adouble_array& a)
{
    return tan(a);
}
const adouble_array adarr_asin(const adouble_array& a)
{
    return asin(a);
}
const adouble_array adarr_acos(const adouble_array& a)
{
    return acos(a);
}
const adouble_array adarr_atan(const adouble_array& a)
{
    return atan(a);
}

const adouble adarr_sum(const adouble_array& a)
{
    return Sum(a, false);
}
const adouble adarr_product(const adouble_array& a)
{
    return Product(a, false);
}
const adouble adarr_min(const adouble_array& a)
{
    return Min(a);
}
const adouble adarr_max(const adouble_array& a)
{
    return Max(a);
}
const adouble adarr_average(const adouble_array& a)
{
    return Average(a);
}
const adouble adarr_integral(const adouble_array& a)
{
    return Integral(a);
}

adouble adouble_array__call__(adouble_array& a, boost::python::object index)
{
    daeDomainIndex domainIndex = CreateDomainIndex(index);

    if(!a.node)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid adouble_array specified for function call (__call__) function";
        throw e;
    }
    return a(domainIndex);
}

/*******************************************************
    adouble_array
*******************************************************/
boost::python::list daeCondition_GetExpressions(daeCondition& self)
{
    std::vector<adNode*> ptrarrExpressions;
    self.GetExpressionsArray(ptrarrExpressions);
    return getListFromVector(ptrarrExpressions);
}

/*******************************************************
    daeObject
*******************************************************/
daeObject* daeObject_GetModel(daeObject& self)
{
    return dynamic_cast<daeObject*>(self.GetModel());
}

string daeGetRelativeName_1(const daeObject* parent, const daeObject* child)
{
    return daeGetRelativeName(parent, child);
}

string daeGetRelativeName_2(const string& strParent, const string& strChild)
{
    return daeGetRelativeName(strParent, strChild);
}

/*******************************************************
    daeDomain
*******************************************************/
boost::python::object GetNumPyArrayDomain(daeDomain& domain)
{
/* NUMPY
    size_t nType;
    npy_intp dimensions;

    nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
    dimensions = domain.GetNumberOfPoints();

    boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
    real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
    for(size_t k = 0; k < (size_t)dimensions; k++)
        values[k] = *domain.GetPoint(k);

    return numpy_array;
*/
    // Import numpy
    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import numpy", main_namespace);
    boost::python::object numpy = main_namespace["numpy"];

    // Create shape
    int dimensions = domain.GetNumberOfPoints();
    boost::python::tuple shape = boost::python::make_tuple(dimensions);

    // Create a flat list of values
    boost::python::list lvalues;
    for(size_t k = 0; k < (size_t)dimensions; k++)
        lvalues.append(*domain.GetPoint(k));

    // Create a flat ndarray
    boost::python::dict kwargs;
    if(typeid(real_t) == typeid(double))
        kwargs["dtype"] = numpy.attr("float64");
    else
        kwargs["dtype"] = numpy.attr("float32");
    boost::python::tuple args = boost::python::make_tuple(lvalues);
    boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

    // Return a re-shaped ndarray
    return ndarray.attr("reshape")(shape);
}

boost::python::list GetDomainPoints(daeDomain& domain)
{
    boost::python::list l;

    for(size_t i = 0; i < domain.GetNumberOfPoints(); i++)
        l.append(*domain.GetPoint(i));

    return l;
}

boost::python::list GetDomainCoordinates(daeDomain& domain)
{
    boost::python::list l;

    const std::vector<daePoint>& coords = domain.GetCoordinates();

    for(size_t i = 0; i < coords.size(); i++)
        l.append(make_tuple(coords[i].x, coords[i].y, coords[i].z));

    return l;
}

void CreateUnstructuredGrid(daeDomain& domain, boost::python::list coords)
{
    real_t x, y, z;
    boost::python::tuple point;
    std::vector<daePoint> arrCoords;

    boost::python::ssize_t n = boost::python::len(coords);
    arrCoords.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        point = extract<boost::python::tuple>(coords[i]);
        arrCoords[i].x = extract<real_t>(point[0]);
        arrCoords[i].y = extract<real_t>(point[1]);
        arrCoords[i].z = extract<real_t>(point[2]);
    }

    domain.CreateUnstructuredGrid(arrCoords);
}

void SetDomainPoints(daeDomain& domain, boost::python::list l)
{
    real_t point;
    std::vector<real_t> darrPoints;

    boost::python::ssize_t n = boost::python::len(l);
    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        point = extract<real_t>(l[i]);
        darrPoints.push_back(point);
    }

    domain.SetPoints(darrPoints);
}

adouble_array DomainArray(daeDomain& domain, boost::python::object indexes)
{
    daeArrayRange arrayRange = CreateArrayRange(indexes, &domain);

    if(arrayRange.m_eType == eRangeDomainIndex)
    {
        if(arrayRange.m_domainIndex.m_eType == eDomainIterator || arrayRange.m_domainIndex.m_eType == eIncrementedDomainIterator)
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid argument for the daeDomain::array function (must not be the daeDEDI object)";
            throw e;
        }
    }

    return domain.array(arrayRange);
}

daeIndexRange* __init__daeIndexRange(daeDomain* pDomain, boost::python::list CustomPoints)
{
    size_t index;
    std::vector<size_t> narrCustomPoints;

    boost::python::ssize_t n = boost::python::len(CustomPoints);
    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        index = extract<size_t>(CustomPoints[i]);
        narrCustomPoints.push_back(index);
    }
    return new daeIndexRange(pDomain, narrCustomPoints);
}

daeDomain* daeIndexRange_GetDomain(daeIndexRange& self)
{
    return self.m_pDomain;
}

/*
daeIndexRange FunctionCallDomain1(daeDomain& domain, int start, int end, int step)
{
    return domain(start, end, step);
}

daeIndexRange FunctionCallDomain2(daeDomain& domain, boost::python::list l)
{
    size_t index;
    std::vector<size_t> narrDomainIndexes;
    boost::python::ssize_t n = boost::python::len(l);
    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        index = extract<size_t>(l[i]);
        narrDomainIndexes.push_back(index);
    }

    return domain(narrDomainIndexes);
}

daeIndexRange FunctionCallDomain3(daeDomain& domain)
{
    return domain();
}
*/

/*******************************************************
    daeParameter
*******************************************************/
daeParameter* daeParameter_init1(string strName, const unit& units, daeModel* pModel, string strDescription, boost::python::list domains)
{
    daeDomain* pDomain;
    boost::python::ssize_t n = boost::python::len(domains);

    daeParameter* pParameter = new daeParameter(strName, units, pModel, strDescription);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        pDomain = boost::python::extract<daeDomain*>(domains[i]);
        pParameter->DistributeOnDomain(*pDomain);
    }

    return pParameter;
}

daeParameter* daeParameter_init2(string strName, const unit& units, daePort* pPort, string strDescription, boost::python::list domains)
{
    daeDomain* pDomain;
    boost::python::ssize_t n = boost::python::len(domains);

    daeParameter* pParameter = new daeParameter(strName, units, pPort, strDescription);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        pDomain = boost::python::extract<daeDomain*>(domains[i]);
        pParameter->DistributeOnDomain(*pDomain);
    }

    return pParameter;
}

boost::python::list daeParameter_GetDomains(daeParameter& self)
{
    return getListFromVector(self.Domains());
}

boost::python::dict daeParameter_GetDomainsIndexesMap1(daeParameter& self, size_t nIndexBase)
{
   // Returns dictionary {integer : [list of integers]}
    boost::python::dict d;
    std::map<size_t, std::vector<size_t> > mapIndexes;
    typedef std::map<size_t, std::vector<size_t> >::iterator c_iterator;

    self.GetDomainsIndexesMap(mapIndexes, nIndexBase);

    for(c_iterator iter = mapIndexes.begin(); iter != mapIndexes.end(); iter++)
        d[iter->first] = getListFromVectorByValue<size_t>(iter->second);

    return d;
}

real_t lGetParameterValue(daeParameter& self, boost::python::list indexes)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in GetValue for parameter " << self.GetCanonicalName();
            throw e;
        }
    }
    return self.GetValue(narrIndexes);
}

quantity lGetParameterQuantity(daeParameter& self, boost::python::list indexes)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in GetQuantity for parameter " << self.GetCanonicalName();
            throw e;
        }
    }
    return self.GetQuantity(narrIndexes);
}

real_t GetParameterValue0(daeParameter& self)
{
    return self.GetValue();
}

real_t GetParameterValue1(daeParameter& self, size_t n1)
{
    return self.GetValue(n1);
}

real_t GetParameterValue2(daeParameter& self, size_t n1, size_t n2)
{
    return self.GetValue(n1, n2);
}

real_t GetParameterValue3(daeParameter& self, size_t n1, size_t n2, size_t n3)
{
    return self.GetValue(n1, n2, n3);
}

real_t GetParameterValue4(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4)
{
    return self.GetValue(n1, n2, n3, n4);
}

real_t GetParameterValue5(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
{
    return self.GetValue(n1, n2, n3, n4, n5);
}

real_t GetParameterValue6(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
{
    return self.GetValue(n1, n2, n3, n4, n5, n6);
}

real_t GetParameterValue7(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
{
    return self.GetValue(n1, n2, n3, n4, n5, n6, n7);
}

real_t GetParameterValue8(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
{
    return self.GetValue(n1, n2, n3, n4, n5, n6, n7, n8);
}

quantity GetParameterQuantity0(daeParameter& self)
{
    return self.GetQuantity();
}

quantity GetParameterQuantity1(daeParameter& self, size_t n1)
{
    return self.GetQuantity(n1);
}

quantity GetParameterQuantity2(daeParameter& self, size_t n1, size_t n2)
{
    return self.GetQuantity(n1, n2);
}

quantity GetParameterQuantity3(daeParameter& self, size_t n1, size_t n2, size_t n3)
{
    return self.GetQuantity(n1, n2, n3);
}

quantity GetParameterQuantity4(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4)
{
    return self.GetQuantity(n1, n2, n3, n4);
}

quantity GetParameterQuantity5(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
{
    return self.GetQuantity(n1, n2, n3, n4, n5);
}

quantity GetParameterQuantity6(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
{
    return self.GetQuantity(n1, n2, n3, n4, n5, n6);
}

quantity GetParameterQuantity7(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
{
    return self.GetQuantity(n1, n2, n3, n4, n5, n6, n7);
}

quantity GetParameterQuantity8(daeParameter& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
{
    return self.GetQuantity(n1, n2, n3, n4, n5, n6, n7, n8);
}


boost::python::object GetNumPyArrayParameter(daeParameter& self)
{
/* NUMPY
    size_t nType, nDomains, nTotalSize;
    real_t* data;
    npy_intp* dimensions;
    vector<daeDomain_t*> ptrarrDomains;

    nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
    data = param.GetValuePointer();
    param.GetDomains(ptrarrDomains);
    nDomains = ptrarrDomains.size();

    if(nDomains == 0)
    {
        return boost::python::object(param.GetValue());
    }
    else
    {
        dimensions = new npy_intp[nDomains];
        nTotalSize = 1;
        for(size_t i = 0; i < nDomains; i++)
        {
            dimensions[i] = ptrarrDomains[i]->GetNumberOfPoints();
            nTotalSize *= dimensions[i];
        }

        boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(nDomains, dimensions, nType))));
        real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
        for(size_t k = 0; k < nTotalSize; k++)
            values[k] = data[k];

        delete[] dimensions;
        return numpy_array;
    }
*/
    const std::vector<daeDomain*>& domains = self.Domains();
    real_t* data = self.GetValuePointer();
    size_t nTotalSize = self.GetNumberOfPoints();

    if(domains.size() == 0)
    {
        return boost::python::object(self.GetValue());
    }
    else
    {
        // Import numpy
        boost::python::object main_module = import("__main__");
        boost::python::object main_namespace = main_module.attr("__dict__");
        exec("import numpy", main_namespace);
        boost::python::object numpy = main_namespace["numpy"];

        // Create shape
        boost::python::list ldimensions;
        for(size_t i = 0; i < domains.size(); i++)
            ldimensions.append(domains[i]->GetNumberOfPoints());
        boost::python::tuple shape = boost::python::tuple(ldimensions);

        // Create a flat list of values
        boost::python::list lvalues;
        for(size_t k = 0; k < nTotalSize; k++)
            lvalues.append(data[k]);

        // Create a flat ndarray
        boost::python::dict kwargs;
        if(typeid(real_t) == typeid(double))
            kwargs["dtype"] = numpy.attr("float64");
        else
            kwargs["dtype"] = numpy.attr("float32");
        boost::python::tuple args = boost::python::make_tuple(lvalues);
        boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

        // Return a re-shaped ndarray
        return ndarray.attr("reshape")(shape);
    }
}

adouble FunctionCallParameter0(daeParameter& param)
{
    return param();
}

adouble FunctionCallParameter1(daeParameter& param, object o1)
{
    return param(CreateDomainIndex(o1));
}

adouble FunctionCallParameter2(daeParameter& param, object o1, object o2)
{
    return param(CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble FunctionCallParameter3(daeParameter& param, object o1, object o2, object o3)
{
    return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble FunctionCallParameter4(daeParameter& param, object o1, object o2, object o3, object o4)
{
    return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble FunctionCallParameter5(daeParameter& param, object o1, object o2, object o3, object o4, object o5)
{
    return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble FunctionCallParameter6(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble FunctionCallParameter7(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble FunctionCallParameter8(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return param(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

void lSetParameterValue(daeParameter& param, boost::python::list indexes, real_t value)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in SetValue for parameter " << param.GetCanonicalName();
            throw e;
        }
    }
    param.SetValue(narrIndexes, value);
}

void lSetParameterQuantity(daeParameter& param, boost::python::list indexes, const quantity& value)
{
    real_t val = value.scaleTo(param.GetUnits()).getValue();
    lSetParameterValue(param, indexes, val);
}

void SetParameterValue0(daeParameter& param, real_t value)
{
    param.SetValue(value);
}

void SetParameterValue1(daeParameter& param, size_t n1, real_t value)
{
    param.SetValue(n1, value);
}

void SetParameterValue2(daeParameter& param, size_t n1, size_t n2, real_t value)
{
    param.SetValue(n1, n2, value);
}

void SetParameterValue3(daeParameter& param, size_t n1, size_t n2, size_t n3, real_t value)
{
    param.SetValue(n1, n2, n3, value);
}

void SetParameterValue4(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
    param.SetValue(n1, n2, n3, n4, value);
}

void SetParameterValue5(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
    param.SetValue(n1, n2, n3, n4, n5, value);
}

void SetParameterValue6(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
    param.SetValue(n1, n2, n3, n4, n5, n6, value);
}

void SetParameterValue7(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
    param.SetValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void SetParameterValue8(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
    param.SetValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void SetParameterQuantity0(daeParameter& param, const quantity& q)
{
    param.SetValue(q);
}

void SetParameterQuantity1(daeParameter& param, size_t n1, const quantity& q)
{
    param.SetValue(n1, q);
}

void SetParameterQuantity2(daeParameter& param, size_t n1, size_t n2, const quantity& q)
{
    param.SetValue(n1, n2, q);
}

void SetParameterQuantity3(daeParameter& param, size_t n1, size_t n2, size_t n3, const quantity& q)
{
    param.SetValue(n1, n2, n3, q);
}

void SetParameterQuantity4(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& q)
{
    param.SetValue(n1, n2, n3, n4, q);
}

void SetParameterQuantity5(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& q)
{
    param.SetValue(n1, n2, n3, n4, n5, q);
}

void SetParameterQuantity6(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& q)
{
    param.SetValue(n1, n2, n3, n4, n5, n6, q);
}

void SetParameterQuantity7(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& q)
{
    param.SetValue(n1, n2, n3, n4, n5, n6, n7, q);
}

void SetParameterQuantity8(daeParameter& param, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& q)
{
    param.SetValue(n1, n2, n3, n4, n5, n6, n7, n8, q);
}

void SetParameterValues(daeParameter& param, real_t values)
{
    param.SetValues(values);
}

void qSetParameterValues(daeParameter& param, const quantity& q)
{
    param.SetValues(q);
}

void lSetParameterValues(daeParameter& param, boost::python::object nd_values)
{
/* NUMPY */
    // Check the shape of ndarray
    boost::python::tuple shape = boost::python::extract<boost::python::tuple>(nd_values.attr("shape"));
    if(len(shape) != param.GetNumberOfDomains())
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid number of dimensions (" << len(shape) << ") of the array of values in SetValues for parameter " << param.GetCanonicalName()
          << "; the required number of dimensions is " << param.GetNumberOfDomains();
        throw e;
    }
    for(size_t k = 0; k < len(shape); k++)
    {
        size_t dim_avail = boost::python::extract<size_t>(shape[k]);
        size_t dim_req   = param.GetDomain(k)->GetNumberOfPoints();
        if(dim_req != dim_avail)
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid shape of the array of values in SetValues for parameter " << param.GetCanonicalName()
              << "; dimension " << k << " has " << dim_avail << " points (required is " << dim_req << ")";
            throw e;
        }
    }

    // The ndarray must be flattened before use (in the row-major c-style order)
    boost::python::object arg("C");
    boost::python::object values = nd_values.attr("ravel")(arg);
    boost::python::ssize_t n = boost::python::extract<boost::python::ssize_t>(values.attr("size"));
    std::vector<quantity> q_values;
    q_values.resize(n);

    unit u = param.GetUnits();

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::object value = values.attr("__getitem__")(i);
        boost::python::extract<real_t>   rValue(value);
        boost::python::extract<quantity> qValue(value);

        if(rValue.check())
            q_values[i] = quantity(rValue(), u);
        else if(qValue.check())
            q_values[i] = qValue;
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of item [" << i << "] in the list of values in SetValues for parameter " << param.GetCanonicalName();
            throw e;
        }
    }
    param.SetValues(q_values);
}

adouble_array ParameterArray1(daeParameter& param, object o1)
{
    return param.array(parRANGE(1));
}

adouble_array ParameterArray2(daeParameter& param, object o1, object o2)
{
    return param.array(parRANGE(1), parRANGE(2));
}

adouble_array ParameterArray3(daeParameter& param, object o1, object o2, object o3)
{
    return param.array(parRANGE(1), parRANGE(2), parRANGE(3));
}

adouble_array ParameterArray4(daeParameter& param, object o1, object o2, object o3, object o4)
{
    return param.array(parRANGE(1), parRANGE(2), parRANGE(3), parRANGE(4));
}

adouble_array ParameterArray5(daeParameter& param, object o1, object o2, object o3, object o4, object o5)
{
    return param.array(parRANGE(1), parRANGE(2), parRANGE(3), parRANGE(4), parRANGE(5));
}

adouble_array ParameterArray6(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return param.array(parRANGE(1), parRANGE(2), parRANGE(3), parRANGE(4), parRANGE(5), parRANGE(6));
}

adouble_array ParameterArray7(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return param.array(parRANGE(1), parRANGE(2), parRANGE(3), parRANGE(4), parRANGE(5), parRANGE(6), parRANGE(7));
}

adouble_array ParameterArray8(daeParameter& param, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return param.array(parRANGE(1), parRANGE(2), parRANGE(3), parRANGE(4), parRANGE(5), parRANGE(6), parRANGE(7), parRANGE(8));
}

/*******************************************************
    daeVariable
*******************************************************/
daeVariable* daeVariable_init1(string strName, const daeVariableType& varType, daeModel* pModel, string strDescription, boost::python::list domains)
{
    daeDomain* pDomain;
    boost::python::ssize_t n = boost::python::len(domains);

    daeVariable* pVariable = new daeVariable(strName, varType, pModel, strDescription);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        pDomain = boost::python::extract<daeDomain*>(domains[i]);
        pVariable->DistributeOnDomain(*pDomain);
    }

    return pVariable;
}

daeVariable* daeVariable_init2(string strName, const daeVariableType& varType, daePort* pPort, string strDescription, boost::python::list domains)
{
    daeDomain* pDomain;
    boost::python::ssize_t n = boost::python::len(domains);

    daeVariable* pVariable = new daeVariable(strName, varType, pPort, strDescription);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        pDomain = boost::python::extract<daeDomain*>(domains[i]);
        pVariable->DistributeOnDomain(*pDomain);
    }

    return pVariable;
}

boost::python::list daeVariable_GetDomains(daeVariable& self)
{
    return getListFromVector(self.Domains());
}

daeVariableType* daeVariable_GetVariableType(daeVariable& self)
{
    daeVariableType_t* vt = const_cast<daeVariableType_t*>(self.GetVariableType());
    return dynamic_cast<daeVariableType*>(vt);
}

real_t daeVariable_lGetVariableValue(daeVariable& self, boost::python::list indexes)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in GetValue for variable " << self.GetCanonicalName();
            throw e;
        }
    }
    return self.GetValue(narrIndexes);
}

quantity daeVariable_lGetVariableQuantity(daeVariable& self, boost::python::list indexes)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in GetQuantity for variable " << self.GetCanonicalName();
            throw e;
        }
    }
    return self.GetQuantity(narrIndexes);
}

real_t daeVariable_GetVariableValue0(daeVariable& self)
{
    return self.GetValue();
}

real_t daeVariable_GetVariableValue1(daeVariable& self, size_t n1)
{
    return self.GetValue(n1);
}

real_t daeVariable_GetVariableValue2(daeVariable& self, size_t n1, size_t n2)
{
    return self.GetValue(n1, n2);
}

real_t daeVariable_GetVariableValue3(daeVariable& self, size_t n1, size_t n2, size_t n3)
{
    return self.GetValue(n1, n2, n3);
}

real_t daeVariable_GetVariableValue4(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4)
{
    return self.GetValue(n1, n2, n3, n4);
}

real_t daeVariable_GetVariableValue5(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
{
    return self.GetValue(n1, n2, n3, n4, n5);
}

real_t daeVariable_GetVariableValue6(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
{
    return self.GetValue(n1, n2, n3, n4, n5, n6);
}

real_t daeVariable_GetVariableValue7(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
{
    return self.GetValue(n1, n2, n3, n4, n5, n6, n7);
}

real_t daeVariable_GetVariableValue8(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
{
    return self.GetValue(n1, n2, n3, n4, n5, n6, n7, n8);
}

quantity daeVariable_GetVariableQuantity0(daeVariable& self)
{
    return self.GetQuantity();
}

quantity daeVariable_GetVariableQuantity1(daeVariable& self, size_t n1)
{
    return self.GetQuantity(n1);
}

quantity daeVariable_GetVariableQuantity2(daeVariable& self, size_t n1, size_t n2)
{
    return self.GetQuantity(n1, n2);
}

quantity daeVariable_GetVariableQuantity3(daeVariable& self, size_t n1, size_t n2, size_t n3)
{
    return self.GetQuantity(n1, n2, n3);
}

quantity daeVariable_GetVariableQuantity4(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4)
{
    return self.GetQuantity(n1, n2, n3, n4);
}

quantity daeVariable_GetVariableQuantity5(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
{
    return self.GetQuantity(n1, n2, n3, n4, n5);
}

quantity daeVariable_GetVariableQuantity6(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
{
    return self.GetQuantity(n1, n2, n3, n4, n5, n6);
}

quantity daeVariable_GetVariableQuantity7(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
{
    return self.GetQuantity(n1, n2, n3, n4, n5, n6, n7);
}

quantity daeVariable_GetVariableQuantity8(daeVariable& self, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
{
    return self.GetQuantity(n1, n2, n3, n4, n5, n6, n7, n8);
}



boost::python::object daeVariable_TimeDerivatives(daeVariable& self)
{
/* NUMPY
    size_t i, k, nType, nDomains, nStart, nEnd;
    npy_intp* dimensions;
    vector<daeDomain_t*> ptrarrDomains;

    nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
    var.GetDomains(ptrarrDomains);
    nDomains = ptrarrDomains.size();
    daeModel* pModel = dynamic_cast<daeModel*>(var.GetModel());
    boost::shared_ptr<daeDataProxy_t> pDataProxy = pModel->GetDataProxy();
    const std::vector<real_t*>& dtRefs = pDataProxy->GetTimeDerivativesReferences();

    if(nDomains == 0)
    {
        // Assigned variables do not have time derivatives mapped!!
        if(dtRefs[var.GetOverallIndex()])
            return boost::python::object(*dtRefs[var.GetOverallIndex()]);
        else
            return boost::python::object(0.0);
    }
    else
    {
        dimensions = new npy_intp[nDomains];
        for(i = 0; i < nDomains; i++)
            dimensions[i] = ptrarrDomains[i]->GetNumberOfPoints();
        nStart = var.GetOverallIndex();
        nEnd   = var.GetOverallIndex() + var.GetNumberOfPoints();

        boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(nDomains, dimensions, nType))));
        real_t* values = static_cast<real_t*>(PyArray_DATA(numpy_array.ptr()));

        // Assigned variables do not have time derivatives mapped!!
        for(k = 0, i = nStart; i < nEnd; i++, k++)
        {
            if(dtRefs[i])
                values[k] = *dtRefs[i];
            else
                values[k] = 0.0;
        }

        delete[] dimensions;
        return numpy_array;
    }
*/
    const std::vector<daeDomain*>& domains = self.Domains();
    daeModel* pModel = dynamic_cast<daeModel*>(self.GetModel());
    boost::shared_ptr<daeDataProxy_t> pDataProxy = pModel->GetDataProxy();
    size_t nStart = self.GetOverallIndex();
    size_t nEnd   = self.GetOverallIndex() + self.GetNumberOfPoints();
    const std::vector<real_t*>& dtRefs = pDataProxy->GetTimeDerivativesReferences();

    if(domains.size() == 0)
    {
        // Assigned variables do not have time derivatives mapped!!
        if(dtRefs[self.GetOverallIndex()])
            return boost::python::object(*dtRefs[self.GetOverallIndex()]);
        else
            return boost::python::object(0.0);
    }
    else
    {
        // Import numpy
        boost::python::object main_module = import("__main__");
        boost::python::object main_namespace = main_module.attr("__dict__");
        exec("import numpy", main_namespace);
        boost::python::object numpy = main_namespace["numpy"];

        // Create shape
        boost::python::list ldimensions;
        for(size_t i = 0; i < domains.size(); i++)
            ldimensions.append(domains[i]->GetNumberOfPoints());
        boost::python::tuple shape = boost::python::tuple(ldimensions);

        // Create a flat list of dt values
        boost::python::list lvalues;
        for(size_t i = nStart; i < nEnd; i++)
        {
            if(dtRefs[i])
                lvalues.append(*dtRefs[i]);
            else
                lvalues.append(0.0);
        }

        // Create a flat ndarray
        boost::python::dict kwargs;
        if(typeid(real_t) == typeid(double))
            kwargs["dtype"] = numpy.attr("float64");
        else
            kwargs["dtype"] = numpy.attr("float32");
        boost::python::tuple args = boost::python::make_tuple(lvalues);
        boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

        // Return a re-shaped ndarray
        return ndarray.attr("reshape")(shape);
    }
}

boost::python::object daeVariable_Values(daeVariable& self)
{
/* NUMPY
    size_t i, k, nType, nDomains, nStart, nEnd;
    npy_intp* dimensions;
    vector<daeDomain_t*> ptrarrDomains;

    nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
    var.GetDomains(ptrarrDomains);
    nDomains = ptrarrDomains.size();
    daeModel* pModel = dynamic_cast<daeModel*>(var.GetModel());
    boost::shared_ptr<daeDataProxy_t> pDataProxy = pModel->GetDataProxy();

    if(nDomains == 0)
    {
        return boost::python::object(pDataProxy->GetValue(var.GetOverallIndex()));
    }
    else
    {
        dimensions = new npy_intp[nDomains];
        for(i = 0; i < nDomains; i++)
            dimensions[i] = ptrarrDomains[i]->GetNumberOfPoints();
        nStart = var.GetOverallIndex();
        nEnd   = var.GetOverallIndex() + var.GetNumberOfPoints();

        boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(nDomains, dimensions, nType))));
        real_t* values = static_cast<real_t*>(PyArray_DATA(numpy_array.ptr()));

        for(k = 0, i = nStart; i < nEnd; i++, k++)
            values[k] = pDataProxy->GetValue(i);

        delete[] dimensions;
        return numpy_array;
    }
*/
    const std::vector<daeDomain*>& domains = self.Domains();
    daeModel* pModel = dynamic_cast<daeModel*>(self.GetModel());
    boost::shared_ptr<daeDataProxy_t> pDataProxy = pModel->GetDataProxy();
    size_t nStart = self.GetOverallIndex();
    size_t nEnd   = self.GetOverallIndex() + self.GetNumberOfPoints();

    if(domains.size() == 0)
    {
        return boost::python::object(pDataProxy->GetValue(self.GetOverallIndex()));
    }
    else
    {
        // Import numpy
        boost::python::object main_module = import("__main__");
        boost::python::object main_namespace = main_module.attr("__dict__");
        exec("import numpy", main_namespace);
        boost::python::object numpy = main_namespace["numpy"];

        // Create shape
        boost::python::list ldimensions;
        for(size_t i = 0; i < domains.size(); i++)
            ldimensions.append(domains[i]->GetNumberOfPoints());
        boost::python::tuple shape = boost::python::tuple(ldimensions);

        // Create a flat list of values
        boost::python::list lvalues;
        for(size_t i = nStart; i < nEnd; i++)
            lvalues.append(pDataProxy->GetValue(i));

        // Create a flat ndarray
        boost::python::dict kwargs;
        if(typeid(real_t) == typeid(double))
            kwargs["dtype"] = numpy.attr("float64");
        else
            kwargs["dtype"] = numpy.attr("float32");
        boost::python::tuple args = boost::python::make_tuple(lvalues);
        boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

        // Return a re-shaped ndarray
        return ndarray.attr("reshape")(shape);
    }
}

boost::python::object daeVariable_IDs(daeVariable& self)
{
/* NUMPY
    size_t i, k, nType, nDomains, nStart, nEnd;
    npy_intp* dimensions;
    vector<daeDomain_t*> ptrarrDomains;

    nType = NPY_INT;
    var.GetDomains(ptrarrDomains);
    nDomains = ptrarrDomains.size();
    daeModel* pModel = dynamic_cast<daeModel*>(var.GetModel());
    boost::shared_ptr<daeDataProxy_t> pDataProxy = pModel->GetDataProxy();

    if(nDomains == 0)
    {
        return boost::python::object(pDataProxy->GetVariableType(var.GetOverallIndex()));
    }
    else
    {
        dimensions = new npy_intp[nDomains];
        for(i = 0; i < nDomains; i++)
            dimensions[i] = ptrarrDomains[i]->GetNumberOfPoints();
        nStart = var.GetOverallIndex();
        nEnd   = var.GetOverallIndex() + var.GetNumberOfPoints();

        boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(nDomains, dimensions, nType))));
        npy_int* values = static_cast<npy_int*>(PyArray_DATA(numpy_array.ptr()));

        for(k = 0, i = nStart; i < nEnd; i++, k++)
            values[k] = static_cast<npy_int>(pDataProxy->GetVariableType(i));

        delete[] dimensions;
        return numpy_array;
    }
*/
    const std::vector<daeDomain*>& domains = self.Domains();
    daeModel* pModel = dynamic_cast<daeModel*>(self.GetModel());
    boost::shared_ptr<daeDataProxy_t> pDataProxy = pModel->GetDataProxy();
    size_t nStart = self.GetOverallIndex();
    size_t nEnd   = self.GetOverallIndex() + self.GetNumberOfPoints();

    if(domains.size() == 0)
    {
        return boost::python::object(pDataProxy->GetVariableType(self.GetOverallIndex()));
    }
    else
    {
        // Import numpy
        boost::python::object main_module = import("__main__");
        boost::python::object main_namespace = main_module.attr("__dict__");
        exec("import numpy", main_namespace);
        boost::python::object numpy = main_namespace["numpy"];

        // Create shape
        boost::python::list ldimensions;
        for(size_t i = 0; i < domains.size(); i++)
            ldimensions.append(domains[i]->GetNumberOfPoints());
        boost::python::tuple shape = boost::python::tuple(ldimensions);

        // Create a flat list of values
        boost::python::list lvalues;
        for(size_t i = nStart; i < nEnd; i++)
            lvalues.append(pDataProxy->GetVariableType(i));

        // Create a flat ndarray
        boost::python::dict kwargs;
        kwargs["dtype"] = numpy.attr("int32");
        boost::python::tuple args = boost::python::make_tuple(lvalues);
        boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

        // Return a re-shaped ndarray
        return ndarray.attr("reshape")(shape);
    }
}

boost::python::object daeVariable_GatheredIDs(daeVariable& self)
{
/* NUMPY
    size_t i, k, nType, nDomains, nStart, nEnd;
    npy_intp* dimensions;
    vector<daeDomain_t*> ptrarrDomains;

    nType = NPY_INT;
    var.GetDomains(ptrarrDomains);
    nDomains = ptrarrDomains.size();
    daeModel* pModel = dynamic_cast<daeModel*>(var.GetModel());
    boost::shared_ptr<daeDataProxy_t> pDataProxy = pModel->GetDataProxy();

    if(nDomains == 0)
    {
        return boost::python::object(pDataProxy->GetVariableTypeGathered(var.GetOverallIndex()));
    }
    else
    {
        dimensions = new npy_intp[nDomains];
        for(i = 0; i < nDomains; i++)
            dimensions[i] = ptrarrDomains[i]->GetNumberOfPoints();
        nStart = var.GetOverallIndex();
        nEnd   = var.GetOverallIndex() + var.GetNumberOfPoints();

        boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(nDomains, dimensions, nType))));
        npy_int* values = static_cast<npy_int*>(PyArray_DATA(numpy_array.ptr()));

        for(k = 0, i = nStart; i < nEnd; i++, k++)
            values[k] = static_cast<npy_int>(pDataProxy->GetVariableTypeGathered(i));

        delete[] dimensions;
        return numpy_array;
    }
*/
    const std::vector<daeDomain*>& domains = self.Domains();
    daeModel* pModel = dynamic_cast<daeModel*>(self.GetModel());
    boost::shared_ptr<daeDataProxy_t> pDataProxy = pModel->GetDataProxy();
    size_t nStart = self.GetOverallIndex();
    size_t nEnd   = self.GetOverallIndex() + self.GetNumberOfPoints();

    if(domains.size() == 0)
    {
        return boost::python::object(pDataProxy->GetVariableTypeGathered(self.GetOverallIndex()));
    }
    else
    {
        // Import numpy
        boost::python::object main_module = import("__main__");
        boost::python::object main_namespace = main_module.attr("__dict__");
        exec("import numpy", main_namespace);
        boost::python::object numpy = main_namespace["numpy"];

        // Create shape
        boost::python::list ldimensions;
        for(size_t i = 0; i < domains.size(); i++)
            ldimensions.append(domains[i]->GetNumberOfPoints());
        boost::python::tuple shape = boost::python::tuple(ldimensions);

        // Create a flat list of values
        boost::python::list lvalues;
        for(size_t i = nStart; i < nEnd; i++)
            lvalues.append(pDataProxy->GetVariableTypeGathered(i));

        // Create a flat ndarray
        boost::python::dict kwargs;
        kwargs["dtype"] = numpy.attr("int32");
        boost::python::tuple args = boost::python::make_tuple(lvalues);
        boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

        // Return a re-shaped ndarray
        return ndarray.attr("reshape")(shape);
    }
}

boost::python::dict daeVariable_GetDomainsIndexesMap(daeVariable& self, size_t nIndexBase)
{
   // Returns dictionary {integer : [list of integers]}
    boost::python::dict d;
    std::map<size_t, std::vector<size_t> > mapIndexes;
    typedef std::map<size_t, std::vector<size_t> >::iterator c_iterator;

    self.GetDomainsIndexesMap(mapIndexes, nIndexBase);

    for(c_iterator iter = mapIndexes.begin(); iter != mapIndexes.end(); iter++)
        d[iter->first] = getListFromVectorByValue<size_t>(iter->second);

    return d;
}

adouble VariableFunctionCall0(daeVariable& var)
{
    return var();
}

adouble VariableFunctionCall1(daeVariable& var, object o1)
{
    return var(CreateDomainIndex(o1));
}

adouble VariableFunctionCall2(daeVariable& var, object o1, object o2)
{
    return var(CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble VariableFunctionCall3(daeVariable& var, object o1, object o2, object o3)
{
    return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble VariableFunctionCall4(daeVariable& var, object o1, object o2, object o3, object o4)
{
    return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble VariableFunctionCall5(daeVariable& var, object o1, object o2, object o3, object o4, object o5)
{
    return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble VariableFunctionCall6(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble VariableFunctionCall7(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble VariableFunctionCall8(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return var(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

void lAssignValue1(daeVariable& var, boost::python::list indexes, real_t value)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in AssignValue for variable " << var.GetCanonicalName();
            throw e;
        }
    }
    var.AssignValue(narrIndexes, value);
}

void lAssignValue2(daeVariable& var, boost::python::list indexes, const quantity& value)
{
    const daeVariableType* varType = dynamic_cast<const daeVariableType*>(var.GetVariableType());
    real_t val = value.scaleTo(varType->GetUnits()).getValue();
    lAssignValue1(var, indexes, val);
}

void AssignValue0(daeVariable& var, real_t value)
{
    var.AssignValue(value);
}

void AssignValue1(daeVariable& var, size_t n1, real_t value)
{
    var.AssignValue(n1, value);
}

void AssignValue2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
    var.AssignValue(n1, n2, value);
}

void AssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
    var.AssignValue(n1, n2, n3, value);
}

void AssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
    var.AssignValue(n1, n2, n3, n4, value);
}

void AssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
    var.AssignValue(n1, n2, n3, n4, n5, value);
}

void AssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
    var.AssignValue(n1, n2, n3, n4, n5, n6, value);
}

void AssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
    var.AssignValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void AssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
    var.AssignValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void lReAssignValue1(daeVariable& var, boost::python::list indexes, real_t value)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in ReAssignValue for variable " << var.GetCanonicalName();
            throw e;
        }
    }
    var.ReAssignValue(narrIndexes, value);
}

void lReAssignValue2(daeVariable& var, boost::python::list indexes, const quantity& value)
{
    const daeVariableType* varType = dynamic_cast<const daeVariableType*>(var.GetVariableType());
    real_t val = value.scaleTo(varType->GetUnits()).getValue();
    lReAssignValue1(var, indexes, val);
}

void ReAssignValue0(daeVariable& var, real_t value)
{
    var.ReAssignValue(value);
}

void ReAssignValue1(daeVariable& var, size_t n1, real_t value)
{
    var.ReAssignValue(n1, value);
}

void ReAssignValue2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
    var.ReAssignValue(n1, n2, value);
}

void ReAssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
    var.ReAssignValue(n1, n2, n3, value);
}

void ReAssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
    var.ReAssignValue(n1, n2, n3, n4, value);
}

void ReAssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
    var.ReAssignValue(n1, n2, n3, n4, n5, value);
}

void ReAssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
    var.ReAssignValue(n1, n2, n3, n4, n5, n6, value);
}

void ReAssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
    var.ReAssignValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void ReAssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
    var.ReAssignValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void qAssignValue0(daeVariable& var, const quantity& value)
{
    var.AssignValue(value);
}

void qAssignValue1(daeVariable& var, size_t n1, const quantity& value)
{
    var.AssignValue(n1, value);
}

void qAssignValue2(daeVariable& var, size_t n1, size_t n2, const quantity& value)
{
    var.AssignValue(n1, n2, value);
}

void qAssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& value)
{
    var.AssignValue(n1, n2, n3, value);
}

void qAssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& value)
{
    var.AssignValue(n1, n2, n3, n4, value);
}

void qAssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& value)
{
    var.AssignValue(n1, n2, n3, n4, n5, value);
}

void qAssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& value)
{
    var.AssignValue(n1, n2, n3, n4, n5, n6, value);
}

void qAssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& value)
{
    var.AssignValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void qAssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& value)
{
    var.AssignValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void qReAssignValue0(daeVariable& var, const quantity& value)
{
    var.ReAssignValue(value);
}

void qReAssignValue1(daeVariable& var, size_t n1, const quantity& value)
{
    var.ReAssignValue(n1, value);
}

void qReAssignValue2(daeVariable& var, size_t n1, size_t n2, const quantity& value)
{
    var.ReAssignValue(n1, n2, value);
}

void qReAssignValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& value)
{
    var.ReAssignValue(n1, n2, n3, value);
}

void qReAssignValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& value)
{
    var.ReAssignValue(n1, n2, n3, n4, value);
}

void qReAssignValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& value)
{
    var.ReAssignValue(n1, n2, n3, n4, n5, value);
}

void qReAssignValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& value)
{
    var.ReAssignValue(n1, n2, n3, n4, n5, n6, value);
}

void qReAssignValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& value)
{
    var.ReAssignValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void qReAssignValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& value)
{
    var.ReAssignValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

adouble Get_dt0(daeVariable& var)
{
    return var.dt();
}

adouble Get_dt1(daeVariable& var, object o1)
{
    return var.dt(CreateDomainIndex(o1));
}

adouble Get_dt2(daeVariable& var, object o1, object o2)
{
    return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble Get_dt3(daeVariable& var, object o1, object o2, object o3)
{
    return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble Get_dt4(daeVariable& var, object o1, object o2, object o3, object o4)
{
    return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble Get_dt5(daeVariable& var, object o1, object o2, object o3, object o4, object o5)
{
    return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble Get_dt6(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble Get_dt7(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble Get_dt8(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return var.dt(CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

adouble Get_d1(daeVariable& var, daeDomain& d, object o1)
{
    return var.d(d, CreateDomainIndex(o1));
}

adouble Get_d2(daeVariable& var, daeDomain& d, object o1, object o2)
{
    return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble Get_d3(daeVariable& var, daeDomain& d, object o1, object o2, object o3)
{
    return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble Get_d4(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4)
{
    return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble Get_d5(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5)
{
    return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble Get_d6(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble Get_d7(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble Get_d8(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return var.d(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

adouble Get_d21(daeVariable& var, daeDomain& d, object o1)
{
    return var.d2(d, CreateDomainIndex(o1));
}

adouble Get_d22(daeVariable& var, daeDomain& d, object o1, object o2)
{
    return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2));
}

adouble Get_d23(daeVariable& var, daeDomain& d, object o1, object o2, object o3)
{
    return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3));
}

adouble Get_d24(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4)
{
    return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4));
}

adouble Get_d25(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5)
{
    return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5));
}

adouble Get_d26(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6));
}

adouble Get_d27(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7));
}

adouble Get_d28(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return var.d2(d, CreateDomainIndex(o1), CreateDomainIndex(o2), CreateDomainIndex(o3), CreateDomainIndex(o4), CreateDomainIndex(o5), CreateDomainIndex(o6), CreateDomainIndex(o7), CreateDomainIndex(o8));
}

adouble_array VariableArray1(daeVariable& var, object o1)
{
    return var.array(varRANGE(1));
}

adouble_array VariableArray2(daeVariable& var, object o1, object o2)
{
    return var.array(varRANGE(1), varRANGE(2));
}

adouble_array VariableArray3(daeVariable& var, object o1, object o2, object o3)
{
    return var.array(varRANGE(1), varRANGE(2), varRANGE(3));
}

adouble_array VariableArray4(daeVariable& var, object o1, object o2, object o3, object o4)
{
    return var.array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4));
}

adouble_array VariableArray5(daeVariable& var, object o1, object o2, object o3, object o4, object o5)
{
    return var.array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5));
}

adouble_array VariableArray6(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return var.array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6));
}

adouble_array VariableArray7(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return var.array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6), varRANGE(7));
}

adouble_array VariableArray8(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return var.array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6), varRANGE(7), varRANGE(8));
}

adouble_array Get_dt_array1(daeVariable& var, object o1)
{
    return var.dt_array(varRANGE(1));
}

adouble_array Get_dt_array2(daeVariable& var, object o1, object o2)
{
    return var.dt_array(varRANGE(1), varRANGE(2));
}

adouble_array Get_dt_array3(daeVariable& var, object o1, object o2, object o3)
{
    return var.dt_array(varRANGE(1), varRANGE(2), varRANGE(3));
}

adouble_array Get_dt_array4(daeVariable& var, object o1, object o2, object o3, object o4)
{
    return var.dt_array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4));
}

adouble_array Get_dt_array5(daeVariable& var, object o1, object o2, object o3, object o4, object o5)
{
    return var.dt_array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5));
}

adouble_array Get_dt_array6(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return var.dt_array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6));
}

adouble_array Get_dt_array7(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return var.dt_array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6), varRANGE(7));
}

adouble_array Get_dt_array8(daeVariable& var, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return var.dt_array(varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6), varRANGE(7), varRANGE(8));
}


adouble_array Get_d_array1(daeVariable& var, daeDomain& d, object o1)
{
    return var.d_array(d, varRANGE(1));
}

adouble_array Get_d_array2(daeVariable& var, daeDomain& d, object o1, object o2)
{
    return var.d_array(d, varRANGE(1), varRANGE(2));
}

adouble_array Get_d_array3(daeVariable& var, daeDomain& d, object o1, object o2, object o3)
{
    return var.d_array(d, varRANGE(1), varRANGE(2), varRANGE(3));
}

adouble_array Get_d_array4(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4)
{
    return var.d_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4));
}

adouble_array Get_d_array5(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5)
{
    return var.d_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5));
}

adouble_array Get_d_array6(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return var.d_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6));
}

adouble_array Get_d_array7(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return var.d_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6), varRANGE(7));
}

adouble_array Get_d_array8(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return var.d_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6), varRANGE(7), varRANGE(8));
}

adouble_array Get_d2_array1(daeVariable& var, daeDomain& d, object o1)
{
    return var.d2_array(d, varRANGE(1));
}

adouble_array Get_d2_array2(daeVariable& var, daeDomain& d, object o1, object o2)
{
    return var.d2_array(d, varRANGE(1), varRANGE(2));
}

adouble_array Get_d2_array3(daeVariable& var, daeDomain& d, object o1, object o2, object o3)
{
    return var.d2_array(d, varRANGE(1), varRANGE(2), varRANGE(3));
}

adouble_array Get_d2_array4(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4)
{
    return var.d2_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4));
}

adouble_array Get_d2_array5(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5)
{
    return var.d2_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5));
}

adouble_array Get_d2_array6(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6)
{
    return var.d2_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6));
}

adouble_array Get_d2_array7(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7)
{
    return var.d2_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6), varRANGE(7));
}

adouble_array Get_d2_array8(daeVariable& var, daeDomain& d, object o1, object o2, object o3, object o4, object o5, object o6, object o7, object o8)
{
    return var.d2_array(d, varRANGE(1), varRANGE(2), varRANGE(3), varRANGE(4), varRANGE(5), varRANGE(6), varRANGE(7), varRANGE(8));
}

void lSetVariableValue1(daeVariable& var, boost::python::list indexes, real_t value)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in SetValue for variable " << var.GetCanonicalName();
            throw e;
        }
    }
    var.SetValue(narrIndexes, value);
}

void lSetVariableValue2(daeVariable& var, boost::python::list indexes, const quantity& value)
{
    const daeVariableType* varType = dynamic_cast<const daeVariableType*>(var.GetVariableType());
    real_t val = value.scaleTo(varType->GetUnits()).getValue();
    lSetVariableValue1(var, indexes, val);
}

void SetVariableValue0(daeVariable& var, real_t value)
{
    var.SetValue(value);
}

void SetVariableValue1(daeVariable& var, size_t n1, real_t value)
{
    var.SetValue(n1, value);
}

void SetVariableValue2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
    var.SetValue(n1, n2, value);
}

void SetVariableValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
    var.SetValue(n1, n2, n3, value);
}

void SetVariableValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
    var.SetValue(n1, n2, n3, n4, value);
}

void SetVariableValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
    var.SetValue(n1, n2, n3, n4, n5, value);
}

void SetVariableValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
    var.SetValue(n1, n2, n3, n4, n5, n6, value);
}

void SetVariableValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
    var.SetValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void SetVariableValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
    var.SetValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void lSetInitialGuess1(daeVariable& var, boost::python::list indexes, real_t value)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in SetInitialGuess for variable " << var.GetCanonicalName();
            throw e;
        }
    }
    var.SetInitialGuess(narrIndexes, value);
}

void lSetInitialGuess2(daeVariable& var, boost::python::list indexes, const quantity& value)
{
    const daeVariableType* varType = dynamic_cast<const daeVariableType*>(var.GetVariableType());
    real_t val = value.scaleTo(varType->GetUnits()).getValue();
    lSetInitialGuess1(var, indexes, val);
}

void SetInitialGuess0(daeVariable& var, real_t value)
{
    var.SetInitialGuess(value);
}

void SetInitialGuess1(daeVariable& var, size_t n1, real_t value)
{
    var.SetInitialGuess(n1, value);
}

void SetInitialGuess2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
    var.SetInitialGuess(n1, n2, value);
}

void SetInitialGuess3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
    var.SetInitialGuess(n1, n2, n3, value);
}

void SetInitialGuess4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
    var.SetInitialGuess(n1, n2, n3, n4, value);
}

void SetInitialGuess5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
    var.SetInitialGuess(n1, n2, n3, n4, n5, value);
}

void SetInitialGuess6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
    var.SetInitialGuess(n1, n2, n3, n4, n5, n6, value);
}

void SetInitialGuess7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
    var.SetInitialGuess(n1, n2, n3, n4, n5, n6, n7, value);
}

void SetInitialGuess8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
    var.SetInitialGuess(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void lSetInitialCondition1(daeVariable& var, boost::python::list indexes, real_t value)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in SetInitialCondition for variable " << var.GetCanonicalName();
            throw e;
        }
    }
    var.SetInitialCondition(narrIndexes, value);
}

void lSetInitialCondition2(daeVariable& var, boost::python::list indexes, const quantity& value)
{
    const daeVariableType* varType = dynamic_cast<const daeVariableType*>(var.GetVariableType());
    real_t val = value.scaleTo(varType->GetUnits()).getValue();
    lSetInitialCondition1(var, indexes, val);
}

void SetInitialCondition0(daeVariable& var, real_t value)
{
    var.SetInitialCondition(value);
}

void SetInitialCondition1(daeVariable& var, size_t n1, real_t value)
{
    var.SetInitialCondition(n1, value);
}

void SetInitialCondition2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
    var.SetInitialCondition(n1, n2, value);
}

void SetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
    var.SetInitialCondition(n1, n2, n3, value);
}

void SetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
    var.SetInitialCondition(n1, n2, n3, n4, value);
}

void SetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
    var.SetInitialCondition(n1, n2, n3, n4, n5, value);
}

void SetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
    var.SetInitialCondition(n1, n2, n3, n4, n5, n6, value);
}

void SetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
    var.SetInitialCondition(n1, n2, n3, n4, n5, n6, n7, value);
}

void SetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
    var.SetInitialCondition(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void lReSetInitialCondition1(daeVariable& var, boost::python::list indexes, real_t value)
{
    std::vector<size_t> narrIndexes;
    boost::python::ssize_t n = boost::python::len(indexes);
    narrIndexes.resize(n);

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        boost::python::extract<size_t> index(indexes[i]);

        if(index.check())
            narrIndexes[i] = index();
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid type of index [" << i << "] in the list of indexes in ReSetInitialCondition for variable " << var.GetCanonicalName();
            throw e;
        }
    }
    var.ReSetInitialCondition(narrIndexes, value);
}

void lReSetInitialCondition2(daeVariable& var, boost::python::list indexes, const quantity& value)
{
    const daeVariableType* varType = dynamic_cast<const daeVariableType*>(var.GetVariableType());
    real_t val = value.scaleTo(varType->GetUnits()).getValue();
    lReSetInitialCondition1(var, indexes, val);
}

void ReSetInitialCondition0(daeVariable& var, real_t value)
{
    var.ReSetInitialCondition(value);
}

void ReSetInitialCondition1(daeVariable& var, size_t n1, real_t value)
{
    var.ReSetInitialCondition(n1, value);
}

void ReSetInitialCondition2(daeVariable& var, size_t n1, size_t n2, real_t value)
{
    var.ReSetInitialCondition(n1, n2, value);
}

void ReSetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, real_t value)
{
    var.ReSetInitialCondition(n1, n2, n3, value);
}

void ReSetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, real_t value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, value);
}

void ReSetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, real_t value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, n5, value);
}

void ReSetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, real_t value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, n5, n6, value);
}

void ReSetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, real_t value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, n5, n6, n7, value);
}

void ReSetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, real_t value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void qSetVariableValue0(daeVariable& var, const quantity& value)
{
    var.SetValue(value);
}

void qSetVariableValue1(daeVariable& var, size_t n1, const quantity& value)
{
    var.SetValue(n1, value);
}

void qSetVariableValue2(daeVariable& var, size_t n1, size_t n2, const quantity& value)
{
    var.SetValue(n1, n2, value);
}

void qSetVariableValue3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& value)
{
    var.SetValue(n1, n2, n3, value);
}

void qSetVariableValue4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& value)
{
    var.SetValue(n1, n2, n3, n4, value);
}

void qSetVariableValue5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& value)
{
    var.SetValue(n1, n2, n3, n4, n5, value);
}

void qSetVariableValue6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& value)
{
    var.SetValue(n1, n2, n3, n4, n5, n6, value);
}

void qSetVariableValue7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& value)
{
    var.SetValue(n1, n2, n3, n4, n5, n6, n7, value);
}

void qSetVariableValue8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& value)
{
    var.SetValue(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void qSetInitialGuess0(daeVariable& var, const quantity& value)
{
    var.SetInitialGuess(value);
}

void qSetInitialGuess1(daeVariable& var, size_t n1, const quantity& value)
{
    var.SetInitialGuess(n1, value);
}

void qSetInitialGuess2(daeVariable& var, size_t n1, size_t n2, const quantity& value)
{
    var.SetInitialGuess(n1, n2, value);
}

void qSetInitialGuess3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& value)
{
    var.SetInitialGuess(n1, n2, n3, value);
}

void qSetInitialGuess4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& value)
{
    var.SetInitialGuess(n1, n2, n3, n4, value);
}

void qSetInitialGuess5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& value)
{
    var.SetInitialGuess(n1, n2, n3, n4, n5, value);
}

void qSetInitialGuess6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& value)
{
    var.SetInitialGuess(n1, n2, n3, n4, n5, n6, value);
}

void qSetInitialGuess7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& value)
{
    var.SetInitialGuess(n1, n2, n3, n4, n5, n6, n7, value);
}

void qSetInitialGuess8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& value)
{
    var.SetInitialGuess(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void qSetInitialCondition0(daeVariable& var, const quantity& value)
{
    var.SetInitialCondition(value);
}

void qSetInitialCondition1(daeVariable& var, size_t n1, const quantity& value)
{
    var.SetInitialCondition(n1, value);
}

void qSetInitialCondition2(daeVariable& var, size_t n1, size_t n2, const quantity& value)
{
    var.SetInitialCondition(n1, n2, value);
}

void qSetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& value)
{
    var.SetInitialCondition(n1, n2, n3, value);
}

void qSetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& value)
{
    var.SetInitialCondition(n1, n2, n3, n4, value);
}

void qSetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& value)
{
    var.SetInitialCondition(n1, n2, n3, n4, n5, value);
}

void qSetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& value)
{
    var.SetInitialCondition(n1, n2, n3, n4, n5, n6, value);
}

void qSetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& value)
{
    var.SetInitialCondition(n1, n2, n3, n4, n5, n6, n7, value);
}

void qSetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& value)
{
    var.SetInitialCondition(n1, n2, n3, n4, n5, n6, n7, n8, value);
}

void qReSetInitialCondition0(daeVariable& var, const quantity& value)
{
    var.ReSetInitialCondition(value);
}

void qReSetInitialCondition1(daeVariable& var, size_t n1, const quantity& value)
{
    var.ReSetInitialCondition(n1, value);
}

void qReSetInitialCondition2(daeVariable& var, size_t n1, size_t n2, const quantity& value)
{
    var.ReSetInitialCondition(n1, n2, value);
}

void qReSetInitialCondition3(daeVariable& var, size_t n1, size_t n2, size_t n3, const quantity& value)
{
    var.ReSetInitialCondition(n1, n2, n3, value);
}

void qReSetInitialCondition4(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, const quantity& value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, value);
}

void qReSetInitialCondition5(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, const quantity& value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, n5, value);
}

void qReSetInitialCondition6(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, const quantity& value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, n5, n6, value);
}

void qReSetInitialCondition7(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, const quantity& value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, n5, n6, n7, value);
}

void qReSetInitialCondition8(daeVariable& var, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8, const quantity& value)
{
    var.ReSetInitialCondition(n1, n2, n3, n4, n5, n6, n7, n8, value);
}



void AssignValues(daeVariable& var, real_t values)
{
    var.AssignValues(values);
}

void AssignValues2(daeVariable& var, boost::python::object nd_values)
{
/* NUMPY */
    boost::python::object numpy = boost::python::import("numpy");

    // Check the shape of ndarray
    boost::python::tuple shape = boost::python::extract<boost::python::tuple>(nd_values.attr("shape"));
    if(len(shape) != var.GetNumberOfDomains())
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid number of dimensions (" << len(shape) << ") of the array of values in AssignValues for variable " << var.GetCanonicalName()
          << "; the required number of dimensions is " << var.GetNumberOfDomains();
        throw e;
    }
    for(size_t k = 0; k < len(shape); k++)
    {
        size_t dim_avail = boost::python::extract<size_t>(shape[k]);
        size_t dim_req   = var.GetDomain(k)->GetNumberOfPoints();
        if(dim_req != dim_avail)
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid shape of the array of values in AssignValues for variable " << var.GetCanonicalName()
              << "; dimension " << k << " has " << dim_avail << " points (required is " << dim_req << ")";
            throw e;
        }
    }

    // The ndarray must be flattened before use (in the row-major c-style order)
    boost::python::object arg("C");
    boost::python::object values = nd_values.attr("ravel")(arg);
    boost::python::ssize_t n = boost::python::extract<boost::python::ssize_t>(values.attr("size"));
    std::vector<quantity> q_values;
    q_values.resize(n);

    unit u = var.GetVariableType()->GetUnits();

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        // ACHTUNG!! If an item is None set its value to cnUnsetValue (by design: that means unset value)
        boost::python::object obj = values.attr("__getitem__")(i);
        if(obj.is_none() || numpy.attr("isnan")(obj))
        {
            q_values[i] = quantity(cnUnsetValue, u);
        }
        else
        {
            boost::python::extract<real_t>   rValue(obj);
            boost::python::extract<quantity> qValue(obj);

            if(rValue.check())
                q_values[i] = quantity(rValue(), u);
            else if(qValue.check())
                q_values[i] = qValue();
            else
            {
                daeDeclareException(exInvalidCall);
                e << "Invalid type of item [" << i << "] in the list of values in AssignValues for variable " << var.GetCanonicalName();
                throw e;
            }
        }
    }
    var.AssignValues(q_values);
}

void qAssignValues(daeVariable& var, const quantity& q)
{
    var.AssignValues(q);
}

void ReAssignValues(daeVariable& var, real_t values)
{
    var.ReAssignValues(values);
}

void ReAssignValues2(daeVariable& var, boost::python::object nd_values)
{
/* NUMPY */
    boost::python::object numpy = boost::python::import("numpy");

    // Check the shape of ndarray
    boost::python::tuple shape = boost::python::extract<boost::python::tuple>(nd_values.attr("shape"));
    if(len(shape) != var.GetNumberOfDomains())
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid number of dimensions (" << len(shape) << ") of the array of values in ReAssignValues for variable " << var.GetCanonicalName()
          << "; the required number of dimensions is " << var.GetNumberOfDomains();
        throw e;
    }
    for(size_t k = 0; k < len(shape); k++)
    {
        size_t dim_avail = boost::python::extract<size_t>(shape[k]);
        size_t dim_req   = var.GetDomain(k)->GetNumberOfPoints();
        if(dim_req != dim_avail)
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid shape of the array of values in ReAssignValues for variable " << var.GetCanonicalName()
              << "; dimension " << k << " has " << dim_avail << " points (required is " << dim_req << ")";
            throw e;
        }
    }

    // The ndarray must be flattened before use (in the row-major c-style order)
    boost::python::object arg("C");
    boost::python::object values = nd_values.attr("ravel")(arg);
    boost::python::ssize_t n = boost::python::extract<boost::python::ssize_t>(values.attr("size"));
    std::vector<quantity> q_values;
    q_values.resize(n);

    unit u = var.GetVariableType()->GetUnits();

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        // ACHTUNG!! If an item is None set its value to cnUnsetValue (by design: that means unset value)
        boost::python::object obj = values.attr("__getitem__")(i);
        if(obj.is_none() || numpy.attr("isnan")(obj))
        {
            q_values[i] = quantity(cnUnsetValue, u);
        }
        else
        {
            boost::python::extract<real_t>   rValue(obj);
            boost::python::extract<quantity> qValue(obj);

            if(rValue.check())
                q_values[i] = quantity(rValue(), u);
            else if(qValue.check())
                q_values[i] = qValue();
            else
            {
                daeDeclareException(exInvalidCall);
                e << "Invalid type of item [" << i << "] in the list of values in ReAssignValues for variable " << var.GetCanonicalName();
                throw e;
            }
        }
    }
    var.ReAssignValues(q_values);
}

void qReAssignValues(daeVariable& var, const quantity& q)
{
    var.ReAssignValues(q);
}

void SetInitialConditions(daeVariable& var, real_t values)
{
    var.SetInitialConditions(values);
}

void SetInitialConditions2(daeVariable& var, boost::python::object nd_values)
{
/* NUMPY */
    boost::python::object numpy = boost::python::import("numpy");

    // Check the shape of ndarray
    boost::python::tuple shape = boost::python::extract<boost::python::tuple>(nd_values.attr("shape"));
    if(len(shape) != var.GetNumberOfDomains())
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid number of dimensions (" << len(shape) << ") of the array of values in SetInitialConditions for variable " << var.GetCanonicalName()
          << "; the required number of dimensions is " << var.GetNumberOfDomains();
        throw e;
    }
    for(size_t k = 0; k < len(shape); k++)
    {
        size_t dim_avail = boost::python::extract<size_t>(shape[k]);
        size_t dim_req   = var.GetDomain(k)->GetNumberOfPoints();
        if(dim_req != dim_avail)
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid shape of the array of values in SetInitialConditions for variable " << var.GetCanonicalName()
              << "; dimension " << k << " has " << dim_avail << " points (required is " << dim_req << ")";
            throw e;
        }
    }

    // The ndarray must be flattened before use (in the row-major c-style order)
    boost::python::object arg("C");
    boost::python::object values = nd_values.attr("ravel")(arg);
    boost::python::ssize_t n = boost::python::extract<boost::python::ssize_t>(values.attr("size"));
    std::vector<quantity> q_values;
    q_values.resize(n);

    unit u = var.GetVariableType()->GetUnits();

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        // ACHTUNG!! If an item is None set its value to cnUnsetValue (by design: that means unset value)
        boost::python::object obj = values.attr("__getitem__")(i);
        if(obj.is_none() || numpy.attr("isnan")(obj))
        {
            q_values[i] = quantity(cnUnsetValue, u);
        }
        else
        {
            boost::python::extract<real_t>   rValue(obj);
            boost::python::extract<quantity> qValue(obj);

            if(rValue.check())
                q_values[i] = quantity(rValue(), u);
            else if(qValue.check())
                q_values[i] = qValue();
            else
            {
                daeDeclareException(exInvalidCall);
                e << "Invalid type of item [" << i << "] in the list of values in SetInitialConditions for variable " << var.GetCanonicalName();
                throw e;
            }
        }
    }
    var.SetInitialConditions(q_values);
}

void qSetInitialConditions(daeVariable& var, const quantity& q)
{
    var.SetInitialConditions(q);
}

void ReSetInitialConditions(daeVariable& var, real_t values)
{
    var.ReSetInitialConditions(values);
}

void ReSetInitialConditions2(daeVariable& var, boost::python::object nd_values)
{
/* NUMPY */
    boost::python::object numpy = boost::python::import("numpy");

    // Check the shape of ndarray
    boost::python::tuple shape = boost::python::extract<boost::python::tuple>(nd_values.attr("shape"));
    if(len(shape) != var.GetNumberOfDomains())
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid number of dimensions (" << len(shape) << ") of the array of values in ReSetInitialConditions for variable " << var.GetCanonicalName()
          << "; the required number of dimensions is " << var.GetNumberOfDomains();
        throw e;
    }
    for(size_t k = 0; k < len(shape); k++)
    {
        size_t dim_avail = boost::python::extract<size_t>(shape[k]);
        size_t dim_req   = var.GetDomain(k)->GetNumberOfPoints();
        if(dim_req != dim_avail)
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid shape of the array of values in ReSetInitialConditions for variable " << var.GetCanonicalName()
              << "; dimension " << k << " has " << dim_avail << " points (required is " << dim_req << ")";
            throw e;
        }
    }

    // The ndarray must be flattened before use (in the row-major c-style order)
    boost::python::object arg("C");
    boost::python::object values = nd_values.attr("ravel")(arg);
    boost::python::ssize_t n = boost::python::extract<boost::python::ssize_t>(values.attr("size"));
    std::vector<quantity> q_values;
    q_values.resize(n);

    unit u = var.GetVariableType()->GetUnits();

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        // ACHTUNG!! If an item is None set its value to cnUnsetValue (by design: that means unset value)
        boost::python::object obj = values.attr("__getitem__")(i);
        if(obj.is_none() || numpy.attr("isnan")(obj))
        {
            q_values[i] = quantity(cnUnsetValue, u);
        }
        else
        {
            boost::python::extract<real_t>   rValue(obj);
            boost::python::extract<quantity> qValue(obj);

            if(rValue.check())
                q_values[i] = quantity(rValue(), u);
            else if(qValue.check())
                q_values[i] = qValue();
            else
            {
                daeDeclareException(exInvalidCall);
                e << "Invalid type of item [" << i << "] in the list of values in ReSetInitialConditions for variable " << var.GetCanonicalName();
                throw e;
            }
        }
    }
    var.ReSetInitialConditions(q_values);
}

void qReSetInitialConditions(daeVariable& var, const quantity& q)
{
    var.ReSetInitialConditions(q);
}

void SetInitialGuesses(daeVariable& var, real_t values)
{
    var.SetInitialGuesses(values);
}

void SetInitialGuesses2(daeVariable& var, boost::python::object nd_values)
{
/* NUMPY */
    boost::python::object numpy = boost::python::import("numpy");

    // Check the shape of ndarray
    boost::python::tuple shape = boost::python::extract<boost::python::tuple>(nd_values.attr("shape"));
    if(len(shape) != var.GetNumberOfDomains())
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid number of dimensions (" << len(shape) << ") of the array of values in SetInitialGuesses for variable " << var.GetCanonicalName()
          << "; the required number of dimensions is " << var.GetNumberOfDomains();
        throw e;
    }
    for(size_t k = 0; k < len(shape); k++)
    {
        size_t dim_avail = boost::python::extract<size_t>(shape[k]);
        size_t dim_req   = var.GetDomain(k)->GetNumberOfPoints();
        if(dim_req != dim_avail)
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid shape of the array of values in SetInitialGuesses for variable " << var.GetCanonicalName()
              << "; dimension " << k << " has " << dim_avail << " points (required is " << dim_req << ")";
            throw e;
        }
    }

    // The ndarray must be flattened before use (in the row-major c-style order)
    boost::python::object arg("C");
    boost::python::object values = nd_values.attr("ravel")(arg);
    boost::python::ssize_t n = boost::python::extract<boost::python::ssize_t>(values.attr("size"));
    std::vector<quantity> q_values;
    q_values.resize(n);

    unit u = var.GetVariableType()->GetUnits();

    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        // ACHTUNG!! If an item is None set its value to cnUnsetValue (by design: that means unset value)
        boost::python::object obj = values.attr("__getitem__")(i);
        if(obj.is_none() || numpy.attr("isnan")(obj))
        {
            q_values[i] = quantity(cnUnsetValue, u);
        }
        else
        {
            boost::python::extract<real_t>   rValue(obj);
            boost::python::extract<quantity> qValue(obj);

            if(rValue.check())
                q_values[i] = quantity(rValue(), u);
            else if(qValue.check())
                q_values[i] = qValue();
            else
            {
                daeDeclareException(exInvalidCall);
                e << "Invalid type of item [" << i << "] in the list of values in SetInitialGuesses for variable " << var.GetCanonicalName();
                throw e;
            }
        }
    }
    var.SetInitialGuesses(q_values);
}

void qSetInitialGuesses(daeVariable& var, const quantity& q)
{
    var.SetInitialGuesses(q);
}

/*******************************************************
    daePort
*******************************************************/
boost::python::list daePort_GetDomains(daePort& self)
{
    return getListFromVector(self.Domains());
}

boost::python::list daePort_GetParameters(daePort& self)
{
    return getListFromVector(self.Parameters());
}

boost::python::list daePort_GetVariables(daePort& self)
{
    return getListFromVector(self.Variables());
}

boost::python::dict daePort_dictDomains(daePort& self)
{
    return getDictFromObjectArray(self.Domains());
}

boost::python::dict daePort_dictParameters(daePort& self)
{
    return getDictFromObjectArray(self.Parameters());
}

boost::python::dict daePort_dictVariables(daePort& self)
{
    return getDictFromObjectArray(self.Variables());
}

/*******************************************************
    daeEventPort
*******************************************************/
boost::python::list GetEventPortEventsList(daeEventPort& self)
{
    boost::python::list events;
    std::list< std::pair<real_t, real_t> >::const_iterator citer;
    const std::list< std::pair<real_t, real_t> >& listEvents = self.GetListOfEvents();

    for(citer = listEvents.begin(); citer != listEvents.end(); citer++)
        events.append(boost::python::make_tuple(citer->first, citer->second));

    return events;
}

/*******************************************************
    daeEquation
*******************************************************/
daeDEDI* daeEquation_DistributeOnDomain1(daeEquation& self, daeDomain& rDomain, daeeDomainBounds eDomainBounds, const string& strName)
{
    return self.DistributeOnDomain(rDomain, eDomainBounds, strName);
}

daeDEDI* daeEquation_DistributeOnDomain2(daeEquation& self, daeDomain& rDomain, boost::python::list l, const string& strName)
{
    size_t index;
    std::vector<size_t> narrDomainIndexes;
    boost::python::ssize_t n = boost::python::len(l);
    for(boost::python::ssize_t i = 0; i < n; i++)
    {
        index = extract<size_t>(l[i]);
        narrDomainIndexes.push_back(index);
    }

     return self.DistributeOnDomain(rDomain, narrDomainIndexes, strName);
}

boost::python::list daeEquation_GetEquationExecutionInfos(daeEquation& self)
{
    std::vector<daeEquationExecutionInfo*> ptrarr;
    self.GetEquationExecutionInfos(ptrarr);
    return getListFromVector(ptrarr);
}

boost::python::list daeEquation_DistributedEquationDomainInfos(daeEquation& self)
{
    std::vector<daeDistributedEquationDomainInfo_t*> ptrarr;
    self.GetDomainDefinitions(ptrarr);
    return getListFromVectorAndCastPointer<daeDistributedEquationDomainInfo_t*, daeDistributedEquationDomainInfo*>(ptrarr);
}

/*******************************************************
    daeEquationExecutionInfo
*******************************************************/
adNode* daeEquationExecutionInfo_GetNode(daeEquationExecutionInfo& self)
{
    return self.GetEquationEvaluationNodeRawPtr();
}

//void daeFiniteElementEquationExecutionInfo_SetNode(daeFiniteElementEquationExecutionInfo& self, adouble a)
//{
//    if(!a.node)
//    {
//        daeDeclareException(exInvalidCall);
//        e << "Invalid node argument in daeFiniteElementEquationExecutionInfo.SetNode function";
//        throw e;
//    }

//    self.SetEvaluationNode(a.node);
//}

boost::python::list daeEquationExecutionInfo_GetVariableIndexes(daeEquationExecutionInfo& self)
{
    std::vector<size_t> narr;
    self.GetVariableIndexes(narr);
    return getListFromVectorByValue(narr);
}

boost::python::dict daeEquationExecutionInfo_JacobianExpressions(daeEquationExecutionInfo& self)
{
    boost::python::dict d;
    std::map< size_t, std::pair<size_t, adNodePtr> >::const_iterator iter;
    const std::map< size_t, std::pair<size_t, adNodePtr> >& mapJacobianExpressions = self.GetJacobianExpressions();

    for(iter = mapJacobianExpressions.begin(); iter != mapJacobianExpressions.end(); iter++)
        d[iter->first] = boost::python::make_tuple(iter->second.first, boost::cref(iter->second.second.get()));

    return d;
}

/*******************************************************
    daeDEDI
*******************************************************/
daeDomain* daeDEDI_GetDomain(daeDEDI& self)
{
    daeDomain* pDomain = dynamic_cast<daeDomain*>(self.GetDomain());
    return pDomain;
}

boost::python::list daeDEDI_GetDomainPoints(daeDEDI& self)
{
    std::vector<size_t> narrDomainPoints;
    self.GetDomainPoints(narrDomainPoints);
    return getListFromVectorByValue(narrDomainPoints);
}

/*******************************************************
    daePortConnection
*******************************************************/
daeObject* daePortConnection_GetPortFrom(daePortConnection& self)
{
    return dynamic_cast<daeObject*>(self.GetPortFrom());
}

daeObject* daePortConnection_GetPortTo(daePortConnection& self)
{
    return dynamic_cast<daeObject*>(self.GetPortTo());
}

boost::python::list daePortConnection_GetEquations(daePortConnection& self)
{
    std::vector<daeEquation*> ptrarr;
    self.GetEquations(ptrarr);
    return getListFromVector(ptrarr);
}

/*******************************************************
    daeSTN
*******************************************************/
boost::python::list daeSTN_States(daeSTN& self)
{
    return getListFromVector(self.States());
}

boost::python::dict daeSTN_dictStates(daeSTN& self)
{
    return getDictFromObjectArray(self.States());
}

/*******************************************************
    daeOnEventActions
*******************************************************/
boost::python::list daeOnEventActions_Actions(const daeOnEventActions& self)
{
    return getListFromVector(self.Actions());
}

boost::python::list daeOnEventActions_UserDefinedActions(const daeOnEventActions& self)
{
    return getListFromVector(self.UserDefinedActions());
}

/*******************************************************
    daeState
*******************************************************/
boost::python::list daeState_GetEquations(daeState& self)
{
    return getListFromVector(self.Equations());
}

boost::python::list daeState_GetNestedSTNs(daeState& self)
{
    return getListFromVector(self.NestedSTNs());
}

boost::python::list daeState_GetOnEventActions(daeState& self)
{
    return getListFromVector(self.OnEventActions());
}

boost::python::list daeState_GetOnConditionActions(daeState& self)
{
    return getListFromVector(self.OnConditionActions());
}

/*******************************************************
    daeOnConditionActions
*******************************************************/
daeCondition* daeOnConditionActions_Condition(daeOnConditionActions& self)
{
    return self.GetCondition();
}

boost::python::list daeOnConditionActions_Actions(daeOnConditionActions& self)
{
    return getListFromVector(self.Actions());
}

boost::python::list daeOnConditionActions_UserDefinedActions(daeOnConditionActions& self)
{
    return getListFromVector(self.UserDefinedActions());
}

/*******************************************************
    daeSparseMatrixRowIterator_python
*******************************************************/
daeSparseMatrixRowIterator__iter__* daeSparseMatrixRowIterator_iter(daeSparseMatrixRowIterator& self)
{
    self.first();
    return new daeSparseMatrixRowIterator__iter__(self);
}

/*******************************************************
    daeModel
*******************************************************/
void daeModel_def_InitializeModel(daeModel& self, const std::string& jsonInit)
{
    self.InitializeModel(jsonInit);
}

void daeModel_ON_CONDITION(daeModel& self, const daeCondition& rCondition,
                                           boost::python::list switchToStates,
                                           boost::python::list setVariableValues,
                                           boost::python::list triggerEvents,
                                           boost::python::list userDefinedActions,
                                           real_t dEventTolerance)
{
    daeAction* pAction;
    string strSTN;
    string strStateTo;
    std::vector< std::pair<string, string> > arrSwitchToStates;
    daeEventPort* pEventPort;
    std::vector< std::pair<daeVariableWrapper, adouble> > arrSetVariables;
    std::vector< std::pair<daeEventPort*, adouble> > arrTriggerEvents;
    std::vector<daeAction*> ptrarrUserDefinedActions;
    boost::python::ssize_t i, n;
    boost::python::tuple t;

    n = boost::python::len(switchToStates);
    for(i = 0; i < n; i++)
    {
        t = boost::python::extract<boost::python::tuple>(switchToStates[i]);
        if(boost::python::len(t) != 2)
            daeDeclareAndThrowException(exInvalidCall);

        strSTN     = boost::python::extract<string>(t[0]);
        strStateTo = boost::python::extract<string>(t[1]);

        std::pair<string, string> p(strSTN, strStateTo);
        arrSwitchToStates.push_back(p);
    }

    n = boost::python::len(setVariableValues);
    for(i = 0; i < n; i++)
    {
        t = boost::python::extract<boost::python::tuple>(setVariableValues[i]);
        if(boost::python::len(t) != 2)
            daeDeclareAndThrowException(exInvalidCall);

        boost::python::object var = boost::python::extract<boost::python::object>(t[0]);
        boost::python::object o   = boost::python::extract<boost::python::object>(t[1]);

        boost::python::extract<daeVariable*> pvar(var);
        boost::python::extract<adouble>      avar(var);

        boost::python::extract<real_t>  dValue(o);
        boost::python::extract<adouble> aValue(o);

        std::pair<daeVariableWrapper, adouble> p;

        if(pvar.check())
        {
            p.first = daeVariableWrapper(*pvar());
        }
        else if(avar.check())
        {
            adouble a = avar();
            p.first = daeVariableWrapper(a);
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid setVariableValues argument in ON_CONDITION function";
            throw e;
        }

        if(aValue.check())
        {
            p.second = aValue();
        }
        else if(dValue.check())
        {
            p.second = adouble(dValue());
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid setVariableValues argument in ON_CONDITION function";
            throw e;
        }

        arrSetVariables.push_back(p);
    }

    n = boost::python::len(triggerEvents);
    for(i = 0; i < n; i++)
    {
        t = boost::python::extract<boost::python::tuple>(triggerEvents[i]);
        if(boost::python::len(t) != 2)
            daeDeclareAndThrowException(exInvalidCall);

        pEventPort              = boost::python::extract<daeEventPort*>(t[0]);
        boost::python::object o = boost::python::extract<boost::python::object>(t[1]);

        boost::python::extract<real_t>  dValue(o);
        boost::python::extract<adouble> aValue(o);

        std::pair<daeEventPort*, adouble> p;

        p.first = pEventPort;
        if(aValue.check())
        {
            p.second = aValue();
        }
        else if(dValue.check())
        {
            p.second = adouble(dValue());
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid trigger events argument in ON_CONDITION function";
            throw e;
        }

        arrTriggerEvents.push_back(p);
    }

    n = boost::python::len(userDefinedActions);
    for(i = 0; i < n; i++)
    {
        pAction = boost::python::extract<daeAction*>(userDefinedActions[i]);
        if(!pAction)
            daeDeclareAndThrowException(exInvalidPointer);

        ptrarrUserDefinedActions.push_back(pAction);
    }

    self.ON_CONDITION(rCondition,
                      arrSwitchToStates,
                      arrSetVariables,
                      arrTriggerEvents,
                      ptrarrUserDefinedActions,
                      dEventTolerance);
}

void daeModel_ON_EVENT(daeModel& self, daeEventPort* pTriggerEventPort,
                                       boost::python::list switchToStates,
                                       boost::python::list setVariableValues,
                                       boost::python::list triggerEvents,
                                       boost::python::list userDefinedActions)
{
    daeAction* pAction;
    daeEventPort* pEventPort;
    string strSTN;
    string strStateTo;
    std::vector< std::pair<string, string> > arrSwitchToStates;
    std::vector< std::pair<daeVariableWrapper, adouble> > arrSetVariables;
    std::vector< std::pair<daeEventPort*, adouble> > arrTriggerEvents;
    std::vector<daeAction*> ptrarrUserDefinedActions;
    boost::python::ssize_t i, n;
    boost::python::tuple t;

    n = boost::python::len(switchToStates);
    for(i = 0; i < n; i++)
    {
        t = boost::python::extract<boost::python::tuple>(switchToStates[i]);
        if(boost::python::len(t) != 2)
            daeDeclareAndThrowException(exInvalidCall);

        strSTN     = boost::python::extract<string>(t[0]);
        strStateTo = boost::python::extract<string>(t[1]);

        std::pair<string, string> p(strSTN, strStateTo);
        arrSwitchToStates.push_back(p);
    }

    n = boost::python::len(setVariableValues);
    for(i = 0; i < n; i++)
    {
        t = boost::python::extract<boost::python::tuple>(setVariableValues[i]);
        if(boost::python::len(t) != 2)
            daeDeclareAndThrowException(exInvalidCall);

        boost::python::object var = boost::python::extract<boost::python::object>(t[0]);
        boost::python::object o   = boost::python::extract<boost::python::object>(t[1]);

        boost::python::extract<daeVariable*> pvar(var);
        boost::python::extract<adouble>      avar(var);

        boost::python::extract<real_t>  dValue(o);
        boost::python::extract<adouble> aValue(o);

        std::pair<daeVariableWrapper, adouble> p;

        if(pvar.check())
        {
            p.first = daeVariableWrapper(*pvar());
        }
        else if(avar.check())
        {
            adouble a = avar();
            p.first = daeVariableWrapper(a);
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid setVariableValues argument in ON_CONDITION function";
            throw e;
        }

        if(aValue.check())
        {
            p.second = aValue();
        }
        else if(dValue.check())
        {
            p.second = adouble(dValue());
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid setVariableValues argument in ON_EVENT function";
            throw e;
        }

        arrSetVariables.push_back(p);
    }

    n = boost::python::len(triggerEvents);
    for(i = 0; i < n; i++)
    {
        t = boost::python::extract<boost::python::tuple>(triggerEvents[i]);
        if(boost::python::len(t) != 2)
            daeDeclareAndThrowException(exInvalidCall);

        pEventPort              = boost::python::extract<daeEventPort*>(t[0]);
        boost::python::object o = boost::python::extract<boost::python::object>(t[1]);

        boost::python::extract<real_t>  dValue(o);
        boost::python::extract<adouble> aValue(o);

        std::pair<daeEventPort*, adouble> p;

        p.first = pEventPort;
        if(aValue.check())
        {
            p.second = aValue();
        }
        else if(dValue.check())
        {
            p.second = adouble(dValue());
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid triggerEvents argument in ON_EVENT function";
            throw e;
        }

        arrTriggerEvents.push_back(p);
    }

    n = boost::python::len(userDefinedActions);
    for(i = 0; i < n; i++)
    {
        pAction = boost::python::extract<daeAction*>(userDefinedActions[i]);
        if(!pAction)
            daeDeclareAndThrowException(exInvalidPointer);

        ptrarrUserDefinedActions.push_back(pAction);
    }

    self.ON_EVENT(pTriggerEventPort,
                  arrSwitchToStates,
                  arrSetVariables,
                  arrTriggerEvents,
                  ptrarrUserDefinedActions);
}

boost::python::dict daeModel_GetOverallIndex_BlockIndex_VariableNameMap(daeModel& self)
{
    boost::python::dict d;
    std::map<size_t, std::pair<size_t, string> > mapOverallIndex_BlockIndex_VariableName;
    std::map<size_t, std::pair<size_t, string> >::iterator iter;
    const std::map<size_t, size_t>& mapOverallIndex_BlockIndex = self.GetDataProxy()->GetBlock()->m_mapVariableIndexes;

    self.CreateOverallIndex_BlockIndex_VariableNameMap(mapOverallIndex_BlockIndex_VariableName, mapOverallIndex_BlockIndex);

    for(iter = mapOverallIndex_BlockIndex_VariableName.begin(); iter != mapOverallIndex_BlockIndex_VariableName.end(); iter++)
    {
        d[iter->first] = boost::python::make_tuple(iter->second.first, iter->second.second);
    }

    return d;
}

boost::python::list daeModel_GetDomains(daeModel& self)
{
    return getListFromVector(self.Domains());
}

boost::python::list daeModel_GetParameters(daeModel& self)
{
    return getListFromVector(self.Parameters());
}

boost::python::list daeModel_GetVariables(daeModel& self)
{
    return getListFromVector(self.Variables());
}

boost::python::list daeModel_GetPorts(daeModel& self)
{
    return getListFromVector(self.Ports());
}

boost::python::list daeModel_GetEventPorts(daeModel& self)
{
    return getListFromVector(self.EventPorts());
}

boost::python::list daeModel_GetOnEventActions(daeModel& self)
{
    return getListFromVector(self.OnEventActions());
}

boost::python::list daeModel_GetOnConditionActions(daeModel& self)
{
    return getListFromVector(self.OnConditionActions());
}

boost::python::list daeModel_GetPortArrays(daeModel& self)
{
    return getListFromVector(self.PortArrays());
}

boost::python::list daeModel_GetComponents(daeModel& self)
{
    return getListFromVector(self.Models());
}

boost::python::list daeModel_GetComponentArrays(daeModel& self)
{
    return getListFromVector(self.ModelArrays());
}

boost::python::list daeModel_GetSTNs(daeModel& self)
{
    return getListFromVector(self.STNs());
}

boost::python::list daeModel_GetEquations(daeModel& self)
{
    return getListFromVector(self.Equations());
}

boost::python::list daeModel_GetPortConnections(daeModel& self)
{
    return getListFromVector(self.PortConnections());
}

boost::python::list daeModel_GetEventPortConnections(daeModel& self)
{
    return getListFromVector(self.EventPortConnections());
}


boost::python::dict daeModel_dictDomains(daeModel& self)
{
    return getDictFromObjectArray(self.Domains());
}

boost::python::dict daeModel_dictParameters(daeModel& self)
{
    return getDictFromObjectArray(self.Parameters());
}

boost::python::dict daeModel_dictVariables(daeModel& self)
{
    return getDictFromObjectArray(self.Variables());
}

boost::python::dict daeModel_dictPorts(daeModel& self)
{
    return getDictFromObjectArray(self.Ports());
}

boost::python::dict daeModel_dictEventPorts(daeModel& self)
{
    return getDictFromObjectArray(self.EventPorts());
}

boost::python::dict daeModel_dictOnEventActions(daeModel& self)
{
    return getDictFromObjectArray(self.OnEventActions());
}

boost::python::dict daeModel_dictOnConditionActions(daeModel& self)
{
    return getDictFromObjectArray(self.OnConditionActions());
}

boost::python::dict daeModel_dictPortArrays(daeModel& self)
{
    return getDictFromObjectArray(self.PortArrays());
}

boost::python::dict daeModel_dictComponents(daeModel& self)
{
    return getDictFromObjectArray(self.Models());
}

boost::python::dict daeModel_dictComponentArrays(daeModel& self)
{
    return getDictFromObjectArray(self.ModelArrays());
}

boost::python::dict daeModel_dictSTNs(daeModel& self)
{
    return getDictFromObjectArray(self.STNs());
}

boost::python::dict daeModel_dictEquations(daeModel& self)
{
    return getDictFromObjectArray(self.Equations());
}

boost::python::dict daeModel_dictPortConnections(daeModel& self)
{
    return getDictFromObjectArray(self.PortConnections());
}

boost::python::dict daeModel_dictEventPortConnections(daeModel& self)
{
    return getDictFromObjectArray(self.EventPortConnections());
}

/*******************************************************
    daeObjectiveFunction, daeOptimizationConstraint
*******************************************************/
boost::python::object GetGradientsObjectiveFunction(daeObjectiveFunction& self)
{
/* NUMPY
    size_t nType;
    npy_intp dimensions;

    nType      = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
    dimensions = o.GetNumberOfOptimizationVariables();

    boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
    real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
    ::memset(values, 0, dimensions * sizeof(real_t));
    o.GetGradients(values, dimensions);

    return numpy_array;
*/
    // Import numpy
    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import numpy", main_namespace);
    boost::python::object numpy = main_namespace["numpy"];

    // Create shape
    int nVariables = self.GetNumberOfOptimizationVariables();
    boost::python::tuple shape = boost::python::make_tuple(nVariables);

    // Fill gradient array with zeros and then get gradients
    std::vector<real_t> gradients;
    gradients.resize(nVariables, 0.0);
    self.GetGradients(&gradients[0], nVariables);

    // Create a flat list of values from the gradients array
    boost::python::list lvalues;
    for(size_t i = 0; i < nVariables; i++)
        lvalues.append(gradients[i]);

    // Create a flat ndarray
    boost::python::dict kwargs;
    if(typeid(real_t) == typeid(double))
        kwargs["dtype"] = numpy.attr("float64");
    else
        kwargs["dtype"] = numpy.attr("float32");
    boost::python::tuple args = boost::python::make_tuple(lvalues);
    boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

    // Return a re-shaped ndarray
    return ndarray.attr("reshape")(shape);
}

boost::python::object GetGradientsOptimizationConstraint(daeOptimizationConstraint& self)
{
/* NUMPY
    size_t nType;
    npy_intp dimensions;

    nType      = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
    dimensions = o.GetNumberOfOptimizationVariables();

    boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
    real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
    ::memset(values, 0, dimensions * sizeof(real_t));
    o.GetGradients(values, dimensions);

    return numpy_array;
*/
    // Import numpy
    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import numpy", main_namespace);
    boost::python::object numpy = main_namespace["numpy"];

    // Create shape
    int nVariables = self.GetNumberOfOptimizationVariables();
    boost::python::tuple shape = boost::python::make_tuple(nVariables);

    // Fill gradient array with zeros and then get gradients
    std::vector<real_t> gradients;
    gradients.resize(nVariables, 0.0);
    self.GetGradients(&gradients[0], nVariables);

    // Create a flat list of values from the gradients array
    boost::python::list lvalues;
    for(size_t i = 0; i < nVariables; i++)
        lvalues.append(gradients[i]);

    // Create a flat ndarray
    boost::python::dict kwargs;
    if(typeid(real_t) == typeid(double))
        kwargs["dtype"] = numpy.attr("float64");
    else
        kwargs["dtype"] = numpy.attr("float32");
    boost::python::tuple args = boost::python::make_tuple(lvalues);
    boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

    // Return a re-shaped ndarray
    return ndarray.attr("reshape")(shape);
}

boost::python::object GetGradientsMeasuredVariable(daeMeasuredVariable& self)
{
/* NUMPY
    size_t nType;
    npy_intp dimensions;

    nType      = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
    dimensions = o.GetNumberOfOptimizationVariables();

    boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
    real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
    ::memset(values, 0, dimensions * sizeof(real_t));
    o.GetGradients(values, dimensions);

    return numpy_array;
*/
    // Import numpy
    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import numpy", main_namespace);
    boost::python::object numpy = main_namespace["numpy"];

    // Create shape
    int nVariables = self.GetNumberOfOptimizationVariables();
    boost::python::tuple shape = boost::python::make_tuple(nVariables);

    // Fill gradient array with zeros and then get gradients
    std::vector<real_t> gradients;
    gradients.resize(nVariables, 0.0);
    self.GetGradients(&gradients[0], nVariables);

    // Create a flat list of values from the gradients array
    boost::python::list lvalues;
    for(size_t i = 0; i < nVariables; i++)
        lvalues.append(gradients[i]);

    // Create a flat ndarray
    boost::python::dict kwargs;
    if(typeid(real_t) == typeid(double))
        kwargs["dtype"] = numpy.attr("float64");
    else
        kwargs["dtype"] = numpy.attr("float32");
    boost::python::tuple args = boost::python::make_tuple(lvalues);
    boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

    // Return a re-shaped ndarray
    return ndarray.attr("reshape")(shape);
}

}
