#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#define NO_IMPORT_ARRAY
#include <noprefix.h>
using namespace std;
using namespace boost::python;
#include <boost/property_tree/json_parser.hpp>
#include "../Core/units_io.h"

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
    string strValues = "[";
    for(size_t i = 0; i < self.m_arrValues.size(); i++)
        strValues += (boost::format("%1%%2%") % (i==0 ? "" : ", ") % adouble__str__(self.m_arrValues[i])).str();
    strValues += "]";
    return strValues;
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
			e << "Invalid item for Vector function (must be a floating point value or a quantity object)";
			throw e;
		}
	}
	
	return Array(qarrValues);
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
	return Sum(a);
}
const adouble adarr_product(const adouble_array& a)
{
	return Product(a);
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
	size_t nType;
	npy_intp dimensions;

	nType = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = domain.GetNumberOfPoints();

	boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	for(size_t k = 0; k < (size_t)dimensions; k++)
		values[k] = *domain.GetPoint(k);

	return numpy_array;
}

boost::python::list GetDomainPoints(daeDomain& domain)
{
	boost::python::list l;

	for(size_t i = 0; i < domain.GetNumberOfPoints(); i++)
		l.append(*domain.GetPoint(i));

	return l;
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

adouble_array DomainArray1(daeDomain& domain)
{
	return domain.array();
}

adouble_array DomainArray2(daeDomain& domain, boost::python::slice s)
{
	extract<int> get_start(s.start());
	extract<int> get_end(s.stop());
	extract<int> get_step(s.step());

	int start = get_start.check() ? get_start() : 0;
	int end   = get_end.check()   ? get_end() : domain.GetNumberOfPoints()-1;
	int step  = get_step.check()  ? get_step() : 1;

	return domain.array(start, end, step);
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
boost::python::object GetNumPyArrayParameter(daeParameter& param)
{
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
boost::python::object daeVariable_Values(daeVariable& var)
{
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
        for(size_t i = 0; i < nDomains; i++)
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
}

boost::python::object daeVariable_IDs(daeVariable& var)
{
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

void qAssignValues(daeVariable& var, const quantity& q)
{
    var.AssignValues(q);
}

void ReAssignValues(daeVariable& var, real_t values)
{
	var.ReAssignValues(values);
}

void qReAssignValues(daeVariable& var, const quantity& q)
{
    var.ReAssignValues(q);
}

void SetInitialConditions(daeVariable& var, real_t values)
{
	var.SetInitialConditions(values);
}

void qSetInitialConditions(daeVariable& var, const quantity& q)
{
    var.SetInitialConditions(q);
}

void ReSetInitialConditions(daeVariable& var, real_t values)
{
	var.ReSetInitialConditions(values);
}

void qReSetInitialConditions(daeVariable& var, const quantity& q)
{
    var.ReSetInitialConditions(q);
}

void SetInitialGuesses(daeVariable& var, real_t values)
{
	var.SetInitialGuesses(values);
}

void qSetInitialGuesses(daeVariable& var, const quantity& q)
{
    var.SetInitialGuesses(q);
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
daeDEDI* daeEquation_DistributeOnDomain1(daeEquation& self, daeDomain& rDomain, daeeDomainBounds eDomainBounds)
{
	return self.DistributeOnDomain(rDomain, eDomainBounds);
}

daeDEDI* daeEquation_DistributeOnDomain2(daeEquation& self, daeDomain& rDomain, boost::python::list l)
{
	size_t index;
	std::vector<size_t> narrDomainIndexes;
	boost::python::ssize_t n = boost::python::len(l);
	for(boost::python::ssize_t i = 0; i < n; i++)
	{
		index = extract<size_t>(l[i]);
		narrDomainIndexes.push_back(index);
	}

	 return self.DistributeOnDomain(rDomain, narrDomainIndexes);
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
boost::python::list daeEquationExecutionInfo_GetVariableIndexes(daeEquationExecutionInfo& self)
{
    std::vector<size_t> narr;
    self.GetVariableIndexes(narr);
    return getListFromVectorByValue(narr);
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
boost::python::list GetStatesSTN(daeSTN& self)
{
    std::vector<daeState_t*> ptrarrStates;
	self.GetStates(ptrarrStates);
    return getListFromVectorAndCastPointer<daeState_t*, daeState*>(ptrarrStates);
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
	daeObjectiveFunction, daeOptimizationConstraint
*******************************************************/
boost::python::numeric::array GetGradientsObjectiveFunction(daeObjectiveFunction& o)
{
	size_t nType;
	npy_intp dimensions;

	nType      = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = o.GetNumberOfOptimizationVariables();

	boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	::memset(values, 0, dimensions * sizeof(real_t));
	o.GetGradients(values, dimensions);

	return numpy_array;
}

boost::python::numeric::array GetGradientsOptimizationConstraint(daeOptimizationConstraint& o)
{
	size_t nType;
	npy_intp dimensions;

	nType      = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = o.GetNumberOfOptimizationVariables();

	boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	::memset(values, 0, dimensions * sizeof(real_t));
	o.GetGradients(values, dimensions);

	return numpy_array;
}

boost::python::numeric::array GetGradientsMeasuredVariable(daeMeasuredVariable& o)
{
	size_t nType;
	npy_intp dimensions;

	nType      = (typeid(real_t) == typeid(double) ? NPY_DOUBLE : NPY_FLOAT);
	dimensions = o.GetNumberOfOptimizationVariables();

	boost::python::numeric::array numpy_array(static_cast<boost::python::numeric::array>(handle<>(PyArray_SimpleNew(1, &dimensions, nType))));
	real_t* values = static_cast<real_t*> PyArray_DATA(numpy_array.ptr());
	::memset(values, 0, dimensions * sizeof(real_t));
	o.GetGradients(values, dimensions);

	return numpy_array;
}

}
