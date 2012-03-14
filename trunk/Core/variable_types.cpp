#include "stdafx.h"
#include "coreimpl.h"
#include "units_io.h"

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	daeVariableType
**********************************************************************************************/
daeVariableType::daeVariableType()
{

}

daeVariableType::daeVariableType(string strName,
								 unit   units,
								 real_t dLowerBound,
								 real_t dUpperBound,
								 real_t dInitialGuess,
								 real_t dAbsoluteTolerance)
{
	m_dInitialGuess		  = dInitialGuess;
	m_dLowerBound		  = dLowerBound;
	m_dUpperBound		  = dUpperBound;
	m_Units               = units;
	m_strName             = strName;
	m_dAbsoluteTolerance  = dAbsoluteTolerance;
}

bool daeVariableType::operator ==(const daeVariableType& other)
{
	return  (m_dInitialGuess	   == other.m_dInitialGuess) &&
			(m_dLowerBound		   == other.m_dLowerBound)   &&
			(m_dUpperBound		   == other.m_dUpperBound)   &&
			(m_Units               == other.m_Units)         &&
			(m_strName             == other.m_strName)       &&
			(m_dAbsoluteTolerance  == other.m_dAbsoluteTolerance);
}

bool daeVariableType::operator !=(const daeVariableType& other)
{
	return !(*this == other);
}

daeVariableType::~daeVariableType()
{
}

real_t daeVariableType::GetLowerBound(void) const
{
	return m_dLowerBound;
}

void daeVariableType::SetLowerBound(real_t dValue)
{
	m_dLowerBound = dValue;
}

real_t daeVariableType::GetUpperBound(void) const
{
	return m_dUpperBound;
}

void daeVariableType::SetUpperBound(real_t dValue)
{
	m_dUpperBound = dValue;
}

real_t daeVariableType::GetInitialGuess(void) const
{
	return m_dInitialGuess;
}

void daeVariableType::SetInitialGuess(real_t dValue)
{
	m_dInitialGuess = dValue;
}

unit daeVariableType::GetUnits(void) const
{
	return m_Units;
}

void daeVariableType::SetUnits(const unit& units)
{
	m_Units = units;
}

string daeVariableType::GetName(void) const
{
	return m_strName;
}

void daeVariableType::SetName(string strName)
{
	m_strName = strName;
}

real_t daeVariableType::GetAbsoluteTolerance(void) const
{
	return m_dAbsoluteTolerance;
}

void daeVariableType::SetAbsoluteTolerance(real_t dTolerance)
{
	m_dAbsoluteTolerance = dTolerance;
}

void daeVariableType::Open(io::xmlTag_t* pTag)
{
	string strName;

	strName = "Name";
	pTag->Open(strName, m_strName);

//	strName = "Units";
//	pTag->Open(strName, m_Units);

	strName = "LowerBound";
	pTag->Open(strName, m_dLowerBound);

	strName = "UpperBound";
	pTag->Open(strName, m_dUpperBound);

	strName = "InitialGuess";
	pTag->Open(strName, m_dInitialGuess);

	strName = "AbsoluteTolerance";
	pTag->Open(strName, m_dAbsoluteTolerance);
}

void daeVariableType::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_strName);

	strName = "Units";
	pTag->Save(strName, m_Units.toString());

	strName = "LowerBound";
	pTag->Save(strName, m_dLowerBound);

	strName = "UpperBound";
	pTag->Save(strName, m_dUpperBound);

	strName = "InitialGuess";
	pTag->Save(strName, m_dInitialGuess);

	strName = "AbsoluteTolerance";
	pTag->Save(strName, m_dAbsoluteTolerance);
}

void daeVariableType::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strFile;
	boost::format fmtFile;
	
	if(eLanguage == ePYDAE)
	{
		strFile = c.CalculateIndent(c.m_nPythonIndentLevel) + "%1% = daeVariableType(\"%2%\", \"%3%\", %4%, %5%, %6%, %7%)\n";
		fmtFile.parse(strFile);
		fmtFile % m_strName 
				% m_strName 
		        % units::Export(eLanguage, c, m_Units)
				% m_dLowerBound
				% m_dUpperBound
				% m_dInitialGuess
				% m_dAbsoluteTolerance;
	}
	else if(eLanguage == eCDAE)
	{
		strFile = c.CalculateIndent(c.m_nPythonIndentLevel) + "daeVariableType %1%(\"%2%\", \"%3%\", %4%, %5%, %6%, %7%);\n";
		fmtFile.parse(strFile);
		fmtFile % m_strName 
				% m_strName 
		        % units::Export(eLanguage, c, m_Units)
				% m_dLowerBound
				% m_dUpperBound
				% m_dInitialGuess
				% m_dAbsoluteTolerance;
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented); 
	}
	
	strContent += fmtFile.str();	
}

void daeVariableType::OpenRuntime(io::xmlTag_t* /*pTag*/)
{
}

void daeVariableType::SaveRuntime(io::xmlTag_t* /*pTag*/) const
{
}

bool daeVariableType::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

	if(m_strName.empty())
	{
		strError = "Invalid name in variable type";
		strarrErrors.push_back(strError);
		return false;
	}

//	if(m_strUnits.empty())
//	{
//		strError = "Invalid units in variable type [" + m_strName + "]";
//		strarrErrors.push_back(strError);
//		bCheck = false;
//	}

	if(m_dLowerBound >= m_dUpperBound)
	{
		strError = "Invalid bounds in variable type [" + m_strName + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

	if(m_dInitialGuess < m_dLowerBound || m_dInitialGuess > m_dUpperBound)
	{
		strError = "Invalid initial guess in variable type [" + m_strName + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

	if(m_dAbsoluteTolerance <= 0 || m_dAbsoluteTolerance > 1)
	{
		strError = "Invalid absolute tolerance in variable type [" + m_strName + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

	return bCheck;
}

}
}
