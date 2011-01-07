/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_CORE_IMPL_H
#define DAE_CORE_IMPL_H

#include <boost/smart_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/cstdint.hpp>

#include "../config.h"
#include "io_impl.h"
#include "helpers.h"
#include "core.h"
#include "class_factory.h"
#include "adouble.h"

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
// Some M$ crap

#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)

#ifdef AddPort
#undef AddPort
#endif

#ifdef GetCurrentTime
#undef GetCurrentTime
#endif

#ifdef DAEDLL
#ifdef MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllexport)
#else // MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllimport)
#endif // MODEL_EXPORTS
#else // DAEDLL
#define DAE_CORE_API
#endif // DAEDLL

#else // WIN32
#define DAE_CORE_API 
#endif // WIN32


namespace dae 
{
namespace core 
{
enum daeeObjectType
{
	eObjectTypeUnknown = 0,
	eObjectTypeParameter,
	eObjectTypeDomain,
	eObjectTypeVariable,
	eObjectTypePort,
	eObjectTypeModel,
	eObjectTypeSTN,
	eObjectTypeIF,
	eObjectTypeState,
	eObjectTypeModelArray,
	eObjectTypePortArray
};

enum daeeOperandType
{
	eOTUnknown = 0,
	eConstant,
	eDomain,
	eValue,
	eTimeDerivative,
	eFromStack
};

enum daeeResultType
{
	eRTUnknown = 0,
	eToStack
};

/******************************************************************
	daeRuntimeCheck
*******************************************************************/
class daeRuntimeCheck
{
public:
	virtual ~daeRuntimeCheck(void){}

public:
	virtual bool CheckObject(std::vector<string>& strarrErrors) const = 0;
};

/******************************************************************
	daeVariableType
*******************************************************************/
class DAE_CORE_API daeVariableType : public daeVariableType_t,
	                                 public io::daeSerializable,
									 public daeRuntimeCheck
{
public:
	daeDeclareDynamicClass(daeVariableType)
	daeVariableType(void);
	daeVariableType(string strName,
	                string strUnits,
					real_t dLowerBound,
					real_t dUpperBound,
					real_t dInitialGuess,
					real_t dAbsoluteTolerance);
	virtual ~daeVariableType(void);

public:	
	virtual string	GetName(void) const;
	virtual void	SetName(string strName);
	virtual real_t	GetLowerBound(void) const;
	virtual void	SetLowerBound(real_t dValue);
	virtual real_t	GetUpperBound(void) const;
	virtual void	SetUpperBound(real_t dValue);
	virtual real_t	GetInitialGuess(void) const;
	virtual void	SetInitialGuess(real_t dValue);
	virtual string	GetUnits(void) const;
	virtual void	SetUnits(string strName);
	virtual real_t	GetAbsoluteTolerance(void) const;
	virtual void	SetAbsoluteTolerance(real_t dTolerance);

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	bool CheckObject(std::vector<string>& strarrErrors) const;

protected:
	string	m_strName;
	string	m_strUnits;
	real_t	m_dLowerBound;
	real_t	m_dUpperBound;
	real_t	m_dInitialGuess;
	real_t	m_dAbsoluteTolerance;
	friend class daeModel;
	friend class daeVariable;
};

/******************************************************************
	daeObject
*******************************************************************/
class daeModel;
class DAE_CORE_API daeObject : virtual public daeObject_t, 
	                           virtual public io::daeSerializable,
	                           public daeRuntimeCheck
	                           
{
public:
	daeDeclareDynamicClass(daeObject)
	daeObject(void);
	virtual ~daeObject(void);

// Public interface
public:	
	virtual string			GetCanonicalName(void) const;
	virtual string			GetDescription(void) const;
	virtual string			GetName(void) const;
	virtual daeModel_t*		GetModel(void) const;
	virtual void			LogMessage(const string& strMessage, size_t nSeverity) const;

// Local interface
public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	void SaveNameAsMathML(io::xmlTag_t* pTag, string strMathMLTag) const;
	void SaveRelativeNameAsMathML(io::xmlTag_t* pTag, string strMathMLTag, const daeObject* pParent = NULL) const;
	
	bool CheckObject(std::vector<string>& strarrErrors) const;

	void SetCanonicalName(const string& strCanonicalName);
	void SetName(const string& strName);
	void SetDescription(const string& strName);
	void SetModel(daeModel* pModel);

	static string GetRelativeName(const daeObject* parent, const daeObject* child);
	static string GetRelativeName(const string& strParent, const string& strChild);
	string GetNameRelativeToParentModel(void) const;
	
protected:
	string			m_strCanonicalName;
	string			m_strDescription;
	string 			m_strShortName;
	daeModel*		m_pModel;
	friend class daeDomain;
	friend class daeParameter;
	friend class daeVariable;
	friend class daeEquation;
	friend class daePort;
	friend class daeSTN;
	friend class daeIF;
	friend class daeState;
	friend class daeStateTransition;
	friend class daeModel;
	friend class daePortArray;
	friend class daeModelArray;
	friend class daeDistributedEquationDomainInfo;
};

/******************************************************************
	daePartialDerivativeVariable
*******************************************************************/
class daeDomain;
class daeVariable;
class DAE_CORE_API daePartialDerivativeVariable
{
public:
	daePartialDerivativeVariable(const size_t		nOrder,
		                         const daeVariable&	rVariable,
								 const daeDomain&	rDomain,
								 const size_t		nDomainIndex,
								 const size_t		nNoIndexes,
								 const size_t*		pIndexes);
	virtual ~daePartialDerivativeVariable(void);

public:
	adouble				operator()(size_t nIndex);
	adouble				operator[](size_t nIndex);
	size_t				GetOrder(void) const;
	size_t				GetPoint(void) const;
	const daeDomain&	GetDomain(void) const;
	const daeVariable&	GetVariable(void) const;

public:
	size_t*				m_pIndexes;
	const size_t		m_nOrder;
	const daeVariable&	m_rVariable;
	const daeDomain&	m_rDomain;
	const size_t		m_nDomainIndex;
	const size_t		m_nNoIndexes;
};

/******************************************************************
	daeIndexRange
*******************************************************************/
class daeDomain;
class DAE_CORE_API daeIndexRange
{
public:
	daeIndexRange(void);
	daeIndexRange(daeDomain* pDomain);
	daeIndexRange(daeDomain* pDomain, const std::vector<size_t>& narrCustomPoints);
	daeIndexRange(daeDomain* pDomain, int iStartIndex, int iEndIndex, int iStride);
	
public:
	void	Open(io::xmlTag_t* pTag);
	void	Save(io::xmlTag_t* pTag) const;

	size_t	GetNoPoints(void) const;
	void	GetPoints(std::vector<size_t>& narrCustomPoints) const;
	string  ToString(void) const;
	
public:	
	daeDomain*			m_pDomain;
	daeIndexRangeType	m_eType;
	int					m_iStartIndex;
	int					m_iEndIndex;
	int					m_iStride;
	std::vector<size_t>	m_narrCustomPoints;
};

/******************************************************************
	daeDomain
*******************************************************************/
class daePort;
class DAE_CORE_API daeDomain : virtual public daeObject,
	                           virtual public daeDomain_t
{
public:
	daeDeclareDynamicClass(daeDomain)
	daeDomain(void);
	daeDomain(string strName, daeModel* pModel, string strDescription = "");
	daeDomain(string strName, daePort* pPort, string strDescription = "");
	virtual ~daeDomain(void);

// Public interface
public:	
// Common for both Discrete and Distributed domains
	virtual daeeDomainType				GetType(void) const;
	virtual size_t						GetNumberOfIntervals(void) const;
	virtual size_t						GetNumberOfPoints(void) const;
	virtual real_t						GetPoint(size_t nIndex) const;

// Only for Distributed domains
	virtual daeeDiscretizationMethod	GetDiscretizationMethod(void) const;
	virtual size_t						GetDiscretizationOrder(void) const;
	virtual real_t						GetLowerBound(void) const;
	virtual real_t						GetUpperBound(void) const;

	virtual void						SetPoints(std::vector<real_t>& darrPoints);

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	bool CheckObject(std::vector<string>& strarrErrors) const;

	adouble_array	array(void);
	adouble_array	array(int start, int end, int step);
	
	adouble	partial(daePartialDerivativeVariable& pdv) const;

	adouble			operator[](size_t nIndex) const;
	daeIndexRange	operator()(void);
	daeIndexRange	operator()(int start, int end, int step);
	daeIndexRange	operator()(const std::vector<size_t>& narrCustomPoints);
	
	void	CreateArray(size_t nNoIntervals);
	void	CreateDistributed(daeeDiscretizationMethod eMethod, size_t nOrder, size_t nNoIntervals, real_t dLB, real_t dRB);
	
protected:
	void	CreatePoints(void);
	void	SetType(daeeDomainType eDomainType);
	virtual adouble pd_BFD(daePartialDerivativeVariable& pdv) const;
	virtual adouble pd_FFD(daePartialDerivativeVariable& pdv) const;
	virtual adouble pd_CFD(daePartialDerivativeVariable& pdv) const;
	virtual adouble customPartialDerivative(daePartialDerivativeVariable& pdv) const;

protected:
	daeeDiscretizationMethod		m_eDiscretizationMethod;
	size_t							m_nDiscretizationOrder;
	daeeDomainType					m_eDomainType;
	real_t							m_dLowerBound;
	real_t							m_dUpperBound;
	size_t							m_nNumberOfPoints;
	size_t							m_nNumberOfIntervals;
	std::vector<real_t>				m_darrPoints;
	friend class daeParameter;
	friend class daeDistributedEquationDomainInfo;
	friend class daeVariable;
	friend class adSetupDomainIteratorNode;
};

/******************************************************************
	daeBoolArray
*******************************************************************/
class DAE_CORE_API daeBoolArray : public std::vector<bool>
{
public:
	daeBoolArray(void);
	virtual ~daeBoolArray(void);

public:
	void OR(const daeBoolArray& rArray);
	bool CheckOverlapping(const daeBoolArray& rArray);
};

/******************************************************************
	daeEquationExecutionInfo
*******************************************************************/
class daeBlock;
class daeVariable;
class daeEquation;
class DAE_CORE_API daeEquationExecutionInfo : public io::daeSerializable
{
public:
	daeDeclareDynamicClass(daeEquationExecutionInfo)
	daeEquationExecutionInfo(void);
	virtual ~daeEquationExecutionInfo(void);

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

public:
	void GatherInfo(void);
	void Residual(void);
	void Jacobian(void);
	void Sensitivities(const std::vector<size_t>& narrParameterIndexes);
	void Gradients(const std::vector<size_t>& narrParameterIndexes);
	void AddVariableInEquation(size_t nIndex);
	void GetVariableIndexes(std::vector<size_t>& narrVariableIndexes) const;

	size_t GetEquationIndexInBlock(void) const;
	boost::shared_ptr<adNode> GetEquationEvaluationNode(void) const;
	
protected:
	daeBlock*						m_pBlock;
	daeModel*						m_pModel;
	daeEquation*					m_pEquation;
	size_t							m_nEquationIndexInBlock;
	std::vector<size_t>				m_narrDomainIndexes;
	std::map<size_t, size_t> 		m_mapIndexes;
	std::vector<daeDomain*>			m_ptrarrDomains;
	boost::shared_ptr<adNode>		m_EquationEvaluationNode;
	daePtrVector<daeFPUCommand*>	m_ptrarrEquationCommands;
	friend class daeEquation;
	friend class daeSTN;
	friend class daeIF;
	friend class daeModel;
	friend class daeState;
	friend class daeBlock;
};

/******************************************************************
	daeDistributedEquationDomainInfo
*******************************************************************/
class DAE_CORE_API daeDistributedEquationDomainInfo : virtual public daeObject,
	                                                  virtual public daeDistributedEquationDomainInfo_t
{
public:
	daeDeclareDynamicClass(daeDistributedEquationDomainInfo)
	daeDistributedEquationDomainInfo(void);
	daeDistributedEquationDomainInfo(daeEquation* pEquation, daeDomain* pDomain, daeeDomainBounds eDomainBounds);
	daeDistributedEquationDomainInfo(daeEquation* pEquation, daeDomain* pDomain, const std::vector<size_t>& narrDomainIndexes);
	virtual ~daeDistributedEquationDomainInfo(void);

public:
	virtual daeDomain_t*		GetDomain(void) const;
	virtual daeeDomainBounds	GetDomainBounds(void) const; 
	virtual void				GetDomainPoints(std::vector<size_t>& narrDomainPoints) const;

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;
	bool CheckObject(std::vector<string>& strarrErrors) const;
	
	daeEquation*	GetEquation(void) const;
	size_t			GetCurrentIndex(void) const;
	adouble			operator()(void) const;
	void			Initialize(void);

protected:
//	void Initialize(daeDomain* pDomain, daeeDomainBounds eDomainBounds);
//	void Initialize(daeDomain* pDomain, const std::vector<size_t>& narrDomainIndexes);

protected:
	daeDomain*			m_pDomain;
	daeeDomainBounds	m_eDomainBounds; 
	size_t				m_nCurrentIndex;
	daeEquation*		m_pEquation;
	std::vector<size_t>	m_narrDomainPoints;
	friend class daeModel;
	friend class daeDomain;
	friend class daeEquation;
	friend class daeParameter;
	friend class daeVariable;
};
typedef daeDistributedEquationDomainInfo daeDEDI;


/*********************************************************************************************
	daeDomainIndex
**********************************************************************************************/
struct DAE_CORE_API daeDomainIndex
{
public:
	daeDomainIndex(void);
	daeDomainIndex(size_t nIndex);
	daeDomainIndex(daeDistributedEquationDomainInfo* pDEDI) ;

public:
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	
public:
	daeeDomainIndexType					m_eType;
	size_t								m_nIndex;
	daeDistributedEquationDomainInfo*	m_pDEDI;
};

/*********************************************************************************************
	daeArrayRange
**********************************************************************************************/
struct DAE_CORE_API daeArrayRange
{
public:
	daeArrayRange(void);
	daeArrayRange(size_t nIndex);
	daeArrayRange(daeDistributedEquationDomainInfo* pDEDI);
	daeArrayRange(daeIndexRange range);

public:
	size_t GetNoPoints(void) const;

	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	
public:
	daeeRangeType						m_eType;
	daeIndexRange						m_Range;
	size_t								m_nIndex;
	daeDistributedEquationDomainInfo*	m_pDEDI;
};

/******************************************************************
	daeExpressionInfo
*******************************************************************/
class daeStateTransition;
class daeExpressionInfo
{
public:
	daeExpressionInfo(void)
	{
		m_pStateTransition = NULL;
	};

	boost::shared_ptr<adNode>	m_pExpression;
	daeStateTransition* m_pStateTransition;
};

/******************************************************************
	daeExecutionContext
*******************************************************************/
class daeBlock;
class daeDataProxy_t;
class DAE_CORE_API daeExecutionContext
{
public:
	daeExecutionContext(void);

public:
	daeBlock*					m_pBlock;
	daeDataProxy_t*				m_pDataProxy;
	daeEquationExecutionInfo*	m_pEquationExecutionInfo;
	daeeEquationCalculationMode m_eEquationCalculationMode;
	size_t						m_nCurrentVariableIndexForJacobianEvaluation;
	size_t						m_nCurrentParameterIndexForSensitivityEvaluation;
	size_t						m_nIndexInTheArrayOfCurrentParameterForSensitivityEvaluation;
	real_t						m_dInverseTimeStep;
};

/******************************************************************
	daeDataProxy_t
*******************************************************************/
class daeDataProxy_t
{
public:
	daeDataProxy_t(void)
	{
		m_eInitialConditionMode		= eAlgebraicValuesProvided;
		m_pdValues					= NULL;
		m_pdTimeDerivatives			= NULL;
		m_pdInitialConditions		= NULL;
		m_pnVariablesTypes			= NULL;
		m_pnVariablesTypesGathered	= NULL;
		m_pdAbsoluteTolerances		= NULL;
		//m_pExecutionContexts		= NULL;
		m_pTopLevelModel			= NULL;
		m_pLog						= NULL;
		m_bGatherInfo				= false;
		m_nTotalNumberOfVariables	= 0;
		m_nNumberOfParameters		= 0;
		m_eModelType				= eMTUnknown;
//		m_pCondition				= NULL;
		m_pmatSValues				= NULL;
		m_pmatSTimeDerivatives		= NULL; 
		m_pmatSResiduals			= NULL;
	}

	
	virtual ~daeDataProxy_t(void)
	{
		if(m_pdValues)
		{
			delete[] m_pdValues;
			m_pdValues = NULL;
		}
		if(m_pdTimeDerivatives)
		{
			delete[] m_pdTimeDerivatives;
			m_pdTimeDerivatives = NULL;
		}
		if(m_pdInitialConditions)
		{
			delete[] m_pdInitialConditions;
			m_pdInitialConditions = NULL;
		}
		if(m_pnVariablesTypes)
		{
			delete[] m_pnVariablesTypes;
			m_pnVariablesTypes = NULL;
		}
		if(m_pnVariablesTypesGathered)
		{
			delete[] m_pnVariablesTypesGathered;
			m_pnVariablesTypesGathered = NULL;
		}
		if(m_pdAbsoluteTolerances)
		{
			delete[] m_pdAbsoluteTolerances;
			m_pdAbsoluteTolerances = NULL;
		}
		/*
		if(m_pExecutionContexts)
		{
			delete[] m_pExecutionContexts;
			m_pExecutionContexts = NULL;
		}
		*/
//		if(m_pCondition)
//		{
//			delete m_pCondition;
//			m_pCondition = NULL;
//		}
	}

	void Initialize(daeModel_t* pTopLevelModel, daeLog_t* pLog, size_t nTotalNumberOfVariables)
	{
		m_nTotalNumberOfVariables = nTotalNumberOfVariables;

		m_pdValues			= new real_t[m_nTotalNumberOfVariables];
		memset(m_pdValues, 0, m_nTotalNumberOfVariables * sizeof(real_t));

		m_pdTimeDerivatives	= new real_t[m_nTotalNumberOfVariables];
		memset(m_pdTimeDerivatives, 0, m_nTotalNumberOfVariables * sizeof(real_t));

		m_pdInitialConditions	= new real_t[m_nTotalNumberOfVariables];
		memset(m_pdInitialConditions, 0, m_nTotalNumberOfVariables * sizeof(real_t));

		m_pnVariablesTypes			= new int[m_nTotalNumberOfVariables];
		m_pnVariablesTypesGathered	= new int[m_nTotalNumberOfVariables];
		memset(m_pnVariablesTypes,         cnNormal, m_nTotalNumberOfVariables * sizeof(int));
		memset(m_pnVariablesTypesGathered, cnNormal, m_nTotalNumberOfVariables * sizeof(int));

		m_pdAbsoluteTolerances	= new real_t[m_nTotalNumberOfVariables];
		memset(m_pdAbsoluteTolerances, 0, m_nTotalNumberOfVariables * sizeof(real_t));

		//m_pExecutionContexts = new daeExecutionContext[m_nTotalNumberOfVariables];

		m_pTopLevelModel = pTopLevelModel;
		m_pLog           = pLog;

		m_bGatherInfo	 = false;

	}

	void Load(const std::string& strFileName)
	{
		double dValue;
		boost::uint32_t nTotalNumberOfVariables, nFloatTypeSize;
		size_t counter, nFileSize, nRequiredFileSize;
		std::ifstream file;
		
		try
		{
			if(!m_pLog)
				std::cout << "Invalid Log in daeDataProxy_t" << std::endl;
			
			if(m_nTotalNumberOfVariables == 0 || (!m_pdValues))
			{
				m_pLog->Message(string("daeDataProxy_t has not been initialized"), 0);
				return;
			}

			file.open(strFileName.c_str(), std::ios_base::in | std::ios_base::binary);
			if(!file.is_open())
			{
				m_pLog->Message(string("Cannot open the initialization file: ") + strFileName, 0);
				return;
			}
			
			file.seekg(0, std::ifstream::end);
			nFileSize = file.tellg();
			file.seekg(0);
			
			nRequiredFileSize = sizeof(boost::uint32_t) + m_nTotalNumberOfVariables * sizeof(double);
			if(nFileSize != nRequiredFileSize)
			{
				m_pLog->Message(string("The file size of the initialization file ") + 
						        strFileName + string(" does not match; required: ") + 
						        toString<size_t>(nRequiredFileSize) + string(", but available: ") + 
								toString<size_t>(nFileSize), 0);
				return;
			}
			
			file.read((char*)(&nTotalNumberOfVariables), sizeof(boost::uint32_t));
			if(m_nTotalNumberOfVariables != nTotalNumberOfVariables)
			{
				m_pLog->Message(string("The number of variables in the initialization file ") + 
								strFileName + string(": ") + toString<size_t>(nTotalNumberOfVariables) + 
								string(" does not match the number of variables in the simulation: ") + 
								toString<size_t>(m_nTotalNumberOfVariables), 0);
				return;
			}

			counter = 0;
			while(file.good() && counter < m_nTotalNumberOfVariables)
			{
				file.read((char*)(&dValue), sizeof(double));
				m_pdValues[counter] = static_cast<real_t>(dValue);
				counter++;
			}
			
			if(counter < m_nTotalNumberOfVariables)
			{
				m_pLog->Message(string("The initialization file does not contain: ") + 
								toString<size_t>(m_nTotalNumberOfVariables) + 
								string(" variables; found: ") + 
								toString<size_t>(counter), 0);
				return;
			}
			
			file.close();
			m_pLog->Message(string("The initialization file: ") + 
							strFileName + 
							string(" loaded successfuly!"), 0);
		}
		catch(std::exception& e)
		{
			m_pLog->Message(string("An error occured while loading the initialization file: ") + 
							strFileName + 
							string("; ") + 
							e.what(), 0);
		}
	}

	void Store(const std::string& strFileName) const
	{
		double dValue;
		std::ofstream file;
		boost::uint32_t nTotalNumberOfVariables;
		
		try
		{
			if(!m_pLog)
				std::cout << "Invalid Log in daeDataProxy_t" << std::endl;
			
			if(m_nTotalNumberOfVariables == 0 || (!m_pdValues))
			{
				m_pLog->Message(string("daeDataProxy_t has not been initialized"), 0);
				return;
			}
			
			file.open(strFileName.c_str(), std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
			if(!file.is_open())
			{
				m_pLog->Message(string("Cannot open the initialization file: ") + strFileName, 0);
				return;
			}
	
			nTotalNumberOfVariables = static_cast<boost::uint32_t>(m_nTotalNumberOfVariables);
			file.write((char*)(&nTotalNumberOfVariables), sizeof(boost::uint32_t));
			
			for(size_t i = 0; i < m_nTotalNumberOfVariables; i++)
			{
				dValue = static_cast<double>(m_pdValues[i]);
				file.write((char*)(&dValue), sizeof(double));
			}
			
			file.flush();
			file.close();

			m_pLog->Message(string("The initialization file: ") + 
							strFileName + 
							string(" stored successfuly!"), 0);
		}
		catch(std::exception& e)
		{
			m_pLog->Message(string("An error occured while storing the initialization file: ") + 
							strFileName + 
							string("; ") + 
							e.what(), 0);
		}
	}

	size_t GetTotalNumberOfVariables(void) const
	{
		return m_nTotalNumberOfVariables;
	}

	real_t* GetValue(size_t nIndex) const
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		return &(m_pdValues[nIndex]);
	}
	
	void SetValue(size_t nIndex, real_t Value)
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		m_pdValues[nIndex] = Value;
	}

	real_t* GetTimeDerivative(size_t nIndex) const
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		return &(m_pdTimeDerivatives[nIndex]);
	}
	
	void SetTimeDerivative(size_t nIndex, real_t Value)
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		m_pdTimeDerivatives[nIndex] = Value;
	}

// S value: S[nParameterIndex][nVariableIndex]
	real_t GetSValue(size_t nParameterIndex, size_t nVariableIndex) const
	{
		if(!m_pmatSValues)
			daeDeclareAndThrowException(exInvalidPointer);

		if(nParameterIndex >= m_pmatSValues->GetNrows() ||
		   nVariableIndex  >= m_pmatSValues->GetNcols()  )
			daeDeclareAndThrowException(exOutOfBounds)

		return m_pmatSValues->GetItem(nParameterIndex, nVariableIndex);
	}
	
// SD value: SD[nParameterIndex][nVariableIndex]
	real_t GetSDValue(size_t nParameterIndex, size_t nVariableIndex) const
	{
		if(!m_pmatSTimeDerivatives)
			daeDeclareAndThrowException(exInvalidPointer);

		if(nParameterIndex >= m_pmatSTimeDerivatives->GetNrows() ||
		   nVariableIndex  >= m_pmatSTimeDerivatives->GetNcols()  )
			daeDeclareAndThrowException(exOutOfBounds)

		return m_pmatSTimeDerivatives->GetItem(nParameterIndex, nVariableIndex);
	}
	
// Sresidual value: SRes[nParameterIndex][nEquationIndex]
	void SetSResValue(size_t nParameterIndex, size_t nEquationIndex, real_t value)
	{
		if(!m_pmatSResiduals)
			daeDeclareAndThrowException(exInvalidPointer);

		if(nParameterIndex >= m_pmatSResiduals->GetNrows() ||
		   nEquationIndex  >= m_pmatSResiduals->GetNcols()  )
			daeDeclareAndThrowException(exOutOfBounds)

		return m_pmatSResiduals->SetItem(nParameterIndex, nEquationIndex, value);
	}	

	real_t* GetInitialCondition(size_t nIndex) const
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		return &(m_pdInitialConditions[nIndex]);
	}
	
	void SetInitialCondition(size_t nIndex, real_t Value)
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		m_pdInitialConditions[nIndex] = Value;
	}

	int GetVariableType(size_t nIndex) const
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		return m_pnVariablesTypes[nIndex];
	}
	
	void SetVariableType(size_t nIndex, int Value)
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		m_pnVariablesTypes[nIndex] = Value;
	}

	int GetVariableTypeGathered(size_t nIndex) const
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		return m_pnVariablesTypesGathered[nIndex];
	}
	
	void SetVariableTypeGathered(size_t nIndex, int Value)
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		m_pnVariablesTypesGathered[nIndex] = Value;
	}

	real_t* GetAbsoluteTolerance(size_t nIndex) const
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		return &(m_pdAbsoluteTolerances[nIndex]);
	}
	
	void SetAbsoluteTolerance(size_t nIndex, real_t Value)
	{
		if(nIndex >= m_nTotalNumberOfVariables)
			daeDeclareAndThrowException(exOutOfBounds)

		m_pdAbsoluteTolerances[nIndex] = Value;
	}

	void LogMessage(const string& strMessage, size_t nSeverity) const
	{
		if(m_pLog)
			m_pLog->Message(strMessage, nSeverity);
	}

	daeModel_t* GetTopLevelModel(void) const
	{
		return m_pTopLevelModel;
	}

	bool GetGatherInfo(void) const
	{
		return m_bGatherInfo;
	}

	void SetGatherInfo(bool bGatherInfo)
	{
		m_bGatherInfo = bGatherInfo;
	}

//	daeExecutionContext* GetExecutionContext(size_t nIndex) const
//	{
//		return &(m_pExecutionContexts[nIndex]);
//	}
	
	daeeInitialConditionMode GetInitialConditionMode(void) const
	{
		return m_eInitialConditionMode;
	}
	
	void SetInitialConditionMode(daeeInitialConditionMode eMode)
	{
		m_eInitialConditionMode = eMode;
	}

	const std::vector<size_t>& GetOptimizationParametersIndexes(void) const
	{
		return m_narrOptimizationParametersIndexes;
	}
	
	daeeModelType GetModelType(void)
	{
		if(m_eModelType == eMTUnknown)
		{
			if(!m_pnVariablesTypesGathered || m_nTotalNumberOfVariables == 0)
				daeDeclareAndThrowException(exInvalidCall)
			
			bool bFoundDiffVariables = false;
			for(size_t i = 0; i < m_nTotalNumberOfVariables; i++)
			{
				if(m_pnVariablesTypesGathered[i] == cnDifferential)
				{
					bFoundDiffVariables = true;
					break;
				}
			}
			if(bFoundDiffVariables)
				m_eModelType = eDynamicModel;
			else
				m_eModelType = eSteadyStateModel;
		}
		
		return m_eModelType;
	}
	
//	void SetGlobalCondition(daeCondition condition)
//	{
//		daeExecutionContext EC;
//		EC.m_pDataProxy					= this;
//		EC.m_eEquationCalculationMode	= eCreateFunctionsIFsSTNs;
//
//		if(m_pCondition)
//			delete m_pCondition;
//		m_pCondition = new daeCondition;
//		
//		m_pCondition->m_pConditionNode = condition.m_pConditionNode;
//		m_pCondition->m_pModel         = (daeModel*)m_pTopLevelModel;
//		m_pCondition->BuildExpressionsArray(&EC);
//	}
//	
//	void ResetGlobalCondition(void)
//	{
//		if(m_pCondition)
//			delete m_pCondition;
//		m_pCondition = NULL;
//	}
//	
//	daeCondition* GetGlobalCondition() const
//	{
//		return m_pCondition;	
//	}
	
	void SetSensitivityMatrixes(daeMatrix<real_t>* pSValues,
			 				    daeMatrix<real_t>* pSTimeDerivatives,
								daeMatrix<real_t>* pSResiduals)
	{
	// Only SResiduals matrix must not be NULL
		if(!pSResiduals)
			daeDeclareAndThrowException(exInvalidPointer);

		m_pmatSValues          = pSValues;
		m_pmatSTimeDerivatives = pSTimeDerivatives; 
		m_pmatSResiduals       = pSResiduals;
	}
	
	void ResetSensitivityMatrixes(void)
	{
		m_pmatSValues          = NULL;
		m_pmatSTimeDerivatives = NULL; 
		m_pmatSResiduals       = NULL;
	}
	
	bool IsModelDynamic() const
	{
		if(!m_pnVariablesTypesGathered)
			daeDeclareAndThrowException(exInvalidPointer)
			
		for(size_t i = 0; i < m_nTotalNumberOfVariables; i++)
			if(m_pnVariablesTypesGathered[i] == cnDifferential)
				return true;
		return false;		
	}
	
protected:
//	daeCondition*					m_pCondition;
	daeLog_t*						m_pLog;
	daeModel_t*						m_pTopLevelModel;
	size_t							m_nTotalNumberOfVariables;
	real_t*							m_pdValues;
	real_t*							m_pdTimeDerivatives;
	real_t*							m_pdInitialConditions;
	real_t*							m_pdAbsoluteTolerances;
	int*							m_pnVariablesTypes;
	int*							m_pnVariablesTypesGathered;
	bool							m_bGatherInfo;
	//daeExecutionContext*			m_pExecutionContexts;
	daeeInitialConditionMode		m_eInitialConditionMode;
	size_t							m_nNumberOfParameters;
	daeeModelType					m_eModelType;
	std::vector<size_t>				m_narrOptimizationParametersIndexes;
	daeMatrix<real_t>*				m_pmatSValues;
	daeMatrix<real_t>*				m_pmatSTimeDerivatives; 
	daeMatrix<real_t>*				m_pmatSResiduals;
};

/******************************************************************
	daeBlock
*******************************************************************/
class daeSTN;
class DAE_CORE_API daeBlock : public daeBlock_t,
						      public io::daeSerializable,
							  public daeRuntimeCheck
{
public:
	daeDeclareDynamicClass(daeBlock)
	daeBlock(void);
	virtual ~daeBlock(void);

public:
	void	Open(io::xmlTag_t* pTag);
	void	Save(io::xmlTag_t* pTag) const;

// Public block interface used by a solver
	virtual void	Initialize(void);

	virtual void	CalculateResiduals(real_t				dTime, 
									   daeArray<real_t>&	arrValues, 
									   daeArray<real_t>&	arrResiduals, 
									   daeArray<real_t>&	arrTimeDerivatives);

	virtual void	CalculateJacobian(real_t				dTime, 
									  daeArray<real_t>&		arrValues, 
									  daeArray<real_t>&		arrResiduals, 
									  daeArray<real_t>&		arrTimeDerivatives, 
									  daeMatrix<real_t>&	matJacobian, 
									  real_t				dInverseTimeStep);

	virtual void	CalculateSensitivities(real_t					  dTime, 
										   const std::vector<size_t>& narrParameterIndexes,
										   daeArray<real_t>&		  arrValues, 
										   daeArray<real_t>&		  arrTimeDerivatives, 
										   daeMatrix<real_t>&		  matSValues, 
										   daeMatrix<real_t>&		  matSTimeDerivatives, 
										   daeMatrix<real_t>&		  matSResiduals);

	virtual void	CalculateGradients(const std::vector<size_t>& narrParameterIndexes,
									   daeArray<real_t>&		  arrValues, 
									   daeMatrix<real_t>&		  matSResiduals);

	virtual void	CalculateConditions(real_t					dTime, 
									    daeArray<real_t>&		arrValues, 
									    daeArray<real_t>&		arrTimeDerivatives, 
									    std::vector<real_t>&	arrResults);

	virtual void	SetInitialConditionsAndInitialGuesses(daeArray<real_t>& arrValues, 
		                                                  daeArray<real_t>& arrTimeDerivatives, 
														  daeArray<real_t>& arrInitialConditionsTypes);

	virtual void	FillAbsoluteTolerancesArray(daeArray<real_t>& arrAbsoluteTolerances);

	virtual size_t	GetNumberOfEquations(void) const;

	virtual size_t	GetNumberOfRoots(void) const;

	virtual daeeDiscontinuityType CheckDiscontinuities(void);
	
	virtual void	CalcNonZeroElements(int& NNZ);
	virtual void	FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix);

	virtual void	CopyValuesFromSolver(daeArray<real_t>& arrValues);
	virtual void	CopyTimeDerivativesFromSolver(daeArray<real_t>& arrTimeDerivatives);
	virtual void	CopyValuesToSolver(daeArray<real_t>& arrValues);
	virtual void	CopyTimeDerivativesToSolver(daeArray<real_t>& arrTimeDerivatives);

	virtual real_t	GetTime(void) const;
	virtual void	SetTime(real_t time);

	virtual size_t	FindVariableBlockIndex(size_t nVariableOverallIndex) const;
	
	virtual bool	IsModelDynamic() const;
	
public:
	daeDataProxy_t*	GetDataProxy(void) const;
	void			SetDataProxy(daeDataProxy_t* pDataProxy);
	string			GetCanonicalName(void) const;
	string			GetName(void) const;
	void			SetName(const string& strName);

public:
// Used by equations to set Residuals/Jacobian/Hesian values
	real_t	GetResidual(size_t nIndex) const;
	void	SetResidual(size_t nIndex, real_t dResidual);

	real_t	GetJacobian(size_t nEquationIndex, size_t nVariableindexInBlock) const;
	void	SetJacobian(size_t nEquationIndex, size_t nVariableindexInBlock, real_t dJacobianItem);

	real_t	GetInverseTimeStep(void) const;
	void	SetInverseTimeStep(real_t dInverseTimeStep);

public:			
// Used by adNode members during calculation
	real_t GetValue(size_t nOverallIndex) const;
	real_t GetADValue(size_t nOverallIndex) const;
	real_t GetTimeDerivative(size_t nOverallIndex) const;

public:
	void AddEquationExecutionInfo(daeEquationExecutionInfo* pEquationExecutionInfo);
	void GetEquationExecutionInfo(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfos);

	bool CheckOverlappingAndAddVariables(const std::vector<size_t>& narrVariablesInEquation);
	void AddVariables(const std::map<size_t, size_t>& mapIndexes);

	bool GetInitializeMode(void) const;
	void SetInitializeMode(bool bMode);

	bool CheckObject(std::vector<string>& strarrErrors) const;

	// first - index in block;   second - index in core
	std::map<size_t, size_t>& GetVariableIndexesMap(void);

protected:
// Used internally by the block during calculation of Residuals/Jacobian/Hesian
	daeMatrix<real_t>*	GetJacobianMatrix(void) const;
	void				SetJacobianMatrix(daeMatrix<real_t>* pJacobian);

	daeArray<real_t>*	GetResidualArray(void) const;
	void				SetResidualArray(daeArray<real_t>* pResidual);
	
	void				RebuildExpressionMap(void);
	
	void				SetSValuesMatrix(daeMatrix<real_t>* pSValues);
	daeMatrix<real_t>*	GetSValuesMatrix(void) const;

	void				SetSTimeDerivativesMatrix(daeMatrix<real_t>* pSTimeDerivatives);
	daeMatrix<real_t>*	GetSTimeDerivativesMatrix(void) const;

	void				SetSResidualsMatrix(daeMatrix<real_t>* pSResiduals);
	daeMatrix<real_t>*	GetSResidualsMatrix(void) const;

public:
	bool	m_bInitializeMode;
	string	m_strName;

	std::vector<daeEquationExecutionInfo*>	m_ptrarrEquationExecutionInfos;
	std::vector<daeSTN*>					m_ptrarrSTNs;

	std::map<size_t, daeExpressionInfo>		m_mapExpressionInfos; 
	std::map<size_t, size_t>				m_mapVariableIndexes;

	daeDataProxy_t*	m_pDataProxy;

	size_t	m_nCurrentVariableIndexForJacobianEvaluation;

// Given by a solver during Residual/Jacobian/Hesian calculation
	real_t				m_dCurrentTime;
	real_t				m_dInverseTimeStep;
	daeArray<real_t>*	m_parrResidual; 
	daeMatrix<real_t>*	m_pmatJacobian; 
	daeMatrix<real_t>*	m_pmatSValues;
	daeMatrix<real_t>*	m_pmatSTimeDerivatives; 
	daeMatrix<real_t>*	m_pmatSResiduals;
};

/******************************************************************
	daeParameter
*******************************************************************/
class DAE_CORE_API daeParameter : virtual public daeObject,
	                              virtual public daeParameter_t
{
public:
	daeDeclareDynamicClass(daeParameter)
	daeParameter(string strName, daeeParameterType paramType, daeModel* pModel, string strDescription = "");
	daeParameter(string strName, daeeParameterType paramType, daePort* pPort, string strDescription = "");
	daeParameter(void);
	virtual ~daeParameter(void);

public:	
	virtual daeeParameterType	GetParameterType(void) const;
	virtual void				GetDomains(std::vector<daeDomain_t*>& ptrarrDomains);
	virtual real_t*				GetValuePointer(void);

	virtual void	SetValue(real_t value);
	virtual void	SetValue(size_t nD1, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value);

	virtual real_t	GetValue(void);
	virtual real_t	GetValue(size_t nD1);
	virtual real_t	GetValue(size_t nD1, size_t nD2);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8);

public:	
	void	Open(io::xmlTag_t* pTag);
	void	Save(io::xmlTag_t* pTag) const;
	void	OpenRuntime(io::xmlTag_t* pTag);
	void	SaveRuntime(io::xmlTag_t* pTag) const;

	void	DistributeOnDomain(daeDomain& rDomain);

	adouble	operator()(void);
	template<typename TYPE1>
		adouble	operator()(TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble	operator()(TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);

	template<typename TYPE1>
		adouble_array array(TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble_array array(TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);

	bool CheckObject(std::vector<string>& strarrErrors) const;

protected:
	void SetParameterType(daeeParameterType eParameterType);
	void Initialize(void);

	adouble Create_adouble(const size_t* indexes, const size_t N) const;
	adouble	CreateSetupParameter(const daeDomainIndex* indexes, const size_t N) const;
	adouble_array Create_adouble_array(const daeArrayRange* ranges, const size_t N) const;
	adouble_array CreateSetupParameterArray(const daeArrayRange* ranges, const size_t N) const;
	
	void	Fill_adouble_array(std::vector<adouble>& arrValues, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const;
	size_t  CalculateIndex(const size_t* indexes, const size_t N) const;

protected:
	std::vector<real_t>				m_darrValues;
	daeeParameterType				m_eParameterType;
	std::vector<daeDomain*>			m_ptrDomains;
	friend class daePort;
	friend class daeModel;
	friend class adSetupParameterNode;
	friend class adSetupParameterNodeArray;
};	

/******************************************************************
	daeVariable
*******************************************************************/
class DAE_CORE_API daeVariable : virtual public daeObject,
	                             virtual public daeVariable_t
{
public:
	daeDeclareDynamicClass(daeVariable)
	daeVariable(void);
	daeVariable(string strName, const daeVariableType& varType, daeModel* pModel, string strDescription = "");
	daeVariable(string strName, const daeVariableType& varType, daePort* pPort, string strDescription = "");
	virtual ~daeVariable(void);

public:	
	virtual daeVariableType_t*	GetVariableType(void);
	virtual void				GetDomains(std::vector<daeDomain_t*>& ptrarrDomains);

	virtual size_t	GetNumberOfPoints(void) const;
	virtual real_t*	GetValuePointer(void) const;

	virtual bool	GetReportingOn(void) const;
	virtual void	SetReportingOn(bool bOn);

	virtual void	SetValue(real_t value);
	virtual void	SetValue(size_t nD1, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value);
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value);

	virtual real_t	GetValue(void);
	virtual real_t	GetValue(size_t nD1);
	virtual real_t	GetValue(size_t nD1, size_t nD2);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7);
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8);

	virtual void	AssignValue(real_t value);
	virtual void	AssignValue(size_t nD1, real_t value);
	virtual void	AssignValue(size_t nD1, size_t nD2, real_t value);
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, real_t value);
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value);
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value);
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value);
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value);
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value);

	virtual void	ReAssignValue(real_t value);
	virtual void	ReAssignValue(size_t nD1, real_t value);
	virtual void	ReAssignValue(size_t nD1, size_t nD2, real_t value);
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, real_t value);
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value);
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value);
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value);
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value);
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value);
	
	virtual void	SetInitialGuess(real_t dInitialGuess);
	virtual void	SetInitialGuess(size_t nD1, real_t dInitialGuesses);
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, real_t dInitialGuesses);
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, real_t dInitialGuesses);
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialGuesses);
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialGuesses);
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialGuesses);
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialGuesses);
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialGuesses);
	virtual void	SetInitialGuesses(real_t dInitialGuesses);

	virtual void	SetInitialCondition(real_t dInitialCondition);
	virtual void	SetInitialCondition(size_t nD1, real_t dInitialCondition);
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, real_t dInitialCondition);
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, real_t dInitialCondition);
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialCondition);
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialCondition);
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialCondition);
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialCondition);
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialCondition);

	virtual void	ReSetInitialCondition(real_t dInitialCondition);
	virtual void	ReSetInitialCondition(size_t nD1, real_t dInitialCondition);
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, real_t dInitialCondition);
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, real_t dInitialCondition);
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialCondition);
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialCondition);
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialCondition);
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialCondition);
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialCondition);

	virtual void	SetAbsoluteTolerances(real_t dAbsTolerances);

	virtual real_t	TimeDerivative(void);
	virtual real_t	TimeDerivative(size_t nD1);
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2);
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3);
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4);
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5);
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6);
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7);
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8);

	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1);
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2);
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3);
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4);
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5);
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6);
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7);
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8);

	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1);
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2);
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3);
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4);
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5);
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6);
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7);
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8);

public:
	void	Open(io::xmlTag_t* pTag);
	void	Save(io::xmlTag_t* pTag) const;
	void	OpenRuntime(io::xmlTag_t* pTag);
	void	SaveRuntime(io::xmlTag_t* pTag) const;

	void	DistributeOnDomain(daeDomain& rDomain);

public:
	adouble	operator()(void);
	template<typename TYPE1>
		adouble	operator()(TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble	operator()(TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble	operator()(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);

	adouble dt(void);
	template<typename TYPE1>
		adouble	dt(TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble	dt(TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble	dt(TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble	dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble	dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble	dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble	dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble	dt(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);

	template<typename TYPE1>
		adouble	d(const daeDomain_t& rDomain, TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble	d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble	d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble	d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble	d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble	d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble	d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble	d(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);

	template<typename TYPE1>
		adouble	d2(const daeDomain_t& rDomain, TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble	d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble	d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble	d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble	d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble	d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble	d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble	d2(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);
	
	template<typename TYPE1>
		adouble_array array(TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble_array array(TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble_array array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);

	template<typename TYPE1>
		adouble_array dt_array(TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble_array dt_array(TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble_array dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble_array dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble_array dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble_array dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble_array dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble_array	dt_array(TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);

	template<typename TYPE1>
		adouble_array d_array(const daeDomain_t& rDomain, TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble_array d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble_array d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble_array d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble_array d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble_array d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble_array d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble_array d_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);

	template<typename TYPE1>
		adouble_array d2_array(const daeDomain_t& rDomain, TYPE1 d1);
	template<typename TYPE1, typename TYPE2>
		adouble_array d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2);
	template<typename TYPE1, typename TYPE2, typename TYPE3>
		adouble_array d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4>
		adouble_array d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5>
		adouble_array d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6>
		adouble_array d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7>
		adouble_array d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7);
	template<typename TYPE1, typename TYPE2, typename TYPE3, typename TYPE4, typename TYPE5, typename TYPE6, typename TYPE7, typename TYPE8>
		adouble_array d2_array(const daeDomain_t& rDomain, TYPE1 d1, TYPE2 d2, TYPE3 d3, TYPE4 d4, TYPE5 d5, TYPE6 d6, TYPE7 d7, TYPE8 d8);

	bool CheckObject(std::vector<string>& strarrErrors) const;
	
protected:
	adouble	Create_adouble(const size_t* indexes, const size_t N) const;
	adouble	Calculate_dt(const size_t* indexes, const size_t N) const;
	adouble	partial(const size_t nOrder, const daeDomain_t& rDomain, const size_t* indexes, const size_t N) const;
	adouble	CreateSetupVariable(const daeDomainIndex* indexes, const size_t N) const;
	adouble	CreateSetupTimeDerivative(const daeDomainIndex* indexes, const size_t N) const;
	adouble	CreateSetupPartialDerivative(const size_t nOrder, const daeDomain_t& rDomain, const daeDomainIndex* indexes, const size_t N) const;

	adouble_array Create_adouble_array(const daeArrayRange* ranges, const size_t N) const;
	adouble_array Calculate_dt_array(const daeArrayRange* ranges, const size_t N) const;
	adouble_array partial_array(const size_t nOrder, const daeDomain_t& rDomain, const daeArrayRange* ranges, const size_t N) const;
	adouble_array CreateSetupVariableArray(const daeArrayRange* ranges, const size_t N) const;
	adouble_array CreateSetupTimeDerivativeArray(const daeArrayRange* ranges, const size_t N) const;
	adouble_array CreateSetupPartialDerivativeArray(const size_t nOrder, const daeDomain_t& rDomain, const daeArrayRange* ranges, const size_t N) const;
	
//	real_t	GetValueAt(size_t nIndex) const;
//	real_t	GetADValueAt(size_t nIndex) const;
	void	SetVariableType(const daeVariableType& VariableType);
	real_t 	GetInitialCondition(const size_t* indexes, const size_t N);
	void	Fill_adouble_array(std::vector<adouble>& arrValues, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const;
	void	Fill_dt_array(std::vector<adouble>& arrValues, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const;
	void	Fill_partial_array(std::vector<adouble>& arrValues, size_t nOrder, const daeDomain_t& rDomain, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const;

	size_t	CalculateIndex(const size_t* indexes, size_t N) const; 
	size_t	CalculateIndex(const std::vector<size_t>& narrDomainIndexes) const;

protected:
	size_t					m_nOverallIndex;
	daeVariableType			m_VariableType;
	bool					m_bReportingOn;
	std::vector<daeDomain*>	m_ptrDomains;
	friend class daePort;
	friend class daeModel;
	friend class daePortConnection;
	friend class daePartialDerivativeVariable;
	friend class adSetupVariableNode;
	friend class adSetupTimeDerivativeNode;
	friend class adSetupPartialDerivativeNode;
	friend class adSetupVariableNodeArray;
	friend class adSetupTimeDerivativeNodeArray;
	friend class adSetupPartialDerivativeNodeArray;
	friend class adSetupExpressionDerivativeNode;
	friend class adSetupExpressionPartialDerivativeNode;
	friend class daeOptimizationVariable;
	friend class daeObjectiveFunction;
	friend class daeOptimizationConstraint;
};	

/******************************************************************
	daeObjectType
*******************************************************************/
class DAE_CORE_API daeObjectType
{
public:
	daeObjectType(void);
	virtual ~daeObjectType(void);

public:
	daeeObjectType		m_eObjectType;
	string				m_strFullName;
	string				m_strName;
	std::vector<size_t>	m_narrDomains;
	daeObject_t*		m_pObject;

};

/******************************************************************
	daePort
*******************************************************************/
class DAE_CORE_API daePort : virtual public daeObject,
	                         virtual public daePort_t
{
public:
	daeDeclareDynamicClass(daePort)
	daePort(void);
	daePort(string strName, daeePortType portType, daeModel* parent, string strDescription = "");
	virtual ~daePort(void);

public:
	virtual daeePortType	GetType(void) const;
	virtual void			GetDomains(std::vector<daeDomain_t*>& ptrarrDomains);
	virtual void			GetVariables(std::vector<daeVariable_t*>& ptrarrVariables);
	virtual void			GetParameters(std::vector<daeParameter_t*>& ptrarrParameters);

	void AddDomain(daeDomain* pDomain);
	void AddVariable(daeVariable* pVariable);
	void AddParameter(daeParameter* pParameter);
	
	virtual void	SetReportingOn(bool bOn);

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	void SetType(daeePortType eType);

	bool CheckObject(std::vector<string>& strarrErrors) const;
	
	void AddDomain(daeDomain& rDomain, const string& strName, string strDescription = "");
	void AddVariable(daeVariable& rVariable, const string& strName, const daeVariableType& rVariableType, string strDescription = "");
	void AddParameter(daeParameter& rParameter, const string& strName, daeeParameterType eParameterType, string strDescription = "");

protected:
	void			InitializeParameters(void);
	void			InitializeVariables(void);
	size_t			GetNumberOfVariables(void) const;

	size_t			GetVariablesStartingIndex(void) const;
	void			SetVariablesStartingIndex(size_t nVariablesStartingIndex);

	bool			FindObject(std::vector<string>& strarrHierarchy, daeObjectType& ObjectType);
	bool			DetectObject(string& strShortName, std::vector<size_t>& narrDomains, daeeObjectType& eType, daeObject** ppObject);

	daeDomain*		FindDomain(string& strName);
	daeParameter*	FindParameter(string& strName);
	daeVariable*	FindVariable(string& strName);
	daeVariable*	FindVariable(unsigned long nID) const;
	daeDomain*		FindDomain(unsigned long nID) const;

protected:
	void			SetModelAndCanonicalName(daeObject* pObject);

protected:
	daeePortType				m_ePortType;
	daePtrVector<daeVariable*>	m_ptrarrVariables;
	daePtrVector<daeDomain*>	m_ptrarrDomains;
	daePtrVector<daeParameter*>	m_ptrarrParameters;

	size_t					m_nVariablesStartingIndex;
	size_t					_currentVariablesIndex;
	friend class daeModel;
	friend class daePortConnection;
};

/******************************************************************
	daePortConnection
*******************************************************************/
class DAE_CORE_API daePortConnection : virtual public daeObject,
						               virtual public daePortConnection_t
{
public:
	daeDeclareDynamicClass(daePortConnection)
	daePortConnection(void);
	virtual ~daePortConnection(void);

public:
	virtual daePort_t*	GetPortFrom(void) const;
	virtual daePort_t*	GetPortTo(void) const;

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	bool   CheckObject(std::vector<string>& strarrErrors) const;
	void   GetEquations(std::vector<daeEquation*>& ptrarrEquations) const;
	size_t GetTotalNumberOfEquations(void) const;

protected:
	void CreateEquations(void);
	void SetModelAndCanonicalName(daeObject* pObject);

protected:
	daePort*					m_pPortFrom;
	daePort*					m_pPortTo;
	daePtrVector<daeEquation*>	m_ptrarrEquations;
	friend class daeModel;
};

/******************************************************************
	daeModel
*******************************************************************/
class daeState;
class daeIF;
class daeSTN;
class daeModelArray;
class daePortArray;
class DAE_CORE_API daeModel : virtual public daeObject,
						      virtual public daeModel_t
{
public:
	daeDeclareDynamicClass(daeModel)
	daeModel(void);
	daeModel(string strName, daeModel* pModel = NULL, string strDescription = "");
	virtual ~daeModel(void);

public:
	virtual void	GetModelInfo(daeModelInfo& mi) const;
	virtual void	GetSTNs(std::vector<daeSTN_t*>& ptrarrSTNs);
	virtual void	GetPorts(std::vector<daePort_t*>& ptrarrPorts);
	virtual void	GetEquations(std::vector<daeEquation_t*>& ptrarrEquations);
	virtual void	GetModels(std::vector<daeModel_t*>& ptrarrModels);
	virtual void	GetDomains(std::vector<daeDomain_t*>& ptrarrDomains);
	virtual void	GetVariables(std::vector<daeVariable_t*>& ptrarrVariables);
	virtual void	GetParameters(std::vector<daeParameter_t*>& ptrarrParameters);
	virtual void	GetPortConnections(std::vector<daePortConnection_t*>& ptrarrPortConnections);
	virtual void	GetPortArrays(std::vector<daePortArray_t*>& ptrarrPortArrays);
	virtual void	GetModelArrays(std::vector<daeModelArray_t*>& ptrarrModelArrays);

	virtual void	InitializeStage1(void);
	virtual void	InitializeStage2(void);
	virtual void	InitializeStage3(daeLog_t* pLog);
	virtual void	InitializeStage4(void);
	virtual void	InitializeStage5(bool bDoBlockDecomposition, std::vector<daeBlock_t*>& ptrarrBlocks);

	virtual void	SaveModelReport(const string& strFileName) const;
	virtual void	SaveRuntimeModelReport(const string& strFileName) const;

	virtual daeDomain_t*		FindDomain(string& strCanonicalName);
	virtual daeParameter_t*		FindParameter(string& strCanonicalName);
	virtual daeVariable_t*		FindVariable(string& strCanonicalName);
	virtual daePort_t*			FindPort(string& strCanonicalName);
	virtual daeModel_t*			FindModel(string& strCanonicalName);
	virtual daePortArray_t*		FindPortArray(string& strCanonicalName);
	virtual daeModelArray_t*	FindModelArray(string& strCanonicalName);

	virtual void	SetReportingOn(bool bOn);
	
	virtual daeeInitialConditionMode GetInitialConditionMode(void) const;
	virtual void SetInitialConditionMode(daeeInitialConditionMode eMode);
	virtual void SetInitialConditions(real_t value);
	
	virtual void StoreInitializationValues(const std::string& strFileName) const;
	virtual void LoadInitializationValues(const std::string& strFileName) const;
	
	virtual bool IsModelDynamic() const;

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	void daeSaveRuntimeNodes(string strFileName);

	virtual size_t	GetTotalNumberOfVariables(void) const;
	virtual size_t	GetTotalNumberOfEquations(void) const;

	size_t			GetNumberOfSTNs(void) const;

	void SetGlobalConditionContext(void);
	void UnsetGlobalConditionContext(void);
	void SetGlobalCondition(daeCondition condition);
	void ResetGlobalCondition(void);
	daeCondition* GetGlobalCondition() const;

	const adouble dt(const adouble& a) const;
	const adouble d(const adouble& a, daeDomain& domain) const;
	const adouble sum(const adouble_array& a) const;
	const adouble product(const adouble_array& a) const;
	const adouble min(const adouble_array& a) const;
	const adouble max(const adouble_array& a) const;
	const adouble average(const adouble_array& a) const;
	const adouble integral(const adouble_array& a) const;

// Internal functions
	const adouble __sum__(const adouble_array& a) const;
	const adouble __product__(const adouble_array& a) const;
	const adouble __min__(const adouble_array& a) const;
	const adouble __max__(const adouble_array& a) const;
	const adouble __average__(const adouble_array& a) const;
	const adouble __integral__(const adouble_array& a, daeDomain* pDomain, const std::vector<size_t>& narrPoints) const;

	bool CheckObject(std::vector<string>& strarrErrors) const;

	daeEquation* CreateEquation(const string& strName, string strDescription = "");

	void AddEquation(daeEquation* pEquation);
	void AddDomain(daeDomain* pDomain);
	void AddVariable(daeVariable* pVariable);
	void AddParameter(daeParameter* pParameter);
	void AddModel(daeModel* pModel);
	void AddPort(daePort* pPort);
	void AddPortConnection(daePortConnection* pPortConnection);
	void AddPortArray(daePortArray* pPortArray);
	void AddModelArray(daeModelArray* pModelArray);

// Overridables		
public:
	//virtual void DeclareData(void);
	virtual void DeclareEquations(void);

protected:
	void AddDomain(daeDomain& rDomain, const string& strName, string strDescription = "");
	void AddVariable(daeVariable& rVariable, const string& strName, const daeVariableType& rVariableType, string strDescription = "");
	void AddParameter(daeParameter& rParameter, const string& strName, daeeParameterType eParameterType, string strDescription = "");
	void AddModel(daeModel& rModel, const string& strName, string strDescription = "");
	void AddPort(daePort& rPort, const string& strName, daeePortType ePortType, string strDescription = "");
	void AddPortArray(daePortArray& rPortArray, const string& strName, daeePortType ePortType, string strDescription = "");
	void AddModelArray(daeModelArray& rModelArray, const string& strName, string strDescription = "");

	//void DeclareDataBase(void);
	void DeclareEquationsBase(void);

	daeSTN* AddSTN(const string& strName);
	daeIF*  AddIF(const string& strCondition);

	void IF(const daeCondition& rCondition, real_t dEventTolerance = 0);
	void ELSE_IF(const daeCondition& rCondition, real_t dEventTolerance = 0);
	void ELSE(void);
	void END_IF(void);
	
	daeSTN*   STN(const string& strSTN);
	daeState* STATE(const string& strState);
	void      END_STN(void);
	void      SWITCH_TO(const string& strState, const daeCondition& rCondition, real_t dEventTolerance = 0);

	template<typename Model>
		daeEquation* AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(void));
	template<typename Model>
		daeEquation* AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t));
	template<typename Model>
		daeEquation* AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t));
	template<typename Model>
		daeEquation* AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t));
	template<typename Model>
		daeEquation* AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t));
	template<typename Model>
		daeEquation* AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t));
	template<typename Model>
		daeEquation* AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t));
	template<typename Model>
		daeEquation* AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t));
	template<typename Model>
		daeEquation* AddEquation(const string& strFunctionName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t));

	void ConnectPorts(daePort* pPortFrom, daePort* pPortTo);

public:
	daeDomain*		FindDomain(unsigned long nID) const;
	daePort*		FindPort(unsigned long nID) const;
	daeVariable*	FindVariable(unsigned long nID) const;

// Internal functions
protected:
	void		InitializeParameters(void);
	void		InitializeVariables(void);
	void		InitializeEquations(void);
	void		InitializeDEDIs(void);
	void		InitializePortAndModelArrays(void);
	void		InitializeSTNs(void);
	void		DoBlockDecomposition(bool bDoBlockDecomposition, std::vector<daeBlock_t*>& ptrarrBlocks);
	void		SetDefaultAbsoluteTolerances(void);
	void		SetDefaultInitialGuesses(void);	
	void		BuildUpSTNsAndEquations(void);
	void		BuildUpPortConnectionEquations();
	void		CreatePortConnectionEquations(void);

	void		PropagateDataProxy(boost::shared_ptr<daeDataProxy_t> pDataProxy);
	void		PropagateGlobalExecutionContext(daeExecutionContext* pExecutionContext);

	size_t		GetVariablesStartingIndex(void) const;
	void		SetVariablesStartingIndex(size_t nVariablesStartingIndex);

	void		AddEquationExecutionInfo(daeEquationExecutionInfo* pEquationExecutionInfo);
	void		GetEquationExecutionInfo(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfos);
	void		CreateEquationExecutionInfo(daeEquation* pEquation, std::vector<daeEquationExecutionInfo*>& ptrarrEqnExecutionInfosCreated, bool bAddToTheModel);

	bool		FindObject(string& strCanonicalName, daeObjectType& ObjectType);
	bool		FindObject(std::vector<string>& strarrHierarchy, daeObjectType& ObjectType);

protected:
	void		CollectAllSTNs(std::vector<daeSTN*>& ptrarrSTNs) const;
	void		CollectEquationExecutionInfosFromModels(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const;
	void		CollectEquationExecutionInfosFromSTNs(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const;
	void		SetModelAndCanonicalName(daeObject* pObject);
	bool		DetectObject(string& strShortName, std::vector<size_t>& narrDomains, daeeObjectType& eType, daeObject_t** ppObject);

// Used to nest States
	daeState*	GetStateFromStack(void);
	void		PutStateToStack(daeState* pState);
	void		RemoveStateFromStack(void);

protected:
	daeCondition*						m_pCondition;
	boost::shared_ptr<daeDataProxy_t>	m_pDataProxy;
	size_t								m_nVariablesStartingIndex;
	size_t								m_nTotalNumberOfVariables;

	daePtrVector<daeModel*>			m_ptrarrModels;
	daePtrVector<daeEquation*>		m_ptrarrEquations;
	daePtrVector<daeSTN*>			m_ptrarrSTNs;
	daePtrVector<daePortConnection*> m_ptrarrPortConnections;
// When used programmatically they dont own pointers
	daePtrVector<daeDomain*>		m_ptrarrDomains;
	daePtrVector<daeParameter*>		m_ptrarrParameters;
	daePtrVector<daeVariable*>		m_ptrarrVariables;
	daePtrVector<daePort*>			m_ptrarrPorts;
	daePtrVector<daePortArray*>		m_ptrarrPortArrays;
	daePtrVector<daeModelArray*>	m_ptrarrModelArrays;

	daePtrVector<daeEquationExecutionInfo*> m_ptrarrEquationExecutionInfos;

	boost::shared_array<daeExecutionContext>	m_pExecutionContexts;

// Used to nest STNS
	std::vector<daeState*> m_ptrarrStackStates;

	daeSTN*	_currentSTN;

	size_t	_currentVariablesIndex;
// Used only during GatherInfo
	daeExecutionContext*	m_pExecutionContextForGatherInfo;
	
	daeExecutionContext		m_globalConditionContext;

	friend class daeIF;
	friend class daeSTN;
	friend class daeStateTransition;
	friend class daePort;
	friend class daeObject;
	friend class daeDomain;
	friend class daeVariable;
	friend class daeParameter;
	friend class daeEquation;
	friend class daeEquationExecutionInfo;
	friend class daeDistributedEquationDomainInfo;
	friend class daeOptimizationVariable;
};

/******************************************************************
	daeModelArray
*******************************************************************/
class DAE_CORE_API daeModelArray : virtual public daeObject,
	                               virtual public daeModelArray_t
{
public:
	daeDeclareDynamicClass(daeModelArray)
	daeModelArray(int n);
	virtual ~daeModelArray(void);
	
public:
	virtual size_t	GetDimensions(void) const;
	virtual void	GetDomains(std::vector<daeDomain_t*>& ptrarrDomains);

	virtual daeModel_t* GetModel(size_t n1);
	virtual daeModel_t* GetModel(size_t n1, size_t n2);
	virtual daeModel_t* GetModel(size_t n1, size_t n2, size_t n3);
	virtual daeModel_t* GetModel(size_t n1, size_t n2, size_t n3, size_t n4);
	virtual daeModel_t* GetModel(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5);

public:
	void DistributeOnDomain(daeDomain& rDomain);
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;	
	bool CheckObject(std::vector<string>& strarrErrors) const;
	
	size_t GetVariablesStartingIndex(void) const;
	void   SetVariablesStartingIndex(size_t nVariablesStartingIndex);

protected:
	virtual size_t	GetTotalNumberOfVariables(void) const																		= 0;
	virtual size_t	GetTotalNumberOfEquations(void) const																		= 0;
	virtual void DeclareData(void)																								= 0;
	virtual void DeclareEquations(void)																							= 0;
	virtual void InitializeParameters(void)																						= 0;
	virtual void InitializeVariables(void)																						= 0;
	virtual void InitializeEquations(void)																						= 0;
	virtual void InitializeSTNs(void)																							= 0;
	virtual void InitializeDEDIs(void)																		                    = 0;
	virtual void CreatePortConnectionEquations(void)																			= 0;
	virtual void PropagateDataProxy(boost::shared_ptr<daeDataProxy_t> pDataProxy)												= 0;
	virtual void PropagateGlobalExecutionContext(daeExecutionContext* pExecutionContext)										= 0;
	virtual void CollectAllSTNs(std::vector<daeSTN*>& ptrarrSTNs) const																= 0;
	virtual void CollectEquationExecutionInfosFromModels(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const	= 0;
	virtual void CollectEquationExecutionInfosFromSTNs(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const	= 0;
	virtual void SetDefaultAbsoluteTolerances(void)																				= 0;
	virtual void SetDefaultInitialGuesses(void)																					= 0;	
	
protected:
	virtual void Create(void);
	daeModel_t*	GetModel(std::vector<size_t>& narrIndexes);

protected:
	size_t					_currentVariablesIndex;
	size_t					m_nVariablesStartingIndex;
	const int				N;
	std::vector<daeDomain*>	m_ptrarrDomains;
	friend class daeModel;
};

/******************************************************************
	daePortArray
*******************************************************************/
class DAE_CORE_API daePortArray : virtual public daeObject,
	                              virtual public daePortArray_t
{
public:
	daeDeclareDynamicClass(daePortArray)
	daePortArray(int n);
	virtual ~daePortArray(void);
	
public:
	virtual size_t	GetDimensions(void) const;
	virtual void	GetDomains(std::vector<daeDomain_t*>& ptrarrDomains);

	virtual daePort_t* GetPort(size_t n1);
	virtual daePort_t* GetPort(size_t n1, size_t n2);
	virtual daePort_t* GetPort(size_t n1, size_t n2, size_t n3);
	virtual daePort_t* GetPort(size_t n1, size_t n2, size_t n3, size_t n4);
	virtual daePort_t* GetPort(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5);
	
public:
	void DistributeOnDomain(daeDomain& rDomain);
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	bool   CheckObject(std::vector<string>& strarrErrors) const;
	size_t GetVariablesStartingIndex(void) const;
	void   SetVariablesStartingIndex(size_t nVariablesStartingIndex);
	
protected:
	virtual void DeclareData(void)					= 0;
	virtual void InitializeParameters(void)			= 0;
	virtual void InitializeVariables(void)			= 0;
	virtual void SetDefaultAbsoluteTolerances(void)	= 0;
	virtual void SetDefaultInitialGuesses(void)		= 0;	

protected:
	virtual void Create(void);
	daePort_t*	 GetPort(std::vector<size_t>& narrIndexes);

protected:
	size_t					_currentVariablesIndex;
	size_t					m_nVariablesStartingIndex;
	const int				N;
	daeePortType			m_ePortType;
	std::vector<daeDomain*>	m_ptrarrDomains;
	friend class daeModel;
};

/******************************************************************
	daeState
*******************************************************************/
class daeStateTransition;
class daeSTN;
class DAE_CORE_API daeState : virtual public daeObject,
						      virtual public daeState_t
{
public:
	daeDeclareDynamicClass(daeState)
	daeState(void);
	virtual ~daeState(void);

public:
	virtual void	GetStateTransitions(std::vector<daeStateTransition_t*>& ptrarrStateTransitions);
	virtual void	GetEquations(std::vector<daeEquation_t*>& ptrarrEquations);
	virtual void	GetNestedSTNs(std::vector<daeSTN_t*>& ptrarrSTNs);

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	void AddEquation(daeEquation* pEquation);
	//daeEquation* AddEquation(const string& strEquationExpression);

	size_t GetNumberOfEquations(void) const;
	size_t GetNumberOfStateTransitions(void) const;
	size_t GetNumberOfSTNs(void) const;

	template<class Model>
		daeEquation* AddEquation(const string& strName, adouble (Model::*Calculate)(void));
	template<class Model>
		daeEquation* AddEquation(const string& strName, adouble (Model::*Calculate)(size_t));
	template<class Model>
		daeEquation* AddEquation(const string& strName, adouble (Model::*Calculate)(size_t, size_t));
	template<class Model>
		daeEquation* AddEquation(const string& strName, adouble (Model::*Calculate)(size_t, size_t, size_t));
	template<class Model>
		daeEquation* AddEquation(const string& strName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t));
	template<class Model>
		daeEquation* AddEquation(const string& strName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t));
	template<class Model>
		daeEquation* AddEquation(const string& strName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t));
	template<class Model>
		daeEquation* AddEquation(const string& strName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t));
	template<class Model>
		daeEquation* AddEquation(const string& strName, adouble (Model::*Calculate)(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t));

	bool CheckObject(std::vector<string>& strarrErrors) const;
	void CalcNonZeroElements(int& NNZ);
	void FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix);

protected:
	void	Create(const string& strName, daeSTN* pSTN);
	void	InitializeStateTransitions(void);
	void	InitializeDEDIs(void);

	void	AddStateTransition(daeStateTransition* pStateTransition);
	void	AddNestedSTN(daeSTN* pSTN);

	daeSTN*	GetSTN(void) const;
	void	SetSTN(daeSTN* pSTN);

	void	SetModelAndCanonicalName(daeObject* pObject);

protected:
	daeSTN*									m_pSTN;
	daePtrVector<daeEquation*>				m_ptrarrEquations;
	daePtrVector<daeStateTransition*>		m_ptrarrStateTransitions;
	daePtrVector<daeEquationExecutionInfo*> m_ptrarrEquationExecutionInfos;
	daePtrVector<daeSTN*>					m_ptrarrSTNs;
	friend class daeIF;
	friend class daeSTN;
	friend class daeModel;
	friend class daeStateTransition;
};

/******************************************************************
	daeStateTransition
*******************************************************************/
class DAE_CORE_API daeStateTransition : virtual public daeObject,
						                virtual public daeStateTransition_t
{
public:
	daeDeclareDynamicClass(daeStateTransition)
	daeStateTransition(void);
	virtual ~daeStateTransition(void);

public:
	virtual daeState_t* GetStateTo(void) const;
	virtual daeState_t*	GetStateFrom(void) const;

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	void Initialize(void);
	void CreateSTN(const string& strCondition, daeState* pStateFrom, const string& strStateToName, const daeCondition& rCondition, real_t dEventTolerance);
	void CreateIF(const string& strCondition,  daeState* pStateTo, const daeCondition& rCondition, real_t dEventTolerance);
	bool CheckObject(std::vector<string>& strarrErrors) const;
	string GetConditionAsString() const;
	
protected:
	daeCondition*		GetCondition(void);
	void				SetCondition(daeCondition& rCondition);
	void				SetStateTo(daeState* pState);
	void				SetStateFrom(daeState* pState);

protected:
	string								m_strStateToName;
	daeSTN*								m_pSTN;
	daeState*							m_pStateFrom;
	daeState*							m_pStateTo;
	daeCondition						m_Condition;
	std::map<size_t, daeExpressionInfo>	m_mapExpressionInfos; 

// Internal variables, used only during Open
	long	m_nStateFromID;
	long	m_nStateToID;
	friend class daeIF;
	friend class daeSTN;
};

/******************************************************************
	daeSTN
*******************************************************************/
class DAE_CORE_API daeSTN : virtual public daeObject,
						    virtual public daeSTN_t
{
public:
	daeDeclareDynamicClass(daeSTN)
	daeSTN(void);
	virtual ~daeSTN(void);

public:
	virtual void		GetStates(std::vector<daeState_t*>& ptrarrStates);
	virtual daeState_t*	GetActiveState(void);

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	virtual void		Initialize(void);
	virtual bool		CheckDiscontinuities(void);
	virtual size_t		GetNumberOfEquations(void) const;
	virtual daeState*	AddState(string strName);

	size_t			GetNumberOfStates(void) const;
	void			SetActiveState(const string& strStateName);
	void			SetActiveState(daeState* pState);
	bool			CheckObject(std::vector<string>& strarrErrors) const;
	void			CalcNonZeroElements(int& NNZ);
	void			FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix);

	daeState*		GetParentState(void) const;
	void			SetParentState(daeState* pParentState);
	
protected:
	virtual void	AddExpressionsToBlock(daeBlock* pBlock);

	void			InitializeStateTransitions(void);
	void			InitializeDEDIs(void);

	bool			CheckState(daeState* pState);
	void			BuildExpressions(daeBlock* pBlock);
	void			CreateEquationExecutionInfo(void);
	void			CollectEquationExecutionInfos(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo);
	void			CollectVariableIndexes(std::map<size_t, size_t>& mapVariableIndexes);
	void			SetIndexesWithinBlockToEquationExecutionInfos(daeBlock* pBlock, size_t& nEquationIndex);

	void			CalculateResiduals(void);
	void			CalculateJacobian(void);
	void			CalculateSensitivities(const std::vector<size_t>& narrParameterIndexes);
	void			CalculateGradients(const std::vector<size_t>& narrParameterIndexes);

	size_t			GetNumberOfEquationsInState(daeState* pState) const;

	daeState*		FindState(long nID);
	daeState*		FindState(const string& strName);

	void			ReconnectStateTransitionsAndStates(void);

protected:
	daeState*				m_pParentState;
	daePtrVector<daeState*>	m_ptrarrStates;
	daeState*				m_pActiveState;
	daeeSTNType				m_eSTNType;
	bool					m_bInitialized;
	friend class daeIF;
	friend class daeModel;
	friend class daeState;
	friend class daeBlock;
};

/******************************************************************
	daeIF
*******************************************************************/
class DAE_CORE_API daeIF : public daeSTN
{
public:
	daeDeclareDynamicClass(daeIF)
	daeIF(void);
	virtual ~daeIF(void);

public:	
	virtual void		Initialize(void);
	virtual bool		CheckDiscontinuities(void);
	virtual daeState*	AddState(string strName);

public:	
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	daeState* CreateElse(void);
	bool      CheckObject(std::vector<string>& strarrErrors) const;

protected:
	virtual void	AddExpressionsToBlock(daeBlock* pBlock);
	bool			CheckState(daeState* pState);
};

/******************************************************************
	daeEquation
*******************************************************************/
class DAE_CORE_API daeEquation : virtual public daeObject,
	                             virtual public daeEquation_t
{
public:
	daeDeclareDynamicClass(daeEquation)
	daeEquation(void);
	virtual ~daeEquation(void);

public:
	virtual void GetDomainDefinitions(std::vector<daeDistributedEquationDomainInfo_t*>& arrDistributedEquationDomainInfo);

public:	
	void	 SetResidual(adouble res);
	adouble	 GetResidual(void) const;
	daeDEDI* DistributeOnDomain(daeDomain& rDomain, daeeDomainBounds eDomainBounds);
	daeDEDI* DistributeOnDomain(daeDomain& rDomain, const std::vector<size_t>& narrDomainIndexes);

	daeeEquationDefinitionMode	GetEquationDefinitionMode(void) const;
	daeeEquationEvaluationMode	GetEquationEvaluationMode(void) const;
	void						SetEquationEvaluationMode(daeeEquationEvaluationMode eMode);

	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;
	bool CheckObject(std::vector<string>& strarrErrors) const;
	void InitializeDEDIs(void);

	virtual size_t	GetNumberOfEquations(void) const;
	
	void GetEquationExecutionInfos(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfos) const;

protected:
//  virtual adouble		Calculate(void);
//  virtual adouble		Calculate(size_t nDomain1);
//  virtual adouble		Calculate(size_t nDomain1, size_t nDomain2);
//  virtual adouble		Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3);
//  virtual adouble		Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4);
//  virtual adouble		Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5);
//  virtual adouble		Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6);
//  virtual adouble		Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7);
//  virtual adouble		Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7, size_t nDomain8);

	void GatherInfo(const std::vector<size_t>& narrDomainIndexes, const daeExecutionContext& EC, boost::shared_ptr<adNode>& node);
//	void Residual  (const std::vector<size_t>& narrDomainIndexes, const daeExecutionContext& EC);
//	void Jacobian  (const std::vector<size_t>& narrDomainIndexes, const std::map<size_t, size_t>& mapVariableIndexesInEquation, daeExecutionContext& EC);

	void SetResidualValue(size_t nEquationIndex, real_t dResidual, daeBlock* pBlock);
	void SetJacobianItem(size_t nEquationIndex, size_t nVariableIndex, real_t dJacobValue, daeBlock* pBlock);
	
	void SaveNodeAsMathML(io::xmlTag_t* pTag, const string& strObjectName) const;
	void SetModelAndCanonicalName(daeObject* pObject);

protected:
	daeeEquationDefinitionMode							m_eEquationDefinitionMode;
	daeeEquationEvaluationMode							m_eEquationEvaluationMode;
	boost::shared_ptr<adNode>							m_pResidualNode;
	daePtrVector<daeDistributedEquationDomainInfo*>		m_ptrarrDistributedEquationDomainInfos;
// This vector is redundant - all EquationExecutionInfos already exist in models and states
// However, it is useful when saving RuntimeReport
	std::vector<daeEquationExecutionInfo*>				m_ptrarrEquationExecutionInfos;
	
	friend class daeModel;
	friend class daeEquationExecutionInfo;
	friend class daeOptimizationConstraint;
	friend class daeObjectiveFunction;
};

/******************************************************************
	daePortEqualityEquation
*******************************************************************/
class DAE_CORE_API daePortEqualityEquation : public daeEquation
{
public:
	daeDeclareDynamicClass(daePortEqualityEquation)
	daePortEqualityEquation(void);
	virtual ~daePortEqualityEquation(void);

public:
	void Open(io::xmlTag_t* pTag);
	void Save(io::xmlTag_t* pTag) const;
	void OpenRuntime(io::xmlTag_t* pTag);
	void SaveRuntime(io::xmlTag_t* pTag) const;

	void	Initialize(daeVariable* pLeft, daeVariable* pRight);
	bool	CheckObject(std::vector<string>& strarrErrors) const;
	size_t	GetNumberOfEquations(void) const;

protected:
	daeVariable* m_pLeft;
	daeVariable* m_pRight;
};

/******************************************************************
	daeDistributedEquationDomainInfo
*******************************************************************/
class DAE_CORE_API daeSaveAsMathMLContext
{
public:
	daeSaveAsMathMLContext(const daeModel* pModel)
		: m_pModel(pModel)
	{
	}
	daeSaveAsMathMLContext(const daeModel* pModel, const std::vector<daeDEDI*>& ptrarrDEDIs)
		: m_pModel(pModel), m_ptrarrDEDIs(ptrarrDEDIs)
	{
	}
	const daeModel*			m_pModel;
	std::vector<daeDEDI*>	m_ptrarrDEDIs;
};


/******************************************************************
	daeOptimizationVariable
*******************************************************************/
class DAE_CORE_API daeOptimizationVariable : public daeOptimizationVariable_t,
		                                     public daeRuntimeCheck
{
public:
	daeDeclareDynamicClass(daeOptimizationVariable)
	daeOptimizationVariable(daeVariable* pVariable, size_t nOptimizationVariableIndex, const std::vector<size_t>& narrDomainIndexes, real_t LB, real_t UB, real_t defaultValue);
	daeOptimizationVariable(daeVariable* pVariable, size_t nOptimizationVariableIndex, const std::vector<size_t>& narrDomainIndexes, int LB, int UB, int defaultValue);
	daeOptimizationVariable(daeVariable* pVariable, size_t nOptimizationVariableIndex, const std::vector<size_t>& narrDomainIndexes, bool defaultValue);
	virtual ~daeOptimizationVariable(void);
	
public:
	std::string GetName(void) const;
	
	size_t GetOverallIndex(void) const;
	size_t GetOptimizationVariableIndex(void) const;
	
	void SetValue(real_t value);
	real_t GetValue(void) const;
	
	void                         SetType(daeeOptimizationVariableType value);
	daeeOptimizationVariableType GetType(void) const;

	void SetStartingPoint(real_t value);
	real_t GetStartingPoint(void) const;

	void SetLB(real_t value);
	real_t GetLB(void) const;

	void SetUB(real_t value);
	real_t GetUB(void) const;
	
	bool CheckObject(std::vector<string>& strarrErrors) const;
	
protected:
	daeVariable*					m_pVariable;
	real_t							m_dLB;
	real_t							m_dUB;
	real_t							m_dDefaultValue;
	daeeOptimizationVariableType	m_eType;
	size_t							m_nOptimizationVariableIndex; // index in the array of opt. variables: 0 - Nopt.vars. 
	std::vector<size_t>				m_narrDomainIndexes;
};

/******************************************************************
	daeObjectiveFunction
*******************************************************************/
class DAE_CORE_API daeObjectiveFunction : public daeObjectiveFunction_t,
		                                  public daeRuntimeCheck
{
public:
	daeDeclareDynamicClass(daeObjectiveFunction)
	daeObjectiveFunction(daeModel* pModel, real_t abstol);
	virtual ~daeObjectiveFunction(void);

public:
	bool CheckObject(std::vector<string>& strarrErrors) const;

	bool IsLinear(void) const;

	std::string GetName(void) const;
	real_t GetValue(void) const;
	void GetGradients(const daeMatrix<real_t>& matSensitivities, real_t* gradients, size_t Nparams) const;

	void   GetOptimizationVariableIndexes(std::vector<size_t>& narrOptimizationVariablesIndexes) const;

	void	 SetResidual(adouble res);
	adouble	 GetResidual(void) const;
	
	void Initialize(const std::vector< boost::shared_ptr<daeOptimizationVariable> >& arrOptimizationVariables, daeBlock_t* pBlock);
	
protected:
	daeModel*						m_pModel;
	boost::shared_ptr<daeVariable>	m_pObjectiveVariable;
	daeEquation*					m_pObjectiveFunction;
	size_t							m_nEquationIndexInBlock;
	size_t							m_nVariableIndexInBlock;
	std::vector<size_t>				m_narrOptimizationVariablesIndexes;
};

/******************************************************************
	daeOptimizationConstraint
*******************************************************************/
class DAE_CORE_API daeOptimizationConstraint : public daeOptimizationConstraint_t,
                                               public daeRuntimeCheck
{
public:
	daeDeclareDynamicClass(daeOptimizationConstraint)
	daeOptimizationConstraint(daeModel* pModel, real_t LB, real_t UB, real_t abstol, size_t N, string strDescription);
	daeOptimizationConstraint(daeModel* pModel, real_t Value, real_t abstol, size_t N, string strDescription);
	virtual ~daeOptimizationConstraint(void);

public:
	bool CheckObject(std::vector<string>& strarrErrors) const;

	void               SetType(daeeConstraintType value);
	daeeConstraintType GetType(void) const;
	
	bool IsLinear(void) const;
	
	void SetLB(real_t value);
	real_t GetLB(void) const;

	void SetUB(real_t value);
	real_t GetUB(void) const;
	
	void SetEqualityValue(real_t value);
	real_t GetEqualityValue(void) const;

	std::string GetName(void) const;
	real_t GetValue(void) const;
	void GetGradients(const daeMatrix<real_t>& matSensitivities, real_t* gradients, size_t Nparams) const;

	void   GetOptimizationVariableIndexes(std::vector<size_t>& narrOptimizationVariablesIndexes) const;

	void	 SetResidual(adouble res);
	adouble	 GetResidual(void) const;
	
	void Initialize(const std::vector< boost::shared_ptr<daeOptimizationVariable> >& arrOptimizationVariables, daeBlock_t* pBlock);

protected:
	daeeConstraintType				m_eConstraintType;
	real_t							m_dLB;
	real_t							m_dUB;
	daeModel*						m_pModel;
	size_t							m_nEquationIndexInBlock;
	size_t							m_nVariableIndexInBlock;
	boost::shared_ptr<daeVariable>	m_pConstraintVariable;
	daeEquation*					m_pConstraintFunction;
	std::vector<size_t>				m_narrOptimizationVariablesIndexes;
};

/******************************************************************
	daeCoreClassFactory
*******************************************************************/
typedef daeCreateObjectDelegate<daeVariableType>*		pfnVariableType;
typedef daeCreateObjectDelegate<daeParameter>*			pfnCreateParameter;
typedef daeCreateObjectDelegate<daeDomain>*				pfnCreateDomain;
typedef daeCreateObjectDelegate<daeVariable>*			pfnCreateVariable;
typedef daeCreateObjectDelegate<daeEquation>*			pfnCreateEquation;
typedef daeCreateObjectDelegate<daeSTN>*				pfnCreateSTN;
typedef daeCreateObjectDelegate<daeIF>*					pfnCreateIF;
typedef daeCreateObjectDelegate<daePort>*				pfnCreatePort;
typedef daeCreateObjectDelegate<daeState>*				pfnCreateState;
typedef daeCreateObjectDelegate<daeStateTransition>*	pfnCreateStateTransition;
typedef daeCreateObjectDelegate<daeModel>*				pfnCreateModel;
typedef daeCreateObjectDelegate<daePortConnection>*		pfnCreatePortConnection;

class DAE_CORE_API daeCoreClassFactory : public daeCoreClassFactory_t
{
public:
	daeCoreClassFactory(string strName, string strDescription, string strAuthorInfo, string strLicenceInfo, string strVersion);
	virtual ~daeCoreClassFactory(void);

public:
    string   GetName(void) const;
    string   GetDescription(void) const;
    string   GetAuthorInfo(void) const;
    string   GetLicenceInfo(void) const;
    string   GetVersion(void) const;

    daeVariableType_t*		CreateVariableType(const string& strClass);
	daePort_t*				CreatePort(const string& strClass);
	daeModel_t*				CreateModel(const string& strClass);
	
//	daeParameter_t*			CreateParameter(const string& strClass);
//	daeDomain_t*			CreateDomain(const string& strClass);
//	daeVariable_t*			CreateVariable(const string& strClass);
//	daeEquation_t*			CreateEquation(const string& strClass);
//	daeSTN_t*				CreateSTN(const string& strClass);
//	daeState_t*				CreateState(const string& strClass);
//	daeStateTransition_t*	CreateStateTransition(const string& strClass);
//	daePortConnection_t*	CreatePortConnection(const string& strClass);

	void SupportedVariableTypes(std::vector<string>& strarrClasses);
	void SupportedPorts(std::vector<string>& strarrClasses);
	void SupportedModels(std::vector<string>& strarrClasses);
	
//	void SupportedParameters(std::vector<string>& strarrClasses);
//	void SupportedDomains(std::vector<string>& strarrClasses);
//	void SupportedVariables(std::vector<string>& strarrClasses);
//	void SupportedEquations(std::vector<string>& strarrClasses);
//	void SupportedSTNs(std::vector<string>& strarrClasses);
//	void SupportedStates(std::vector<string>& strarrClasses);
//	void SupportedStateTransitions(std::vector<string>& strarrClasses);
//	void SupportedPortConnections(std::vector<string>& strarrClasses);

	bool RegisterVariableType(string strClass, pfnVariableType pfn);	
	bool RegisterPort(string strClass, pfnCreatePort pfn);
	bool RegisterModel(string strClass, pfnCreateModel pfn);
	
//	bool RegisterParameter(string strClass, pfnCreateParameter pfn);
//	bool RegisterDomain(string strClass, pfnCreateDomain pfn);
//	bool RegisterVariable(string strClass, pfnCreateVariable pfn);
//	bool RegisterEquation(string strClass, pfnCreateEquation pfn);
//	bool RegisterSTN(string strClass, pfnCreateSTN pfn);
//	bool RegisterIF(string strClass, pfnCreateIF pfn);
//	bool RegisterState(string strClass, pfnCreateState pfn);
//	bool RegisterStateTransition(string strClass, pfnCreateStateTransition pfn);
//	bool RegisterPortConnection(string strClass, pfnCreatePortConnection pfn);

public:
    string   m_strName;
    string   m_strDescription;
    string   m_strAuthorInfo;
    string   m_strLicenceInfo;
    string   m_strVersion;

	daePtrMap<string, pfnVariableType>				m_mapCreateVariableType;
	daePtrMap<string, pfnCreatePort>				m_mapCreatePort;
	daePtrMap<string, pfnCreateModel>				m_mapCreateModel;
	
//	daePtrMap<string, pfnCreateParameter>			m_mapCreateParameter;
//	daePtrMap<string, pfnCreateDomain>				m_mapCreateDomain;
//	daePtrMap<string, pfnCreateVariable>			m_mapCreateVariable;
//	daePtrMap<string, pfnCreateEquation>			m_mapCreateEquation;
//	daePtrMap<string, pfnCreateSTN>					m_mapCreateSTN;
//	daePtrMap<string, pfnCreateIF>					m_mapCreateIF;
//	daePtrMap<string, pfnCreateState>				m_mapCreateState;
//	daePtrMap<string, pfnCreateStateTransition>		m_mapCreateStateTransition;
//	daePtrMap<string, pfnCreatePortConnection>		m_mapCreatePortConnection;
};

#include "inlines_equation.h"
#include "inlines_modelarray.h"
#include "inlines_portarray.h"
#include "inlines_io.h"
#include "inlines_varparam_templates.h"
#include "inlines_varparam_array.h"

#include <algorithm>

template<class Object>
bool CheckName(const std::vector<Object*>& arrObjects, const std::string& strName)
{
	for(size_t i = 0; i < arrObjects.size(); i++)
		if(strName == arrObjects[i]->GetName())
			return true;
	return false;
}

}
}

#endif
