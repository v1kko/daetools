/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2015
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
#include <boost/variant.hpp>
#include <boost/tuple/tuple.hpp>

#include "../config.h"
#include "io_impl.h"
#include "helpers.h"
#include "call_stats.h"
#include "core.h"
#include "class_factory.h"
#include "adouble.h"
#include "export.h"
#include "thermo_package.h"

#if defined(DAE_MPI)
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#endif

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

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

enum daeeEvaluationMode
{
    eEvaluationTree_OpenMP = 0,
    eComputeStack_OpenMP,
    eComputeStack_External
};

/******************************************************************
    daeRuntimeCheck_t
*******************************************************************/
class daeRuntimeCheck_t
{
public:
    virtual ~daeRuntimeCheck_t(void){}

public:
    virtual bool CheckObject(std::vector<string>& strarrErrors) const = 0;
};

/******************************************************************
    daeVariableType
*******************************************************************/
class DAE_CORE_API daeVariableType : public daeVariableType_t,
                                     public io::daeSerializable,
                                     public daeRuntimeCheck_t,
                                     public daeExportable_t
{
public:
    daeDeclareDynamicClass(daeVariableType)
    daeVariableType(void);
    daeVariableType(string strName,
                    unit   units,
                    real_t dLowerBound,
                    real_t dUpperBound,
                    real_t dInitialGuess,
                    real_t dAbsoluteTolerance,
                    daeeVariableValueConstraint eValueConstraint = eNoConstraint);
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
    virtual unit	GetUnits(void) const;
    virtual void	SetUnits(const unit& u);
    virtual real_t	GetAbsoluteTolerance(void) const;
    virtual void	SetAbsoluteTolerance(real_t dTolerance);
    virtual daeeVariableValueConstraint	GetValueConstraint(void) const;
    virtual void	SetValueConstraint(daeeVariableValueConstraint eValueConstraint);

public:
    bool operator ==(const daeVariableType& other);
    bool operator !=(const daeVariableType& other);

    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

protected:
    string	m_strName;
    unit	m_Units;
    real_t	m_dLowerBound;
    real_t	m_dUpperBound;
    real_t	m_dInitialGuess;
    real_t	m_dAbsoluteTolerance;
    daeeVariableValueConstraint m_eValueConstraint;
    friend class daeModel;
    friend class daeVariable;
};

/******************************************************************
    daeObject
*******************************************************************/
class daeModel;
class DAE_CORE_API daeObject : virtual public daeObject_t,
                               virtual public io::daeSerializable,
                               public daeRuntimeCheck_t,
                               public daeExportable_t

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
    virtual void Open(io::xmlTag_t* pTag);
    virtual void Save(io::xmlTag_t* pTag) const;
    virtual void OpenRuntime(io::xmlTag_t* pTag);
    virtual void SaveRuntime(io::xmlTag_t* pTag) const;

    void Clone(const daeObject& rObject);

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    void SaveNameAsMathML(io::xmlTag_t* pTag, string strMathMLTag) const;
    void SaveRelativeNameAsMathML(io::xmlTag_t* pTag, string strMathMLTag, const daeObject* pParent = NULL) const;

    void SetName(const string& strName);
    void SetDescription(const string& strDescription);
    void SetModel(daeModel* pModel);

    string GetNameRelativeToParentModel(void) const;
    string GetStrippedName(void) const;
    string GetStrippedNameRelativeToParentModel(void) const;
    virtual string GetCanonicalNameAndPrepend(const std::string& prependToName) const;

protected:
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
    friend class daeOnConditionActions;
    friend class daeModel;
    friend class daePortArray;
    friend class daeModelArray;
    friend class daeDistributedEquationDomainInfo;
    friend class daeOnEventActions;
    friend class daeEventPortConnection;
    friend class daePortConnection;
};

bool   daeIsValidObjectName(const string& strName);
string daeGetRelativeName(const daeObject* parent, const daeObject* child);
string daeGetRelativeName(const string& strParent, const string& strChild);
string daeGetStrippedName(const string& strName);
string daeGetStrippedRelativeName(const daeObject* parent, const daeObject* child);

/******************************************************************
    daePartialDerivativeVariable
*******************************************************************/
class daeDomain;
class daeVariable;
class DAE_CORE_API daePartialDerivativeVariable
{
public:
    daePartialDerivativeVariable(const size_t                               nOrder,
                                 const daeVariable&                         rVariable,
                                 const daeDomain&                           rDomain,
                                 const size_t                               nDomainIndex,
                                 const size_t                               nNoIndexes,
                                 const size_t*                              pIndexes,
                                 const daeeDiscretizationMethod             eDiscretizationMethod,
                                 const std::map<std::string, std::string>&  mapDiscretizationOptions);
    virtual ~daePartialDerivativeVariable(void);

public:
    adouble				operator()(size_t nIndex);
    adouble				operator[](size_t nIndex);
    size_t				GetOrder(void) const;
    size_t				GetPoint(void) const;
    const daeDomain&	GetDomain(void) const;
    const daeVariable&	GetVariable(void) const;

    adouble             CreateSetupNode(void);

public:
    size_t*                         m_pIndexes;
    const size_t                    m_nOrder;
    const daeVariable&              m_rVariable;
    const daeDomain&                m_rDomain;
    const size_t                    m_nDomainIndex;
    const size_t                    m_nNoIndexes;

    const daeeDiscretizationMethod           m_eDiscretizationMethod;
    const std::map<std::string, std::string> m_mapDiscretizationOptions;
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
class daeArrayRange;
class DAE_CORE_API daeDomain : virtual public daeObject,
                               virtual public daeDomain_t
{
public:
    daeDeclareDynamicClass(daeDomain)
    daeDomain(void);
    daeDomain(string strName, daeModel* pModel, const unit& units, string strDescription = "");
    daeDomain(string strName, daePort* pPort, const unit& units, string strDescription = "");
    virtual ~daeDomain(void);

// Public interface
public:
    virtual string GetCanonicalName(void) const;

// Common for both Discrete and Distributed domains
    virtual daeeDomainType				GetType(void) const;
    virtual size_t						GetNumberOfIntervals(void) const;
    virtual size_t						GetNumberOfPoints(void) const;
    virtual const real_t*				GetPoint(size_t nIndex) const;
    virtual unit						GetUnits(void) const;

// Only for Distributed domains
    //virtual daeeDiscretizationMethod	GetDiscretizationMethod(void) const;
    //virtual size_t					GetDiscretizationOrder(void) const;
    virtual real_t						GetLowerBound(void) const;
    virtual real_t						GetUpperBound(void) const;

    virtual void						GetPoints(std::vector<real_t>& darrPoints) const;
    virtual void						SetPoints(const std::vector<real_t>& darrPoints);

    const std::vector<daePoint>& GetCoordinates() const;

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    void Clone(const daeDomain& rObject);

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    adouble_array array(void);
    adouble_array array(const daeIndexRange& indexRange);
    adouble_array array(const daeArrayRange& arrayRange);

    adouble	operator[](size_t nIndex) const;
    adouble	operator()(size_t nIndex) const;

    void CreateArray(size_t nNoPoints);
    void CreateStructuredGrid(size_t nNoIntervals, real_t dLB, real_t dUB);
    void CreateStructuredGrid(size_t nNoIntervals, quantity qLB, quantity qUB);
    void CreateUnstructuredGrid(const std::vector<daePoint>& coordinates);

    daePort* GetParentPort(void) const;

protected:
    void	CreatePoints(void);
    void	SetType(daeeDomainType eDomainType);
    void	SetUnits(const unit& units);

protected:
    unit							m_Unit;
    daeeDomainType					m_eDomainType;
    size_t							m_nNumberOfPoints;
    std::vector<real_t>				m_darrPoints;
    daePort*						m_pParentPort;

    // StructuredGrid:
    real_t							m_dLowerBound;
    real_t							m_dUpperBound;
    size_t							m_nNumberOfIntervals;

    // UnstructuredGrid:
    std::vector<daePoint> m_arrCoordinates;

    friend class daePort;
    friend class daeModel;
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
    daeEquationExecutionInfo(daeEquation* pEquation);
    virtual ~daeEquationExecutionInfo(void);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

public:
    virtual void GatherInfo(daeExecutionContext& EC, daeModel* pModel);
    virtual void Residual(daeExecutionContext& EC);
    virtual void Jacobian(daeExecutionContext& EC);
    virtual void SensitivityResiduals(daeExecutionContext& EC, const std::vector<size_t>& narrParameterIndexes);
    virtual void SensitivityParametersGradients(daeExecutionContext& EC, const std::vector<size_t>& narrParameterIndexes);

    void AddVariableInEquation(size_t nIndex);
    void GetVariableIndexes(std::vector<size_t>& narrVariableIndexes) const;

    size_t GetEquationIndexInBlock(void) const;
    adNodePtr GetEquationEvaluationNode(void) const;

    adNode* GetEquationEvaluationNodeRawPtr(void) const;

    daeeEquationType GetEquationType(void) const;
    daeEquation* GetEquation(void) const;

    std::string GetName(void) const;

    const std::map< size_t, std::pair<size_t, adNodePtr> >& GetJacobianExpressions() const;
    std::pair<size_t,size_t>          GetComputeStackInfo(const std::map<daeeUnaryFunctions, size_t> &unaryOps,
                                                          const std::map<daeeBinaryFunctions, size_t> &binaryOps) const;
    std::vector<csComputeStackItem_t> GetComputeStack() const;
    uint8_t GetComputeStack_max_valueSize() const;
    uint8_t GetComputeStack_max_lvalueSize() const;
    uint8_t GetComputeStack_max_rvalueSize() const;

protected:
    void BuildJacobianExpressions();
    void CreateComputeStack(daeBlock* pBlock);

protected:
    real_t                      m_dScaling;
    size_t                      m_nEquationIndexInBlock;
    std::vector<size_t>         m_narrDomainIndexes;
    std::map<size_t, size_t>    m_mapIndexes;
    daeEquation*                m_pEquation;
    adNodePtr                   m_EquationEvaluationNode;
    std::map< size_t, std::pair<size_t, adNodePtr> > m_mapJacobianExpressions;

    uint32_t                            m_nComputeStackIndex;
    //std::vector<csComputeStackItem_t>   m_EquationComputeStack;
    uint8_t                             m_nComputeStack_max_valueSize;
    uint8_t                             m_nComputeStack_max_lvalueSize;
    uint8_t                             m_nComputeStack_max_rvalueSize;

    friend class daeEquation;
    friend class daeFiniteElementEquation;
    friend class daeSTN;
    friend class daeIF;
    friend class daeModel;
    friend class daeFiniteElementModel;
    friend class daeState;
    friend class daeBlock;
};

/*********************************************************************************************
    daeDomainIndex
**********************************************************************************************/
class daeDistributedEquationDomainInfo;
class DAE_CORE_API daeDomainIndex
{
public:
    daeDomainIndex(void);
    daeDomainIndex(size_t nIndex);
    daeDomainIndex(daeDomain* pLastPointInDomain, int i);
    daeDomainIndex(daeDistributedEquationDomainInfo* pDEDI);
    daeDomainIndex(daeDistributedEquationDomainInfo* pDEDI, int iIncrement);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;

    size_t GetCurrentIndex(void) const;
    string GetIndexAsString(void) const;

public:
    daeeDomainIndexType					m_eType;
    size_t								m_nIndex;
    daeDistributedEquationDomainInfo*	m_pDEDI;
    int									m_iIncrement;
    daeDomain*                          m_pLastPointInDomain;
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

    void Clone(const daeDistributedEquationDomainInfo& rObject);
    bool CheckObject(std::vector<string>& strarrErrors) const;

    daeEquation*	GetEquation(void) const;
    size_t			GetCurrentIndex(void) const;
    adouble			operator()(void) const;
    void			Initialize(void);

    daeDomainIndex	operator+(size_t increment) const;
    daeDomainIndex	operator-(size_t increment) const;

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
    friend class daeFiniteElementEquation;
    friend class daeParameter;
    friend class daeVariable;
    friend class daeEquationExecutionInfo;
};
typedef daeDistributedEquationDomainInfo daeDEDI;

/*********************************************************************************************
    daeArrayRange
**********************************************************************************************/
class DAE_CORE_API daeArrayRange
{
public:
    daeArrayRange(void);
    daeArrayRange(daeDomainIndex domainIndex);
    daeArrayRange(daeIndexRange range);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;

    size_t GetNoPoints(void) const;
    void   GetPoints(std::vector<size_t>& narrPoints) const;
    string GetRangeAsString(void) const;

public:
    daeeRangeType	m_eType;
    daeIndexRange	m_Range;
    daeDomainIndex	m_domainIndex;
};

/******************************************************************
    daeExpressionInfo
*******************************************************************/
class daeOnConditionActions;
class daeExpressionInfo
{
public:
    daeExpressionInfo(void)
    {
        m_pOnConditionActions = NULL;
    }

    adNodePtr	           m_pExpression;
    daeOnConditionActions* m_pOnConditionActions;
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
        m_pdInitialValues			= NULL;
        m_pdInitialConditions		= NULL;
        m_pdVariablesTypes			= NULL;
        m_pdVariablesTypesGathered	= NULL;
        m_pdAbsoluteTolerances		= NULL;
        m_pdVariableValuesConstraints = NULL;
        m_pTopLevelModel			= NULL;
        m_pLog						= NULL;
        m_pBlock					= NULL;
        m_bGatherInfo				= false;
        m_nTotalNumberOfVariables	= 0;
        m_nNumberOfParameters		= 0;
        m_pmatSValues				= NULL;
        m_pmatSTimeDerivatives		= NULL;
        m_pmatSResiduals			= NULL;
        m_dCurrentTime				= 0;
        m_bReinitializationFlag		= false;
        m_bIsModelDynamic			= false;
        m_bCopyDataFromBlock        = false;
        m_pLastSatisfiedCondition   = NULL;

        daeConfig& cfg = daeConfig::GetConfig();
        m_bResetLAMatrixAfterDiscontinuity = cfg.GetBoolean("daetools.core.resetLAMatrixAfterDiscontinuity", true);
        m_bPrintInfo                       = cfg.GetBoolean("daetools.core.printInfo", false);
        m_bCheckForInfinite                = cfg.GetBoolean("daetools.core.checkForInfiniteNumbers", false);

        std::string evaluationMode = cfg.GetString("daetools.core.equations.evaluationMode", "evaluationTree_OpenMP");
        if(evaluationMode == "computeStack_OpenMP")
            m_eEvaluationMode = eComputeStack_OpenMP;
        else if(evaluationMode == "evaluationTree_OpenMP")
            m_eEvaluationMode = eEvaluationTree_OpenMP;
        else
            m_eEvaluationMode = eComputeStack_External;
    }


    virtual ~daeDataProxy_t(void)
    {
        if(m_pdInitialValues)
        {
            delete[] m_pdInitialValues;
            m_pdInitialValues = NULL;
        }
        if(m_pdInitialConditions)
        {
            delete[] m_pdInitialConditions;
            m_pdInitialConditions = NULL;
        }
        if(m_pdVariablesTypes)
        {
            delete[] m_pdVariablesTypes;
            m_pdVariablesTypes = NULL;
        }
        if(m_pdVariablesTypesGathered)
        {
            delete[] m_pdVariablesTypesGathered;
            m_pdVariablesTypesGathered = NULL;
        }
        if(m_pdAbsoluteTolerances)
        {
            delete[] m_pdAbsoluteTolerances;
            m_pdAbsoluteTolerances = NULL;
        }
        if(m_pdVariableValuesConstraints)
        {
            delete[] m_pdVariableValuesConstraints;
            m_pdVariableValuesConstraints = NULL;
        }
//		if(m_pCondition)
//		{
//			delete m_pCondition;
//			m_pCondition = NULL;
//		}
    }

    void Initialize(daeModel_t* pTopLevelModel, daeLog_t* pLog, size_t nTotalNumberOfVariables)
    {
        m_nTotalNumberOfVariables = nTotalNumberOfVariables;

        m_pdInitialValues = new real_t[m_nTotalNumberOfVariables];
        memset(m_pdInitialValues, 0, m_nTotalNumberOfVariables * sizeof(real_t));

        m_pdInitialConditions	= new real_t[m_nTotalNumberOfVariables];
        memset(m_pdInitialConditions, 0, m_nTotalNumberOfVariables * sizeof(real_t));

        m_pdVariablesTypes			= new real_t[m_nTotalNumberOfVariables];
        m_pdVariablesTypesGathered	= new real_t[m_nTotalNumberOfVariables];
        memset(m_pdVariablesTypes,         cnAlgebraic, m_nTotalNumberOfVariables * sizeof(real_t));
        memset(m_pdVariablesTypesGathered, cnAlgebraic, m_nTotalNumberOfVariables * sizeof(real_t));

        m_pdAbsoluteTolerances	= new real_t[m_nTotalNumberOfVariables];
        memset(m_pdAbsoluteTolerances, 0, m_nTotalNumberOfVariables * sizeof(real_t));

        m_pdVariableValuesConstraints	= new real_t[m_nTotalNumberOfVariables];
        memset(m_pdVariableValuesConstraints, 0, m_nTotalNumberOfVariables * sizeof(real_t));

        m_pTopLevelModel = pTopLevelModel;
        m_pLog           = pLog;

        m_bGatherInfo	 = false;
    }

    void Load(const std::string& strFileName)
    {
        double dValue;
        boost::uint32_t nTotalNumberOfVariables;
        size_t counter;
        std::streamoff lRequiredFileSize;
        std::streamoff lFileSize;
        std::ifstream file;

        if(m_nTotalNumberOfVariables == 0 || m_pdarrValuesReferences.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot load initialization file: " << strFileName << ": simulation has not been initialized";
            throw e;
        }
        if(m_nTotalNumberOfVariables != m_pdarrValuesReferences.size())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot load initialization file: " << strFileName << ": invalid number of variables";
            throw e;
        }

        try
        {
            file.open(strFileName.c_str(), std::ios_base::in | std::ios_base::binary);
            file.seekg(0, std::ifstream::end);
            lFileSize = file.tellg();
            file.seekg(0);
            file.read((char*)(&nTotalNumberOfVariables), sizeof(boost::uint32_t));
        }
        catch(std::exception& exc)
        {
            daeDeclareException(exInvalidCall);
            e << "An error occured while loading the initialization file: " <<
                 strFileName << "; " << exc.what();
            throw e;
        }

        lRequiredFileSize = sizeof(boost::uint32_t) + m_nTotalNumberOfVariables * sizeof(double);
        if(lFileSize != lRequiredFileSize)
        {
            daeDeclareException(exInvalidCall);
            e << string("The file size of the initialization file ") <<
                 strFileName + string(" does not match; required: ") <<
                 toString<std::streamoff>(lRequiredFileSize) + string(", but available: ") <<
                 toString<std::streamoff>(lFileSize);
            throw e;
        }

        if(m_nTotalNumberOfVariables != nTotalNumberOfVariables)
        {
            daeDeclareException(exInvalidCall);
            e << string("The number of variables in the initialization file ") <<
                 strFileName + string(": ") << toString<size_t>(nTotalNumberOfVariables) <<
                 string(" does not match the number of variables in the simulation: ") <<
                 toString<size_t>(m_nTotalNumberOfVariables);
            throw e;
        }

        try
        {
            counter = 0;
            while(file.good() && counter < m_nTotalNumberOfVariables)
            {
                file.read((char*)(&dValue), sizeof(double));
                *(m_pdarrValuesReferences[counter]) = static_cast<real_t>(dValue);
                counter++;
            }
            file.close();
        }
        catch(std::exception& exc)
        {
            daeDeclareException(exInvalidCall);
            e << "An error occured while loading the initialization file: " <<
                 strFileName << "; " << exc.what();
            throw e;
        }

        if(counter < m_nTotalNumberOfVariables)
        {
            daeDeclareException(exInvalidCall);
            e << string("The initialization file does not contain: ") <<
                 toString<size_t>(m_nTotalNumberOfVariables) <<
                 string(" variables; found: ") << toString<size_t>(counter);
            throw e;
        }
    }

    void Store(const std::string& strFileName) const
    {
        double dValue;
        std::ofstream file;
        boost::uint32_t nTotalNumberOfVariables;

        if(m_nTotalNumberOfVariables == 0 || m_pdarrValuesReferences.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot store initialization file: " << strFileName << ": simulation has not been initialized";
            throw e;
        }
        if(m_nTotalNumberOfVariables != m_pdarrValuesReferences.size())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot store initialization file: " << strFileName << ": invalid number of variables";
            throw e;
        }

        try
        {
            file.open(strFileName.c_str(), std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

            nTotalNumberOfVariables = static_cast<boost::uint32_t>(m_nTotalNumberOfVariables);
            file.write((char*)(&nTotalNumberOfVariables), sizeof(boost::uint32_t));

            for(size_t i = 0; i < m_nTotalNumberOfVariables; i++)
            {
                dValue = *(m_pdarrValuesReferences[i]);
                file.write((char*)(&dValue), sizeof(double));
            }

            file.flush();
            file.close();
        }
        catch(std::exception& exc)
        {
            daeDeclareException(exInvalidCall);
            e << "An error occured while storing the initialization file: " <<
                 strFileName << "; " << exc.what();
            throw e;
        }
    }

    size_t GetTotalNumberOfVariables(void) const
    {
        return m_nTotalNumberOfVariables;
    }

/*
  ACHTUNG!! Normally, the initialization phase functions only!
  Used only to set initial guesses/initial conditions/abs. tolerances/assigned values during the initialization phase.
  The following arrays MIGHT be deleted after a successful SolveInitial() call:
    - m_pdInitialValues
    - m_pdInitialConditions
    - m_pdVariablesTypesGathered
    - m_pdAbsoluteTolerances

  However, some of the above functions may be called even after the Initialization. For instance, during an optimization
  the function SetUpVariables is called before each iteration which may set initial guesses, absolute
  tolerances, initial conditions and assign the varoable values. Therefore, these functions must be safe
  from exceptions if called
*/
    void SetInitialGuess(size_t nOverallIndex, real_t Value)
    {
    /*
      First check if m_pdarrValuesReferences has already been initialized.
      If so, use it. If not use m_pdInitialValues.
    */
        if(!m_pdarrValuesReferences.empty())
        {
            *(m_pdarrValuesReferences[nOverallIndex]) = Value;
        }
        else if(m_pdInitialValues)
        {
            m_pdInitialValues[nOverallIndex] = Value;
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot set the initial guess";
            throw e;
        }
    }

    void AssignValue(size_t nOverallIndex, real_t Value)
    {
        std::map<size_t, size_t>::iterator it = m_mapAssignedVarsIndexes.find(nOverallIndex);

        if(it == m_mapAssignedVarsIndexes.end())
        {
            // The overall index does not exist in the map and should be inserted into the map and the vector.
            // In this case, the local index is equal to current vector size.
            size_t dofIndex = m_arrAssignedVarsValues.size();

            m_mapAssignedVarsIndexes[nOverallIndex] = dofIndex;
            m_arrAssignedVarsValues.push_back(Value);
        }
        else
        {
            // The overall index exists in the map.
            // The local index is already in the map.
            size_t dofIndex = it->second;
            m_arrAssignedVarsValues[dofIndex] = Value;
        }

        // This should not happen, but just in case something went wrong.
        if(m_mapAssignedVarsIndexes.size() != m_arrAssignedVarsValues.size())
        {
            daeDeclareException(exInvalidCall);
            e << "The size of m_mapAssignedVarsIndexes map and  m_mapAssignedVarsIndexes vector does not mach";
            throw e;
        }
    }

    void SetInitialCondition(size_t nOverallIndex, real_t Value, daeeInitialConditionMode eMode)
    {
    /*
      Here we have to be able to set initial conditions repeatedly (during optimization).
      However, after the first solve_initial the initialization data might be deleted.
      Therefore, the first thing that has to be done is checking if m_pdarrValuesReferences
      has already been initialized. If so, use it. If not use m_pdInitialValues.
    */
        if(!m_pdarrValuesReferences.empty())
        {
            if(eMode == eAlgebraicValuesProvided)
            {
                *(m_pdarrValuesReferences[nOverallIndex]) = Value;
            }
            else
            {
                daeDeclareAndThrowException(exInvalidCall);
            }
        }
        else if(m_pdInitialValues)
        {
            if(eMode == eAlgebraicValuesProvided)
            {
                m_pdInitialValues[nOverallIndex] = Value;
            }
            else
            {
                daeDeclareAndThrowException(exInvalidCall);
            }
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot set the initial condition";
            throw e;
        }
    }

    int GetVariableType(size_t nOverallIndex) const
    {
        if(!m_pdVariablesTypes)
            daeDeclareAndThrowException(exInvalidPointer);
        return static_cast<int>(m_pdVariablesTypes[nOverallIndex]);
    }

    void SetVariableType(size_t nOverallIndex, int Value)
    {
        if(!m_pdVariablesTypes)
            daeDeclareAndThrowException(exInvalidPointer);
        m_pdVariablesTypes[nOverallIndex] = static_cast<real_t>(Value);
    }

    int GetVariableTypeGathered(size_t nOverallIndex) const
    {
        if(!m_pdVariablesTypesGathered)
            daeDeclareAndThrowException(exInvalidPointer);
        return static_cast<int>(m_pdVariablesTypesGathered[nOverallIndex]);
    }

    void SetVariableTypeGathered(size_t nOverallIndex, int Value)
    {
        if(!m_pdVariablesTypesGathered)
            daeDeclareAndThrowException(exInvalidPointer);
        if(Value == cnDifferential)
            m_bIsModelDynamic = true;
        m_pdVariablesTypesGathered[nOverallIndex] = static_cast<real_t>(Value);
    }

    real_t GetAbsoluteTolerance(size_t nOverallIndex) const
    {
        if(!m_pdAbsoluteTolerances)
            daeDeclareAndThrowException(exInvalidPointer);
        return m_pdAbsoluteTolerances[nOverallIndex];
    }

    void SetAbsoluteTolerance(size_t nOverallIndex, real_t Value)
    {
    // If called repeatedly during optimization it has no effect
        if(m_pdAbsoluteTolerances)
            m_pdAbsoluteTolerances[nOverallIndex] = Value;
    }

    daeeVariableValueConstraint GetValueConstraint(size_t nOverallIndex) const
    {
        if(!m_pdVariableValuesConstraints)
            daeDeclareAndThrowException(exInvalidPointer);
        int constraint = (int)m_pdVariableValuesConstraints[nOverallIndex];
        return (daeeVariableValueConstraint)constraint;
    }

    void SetValueConstraint(size_t nOverallIndex, daeeVariableValueConstraint constraint)
    {
    // If called repeatedly during optimization it has no effect
        if(m_pdVariableValuesConstraints)
            m_pdVariableValuesConstraints[nOverallIndex] = (real_t)constraint;
    }

    real_t* GetInitialValuesPointer(void) const
    {
        return m_pdInitialValues;
    }

    real_t* GetInitialConditionsPointer(void) const
    {
        return m_pdInitialConditions;
    }

    real_t* GetVariableTypesGatheredPointer(void) const
    {
    // Filled during DeclareEquations function
    // VariablesTypesGathered does not contain information about assigned variables!!
    // It contains only 0 and 1
        return m_pdVariablesTypesGathered;
    }

    real_t* GetVariableTypesPointer(void) const
    {
    // Filled during SetUpVariables function
    // VariablesTypes contains information about assigned variables!!
    // It contains 0, 1 and 2
        return m_pdVariablesTypes;
    }

    real_t* GetAbsoluteTolerancesPointer(void) const
    {
        return m_pdAbsoluteTolerances;
    }

    real_t* GetVariableValuesConstraintsPointer(void) const
    {
        return m_pdVariableValuesConstraints;
    }
/* End of initialization phase functions */

/*
  Functions that can be used ONLY when changing some value; for instance:
    - During integration in an operating procedure to re-assign or reinitialize some variable
    - In daeActions after an event to re-assign or reinitialize some variable
    - During optimization, before each iteration new values of opt. variables has to be set
  ACHTUNG!! After any of these uses the solver has to be reinitialized with the new values!!

  Values/TimeDerivatives references point to:
    - DAESolver's data (for state variables)
    - Internal map of assigned variables (for assigned variables)

  ACHTUNG!! Get/Set Value/TimeDerivative must not be used in adNode derived classes to calculate residualS/jacobian/...
            They MUST access the data through the functions in daeBlock (using block indexes).

  NOTE: After each successful call to IDASolver (any daeSimulation::Integrate_XXX function) the new Values and
  TimeDerivatives are set (also the new values are set before any check for discontinuities).
*/
    real_t GetValue(size_t nOverallIndex) const
    {
        if(m_pdarrValuesReferences.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot get the variable value for the system has not been initialized yet";
            throw e;
        }
        return *(m_pdarrValuesReferences[nOverallIndex]);
    }

    void SetValue(size_t nOverallIndex, real_t Value)
    {
        if(m_pdarrValuesReferences.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot set the variable value for the system has not been initialized yet";
            throw e;
        }
        *(m_pdarrValuesReferences[nOverallIndex]) = Value;
    }

    real_t GetTimeDerivative(size_t nOverallIndex) const
    {
        if(m_pdarrTimeDerivativesReferences.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot get the time derivative for the system has not been initialized yet";
            throw e;
        }
        // Assigned variables do not have time derivatives indexes mapped!!
        if(m_pdarrTimeDerivativesReferences[nOverallIndex] == NULL)
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot get the time derivative for the assigned variable";
            throw e;
        }
        return *(m_pdarrTimeDerivativesReferences[nOverallIndex]);
    }

    void SetTimeDerivative(size_t nOverallIndex, real_t Value)
    {
        if(m_pdarrTimeDerivativesReferences.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot set the time derivative for the system has not been initialized yet";
            throw e;
        }
        *(m_pdarrTimeDerivativesReferences[nOverallIndex]) = Value;
    }

    // This function is only used for reporting time derivatives in ReportData(time).
    // It returns 0.0 for non-differential variables (does not throw exceptions).
    real_t GetTimeDerivative_or_zero(size_t nOverallIndex) const
    {
        if(m_pdarrTimeDerivativesReferences.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot get the time derivative for the system has not been initialized yet";
            throw e;
        }

        // Assigned variables do not have time derivatives indexes mapped so return zero
        if(m_pdarrTimeDerivativesReferences[nOverallIndex] == NULL)
            return real_t(0);
        else
            return *(m_pdarrTimeDerivativesReferences[nOverallIndex]);
    }

    void ReSetInitialCondition(size_t nOverallIndex, real_t Value, daeeInitialConditionMode eMode)
    {
        if(m_pdarrValuesReferences.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot re-set the initial condition for the system has not been initialized yet";
            throw e;
        }
        *(m_pdarrValuesReferences[nOverallIndex]) = Value;

        SetReinitializationFlag(true);
        SetCopyDataFromBlock(true);
    }

    void ReAssignValue(size_t nOverallIndex, real_t Value)
    {
        std::map<size_t, size_t>::iterator it = m_mapAssignedVarsIndexes.find(nOverallIndex);

        if(it == m_mapAssignedVarsIndexes.end())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot re-assign the variable value - the overall index is not in the map";
            throw e;
        }

        size_t dofIndex = it->second;
        m_arrAssignedVarsValues[dofIndex] = Value;

        SetReinitializationFlag(true);
    }

    //real_t GetReAssignedValue(size_t nOverallIndex) const
    //{
    //    return m_mapAssignedValues.find(nOverallIndex)->second;
    //}

// S value: S[nParameterIndex][nVariableIndex]
    real_t GetSValue(size_t nParameterIndex, size_t nVariableIndex) const
    {
#ifdef DAE_DEBUG
        if(!m_pmatSValues)
            daeDeclareAndThrowException(exInvalidPointer);

        if(nParameterIndex >= m_pmatSValues->GetNrows() ||
           nVariableIndex  >= m_pmatSValues->GetNcols()  )
            daeDeclareAndThrowException(exOutOfBounds);
#endif
        return m_pmatSValues->GetItem(nParameterIndex, nVariableIndex);
    }

// SD value: SD[nParameterIndex][nVariableIndex]
    real_t GetSDValue(size_t nParameterIndex, size_t nVariableIndex) const
    {
#ifdef DAE_DEBUG
        if(!m_pmatSTimeDerivatives)
            daeDeclareAndThrowException(exInvalidPointer);

        if(nParameterIndex >= m_pmatSTimeDerivatives->GetNrows() ||
           nVariableIndex  >= m_pmatSTimeDerivatives->GetNcols()  )
            daeDeclareAndThrowException(exOutOfBounds);
#endif
        return m_pmatSTimeDerivatives->GetItem(nParameterIndex, nVariableIndex);
    }

// Sresidual value: SRes[nParameterIndex][nEquationIndex]
    void SetSResValue(size_t nParameterIndex, size_t nEquationIndex, real_t value)
    {
#ifdef DAE_DEBUG
        if(!m_pmatSResiduals)
            daeDeclareAndThrowException(exInvalidPointer);

        if(nParameterIndex >= m_pmatSResiduals->GetNrows() ||
           nEquationIndex  >= m_pmatSResiduals->GetNcols()  )
            daeDeclareAndThrowException(exOutOfBounds);
#endif
        return m_pmatSResiduals->SetItem(nParameterIndex, nEquationIndex, value);
    }
/* End of integration phase functions */

    const std::vector<real_t*>& GetValuesReferences(void) const
    {
        return m_pdarrValuesReferences;
    }

    const std::vector<real_t*>& GetTimeDerivativesReferences(void) const
    {
        return m_pdarrTimeDerivativesReferences;
    }

    const std::map<size_t, size_t>& GetAssignedVarsIndexes(void) const
    {
        return m_mapAssignedVarsIndexes;
    }

    const std::vector<real_t>& GetAssignedVarsValues(void) const
    {
        return m_arrAssignedVarsValues;
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

    real_t GetCurrentTime_(void) const
    {
        return m_dCurrentTime;
    }

    void SetCurrentTime(real_t time)
    {
        m_dCurrentTime = time;
    }

    bool GetReinitializationFlag(void) const
    {
        return m_bReinitializationFlag;
    }

    void SetReinitializationFlag(bool bReinitializationFlag)
    {
        m_bReinitializationFlag = bReinitializationFlag;
    }

/*
  The mappings sets the pointers to actual data so there is no need to copy the data back and forth from/to the DAE solver.

  ACHTUNG #1:
     These pointers cannot be used for calculating residuals/jacobian/sensitivities. Therefore, adNode_XXX classes
     MUST access the data through functions from daeBlock (using block indexes).

  ACHTUNG #2:
     Some of time derivatives remain unset (those of assigned variables, for instance) and their pointers are NULL!!
*/
    void CreateIndexMappings(const std::map<size_t, size_t>& mapVariableIndexes, real_t* pdValues, real_t* pdTimeDerivatives)
    {
        m_pdarrValuesReferences.resize(m_nTotalNumberOfVariables, NULL);
        m_pdarrTimeDerivativesReferences.resize(m_nTotalNumberOfVariables, NULL);

    // mapVariableIndexes<nOverallIndex, nBlockIndex>
        for(std::map<size_t, size_t>::const_iterator iter = mapVariableIndexes.begin(); iter != mapVariableIndexes.end(); iter++)
        {
        // Achtung: They must not be previously set!
            if(m_pdarrValuesReferences[iter->first] || m_pdarrTimeDerivativesReferences[iter->first])
                daeDeclareAndThrowException(exInvalidCall);

            m_pdarrValuesReferences[iter->first]          = &pdValues[iter->second];
            m_pdarrTimeDerivativesReferences[iter->first] = &pdTimeDerivatives[iter->second];
        }

    // m_mapAssignedVarsIndexes<overallIndex, dofIndex>
    // dofIndex is an index in the m_arrAssignedVarsValues array.
        for(std::map<size_t, size_t>::iterator iter = m_mapAssignedVarsIndexes.begin(); iter != m_mapAssignedVarsIndexes.end(); iter++)
        {
        // Achtung: They must not be previously set!
            if(m_pdarrValuesReferences[iter->first])
                daeDeclareAndThrowException(exInvalidCall);

            size_t overallIndex = iter->first;
            size_t dofIndex     = iter->second;
            m_pdarrValuesReferences[overallIndex] = &m_arrAssignedVarsValues[dofIndex];
        }
    }

    void CleanUpSetupData(void)
    {
    /*	Achtung, Achtung!!
        m_pdVariablesTypes should not be deleted for it might be used during an integration.
    */
        if(m_pdInitialValues)
        {
            delete[] m_pdInitialValues;
            m_pdInitialValues = NULL;
        }
        if(m_pdInitialConditions)
        {
            delete[] m_pdInitialConditions;
            m_pdInitialConditions = NULL;
        }
        if(m_pdVariablesTypesGathered)
        {
            delete[] m_pdVariablesTypesGathered;
            m_pdVariablesTypesGathered = NULL;
        }
        if(m_pdAbsoluteTolerances)
        {
            delete[] m_pdAbsoluteTolerances;
            m_pdAbsoluteTolerances = NULL;
        }
        if(m_pdVariableValuesConstraints)
        {
            delete[] m_pdVariableValuesConstraints;
            m_pdVariableValuesConstraints = NULL;
        }

        m_pTopLevelModel->CleanUpSetupData();
    }

//	void SetGlobalCondition(daeCondition condition)
//	{
//		daeExecutionContext EC;
//		EC.m_pDataProxy					= this;
//		EC.m_pBlock						= m_pBlock;
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

    bool GetCopyDataFromBlock(void) const
    {
        return m_bCopyDataFromBlock;
    }

    void SetCopyDataFromBlock(bool bCopyDataFromBlock)
    {
        m_bCopyDataFromBlock = bCopyDataFromBlock;
    }

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

    daeMatrix<real_t>* GetSTimeDerivatives()
    {
        return m_pmatSTimeDerivatives;
    }

    daeMatrix<real_t>* GetSValues()
    {
        return m_pmatSValues;
    }

    daeMatrix<real_t>* GetSResiduals()
    {
        return m_pmatSResiduals;
    }

    void ResetSensitivityMatrixes(void)
    {
        m_pmatSValues          = NULL;
        m_pmatSTimeDerivatives = NULL;
        m_pmatSResiduals       = NULL;
    }

    bool IsModelDynamic(void) const
    {
        return m_bIsModelDynamic;
    }

    bool ResetLAMatrixAfterDiscontinuity(void) const
    {
        return m_bResetLAMatrixAfterDiscontinuity;
    }

    bool PrintInfo(void) const
    {
        return m_bPrintInfo;
    }

    bool CheckForInfiniteNumbers(void) const
    {
        return m_bCheckForInfinite;
    }

    daeBlock* GetBlock(void) const
    {
        return m_pBlock;
    }

    void SetBlock(daeBlock* pBlock)
    {
        m_pBlock = pBlock;
    }

    daeCondition* GetLastSatisfiedCondition(void) const
    {
        return m_pLastSatisfiedCondition;
    }

    void SetLastSatisfiedCondition(daeCondition* pCondition)
    {
        m_pLastSatisfiedCondition = pCondition;
    }

    void PrintAssignedVariables() const
    {
        std::cout << "PrintAssignedVariables" << std::endl;
        for(std::map<size_t, size_t>::const_iterator iter = m_mapAssignedVarsIndexes.begin(); iter != m_mapAssignedVarsIndexes.end(); iter++)
            std::cout << "(" << iter->first << ", " << m_arrAssignedVarsValues[iter->second] << ") ";
        std::cout << std::endl;
    }

    daeeEvaluationMode GetEvaluationMode(void) const
    {
        return m_eEvaluationMode;
    }

    void SetEvaluationMode(daeeEvaluationMode eEvaluationMode)
    {
        m_eEvaluationMode = eEvaluationMode;
    }

protected:
    daeLog_t*						m_pLog;
    daeModel_t*						m_pTopLevelModel;
    size_t							m_nTotalNumberOfVariables;
    daeBlock*						m_pBlock;
    daeCondition*                   m_pLastSatisfiedCondition;

    std::vector<real_t*>			m_pdarrValuesReferences;
    std::vector<real_t*>			m_pdarrTimeDerivativesReferences;
    std::map<size_t, size_t>		m_mapAssignedVarsIndexes;
    std::vector<real_t>             m_arrAssignedVarsValues;

// These are just temporary arrays used during the initialization phase
    real_t*							m_pdInitialValues;
    real_t*							m_pdInitialConditions;
    real_t*							m_pdAbsoluteTolerances;
    real_t*							m_pdVariablesTypes;
    real_t*							m_pdVariablesTypesGathered;
    real_t*							m_pdVariableValuesConstraints;

    bool							m_bGatherInfo;
    real_t							m_dCurrentTime;
    bool							m_bReinitializationFlag;
    bool							m_bResetLAMatrixAfterDiscontinuity;
    bool							m_bPrintInfo;
    bool                            m_bCheckForInfinite;
    bool							m_bIsModelDynamic;
    bool							m_bCopyDataFromBlock;
    daeeEvaluationMode              m_eEvaluationMode;

    daeeInitialConditionMode		m_eInitialConditionMode;
    size_t							m_nNumberOfParameters;
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
                              public daeRuntimeCheck_t
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

    virtual void	CalculateSensitivityResiduals(real_t					  dTime,
                                                  const std::vector<size_t>& narrParameterIndexes,
                                                  daeArray<real_t>&		  arrValues,
                                                  daeArray<real_t>&		  arrTimeDerivatives,
                                                  daeMatrix<real_t>&		  matSValues,
                                                  daeMatrix<real_t>&		  matSTimeDerivatives,
                                                  daeMatrix<real_t>&		  matSResiduals);

    virtual void	CalculateSensitivityParametersGradients(const std::vector<size_t>& narrParameterIndexes,
                                                            daeArray<real_t>&		   arrValues,
                                                            daeArray<real_t>&		   arrTimeDerivatives,
                                                            daeMatrix<real_t>&		   matSResiduals);

    virtual void	CalculateConditions(real_t				dTime,
                                        daeArray<real_t>&	arrValues,
                                        daeArray<real_t>&	arrTimeDerivatives,
                                        daeArray<real_t>&	arrResults);

    virtual void	FillAbsoluteTolerancesInitialConditionsAndInitialGuesses(daeArray<real_t>& arrValues,
                                                                             daeArray<real_t>& arrTimeDerivatives,
                                                                             daeArray<real_t>& arrInitialConditionsTypes,
                                                                             daeArray<real_t>& arrAbsoluteTolerances,
                                                                             daeArray<real_t>& arrValueConstraints);

    virtual size_t	GetNumberOfEquations(void) const;
    virtual size_t	GetNumberOfRoots(void) const;

    virtual bool	              CheckForDiscontinuities(void);
    virtual daeeDiscontinuityType ExecuteOnConditionActions(void);

    virtual void	CalcNonZeroElements(int& NNZ);
    virtual void	FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix);

    virtual void	SetBlockData(daeArray<real_t>& arrValues, daeArray<real_t>& arrTimeDerivatives);
    virtual void	CreateIndexMappings(real_t* pdValues, real_t* pdTimeDerivatives);
    virtual void	RebuildActiveEquationSetAndRootExpressions(bool bCalculateSensitivities);

    virtual real_t	GetTime(void) const;
    virtual void	SetTime(real_t time);

    virtual size_t	FindVariableBlockIndex(size_t nVariableOverallIndex) const;

    virtual bool	IsModelDynamic() const;
    virtual void	CleanUpSetupData();

    virtual std::vector<size_t>           GetActiveEquationSetMemory() const;
    virtual std::map<std::string, size_t> GetActiveEquationSetNodeCount() const;

    virtual std::map<std::string, call_stats::TimeAndCount> GetCallStats() const;

//	virtual real_t* GetValuesPointer();
//	virtual real_t* GetTimeDerivativesPointer();
//	virtual real_t* GetAbsoluteTolerancesPointer();
//	virtual real_t* GetVariableTypesPointer();

public:
    daeDataProxy_t*	GetDataProxy(void) const;
    void			SetDataProxy(daeDataProxy_t* pDataProxy);
    string			GetCanonicalName(void) const;
    string			GetName(void) const;
    void			SetName(const string& strName);

public:
    real_t	GetValue(size_t nBlockIndex) const;
    void	SetValue(size_t nBlockIndex, real_t dValue);

    real_t	GetTimeDerivative(size_t nBlockIndex) const;
    void	SetTimeDerivative(size_t nBlockIndex, real_t dTimeDerivative);

    real_t	GetResidual(size_t nEquationIndex) const;
    void	SetResidual(size_t nEquationIndex, real_t dResidual);

    real_t	GetJacobian(size_t nEquationIndex, size_t nVariableindexInBlock) const;
    void	SetJacobian(size_t nEquationIndex, size_t nVariableindexInBlock, real_t dJacobianItem);

    real_t	GetInverseTimeStep(void) const;
    void	SetInverseTimeStep(real_t dInverseTimeStep);

public:
    void AddEquationExecutionInfo(daeEquationExecutionInfo* pEquationExecutionInfo);
    std::vector<daeEquationExecutionInfo*>& GetEquationExecutionInfos_ActiveSet();

    bool CheckOverlappingAndAddVariables(const std::vector<size_t>& narrVariablesInEquation);
    void AddVariables(const std::map<size_t, size_t>& mapIndexes);

    bool GetInitializeMode(void) const;
    void SetInitializeMode(bool bMode);

    bool CheckObject(std::vector<string>& strarrErrors) const;

    // first - index in block;   second - index in core
    std::map<size_t, size_t>& GetVariableIndexesMap(void);

    csComputeStackEvaluator_t* GetComputeStackEvaluator();
    void SetComputeStackEvaluator(csComputeStackEvaluator_t* computeStackEvaluator);

    void BuildComputeStackStructs();
    void CleanComputeStackStructs();
    void ExportComputeStackStructs(const std::string& filenameComputeStacks,
                                   const std::string& filenameJacobianIndexes,
                                   int startEquationIndex = 0,
                                   int endEquationIndex = -1,
                                   const std::map<int,int>& bi_to_bi_local = std::map<int,int>());
    void ExportComputeStackStructs(const std::string& filenameComputeStacks,
                                   const std::string& filenameJacobianIndexes,
                                   const std::vector<uint32_t>& equationIndexes,
                                   const std::map<uint32_t,uint32_t>& bi_to_bi_local);

public:
// Used internally by the block during calculation of Residuals/Jacobian/Hesian
    void				SetValuesArray(daeArray<real_t>* pValues);
    daeArray<real_t>*	GetValuesArray(void) const;

    void				SetTimeDerivativesArray(daeArray<real_t>* pTimeDerivatives);
    daeArray<real_t>*	GetTimeDerivativesArray(void) const;

    daeMatrix<real_t>*	GetJacobianMatrix(void) const;
    void				SetJacobianMatrix(daeMatrix<real_t>* pJacobian);

    daeArray<real_t>*	GetResidualArray(void) const;
    void				SetResidualArray(daeArray<real_t>* pResidual);

    //void				SetSValuesMatrix(daeMatrix<real_t>* pSValues);
    //daeMatrix<real_t>*	GetSValuesMatrix(void) const;

    //void				SetSTimeDerivativesMatrix(daeMatrix<real_t>* pSTimeDerivatives);
    //daeMatrix<real_t>*	GetSTimeDerivativesMatrix(void) const;

    //void				SetSResidualsMatrix(daeMatrix<real_t>* pSResiduals);
    //daeMatrix<real_t>*	GetSResidualsMatrix(void) const;

public:
    bool	m_bInitializeMode;
    string	m_strName;
    size_t	m_nNumberOfEquations;
    size_t  m_nTotalNumberOfVariables;

    // Contains EEIs from models (excluding equations in STN/IF blocks).
    // This is a constant vector populated during the initilisation of the simulation.
    // EEIs from STNs can be obtained from m_ptrarrSTNs vector.
    std::vector<daeEquationExecutionInfo*>	m_ptrarrEquationExecutionInfos;

    // All currently active EEIs: m_ptrarrEquationExecutionInfos + EEIs from m_ptrarrSTNs.
    // It changes after every discontinuity that causes a change in active states.
    std::vector<daeEquationExecutionInfo*>	m_ptrarrEquationExecutionInfos_ActiveSet;

    std::vector<csComputeStackItem_t>   m_arrAllComputeStacks;
    std::vector<csJacobianMatrixItem_t> m_arrComputeStackJacobianItems;
    std::vector<uint32_t>               m_arrActiveEquationSetIndexes;
    std::vector<real_t>                 m_jacobian;

    std::map<std::string, call_stats::TimeAndCount> m_stats;

    // Contains STNs from all models; they may contain nested STNs/IFs.
    // It can be used to colect currently active equations in STNs/IFs.
    std::vector<daeSTN*>					m_ptrarrSTNs;

    std::map<size_t, daeExpressionInfo>		m_mapExpressionInfos;
    std::map<size_t, size_t>				m_mapVariableIndexes;
    daeEquationsIndexes                     m_EquationsIndexes;

    daeDataProxy_t*	m_pDataProxy;

    size_t	m_nCurrentVariableIndexForJacobianEvaluation;

    csComputeStackEvaluator_t* m_computeStackEvaluator;

    int           m_omp_num_threads;
    //std::string m_omp_schedule;
    //int         m_omp_shedule_chunk_size;

// Given by a solver during Residual/Jacobian calculation
    real_t				m_dCurrentTime;
    real_t				m_dInverseTimeStep;
    daeArray<real_t>*	m_parrValues;
    daeArray<real_t>*	m_parrTimeDerivatives;
    daeArray<real_t>*	m_parrResidual;
    daeMatrix<real_t>*	m_pmatJacobian;

    size_t m_nNuberOfResidualsCalls;
    size_t m_nNuberOfJacobianCalls;
    size_t m_nNuberOfSensitivityResidualsCalls;

    double m_dTotalTimeForResiduals;
    double m_dTotalTimeForJacobian;
    double m_dTotalTimeForSensitivityResiduals;

#if defined(DAE_MPI)
    size_t m_nEquationIndexesStart;
    size_t m_nEquationIndexesEnd;
    size_t m_nVariableIndexesStart;
    size_t m_nVariableIndexesEnd;
#endif

    friend class daeEquationExecutionInfo;
};

/******************************************************************
    daeParameter
*******************************************************************/
class DAE_CORE_API daeParameter : virtual public daeObject,
                                  virtual public daeParameter_t
{
public:
    daeDeclareDynamicClass(daeParameter)
    daeParameter(string strName, const unit& units, daeModel* pModel, string strDescription = "",
                 daeDomain* d1 = NULL, daeDomain* d2 = NULL, daeDomain* d3 = NULL, daeDomain* d4 = NULL, daeDomain* d5 = NULL, daeDomain* d6 = NULL, daeDomain* d7 = NULL, daeDomain* d8 = NULL);
    daeParameter(string strName, const unit& units, daePort* pPort, string strDescription = "",
                 daeDomain* d1 = NULL, daeDomain* d2 = NULL, daeDomain* d3 = NULL, daeDomain* d4 = NULL, daeDomain* d5 = NULL, daeDomain* d6 = NULL, daeDomain* d7 = NULL, daeDomain* d8 = NULL);
    daeParameter(void);
    virtual ~daeParameter(void);

public:
    virtual string	GetCanonicalName(void) const;
    virtual unit	GetUnits(void) const;
    virtual void	GetDomains(std::vector<daeDomain_t*>& ptrarrDomains);

    virtual bool	GetReportingOn(void) const;
    virtual void	SetReportingOn(bool bOn);

    virtual size_t	GetNumberOfPoints(void)	const;
    virtual real_t*	GetValuePointer(void);

    virtual void	SetValue(real_t value);
    virtual void	SetValue(size_t nD1, real_t value);
    virtual void	SetValue(size_t nD1, size_t nD2, real_t value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, real_t value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value);
    virtual void    SetValue(const std::vector<size_t>& narrDomainIndexes, real_t value);
    virtual void	SetValues(real_t values);
    virtual void	SetValues(const std::vector<real_t>& values);

    virtual real_t	GetValue(void);
    virtual real_t	GetValue(size_t nD1);
    virtual real_t	GetValue(size_t nD1, size_t nD2);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8);
    virtual real_t  GetValue(const std::vector<size_t>& narrDomainIndexes);
    virtual void	GetValues(std::vector<real_t>& values) const;
    virtual void    GetValues(std::vector<quantity>& quantities) const;

    virtual void	SetValue(const quantity& value);
    virtual void	SetValue(size_t nD1, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& value);
    virtual void    SetValue(const std::vector<size_t>& narrDomainIndexes, const quantity& value);
    virtual void	SetValues(const quantity& values);
    virtual void	SetValues(const std::vector<quantity>& values);

    virtual quantity	GetQuantity(void);
    virtual quantity	GetQuantity(size_t nD1);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8);
    virtual quantity    GetQuantity(const std::vector<size_t>& narrDomainIndexes);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    void Clone(const daeParameter& rObject);

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    size_t GetNumberOfDomains() const;
    void DistributeOnDomain(daeDomain& rDomain);

    daePort* GetParentPort(void) const;
    const std::vector<daeDomain*>& Domains(void) const;

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

    daeDomain* GetDomain(size_t nIndex) const;

    void GetDomainsIndexesMap(std::map<size_t, std::vector<size_t> >& mapDomainsIndexes, size_t nIndexBase) const;

protected:
    void SetUnits(const unit& units);
    void Initialize(void);

    adouble Create_adouble(const size_t* indexes, const size_t N) const;
    adouble	CreateSetupParameter(const daeDomainIndex* indexes, const size_t N) const;
    adouble_array Create_adouble_array(const daeArrayRange* ranges, const size_t N) const;
    adouble_array CreateSetupParameterArray(const daeArrayRange* ranges, const size_t N) const;

    void	Fill_adouble_array(std::vector<adouble>& arrValues, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const;
    size_t  CalculateIndex(const size_t* indexes, const size_t N) const;

protected:
    bool							m_bReportingOn;
    std::vector<real_t>				m_darrValues;
    unit							m_Unit;
    std::vector<daeDomain*>			m_ptrDomains;
    daePort*						m_pParentPort;
    std::vector< adNodePtr > m_ptrarrRuntimeNodes;
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
    daeVariable(string strName, const daeVariableType& varType, daeModel* pModel, string strDescription = "",
                daeDomain* d1 = NULL, daeDomain* d2 = NULL, daeDomain* d3 = NULL, daeDomain* d4 = NULL, daeDomain* d5 = NULL, daeDomain* d6 = NULL, daeDomain* d7 = NULL, daeDomain* d8 = NULL);
    daeVariable(string strName, const daeVariableType& varType, daePort* pPort, string strDescription = "",
                daeDomain* d1 = NULL, daeDomain* d2 = NULL, daeDomain* d3 = NULL, daeDomain* d4 = NULL, daeDomain* d5 = NULL, daeDomain* d6 = NULL, daeDomain* d7 = NULL, daeDomain* d8 = NULL);

    virtual ~daeVariable(void);

public:
    virtual string                   GetCanonicalName(void) const;
    virtual const daeVariableType_t* GetVariableType(void) const;
    virtual void                     GetDomains(std::vector<daeDomain_t*>& ptrarrDomains);

    virtual size_t	GetOverallIndex(void)	const;
    virtual size_t	GetNumberOfPoints(void) const;

    virtual int     GetType() const; // Can be algebraic, assigned, differential, some-points-assigned or some-points-differential

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
    virtual void    SetValue(const std::vector<size_t>& narrDomainIndexes, real_t value);
    virtual void	SetValues(const std::vector<real_t>& values);
    virtual void    SetValues(const std::vector<quantity>& quantities);

    virtual real_t	GetValue(void);
    virtual real_t	GetValue(size_t nD1);
    virtual real_t	GetValue(size_t nD1, size_t nD2);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7);
    virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8);
    virtual real_t  GetValue(const std::vector<size_t>& narrDomainIndexes);
    virtual void	GetValues(std::vector<real_t>& values) const;
    virtual void    GetValues(std::vector<quantity>& quantities) const;

    virtual void	SetValue(const quantity& value);
    virtual void	SetValue(size_t nD1, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& value);
    virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& value);
    virtual void    SetValue(const std::vector<size_t>& narrDomainIndexes, const quantity& value);

    virtual quantity	GetQuantity(void);
    virtual quantity	GetQuantity(size_t nD1);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7);
    virtual quantity	GetQuantity(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8);
    virtual quantity    GetQuantity(const std::vector<size_t>& narrDomainIndexes);

    virtual void	AssignValue(real_t value);
    virtual void	AssignValue(size_t nD1, real_t value);
    virtual void	AssignValue(size_t nD1, size_t nD2, real_t value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, real_t value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value);
    virtual void    AssignValue(const std::vector<size_t>& narrDomainIndexes, real_t value);
    virtual void	AssignValues(real_t values);
    virtual void	AssignValues(const std::vector<real_t>& values);

    virtual void	ReAssignValue(real_t value);
    virtual void	ReAssignValue(size_t nD1, real_t value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, real_t value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, real_t value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value);
    virtual void    ReAssignValue(const std::vector<size_t>& narrDomainIndexes, real_t value);
    virtual void	ReAssignValues(real_t values);
    virtual void	ReAssignValues(const std::vector<real_t>& values);

    virtual void	SetInitialGuess(real_t dInitialGuess);
    virtual void	SetInitialGuess(size_t nD1, real_t dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, real_t dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, real_t dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialGuesses);
    virtual void    SetInitialGuess(const std::vector<size_t>& narrDomainIndexes, real_t dInitialGuess);
    virtual void	SetInitialGuesses(real_t dInitialGuesses);
    virtual void	SetInitialGuesses(const std::vector<real_t>& initialGuesses);

    virtual void	SetInitialCondition(real_t dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, real_t dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, real_t dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, real_t dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialCondition);
    virtual void    SetInitialCondition(const std::vector<size_t>& narrDomainIndexes, real_t dInitialCondition);
    virtual void	SetInitialConditions(real_t dInitialConditions);
    virtual void	SetInitialConditions(const std::vector<real_t>& initialConditions);

    virtual void	ReSetInitialCondition(real_t dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, real_t dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, real_t dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, real_t dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialCondition);
    virtual void    ReSetInitialCondition(const std::vector<size_t>& narrDomainIndexes, real_t dInitialCondition);
    virtual void	ReSetInitialConditions(real_t dInitialConditions);
    virtual void	ReSetInitialConditions(const std::vector<real_t>& initialConditions);

    virtual void	AssignValue(const quantity& value);
    virtual void	AssignValue(size_t nD1, const quantity& value);
    virtual void	AssignValue(size_t nD1, size_t nD2, const quantity& value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, const quantity& value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& value);
    virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& value);
    virtual void    AssignValue(const std::vector<size_t>& narrDomainIndexes, const quantity& value);
    virtual void	AssignValues(const quantity& values);
    virtual void	AssignValues(const std::vector<quantity>& values);

    virtual void	ReAssignValue(const quantity& value);
    virtual void	ReAssignValue(size_t nD1, const quantity& value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, const quantity& value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, const quantity& value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& value);
    virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& value);
    virtual void    ReAssignValue(const std::vector<size_t>& narrDomainIndexes, const quantity& value);
    virtual void	ReAssignValues(const quantity& values);
    virtual void	ReAssignValues(const std::vector<quantity>& values);

    virtual void	SetInitialGuess(const quantity& dInitialGuess);
    virtual void	SetInitialGuess(size_t nD1, const quantity& dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, const quantity& dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, const quantity& dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& dInitialGuesses);
    virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& dInitialGuesses);
    virtual void    SetInitialGuess(const std::vector<size_t>& narrDomainIndexes, const quantity& dInitialGuesses);
    virtual void	SetInitialGuesses(const quantity& dInitialGuesses);
    virtual void	SetInitialGuesses(const std::vector<quantity>& initialGuesses);

    virtual void	SetInitialCondition(const quantity& dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, const quantity& dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, const quantity& dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, const quantity& dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& dInitialCondition);
    virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& dInitialCondition);
    virtual void    SetInitialCondition(const std::vector<size_t>& narrDomainIndexes, const quantity& dInitialCondition);
    virtual void	SetInitialConditions(const quantity& dInitialConditions);
    virtual void	SetInitialConditions(const std::vector<quantity>& initialConditions);

    virtual void	ReSetInitialCondition(const quantity& dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, const quantity& dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, const quantity& dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, const quantity& dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, const quantity& dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, const quantity& dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, const quantity& dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, const quantity& dInitialCondition);
    virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, const quantity& dInitialCondition);
    virtual void    ReSetInitialCondition(const std::vector<size_t>& narrDomainIndexes, const quantity& dInitialCondition);
    virtual void	ReSetInitialConditions(const quantity& dInitialConditions);
    virtual void	ReSetInitialConditions(const std::vector<quantity>& initialConditions);

    virtual void	SetValueConstraint(daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraint(size_t nD1, daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraint(size_t nD1, size_t nD2, daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraint(size_t nD1, size_t nD2, size_t nD3, daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraint(size_t nD1, size_t nD2, size_t nD3, size_t nD4, daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraint(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraint(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraint(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraint(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, daeeVariableValueConstraint eConstraint);
    virtual void    SetValueConstraint(const std::vector<size_t>& narrDomainIndexes, daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraints(daeeVariableValueConstraint eConstraint);
    virtual void	SetValueConstraints(const std::vector<daeeVariableValueConstraint>& constraints);

    virtual void	SetAbsoluteTolerances(real_t dAbsTolerances);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    void Clone(const daeVariable& rObject);

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    size_t GetNumberOfDomains() const;
    void DistributeOnDomain(daeDomain& rDomain);

    daeDomain* GetDomain(size_t nIndex) const;

    void GetDomainsIndexesMap(std::map<size_t, std::vector<size_t> >& mapDomainsIndexes, size_t nIndexBase) const;

    daePort* GetParentPort(void) const;
    const std::vector<daeDomain*>& Domains(void) const;

    virtual string GetCanonicalNameAndPrepend(const std::string& prependToName) const;

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

    const std::vector<size_t>& GetBlockIndexes() const;

protected:
    adouble	Create_adouble(const size_t* indexes, const size_t N) const;
    adouble	Calculate_dt(const size_t* indexes, const size_t N) const;
    adouble	partial(const size_t nOrder,
                    const daeDomain_t& rDomain,
                    const size_t* indexes,
                    const size_t N,
                    const daeeDiscretizationMethod  eDiscretizationMethod,
                    const std::map<std::string, std::string>& mapDiscretizationOptions) const;
    adouble	CreateSetupVariable(const daeDomainIndex* indexes, const size_t N) const;
    adouble	CreateSetupTimeDerivative(const daeDomainIndex* indexes, const size_t N) const;
    adouble	CreateSetupPartialDerivative(const size_t nOrder,
                                         const daeDomain_t& rDomain,
                                         const daeDomainIndex* indexes,
                                         const size_t N,
                                         const daeeDiscretizationMethod  eDiscretizationMethod,
                                         const std::map<std::string, std::string>& mapDiscretizationOptions) const;

    adouble_array Create_adouble_array(const daeArrayRange* ranges, const size_t N) const;
    adouble_array Calculate_dt_array(const daeArrayRange* ranges, const size_t N) const;
    adouble_array partial_array(const size_t nOrder,
                                const daeDomain_t& rDomain,
                                const daeArrayRange* ranges,
                                const size_t N,
                                const daeeDiscretizationMethod eDiscretizationMethod,
                                const std::map<std::string, std::string>& mapDiscretizationOptions) const;
    adouble_array CreateSetupVariableArray(const daeArrayRange* ranges, const size_t N) const;
    adouble_array CreateSetupTimeDerivativeArray(const daeArrayRange* ranges, const size_t N) const;
    adouble_array CreateSetupPartialDerivativeArray(const size_t nOrder,
                                                    const daeDomain_t& rDomain,
                                                    const daeArrayRange* ranges,
                                                    const size_t N,
                                                    const daeeDiscretizationMethod  eDiscretizationMethod,
                                                    const std::map<std::string, std::string>& mapDiscretizationOptions) const;

    void	SetVariableType(const daeVariableType& VariableType);
    void	Fill_adouble_array(std::vector<adouble>& arrValues, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const;
    void	Fill_dt_array(std::vector<adouble>& arrValues, const daeArrayRange* ranges, size_t* indexes, const size_t N, size_t currentN) const;
    void	Fill_partial_array(std::vector<adouble>& arrValues,
                               size_t nOrder,
                               const daeDomain_t& rDomain,
                               const daeArrayRange* ranges,
                               size_t* indexes,
                               const size_t N,
                               size_t currentN,
                               daeeDiscretizationMethod  eDiscretizationMethod,
                               const std::map<std::string, std::string>& mapDiscretizationOptions) const;

    size_t	CalculateIndex(const size_t* indexes, size_t N) const;
    size_t	CalculateIndex(const std::vector<size_t>& narrDomainIndexes) const;

    void    InitializeBlockIndexes(const std::map<size_t, size_t>& mapOverallIndex_BlockIndex);

protected:
    size_t					m_nOverallIndex;
    daeVariableType			m_VariableType;
    bool					m_bReportingOn;
    std::vector<daeDomain*>	m_ptrDomains;
    daePort*				m_pParentPort;

    std::vector<size_t>     m_narrBlockIndexes;

    std::map<size_t, adouble> m_mapSetupVariableNodes;
    std::map<size_t, adouble> m_mapSetupTimeDerivativeNodes;
    std::map<size_t, adouble> m_mapRuntimeVariableNodes;
    std::map<size_t, adouble> m_mapRuntimeTimeDerivativeNodes;

    friend class daePort;
    friend class daeModel;
    friend class daeFiniteElementEquation;
    friend class daeAction;
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
    friend class adSetupExpressionPartialDerivativeNodeArray;
    friend class adRuntimeVariableNode;
    friend class adRuntimeTimeDerivativeNode;
    friend class adRuntimePartialDerivativeNode;
    friend class daeOptimizationVariable;
    friend class daeFunctionWithGradients;
    friend class daeVariableWrapper;
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
    daePort(string strName, daeePortType portType, daeModel* parent, string strDescription = string(""));
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

    virtual daeObject_t* FindObject(string& strName);
    virtual daeObject_t* FindObjectFromRelativeName(string& strRelativeName);
    virtual daeObject_t* FindObjectFromRelativeName(std::vector<string>& strarrNames);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    void DetectVariableTypesForExport(std::vector<const daeVariableType*>& ptrarrVariableTypes) const;
    void CreateDefinition(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    void Clone(const daePort& rObject);

    void CleanUpSetupData(void);

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void SetType(daeePortType eType);
    void AddDomain(daeDomain& rDomain, const string& strName, const unit& units, string strDescription = "");
    void AddVariable(daeVariable& rVariable, const string& strName, const daeVariableType& rVariableType, string strDescription = "");
    void AddParameter(daeParameter& rParameter, const string& strName, const unit& units, string strDescription = "");

    const std::vector<daeDomain*>& Domains() const;
    const std::vector<daeParameter*>& Parameters() const;
    const std::vector<daeVariable*>& Variables() const;

    void CreateOverallIndex_BlockIndex_VariableNameMap(std::map<size_t, std::pair<size_t, string> >& mapOverallIndex_BlockIndex_VariableName,
                                                       const std::map<size_t, size_t>& mapOverallIndex_BlockIndex);

    void PropagateDomain(daeDomain& propagatedDomain);
    void PropagateParameter(daeParameter& propagatedParameter);

protected:
    void			InitializeParameters(void);
    void			InitializeVariables(void);
    void            InitializeBlockIndexes(const std::map<size_t, size_t>& mapOverallIndex_BlockIndex);
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
    daeEventPort
*******************************************************************/
class daeEventPort : virtual public daeObject,
                     virtual public daeEventPort_t
{
public:
    daeDeclareDynamicClass(daeEventPort)
    daeEventPort(void);
    daeEventPort(string strName, daeePortType eType, daeModel* pModel, const string& strDescription = string(""));
    virtual ~daeEventPort(void);

public:
    virtual daeePortType	GetType(void) const;
    virtual void			SetType(daeePortType eType);
    virtual void			SendEvent(real_t data);

    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    void Clone(const daeEventPort& rObject);

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Update(daeEventPort_t* pSubject, void* data);

    void Initialize(void);

    real_t GetEventData(void);

    bool GetRecordEvents() const;
    void SetRecordEvents(bool bRecordEvents);
    const std::list< std::pair<real_t, real_t> >& GetListOfEvents(void) const;

    adouble operator()(void);

    void ReceiveEvent(real_t data);

protected:
    real_t									m_dEventData;
    daeePortType							m_ePortType;
    bool									m_bRecordEvents;
    std::list< std::pair<real_t, real_t> >	m_listEvents;
};

/******************************************************************
    daeAction
*******************************************************************/
class daeState;
class daeVariableWrapper;
class daeAction : virtual public daeObject,
                  virtual public daeAction_t
{
public:
    daeDeclareDynamicClass(daeAction)
    daeAction(void);
    daeAction(const string& strName, daeModel* pModel, daeSTN* pSTN, const string& strStateTo, const string& strDescription);
    daeAction(const string& strName, daeModel* pModel, const string& strSTN, const string& strStateTo, const string& strDescription);
    daeAction(const string& strName, daeModel* pModel, daeEventPort* pPort, adouble data, const string& strDescription);
    daeAction(const string& strname, daeModel* pModel, const daeVariableWrapper& variable, const adouble value, const string& strDescription);
    virtual ~daeAction(void);

public:
    virtual daeeActionType	GetType(void) const;
    virtual void			Execute(void);

    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    void Clone(const daeAction& rObject);
    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Initialize(void);

    daeSTN* GetSTN() const;
    daeState* GetStateTo() const;
    daeEventPort* GetSendEventPort() const;
    daeVariableWrapper* GetVariableWrapper() const;
    adNodePtr getSetupNode() const;
    adNode* getSetupNodeRawPtr() const;
    adNodePtr getRuntimeNode() const;
    adNode* getRuntimeNodeRawPtr() const;

protected:
    void SaveNodeAsMathML(adNode* node, io::xmlTag_t* pTag, const string& strObjectName) const;

protected:
    daeeActionType m_eActionType;

// For eChangeState:
    daeSTN*		m_pSTN;
    string      m_strSTN;
    string		m_strStateTo;
    daeState*	m_pStateTo;

// For eSendEvent:
    daeEventPort* m_pSendEventPort;

// For eReAssignOrReInitializeVariable:
    boost::shared_ptr<daeVariableWrapper> m_pVariableWrapper;

// Common for eSendEvent and eReAssignOrReInitializeVariable:
    adNodePtr	m_pSetupNode;
    adNodePtr	m_pNode;
};

/******************************************************************
    daeOnEventActions
*******************************************************************/
class daeOnEventActions : virtual public daeObject,
                          virtual public daeOnEventActions_t
{
public:
    daeDeclareDynamicClass(daeOnEventActions)
    daeOnEventActions(void);
    daeOnEventActions(daeEventPort* pEventPort,
                      daeModel* pModel,
                      std::vector<daeAction*>& ptrarrOnEventActions,
                      std::vector<daeAction*>& ptrarrUserDefinedOnEventActions,
                      const string& strDescription);
    daeOnEventActions(daeEventPort* pEventPort,
                      daeState* pState,
                      std::vector<daeAction*>& ptrarrOnEventActions,
                      std::vector<daeAction*>& ptrarrUserDefinedOnEventActions,
                      const string& strDescription);
    virtual ~daeOnEventActions(void);

public:
    virtual string GetCanonicalName(void) const;
    virtual void   Execute(void);
    virtual void   Update(daeEventPort_t *pSubject, void* data);

    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;

    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    void Clone(const daeOnEventActions& rObject);
    bool CheckObject(std::vector<string>& strarrErrors) const;

    daeEventPort* GetEventPort(void) const;
    const std::vector<daeAction*>& Actions() const;
    const std::vector<daeAction*>& UserDefinedActions() const;

    void Initialize(void);

protected:
    daeEventPort*            m_pEventPort;
    daePtrVector<daeAction*> m_ptrarrOnEventActions;
    std::vector<daeAction*>  m_ptrarrUserDefinedOnEventActions;
    daeState*				 m_pParentState;
    friend class daeState;
};

/******************************************************************
    daeOnConditionActions
*******************************************************************/
class DAE_CORE_API daeOnConditionActions : virtual public daeObject,
                                           virtual public daeOnConditionActions_t
{
public:
    daeDeclareDynamicClass(daeOnConditionActions)
    daeOnConditionActions(void);
    virtual ~daeOnConditionActions(void);

public:
    string GetCanonicalName(void) const;
    void Execute(void);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    bool CheckObject(std::vector<string>& strarrErrors) const;
    void CleanUpSetupData();
    void Clone(const daeOnConditionActions& rObject);
    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    void Initialize(void);
    void Create_SWITCH_TO(daeState* pStateFrom, const string& strStateToName, const daeCondition& rCondition, real_t dEventTolerance);
    void Create_IF(daeState* pStateTo, const daeCondition& rCondition, real_t dEventTolerance);
    void Create_ON_CONDITION(daeState* pStateFrom,
                             daeModel* pModel,
                             const daeCondition& rCondition,
                             std::vector<daeAction*>& ptrarrActions,
                             std::vector<daeAction*>& ptrarrUserDefinedActions,
                             real_t dEventTolerance);

    string GetConditionAsString() const;

    daeCondition*	GetCondition(void);
    void			SetCondition(daeCondition& rCondition);

    const std::vector<daeAction*>& Actions(void) const;
    const std::vector<daeAction*>& UserDefinedActions(void) const;

protected:
    daePtrVector<daeAction*>			m_ptrarrActions;
    std::vector<daeAction*>				m_ptrarrUserDefinedActions;
    daeCondition						m_Condition;
    std::map<size_t, daeExpressionInfo>	m_mapExpressionInfos;
    daeState*                           m_pParentState;

// Internal variables, used only during Open
    long	m_nStateFromID;
    long	m_nStateToID;
    friend class daeIF;
    friend class daeSTN;
    friend class daeModel;
    friend class daeState;
};

/******************************************************************
    daePortConnection
*******************************************************************/
class DAE_CORE_API daePortConnection : virtual public daeObject,
                                       virtual public daePortConnection_t
{
public:
    daeDeclareDynamicClass(daePortConnection)
    daePortConnection();
    daePortConnection(daePort* pPortFrom, daePort* pPortTo);
    virtual ~daePortConnection(void);

public:
    virtual daePort_t*	GetPortFrom(void) const;
    virtual daePort_t*	GetPortTo(void) const;

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    bool CheckObject(std::vector<string>& strarrErrors) const;
    void Clone(const daePortConnection& rObject);
    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    void   GetEquations(std::vector<daeEquation*>& ptrarrEquations) const;
    size_t GetTotalNumberOfEquations(void) const;

protected:
    void CreateEquations(void);

protected:
    daePort*					m_pPortFrom;
    daePort*					m_pPortTo;
    daePtrVector<daeEquation*>	m_ptrarrEquations;
    friend class daeModel;
};

/******************************************************************
    daeEventPortConnection
*******************************************************************/
class daeRemoteEventReceiver;
class daeRemoteEventSender;

class DAE_CORE_API daeEventPortConnection : virtual public daeObject,
                                            virtual public daeEventPortConnection_t
{
public:
    daeDeclareDynamicClass(daeEventPortConnection)
    daeEventPortConnection();
    daeEventPortConnection(daeEventPort* pPortFrom, daeEventPort* pPortTo);
    virtual ~daeEventPortConnection(void);

public:
    virtual daeEventPort_t*	GetPortFrom(void) const;
    virtual daeEventPort_t*	GetPortTo(void) const;

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    bool CheckObject(std::vector<string>& strarrErrors) const;
    void Clone(const daePortConnection& rObject);
    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

protected:
    boost::shared_ptr<daeRemoteEventReceiver> receiver;
    boost::shared_ptr<daeRemoteEventSender>   sender;

    daeEventPort* m_pPortFrom;
    daeEventPort* m_pPortTo;
    friend class daeModel;
};

/******************************************************************
    Opaque support or Python threads and GIL support
*******************************************************************/
class daeGILState_t
{
public:
    virtual ~daeGILState_t(){}
};

class daeAllowThreads_t
{
public:
    virtual ~daeAllowThreads_t(){}
};

/******************************************************************
    daeModel
*******************************************************************/
class daeState;
class daeIF;
class daeSTN;
class daeModelArray;
class daePortArray;
class daeExternalFunction_t;
class DAE_CORE_API daeModel : virtual public daeObject,
                              virtual public daeModel_t
{
public:
    daeDeclareDynamicClass(daeModel)
    daeModel(void);
    daeModel(string strName, daeModel* pModel = NULL, string strDescription = "");
    virtual ~daeModel(void);

public:
    virtual void        InitializeModel(const std::string& jsonInit);

    virtual void        InitializeStage1(void);
    virtual void    	InitializeStage2(void);
    virtual void        InitializeStage3(daeLog_t* pLog);
    virtual void        InitializeStage4(void);
    virtual daeBlock_t*	InitializeStage5(void);
    virtual void        InitializeStage6(daeBlock_t* pBlock);

    virtual void	CleanUpSetupData(void);
    virtual void    UpdateEquations();

    virtual void	SaveModelReport(const string& strFileName) const;
    virtual void	SaveRuntimeModelReport(const string& strFileName) const;

    virtual void	SetReportingOn(bool bOn);

    virtual daeeInitialConditionMode GetInitialConditionMode(void) const;
    virtual void SetInitialConditionMode(daeeInitialConditionMode eMode);
    //virtual void SetInitialConditions(real_t value);

    virtual void StoreInitializationValues(const std::string& strFileName) const;
    virtual void LoadInitializationValues(const std::string& strFileName) const;

    virtual bool IsModelDynamic() const;
    virtual daeeModelType GetModelType() const;

    //boost::shared_ptr<daeExternalObject_t> LoadExternalObject(const string& strPath);

    virtual void	GetModelInfo(daeModelInfo& mi) const;

    virtual void	GetSTNs(std::vector<daeSTN_t*>& ptrarrSTNs);
    virtual void	GetPorts(std::vector<daePort_t*>& ptrarrPorts);
    virtual void	GetEquations(std::vector<daeEquation_t*>& ptrarrEquations);
    virtual void	GetModels(std::vector<daeModel_t*>& ptrarrModels);
    virtual void	GetDomains(std::vector<daeDomain_t*>& ptrarrDomains);
    virtual void	GetVariables(std::vector<daeVariable_t*>& ptrarrVariables);
    virtual void	GetParameters(std::vector<daeParameter_t*>& ptrarrParameters);
    virtual void	GetPortConnections(std::vector<daePortConnection_t*>& ptrarrPortConnections);
    virtual void	GetEventPortConnections(std::vector<daeEventPortConnection_t*>& ptrarrEventPortConnections);
    virtual void	GetPortArrays(std::vector<daePortArray_t*>& ptrarrPortArrays);
    virtual void	GetModelArrays(std::vector<daeModelArray_t*>& ptrarrModelArrays);

    virtual void	CollectAllDomains(std::map<dae::string, daeDomain_t*>& mapDomains) const;
    virtual void	CollectAllParameters(std::map<dae::string, daeParameter_t*>& mapParameters) const;
    virtual void	CollectAllVariables(std::map<dae::string, daeVariable_t*>& mapVariables) const;
    virtual void	CollectAllSTNs(std::map<dae::string, daeSTN_t*>& mapSTNs) const;
    virtual void	CollectAllPorts(std::map<dae::string, daePort_t*>& mapPorts) const;

    virtual void    GetCoSimulationInterface(std::vector<daeParameter_t*>& ptrarrParameters,
                                             std::vector<daeVariable_t*>&  ptrarrInputs,
                                             std::vector<daeVariable_t*>&  ptrarrOutputs,
                                             std::vector<daeVariable_t*>&  ptrarrModelVariables,
                                             std::vector<daeSTN_t*>&       ptrarrSTNs);
    virtual void    GetFMIInterface(std::map<size_t, daeFMI2Object_t>& mapInterface);

    virtual daeDomain_t*		FindDomain(string& strCanonicalName);
    virtual daeParameter_t*		FindParameter(string& strCanonicalName);
    virtual daeVariable_t*		FindVariable(string& strCanonicalName);
    virtual daePort_t*			FindPort(string& strCanonicalName);
    virtual daeModel_t*			FindModel(string& strCanonicalName);
    virtual daeEventPort_t*		FindEventPort(string& strName);
    virtual daeSTN_t*			FindSTN(string& strCanonicalName);
    virtual daePortArray_t*		FindPortArray(string& strCanonicalName);
    virtual daeModelArray_t*	FindModelArray(string& strCanonicalName);

    virtual daeObject_t*		FindObject(string& strName);
    virtual daeObject_t*		FindObjectFromRelativeName(string& strRelativeName);
    virtual daeObject_t*		FindObjectFromRelativeName(std::vector<string>& strarrNames);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    void Clone(const daeModel& rObject);
    bool CheckObject(std::vector<string>& strarrErrors) const;

    string ExportObjects(std::vector<daeExportable_t*>& ptrarrObjects, daeeModelLanguage eLanguage) const;
    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    void DetectVariableTypesForExport(std::vector<const daeVariableType*>& ptrarrVariableTypes) const;
    void CreateDefinition(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    void daeSaveRuntimeNodes(string strFileName);

    virtual size_t	GetTotalNumberOfVariables(void) const;
    virtual size_t	GetTotalNumberOfEquations(void) const;

    virtual boost::shared_ptr<daeAllowThreads_t> CreateAllowThreads();
    virtual boost::shared_ptr<daeGILState_t>     CreateGILState();

    size_t GetNumberOfSTNs(void) const;

    void BuildExpressions(daeBlock* pBlock);
    bool CheckDiscontinuities(void);
    void AddExpressionsToBlock(daeBlock* pBlock);
    void ExecuteOnConditionActions(void);

    daeEquation* CreateEquation(const string& strName, string strDescription = "", real_t dScaling = 1.0);

    void AddEquation(daeEquation* pEquation);
    void AddDomain(daeDomain* pDomain);
    void AddVariable(daeVariable* pVariable);
    void AddParameter(daeParameter* pParameter);
    void AddModel(daeModel* pModel);
    void AddPort(daePort* pPort);
    void AddEventPort(daeEventPort* pPort);
    void AddOnEventAction(daeOnEventActions* pOnEventAction);
    void AddOnConditionAction(daeOnConditionActions* pOnConditionAction);
    void AddPortConnection(daePortConnection* pPortConnection);
    void AddEventPortConnection(daeEventPortConnection* pEventPortConnection);
    void AddPortArray(daePortArray* pPortArray);
    void AddModelArray(daeModelArray* pModelArray);
    void AddExternalFunction(daeExternalFunction_t* pExternalFunction);

    void ConnectPorts(daePort* pPortFrom, daePort* pPortTo);
    void ConnectEventPorts(daeEventPort* pPortFrom, daeEventPort* pPortTo);

    void PropagateDomain(daeDomain& propagatedDomain);
    void PropagateParameter(daeParameter& propagatedParameter);

    const std::vector<daePort*>& Ports() const;
    const std::vector<daeEventPort*>& EventPorts() const;
    const std::vector<daeModel*>& Models() const;
    const std::vector<daeDomain*>& Domains() const;
    const std::vector<daeParameter*>& Parameters() const;
    const std::vector<daeVariable*>& Variables() const;
    const std::vector<daeEquation*>& Equations() const;
    const std::vector<daeSTN*>& STNs() const;
    const std::vector<daeOnEventActions*>& OnEventActions() const;
    const std::vector<daeOnConditionActions*>& OnConditionActions() const;
    const std::vector<daePortConnection*>& PortConnections() const;
    const std::vector<daeEventPortConnection*>& EventPortConnections() const;
    const std::vector<daePortArray*>& PortArrays() const;
    const std::vector<daeModelArray*>& ModelArrays() const;

    const std::vector<csComputeStackItem_t>& GetComputeStack() const;

// Overridables
public:
    virtual void DeclareEquations(void);

protected:
    void AddDomain(daeDomain& rDomain, const string& strName, const unit& units, string strDescription = "");
    void AddVariable(daeVariable& rVariable, const string& strName, const daeVariableType& rVariableType, string strDescription = "");
    void AddParameter(daeParameter& rParameter, const string& strName, const unit& units, string strDescription = "");
    void AddModel(daeModel& rModel, const string& strName, string strDescription = "");
    void AddPort(daePort& rPort, const string& strName, daeePortType ePortType, string strDescription = "");
    void AddEventPort(daeEventPort& rPort, const string& strName, daeePortType ePortType, string strDescription);
    void AddOnEventAction(daeOnEventActions& rOnEventAction, const string& strName, string strDescription = "");
    void AddPortArray(daePortArray& rPortArray, const string& strName, daeePortType ePortType, string strDescription = "");
    void AddModelArray(daeModelArray& rModelArray, const string& strName, string strDescription = "");

    void DeclareEquationsBase(void);

    daeSTN* AddSTN(const string& strName);
    daeIF*  AddIF(const string& strName);

public:
    void IF(const daeCondition& rCondition, real_t dEventTolerance = 0, const string& strIFName= "", const string& strIFDescription = "",
                                                                        const string& strStateName = "", const string& strStateDescription = "");
    void ELSE_IF(const daeCondition& rCondition, real_t dEventTolerance = 0, const string& strStateName = "", const string& strStateDescription = "");
    void ELSE(const string& strStateDescription = "");
    void END_IF(void);

    daeSTN*   STN(const string& strName, const string& strDescription = "");
    daeState* STATE(const string& strName, const string& strDescription = "");
    void      END_STN(void);
    void      SWITCH_TO(const string& strState, const daeCondition& rCondition, real_t dEventTolerance = 0);

    void ON_CONDITION(const daeCondition&                                       rCondition,
                      std::vector< std::pair<string, string> >&					arrSwitchToStates,
                      std::vector< std::pair<daeVariableWrapper, adouble> >&    arrSetVariables,
                      std::vector< std::pair<daeEventPort*, adouble> >&			arrTriggerEvents,
                      std::vector<daeAction*>&                                  ptrarrUserDefinedActions,
                      real_t                                                    dEventTolerance = 0.0);

    void ON_EVENT(daeEventPort*												pTriggerEventPort,
                  std::vector< std::pair<string, string> >&					arrSwitchToStates,
                  std::vector< std::pair<daeVariableWrapper, adouble> >&	arrSetVariables,
                  std::vector< std::pair<daeEventPort*, adouble> >&			arrTriggerEvents,
                  std::vector<daeAction*>&									ptrarrUserDefinedOnEventActions);

    daeDomain*		FindDomain(unsigned long nID) const;
    daePort*		FindPort(unsigned long nID) const;
    daeEventPort*	FindEventPort(unsigned long nID) const;
    daeVariable*	FindVariable(unsigned long nID) const;

    boost::shared_ptr<daeDataProxy_t> GetDataProxy(void) const;

    void RemoveModel(daeModel* pObject);
    void RemoveEquation(daeEquation* pObject);
    void RemoveSTN(daeSTN* pObject);
    void RemovePortConnection(daePortConnection* pObject);
    void RemoveEventPortConnection(daeEventPortConnection* pObject);
    void RemoveDomain(daeDomain* pObject);
    void RemoveParameter(daeParameter* pObject);
    void RemoveVariable(daeVariable* pObject);
    void RemovePort(daePort* pObject);
    void RemoveEventPort(daeEventPort* pObject);
    void RemoveOnEventAction(daeOnEventActions* pObject);
    void RemoveOnConditionAction(daeOnConditionActions* pObject);
    void RemovePortArray(daePortArray* pObject);
    void RemoveModelArray(daeModelArray* pObject);
    void RemoveExternalFunction(daeExternalFunction_t* pObject);

    void CreateOverallIndex_BlockIndex_VariableNameMap(std::map<size_t, std::pair<size_t, string> >& mapOverallIndex_BlockIndex_VariableName,
                                                       const std::map<size_t, size_t>& mapOverallIndex_BlockIndex);

// Internal functions
protected:
    void		InitializeParameters(void);
    void		InitializeVariables(void);
    void		InitializeEquations(void);
    void		InitializeDEDIs(void);
    void		InitializePortAndModelArrays(void);
    void		InitializeSTNs(void);
    void		InitializeOnEventAndOnConditionActions(void);
    void        InitializeBlockIndexes(const std::map<size_t, size_t>& mapOverallIndex_BlockIndex);
    daeBlock*	DoBlockDecomposition(void);
    void        PopulateBlockIndexes(daeBlock* pBlock);
    void		SetDefaultAbsoluteTolerances(void);
    void		SetDefaultInitialGuessesAndConstraints(void);
    void		BuildUpSTNsAndEquations(void);
    void		CreatePortConnectionEquations(void);

    void		PropagateDataProxy(boost::shared_ptr<daeDataProxy_t> pDataProxy);
    void		PropagateGlobalExecutionContext(daeExecutionContext* pExecutionContext);

    size_t		GetVariablesStartingIndex(void) const;
    void		SetVariablesStartingIndex(size_t nVariablesStartingIndex);

    void		AddEquationExecutionInfo(daeEquationExecutionInfo* pEquationExecutionInfo);
    void		GetEquationExecutionInfos(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfos);

    bool		FindObject(string& strCanonicalName, daeObjectType& ObjectType);
    bool		FindObject(std::vector<string>& strarrHierarchy, daeObjectType& ObjectType);

protected:
    void		CollectAllSTNsAsVector(std::vector<daeSTN*>& ptrarrSTNs) const;
    void		CollectEquationExecutionInfosFromModels(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const;
    void		CollectEquationExecutionInfosFromSTNs(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const;
    bool		DetectObject(string& strShortName, std::vector<size_t>& narrDomains, daeeObjectType& eType, daeObject_t** ppObject);

protected:
    boost::shared_ptr<daeDataProxy_t>	m_pDataProxy;
    size_t								m_nVariablesStartingIndex;
    size_t								m_nTotalNumberOfVariables;

    daePtrVector<daeModel*>					m_ptrarrComponents;
    daePtrVector<daeEquation*>				m_ptrarrEquations;
    daePtrVector<daeSTN*>					m_ptrarrSTNs;
    daePtrVector<daePortConnection*>		m_ptrarrPortConnections;
    daePtrVector<daeEventPortConnection*>	m_ptrarrEventPortConnections;

// When used programmatically they dont own pointers
    daePtrVector<daeDomain*>				m_ptrarrDomains;
    daePtrVector<daeParameter*>				m_ptrarrParameters;
    daePtrVector<daeVariable*>				m_ptrarrVariables;
    daePtrVector<daePort*>					m_ptrarrPorts;
    daePtrVector<daeEventPort*>				m_ptrarrEventPorts;
    daePtrVector<daeOnEventActions*>		m_ptrarrOnEventActions;
    daePtrVector<daeOnConditionActions*>	m_ptrarrOnConditionActions;
    daePtrVector<daePortArray*>				m_ptrarrPortArrays;
    daePtrVector<daeModelArray*>			m_ptrarrComponentArrays;
    daePtrVector<daeExternalFunction_t*>	m_ptrarrExternalFunctions;

    // daeEquation owns the pointers
    std::vector<daeEquationExecutionInfo*> m_ptrarrEquationExecutionInfos;

// Used to nest STNs/IFs
    std::stack<daeState*> m_ptrarrStackStates;
    std::stack<daeSTN*>   m_ptrarrStackSTNs;

    size_t	_currentVariablesIndex;
// Used only during GatherInfo
    daeExecutionContext*	m_pExecutionContextForGatherInfo;

    std::vector<daeEquationExecutionInfo*> m_ptrarrEEIfromModels;
    std::vector<daeEquationExecutionInfo*> m_ptrarrEEIfromSTNs;
    std::vector<daeSTN*>                   m_ptrarrAllSTNs;

    friend class daeIF;
    friend class daeState;
    friend class daeSTN;
    friend class daeOnConditionActions;
    friend class daePort;
    friend class daeEventPort;
    friend class daeAction;
    friend class daeOnEventActions;
    friend class daeObject;
    friend class daeDomain;
    friend class daeVariable;
    friend class daeParameter;
    friend class daeEquation;
    friend class daeFiniteElementEquation;
    friend class daeEquationExecutionInfo;
    friend class daeDistributedEquationDomainInfo;
    friend class daeFunctionWithGradients;
    friend class daeOptimizationVariable;
    friend class daeVariableWrapper;
    friend class daeExternalFunction_t;
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
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual void DetectVariableTypesForExport(std::vector<const daeVariableType*>& ptrarrVariableTypes) const;

    void DistributeOnDomain(daeDomain& rDomain);

    size_t GetVariablesStartingIndex(void) const;
    void   SetVariablesStartingIndex(size_t nVariablesStartingIndex);

protected:
    virtual size_t	GetTotalNumberOfVariables(void) const               															= 0;
    virtual size_t	GetTotalNumberOfEquations(void) const                   														= 0;
    virtual void DeclareData(void)                                              													= 0;
    virtual void DeclareEquations(void)                                             												= 0;
    virtual void InitializeParameters(void)                                             											= 0;
    virtual void InitializeVariables(void)                                                  										= 0;
    virtual void InitializeEquations(void)                                                      									= 0;
    virtual void InitializeSTNs(void)                                                               								= 0;
    virtual void InitializeOnEventAndOnConditionActions(void)																		= 0;
    virtual void InitializeDEDIs(void)                                                                  		                    = 0;
    virtual void InitializeBlockIndexes(const std::map<size_t, size_t>& mapOverallIndex_BlockIndex)                                 = 0;

    virtual void CreatePortConnectionEquations(void)                                                        						= 0;
    virtual void PropagateDataProxy(boost::shared_ptr<daeDataProxy_t> pDataProxy)                               					= 0;
    virtual void PropagateGlobalExecutionContext(daeExecutionContext* pExecutionContext)                            				= 0;
    virtual void CollectAllSTNsAsVector(std::vector<daeSTN*>& ptrarrSTNs) const         											= 0;
    virtual void CollectEquationExecutionInfosFromModels(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const	= 0;
    virtual void CollectEquationExecutionInfosFromSTNs(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const	= 0;
    virtual void SetDefaultAbsoluteTolerances(void)                                                                                 = 0;
    virtual void SetDefaultInitialGuessesAndConstraints(void)                                                                       = 0;
    virtual void BuildExpressions(daeBlock* pBlock)                                                                                 = 0;
    virtual bool CheckDiscontinuities(void)                                                                                         = 0;
    virtual void AddExpressionsToBlock(daeBlock* pBlock)                                                                            = 0;
    virtual void ExecuteOnConditionActions(void)                                                                                    = 0;
    virtual void CreateOverallIndex_BlockIndex_VariableNameMap(std::map<size_t, std::pair<size_t, string> >& mapOverallIndex_BlockIndex_VariableName,
                                                               const std::map<size_t, size_t>& mapOverallIndex_BlockIndex)          = 0;
    virtual void UpdateEquations()                                     = 0;
    virtual void PropagateDomain(daeDomain& propagatedDomain)          = 0;
    virtual void PropagateParameter(daeParameter& propagatedParameter) = 0;

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
    daeVariableWrapper
*******************************************************************/
void daeGetVariableAndIndexesFromNode(adouble& a, daeVariable** variable, std::vector<size_t>& narrDomainIndexes);

class DAE_CORE_API daeVariableWrapper : public daeVariableWrapper_t
{
public:
    daeVariableWrapper();
    daeVariableWrapper(daeVariable& variable, std::string strName = "");
    daeVariableWrapper(adouble& a, std::string strName = "");
    virtual ~daeVariableWrapper(void);

public:
    void Initialize(daeVariable* pVariable, std::string strName, const std::vector<size_t>& narrDomainIndexes);
    string GetName(void) const;
    size_t GetOverallIndex(void) const;
    int GetVariableType(void) const;
    real_t GetValue(void) const;
    void SetValue(real_t value);

public:
    string GetVariableCanonicalNameWithDomains(void) const;

public:
    std::string			m_strName;
    daeVariable*		m_pVariable;
    std::vector<size_t>	m_narrDomainIndexes;
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
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual void DetectVariableTypesForExport(std::vector<const daeVariableType*>& ptrarrVariableTypes) const;

    void DistributeOnDomain(daeDomain& rDomain);

    size_t GetVariablesStartingIndex(void) const;
    void   SetVariablesStartingIndex(size_t nVariablesStartingIndex);

protected:
    virtual void DeclareData(void)                              = 0;
    virtual void InitializeParameters(void)                     = 0;
    virtual void InitializeVariables(void)                      = 0;
    virtual void SetDefaultAbsoluteTolerances(void)             = 0;
    virtual void SetDefaultInitialGuessesAndConstraints(void)	= 0;
    virtual void InitializeBlockIndexes(const std::map<size_t, size_t>& mapOverallIndex_BlockIndex) = 0;
    virtual void CreateOverallIndex_BlockIndex_VariableNameMap(std::map<size_t, std::pair<size_t, string> >& mapOverallIndex_BlockIndex_VariableName,
                                                               const std::map<size_t, size_t>& mapOverallIndex_BlockIndex) = 0;
    virtual void PropagateDomain(daeDomain& propagatedDomain)          = 0;
    virtual void PropagateParameter(daeParameter& propagatedParameter) = 0;

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
class daeOnConditionActions;
class daeSTN;
class DAE_CORE_API daeState : virtual public daeObject,
                              virtual public daeState_t
{
public:
    daeDeclareDynamicClass(daeState)
    daeState(void);
    virtual ~daeState(void);

public:
    virtual string	GetCanonicalName(void) const;
    virtual void	GetOnConditionActions(std::vector<daeOnConditionActions_t*>& ptrarrOnConditionActions);
    virtual void	GetOnEventActions(std::vector<daeOnEventActions_t*>& ptrarrOnEventActions);
    virtual void	GetEquations(std::vector<daeEquation_t*>& ptrarrEquations);
    virtual void	GetNestedSTNs(std::vector<daeSTN_t*>& ptrarrSTNs);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    bool CheckObject(std::vector<string>& strarrErrors) const;
    void Clone(const daeState& rObject);
    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    void CleanUpSetupData();
    void UpdateEquations();

    void AddEquation(daeEquation* pEquation);

//	size_t GetNumberOfEquations(void) const;
//	size_t GetNumberOfSTNs(void) const;

    void AddOnConditionAction(daeOnConditionActions& rOnConditionActions, const string& strName, string strDescription);
    void AddOnEventAction(daeOnEventActions& rOnEventAction, const string& strName, string strDescription);

    void CalcNonZeroElements(int& NNZ);
    void FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix);
    void ConnectOnEventActions(void);
    void DisconnectOnEventActions(void);

    const std::vector<daeEquation*>& Equations() const;
    const std::vector<daeSTN*>& NestedSTNs() const;
    const std::vector<daeOnEventActions*>& OnEventActions() const;
    const std::vector<daeOnConditionActions*>& OnConditionActions() const;

protected:
    void	Create(const string& strName, daeSTN* pSTN);
    void	InitializeOnEventAndOnConditionActions(void);
    void	InitializeDEDIs(void);

    void	AddNestedSTN(daeSTN* pSTN);

    daeSTN*	GetSTN(void) const;
    void	SetSTN(daeSTN* pSTN);

    void	AddIndexesFromAllEquations(std::vector< std::map<size_t, size_t> >& arrIndexes, size_t& nCurrentEquaton);

protected:
    daeSTN*									m_pSTN;
    daePtrVector<daeEquation*>				m_ptrarrEquations;
    daePtrVector<daeOnConditionActions*>	m_ptrarrOnConditionActions;
    daePtrVector<daeOnEventActions*>		m_ptrarrOnEventActions;
    daePtrVector<daeSTN*>					m_ptrarrSTNs;
    // daeEquation owns the pointers
    std::vector<daeEquationExecutionInfo*>  m_ptrarrEquationExecutionInfos;

    friend class daeIF;
    friend class daeSTN;
    friend class daeModel;
    friend class daeFiniteElementModel;
    friend class daeOnConditionActions;
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
    virtual string      GetCanonicalName(void) const;
    virtual void		GetStates(std::vector<daeState_t*>& ptrarrStates);
    virtual daeState_t*	GetActiveState(void);
    virtual void		SetActiveState(const std::string& strStateName);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    void CleanUpSetupData();
    void UpdateEquations(void);

    bool CheckObject(std::vector<string>& strarrErrors) const;
    virtual void Clone(const daeSTN& rObject);

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    virtual void		FinalizeDeclaration(void);
    virtual bool		CheckDiscontinuities(void);
    virtual void		ExecuteOnConditionActions(void);
    virtual size_t		GetNumberOfEquations(void) const;
    virtual daeState*	AddState(const string& strName);

    size_t			GetNumberOfStates(void) const;
    string			GetActiveState2(void) const;
    void			SetActiveState(daeState* pState);
    void			CalcNonZeroElements(int& NNZ);
    void			FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix);

    const std::vector<daeState*>& States() const;

    daeState*		GetParentState(void) const;
    void			SetParentState(daeState* pParentState);

    daeeSTNType		GetType(void) const;
    void			SetType(daeeSTNType eType);

    daeState*		FindState(long nID);
    daeState*		FindState(const string& strName);

    void CollectAllSTNs(std::map<dae::string, daeSTN_t*>& mapSTNs) const;

protected:
    virtual void	AddExpressionsToBlock(daeBlock* pBlock);

    void			InitializeOnEventAndOnConditionActions(void);
    void			InitializeDEDIs(void);

    bool			CheckState(daeState* pState);
    void			BuildExpressions(daeBlock* pBlock);
    void			CreateEquationExecutionInfo(void);
    void			CollectEquationExecutionInfos(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo);
    void			CollectVariableIndexes(std::map<size_t, size_t>& mapVariableIndexes);
    void            AddEquationsOverallIndexes(std::map<size_t, std::vector<size_t> >& mapIndexes);
    void			SetIndexesWithinBlockToEquationExecutionInfos(daeBlock* pBlock, size_t& nEquationIndex);
    void            BuildJacobianExpressions();
    void            CreateComputeStack(daeBlock* pBlock);
    uint32_t        GetComputeStackSize();

    void			CalculateResiduals(daeExecutionContext& EC);
    void			CalculateJacobian(daeExecutionContext& EC);
    void			CalculateSensitivityResiduals(daeExecutionContext& EC, const std::vector<size_t>& narrParameterIndexes);
    void			CalculateSensitivityParametersGradients(daeExecutionContext& EC, const std::vector<size_t>& narrParameterIndexes);

    size_t			GetNumberOfEquationsInState(daeState* pState) const;

    void			ReconnectStateTransitionsAndStates(void);
    void			AddIndexesFromAllEquations(std::vector< std::map<size_t, size_t> >& arrIndexes, size_t& nCurrentEquaton);

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
    virtual void		FinalizeDeclaration(void);
    virtual bool		CheckDiscontinuities(void);
    virtual void		ExecuteOnConditionActions(void);
    virtual daeState*	AddState(const string& strName);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    bool CheckObject(std::vector<string>& strarrErrors) const;
    virtual void Clone(const daeIF& rObject);
    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    daeState* CreateElse(const string& strName);

protected:
    virtual void	AddExpressionsToBlock(daeBlock* pBlock);
    void			CheckState(daeState* pState);
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
    virtual string GetCanonicalName(void) const;
    virtual void   GetDomainDefinitions(std::vector<daeDistributedEquationDomainInfo_t*>& arrDistributedEquationDomainInfo);

public:
    // Updates equation before every call to Residuals/Jacobian/Sens.Residuals. Does nothing by default.
    virtual void Update();

    virtual void CreateEquationExecutionInfos(daeModel* pModel, std::vector<daeEquationExecutionInfo*>& ptrarrEqnExecutionInfosCreated, bool bAddToTheModel);

    void	SetResidual(adouble res);
    adouble	GetResidual(void) const;

    real_t	GetScaling(void) const;
    void	SetScaling(real_t dScaling);

    bool GetSimplifyExpressions(void) const;
    void SetSimplifyExpressions(bool bSimplifyExpressions);

    bool GetCheckUnitsConsistency(void) const;
    void SetCheckUnitsConsistency(bool bCheck);

    bool GetBuildJacobianExpressions(void) const;
    void SetBuildJacobianExpressions(bool bBuildJacobianExpressions);

    virtual daeDEDI* DistributeOnDomain(daeDomain& rDomain, daeeDomainBounds eDomainBounds, const string& strName = string(""));
    virtual daeDEDI* DistributeOnDomain(daeDomain& rDomain, const std::vector<size_t>& narrDomainIndexes, const string& strName = string(""));
    virtual daeDEDI* DistributeOnDomain(daeDomain& rDomain, const size_t* pnarrDomainIndexes, size_t n, const string& strName = string(""));

    daeeEquationType GetEquationType(void) const;

    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    bool CheckObject(std::vector<string>& strarrErrors) const;
    void Clone(const daeEquation& rObject);
    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    void InitializeDEDIs(void);

    virtual size_t	GetNumberOfEquations(void) const;

    void GetEquationExecutionInfos(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfos) const;

    daeState* GetParentState() const;
    std::vector<daeDEDI*> GetDEDIs() const;

protected:
    void SetResidualValue(size_t nEquationIndex, real_t dResidual, daeBlock* pBlock);
    void SetJacobianItem(size_t nEquationIndex, size_t nVariableIndex, real_t dJacobValue, daeBlock* pBlock);

    void SaveNodeAsMathML(io::xmlTag_t* pTag, const string& strObjectName) const;
    void SaveNodeAsLatex(io::xmlTag_t* pTag, const string& strObjectName) const;

protected:
    real_t												m_dScaling;
    bool                                                m_bSimplifyExpressions;
    bool                                                m_bCheckUnitsConsistency;
    bool                                                m_bBuildJacobianExpressions;
    daeState*											m_pParentState;
    adNodePtr											m_pResidualNode;
    daePtrVector<daeDistributedEquationDomainInfo*>		m_ptrarrDistributedEquationDomainInfos;
// This vector is redundant - all EquationExecutionInfos already exist in models and states;
// However, daeEquation is the owner of pointers and responsible for freeing the memory.
    daePtrVector<daeEquationExecutionInfo*>				m_ptrarrEquationExecutionInfos;

    friend class daeSTN;
    friend class daeModel;
    friend class daeState;
    friend class daeEquationExecutionInfo;
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

    bool CheckObject(std::vector<string>& strarrErrors) const;
    void Clone(const daePortEqualityEquation& rObject);
    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    void Initialize(daeVariable* pLeft, daeVariable* pRight);
    size_t GetNumberOfEquations(void) const;

protected:
    daeVariable* m_pLeft;
    daeVariable* m_pRight;
};

/******************************************************************
    daeFiniteElementVariable
*******************************************************************/
struct daeFiniteElementVariableInfo
{
    std::string         m_strName;
    std::string         m_strDescription;
    unsigned int        m_nMultiplicity;
    unsigned int        m_nNumberOfDOFs;
    daeVariableType     m_VariableType;
};

/******************************************************************
    daeFiniteElementObjectInfo
*******************************************************************/
struct daeFiniteElementObjectInfo
{
    unsigned int                                m_nTotalNumberDOFs;
    unsigned int                                m_nNumberOfDOFsPerVariable;
    std::vector<daeFiniteElementVariableInfo>   m_VariableInfos;
};

/******************************************************************
    daeFiniteElementObject_t
*******************************************************************/
class daeFiniteElementModel;
class daeFiniteElementObject_t
{
public:
    virtual ~daeFiniteElementObject_t() {}

    virtual void SetModel(daeFiniteElementModel* pFEModel) = 0;

    virtual void AssembleSystem()       = 0;
    virtual void ClearAssembledSystem() = 0;
    virtual bool NeedsReAssembling()    = 0;
    virtual void ReAssembleSystem()     = 0;

    virtual void RowIndices(unsigned int row, std::vector<unsigned int>& narrIndices) const = 0;

    virtual daeFiniteElementObjectInfo GetObjectInfo() const = 0;

    virtual dae::daeMatrix<adouble>*                                                    Asystem() const = 0;
    virtual dae::daeMatrix<adouble>*                                                    Msystem() const = 0;
    virtual dae::daeArray<adouble>*                                                     Fload()   const = 0;
    virtual const std::map< unsigned int, std::vector< std::pair<adouble,adouble> > >*  SurfaceIntegrals() const = 0;
    virtual const std::vector< std::pair<adouble,adouble> >*                            VolumeIntegrals()  const = 0;
};

/******************************************************************
    daeFiniteElementModel
******************************************************************/
class daeFiniteElementEquation;
class DAE_CORE_API daeFiniteElementModel : public daeModel
{
public:
    daeDeclareDynamicClass(daeFiniteElementModel)
    daeFiniteElementModel(std::string               strName,
                          daeModel*                 pModel,
                          std::string               strDescription,
                          daeFiniteElementObject_t* fe);

public:
    void DeclareEquations(void);
    void DeclareEquationsForWeakForm(void);
    void UpdateEquations();

protected:
    daeFiniteElementObject_t*               m_fe;
    daeDomain                               m_omega;
    daePtrVector<daeDomain*>                m_ptrarrFESubDomains;
    daePtrVector<daeVariable*>              m_ptrarrFEVariables;
    boost::shared_ptr< daeMatrix<adouble> > m_Aij; // Stiffness matrix
    boost::shared_ptr< daeMatrix<adouble> > m_Mij; // Mass matrix
    boost::shared_ptr< daeArray<adouble> >  m_Fi;  // Load vector

    friend class daeFiniteElementEquation;
};

/******************************************************************
    daeFiniteElementEquation
*******************************************************************/
class DAE_CORE_API daeFiniteElementEquation : public daeEquation
{
public:
    daeDeclareDynamicClass(daeFiniteElementEquation)
    daeFiniteElementEquation(const daeFiniteElementModel& feModel, const std::vector<daeVariable*>& arrVariables, size_t startRow, size_t endRow);
    virtual ~daeFiniteElementEquation(void);

public:
    void CreateEquationExecutionInfos(daeModel* pModel, std::vector<daeEquationExecutionInfo*>& ptrarrEqnExecutionInfosCreated, bool bAddToTheModel);
    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    virtual daeDEDI* DistributeOnDomain(daeDomain& rDomain, daeeDomainBounds eDomainBounds, const string& strName = string(""));
    virtual daeDEDI* DistributeOnDomain(daeDomain& rDomain, const std::vector<size_t>& narrDomainIndexes, const string& strName = string(""));
    virtual daeDEDI* DistributeOnDomain(daeDomain& rDomain, const size_t* pnarrDomainIndexes, size_t n, const string& strName = string(""));

public:
    const daeFiniteElementModel&     m_FEModel;
    const std::vector<daeVariable*>& m_ptrarrVariables;
    const size_t                     m_startRow;
    const size_t                     m_endRow;

    friend class daeModel;
    friend class daeFiniteElementModel;
    friend class daeEquationExecutionInfo;
};


/******************************************************************
    daeODEModel
******************************************************************/
class daeODEEquation;
class DAE_CORE_API daeODEModel : public daeModel
{
public:
    daeDeclareDynamicClass(daeODEModel)
    daeODEModel(std::string strName, daeModel* pModel, std::string strDescription);

public:
    bool CheckObject(std::vector<string>& strarrErrors) const;

    friend class daeODEEquation;
};

/******************************************************************
    daeODEEquation
*******************************************************************/
class DAE_CORE_API daeODEEquation : public daeEquation
{
public:
    daeDeclareDynamicClass(daeODEEquation)
    daeODEEquation();
    virtual ~daeODEEquation(void);

public:
    void Open(io::xmlTag_t* pTag);
    void Save(io::xmlTag_t* pTag) const;
    void OpenRuntime(io::xmlTag_t* pTag);
    void SaveRuntime(io::xmlTag_t* pTag) const;

    void CreateEquationExecutionInfos(daeModel* pModel, std::vector<daeEquationExecutionInfo*>& ptrarrEqnExecutionInfosCreated, bool bAddToTheModel);
    bool CheckObject(std::vector<string>& strarrErrors) const;

    void	SetODEVariable(adouble odeVariable);
    adouble	GetODEVariable(void) const;

protected:
    // adSetupVariableNode in dVar/dt = residual
    adNodePtr m_pODEVariableNode;

    friend class daeModel;
    friend class daeODEModel;
    friend class daeEquationExecutionInfo;
};


/******************************************************************
    daeNodeSaveAsContext
*******************************************************************/
class DAE_CORE_API daeNodeSaveAsContext
{
public:
    daeNodeSaveAsContext(const daeModel* pModel = NULL) : m_pModel(pModel)
    {
    }
    const daeModel* m_pModel;
};


/******************************************************************
    daeOptimizationVariable
*******************************************************************/
class DAE_CORE_API daeOptimizationVariable : public daeOptimizationVariable_t,
                                             public daeRuntimeCheck_t,
                                             public daeExportable_t
{
public:
    daeDeclareDynamicClass(daeOptimizationVariable)
    daeOptimizationVariable(void);
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

    real_t GetScaling() const;
    void   SetScaling(real_t scaling);

    unit GetUnits() const;

    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

protected:
    daeVariable*					m_pVariable;
    real_t							m_dLB;
    real_t							m_dUB;
    real_t							m_dDefaultValue;
    daeeOptimizationVariableType	m_eType;
    size_t							m_nOptimizationVariableIndex; // index in the array of opt. variables: 0 - Nopt.vars.
    std::vector<size_t>				m_narrDomainIndexes;
    real_t                          m_dScaling;
};

/******************************************************************
    daeFunctionWithGradients
*******************************************************************/
class DAE_CORE_API daeFunctionWithGradients : virtual public daeFunctionWithGradients_t,
                                              virtual public daeRuntimeCheck_t,
                                              virtual public daeExportable_t
{
public:
    daeDeclareDynamicClass(daeFunctionWithGradients)
    daeFunctionWithGradients(void);
    daeFunctionWithGradients(daeModel* pModel,
                             daeDAESolver_t* pDAESolver,
                             real_t abstol,
                             const string& strVariableName,
                             const string& strEquationName,
                             const string& strDescription);
    virtual ~daeFunctionWithGradients(void);

public:
    bool CheckObject(std::vector<string>& strarrErrors) const;

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;

    bool IsLinear(void) const;

    std::string GetName(void) const;
    real_t GetValue(void) const;

    void GetGradients(const daeMatrix<real_t>& matSensitivities, real_t* gradients, size_t Nparams) const;
    void GetGradients(real_t* gradients, size_t Nparams) const;

    void GetOptimizationVariableIndexes(std::vector<size_t>& narrOptimizationVariablesIndexes) const;
    size_t GetNumberOfOptimizationVariables(void) const;

    real_t GetAbsTolerance() const;
    void   SetAbsTolerance(real_t abstol);

    real_t GetScaling() const;
    void   SetScaling(real_t scaling);

    void	 SetResidual(adouble res);
    adouble	 GetResidual(void) const;

    void Initialize(const std::vector< boost::shared_ptr<daeOptimizationVariable> >& arrOptimizationVariables, daeBlock_t* pBlock);

    void RemoveEquationFromModel(void);

protected:
    daeMatrix<real_t>& GetSensitivitiesMatrix(void) const;

protected:
    daeModel*						m_pModel;
    daeDAESolver_t*					m_pDAESolver;
    boost::shared_ptr<daeVariable>	m_pVariable;
    daeEquation*					m_pEquation;
    daeEquationExecutionInfo*		m_pEquationExecutionInfo;
    size_t							m_nEquationIndexInBlock;
    size_t							m_nVariableIndexInBlock;
    size_t							m_nNumberOfOptimizationVariables;
    std::vector<size_t>				m_narrOptimizationVariablesIndexes;
    real_t                          m_dScaling;
};

/******************************************************************
    daeObjectiveFunction
*******************************************************************/
class DAE_CORE_API daeObjectiveFunction : virtual public daeFunctionWithGradients,
                                          virtual public daeObjectiveFunction_t
{
public:
    daeDeclareDynamicClass(daeObjectiveFunction)
    daeObjectiveFunction(void);
    daeObjectiveFunction(daeModel* pModel,
                         daeDAESolver_t* pDAESolver,
                         real_t abstol,
                         size_t nEquationIndex,
                         const string& strDescription);
    virtual ~daeObjectiveFunction(void);
};

/******************************************************************
    daeOptimizationConstraint
*******************************************************************/
class DAE_CORE_API daeOptimizationConstraint : virtual public daeFunctionWithGradients,
                                               virtual public daeOptimizationConstraint_t
{
public:
    daeDeclareDynamicClass(daeOptimizationConstraint)
    daeOptimizationConstraint(void);
    daeOptimizationConstraint(daeModel* pModel,
                              daeDAESolver_t* pDAESolver,
                              bool bIsInequalityConstraint,
                              real_t abstol,
                              size_t nEquationIndex,
                              const string& strDescription);
    virtual ~daeOptimizationConstraint(void);

public:
    void               SetType(daeeConstraintType value);
    daeeConstraintType GetType(void) const;

protected:
    daeeConstraintType m_eConstraintType;
};

/******************************************************************
    daeMeasuredVariable
*******************************************************************/
class DAE_CORE_API daeMeasuredVariable : virtual public daeFunctionWithGradients,
                                         virtual public daeMeasuredVariable_t
{
public:
    daeDeclareDynamicClass(daeMeasuredVariable)
    daeMeasuredVariable(void);
    daeMeasuredVariable(daeModel* pModel,
                        daeDAESolver_t* pDAESolver,
                        real_t abstol,
                        size_t nEquationIndex,
                        const string& strDescription);
    virtual ~daeMeasuredVariable(void);
};

/******************************************************************
    Find functions
*******************************************************************/
daeDomain*		FindDomain(const daeDomain* pSource, daeModel* pParentModel);
daeEventPort*	FindEventPort(const daeEventPort* pSource, daeModel* pParentModel);
void			FindDomains(const std::vector<daeDomain*>& ptrarrSource, std::vector<daeDomain*>& ptrarrDestination, daeModel* pParentModel);

/*********************************************************************************************
    daeExternalFunction_t
**********************************************************************************************/
typedef boost::variant<adouble, adouble_array>						  daeExternalFunctionArgument_t;
typedef std::map<std::string, daeExternalFunctionArgument_t>		  daeExternalFunctionArgumentMap_t;
typedef boost::variant<adouble, std::vector<adouble> >				  daeExternalFunctionArgumentValue_t;
typedef std::map<std::string, daeExternalFunctionArgumentValue_t>	  daeExternalFunctionArgumentValueMap_t;
typedef boost::variant<adNodePtr, adNodeArrayPtr >	                  daeExternalFunctionNode_t;
typedef std::map<std::string, daeExternalFunctionNode_t>			  daeExternalFunctionNodeMap_t;

class DAE_CORE_API daeExternalFunction_t : public daeObject
{
public:
    daeExternalFunction_t(const string& strName, daeModel* pModel, const unit& units);
    virtual ~daeExternalFunction_t(void);

public:
    void									InitializeArguments(const daeExecutionContext* pExecutionContext);
    void									SetArguments(const daeExternalFunctionArgumentMap_t& mapArguments);
    const daeExternalFunctionArgumentMap_t&	GetArgumentNodes(void) const;
    const daeExternalFunctionNodeMap_t&		GetSetupArgumentNodes(void) const;
    unit									GetUnits(void) const;

protected:
    unit							 m_Unit;
    daeExternalFunctionNodeMap_t	 m_mapSetupArgumentNodes;
    daeExternalFunctionArgumentMap_t m_mapArgumentNodes;
};

/*********************************************************************************************
    daeScalarExternalFunction
**********************************************************************************************/
class DAE_CORE_API daeScalarExternalFunction : public daeExternalFunction_t
{
public:
    daeScalarExternalFunction(const string& strName, daeModel* pModel, const unit& units);
    virtual ~daeScalarExternalFunction(void);

public:
    virtual adouble	Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const;
    virtual adouble	operator() (void);
};

/*********************************************************************************************
    daeLinearInterpolationFunction
**********************************************************************************************/
class DAE_CORE_API daeLinearInterpolationFunction : public daeScalarExternalFunction
{
public:
    daeLinearInterpolationFunction(const string& strName,
                                   daeModel* pModel,
                                   const unit& units);
    virtual ~daeLinearInterpolationFunction(void);

public:
    virtual adouble	Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const;

    void InitData(const std::vector<real_t> &x, const std::vector<real_t> &y, adouble &arg);

protected:
    std::vector<real_t> x_arr;
    std::vector<real_t> y_arr;
};

/*********************************************************************************************
    daeCTypesExternalFunction
**********************************************************************************************/
class DAE_CORE_API daeCTypesExternalFunction : public daeScalarExternalFunction
{
public:
    class adouble_c
    {
    public:
        adouble_c(real_t val, real_t deriv)
        {
            value      = val;
            derivative = deriv;
        }

        real_t value;
        real_t derivative;
    };
    typedef adouble_c (*external_lib_function)(const adouble_c[], const char*[], int);

    daeCTypesExternalFunction(const string& strName,
                           daeModel* pModel,
                           const unit& units,
                           external_lib_function fun_ptr);
    virtual ~daeCTypesExternalFunction(void);

public:
    virtual adouble	Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const;

protected:
    external_lib_function m_external_lib_function;
};

/*********************************************************************************************
    daeVectorExternalFunction
**********************************************************************************************/
class DAE_CORE_API daeVectorExternalFunction : public daeExternalFunction_t
{
public:
    daeVectorExternalFunction(const string& strName, daeModel* pModel, const unit& units, size_t nNumberofArguments);
    virtual ~daeVectorExternalFunction(void);

public:
    virtual std::vector<adouble> Calculate(daeExternalFunctionArgumentValueMap_t& mapValues) const;
    virtual adouble_array operator() (void);
    virtual size_t GetNumberOfResults(void) const;

protected:
    const size_t m_nNumberofArguments;
};


/*********************************************************************************************
    daeCapeOpenThermoPhysicalPropertyPackage
**********************************************************************************************/
using dae::tpp::daeeThermoPackagePropertyType;
using dae::tpp::daeeThermoPackagePhase;
using dae::tpp::daeeThermoPackageBasis;
using dae::tpp::daeThermoPhysicalPropertyPackage_t;
using dae::tpp::eMole;

class DAE_CORE_API daeThermoPhysicalPropertyPackage : public daeObject
{
public:
    daeDeclareDynamicClass(daeThermoPhysicalPropertyPackage)
    daeThermoPhysicalPropertyPackage(const string& strName, daeModel* pModel, const string& strDescription = "");
    virtual ~daeThermoPhysicalPropertyPackage(void);

public:
    void Load_CoolProp_TPP(const std::vector<std::string>& strarrCompoundIDs,
                           const std::vector<std::string>& strarrCompoundCASNumbers,
                           const std::map<std::string,daeeThermoPackagePhase>& mapAvailablePhases,
                           daeeThermoPackageBasis defaultBasis = eMole,
                           const std::map<std::string,std::string>& mapOptions = std::map<std::string,std::string>());

    void Load_CapeOpen_TPP(const std::string& strPackageManager,
                           const std::string& strPackageName,
                           const std::vector<std::string>& strarrCompoundIDs,
                           const std::vector<std::string>& strarrCompoundCASNumbers,
                           const std::map<std::string,daeeThermoPackagePhase>& mapAvailablePhases,
                           daeeThermoPackageBasis defaultBasis = eMole,
                           const std::map<std::string,std::string>& mapOptions = std::map<std::string,std::string>());

public:
    std::string GetTPPName();

    adouble GetCompoundConstant(const std::string& property, const std::string& compound);

    adouble GetTDependentProperty(const std::string& property,
                                  const adouble& T,
                                  const std::string& compound);

    adouble GetPDependentProperty(const std::string& property,
                                  const adouble& P,
                                  const std::string& compound);

    adouble CalcSinglePhaseScalarProperty(const std::string& property,
                                          const adouble& P,
                                          const adouble& T,
                                          const adouble_array& X,
                                          const std::string& phase,
                                          daeeThermoPackageBasis basis = dae::tpp::eMole);

    adouble_array CalcSinglePhaseVectorProperty(const std::string& property,
                                                const adouble& P,
                                                const adouble& T,
                                                const adouble_array& X,
                                                const std::string& phase,
                                                daeeThermoPackageBasis basis = dae::tpp::eMole);

    adouble CalcTwoPhaseScalarProperty(const std::string& property,
                                       const adouble& P1,
                                       const adouble& T1,
                                       const adouble_array& X1,
                                       const std::string& phase1,
                                       const adouble& P2,
                                       const adouble& T2,
                                       const adouble_array& X2,
                                       const std::string& phase2,
                                       daeeThermoPackageBasis basis = dae::tpp::eMole);

     adouble_array CalcTwoPhaseVectorProperty(const std::string& property,
                                              const adouble& P1,
                                              const adouble& T1,
                                              const adouble_array& X1,
                                              const std::string& phase1,
                                              const adouble& P2,
                                              const adouble& T2,
                                              const adouble_array& X2,
                                              const std::string& phase2,
                                              daeeThermoPackageBasis basis = dae::tpp::eMole);

    unit GetUnits(const std::string& property, daeeThermoPackageBasis basis = dae::tpp::eMole);

public:
    daeThermoPhysicalPropertyPackage_t* m_package;
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

//#include "inlines_equation.h"
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
