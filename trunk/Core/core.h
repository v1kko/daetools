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
#ifndef DAE_CORE_H
#define DAE_CORE_H

#include "definitions.h"
#include "log.h"
using namespace dae::logging;

namespace dae 
{
namespace core 
{
const int cnNormal		 = 0;
const int cnDifferential = 1;
const int cnFixed		 = 2;

enum daeeDomainType
{
	eDTUnknown = 0,
	eArray,
	eDistributed
};
		
enum daeeParameterType
{
	ePTUnknown = 0,
	eReal,
	eInteger,
	eBool
};

enum daeeDomainBounds
{
	eDBUnknown = 0,
	eOpenOpen,
	eOpenClosed,
	eClosedOpen,
	eClosedClosed,
	eLowerBound,
	eUpperBound,
	eFunctor,
	eCustomBound
};

enum daeeDiscretizationMethod
{
	eDMUnknown = 0,
	eCFDM,
	eFFDM,
	eBFDM,
	eCustomDM
};

enum daeeInitialConditionMode
{
	eICTUnknown = 0,
	eAlgebraicValuesProvided,
	eDifferentialValuesProvided,
	eQuasySteadyState
};

enum daeeEquationCalculationMode
{
	eECMUnknown = 0,
	eGatherInfo,
	eCalculate,
	eCreateFunctionsIFsSTNs,
	eCalculateJacobian,
	eCalculateSensitivityResiduals,
	eCalculateSensitivityParametersGradients
};

enum daeeModelType
{
	eMTUnknown = 0,
	eSteadyStateModel,
	eDynamicModel
};

enum daeeEquationDefinitionMode
{
	eEDMUnknown = 0,
	eMemberFunctionPointer,
	eResidualNode
};

enum daeeEquationEvaluationMode
{
	eEEMUnknown = 0,
	eResidualNodeEvaluation,
	eFunctionEvaluation,
	eCommandStackEvaluation
};

enum daeePortType
{
	eUnknownPort = 0,
	eInletPort,
	eOutletPort
};

enum daeeActionType
{
	eUnknownAction = 0,
	eChangeState,
	eSendEvent,
	eReAssignOrReInitializeVariable
};

enum daeeFunctionType
{
	eFTUnknown = 0,
	eUnary,
	eBinary
};

enum daeeUnaryFunctions
{
	eUFUnknown = 0,
	eSign,
	eSqrt,
	eExp,
	eLog,
	eLn,
	eAbs,
	eSin,
	eCos,
	eTan,
	eArcSin,
	eArcCos,
	eArcTan, 
	eCeil,
	eFloor
};

enum daeeBinaryFunctions
{
	eBFUnknown = 0,
	ePlus,
	eMinus,
	eMulti,
	eDivide,
	ePower,
	eMin,
	eMax
};

enum daeeSpecialUnaryFunctions
{
	eSUFUnknown = 0,
	eSum,
	eProduct,
	eMinInArray,
	eMaxInArray,
	eAverage
};

enum daeeIntegralFunctions
{
	eIFUnknown = 0,
	eSingleIntegral
};

enum daeeLogicalUnaryOperator
{
	eUOUnknown = 0,
	eNot
};

enum daeeLogicalBinaryOperator
{
	eBOUnknown = 0,
	eAnd,
	eOr
};

enum daeeConditionType
{
	eCTUnknown = 0,
	eNotEQ, // !=
	eEQ,    // ==
	eGT,    // >
	eGTEQ,  // >=
	eLT,    // <
	eLTEQ   // <=
};

enum daeeDiscontinuityType
{
	eDCTUnknown = 0,
	eGlobalDiscontinuity,
	eModelDiscontinuity,
	eModelDiscontinuityWithDataChange,
	eNoDiscontinuity	
};

enum daeeStopCriterion
{
	eStopAtGlobalDiscontinuity = 0,
	eStopAtModelDiscontinuity,
	eDoNotStopAtDiscontinuity
};

enum daeeIntegrationMode
{
	eIMUnknown = 0,
	eContinueFor,
	eContinueUntil
};

enum daeeSTNType
{
	eSTNTUnknown = 0,
	eSTN,
	eIF
};

enum daeeDomainIndexType
{
	eDITUnknown = 0,
	eConstantIndex, 
	eDomainIterator,
	eIncrementedDomainIterator
};

enum daeeRangeType
{
	eRaTUnknown = 0,
	eRangeDomainIndex, 
	eRange
};

enum daeIndexRangeType
{
	eIRTUnknown = 0,
	eAllPointsInDomain,
	eRangeOfIndexes,
	eCustomRange
};

enum daeeConstraintType
{
	eInequalityConstraint = 0,
	eEqualityConstraint	
};

enum daeeOptimizationVariableType
{
	eIntegerVariable = 0,
	eBinaryVariable,
	eContinuousVariable
};

enum daeeModelLanguage
{
	eMLNone = 0,
	eCDAE,
	ePYDAE
};

		
/******************************************************************
	daeObject_t
*******************************************************************/
class daeModel_t;
class daeObject_t
{
public:
	virtual ~daeObject_t(void){}

public:	
	virtual string			GetCanonicalName(void) const									= 0;
	virtual string			GetName(void) const												= 0;
	virtual string			GetDescription(void) const										= 0;
	virtual daeModel_t*		GetModel(void) const											= 0;
	virtual void			LogMessage(const string& strMessage, size_t nSeverity) const	= 0;
};

/******************************************************************
	daeVariableType_t
*******************************************************************/
class daeVariableType_t
{
public:
	virtual ~daeVariableType_t(void){}

	virtual string	GetName(void) const						= 0;
	virtual void	SetName(string strName)					= 0;
	virtual real_t	GetLowerBound(void) const				= 0;
	virtual void	SetLowerBound(real_t dValue)			= 0;
	virtual real_t	GetUpperBound(void) const				= 0;
	virtual void	SetUpperBound(real_t dValue)			= 0;
	virtual real_t	GetInitialGuess(void) const				= 0;
	virtual void	SetInitialGuess(real_t dValue)			= 0;
	virtual string	GetUnits(void) const					= 0;
	virtual void	SetUnits(string strName)				= 0;
	virtual real_t	GetAbsoluteTolerance(void) const		= 0;
	virtual void	SetAbsoluteTolerance(real_t dTolerance)	= 0;
};

/******************************************************************
	daeDomain_t
*******************************************************************/
class daeDomain_t : virtual public daeObject_t
{
public:
// Common for both Discrete and Distributed domains
	virtual daeeDomainType				GetType(void) const						= 0;
	virtual size_t						GetNumberOfIntervals(void) const		= 0;
	virtual size_t						GetNumberOfPoints(void) const			= 0;
	virtual real_t						GetPoint(size_t nIndex) const			= 0;

// Only for Distributed domains
	virtual daeeDiscretizationMethod	GetDiscretizationMethod(void) const		= 0;
	virtual size_t						GetDiscretizationOrder(void) const		= 0;
	virtual real_t						GetLowerBound(void) const				= 0;
	virtual real_t						GetUpperBound(void) const				= 0;

	virtual void						GetPoints(std::vector<real_t>& darrPoints) const = 0;
	virtual void						SetPoints(const std::vector<real_t>& darrPoints) = 0;
};


/******************************************************************
	daeDistributedEquationDomainInfo_t
*******************************************************************/
class daeDistributedEquationDomainInfo_t : virtual public daeObject_t
{
public:
	virtual daeDomain_t*		GetDomain(void) const		= 0;
	virtual daeeDomainBounds	GetDomainBounds(void) const = 0; 
	virtual void				GetDomainPoints(std::vector<size_t>& narrDomainPoints) const = 0;
};


/******************************************************************
	daeParameter_t
*******************************************************************/
class daeParameter_t : virtual public daeObject_t
{
public:
	virtual daeeParameterType	GetParameterType(void) const							= 0;
	virtual void				GetDomains(std::vector<daeDomain_t*>& ptrarrDomains)	= 0;
	virtual real_t*				GetValuePointer(void)									= 0;

	virtual void	SetValue(real_t value)																									= 0;
	virtual void	SetValue(size_t nD1, real_t value)																						= 0;
	virtual void	SetValue(size_t nD1, size_t nD2, real_t value)																			= 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, real_t value)																= 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value)													= 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value)										= 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value)							= 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value)				= 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value)	= 0;

	virtual real_t	GetValue(void)																								= 0;
	virtual real_t	GetValue(size_t nD1)																						= 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2) 																			= 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3) 																= 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4) 													= 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5) 										= 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6)							= 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7)				= 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8)	= 0;
};	

/******************************************************************
	daeVariable_t
*******************************************************************/
class daeVariable_t : virtual public daeObject_t
{
public:
	virtual daeVariableType_t*		GetVariableType(void)									= 0;
	virtual void					GetDomains(std::vector<daeDomain_t*>& ptrarrDomains)	= 0;

	virtual size_t					GetNumberOfPoints(void)	const		= 0;
	virtual real_t*					GetValuePointer(void) const			= 0;

	virtual bool					GetReportingOn(void) const			= 0;
	virtual void					SetReportingOn(bool bOn)			= 0;

	virtual void	SetValue(real_t value) = 0;
	virtual void	SetValue(size_t nD1, real_t value) = 0;
	virtual void	SetValue(size_t nD1, size_t nD2, real_t value) = 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, real_t value) = 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value) = 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value) = 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value) = 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value) = 0;
	virtual void	SetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value) = 0;

	virtual real_t	GetValue(void) = 0;
	virtual real_t	GetValue(size_t nD1) = 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2) = 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3) = 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4) = 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5) = 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6) = 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7) = 0;
	virtual real_t	GetValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8) = 0;

	virtual void	AssignValue(real_t value) = 0;
	virtual void	AssignValue(size_t nD1, real_t value) = 0;
	virtual void	AssignValue(size_t nD1, size_t nD2, real_t value) = 0;
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, real_t value) = 0;
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value) = 0;
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value) = 0;
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value) = 0;
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value) = 0;
	virtual void	AssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value) = 0;

	virtual void	ReAssignValue(real_t value) = 0;
	virtual void	ReAssignValue(size_t nD1, real_t value) = 0;
	virtual void	ReAssignValue(size_t nD1, size_t nD2, real_t value) = 0;
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, real_t value) = 0;
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t value) = 0;
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t value) = 0;
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t value) = 0;
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t value) = 0;
	virtual void	ReAssignValue(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t value) = 0;

	virtual void	SetInitialGuess(real_t dInitialGuess) = 0;
	virtual void	SetInitialGuess(size_t nD1, real_t dInitialGuesses) = 0;
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, real_t dInitialGuesses)	= 0;
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, real_t dInitialGuesses) = 0;
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialGuesses) = 0;
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialGuesses) = 0;
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialGuesses) = 0;
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialGuesses) = 0;
	virtual void	SetInitialGuess(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialGuesses) = 0;
// Set them for all indexes
	virtual void	SetInitialGuesses(real_t dInitialGuesses) = 0;

	virtual void	SetInitialCondition(real_t dInitialCondition)	= 0;
	virtual void	SetInitialCondition(size_t nD1, real_t dInitialCondition)	= 0;
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, real_t dInitialCondition)	= 0;
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, real_t dInitialCondition)	= 0;
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialCondition)	= 0;
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialCondition)	= 0;
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialCondition) = 0;
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialCondition) = 0;
	virtual void	SetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialCondition) = 0;

	virtual void	ReSetInitialCondition(real_t dInitialCondition)	= 0;
	virtual void	ReSetInitialCondition(size_t nD1, real_t dInitialCondition)	= 0;
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, real_t dInitialCondition)	= 0;
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, real_t dInitialCondition)	= 0;
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, real_t dInitialCondition)	= 0;
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, real_t dInitialCondition)	= 0;
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, real_t dInitialCondition) = 0;
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, real_t dInitialCondition) = 0;
	virtual void	ReSetInitialCondition(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8, real_t dInitialCondition) = 0;

	virtual void	SetAbsoluteTolerances(real_t dAbsTolerances) = 0;

	virtual real_t	TimeDerivative(void) = 0;
	virtual real_t	TimeDerivative(size_t nD1) = 0;
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2) = 0;
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3) = 0;
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4) = 0;
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5) = 0;
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6) = 0;
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7) = 0;
	virtual real_t	TimeDerivative(size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8) = 0;

	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1) = 0;
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2) = 0;
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3) = 0;
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4) = 0;
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5) = 0;
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6) = 0;
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7) = 0;
	virtual real_t	PartialDerivative1(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8) = 0;

	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1) = 0;
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2) = 0;
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3) = 0;
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4) = 0;
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5) = 0;
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6) = 0;
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7) = 0;
	virtual real_t	PartialDerivative2(const daeDomain_t& rDomain, size_t nD1, size_t nD2, size_t nD3, size_t nD4, size_t nD5, size_t nD6, size_t nD7, size_t nD8) = 0;
};	

/******************************************************************
	daeEquation_t
*******************************************************************/
class daeEquation_t : virtual public daeObject_t
{
public:
	virtual void GetDomainDefinitions(std::vector<daeDistributedEquationDomainInfo_t*>& arrDistributedEquationDomainInfo) = 0;
};

/******************************************************************
	daePort_t
*******************************************************************/
class daePort_t : virtual public daeObject_t
{
public:
	virtual daeePortType	GetType(void) const												= 0;
	virtual void			GetDomains(std::vector<daeDomain_t*>& ptrarrDomains)			= 0;
	virtual void			GetVariables(std::vector<daeVariable_t*>& ptrarrVariables)		= 0;
	virtual void			GetParameters(std::vector<daeParameter_t*>& ptrarrParameters)	= 0;
	virtual void			SetReportingOn(bool bOn)										= 0;
	
	virtual daeObject_t*	FindObject(std::string& strName)									= 0;
	virtual daeObject_t*	FindObjectFromRelativeName(std::string& strRelativeName)			= 0;
	virtual daeObject_t*	FindObjectFromRelativeName(std::vector<std::string>& strarrNames)	= 0;
};

/******************************************************************
	daeEventPort_t
*******************************************************************/
/* It derives from daeSubject<daeEventPort_t> to allow daeOnEventActions_t to observe daeEventPort_t
   It derives from daeObserver<daeEventPort_t> to allow inlet ports to observe outlet daeEventPort_t ports
   How the events are propagated in the following scenario: 
     Outlet event ports are attached to inlet event ports, and actions are attached to inlet event ports
		 [daeOnEventActions] ------ > [inlet daeEventPort] --------> [outlet daeEventport]
	   
	   When outlet port's SendEvent() is called it calls it's Notify() function.   
	   Notify() function calls Update() function in the inlet port which then calls Update() in OnEventActions.
																	  
													  <observer> -------< Notify(data) ------------ <subject>
														  |                                      [EventPort: out]
														  |
														  | observer.Update(data) calls Notify(data)
														  |
														  v
		<observer>-----------< Notify(data) ----------<subject>                         
	 [OnEventActions]                              [EventPort: in]                     
			|
			| Update(data) calls Execute(data)
			v

   The function daeEventPort_t::SendEvent() is used by daeActions to trigger the event 
   (it calls the function Notify(data))
*/
class daeAction_t;
class daeEventPort_t : virtual public daeObject_t,
                       virtual public daeSubject<daeEventPort_t>,
                       virtual public daeObserver<daeEventPort_t>
{
public:
	virtual daeePortType	GetType(void) const			= 0;
	virtual void			SetType(daeePortType eType)	= 0;
	virtual void			SendEvent(real_t data)		= 0;
};

/******************************************************************
	daeAction_t
*******************************************************************/
class daeAction_t : virtual public daeObject_t
{
public:
	virtual daeeActionType	GetType(void) const	= 0;
	virtual void			Execute(void)  		= 0;
};

/******************************************************************
	daeOnEventAction_t
*******************************************************************/
class daeOnEventActions_t : virtual public daeObject_t,
                            virtual public daeObserver<daeEventPort_t>
{
public:
	virtual void Execute(void) = 0;
};

/******************************************************************
	daePortConnection_t
*******************************************************************/
class daePortConnection_t : virtual public daeObject_t
{
public:
	virtual daePort_t*	GetPortFrom(void) const = 0;
	virtual daePort_t*	GetPortTo(void) const	= 0;
};

/******************************************************************
	daeState_t
*******************************************************************/
class daeSTN_t;
class daeStateTransition_t;
class daeState_t : virtual public daeObject_t
{
public:
	virtual void	GetStateTransitions(std::vector<daeStateTransition_t*>& ptrarrStateTransitions)	= 0;
	virtual void	GetEquations(std::vector<daeEquation_t*>& ptrarrEquations)						= 0;
	virtual void	GetNestedSTNs(std::vector<daeSTN_t*>& ptrarrSTNs)								= 0;
};

/******************************************************************
	daeStateTransition_t
*******************************************************************/
class daeStateTransition_t : virtual public daeObject_t
{
public:
//	virtual daeState_t*		GetStateTo(void) const	 = 0;
//	virtual daeState_t*		GetStateFrom(void) const = 0;
	virtual void GetActions(std::vector<daeAction_t*>& ptrarrActions) const	= 0;
	virtual void ExecuteActions(void)										= 0;
};

/******************************************************************
	daeSTN_t
*******************************************************************/
class daeSTN_t : virtual public daeObject_t
{
public:
	virtual void		GetStates(std::vector<daeState_t*>& ptrarrStates)	= 0;
	virtual daeState_t*	GetActiveState(void)								= 0;
};

/******************************************************************
	daePortArray_t
*******************************************************************/
class daePortArray_t : virtual public daeObject_t
{
public:
	virtual size_t	GetDimensions(void) const							 = 0;
	virtual void	GetDomains(std::vector<daeDomain_t*>& ptrarrDomains) = 0;

	virtual void	SetReportingOn(bool bOn)   = 0;

	virtual daePort_t* GetPort(size_t n1)												= 0;
	virtual daePort_t* GetPort(size_t n1, size_t n2)									= 0;
	virtual daePort_t* GetPort(size_t n1, size_t n2, size_t n3)							= 0;
	virtual daePort_t* GetPort(size_t n1, size_t n2, size_t n3, size_t n4)				= 0;
	virtual daePort_t* GetPort(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)	= 0;
};

/******************************************************************
	daeModelArray_t
*******************************************************************/
class daeModelArray_t : virtual public daeObject_t
{
public:
	virtual size_t	GetDimensions(void) const								= 0;
	virtual void	GetDomains(std::vector<daeDomain_t*>& ptrarrDomains)	= 0;

	virtual void	SetReportingOn(bool bOn)   = 0;

	virtual daeModel_t* GetModel(size_t n1)												= 0;
	virtual daeModel_t* GetModel(size_t n1, size_t n2)									= 0;
	virtual daeModel_t* GetModel(size_t n1, size_t n2, size_t n3)						= 0;
	virtual daeModel_t* GetModel(size_t n1, size_t n2, size_t n3, size_t n4)			= 0;
	virtual daeModel_t* GetModel(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5) = 0;
};

/*********************************************************************************************
	daeBlock_t
**********************************************************************************************/
class daeBlock_t  
{
public:
	virtual ~daeBlock_t(void){}

public:
	virtual void	Initialize(void) = 0;

	virtual void	CalculateResiduals(real_t			 dTime, 
									   daeArray<real_t>& arrValues, 
									   daeArray<real_t>& arrResiduals, 
									   daeArray<real_t>& arrTimeDerivatives) = 0;

	virtual void	CalculateJacobian(real_t			 dTime, 
									  daeArray<real_t>&	 arrValues, 
									  daeArray<real_t>&	 arrResiduals, 
									  daeArray<real_t>&	 arrTimeDerivatives, 
									  daeMatrix<real_t>& matJacobian, 
									  real_t			 dInverseTimeStep) = 0;

	virtual void	CalculateSensitivityResiduals(real_t					 dTime, 
												  const std::vector<size_t>& narrParameterIndexes,
												  daeArray<real_t>&			 arrValues, 
												  daeArray<real_t>&			 arrTimeDerivatives, 
												  daeMatrix<real_t>&		 matSValues, 
												  daeMatrix<real_t>&		 matSTimeDerivatives, 
												  daeMatrix<real_t>&		 matSResiduals) = 0;
	
	virtual void	CalculateSensitivityParametersGradients(const std::vector<size_t>& narrParameterIndexes,
															daeArray<real_t>&		   arrValues, 
															daeMatrix<real_t>&		   matSResiduals) = 0;
	
	virtual void	CalculateConditions(real_t				 dTime, 
									    daeArray<real_t>&	 arrValues, 
									    daeArray<real_t>&	 arrTimeDerivatives, 
									    std::vector<real_t>& arrResults) = 0;

	virtual void	SetInitialConditionsAndInitialGuesses(daeArray<real_t>& arrValues, 
		                                                  daeArray<real_t>& arrTimeDerivatives, 
														  daeArray<real_t>& arrInitialConditionsTypes) = 0;

	virtual void	FillAbsoluteTolerancesArray(daeArray<real_t>& arrAbsoluteTolerances) = 0;
	virtual size_t	GetNumberOfEquations(void) const = 0;
	virtual size_t	GetNumberOfRoots(void) const = 0;

	virtual void	CopyValuesFromSolver(daeArray<real_t>& arrValues) = 0;
	virtual void	CopyTimeDerivativesFromSolver(daeArray<real_t>& arrTimeDerivatives) = 0;
	virtual void	CopyValuesToSolver(daeArray<real_t>& arrValues) = 0;
	virtual void	CopyTimeDerivativesToSolver(daeArray<real_t>& arrTimeDerivatives) = 0;
	
	virtual bool	              CheckDiscontinuities(void) = 0;
	virtual daeeDiscontinuityType ExecuteOnConditionActionsAndRebuildExpressionMap(void) = 0;
	
	virtual void	SetAllInitialConditions(real_t value) = 0;

	virtual void	CalcNonZeroElements(int& NNZ) = 0;
	virtual void	FillSparseMatrix(daeSparseMatrix<real_t>* pMatrix) = 0;	
	
	virtual real_t	GetTime(void) const  = 0;
	virtual void	SetTime(real_t time) = 0;
	
	virtual size_t	FindVariableBlockIndex(size_t nVariableOverallIndex) const = 0;
	
	virtual bool	IsModelDynamic() const = 0;
};


/******************************************************************
	daeOptimizationVariable_t
*******************************************************************/
class daeOptimizationVariable_t
{
public:
	virtual ~daeOptimizationVariable_t(void){}
	
public:
	virtual std::string GetName(void) const = 0;
	
	virtual size_t GetOverallIndex(void) const = 0;
	virtual size_t GetOptimizationVariableIndex(void) const = 0;
	
	virtual void SetValue(real_t value) = 0;
	virtual real_t GetValue(void) const = 0;
	
	virtual void SetType(daeeOptimizationVariableType value) = 0;
	virtual daeeOptimizationVariableType GetType(void) const = 0;

	virtual void SetStartingPoint(real_t value) = 0;
	virtual real_t GetStartingPoint(void) const = 0;

	virtual void SetLB(real_t value) = 0;
	virtual real_t GetLB(void) const = 0;

	virtual void SetUB(real_t value) = 0;
	virtual real_t GetUB(void) const = 0;
};

/******************************************************************
	daeFunctionWithGradients_t
*******************************************************************/
class daeFunctionWithGradients_t
{
public:
	virtual ~daeFunctionWithGradients_t(void){}
	
public:
	virtual bool IsLinear(void) const = 0;
	
	virtual std::string GetName(void) const = 0;
	virtual real_t GetValue(void) const = 0;
	virtual void GetGradients(const daeMatrix<real_t>& matSensitivities, real_t* gradients, size_t Nparams) const = 0;
	virtual void GetGradients(real_t* gradients, size_t Nparams) const = 0;
	
	virtual void GetOptimizationVariableIndexes(std::vector<size_t>& narrOptimizationVariablesIndexes) const = 0;
	virtual size_t GetNumberOfOptimizationVariables(void) const = 0;
};

/******************************************************************
	daeObjectiveFunction_t
*******************************************************************/
class daeObjectiveFunction_t : virtual public daeFunctionWithGradients_t
{
public:
	virtual ~daeObjectiveFunction_t(void){}
};

/******************************************************************
	daeOptimizationConstraint_t
*******************************************************************/
class daeOptimizationConstraint_t : virtual public daeFunctionWithGradients_t
{
public:
	virtual ~daeOptimizationConstraint_t(void){}
	
public:
	virtual void               SetType(daeeConstraintType value) = 0;
	virtual daeeConstraintType GetType(void) const = 0;
};

/******************************************************************
	daeMeasuredVariable_t
*******************************************************************/
class daeMeasuredVariable_t : virtual public daeFunctionWithGradients_t
{
public:
	virtual ~daeMeasuredVariable_t(void){}
};

/******************************************************************
	daeVariableWrapper_t
*******************************************************************/
class daeVariableWrapper_t
{
public:
	virtual ~daeVariableWrapper_t(void){}

    virtual real_t GetValue(void) const   = 0;
    virtual void   SetValue(real_t value) = 0;
    virtual string GetName(void) const    = 0;
};

/******************************************************************
	daeModelInfo
*******************************************************************/
struct daeModelInfo
{
	size_t m_nNumberOfVariables;
	size_t m_nNumberOfEquations;
	size_t m_nNumberOfFixedVariables;
	size_t m_nNumberOfStateVariables;
	size_t m_nNumberOfDifferentialVariables;
	size_t m_nNumberOfInitialConditions;
};

/******************************************************************
	daeModel_t
*******************************************************************/
class daeModel_t : virtual public daeObject_t
{
public:
	virtual void	GetModelInfo(daeModelInfo& mi) const											= 0;
	virtual void	GetSTNs(std::vector<daeSTN_t*>& ptrarrSTNs)										= 0;
	virtual void	GetPorts(std::vector<daePort_t*>& ptrarrPorts)									= 0;
	virtual void	GetEquations(std::vector<daeEquation_t*>& ptrarrEquations)						= 0;
	virtual void	GetModels(std::vector<daeModel_t*>& ptrarrModels)								= 0;
	virtual void	GetDomains(std::vector<daeDomain_t*>& ptrarrDomains)							= 0;
	virtual void	GetVariables(std::vector<daeVariable_t*>& ptrarrVariables)						= 0;
	virtual void	GetParameters(std::vector<daeParameter_t*>& ptrarrParameters)					= 0;
	virtual void	GetPortConnections(std::vector<daePortConnection_t*>& ptrarrPortConnections)	= 0;
	virtual void	GetPortArrays(std::vector<daePortArray_t*>& ptrarrPortArrays)					= 0;
	virtual void	GetModelArrays(std::vector<daeModelArray_t*>& ptrarrModelArrays)				= 0;

	virtual void	InitializeStage1(void)															= 0;
	virtual void	InitializeStage2(void)															= 0;
	virtual void	InitializeStage3(daeLog_t* pLog)												= 0;
	virtual void	InitializeStage4(void)															= 0;
	virtual void	InitializeStage5(bool bDoBlockDecomposition, 
									 std::vector<daeBlock_t*>& ptrarrBlocks) = 0;

	virtual daeeInitialConditionMode	GetInitialConditionMode(void) const						= 0;
	virtual void						SetInitialConditionMode(daeeInitialConditionMode eMode)	= 0;
	
//	virtual void	SetInitialConditions(real_t value)							= 0;

	virtual void	StoreInitializationValues(const std::string& strFileName) const		= 0;
	virtual void	LoadInitializationValues(const std::string& strFileName) const		= 0;

	virtual void	SetReportingOn(bool bOn)								= 0;

	virtual void	SaveModelReport(const string& strFileName) const		= 0;
	virtual void	SaveRuntimeModelReport(const string& strFileName) const	= 0;
	virtual bool	IsModelDynamic() const									= 0;

	virtual daeDomain_t*		FindDomain(string& strCanonicalName)		= 0;
	virtual daeParameter_t*		FindParameter(string& strCanonicalName)		= 0;
	virtual daeVariable_t*		FindVariable(string& strCanonicalName)		= 0;
	virtual daePort_t*			FindPort(string& strCanonicalName)			= 0;
	virtual daeModel_t*			FindModel(string& strCanonicalName)			= 0;
	virtual daeEventPort_t*		FindEventPort(string& strName)				= 0;
	virtual daeSTN_t*			FindSTN(string& strCanonicalName)			= 0;
	virtual daePortArray_t*		FindPortArray(string& strCanonicalName)		= 0;
	virtual daeModelArray_t*	FindModelArray(string& strCanonicalName)	= 0;
	
	virtual daeObject_t*		FindObject(std::string& strName)									= 0;
	virtual daeObject_t*		FindObjectFromRelativeName(std::string& strRelativeName)			= 0;
	virtual daeObject_t*		FindObjectFromRelativeName(std::vector<std::string>& strarrNames)	= 0;
};


/*********************************************************************************************
	daeExternalFunctionArgument_t
**********************************************************************************************/
struct daeExternalFunctionArgumentInfo_t;
class daeExternalFunctionArgument_t
{
public:
	virtual ~daeExternalFunctionArgument_t(void){}

public:
	virtual void								GetValues(real_t* values, size_t n)       = 0;
	virtual void								SetValues(const real_t* values, size_t n) = 0;
	virtual real_t								operator [](size_t i) const               = 0;
	virtual daeExternalFunctionArgumentInfo_t	GetInfo(void)                             = 0;
};

/*********************************************************************************************
	daeExternalFunctionXXX info structures
**********************************************************************************************/
struct daeExternalFunctionArgumentInfo_t
{
	std::string	m_strName;	
	size_t		m_nLength;	
};

struct daeExternalFunctionInfo_t
{
	std::string										m_strName;	
	size_t											m_nNumberOfResults;	
	bool											m_bCanCalculateDerivatives;	
	std::vector<daeExternalFunctionArgument_t*>		m_ptrarrArguments;	
};
	
struct daeExternalObjectInfo_t
{
	std::string					m_strName;	
	std::vector<std::string>	m_strarrAvailableFunctions;	
};


/*********************************************************************************************
	daeExternalFunction_t
**********************************************************************************************/
class daeExternalFunction_t
{
public:
	virtual ~daeExternalFunction_t(void){}

public:
// Before a call to Calculate/CalculateDerivatives the aruments' values has to be set
	virtual void						Calculate(real_t* results, size_t n)                 = 0;
	virtual void						CalculateDerivatives(daeMatrix<real_t>& derivatives) = 0;
	virtual daeExternalFunctionInfo_t	GetInfo(void)                                        = 0;
};


/*********************************************************************************************
	daeExternalObject_t
**********************************************************************************************/
class daeExternalObject_t
{
public:
	virtual ~daeExternalObject_t(void){}

public:
	// It is user's responsibility to ensure thread safety if multiple objects of the same 
	// function are created and some vintage code is called
	virtual daeExternalFunction_t*	CreateFunction(const std::string& strFunctionName) = 0;
	virtual daeExternalObjectInfo_t	GetInfo(void)                                      = 0;
};

/*
 C function prototype that external object shared libraries has to implement and export:
	   daeExternalObject_t* GetExternalObject(void);
 Should be defined as extern "C" to avoid a dodgy compiler name-mangling
*/
extern "C"
{
	typedef daeExternalObject_t* (*pfnGetExternalObject)(void);
}


/******************************************************************
	daeCoreClassFactory_t
*******************************************************************/
class daeCoreClassFactory_t
{
public:
	virtual ~daeCoreClassFactory_t(void){}

public:
    virtual string   GetName(void) const			= 0;
    virtual string   GetDescription(void) const		= 0;
    virtual string   GetAuthorInfo(void) const		= 0;
    virtual string   GetLicenceInfo(void) const		= 0;
    virtual string   GetVersion(void) const			= 0;

    virtual daeVariableType_t*		CreateVariableType(const string& strClass)		= 0;
	virtual daePort_t*				CreatePort(const string& strClass)				= 0;
	virtual daeModel_t*				CreateModel(const string& strClass)				= 0;
	
//	virtual daeParameter_t*			CreateParameter(const string& strClass)			= 0;
//	virtual daeDomain_t*			CreateDomain(const string& strClass)			= 0;
//	virtual daeVariable_t*			CreateVariable(const string& strClass)			= 0;
//	virtual daeEquation_t*			CreateEquation(const string& strClass)			= 0;
//	virtual daeSTN_t*				CreateSTN(const string& strClass)				= 0;
//	virtual daeState_t*				CreateState(const string& strClass)				= 0;
//	virtual daeStateTransition_t*	CreateStateTransition(const string& strClass)	= 0;
//	virtual daePortConnection_t*	CreatePortConnection(const string& strClass)	= 0;

	virtual void SupportedVariableTypes(std::vector<string>& strarrClasses)		= 0;
	virtual void SupportedPorts(std::vector<string>& strarrClasses)				= 0;
	virtual void SupportedModels(std::vector<string>& strarrClasses)			= 0;
	
//	virtual void SupportedParameters(std::vector<string>& strarrClasses)		= 0;
//	virtual void SupportedDomains(std::vector<string>& strarrClasses)			= 0;
//	virtual void SupportedVariables(std::vector<string>& strarrClasses)			= 0;
//	virtual void SupportedEquations(std::vector<string>& strarrClasses)			= 0;
//	virtual void SupportedSTNs(std::vector<string>& strarrClasses)				= 0;
//	virtual void SupportedStates(std::vector<string>& strarrClasses)			= 0;
//	virtual void SupportedStateTransitions(std::vector<string>& strarrClasses)	= 0;
//	virtual void SupportedPortConnections(std::vector<string>& strarrClasses)	= 0;
};
typedef daeCoreClassFactory_t* (*pfnGetCoreClassFactory)(void);


}
}

#endif

