#include "stdafx.h"
#include "coreimpl.h"
#include <algorithm>

namespace dae 
{
namespace core 
{
/******************************************************************
	daeObjectiveFunction
*******************************************************************/
daeObjectiveFunction::daeObjectiveFunction(daeModel* pModel, real_t abstol)
{
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer)
		
	const daeVariableType typeObjectiveFunction("typeObjectiveFunction", "-", -1.0e+100, 1.0e+100, 0.0, abstol);
	m_pModel				= pModel;
	m_nEquationIndexInBlock = ULONG_MAX;
	m_pObjectiveVariable	= boost::shared_ptr<daeVariable>(new daeVariable("V_obj", typeObjectiveFunction, pModel, "Objective value"));
	m_pObjectiveVariable->SetReportingOn(true);
	m_pObjectiveFunction = pModel->CreateEquation("F_obj", "Objective function");
}

daeObjectiveFunction::~daeObjectiveFunction(void)
{
}

void daeObjectiveFunction::SetResidual(adouble res)
{
	if(!m_pObjectiveFunction)
		daeDeclareAndThrowException(exInvalidPointer)
	m_pObjectiveFunction->SetResidual( res - (*m_pObjectiveVariable )() );
}

adouble daeObjectiveFunction::GetResidual(void) const
{
	if(!m_pObjectiveFunction)
		daeDeclareAndThrowException(exInvalidPointer)
	return m_pObjectiveFunction->GetResidual();
}

void daeObjectiveFunction::Open(io::xmlTag_t* pTag)
{
	daeObject::Open(pTag);
}

void daeObjectiveFunction::Save(io::xmlTag_t* pTag) const
{
	daeObject::Save(pTag);
}

void daeObjectiveFunction::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daeObjectiveFunction::SaveRuntime(io::xmlTag_t* pTag) const
{
	daeObject::SaveRuntime(pTag);
}

void daeObjectiveFunction::Initialize(const std::vector< boost::shared_ptr<daeOptimizationVariable> >& arrOptimizationVariables)
{
	size_t i;
	map<size_t, size_t> mapVariableIndexes;
	map<size_t, size_t>::iterator iter;
	boost::shared_ptr<daeOptimizationVariable> pOptVariable;
	daeEquationExecutionInfo* pEquationExecutionInfo;
	vector<daeEquationExecutionInfo*> ptrarrEquationExecutionInfos;

	if(!m_pObjectiveFunction)
		daeDeclareAndThrowException(exInvalidPointer)
	
	m_pObjectiveFunction->GetEquationExecutionInfos(ptrarrEquationExecutionInfos);
	if(ptrarrEquationExecutionInfos.size() != 1)
		daeDeclareAndThrowException(exInvalidCall)
		
	pEquationExecutionInfo = ptrarrEquationExecutionInfos[0];
	if(!pEquationExecutionInfo)
		daeDeclareAndThrowException(exInvalidPointer)
		
// 1. Set the equation index in the block it belongs
	m_nEquationIndexInBlock = pEquationExecutionInfo->GetEquationIndexInBlock();

// 2a. Add all optimization variables indexes that are found in this constraint
//     This requires some iterating over the arrays of indexes
	m_narrOptimizationVariablesIndexes.clear();
	
	boost::shared_ptr<adNode> node = pEquationExecutionInfo->GetEquationEvaluationNode();
	node->AddVariableIndexToArray(mapVariableIndexes);
	
//	std::cout << "ObjectiveFunction " << m_pObjectiveFunction->GetName() << " indexes: ";
//	for(iter = mapVariableIndexes.begin(); iter != mapVariableIndexes.end(); iter++)
//		std::cout << iter->first << " ";
//	std::cout << std::endl;
//	std::cout.flush();	

	for(i = 0; i < arrOptimizationVariables.size(); i++)
	{
		pOptVariable = arrOptimizationVariables[i];
	// If I find the index in the optimization variable within the constraint - add it to the array
		if( mapVariableIndexes.find(pOptVariable->GetIndex()) != mapVariableIndexes.end() )
			m_narrOptimizationVariablesIndexes.push_back(pOptVariable->GetIndex());
	}
	
// 2b. Sort the array
	std::sort(m_narrOptimizationVariablesIndexes.begin(), m_narrOptimizationVariablesIndexes.end());

//	std::cout << "ObjectiveFunction " << m_pObjectiveFunction->GetName() << " common indexes: ";
//	for(i = 0; i < m_narrOptimizationVariablesIndexes.size(); i++)
//		std::cout << m_narrOptimizationVariablesIndexes[i] << " ";
//	std::cout << std::endl;
//	std::cout.flush();	
}

bool daeObjectiveFunction::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

// Do not check daeObject since this is not an usual object
//	if(!daeObject::CheckObject(strarrErrors))
//		bCheck = false;

	if(!m_pObjectiveVariable)
	{	
		strError = "Invalid objective variable";
		strarrErrors.push_back(strError);
		return false;
	}
	if(!m_pObjectiveFunction)
	{	
		strError = "Invalid objective function";
		strarrErrors.push_back(strError);
		return false;
	}
	if(m_pObjectiveFunction->GetEquationDefinitionMode() == eEDMUnknown ||
	   m_pObjectiveFunction->GetEquationEvaluationMode() == eEEMUnknown)
	{	
		strError = "Objective function residual not specified [" + m_pObjectiveFunction->GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}
	
	cout << "Objective function is: " << (m_pObjectiveFunction->GetResidual().node->IsLinear() ? "linear" : "non-linear") << endl;

	return bCheck;
}


/******************************************************************
	daeOptimizationConstraint
*******************************************************************/
daeOptimizationConstraint::daeOptimizationConstraint(daeModel* pModel, real_t LB, real_t UB, real_t abstol, size_t N, string strDescription)
{
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer)
		
	const daeVariableType typeConstraint("typeConstraint", "-", -1.0e+100, 1.0e+100, 0.0, abstol);
	string strVName = string("V_constraint") + toString<size_t>(N + 1); 
	string strFName = string("F_constraint") + toString<size_t>(N + 1); 

	m_pModel				= pModel;
	m_eConstraintType		= eInequalityConstraint;
	m_dLB					= LB;
	m_dUB					= UB;
	m_dValue				= -1E20;
	m_nEquationIndexInBlock = ULONG_MAX;
	m_pConstraintVariable	= boost::shared_ptr<daeVariable>(new daeVariable(strVName, typeConstraint, m_pModel, strDescription));
	m_pConstraintVariable->SetReportingOn(true);
	m_pConstraintFunction	= m_pModel->CreateEquation(strFName, strDescription);
}

daeOptimizationConstraint::daeOptimizationConstraint(daeModel* pModel, real_t Value, real_t abstol, size_t N, string strDescription)
{
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer)
		
	const daeVariableType typeConstraint("typeConstraint", "-", -1.0e+100, 1.0e+100, 0.0, abstol);
	string strVName = string("V_constraint") + toString<size_t>(N + 1); 
	string strFName = string("F_constraint") + toString<size_t>(N + 1); 

	m_pModel				= pModel;
	m_eConstraintType		= eEqualityConstraint;
	m_dLB					= Value;
	m_dUB					= Value;
	m_dValue				= Value;
	m_nEquationIndexInBlock = ULONG_MAX;
	m_pConstraintVariable	= boost::shared_ptr<daeVariable>(new daeVariable(strVName, typeConstraint, m_pModel, strDescription));
	m_pConstraintVariable->SetReportingOn(true);
	m_pConstraintFunction	= m_pModel->CreateEquation(strFName, strDescription);
}

daeOptimizationConstraint::~daeOptimizationConstraint(void)
{
}

void daeOptimizationConstraint::SetResidual(adouble res)
{
	if(!m_pConstraintFunction)
		daeDeclareAndThrowException(exInvalidPointer)
	m_pConstraintFunction->SetResidual( res - (*m_pConstraintVariable )() );
}

adouble daeOptimizationConstraint::GetResidual(void) const
{
	if(!m_pConstraintFunction)
		daeDeclareAndThrowException(exInvalidPointer)
	return m_pConstraintFunction->GetResidual();
}

void daeOptimizationConstraint::Open(io::xmlTag_t* pTag)
{
	daeObject::Open(pTag);
}

void daeOptimizationConstraint::Save(io::xmlTag_t* pTag) const
{
	daeObject::Save(pTag);
}

void daeOptimizationConstraint::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daeOptimizationConstraint::SaveRuntime(io::xmlTag_t* pTag) const
{
	daeObject::SaveRuntime(pTag);
}

void daeOptimizationConstraint::Initialize(const std::vector< boost::shared_ptr<daeOptimizationVariable> >& arrOptimizationVariables)
{
	size_t i;
	map<size_t, size_t> mapVariableIndexes;
	map<size_t, size_t>::iterator iter;
	boost::shared_ptr<daeOptimizationVariable> pOptVariable;
	daeEquationExecutionInfo* pEquationExecutionInfo;
	vector<daeEquationExecutionInfo*> ptrarrEquationExecutionInfos;

	if(!m_pConstraintFunction)
		daeDeclareAndThrowException(exInvalidPointer)
	
	m_pConstraintFunction->GetEquationExecutionInfos(ptrarrEquationExecutionInfos);
	if(ptrarrEquationExecutionInfos.size() != 1)
		daeDeclareAndThrowException(exInvalidCall)
		
	pEquationExecutionInfo = ptrarrEquationExecutionInfos[0];
	if(!pEquationExecutionInfo)
		daeDeclareAndThrowException(exInvalidPointer)
	
// 1. Set the equation index in the block it belongs
	m_nEquationIndexInBlock = pEquationExecutionInfo->GetEquationIndexInBlock();
	
// 2a. Add all optimization variables indexes that are found in this constraint
	m_narrOptimizationVariablesIndexes.clear();
	
	boost::shared_ptr<adNode> node = pEquationExecutionInfo->GetEquationEvaluationNode();
	node->AddVariableIndexToArray(mapVariableIndexes);
	
//	std::cout << "Constraint " << m_pConstraintFunction->GetName() << " indexes: ";
//	for(iter = mapVariableIndexes.begin(); iter != mapVariableIndexes.end(); iter++)
//		std::cout << iter->first << " ";
//	std::cout << std::endl;
//	std::cout.flush();

	for(i = 0; i < arrOptimizationVariables.size(); i++)
	{
		pOptVariable = arrOptimizationVariables[i];
	// If I find the index in the optimization variable within the constraint - add it to the array
		if( mapVariableIndexes.find(pOptVariable->GetIndex()) != mapVariableIndexes.end() )
			m_narrOptimizationVariablesIndexes.push_back(pOptVariable->GetIndex());
	}
	
// 2b. Sort the array
	std::sort(m_narrOptimizationVariablesIndexes.begin(), m_narrOptimizationVariablesIndexes.end());
	
//	std::cout << "Constraint " << m_pConstraintFunction->GetName() << " common indexes: ";
//	for(i = 0; i < m_narrOptimizationVariablesIndexes.size(); i++)
//		std::cout << m_narrOptimizationVariablesIndexes[i] << " ";
//	std::cout << std::endl;
//	std::cout.flush();
}

bool daeOptimizationConstraint::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

// Do not check daeObject since this is not an usual object
//	if(!daeObject::CheckObject(strarrErrors))
//		bCheck = false;

	if(!m_pConstraintVariable)
	{	
		strError = "Invalid constraint variable";
		strarrErrors.push_back(strError);
		return false;
	}
	if(!m_pConstraintFunction)
	{	
		strError = "Invalid constraint function";
		strarrErrors.push_back(strError);
		return false;
	}
	if(m_pConstraintFunction->GetEquationDefinitionMode() == eEDMUnknown ||
	   m_pConstraintFunction->GetEquationEvaluationMode() == eEEMUnknown)
	{	
		strError = "Constraint residual not specified [" + m_pConstraintFunction->GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}
	
	// Here I access SetupNode!! Is it wise??
	cout << "Constraint [" << m_pConstraintFunction->GetName() << "] is: " << (m_pConstraintFunction->GetResidual().node->IsLinear() ? "linear" : "non-linear") << endl;

	return bCheck;
}

/******************************************************************
	daeOptimizationVariable
*******************************************************************/
daeOptimizationVariable::daeOptimizationVariable(daeVariable* pVariable, real_t LB, real_t UB, real_t defaultValue)
{
	if(!pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
			
	m_pVariable     = pVariable;
	m_dLB           = LB;
	m_dUB           = UB;
	m_dDefaultValue = defaultValue;
	m_eType         = eContinuousVariable;
}

daeOptimizationVariable::daeOptimizationVariable(daeVariable* pVariable, int LB, int UB, int defaultValue)
{
	if(!pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
			
	m_pVariable     = pVariable;
	m_dLB           = LB;
	m_dUB           = UB;
	m_dDefaultValue = defaultValue;
	m_eType         = eIntegerVariable;
}

daeOptimizationVariable::daeOptimizationVariable(daeVariable* pVariable, bool defaultValue)
{
	if(!pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
			
	m_pVariable     = pVariable;
	m_dLB           = 0;
	m_dUB           = 1;
	m_dDefaultValue = (defaultValue ? 1 : 0);
	m_eType         = eBinaryVariable;
}

daeOptimizationVariable::~daeOptimizationVariable(void)
{
}

size_t daeOptimizationVariable::GetIndex(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
		
	return m_pVariable->m_nOverallIndex;	
}

bool daeOptimizationVariable::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

// Do not check daeObject since this is not an usual object
//	if(!daeObject::CheckObject(strarrErrors))
//		bCheck = false;

	if(!m_pVariable)
	{	
		strError = "Invalid optimization variable specified";
		strarrErrors.push_back(strError);
		return false;
	}
	if(!m_pVariable->m_pModel || !m_pVariable->m_pModel->m_pDataProxy)
	{	
		strError = "Invalid parent model in optimization variable [" + m_pVariable->GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}
			
	if(!m_pVariable->m_ptrDomains.empty())
	{
		strError = "Optimization variable [" + m_pVariable->GetCanonicalName() + "] cannot be distributed";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
  
	int type = m_pVariable->m_pModel->m_pDataProxy->GetVariableType(m_pVariable->m_nOverallIndex);
	if(type != cnFixed)
	{
		strError = "Optimization variable [" + m_pVariable->GetCanonicalName() + "] must be assigned (cannot be a state-variable)";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
	if(m_dDefaultValue < m_dLB)
	{
		strError = "The default value of the optimization variable [" + m_pVariable->GetCanonicalName() + "] is lower than the lower bound";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	if(m_dDefaultValue > m_dUB)
	{
		strError = "The default value of the optimization variable [" + m_pVariable->GetCanonicalName() + "] is greater than the upper bound";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

	return bCheck;
}




}
}
