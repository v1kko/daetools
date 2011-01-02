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
		if( mapVariableIndexes.find(pOptVariable->GetOverallIndex()) != mapVariableIndexes.end() )
			m_narrOptimizationVariablesIndexes.push_back(pOptVariable->GetOptimizationVariableIndex());
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
		if( mapVariableIndexes.find(pOptVariable->GetOverallIndex()) != mapVariableIndexes.end() )
			m_narrOptimizationVariablesIndexes.push_back(pOptVariable->GetOptimizationVariableIndex());
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
daeOptimizationVariable::daeOptimizationVariable(daeVariable* pVariable, size_t nOptimizationVariableIndex, const std::vector<size_t>& narrDomainIndexes, real_t LB, real_t UB, real_t defaultValue)
{
	if(!pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
			
	m_pVariable						= pVariable;
	m_dLB							= LB;
	m_dUB							= UB;
	m_dDefaultValue					= defaultValue;
	m_eType							= eContinuousVariable;
	m_narrDomainIndexes 			= narrDomainIndexes;
	m_nOptimizationVariableIndex	= nOptimizationVariableIndex;
}

daeOptimizationVariable::daeOptimizationVariable(daeVariable* pVariable, size_t nOptimizationVariableIndex, const std::vector<size_t>& narrDomainIndexes, int LB, int UB, int defaultValue)
{
	if(!pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
			
	m_pVariable						= pVariable;
	m_dLB							= LB;
	m_dUB							= UB;
	m_dDefaultValue					= defaultValue;
	m_eType							= eIntegerVariable;
	m_narrDomainIndexes 			= narrDomainIndexes;
	m_nOptimizationVariableIndex	= nOptimizationVariableIndex;
}

daeOptimizationVariable::daeOptimizationVariable(daeVariable* pVariable, size_t nOptimizationVariableIndex, const std::vector<size_t>& narrDomainIndexes, bool defaultValue)
{
	if(!pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
			
	m_pVariable						= pVariable;
	m_dLB							= 0;
	m_dUB							= 1;
	m_dDefaultValue					= (defaultValue ? 1 : 0);
	m_eType							= eBinaryVariable;
	m_narrDomainIndexes				= narrDomainIndexes;
	m_nOptimizationVariableIndex	= nOptimizationVariableIndex;
}

daeOptimizationVariable::~daeOptimizationVariable(void)
{
}

size_t daeOptimizationVariable::GetOverallIndex(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
		
	size_t index = m_pVariable->m_nOverallIndex + 
				   m_pVariable->CalculateIndex(m_narrDomainIndexes);
	return index;	
}

size_t daeOptimizationVariable::GetOptimizationVariableIndex(void) const
{
	return m_nOptimizationVariableIndex;	
}

std::string daeOptimizationVariable::GetName(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidPointer)

	string strName = m_pVariable->GetName();
	if(m_narrDomainIndexes.empty())
		return strName;
	
	strName += "("; 
	for(size_t i = 0; i < m_narrDomainIndexes.size(); i++)
	{
		if(i != 0)
			strName += ",";
		strName += toString<size_t>(m_narrDomainIndexes[i]);
	}
	strName += ")";
	return strName;
}

void daeOptimizationVariable::SetValue(real_t value)
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
		
	size_t n = m_narrDomainIndexes.size();
	
	if(n == 0)
		m_pVariable->ReAssignValue(value);
	else if(n == 1)
		m_pVariable->ReAssignValue(m_narrDomainIndexes[0],
								   value);
	else if(n == 2)
		m_pVariable->ReAssignValue(m_narrDomainIndexes[0],
								   m_narrDomainIndexes[1],
								   value);
	else if(n == 3)
		m_pVariable->ReAssignValue(m_narrDomainIndexes[0],
								   m_narrDomainIndexes[1],
								   m_narrDomainIndexes[2],
								   value);
	else if(n == 4)
		m_pVariable->ReAssignValue(m_narrDomainIndexes[0],
								   m_narrDomainIndexes[1],
								   m_narrDomainIndexes[2],
								   m_narrDomainIndexes[3],
								   value);
	else if(n == 5)
		m_pVariable->ReAssignValue(m_narrDomainIndexes[0],
								   m_narrDomainIndexes[1],
								   m_narrDomainIndexes[2],
								   m_narrDomainIndexes[3],
								   m_narrDomainIndexes[4],
								   value);
	else if(n == 6)
		m_pVariable->ReAssignValue(m_narrDomainIndexes[0],
								   m_narrDomainIndexes[1],
								   m_narrDomainIndexes[2],
								   m_narrDomainIndexes[3],
								   m_narrDomainIndexes[4],
								   m_narrDomainIndexes[5],
								   value);
	else if(n == 7)
		m_pVariable->ReAssignValue(m_narrDomainIndexes[0],
								   m_narrDomainIndexes[1],
								   m_narrDomainIndexes[2],
								   m_narrDomainIndexes[3],
								   m_narrDomainIndexes[4],
								   m_narrDomainIndexes[5],
								   m_narrDomainIndexes[6],
								   value);
	else if(n == 8)
		m_pVariable->ReAssignValue(m_narrDomainIndexes[0],
								   m_narrDomainIndexes[1],
								   m_narrDomainIndexes[2],
								   m_narrDomainIndexes[3],
								   m_narrDomainIndexes[4],
								   m_narrDomainIndexes[5],
								   m_narrDomainIndexes[6],
								   m_narrDomainIndexes[7],
								   value);
	else
		daeDeclareAndThrowException(exInvalidCall)
}

real_t daeOptimizationVariable::GetValue(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidPointer)
	
	size_t n = m_narrDomainIndexes.size();
	
	if(n == 0)
		return m_pVariable->GetValue();
	else if(n == 1)
		return m_pVariable->GetValue(m_narrDomainIndexes[0]);
	else if(n == 2)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1]);
	else if(n == 3)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2]);
	else if(n == 4)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3]);
	else if(n == 5)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3],
									 m_narrDomainIndexes[4]);
	else if(n == 6)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3],
									 m_narrDomainIndexes[4],
									 m_narrDomainIndexes[5]);
	else if(n == 7)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3],
									 m_narrDomainIndexes[4],
									 m_narrDomainIndexes[5],
									 m_narrDomainIndexes[6]);
	else if(n == 8)
		return m_pVariable->GetValue(m_narrDomainIndexes[0],
									 m_narrDomainIndexes[1],
									 m_narrDomainIndexes[2],
									 m_narrDomainIndexes[3],
									 m_narrDomainIndexes[4],
									 m_narrDomainIndexes[5],
									 m_narrDomainIndexes[6],
									 m_narrDomainIndexes[7]);
	else
		daeDeclareAndThrowException(exInvalidCall)
				
	return 0;
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
			
	if(m_pVariable->m_ptrDomains.size() != m_narrDomainIndexes.size())
	{
		strError = "Wrong number of indexes in the optimization variable [" + m_pVariable->GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
  
	size_t index = GetOverallIndex();
	int type = m_pVariable->m_pModel->m_pDataProxy->GetVariableType(index);

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
