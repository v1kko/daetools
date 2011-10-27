#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
using namespace dae;
#include <typeinfo>
#include <stack>

namespace dae 
{
namespace core 
{
void SetValue(daeFPUCommand* cmd, bool isLeft, real_t value);
void SetOperand(daeFPUCommand* cmd, bool isLeft, size_t value);
bool FillOperandInfo(adNode* node, daeFPUCommand* cmd, bool isLeft);

ostream& operator << (ostream& os, const daeFPUCommand& cmd)
{
	os << "info.kind     = ";
	if(cmd.info.kind == eUnary)
		os << "eUnary" << endl;
	else
		os << "eBinary" << endl;

	os << "info.opcode   = ";
	if(cmd.info.opcode == ePlus)
		os << "ePlus" << endl;
	else if(cmd.info.opcode == eMinus)
		os << "eMinus" << endl;
	else if(cmd.info.opcode == eMulti)
		os << "eMulti" << endl;
	else if(cmd.info.opcode == eDivide)
		os << "eDivide" << endl;
	else
		os << "Unknown" << endl;

	os << "info.loperand = ";
	if(cmd.info.loperand == eConstant)
		os << "eConstant" << endl;
	else if(cmd.info.loperand == eDomain)
		os << "eDomain" << endl;
	else if(cmd.info.loperand == eValue)
		os << "eValue" << endl;
	else if(cmd.info.loperand == eTimeDerivative)
		os << "eTimeDerivative" << endl;
	else if(cmd.info.loperand == eFromStack)
		os << "eFromStack" << endl;
	else
		os << "Unknown" << endl;

	os << "info.roperand = ";
	if(cmd.info.roperand == eConstant)
		os << "eConstant" << endl;
	else if(cmd.info.roperand == eDomain)
		os << "eDomain" << endl;
	else if(cmd.info.roperand == eValue)
		os << "eValue" << endl;
	else if(cmd.info.roperand == eTimeDerivative)
		os << "eTimeDerivative" << endl;
	else if(cmd.info.roperand == eFromStack)
		os << "eFromStack" << endl;
	else
		os << "Unknown" << endl;

	os << "info.result   = " << cmd.info.result		<< endl;
	os << "lvalue        = " << cmd.lvalue			<< endl;
	os << "rvalue        = " << cmd.rvalue			<< endl;
	return os;
}

/*********************************************************************************************
	daeFPUCommand
**********************************************************************************************/
void daeFPU::CreateCommandStack(adNode* node, vector<daeFPUCommand*>& ptrarrCommands)
{
	daeFPUCommand* cmd;

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	const type_info& infoNode = typeid(*node);

	if(infoNode == typeid(adUnaryNode))
	{
		adUnaryNode* unary = dynamic_cast<adUnaryNode*>(node);
		cmd = new daeFPUCommand;
		memset(cmd, 0, sizeof(daeFPUCommand));
		ptrarrCommands.push_back(cmd);

		cmd->info.kind   = eUnary;
		cmd->info.opcode = unary->eFunction;

		if(!FillOperandInfo(unary->node.get(), cmd, true))
		{
			cmd->info.loperand = eFromStack;
			CreateCommandStack(unary->node.get(), ptrarrCommands);
		}
	}
	else if(infoNode == typeid(adBinaryNode))
	{
		adBinaryNode* binary = dynamic_cast<adBinaryNode*>(node);
		cmd = new daeFPUCommand;
		memset(cmd, 0, sizeof(daeFPUCommand));
		ptrarrCommands.push_back(cmd);

		cmd->info.kind   = eBinary;
		cmd->info.opcode = binary->eFunction;

		if(!FillOperandInfo(binary->left.get(), cmd, true))
		{
			cmd->info.loperand = eFromStack;
			CreateCommandStack(binary->left.get(), ptrarrCommands);
		}
		if(!FillOperandInfo(binary->right.get(), cmd, false))
		{
			cmd->info.roperand = eFromStack;
			CreateCommandStack(binary->right.get(), ptrarrCommands);
		}
	}
	else if(infoNode == typeid(adRuntimePartialDerivativeNode))
	{
		adRuntimePartialDerivativeNode* partial = dynamic_cast<adRuntimePartialDerivativeNode*>(node);
		CreateCommandStack(partial->pardevnode.get(), ptrarrCommands);
	}
	else
	{
	}
}

void SetValue(daeFPUCommand* cmd, bool isLeft, real_t value)
{
	if(isLeft)
		cmd->lvalue = value;
	else
		cmd->rvalue = value;
}

void SetOperand(daeFPUCommand* cmd, bool isLeft, size_t value)
{
	if(isLeft)
		cmd->info.loperand = value;
	else
		cmd->info.roperand = value;
}

bool FillOperandInfo(adNode* node, daeFPUCommand* cmd, bool isLeft)
{
	const type_info& infoNode = typeid(*node);

	if(infoNode == typeid(adConstantNode))
	{
		adConstantNode* n = dynamic_cast<adConstantNode*>(node);
		SetOperand(cmd, isLeft, eConstant);
		SetValue(cmd, isLeft, n->m_dValue);
		return true;
	}
	//else if(infoNode == typeid(adNoOperationNode))
	//{
	//}
	else if(infoNode == typeid(adRuntimeParameterNode))
	{
		adRuntimeParameterNode* n = dynamic_cast<adRuntimeParameterNode*>(node);
		SetOperand(cmd, isLeft, eConstant);
		SetValue(cmd, isLeft, n->m_dValue);
		return true;
	}
	else if(infoNode == typeid(adDomainIndexNode))
	{
		adDomainIndexNode* n = dynamic_cast<adDomainIndexNode*>(node);
		SetOperand(cmd, isLeft, eDomain);
		SetValue(cmd, isLeft, n->m_pDomain->GetPoint(n->m_nIndex));
		return true;
	}
	else if(infoNode == typeid(adRuntimeVariableNode))
	{
		adRuntimeVariableNode* n = dynamic_cast<adRuntimeVariableNode*>(node);
		SetOperand(cmd, isLeft, eValue);
		SetValue(cmd, isLeft, n->m_nOverallIndex);
		return true;
	}
	else if(infoNode == typeid(adRuntimeTimeDerivativeNode))
	{
		adRuntimeTimeDerivativeNode* n = dynamic_cast<adRuntimeTimeDerivativeNode*>(node);
		SetOperand(cmd, isLeft, eTimeDerivative);
		SetValue(cmd, isLeft, n->m_nOverallIndex);
		return true;
	}
	else if(infoNode == typeid(adRuntimePartialDerivativeNode))
	{
	}
	else if(infoNode == typeid(adUnaryNode))
	{
	}
	else if(infoNode == typeid(adBinaryNode))
	{

	}

	return false;
}

/*********************************************************************************************
	daeFPU Calculator
**********************************************************************************************/
bool daeCommandStackDebugMode = false;

template<class T>
class daeCommandStack : public stack<T>
{
public:
	~daeCommandStack()
	{
		if(daeCommandStackDebugMode)
			cout << endl;
	}
	T& top_pop()
	{
		if(stack<T>::empty())
			daeDeclareAndThrowException(exInvalidCall);
		T& value = stack<T>::top();
		stack<T>::pop();
		if(daeCommandStackDebugMode)
			cout << "pop(" << value << "); ";
		return value;
	}
	void push(T value)
	{
		stack<T>::push(value);
		if(daeCommandStackDebugMode)
			cout << "push(" << value << "); ";
	}

public:
	static bool m_bDebugMode;
};

template<class TYPE>
inline TYPE GetOperand(const daeDataProxy_t* pDataProxy, const size_t operand, const real_t value, daeCommandStack<TYPE>& stack)
{
	if(operand == eConstant)
	{
		return value;
	}
	else if(operand == eDomain)
	{
		return value;
	}
	else if(operand == eValue)
	{
		return *pDataProxy->GetValue((size_t)value);
	}
	else if(operand == eTimeDerivative)
	{
		return *pDataProxy->GetTimeDerivative((size_t)value);
	}
	else if(operand == eFromStack)
	{
		return stack.top_pop();
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall);
	}
}

void daeFPU::fpuResidual(const daeExecutionContext* pEC, const vector<daeFPUCommand*>& ptrarrCommands, real_t& dResult)
{
	real_t left, right;
	daeFPUCommand* cmd;
	daeCommandStack<real_t> stack;
	vector<daeFPUCommand*>::const_reverse_iterator it;

// I have to inverse iterate!!!
	for(it = ptrarrCommands.rbegin(); it != ptrarrCommands.rend(); it++)
	{
		cmd = *it;
		if(cmd->info.kind == eUnary)
		{
			left = GetOperand<real_t>(pEC->m_pDataProxy, cmd->info.loperand, cmd->lvalue, stack);

			if(cmd->info.opcode == eSign)
				stack.push( -left );
			else if(cmd->info.opcode == eSqrt)
				stack.push( ::sqrt(left) );
			else if(cmd->info.opcode == eExp)
				stack.push( ::exp(left) );
			else if(cmd->info.opcode == eLog)
				stack.push( ::log10(left) );
			else if(cmd->info.opcode == eLn)
				stack.push( ::log(left) );
			else if(cmd->info.opcode == eAbs)
				stack.push( ::fabs(left) );
			else if(cmd->info.opcode == eSin)
				stack.push( ::sin(left) );
			else if(cmd->info.opcode == eCos)
				stack.push( ::cos(left) );
			else if(cmd->info.opcode == eTan)
				stack.push( ::sqrt(left) );
			else if(cmd->info.opcode == eArcSin)
				stack.push( ::asin(left) );
			else if(cmd->info.opcode == eArcCos)
				stack.push( ::acos(left) );
			else if(cmd->info.opcode == eArcTan)
				stack.push( ::atan(left) );
			else if(cmd->info.opcode == eCeil)
				stack.push( ::ceil(left) );
			else if(cmd->info.opcode == eFloor)
				stack.push( ::floor(left) );
			else
				daeDeclareAndThrowException(exInvalidCall);
		}
		else if(cmd->info.kind == eBinary)
		{
			left  = GetOperand<real_t>(pEC->m_pDataProxy, cmd->info.loperand, cmd->lvalue, stack);
			right = GetOperand<real_t>(pEC->m_pDataProxy, cmd->info.roperand, cmd->rvalue, stack);

			if(cmd->info.opcode == ePlus)
				stack.push( left + right );
			else if(cmd->info.opcode == eMinus)
				stack.push( left - right );
			else if(cmd->info.opcode == eMulti)
				stack.push( left * right );
			else if(cmd->info.opcode == eDivide)
				stack.push( left / right );
			else if(cmd->info.opcode == ePower)
				stack.push( ::pow(left, right) );
			else if(cmd->info.opcode == eMin)
				stack.push( left < right ? left : right );
			else if(cmd->info.opcode == eMax)
				stack.push( left > right ? left : right );
			else
				daeDeclareAndThrowException(exInvalidCall);
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}
	if(stack.size() != 1)
		daeDeclareAndThrowException(exInvalidCall);
	dResult = stack.top_pop();
}

template<class TYPE>
inline TYPE GetJacobianOperand(const daeDataProxy_t* pDataProxy, const size_t operand, const real_t value, const size_t nCurrentVariableIndex, const real_t dInverseTimeStep, daeCommandStack<TYPE>& stack)
{
	if(operand == eConstant)
	{
		return adouble(value);
	}
	else if(operand == eDomain)
	{
		return adouble(value);
	}
	else if(operand == eValue)
	{
		return adouble(*pDataProxy->GetValue((size_t)value), ((size_t)value == nCurrentVariableIndex ? 1 : 0) );
	}
	else if(operand == eTimeDerivative)
	{
		return adouble(*pDataProxy->GetTimeDerivative((size_t)value), ((size_t)value == nCurrentVariableIndex ? dInverseTimeStep : 0) );
	}
	else if(operand == eFromStack)
	{
		return stack.top_pop();
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall);
	}
}

void daeFPU::fpuJacobian(const daeExecutionContext* pEC, const vector<daeFPUCommand*>& ptrarrCommands, real_t& dResult)
{
	daeFPUCommand* cmd;
	adouble left, right;
	daeCommandStack<adouble> stack;
	vector<daeFPUCommand*>::const_reverse_iterator it;

// I have to inverse iterate!!!
	for(it = ptrarrCommands.rbegin(); it != ptrarrCommands.rend(); it++)
	{
		cmd = *it;
		if(cmd->info.kind == eUnary)
		{
			left = GetJacobianOperand<adouble>(pEC->m_pDataProxy, cmd->info.loperand, cmd->lvalue, pEC->m_nCurrentVariableIndexForJacobianEvaluation, pEC->m_dInverseTimeStep, stack);

			if(cmd->info.opcode == eSign)
				stack.push( -left );
			else if(cmd->info.opcode == eSqrt)
				stack.push( sqrt(left) );
			else if(cmd->info.opcode == eExp)
				stack.push( exp(left) );
			else if(cmd->info.opcode == eLog)
				stack.push( log10(left) );
			else if(cmd->info.opcode == eLn)
				stack.push( log(left) );
			else if(cmd->info.opcode == eAbs)
				stack.push( abs(left) );
			else if(cmd->info.opcode == eSin)
				stack.push( sin(left) );
			else if(cmd->info.opcode == eCos)
				stack.push( cos(left) );
			else if(cmd->info.opcode == eTan)
				stack.push( sqrt(left) );
			else if(cmd->info.opcode == eArcSin)
				stack.push( asin(left) );
			else if(cmd->info.opcode == eArcCos)
				stack.push( acos(left) );
			else if(cmd->info.opcode == eArcTan)
				stack.push( atan(left) );
			else if(cmd->info.opcode == eCeil)
				stack.push( ceil(left) );
			else if(cmd->info.opcode == eFloor)
				stack.push( floor(left) );
			else
				daeDeclareAndThrowException(exInvalidCall);
		}
		else if(cmd->info.kind == eBinary)
		{
			left  = GetJacobianOperand<adouble>(pEC->m_pDataProxy, cmd->info.loperand, cmd->lvalue, pEC->m_nCurrentVariableIndexForJacobianEvaluation, pEC->m_dInverseTimeStep, stack);
			right = GetJacobianOperand<adouble>(pEC->m_pDataProxy, cmd->info.roperand, cmd->rvalue, pEC->m_nCurrentVariableIndexForJacobianEvaluation, pEC->m_dInverseTimeStep, stack);

			if(cmd->info.opcode == ePlus)
				stack.push( left + right );
			else if(cmd->info.opcode == eMinus)
				stack.push( left - right );
			else if(cmd->info.opcode == eMulti)
				stack.push( left * right );
			else if(cmd->info.opcode == eDivide)
				stack.push( left / right );
			else if(cmd->info.opcode == ePower)
				stack.push( pow(left, right) );
			else if(cmd->info.opcode == eMin)
				stack.push( min(left, right) );
			else if(cmd->info.opcode == eMax)
				stack.push( max(left, right) );
			else
				daeDeclareAndThrowException(exInvalidCall);
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}
	if(stack.size() != 1)
		daeDeclareAndThrowException(exInvalidCall);
	dResult = stack.top_pop().getDerivative();
}


}
}
