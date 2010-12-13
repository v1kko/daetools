#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
using namespace dae;
#include <typeinfo>
#include "xmlfunctions.h"
using namespace dae::xml;

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	adSetupParameterNode
**********************************************************************************************/
adSetupParameterNode::adSetupParameterNode(daeParameter* pParameter,
										   vector<daeDomainIndex>& arrDomains) 
                    : m_pParameter(pParameter),
					  m_arrDomains(arrDomains)
{
}

adSetupParameterNode::adSetupParameterNode()
{
	m_pParameter = NULL;
}

adSetupParameterNode::~adSetupParameterNode()
{
}

adouble adSetupParameterNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pParameter)
		daeDeclareAndThrowException(exInvalidCall);

	adouble tmp;
	size_t N        = m_arrDomains.size();
	size_t* indexes = new size_t[N];

	for(size_t i = 0; i < N; i++)
	{
		if(m_arrDomains[i].m_eType == eConstantIndex)
		{
			if(m_arrDomains[i].m_nIndex == ULONG_MAX)
				daeDeclareAndThrowException(exInvalidCall);
			indexes[i] = m_arrDomains[i].m_nIndex;
		}
		else if(m_arrDomains[i].m_eType == eDomainIterator)
		{
			if(!m_arrDomains[i].m_pDEDI)
				daeDeclareAndThrowException(exInvalidCall);
			indexes[i] = m_arrDomains[i].m_pDEDI->GetCurrentIndex();
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}

	tmp = m_pParameter->Create_adouble(indexes, N);
	delete[] indexes;
	return tmp;
}

adNode* adSetupParameterNode::Clone(void) const
{
	return new adSetupParameterNode(*this);
}

string adSetupParameterNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	return textCreator::Variable(strName, strarrIndexes);
}

string adSetupParameterNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	return latexCreator::Variable(strName, strarrIndexes);
}

void adSetupParameterNode::Open(io::xmlTag_t* pTag)
{
}

void adSetupParameterNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Parameter";
	pTag->SaveObjectRef(strName, m_pParameter);

	strName = "DomainIterators";
	pTag->SaveObjectArray(strName, m_arrDomains);
}

void adSetupParameterNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupParameterNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupParameterNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
}

/*********************************************************************************************
	adSetupDomainIteratorNode
**********************************************************************************************/
adSetupDomainIteratorNode::adSetupDomainIteratorNode(daeDistributedEquationDomainInfo* pDEDI)
			             : m_pDEDI(pDEDI)
{
}

adSetupDomainIteratorNode::adSetupDomainIteratorNode()
{
	m_pDEDI = NULL;
}

adSetupDomainIteratorNode::~adSetupDomainIteratorNode()
{
}

adouble adSetupDomainIteratorNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pDEDI)
		daeDeclareAndThrowException(exInvalidCall);

	daeDomain* pDomain = dynamic_cast<daeDomain*>(m_pDEDI->GetDomain());
	size_t nIndex      = m_pDEDI->GetCurrentIndex();
	return (*pDomain)[nIndex];
}

adNode* adSetupDomainIteratorNode::Clone(void) const
{
	return new adSetupDomainIteratorNode(*this);
}

string adSetupDomainIteratorNode::SaveAsPlainText(const daeSaveAsMathMLContext* /*c*/) const
{
	vector<string> strarrIndexes;
	string strName = m_pDEDI->GetName();
	return textCreator::Variable(strName, strarrIndexes);
}

string adSetupDomainIteratorNode::SaveAsLatex(const daeSaveAsMathMLContext* /*c*/) const
{
	vector<string> strarrIndexes;
	string strName = m_pDEDI->GetName();
	return latexCreator::Variable(strName, strarrIndexes);
}

void adSetupDomainIteratorNode::Open(io::xmlTag_t* /*pTag*/)
{
}

void adSetupDomainIteratorNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "DEDI";
	pTag->SaveObjectRef(strName, m_pDEDI);
}

void adSetupDomainIteratorNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	string strName = m_pDEDI->GetName();
	xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupDomainIteratorNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	string strName = m_pDEDI->GetName();
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupDomainIteratorNode::AddVariableIndexToArray(map<size_t, size_t>& /*mapIndexes*/)
{
}

/*********************************************************************************************
	adSetupVariableNode
**********************************************************************************************/
adSetupVariableNode::adSetupVariableNode(daeVariable* pVariable,
										 vector<daeDomainIndex>& arrDomains) 
                    : m_pVariable(pVariable),
					  m_arrDomains(arrDomains)
{
}

adSetupVariableNode::adSetupVariableNode()
{
	m_pVariable = NULL;
}

adSetupVariableNode::~adSetupVariableNode()
{
}

adouble adSetupVariableNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);

	adouble tmp;
	size_t N        = m_arrDomains.size();
	size_t* indexes = new size_t[N];

	for(size_t i = 0; i < N; i++)
	{
		if(m_arrDomains[i].m_eType == eConstantIndex)
		{
			if(m_arrDomains[i].m_nIndex == ULONG_MAX)
				daeDeclareAndThrowException(exInvalidCall);
			indexes[i] = m_arrDomains[i].m_nIndex;
		}
		else if(m_arrDomains[i].m_eType == eDomainIterator)
		{
			if(!m_arrDomains[i].m_pDEDI)
				daeDeclareAndThrowException(exInvalidCall);
			indexes[i] = m_arrDomains[i].m_pDEDI->GetCurrentIndex();
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}

	tmp = m_pVariable->Create_adouble(indexes, N);
	delete[] indexes;
	return tmp;
}

adNode* adSetupVariableNode::Clone(void) const
{
	return new adSetupVariableNode(*this);
}

string adSetupVariableNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return textCreator::Variable(strName, strarrIndexes);
}

string adSetupVariableNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return latexCreator::Variable(strName, strarrIndexes);
}

void adSetupVariableNode::Open(io::xmlTag_t* /*pTag*/)
{
}

void adSetupVariableNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Variable";
	pTag->SaveObjectRef(strName, m_pVariable);

	strName = "DomainIterators";
	pTag->SaveObjectArray(strName, m_arrDomains);
}

void adSetupVariableNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupVariableNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupVariableNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
}

/*********************************************************************************************
	adSetupTimeDerivativeNode
**********************************************************************************************/
adSetupTimeDerivativeNode::adSetupTimeDerivativeNode(daeVariable* pVariable, 
													 size_t nDegree, 
													 vector<daeDomainIndex>& arrDomains)
                         : m_pVariable(pVariable), 
						   m_nDegree(nDegree),
						   m_arrDomains(arrDomains)
{
}

adSetupTimeDerivativeNode::adSetupTimeDerivativeNode()
{
	m_pVariable = NULL;
	m_nDegree   = 0;
}

adSetupTimeDerivativeNode::~adSetupTimeDerivativeNode()
{
}

adouble adSetupTimeDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);

	adouble tmp;
	size_t N        = m_arrDomains.size();
	size_t* indexes = new size_t[N];

	for(size_t i = 0; i < N; i++)
	{
		if(m_arrDomains[i].m_eType == eConstantIndex)
		{
			if(m_arrDomains[i].m_nIndex == ULONG_MAX)
				daeDeclareAndThrowException(exInvalidCall);
			indexes[i] = m_arrDomains[i].m_nIndex;
		}
		else if(m_arrDomains[i].m_eType == eDomainIterator)
		{
			if(!m_arrDomains[i].m_pDEDI)
				daeDeclareAndThrowException(exInvalidCall);
			indexes[i] = m_arrDomains[i].m_pDEDI->GetCurrentIndex();
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}

	tmp = m_pVariable->Calculate_dt(indexes, N);
	delete[] indexes;
	return tmp;
}

adNode* adSetupTimeDerivativeNode::Clone(void) const
{
	return new adSetupTimeDerivativeNode(*this);
}

string adSetupTimeDerivativeNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return textCreator::TimeDerivative(m_nDegree, strName, strarrIndexes);
}

string adSetupTimeDerivativeNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return latexCreator::TimeDerivative(m_nDegree, strName, strarrIndexes);
}

void adSetupTimeDerivativeNode::Open(io::xmlTag_t* pTag)
{
}

void adSetupTimeDerivativeNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Variable";
	pTag->SaveObjectRef(strName, m_pVariable);

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "DomainIterators";
	pTag->SaveObjectArray(strName, m_arrDomains);
}

void adSetupTimeDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlContentCreator::TimeDerivative(pTag, m_nDegree, strName, strarrIndexes);
}

void adSetupTimeDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlPresentationCreator::TimeDerivative(pTag, m_nDegree, strName, strarrIndexes);
}

void adSetupTimeDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
}

/*********************************************************************************************
	adSetupPartialDerivativeNode
**********************************************************************************************/
adSetupPartialDerivativeNode::adSetupPartialDerivativeNode(daeVariable* pVariable, 
														   size_t nDegree, 
														   vector<daeDomainIndex>& arrDomains,
														   daeDomain* pDomain)
                            : m_pVariable(pVariable), 
						  	  m_pDomain(pDomain),
							  m_nDegree(nDegree), 
							  m_arrDomains(arrDomains)
{
}

adSetupPartialDerivativeNode::adSetupPartialDerivativeNode()
{
	m_pVariable = NULL;
	m_pDomain   = NULL;
	m_nDegree   = 0;
}

adSetupPartialDerivativeNode::~adSetupPartialDerivativeNode()
{
}

adouble adSetupPartialDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidCall);

	adouble tmp;
	size_t N        = m_arrDomains.size();
	size_t* indexes = new size_t[N];

	for(size_t i = 0; i < N; i++)
	{
		if(m_arrDomains[i].m_eType == eConstantIndex)
		{
			if(m_arrDomains[i].m_nIndex == ULONG_MAX)
				daeDeclareAndThrowException(exInvalidCall);
			indexes[i] = m_arrDomains[i].m_nIndex;
		}
		else if(m_arrDomains[i].m_eType == eDomainIterator)
		{
			if(!m_arrDomains[i].m_pDEDI)
				daeDeclareAndThrowException(exInvalidCall);
			indexes[i] = m_arrDomains[i].m_pDEDI->GetCurrentIndex();
		}
		else
		{
			daeDeclareAndThrowException(exInvalidCall);
		}
	}

	tmp = m_pVariable->partial(m_nDegree, *m_pDomain, indexes, N);
	delete[] indexes;
	return tmp;
}

adNode* adSetupPartialDerivativeNode::Clone(void) const
{
	return new adSetupPartialDerivativeNode(*this);
}

string adSetupPartialDerivativeNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	return textCreator::PartialDerivative(m_nDegree, strName, strDomainName, strarrIndexes);
}

string adSetupPartialDerivativeNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	return latexCreator::PartialDerivative(m_nDegree, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNode::Open(io::xmlTag_t* pTag)
{
}

void adSetupPartialDerivativeNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Variable";
	pTag->SaveObjectRef(strName, m_pVariable);

	strName = "Domain";
	pTag->SaveObjectRef(strName, m_pDomain);

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "DomainIterators";
	pTag->SaveObjectArray(strName, m_arrDomains);
}

void adSetupPartialDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	xmlContentCreator::PartialDerivative(pTag, m_nDegree, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrDomains, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	xmlPresentationCreator::PartialDerivative(pTag, m_nDegree, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
}



}
}
