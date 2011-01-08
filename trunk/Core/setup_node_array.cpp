#include "stdafx.h"
#include "coreimpl.h"
#include "nodes_array.h"
#include "xmlfunctions.h"
using namespace dae;
#include <typeinfo>
#include "xmlfunctions.h"
using namespace dae::xml;

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	adSetupParameterNodeArray
**********************************************************************************************/
adSetupParameterNodeArray::adSetupParameterNodeArray(daeParameter* pParameter,
													 vector<daeArrayRange>& arrRanges) 
                    : m_pParameter(pParameter),
					  m_arrRanges(arrRanges)
{
}

adSetupParameterNodeArray::adSetupParameterNodeArray()
{
	m_pParameter = NULL;
}

adSetupParameterNodeArray::~adSetupParameterNodeArray()
{
}

void adSetupParameterNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	arrRanges = m_arrRanges;
}

size_t adSetupParameterNodeArray::GetSize(void) const
{
	size_t size = 1;
	for(size_t i = 0; i < m_arrRanges.size(); i++)
		size *= m_arrRanges[i].GetNoPoints();
    return size;
}

adouble_array adSetupParameterNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pParameter)
		daeDeclareAndThrowException(exInvalidCall);
	if(m_arrRanges.empty())
		daeDeclareAndThrowException(exInvalidCall);

	adouble_array tmp;
	size_t N              = m_arrRanges.size();
	daeArrayRange* ranges = new daeArrayRange[N];

	for(size_t i = 0; i < N; i++)
		ranges[i] = m_arrRanges[i];
	tmp = m_pParameter->Create_adouble_array(ranges, N);
	delete[] ranges;
	return tmp;
}

adNodeArray* adSetupParameterNodeArray::Clone(void) const
{
	return new adSetupParameterNodeArray(*this);
}

string adSetupParameterNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	return textCreator::Variable(strName, strarrIndexes);
}

string adSetupParameterNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	return latexCreator::Variable(strName, strarrIndexes);
}

void adSetupParameterNodeArray::Open(io::xmlTag_t* pTag)
{
}

void adSetupParameterNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Parameter";
	pTag->SaveObjectRef(strName, m_pParameter);

	strName = "Ranges";
	pTag->SaveObjectArray(strName, m_arrRanges);
}

void adSetupParameterNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupParameterNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupParameterNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

/*********************************************************************************************
	adSetupVariableNodeArray
**********************************************************************************************/
adSetupVariableNodeArray::adSetupVariableNodeArray(daeVariable* pVariable,
												   vector<daeArrayRange>& arrRanges) 
                    : m_pVariable(pVariable),
					  m_arrRanges(arrRanges)
{
}

adSetupVariableNodeArray::adSetupVariableNodeArray()
{
	m_pVariable = NULL;
}

adSetupVariableNodeArray::~adSetupVariableNodeArray()
{
}

void adSetupVariableNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	arrRanges = m_arrRanges;
}

size_t adSetupVariableNodeArray::GetSize(void) const
{
    size_t size = 1;
    for(size_t i = 0; i < m_arrRanges.size(); i++)
        size *= m_arrRanges[i].GetNoPoints();
    return size;
}

adouble_array adSetupVariableNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);
	if(m_arrRanges.empty())
		daeDeclareAndThrowException(exInvalidCall);

	adouble_array tmp;
	size_t N              = m_arrRanges.size();
	daeArrayRange* ranges = new daeArrayRange[N];

	for(size_t i = 0; i < N; i++)
		ranges[i] = m_arrRanges[i];
	tmp = m_pVariable->Create_adouble_array(ranges, N);
	delete[] ranges;

	return tmp;
}

adNodeArray* adSetupVariableNodeArray::Clone(void) const
{
	return new adSetupVariableNodeArray(*this);
}

string adSetupVariableNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return textCreator::Variable(strName, strarrIndexes);
}

string adSetupVariableNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return latexCreator::Variable(strName, strarrIndexes);
}

void adSetupVariableNodeArray::Open(io::xmlTag_t* /*pTag*/)
{
}

void adSetupVariableNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Variable";
	pTag->SaveObjectRef(strName, m_pVariable);

	strName = "Ranges";
	pTag->SaveObjectArray(strName, m_arrRanges);
}

void adSetupVariableNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupVariableNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupVariableNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	daeDeclareAndThrowException(exInvalidCall)
}

/*********************************************************************************************
	adSetupTimeDerivativeNodeArray
**********************************************************************************************/
adSetupTimeDerivativeNodeArray::adSetupTimeDerivativeNodeArray(daeVariable* pVariable, 
													           size_t nDegree, 
													           vector<daeArrayRange>& arrRanges)
                         : m_pVariable(pVariable), 
						   m_nDegree(nDegree),
						   m_arrRanges(arrRanges)
{
}

adSetupTimeDerivativeNodeArray::adSetupTimeDerivativeNodeArray()
{
	m_pVariable = NULL;
	m_nDegree   = 0;
}

adSetupTimeDerivativeNodeArray::~adSetupTimeDerivativeNodeArray()
{
}

void adSetupTimeDerivativeNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	arrRanges = m_arrRanges;
}

size_t adSetupTimeDerivativeNodeArray::GetSize(void) const
{
    size_t size = 1;
    for(size_t i = 0; i < m_arrRanges.size(); i++)
        size *= m_arrRanges[i].GetNoPoints();
    return size;
}

adouble_array adSetupTimeDerivativeNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);
	if(m_arrRanges.empty())
		daeDeclareAndThrowException(exInvalidCall);

	adouble_array tmp;
	size_t N              = m_arrRanges.size();
	daeArrayRange* ranges = new daeArrayRange[N];

	for(size_t i = 0; i < N; i++)
		ranges[i] = m_arrRanges[i];
	tmp = m_pVariable->Calculate_dt_array(ranges, N);
	delete[] ranges;
	return tmp;
}

adNodeArray* adSetupTimeDerivativeNodeArray::Clone(void) const
{
	return new adSetupTimeDerivativeNodeArray(*this);
}

string adSetupTimeDerivativeNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return textCreator::TimeDerivative(m_nDegree, strName, strarrIndexes);
}

string adSetupTimeDerivativeNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return latexCreator::TimeDerivative(m_nDegree, strName, strarrIndexes);
}

void adSetupTimeDerivativeNodeArray::Open(io::xmlTag_t* pTag)
{
}

void adSetupTimeDerivativeNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Variable";
	pTag->SaveObjectRef(strName, m_pVariable);

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "Ranges";
	pTag->SaveObjectArray(strName, m_arrRanges);
}

void adSetupTimeDerivativeNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlContentCreator::TimeDerivative(pTag, m_nDegree, strName, strarrIndexes);
}

void adSetupTimeDerivativeNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlPresentationCreator::TimeDerivative(pTag, m_nDegree, strName, strarrIndexes);
}

void adSetupTimeDerivativeNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	daeDeclareAndThrowException(exInvalidCall)
}

/*********************************************************************************************
	adSetupPartialDerivativeNodeArray
**********************************************************************************************/
adSetupPartialDerivativeNodeArray::adSetupPartialDerivativeNodeArray(daeVariable* pVariable, 
														   size_t nDegree, 
														   vector<daeArrayRange>& arrRanges,
														   daeDomain* pDomain)
                            : m_pVariable(pVariable), 
							  m_pDomain(pDomain),
							  m_nDegree(nDegree), 
							  m_arrRanges(arrRanges)
{
}

adSetupPartialDerivativeNodeArray::adSetupPartialDerivativeNodeArray()
{
	m_pVariable = NULL;
	m_pDomain   = NULL;
	m_nDegree   = 0;
}

adSetupPartialDerivativeNodeArray::~adSetupPartialDerivativeNodeArray()
{
}

void adSetupPartialDerivativeNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	arrRanges = m_arrRanges;
}

size_t adSetupPartialDerivativeNodeArray::GetSize(void) const
{
    size_t size = 1;
    for(size_t i = 0; i < m_arrRanges.size(); i++)
        size *= m_arrRanges[i].GetNoPoints();
    return size;
}

adouble_array adSetupPartialDerivativeNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);
	if(m_arrRanges.empty())
		daeDeclareAndThrowException(exInvalidCall);

	adouble_array tmp;
	size_t N              = m_arrRanges.size();
	daeArrayRange* ranges = new daeArrayRange[N];

	for(size_t i = 0; i < N; i++)
		ranges[i] = m_arrRanges[i];
	tmp = m_pVariable->partial_array(m_nDegree, *m_pDomain, ranges, N);
	delete[] ranges;
	return tmp;
}

adNodeArray* adSetupPartialDerivativeNodeArray::Clone(void) const
{
	return new adSetupPartialDerivativeNodeArray(*this);
}

string adSetupPartialDerivativeNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	return textCreator::PartialDerivative(m_nDegree, strName, strDomainName, strarrIndexes);
}

string adSetupPartialDerivativeNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	return latexCreator::PartialDerivative(m_nDegree, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNodeArray::Open(io::xmlTag_t* pTag)
{
}

void adSetupPartialDerivativeNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Variable";
	pTag->SaveObjectRef(strName, m_pVariable);

	strName = "Domain";
	pTag->SaveObjectRef(strName, m_pDomain);

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "Ranges";
	pTag->SaveObjectArray(strName, m_arrRanges);
}

void adSetupPartialDerivativeNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	xmlContentCreator::PartialDerivative(pTag, m_nDegree, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	FillDomains(m_arrRanges, strarrIndexes);
	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	xmlPresentationCreator::PartialDerivative(pTag, m_nDegree, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	daeDeclareAndThrowException(exInvalidCall)
}



}
}
