#include "stdafx.h"
#include "coreimpl.h"
#include "nodes_array.h"
#include "xmlfunctions.h"
using namespace dae;
#include <typeinfo>
using namespace dae::xml;
using namespace boost;

namespace dae
{
namespace core
{
/*********************************************************************************************
    adSetupDomainNodeArray
**********************************************************************************************/
adSetupDomainNodeArray::adSetupDomainNodeArray(daeDomain* pDomain,
                                               const daeArrayRange &range)
                      : m_pDomain(pDomain),
                        m_Range(range)
{
}

adSetupDomainNodeArray::adSetupDomainNodeArray()
{
    m_pDomain = NULL;
}

adSetupDomainNodeArray::~adSetupDomainNodeArray()
{
}

size_t adSetupDomainNodeArray::GetSize(void) const
{
    return m_Range.GetNoPoints();
}

adouble_array adSetupDomainNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(!m_pDomain)
        daeDeclareAndThrowException(exInvalidCall);

    adouble_array tmp;
    std::vector<size_t> narrPoints;
    m_Range.GetPoints(narrPoints);

    tmp.Resize(narrPoints.size());
    for(size_t i = 0; i < narrPoints.size(); i++)
    {
        adouble ad = (*m_pDomain)[i];
        tmp.SetItem(i, ad);
    }

    return tmp;
}

const quantity adSetupDomainNodeArray::GetQuantity(void) const
{
    if(!m_pDomain)
        daeDeclareAndThrowException(exInvalidCall);
    return quantity(0.0, m_pDomain->GetUnits());
}

adNodeArray* adSetupDomainNodeArray::Clone(void) const
{
    return new adSetupDomainNodeArray(*this);
}

void adSetupDomainNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    vector<string> strarrIndexes;

    strarrIndexes.push_back(m_Range.GetRangeAsString());

    if(eLanguage == eCDAE)
        strContent += daeGetStrippedRelativeName(c.m_pModel, m_pDomain) + ".array(" + toString(strarrIndexes) + ")";
    else if(eLanguage == ePYDAE)
        strContent += /*"self."*/ daeGetStrippedRelativeName(c.m_pModel, m_pDomain) + ".array(" + toString(strarrIndexes) + ")";
    else
        daeDeclareAndThrowException(exNotImplemented);
}

//string adSetupDomainNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	FillDomains(m_arrRanges, strarrIndexes);
//	string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
//	return textCreator::Variable(strName, strarrIndexes);
//}

string adSetupDomainNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    strarrIndexes.push_back(m_Range.GetRangeAsString());
    string strName = daeGetRelativeName(c->m_pModel, m_pDomain) + ".array";
    return latexCreator::Variable(strName, strarrIndexes);
}

void adSetupDomainNodeArray::Open(io::xmlTag_t* pTag)
{
}

void adSetupDomainNodeArray::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "Domain";
    pTag->SaveObjectRef(strName, m_pDomain);

    strName = "Range";
    pTag->SaveObject(strName, &m_Range);
}

void adSetupDomainNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    strarrIndexes.push_back(m_Range.GetRangeAsString());
    string strName = daeGetRelativeName(c->m_pModel, m_pDomain) + ".array";
    xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupDomainNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    strarrIndexes.push_back(m_Range.GetRangeAsString());
    string strName = daeGetRelativeName(c->m_pModel, m_pDomain) + ".array";
    xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupDomainNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

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

const quantity adSetupParameterNodeArray::GetQuantity(void) const
{
    if(!m_pParameter)
        daeDeclareAndThrowException(exInvalidCall);
    return quantity(0.0, m_pParameter->GetUnits());
}

adNodeArray* adSetupParameterNodeArray::Clone(void) const
{
    return new adSetupParameterNodeArray(*this);
}

void adSetupParameterNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    vector<string> strarrIndexes;

    FillDomains(m_arrRanges, strarrIndexes);

    if(eLanguage == eCDAE)
        strContent += daeGetStrippedRelativeName(c.m_pModel, m_pParameter) + ".array(" + toString(strarrIndexes) + ")";
    else if(eLanguage == ePYDAE)
        strContent += /*"self."*/ daeGetStrippedRelativeName(c.m_pModel, m_pParameter) + ".array(" + toString(strarrIndexes) + ")";
    else
        daeDeclareAndThrowException(exNotImplemented);
}
//string adSetupParameterNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	FillDomains(m_arrRanges, strarrIndexes);
//	string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
//	return textCreator::Variable(strName, strarrIndexes);
//}

string adSetupParameterNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pParameter) + ".array";
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

void adSetupParameterNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pParameter) + ".array";
    xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupParameterNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pParameter) + ".array";
    xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupParameterNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

/*********************************************************************************************
    adCustomNodeArray
**********************************************************************************************/
adCustomNodeArray::adCustomNodeArray(const std::vector<adNodePtr>& ptrarrNodes)
                      : m_ptrarrNodes(ptrarrNodes)
{
}

adCustomNodeArray::adCustomNodeArray()
{
}

adCustomNodeArray::~adCustomNodeArray()
{
}

void adCustomNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{
}

size_t adCustomNodeArray::GetSize(void) const
{
    return m_ptrarrNodes.size();
}

// Here we have to evaluate every node and return adouble_array with adRuntimeCustomNodeArray as a node
adouble_array adCustomNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    adouble_array tmp;
    size_t N = m_ptrarrNodes.size();

    if(pExecutionContext->m_pDataProxy->GetGatherInfo())
    {
        // Evaluate all nodes to obtain runtime ones and reate a new adCustomNodeArray
        tmp.setGatherInfo(true);

        std::vector<adNodePtr> ptrarrNodes;

        ptrarrNodes.resize(N);
        for(size_t i = 0; i < N; i++)
            ptrarrNodes[i] = m_ptrarrNodes[i]->Evaluate(pExecutionContext).node;

        tmp.node = adNodeArrayPtr(new adCustomNodeArray(ptrarrNodes));

        return tmp;
    }

    tmp.Resize(N);
    for(size_t i = 0; i < N; i++)
        tmp[i] = m_ptrarrNodes[i]->Evaluate(pExecutionContext);

    return tmp;
}

const quantity adCustomNodeArray::GetQuantity(void) const
{
    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);
    return m_ptrarrNodes[0]->GetQuantity();
}

adNodeArray* adCustomNodeArray::Clone(void) const
{
    return new adCustomNodeArray(*this);
}

void adCustomNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    size_t N = m_ptrarrNodes.size();

    if(eLanguage == eCDAE)
    {
        strContent += "CustomArray(";
        for(size_t i = 0; i < N; i++)
        {
            if(i != 0)
                strContent += ", ";
            m_ptrarrNodes[i]->Export(strContent, eLanguage, c);
        }
        strContent += ")";
    }
    else if(eLanguage == ePYDAE)
    {
        strContent += "CustomArray(";
        for(size_t i = 0; i < N; i++)
        {
            if(i != 0)
                strContent += ", ";
            m_ptrarrNodes[i]->Export(strContent, eLanguage, c);
        }
        strContent += ")";
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }
}
//string adCustomNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//}

string adCustomNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    string strResult;

    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    strResult  = "{ \\left[";

    size_t N = m_ptrarrNodes.size();
    for(size_t i = 0; i < N; i++)
    {
        if(i != 0)
            strResult += ", ";
        strResult += m_ptrarrNodes[i]->SaveAsLatex(c);
    }

    strResult  += " \\right] } ";
    return strResult;
}

void adCustomNodeArray::Open(io::xmlTag_t* /*pTag*/)
{
}

void adCustomNodeArray::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "Nodes";
    pTag->SaveObjectArray(strName, m_ptrarrNodes);
}

void adCustomNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    daeDeclareAndThrowException(exNotImplemented);
}

void adCustomNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    string strName, strValue;
    io::xmlTag_t *mrow;

    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    strName  = "mrow";
    strValue = "";
    mrow = pTag->AddTag(strName, strValue);

    size_t N = m_ptrarrNodes.size();
    for(size_t i = 0; i < N; i++)
    {
        if(i != 0)
        {
            strName  = "mo";
            strValue = ",";
            mrow->AddTag(strName, strValue);
        }
        m_ptrarrNodes[i]->SaveAsPresentationMathML(mrow, c);
    }
}

void adCustomNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    daeDeclareAndThrowException(exInvalidCall)
}

/*********************************************************************************************
    adRuntimeCustomNodeArray
**********************************************************************************************/
/*
// This is equivalent to adSetupCustomNodeArray, just the nodes here are runtime nodes
adRuntimeCustomNodeArray::adRuntimeCustomNodeArray(const std::vector<adNodePtr>& ptrarrNodes)
                        : m_ptrarrNodes(ptrarrNodes)
{
}

adRuntimeCustomNodeArray::adRuntimeCustomNodeArray()
{
}

adRuntimeCustomNodeArray::~adRuntimeCustomNodeArray()
{
}

void adRuntimeCustomNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{
}

size_t adRuntimeCustomNodeArray::GetSize(void) const
{
    return m_ptrarrNodes.size();
}

adouble_array adRuntimeCustomNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    adouble_array tmp;
    size_t N = m_ptrarrNodes.size();

    tmp.Resize(N);
    for(size_t i = 0; i < N; i++)
        tmp[i] = m_ptrarrNodes[i]->Evaluate(pExecutionContext);

    return tmp;
}

const quantity adRuntimeCustomNodeArray::GetQuantity(void) const
{
    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);
    return m_ptrarrNodes[0]->GetQuantity();
}

adNodeArray* adRuntimeCustomNodeArray::Clone(void) const
{
    return new adRuntimeCustomNodeArray(*this);
}

void adRuntimeCustomNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    size_t N = m_ptrarrNodes.size();

    if(eLanguage == eCDAE)
    {
        strContent += "Array(";
        for(size_t i = 0; i < N; i++)
        {
            if(i != 0)
                strContent += ", ";
            m_ptrarrNodes[i]->Export(strContent, eLanguage, c);
        }
        strContent += ")";
    }
    else if(eLanguage == ePYDAE)
    {
        strContent += "Array(";
        for(size_t i = 0; i < N; i++)
        {
            if(i != 0)
                strContent += ", ";
            m_ptrarrNodes[i]->Export(strContent, eLanguage, c);
        }
        strContent += ")";
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }
}
//string adRuntimeCustomNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//}

string adRuntimeCustomNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    string strResult;

    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    strResult  = "{ ";

    size_t N = m_ptrarrNodes.size();
    for(size_t i = 0; i < N; i++)
    {
        if(i != 0)
            strResult += ", ";
        strResult += m_ptrarrNodes[i]->SaveAsLatex(c);
    }

    strResult  += "} ";
    return strResult;
}

void adRuntimeCustomNodeArray::Open(io::xmlTag_t* pTag)
{
}

void adRuntimeCustomNodeArray::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "Nodes";
    pTag->SaveObjectArray(strName, m_ptrarrNodes);
}

void adRuntimeCustomNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adRuntimeCustomNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    string strName, strValue;
    io::xmlTag_t *mrow;

    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    strName  = "mrow";
    strValue = "";
    mrow = pTag->AddTag(strName, strValue);

    size_t N = m_ptrarrNodes.size();
    for(size_t i = 0; i < N; i++)
    {
        if(i != 0)
        {
            strName  = "mo";
            strValue = ",";
            mrow->AddTag(strName, strValue);
        }
        m_ptrarrNodes[i]->SaveAsPresentationMathML(mrow, c);
    }
}

void adRuntimeCustomNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    size_t N = m_ptrarrNodes.size();
    for(size_t i = 0; i < N; i++)
    {
        m_ptrarrNodes[i]->AddVariableIndexToArray(mapIndexes, bAddFixed);
    }
}

bool adRuntimeCustomNodeArray::IsLinear(void) const
{
    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    size_t N = m_ptrarrNodes.size();
    for(size_t i = 0; i < N; i++)
    {
        if(!m_ptrarrNodes[i]->IsLinear())
            return false;
    }
    return true;
}

bool adRuntimeCustomNodeArray::IsFunctionOfVariables(void) const
{
    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    size_t N = m_ptrarrNodes.size();
    for(size_t i = 0; i < N; i++)
    {
        if(m_ptrarrNodes[i]->IsFunctionOfVariables())
            return true;
    }
    return false;
}

bool adRuntimeCustomNodeArray::IsDifferential(void) const
{
    if(m_ptrarrNodes.empty())
        daeDeclareAndThrowException(exInvalidCall);

    size_t N = m_ptrarrNodes.size();
    for(size_t i = 0; i < N; i++)
    {
        if(m_ptrarrNodes[i]->IsDifferential())
            return true;
    }
    return false;
}
*/

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

const quantity adSetupVariableNodeArray::GetQuantity(void) const
{
    if(!m_pVariable)
        daeDeclareAndThrowException(exInvalidCall);
    return quantity(0.0, m_pVariable->GetVariableType()->GetUnits());
}

adNodeArray* adSetupVariableNodeArray::Clone(void) const
{
    return new adSetupVariableNodeArray(*this);
}

void adSetupVariableNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    string strName;
    vector<string> strarrIndexes;

    FillDomains(m_arrRanges, strarrIndexes);

    if(eLanguage == eCDAE)
        strContent += daeGetStrippedRelativeName(c.m_pModel, m_pVariable) + ".array(" + toString(strarrIndexes) + ")";
    else if(eLanguage == ePYDAE)
        strContent += /*"self."*/ daeGetStrippedRelativeName(c.m_pModel, m_pVariable) + ".array(" + toString(strarrIndexes) + ")";
    else
        daeDeclareAndThrowException(exNotImplemented);
}
//string adSetupVariableNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	FillDomains(m_arrRanges, strarrIndexes);
//	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
//	return textCreator::Variable(strName, strarrIndexes);
//}

string adSetupVariableNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable) + ".array";
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

void adSetupVariableNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable) + ".array";
    xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupVariableNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable) + ".array";
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
                                                               size_t nOrder,
                                                               vector<daeArrayRange>& arrRanges)
                         : m_pVariable(pVariable),
                           m_nOrder(nOrder),
                           m_arrRanges(arrRanges)
{
}

adSetupTimeDerivativeNodeArray::adSetupTimeDerivativeNodeArray()
{
    m_pVariable = NULL;
    m_nOrder   = 0;
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

const quantity adSetupTimeDerivativeNodeArray::GetQuantity(void) const
{
    if(!m_pVariable)
        daeDeclareAndThrowException(exInvalidCall);
    return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / unit("s", 1));
}

adNodeArray* adSetupTimeDerivativeNodeArray::Clone(void) const
{
    return new adSetupTimeDerivativeNodeArray(*this);
}

void adSetupTimeDerivativeNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    string strExport;
    boost::format fmtFile;
    vector<string> strarrIndexes;

    FillDomains(m_arrRanges, strarrIndexes);

    string strName = daeGetStrippedRelativeName(c.m_pModel, m_pVariable);

    if(eLanguage == eCDAE)
    {
        strExport = "%1%.dt_array(%2%)";
        fmtFile.parse(strExport);
        fmtFile % strName % toString(strarrIndexes);
    }
    else if(eLanguage == ePYDAE)
    {
        strExport = /*"self.*/ "%1%.dt_array(%2%)";
        fmtFile.parse(strExport);
        fmtFile % strName % toString(strarrIndexes);
    }
    else
        daeDeclareAndThrowException(exNotImplemented);

    strContent += fmtFile.str();
}
//string adSetupTimeDerivativeNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	FillDomains(m_arrRanges, strarrIndexes);
//	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
//	return textCreator::TimeDerivative(m_nOrder, strName, strarrIndexes);
//}

string adSetupTimeDerivativeNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable) + ".dt_array";
    return latexCreator::TimeDerivative(m_nOrder, strName, strarrIndexes);
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
    pTag->Save(strName, m_nOrder);

    strName = "Ranges";
    pTag->SaveObjectArray(strName, m_arrRanges);
}

void adSetupTimeDerivativeNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable) + ".dt_array";
    xmlContentCreator::TimeDerivative(pTag, m_nOrder, strName, strarrIndexes);
}

void adSetupTimeDerivativeNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable) + ".dt_array";
    xmlPresentationCreator::TimeDerivative(pTag, m_nOrder, strName, strarrIndexes);
}

void adSetupTimeDerivativeNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    daeDeclareAndThrowException(exInvalidCall)
}

bool adSetupTimeDerivativeNodeArray::IsDifferential(void) const
{
    return true;
}

/*********************************************************************************************
    adSetupPartialDerivativeNodeArray
**********************************************************************************************/
adSetupPartialDerivativeNodeArray::adSetupPartialDerivativeNodeArray(daeVariable* pVariable,
                                                                     size_t nOrder,
                                                                     vector<daeArrayRange>& arrRanges,
                                                                     daeDomain* pDomain,
                                                                     daeeDiscretizationMethod  eDiscretizationMethod,
                                                                     const std::map<std::string, std::string>& mapDiscretizationOptions)
                            : m_pVariable(pVariable),
                              m_pDomain(pDomain),
                              m_nOrder(nOrder),
                              m_arrRanges(arrRanges),
                              m_eDiscretizationMethod(eDiscretizationMethod),
                              m_mapDiscretizationOptions(mapDiscretizationOptions)
{
}

adSetupPartialDerivativeNodeArray::adSetupPartialDerivativeNodeArray()
{
    m_pVariable = NULL;
    m_pDomain   = NULL;
    m_nOrder    = 0;
    m_eDiscretizationMethod = eDMUnknown;
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
    if(!m_pDomain)
        daeDeclareAndThrowException(exInvalidCall);
    if(m_arrRanges.empty())
        daeDeclareAndThrowException(exInvalidCall);

    adouble_array tmp;
    size_t N              = m_arrRanges.size();
    daeArrayRange* ranges = new daeArrayRange[N];

    for(size_t i = 0; i < N; i++)
        ranges[i] = m_arrRanges[i];
    tmp = m_pVariable->partial_array(m_nOrder, *m_pDomain, ranges, N, m_eDiscretizationMethod, m_mapDiscretizationOptions);
    delete[] ranges;
    return tmp;
}

const quantity adSetupPartialDerivativeNodeArray::GetQuantity(void) const
{
    if(!m_pVariable)
        daeDeclareAndThrowException(exInvalidCall);
    if(!m_pDomain)
        daeDeclareAndThrowException(exInvalidCall);
    if(m_nOrder == 1)
        return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / m_pDomain->GetUnits());
    else
        return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / (m_pDomain->GetUnits() ^ 2));
}

adNodeArray* adSetupPartialDerivativeNodeArray::Clone(void) const
{
    return new adSetupPartialDerivativeNodeArray(*this);
}

void adSetupPartialDerivativeNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    string strExport;
    boost::format fmtFile;
    vector<string> strarrIndexes;

    FillDomains(m_arrRanges, strarrIndexes);

    string strName       = daeGetStrippedRelativeName(c.m_pModel, m_pVariable);
    string strDomainName = daeGetStrippedRelativeName(c.m_pModel, m_pDomain);

    if(eLanguage == eCDAE)
    {
        strExport = "%1%.%2%(%3%, %4%)";
        fmtFile.parse(strExport);
        fmtFile % strName % (m_nOrder == 1 ? "d_array" : "d2_array") % strDomainName % toString(strarrIndexes);
    }
    else if(eLanguage == ePYDAE)
    {
        strExport = /*"self.*/ "%1%.%2%(self.%3%, %4%)";
        fmtFile.parse(strExport);
        fmtFile % strName % (m_nOrder == 1 ? "d_array" : "d2_array") % strDomainName % toString(strarrIndexes);
    }
    else
        daeDeclareAndThrowException(exNotImplemented);

    strContent += fmtFile.str();
}
//string adSetupPartialDerivativeNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	FillDomains(m_arrRanges, strarrIndexes);
//	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
//	string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
//	return textCreator::PartialDerivative(m_nOrder, strName, strDomainName, strarrIndexes);
//}

string adSetupPartialDerivativeNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable) + (m_nOrder == 1 ? ".d_array" : ".d2_array");
    string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
    return latexCreator::PartialDerivative(m_nOrder, strName, strDomainName, strarrIndexes);
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
    pTag->Save(strName, m_nOrder);

    strName = "Ranges";
    pTag->SaveObjectArray(strName, m_arrRanges);
}

void adSetupPartialDerivativeNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable) + (m_nOrder == 1 ? ".d_array" : ".d2_array");
    string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
    xmlContentCreator::PartialDerivative(pTag, m_nOrder, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrRanges, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable) + (m_nOrder == 1 ? ".d_array" : ".d2_array");
    string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
    xmlPresentationCreator::PartialDerivative(pTag, m_nOrder, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    daeDeclareAndThrowException(exInvalidCall)
}


/*********************************************************************************************
    adSetupExpressionPartialDerivativeNodeArray
**********************************************************************************************/
/*
adSetupExpressionPartialDerivativeNodeArray::adSetupExpressionPartialDerivativeNodeArray(daeDomain*                                pDomain,
                                                                                         size_t                                    nOrder,
                                                                                         daeeDiscretizationMethod                  eDiscretizationMethod,
                                                                                         const std::map<std::string, std::string>& mapDiscretizationOptions,
                                                                                         adNodeArrayPtr                            n)
{
    m_pDomain                  = pDomain;
    m_eDiscretizationMethod    = eDiscretizationMethod;
    m_mapDiscretizationOptions = mapDiscretizationOptions;
    node                       = n;
    m_nOrder                   = nOrder;
}

adSetupExpressionPartialDerivativeNodeArray::adSetupExpressionPartialDerivativeNodeArray()
{
    m_nOrder                = 0;
    m_pDomain               = NULL;
    m_eDiscretizationMethod = eDMUnknown;
}

adSetupExpressionPartialDerivativeNodeArray::~adSetupExpressionPartialDerivativeNodeArray()
{
}

void adSetupExpressionPartialDerivativeNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{
    // Do nothing?
}

size_t adSetupExpressionPartialDerivativeNodeArray::GetSize(void) const
{
    return node->GetSize();
}

adouble_array adSetupExpressionPartialDerivativeNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
#ifdef DAE_DEBUG
    if(!m_pDomain)
        daeDeclareAndThrowException(exInvalidPointer);
#endif

    if(m_nOrder == 0)
        daeDeclareAndThrowException(exInvalidCall);

    adouble_array a, tmp;

    // First get a runtime node
    a = node->Evaluate(pExecutionContext);

    // Set the size of the adouble array
    tmp.setGatherInfo(true);
    size_t N = a.GetSize();
    tmp.Resize(N);

    // Then calculate the partial derivative expressions
    if(m_nOrder == 1)
    {
        for(size_t i = 0; i < N; i++)
            tmp[i].node = adSetupExpressionPartialDerivativeNode::calc_d(a[i].node, m_pDomain, m_eDiscretizationMethod, m_mapDiscretizationOptions, pExecutionContext);
    }
    else if(m_nOrder == 2)
    {
        for(size_t i = 0; i < N; i++)
            tmp[i].node = adSetupExpressionPartialDerivativeNode::calc_d2(a[i].node, m_pDomain, m_eDiscretizationMethod, m_mapDiscretizationOptions, pExecutionContext);
    }
    return tmp;
}

const quantity adSetupExpressionPartialDerivativeNodeArray::GetQuantity(void) const
{
    if(!m_pDomain)
        daeDeclareAndThrowException(exInvalidPointer);

    quantity q = node->GetQuantity();
    unit u = m_pDomain->GetUnits();
    if(m_nOrder == 1)
        return quantity(0.0, q.getUnits() / u);
    else if(m_nOrder == 2)
        return quantity(0.0, q.getUnits() / (u*u));
    else
        return quantity();
}

adNodeArray* adSetupExpressionPartialDerivativeNodeArray::Clone(void) const
{
    return new adSetupExpressionPartialDerivativeNodeArray(*this);
}

void adSetupExpressionPartialDerivativeNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    string strExport, strExpression, strDFunction;
    boost::format fmtFile;

    node->Export(strExpression, eLanguage, c);

    string strDomainName = daeGetStrippedRelativeName(c.m_pModel, m_pDomain);
    if(m_nOrder == 1)
        strDFunction = "d";
    else if(m_nOrder == 2)
        strDFunction = "d2";

    string strDiscretizationMethod;
    if(m_eDiscretizationMethod == eCFDM)
        strDiscretizationMethod = "eCFDM";
    else if(m_eDiscretizationMethod == eFFDM)
        strDiscretizationMethod = "eFFDM";
    else if(m_eDiscretizationMethod == eBFDM)
        strDiscretizationMethod = "eBFDM";
    else if(m_eDiscretizationMethod == eUpwindCCFV)
        strDiscretizationMethod = "eUpwindCCFV";
    else
        daeDeclareAndThrowException(exNotImplemented);

    if(eLanguage == eCDAE)
    {
        // This needs some thinking...
        string strDiscretizationOptions = toString(m_mapDiscretizationOptions);

        strExport = "%1%(%2%, %3%, %4%, \"%5%\")";
        fmtFile.parse(strExport);
        fmtFile % strDFunction % strExpression % strDomainName % strDiscretizationMethod % strDiscretizationOptions;
    }
    else if(eLanguage == ePYDAE)
    {
        string strDiscretizationOptions = "{";
        std::map<string, string>::const_iterator citer;
        for(citer = m_mapDiscretizationOptions.begin(); citer != m_mapDiscretizationOptions.end(); citer++)
        {
            if(citer != m_mapDiscretizationOptions.begin())
                strDiscretizationOptions += ",";
            strDiscretizationOptions += "'" + citer->first + "' : '" + citer->second + "'";
        }
        strDiscretizationOptions += "}";

        strExport = "%1%(%2%, %3%, %4%, %5%)";
        fmtFile.parse(strExport);
        fmtFile % strDFunction % strExpression % strDomainName % strDiscretizationMethod % strDiscretizationOptions;
    }
    else
        daeDeclareAndThrowException(exNotImplemented);

    strContent += fmtFile.str();
}
//string adSetupExpressionPartialDerivativeNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	string strExpression = node->SaveAsPlainText(c);
//	string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
//	return textCreator::PartialDerivative(m_nDegree, strExpression, strDomainName, strarrIndexes, true);
//}

string adSetupExpressionPartialDerivativeNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    string strExpression = node->SaveAsLatex(c);
    string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
    return latexCreator::PartialDerivative(m_nOrder, strExpression, strDomainName, strarrIndexes, true);
}

void adSetupExpressionPartialDerivativeNodeArray::Open(io::xmlTag_t* pTag)
{
    daeDeclareAndThrowException(exNotImplemented)
}

void adSetupExpressionPartialDerivativeNodeArray::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "Order";
    pTag->Save(strName, m_nOrder);

    strName = "Domain";
    pTag->SaveObjectRef(strName, m_pDomain);

    strName = "Node";
    adNodeArray::SaveNode(pTag, strName, node.get());
}

void adSetupExpressionPartialDerivativeNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adSetupExpressionPartialDerivativeNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    string strName, strValue;
    io::xmlTag_t *mfrac, *msup, *mrow1, *mrow2, *mrow0;

    string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);

    strName  = "mrow";
    strValue = "";
    mrow0 = pTag->AddTag(strName, strValue);

    strName  = "mfrac";
    strValue = "";
    mfrac = mrow0->AddTag(strName, strValue);

    strName  = "mrow";
    strValue = "";
    mrow1 = mfrac->AddTag(strName, strValue);

    if(m_nOrder == 1)
    {
        strName  = "mo";
        strValue = "&PartialD;";
        mrow1->AddTag(strName, strValue);
    }
    else
    {
        strName  = "msup";
        strValue = "";
        msup = mrow1->AddTag(strName, strValue);
            strName  = "mo";
            strValue = "&PartialD;";
            msup->AddTag(strName, strValue);

            strName  = "mn";
            strValue = "2";
            msup->AddTag(strName, strValue);
    }

    strName  = "mrow";
    strValue = "";
    mrow2 = mfrac->AddTag(strName, strValue);

    if(m_nOrder == 1)
    {
        strName  = "mo";
        strValue = "&PartialD;";
        mrow2->AddTag(strName, strValue);

        strName  = "mi";
        strValue = strDomainName;
        mrow2->AddTag(strName, strValue);
    }
    else
    {
        strName  = "mo";
        strValue = "&PartialD;";
        mrow2->AddTag(strName, strValue);

        strName  = "msup";
        strValue = "";
        msup = mrow2->AddTag(strName, strValue);

            strName  = "mi";
            strValue = strDomainName;
            xmlPresentationCreator::WrapIdentifier(msup, strValue);

            strName  = "mn";
            strValue = "2";
            msup->AddTag(strName, strValue);
    }

    strName  = "mrow";
    strValue = "";
    mrow2 = mrow0->AddTag(strName, strValue);

    mrow2->AddTag(string("mo"), string("("));
    node->SaveAsPresentationMathML(mrow2, c);
    mrow2->AddTag(string("mo"), string(")"));
}

void adSetupExpressionPartialDerivativeNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    if(!node)
        daeDeclareAndThrowException(exInvalidPointer);
    node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}
*/

/*********************************************************************************************
    adVectorExternalFunctionNode
**********************************************************************************************/
adVectorExternalFunctionNode::adVectorExternalFunctionNode(daeVectorExternalFunction* externalFunction)
{
    m_pExternalFunction = externalFunction;
}

adVectorExternalFunctionNode::~adVectorExternalFunctionNode()
{
}

void adVectorExternalFunctionNode::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{
}

size_t adVectorExternalFunctionNode::GetSize(void) const
{
    if(!m_pExternalFunction)
        daeDeclareAndThrowException(exInvalidPointer);

    return m_pExternalFunction->GetNumberOfResults();
}

adouble_array adVectorExternalFunctionNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(!m_pExternalFunction)
        daeDeclareAndThrowException(exInvalidPointer);

    adouble_array tmp;
// Here I have to initialize arguments (which are at this moment setup nodes)
// Creation of runtime nodes will also add variable indexes into the equation execution info
    daeVectorExternalFunction* pExtFun = const_cast<daeVectorExternalFunction*>(m_pExternalFunction);
    pExtFun->InitializeArguments(pExecutionContext);

    daeExternalFunctionArgumentValue_t value;
    daeExternalFunctionArgumentValueMap_t mapValues;
    daeExternalFunctionArgumentMap_t::const_iterator iter;
    const daeExternalFunctionArgumentMap_t& mapArgumentNodes = m_pExternalFunction->GetArgumentNodes();

    for(iter = mapArgumentNodes.begin(); iter != mapArgumentNodes.end(); iter++)
    {
        std::string                   strName  = iter->first;
        daeExternalFunctionArgument_t argument = iter->second;

        adouble*       ad    = boost::get<adouble>(&argument);
        adouble_array* adarr = boost::get<adouble_array>(&argument);

        if(ad)
        {
            value = (*ad).node->Evaluate(pExecutionContext);
        }
        else if(adarr)
        {
            size_t n = adarr->m_arrValues.size();
            std::vector<adouble> tmp;
            tmp.resize(n);
            for(size_t i = 0; i < n; i++)
                tmp[i] = adarr->m_arrValues[i].node->Evaluate(pExecutionContext);
            value = tmp;
        }
        else
            daeDeclareAndThrowException(exInvalidCall);

        mapValues[strName] = value;
    }

    tmp.m_arrValues = m_pExternalFunction->Calculate(mapValues);
    return tmp;
}

const quantity adVectorExternalFunctionNode::GetQuantity(void) const
{
    if(!m_pExternalFunction)
        daeDeclareAndThrowException(exInvalidPointer);
    return quantity(0.0, m_pExternalFunction->GetUnits());
}

adNodeArray* adVectorExternalFunctionNode::Clone(void) const
{
    return new adVectorExternalFunctionNode(*this);
}

void adVectorExternalFunctionNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    daeDeclareAndThrowException(exNotImplemented);
}
//string adVectorExternalFunctionNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//}

string adVectorExternalFunctionNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    string strLatex;

    strLatex += "{ ";
    strLatex += m_pExternalFunction->GetName();

    daeExternalFunctionNodeMap_t::const_iterator iter;
    const daeExternalFunctionNodeMap_t& mapArgumentNodes = m_pExternalFunction->GetSetupArgumentNodes();

    strLatex += " \\left( ";
    for(iter = mapArgumentNodes.begin(); iter != mapArgumentNodes.end(); iter++)
    {
        std::string               strName  = iter->first;
        daeExternalFunctionNode_t argument = iter->second;

        adNodePtr*      ad    = boost::get<adNodePtr>     (&argument);
        adNodeArrayPtr* adarr = boost::get<adNodeArrayPtr>(&argument);

        if(iter != mapArgumentNodes.begin())
            strLatex += ", ";
        strLatex += strName + " = { ";

        if(ad)
        {
            adNode* node = ad->get();
            strLatex += node->SaveAsLatex(c);
        }
        else if(adarr)
        {
            adNodeArray* nodearray = adarr->get();
            strLatex += nodearray->SaveAsLatex(c);
        }
        else
            daeDeclareAndThrowException(exInvalidCall);

        strLatex += " } ";
    }
    strLatex += " \\right) }";

    return strLatex;
}

void adVectorExternalFunctionNode::Open(io::xmlTag_t* pTag)
{
}

void adVectorExternalFunctionNode::Save(io::xmlTag_t* pTag) const
{
    string strName, strValue;
    daeExternalFunctionNode_t argument;

    strName = "Name";
    strValue = m_pExternalFunction->GetName();
    pTag->Save(strName, strValue);

    strName = "NumberOfResults";
    strValue = toString(m_pExternalFunction->GetNumberOfResults());
    pTag->Save(strName, strValue);

    strName = "Arguments";
    io::xmlTag_t* pArgumentsTag = pTag->AddTag(strName);

    daeExternalFunctionNodeMap_t::const_iterator iter;
    const daeExternalFunctionNodeMap_t& mapArgumentNodes = m_pExternalFunction->GetSetupArgumentNodes();

    for(iter = mapArgumentNodes.begin(); iter != mapArgumentNodes.end(); iter++)
    {
        strName  = iter->first;
        argument = iter->second;

        adNodePtr*      ad    = boost::get<adNodePtr>     (&argument);
        adNodeArrayPtr* adarr = boost::get<adNodeArrayPtr>(&argument);

        if(ad)
            adNode::SaveNode(pArgumentsTag, strName, ad->get());
        else if(adarr)
            adNodeArray::SaveNode(pArgumentsTag, strName, adarr->get());
        else
            daeDeclareAndThrowException(exInvalidCall);
    }
}

void adVectorExternalFunctionNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adVectorExternalFunctionNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    io::xmlTag_t* pRowTag = pTag->AddTag(string("mrow"));

    io::xmlTag_t* pFunctionTag = pRowTag->AddTag(string("mi"), m_pExternalFunction->GetName());
    pFunctionTag->AddAttribute(string("fontstyle"), string("italic"));

    io::xmlTag_t* pFencedTag = pRowTag->AddTag(string("mfenced"));

    daeExternalFunctionNodeMap_t::const_iterator iter;
    const daeExternalFunctionNodeMap_t& mapArgumentNodes = m_pExternalFunction->GetSetupArgumentNodes();

    for(iter = mapArgumentNodes.begin(); iter != mapArgumentNodes.end(); iter++)
    {
        std::string               strName  = iter->first;
        daeExternalFunctionNode_t argument = iter->second;

        adNodePtr*      ad    = boost::get<adNodePtr>     (&argument);
        adNodeArrayPtr* adarr = boost::get<adNodeArrayPtr>(&argument);

        io::xmlTag_t* pArgRowTag = pFencedTag->AddTag(string("mrow"));

        io::xmlTag_t* pArgNameTag = pArgRowTag->AddTag(string("mi"), strName);
        pArgNameTag->AddAttribute(string("fontstyle"), string("italic"));

        pArgRowTag->AddTag(string("mo"), string("="));

        if(ad)
        {
            adNode* node = ad->get();
            node->SaveAsPresentationMathML(pArgRowTag, c);
        }
        else if(adarr)
        {
            adNodeArray* nodearray = adarr->get();
            nodearray->SaveAsPresentationMathML(pArgRowTag, c);
        }
        else
            daeDeclareAndThrowException(exInvalidCall);
    }
}

void adVectorExternalFunctionNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    if(!m_pExternalFunction)
        daeDeclareAndThrowException(exInvalidPointer);

    daeExternalFunctionArgumentMap_t::const_iterator iter;
    const daeExternalFunctionArgumentMap_t& mapArgumentNodes = m_pExternalFunction->GetArgumentNodes();

    // This operates on RuntimeNodes!!
    for(iter = mapArgumentNodes.begin(); iter != mapArgumentNodes.end(); iter++)
    {
        daeExternalFunctionArgument_t argument = iter->second;

        adouble*       ad    = boost::get<adouble>(&argument);
        adouble_array* adarr = boost::get<adouble_array>(&argument);

        if(ad)
            (*ad).node->AddVariableIndexToArray(mapIndexes, bAddFixed);
        else if(adarr)
            for(size_t i = 0; i < adarr->m_arrValues.size(); i++)
                adarr->m_arrValues[i].node->AddVariableIndexToArray(mapIndexes, bAddFixed);
        else
            daeDeclareAndThrowException(exInvalidCall);
    }
}

bool adVectorExternalFunctionNode::IsLinear(void) const
{
    return false;
}

bool adVectorExternalFunctionNode::IsFunctionOfVariables(void) const
{
    return true;
}

}
}
