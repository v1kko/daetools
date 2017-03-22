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
#ifndef DAE_NODES_ARRAY_H
#define DAE_NODES_ARRAY_H

#include "adouble.h"
#include "nodes.h"

namespace dae
{
namespace core
{
inline adNodeArrayPtr CLONE_NODE_ARRAY(adNodeArrayPtr n)
{
    if(n)
        return adNodeArrayPtr(n->Clone());
    else
    {
        daeDeclareAndThrowException(exInvalidCall);
        return adNodeArrayPtr();
    }
}

bool adDoEnclose(const adNodeArray* node);
void adDoEnclose(const adNodeArray* parent, const adNodeArray* left, bool& bEncloseLeft, const adNodeArray* right, bool& bEncloseRight);

/*********************************************************************************************
    adNodeArrayImpl
**********************************************************************************************/
class DAE_CORE_API adNodeArrayImpl : public adNodeArray
{
public:
    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    //void	ExportAsPlainText(string strFileName);
    void ExportAsLatex(string strFileName);

    bool IsLinear(void) const;
    bool IsFunctionOfVariables(void) const;
    bool IsDifferential(void) const;

    virtual void GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
};

/*********************************************************************************************
    adConstantNodeArray
**********************************************************************************************/
class DAE_CORE_API adConstantNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adConstantNodeArray)
    adConstantNodeArray(void);
    adConstantNodeArray(real_t d);
    adConstantNodeArray(real_t d, const unit& units);
    virtual ~adConstantNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool			IsLinear(void) const;
    virtual bool			IsFunctionOfVariables(void) const;
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;

public:
    quantity m_quantity;
};

/*********************************************************************************************
    adVectorNodeArray
**********************************************************************************************/
class DAE_CORE_API adVectorNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adVectorNodeArray)
    adVectorNodeArray(void);
    adVectorNodeArray(const std::vector<real_t>& darrValues);
    adVectorNodeArray(const std::vector<quantity>& qarrValues);
    virtual ~adVectorNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool			IsLinear(void) const;
    virtual bool			IsFunctionOfVariables(void) const;
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;

public:
    std::vector<quantity> m_qarrValues;
};

/*********************************************************************************************
    adRuntimeParameterNodeArray
**********************************************************************************************/
/*
class DAE_CORE_API adRuntimeParameterNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adRuntimeParameterNodeArray)
    adRuntimeParameterNodeArray(void);
    virtual ~adRuntimeParameterNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool			IsLinear(void) const;
    virtual bool			IsFunctionOfVariables(void) const;
    virtual const quantity	GetQuantity(void) const;

public:
// Runtime part
    std::vector<adNodePtr>		m_ptrarrParameterNodes;
// Report/GUI part
    daeParameter*			m_pParameter;
    std::vector<daeArrayRange>	m_arrRanges;
};
*/
/*********************************************************************************************
    adRuntimeVariableNodeArray
**********************************************************************************************/
/*
class DAE_CORE_API adRuntimeVariableNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adRuntimeVariableNodeArray)
    adRuntimeVariableNodeArray(void);
    virtual ~adRuntimeVariableNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool			IsLinear(void) const;
    virtual bool			IsFunctionOfVariables(void) const;
    virtual const quantity	GetQuantity(void) const;

public:
// Runtime part
    std::vector< adNodePtr >		m_ptrarrVariableNodes;
// Report/GUI part
    daeVariable*							m_pVariable;
    std::vector<daeArrayRange>					m_arrRanges;
};
*/
/*********************************************************************************************
    adRuntimeTimeDerivativeNodeArray
**********************************************************************************************/
/*
class DAE_CORE_API adRuntimeTimeDerivativeNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adRuntimeTimeDerivativeNodeArray)
    adRuntimeTimeDerivativeNodeArray(void);
    virtual ~adRuntimeTimeDerivativeNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual const quantity	GetQuantity(void) const;

public:
// Runtime part
    std::vector< adNodePtr >	m_ptrarrTimeDerivativeNodes;
    size_t								m_nDegree;
// Report/GUI part
    daeVariable*						m_pVariable;
    std::vector<daeArrayRange>				m_arrRanges;
};
*/
/*********************************************************************************************
    adRuntimePartialDerivativeNodeArray
**********************************************************************************************/
/*
class DAE_CORE_API adRuntimePartialDerivativeNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adRuntimePartialDerivativeNodeArray)
    adRuntimePartialDerivativeNodeArray(void);
    virtual ~adRuntimePartialDerivativeNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool			IsLinear(void) const;
    virtual bool			IsFunctionOfVariables(void) const;
    virtual const quantity	GetQuantity(void) const;

public:
// Runtime part
    std::vector< adNodePtr >	m_ptrarrPartialDerivativeNodes;
    size_t								m_nDegree;
// Report/GUI part
    daeVariable*						m_pVariable;
    daeDomain*							m_pDomain;
    std::vector<daeArrayRange>				m_arrRanges;
};
*/

/*********************************************************************************************
    adRuntimeIntegralNode
**********************************************************************************************/
/*
class DAE_CORE_API adRuntimeIntegralNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adRuntimeIntegralNode)
    adRuntimeIntegralNode(void);
    adRuntimeIntegralNode(daeeIntegralFunctions eFun,
                          daeModel* pModel,
                          adNodeArrayPtr n,
                          daeDomain* pDomain,
                          const std::vector<const real_t*>& pdarrPoints);
    virtual ~adRuntimeIntegralNode(void);

public:
    virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNode*	Clone(void) const;
    virtual void	Open(io::xmlTag_t* pTag);
    virtual void	Save(io::xmlTag_t* pTag) const;
    virtual string	SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool	IsLinear(void) const;
    virtual bool	IsFunctionOfVariables(void) const;
    virtual const quantity GetQuantity(void) const;

public:
    adNodeArrayPtr	node;
    daeModel*						m_pModel;
    daeeIntegralFunctions			eFunction;
    daeDomain*						m_pDomain;
    std::vector<const real_t*>		m_pdarrPoints;
};
*/

/*********************************************************************************************
    adRuntimeSpecialFunctionForLargeArraysNode
**********************************************************************************************/
class DAE_CORE_API adRuntimeSpecialFunctionForLargeArraysNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adRuntimeSpecialFunctionForLargeArraysNode)
    adRuntimeSpecialFunctionForLargeArraysNode(void);
    adRuntimeSpecialFunctionForLargeArraysNode(daeeSpecialUnaryFunctions eFun,
                                               const std::vector<adNodePtr>& ptrarrRuntimeNodes);
    virtual ~adRuntimeSpecialFunctionForLargeArraysNode(void);

public:
    virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNode*	Clone(void) const;
    virtual void	Open(io::xmlTag_t* pTag);
    virtual void	Save(io::xmlTag_t* pTag) const;
    virtual string	SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool	IsLinear(void) const;
    virtual bool	IsFunctionOfVariables(void) const;
    virtual bool    IsDifferential(void) const;
    virtual const quantity GetQuantity(void) const;

public:
// Runtime part
    std::vector<adNodePtr>     m_ptrarrRuntimeNodes;
    daeeSpecialUnaryFunctions  eFunction;
};

/*********************************************************************************************
    adUnaryNodeArray
**********************************************************************************************/
class DAE_CORE_API adUnaryNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adUnaryNodeArray)
    adUnaryNodeArray(void);
    adUnaryNodeArray(daeeUnaryFunctions eFun, adNodeArrayPtr n);
    virtual ~adUnaryNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool			IsLinear(void) const;
    virtual bool			IsFunctionOfVariables(void) const;
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;
    virtual bool            IsDifferential(void) const;

public:
    adNodeArrayPtr      node;
    daeeUnaryFunctions	eFunction;
};

/*********************************************************************************************
    adBinaryNodeArray
**********************************************************************************************/
class DAE_CORE_API adBinaryNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adBinaryNodeArray)
    adBinaryNodeArray(void);
    adBinaryNodeArray(daeeBinaryFunctions eFun, adNodeArrayPtr l, adNodeArrayPtr r);
    virtual ~adBinaryNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool			IsLinear(void) const;
    virtual bool			IsFunctionOfVariables(void) const;
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;
    virtual bool            IsDifferential(void) const;

public:
    adNodeArrayPtr          left;
    adNodeArrayPtr          right;
    daeeBinaryFunctions		eFunction;
};

/*********************************************************************************************
    adSetupSpecialFunctionNode
**********************************************************************************************/
class DAE_CORE_API adSetupSpecialFunctionNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adSetupSpecialFunctionNode)
    adSetupSpecialFunctionNode(void);
    adSetupSpecialFunctionNode(daeeSpecialUnaryFunctions eFun,
                               adNodeArrayPtr n,
                               bool bIsLargeArray = false);
    virtual ~adSetupSpecialFunctionNode(void);

public:
    virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNode*	Clone(void) const;
    virtual void	Open(io::xmlTag_t* pTag);
    virtual void	Save(io::xmlTag_t* pTag) const;
    virtual string	SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool	IsLinear(void) const;
    virtual bool	IsFunctionOfVariables(void) const;
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;

public:
    adNodeArrayPtr              node;
    daeeSpecialUnaryFunctions	eFunction;
    bool                        m_bIsLargeArray;
};

/*********************************************************************************************
    adSetupIntegralNode
**********************************************************************************************/
class DAE_CORE_API adSetupIntegralNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adSetupIntegralNode)
    adSetupIntegralNode(void);
    adSetupIntegralNode(daeeIntegralFunctions eFun,
                        adNodeArrayPtr n,
                        daeDomain* pDomain,
                        const daeArrayRange& arrayRange);
    virtual ~adSetupIntegralNode(void);

public:
    virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNode*	Clone(void) const;
    virtual void	Open(io::xmlTag_t* pTag);
    virtual void	Save(io::xmlTag_t* pTag) const;
    virtual string	SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;

public:
    adNodeArrayPtr          node;
    daeDomain*				m_pDomain;
    daeeIntegralFunctions	eFunction;
    daeArrayRange			m_ArrayRange;
};

/*********************************************************************************************
    adSingleNodeArray
**********************************************************************************************/
class DAE_CORE_API adSingleNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adSingleNodeArray)
    adSingleNodeArray(void);
    adSingleNodeArray(adNodePtr n);
    virtual ~adSingleNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;
    virtual bool            IsDifferential(void) const;

public:
    adNodePtr	node;
};

/*********************************************************************************************
    adSetupParameterNodeArray
**********************************************************************************************/
class DAE_CORE_API adSetupParameterNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adSetupParameterNodeArray)
    adSetupParameterNodeArray(void);
    adSetupParameterNodeArray(daeParameter* pParameter,
                              std::vector<daeArrayRange>& arrRanges);
    virtual ~adSetupParameterNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;

public:
    daeParameter*               m_pParameter;
    std::vector<daeArrayRange>	m_arrRanges;
};

/*********************************************************************************************
    adSetupDomainNodeArray
**********************************************************************************************/
class DAE_CORE_API adSetupDomainNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adSetupDomainNodeArray)
    adSetupDomainNodeArray(void);
    adSetupDomainNodeArray(daeDomain* pDomain,
                           const daeArrayRange& range);
    virtual ~adSetupDomainNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;

public:
    daeDomain*      m_pDomain;
    daeArrayRange	m_Range;
};

/*********************************************************************************************
    adCustomNodeArray
**********************************************************************************************/
class DAE_CORE_API adCustomNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adCustomNodeArray)
    adCustomNodeArray(void);
    adCustomNodeArray(const std::vector<adNodePtr>& ptrarrNodes);
    virtual ~adCustomNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;

public:
    std::vector<adNodePtr> m_ptrarrNodes;
};

/*********************************************************************************************
    adRuntimeCustomNodeArray
**********************************************************************************************/
/*
class DAE_CORE_API adRuntimeCustomNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adRuntimeCustomNodeArray)
    adRuntimeCustomNodeArray(void);
    adRuntimeCustomNodeArray(const std::vector<adNodePtr>& ptrarrNodes);
    virtual ~adRuntimeCustomNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;
    virtual bool            IsLinear(void) const;
    virtual bool            IsFunctionOfVariables(void) const;
    virtual bool            IsDifferential(void) const;

public:
    std::vector<adNodePtr> m_ptrarrNodes;
};
*/

/*********************************************************************************************
    adSetupVariableNodeArray
**********************************************************************************************/
class DAE_CORE_API adSetupVariableNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adSetupVariableNodeArray)
    adSetupVariableNodeArray(void);
    adSetupVariableNodeArray(daeVariable* pVariable,
                             std::vector<daeArrayRange>& arrRanges);
    virtual ~adSetupVariableNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;

public:
    daeVariable*                m_pVariable;
    std::vector<daeArrayRange>	m_arrRanges;
};

/*********************************************************************************************
    adSetupTimeDerivativeNodeArray
**********************************************************************************************/
class DAE_CORE_API adSetupTimeDerivativeNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adSetupTimeDerivativeNodeArray)
    adSetupTimeDerivativeNodeArray(void);
    adSetupTimeDerivativeNodeArray(daeVariable* pVariable,
                                   size_t nOrder,
                                   std::vector<daeArrayRange>& arrRanges);
    virtual ~adSetupTimeDerivativeNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;
    virtual bool            IsDifferential(void) const;

public:
    daeVariable*                m_pVariable;
    size_t                      m_nOrder;
    std::vector<daeArrayRange>	m_arrRanges;
};

/*********************************************************************************************
    adSetupPartialDerivativeNodeArray
**********************************************************************************************/
class DAE_CORE_API adSetupPartialDerivativeNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adSetupPartialDerivativeNodeArray)
    adSetupPartialDerivativeNodeArray(void);
    adSetupPartialDerivativeNodeArray(daeVariable* pVariable,
                                      size_t nOrder,
                                      std::vector<daeArrayRange>& arrRanges,
                                      daeDomain* pDomain,
                                      const daeeDiscretizationMethod  eDiscretizationMethod,
                                      const std::map<std::string, std::string>& mapDiscretizationOptions);
    virtual ~adSetupPartialDerivativeNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;

public:
    daeVariable*                        m_pVariable;
    daeDomain*                          m_pDomain;
    size_t                              m_nOrder;
    std::vector<daeArrayRange>          m_arrRanges;
    daeeDiscretizationMethod            m_eDiscretizationMethod;
    std::map<std::string, std::string>  m_mapDiscretizationOptions;
};

/*********************************************************************************************
    adSetupExpressionPartialDerivativeNodeArray
**********************************************************************************************/
// Not implemented yet; does not work!!
/*
class DAE_CORE_API adSetupExpressionPartialDerivativeNodeArray : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adSetupExpressionPartialDerivativeNodeArray)
    adSetupExpressionPartialDerivativeNodeArray(void);
    adSetupExpressionPartialDerivativeNodeArray(daeDomain*                                pDomain,
                                                size_t                                    nOrder,
                                                daeeDiscretizationMethod                  eDiscretizationMethod,
                                                const std::map<std::string, std::string>& mapDiscretizationOptions,
                                                adNodeArrayPtr                            n);
    virtual ~adSetupExpressionPartialDerivativeNodeArray(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;

public:
    daeDomain*                          m_pDomain;
    adNodeArrayPtr                      node;
    size_t                              m_nOrder;
    daeeDiscretizationMethod            m_eDiscretizationMethod;
    std::map<std::string, std::string>  m_mapDiscretizationOptions;
};
*/
/*********************************************************************************************
    adVectorExternalFunctionNode
**********************************************************************************************/
class DAE_CORE_API adVectorExternalFunctionNode : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adVectorExternalFunctionNode)
    adVectorExternalFunctionNode(daeVectorExternalFunction* externalFunction);
    virtual ~adVectorExternalFunctionNode(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*	Clone(void) const;
    virtual void			Open(io::xmlTag_t* pTag);
    virtual void			Save(io::xmlTag_t* pTag) const;
    virtual string			SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual bool            IsLinear(void) const;
    virtual bool            IsFunctionOfVariables(void) const;
    virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity	GetQuantity(void) const;

public:
    daeVectorExternalFunction* m_pExternalFunction;
};

/*********************************************************************************************
    adThermoPhysicalPropertyPackageArrayNode
**********************************************************************************************/
class DAE_CORE_API adThermoPhysicalPropertyPackageArrayNode : public adNodeArrayImpl
{
public:
    daeDeclareDynamicClass(adThermoPhysicalPropertyPackageArrayNode)
    adThermoPhysicalPropertyPackageArrayNode(daeeThermoPackagePropertyType propType,
                                              adNodePtr P,
                                              adNodePtr T,
                                              adNodeArrayPtr X,
                                              daeeThermoPhysicalProperty property_,
                                              daeeThermoPackagePhase phase_,
                                              daeeThermoPackageBasis basis_,
                                              daeThermoPhysicalPropertyPackage_t* tpp);
    virtual ~adThermoPhysicalPropertyPackageArrayNode(void);

public:
    virtual size_t			GetSize(void) const;
    virtual void			GetArrayRanges(std::vector<daeArrayRange>& arrRanges) const;
    virtual adouble_array   Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNodeArray*    Clone(void) const;
    virtual void            Open(io::xmlTag_t* pTag);
    virtual void            Save(io::xmlTag_t* pTag) const;
    virtual string          SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void            SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void            SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void            AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool            IsLinear(void) const;
    virtual bool            IsFunctionOfVariables(void) const;
    virtual bool            IsDifferential(void) const;
    virtual void            Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity  GetQuantity(void) const;

public:
    daeeThermoPackagePropertyType       propertyType;
    adNodePtr                           pressure;
    adNodePtr                           temperature;
    adNodeArrayPtr                      composition;
    daeeThermoPhysicalProperty          property;
    daeeThermoPackagePhase              phase;
    daeeThermoPackageBasis              basis;
    daeThermoPhysicalPropertyPackage_t* thermoPhysicalPropertyPackage;
};

inline void FillDomains(const std::vector<daeArrayRange>& arrRanges, std::vector<string>& strarrDomains)
{
    size_t n = arrRanges.size();
    strarrDomains.resize(n);
    for(size_t i = 0; i < n; i++)
    {
        strarrDomains[i] = arrRanges[i].GetRangeAsString();

//		if(arrRanges[i].m_eType == eRangeConstantIndex)
//		{
//			strarrDomains[i] = toString<size_t>(arrRanges[i].m_nIndex);
//		}
//		else if(arrRanges[i].m_eType == eRangeDomainIterator)
//		{
//			if(!arrRanges[i].m_pDEDI)
//				daeDeclareAndThrowException(exInvalidCall);
//			strarrDomains[i] = arrRanges[i].m_pDEDI->GetName();
//		}
//		else if(arrRanges[i].m_eType == eRange)
//		{
//			strarrDomains[i] = arrRanges[i].m_Range.ToString();
//		}
//		else
//		{
//			daeDeclareAndThrowException(exXMLIOError);
//		}
    }
}

}
}

#endif // DAE_NODES_ARRAY_H
