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
#ifndef DAE_NODES_H
#define DAE_NODES_H

#include "adouble.h"

namespace dae 
{
namespace core 
{
bool adDoEnclose(const adNode* node);
void adDoEnclose(const adNode* parent, const adNode* left, bool& bEncloseLeft, const adNode* right, bool& bEncloseRight);
bool condDoEnclose(const adNode* node);
bool condDoEnclose(const condNode* node);

/*********************************************************************************************
	adNodeImpl
**********************************************************************************************/
class DAE_CORE_API adNodeImpl : public adNode
{
public:
	void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	void ExportAsLatex(string strFileName);
	bool IsLinear(void) const;
	bool IsFunctionOfVariables(void) const;
    bool IsDifferential(void) const;
};

daeeEquationType DetectEquationType(adNodePtr node);

/*********************************************************************************************
	adConstantNode
**********************************************************************************************/
class DAE_CORE_API adConstantNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adConstantNode)
	adConstantNode(void);
	adConstantNode(real_t d);
	adConstantNode(real_t d, const unit& units);
	adConstantNode(const quantity& q);
	virtual ~adConstantNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual const quantity GetQuantity(void) const;

public:
	quantity m_quantity;
};

/*********************************************************************************************
	adTimeNode
**********************************************************************************************/
class DAE_CORE_API adTimeNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adTimeNode)
	adTimeNode(void);
	virtual ~adTimeNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual const quantity GetQuantity(void) const;
};

/*********************************************************************************************
	adEventPortDataNode
**********************************************************************************************/
class DAE_CORE_API adEventPortDataNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adEventPortDataNode)
	adEventPortDataNode(daeEventPort* pEventPort);
	virtual ~adEventPortDataNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual const quantity GetQuantity(void) const;

public:
	daeEventPort*	m_pEventPort;
};

/*********************************************************************************************
	adRuntimeParameterNode
**********************************************************************************************/
class DAE_CORE_API adRuntimeParameterNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adRuntimeParameterNode)
	adRuntimeParameterNode(void);
	adRuntimeParameterNode(daeParameter* pParameter, 
	                       std::vector<size_t>& narrDomains, 
                           real_t* pdValue);
	virtual ~adRuntimeParameterNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual const quantity GetQuantity(void) const;

public:
// Runtime part
    real_t*			m_pdValue;
// Report/GUI part
	daeParameter*	m_pParameter;
	std::vector<size_t>	m_narrDomains;
};

/*********************************************************************************************
	adDomainIndexNode
**********************************************************************************************/
class DAE_CORE_API adDomainIndexNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adDomainIndexNode)
	adDomainIndexNode(void);
	adDomainIndexNode(daeDomain* pDomain, size_t nIndex, real_t* pdPointValue);
	virtual ~adDomainIndexNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual const quantity GetQuantity(void) const;

public:
// Report/GUI part
	daeDomain*	m_pDomain;
	size_t		m_nIndex;
	real_t*		m_pdPointValue;
};

/*********************************************************************************************
	adRuntimeVariableNode
**********************************************************************************************/
class DAE_CORE_API adRuntimeVariableNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adRuntimeVariableNode)
	adRuntimeVariableNode(void);
	adRuntimeVariableNode(daeVariable* pVariable, 
	                      size_t nOverallIndex, 
						  std::vector<size_t>& narrDomains);
	virtual ~adRuntimeVariableNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual const quantity GetQuantity(void) const;

public:
// Runtime part
	size_t			m_nOverallIndex;
	size_t			m_nBlockIndex;
	bool			m_bIsAssigned;
// Report/GUI part
	daeVariable*	m_pVariable;
	std::vector<size_t>	m_narrDomains;
};

/*********************************************************************************************
	adRuntimeTimeDerivativeNode
**********************************************************************************************/
class DAE_CORE_API adRuntimeTimeDerivativeNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adRuntimeTimeDerivativeNode)
	adRuntimeTimeDerivativeNode(void);
	adRuntimeTimeDerivativeNode(daeVariable* pVariable, 
	                            size_t nOverallIndex, 
								size_t nDegree, 
								std::vector<size_t>& narrDomains);
	virtual ~adRuntimeTimeDerivativeNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual const quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;

public:
// Runtime part
	real_t*			m_pdTimeDerivative;
	size_t			m_nOverallIndex;
	size_t		    m_nBlockIndex;
	size_t			m_nDegree;
// Report/GUI part
	daeVariable*	m_pVariable;
	std::vector<size_t>	m_narrDomains;
};

/*********************************************************************************************
    adInverseTimeStepNode
**********************************************************************************************/
// Used only in Jacobian expressions!
class DAE_CORE_API adInverseTimeStepNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adInverseTimeStepNode)
    adInverseTimeStepNode(void);
    virtual ~adInverseTimeStepNode(void);

public:
    virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNode* Clone(void) const;
    virtual void	Open(io::xmlTag_t* pTag);
    virtual void	Save(io::xmlTag_t* pTag) const;
    virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool	IsLinear(void) const;
    virtual bool	IsFunctionOfVariables(void) const;
    virtual const quantity GetQuantity(void) const;
};

/*********************************************************************************************
	adRuntimePartialDerivativeNode
**********************************************************************************************/
/*
class DAE_CORE_API adRuntimePartialDerivativeNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adRuntimePartialDerivativeNode)
	adRuntimePartialDerivativeNode(void);
	adRuntimePartialDerivativeNode(daeVariable* pVariable, 
	                               size_t nOverallIndex, 
								   size_t nDegree, 
								   std::vector<size_t>& narrDomains, 
								   daeDomain* pDomain, 
								   adNodePtr pdnode);
	virtual ~adRuntimePartialDerivativeNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual const quantity GetQuantity(void) const;

public:
// Runtime part
	adNodePtr	pardevnode;
	size_t						m_nOverallIndex;
	size_t						m_nDegree;
// Report/GUI part
	daeVariable*		m_pVariable;
	daeDomain*			m_pDomain;
	std::vector<size_t>		m_narrDomains;
};
*/
/*********************************************************************************************
	adUnaryNode
**********************************************************************************************/
class DAE_CORE_API adUnaryNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adUnaryNode)
	adUnaryNode(void);
	adUnaryNode(daeeUnaryFunctions eFun, adNodePtr n);
	virtual ~adUnaryNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;

    adNode* getNodeRawPtr() const 
	{
		return node.get();
	}

public:
	adNodePtr           node;
	daeeUnaryFunctions	eFunction;
};

/*********************************************************************************************
	adBinaryNode
**********************************************************************************************/
class DAE_CORE_API adBinaryNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adBinaryNode)
	adBinaryNode(void);
	adBinaryNode(daeeBinaryFunctions eFun, adNodePtr l, adNodePtr r);
	virtual ~adBinaryNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;
    
    adNode* getLeftRawPtr() const 
	{
		return left.get();
	}
    
    adNode* getRightRawPtr() const 
	{
		return right.get();
	}

public:
	adNodePtr           left;
	adNodePtr           right;
	daeeBinaryFunctions	eFunction;
};

/*********************************************************************************************
	condExpressionNode
**********************************************************************************************/
class DAE_CORE_API condExpressionNode : public condNode
{
public:
	daeDeclareDynamicClass(condExpressionNode)
	condExpressionNode(void);
	condExpressionNode(const adouble& left, daeeConditionType type, const adouble& right);
	condExpressionNode(const adouble& left, daeeConditionType type, real_t right);
	condExpressionNode(real_t left, daeeConditionType type, const adouble& right);
	virtual ~condExpressionNode(void);

public:
	virtual bool			Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual daeCondition	CreateRuntimeNode(const daeExecutionContext* pExecutionContext) const;
	virtual condNode*		Clone(void) const;

	virtual void		Open(io::xmlTag_t* pTag);
	virtual void		Save(io::xmlTag_t* pTag) const;
	virtual string		SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void		SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void		SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void		BuildExpressionsArray(std::vector< adNodePtr > & ptrarrExpressions,
		                                      const daeExecutionContext* pExecutionContext, 
											  real_t dEventTolerance);
	virtual void		AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void		Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual bool		GetQuantity(void) const;

    adNode* getLeftRawPtr() const 
	{
		return m_pLeft.get();
	}
    
    adNode* getRightRawPtr() const 
	{
		return m_pRight.get();
	}
    
public:
	adNodePtr           m_pLeft;
	adNodePtr           m_pRight;
    daeeConditionType	m_eConditionType;
};

/*********************************************************************************************
	condUnaryNode
**********************************************************************************************/
class DAE_CORE_API condUnaryNode : public condNode
{
public:
	daeDeclareDynamicClass(condUnaryNode)
	condUnaryNode(void);
    condUnaryNode(condNodePtr node, daeeLogicalUnaryOperator op);
	virtual ~condUnaryNode(void);

public:
	virtual bool			Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual daeCondition	CreateRuntimeNode(const daeExecutionContext* pExecutionContext) const;
	virtual condNode*		Clone(void) const;

	virtual void		Open(io::xmlTag_t* pTag);
	virtual void		Save(io::xmlTag_t* pTag) const;
	virtual string		SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void		SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void		SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void		BuildExpressionsArray(std::vector< adNodePtr > & ptrarrExpressions,
		                                      const daeExecutionContext* pExecutionContext,
											  real_t dEventTolerance);
	virtual void		AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void		Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual bool		GetQuantity(void) const;
    
    condNode* getNodeRawPtr() const 
	{
		return m_pNode.get();
	}
    
public:
	condNodePtr                 m_pNode;
	daeeLogicalUnaryOperator	m_eLogicalOperator;
};

/*********************************************************************************************
	condBinaryNode
**********************************************************************************************/
class DAE_CORE_API condBinaryNode : public condNode
{
public:
	daeDeclareDynamicClass(condBinaryNode)
	condBinaryNode(void);
    condBinaryNode(condNodePtr left, daeeLogicalBinaryOperator op, condNodePtr right);
	virtual ~condBinaryNode(void);

public:
	virtual bool			Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual daeCondition	CreateRuntimeNode(const daeExecutionContext* pExecutionContext) const;
	virtual condNode*		Clone(void) const;

	virtual void		Open(io::xmlTag_t* pTag);
	virtual void		Save(io::xmlTag_t* pTag) const;
	virtual string		SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void		SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void		SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void		BuildExpressionsArray(std::vector<adNodePtr> & ptrarrExpressions,
		                                      const daeExecutionContext* pExecutionContext,
											  real_t dEventTolerance);
	virtual void		AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void		Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual bool		GetQuantity(void) const;

    condNode* getLeftRawPtr() const 
	{
		return m_pLeft.get();
	}
    
    condNode* getRightRawPtr() const 
	{
		return m_pRight.get();
	}
    
public:
	condNodePtr					m_pLeft;
	condNodePtr					m_pRight;
	daeeLogicalBinaryOperator	m_eLogicalOperator;
};




/*********************************************************************************************
	adSetupParameterNode
**********************************************************************************************/
class DAE_CORE_API adSetupParameterNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adSetupParameterNode)
	adSetupParameterNode(void);
	adSetupParameterNode(daeParameter* pParameter,
	                     std::vector<daeDomainIndex>& arrDomains);
	virtual ~adSetupParameterNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

public:
	daeParameter*			    m_pParameter;
	std::vector<daeDomainIndex>	m_arrDomains;
};

/*********************************************************************************************
	adSetupDomainIteratorNode
**********************************************************************************************/
class DAE_CORE_API adSetupDomainIteratorNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adSetupDomainIteratorNode)
	adSetupDomainIteratorNode(void);
	adSetupDomainIteratorNode(daeDistributedEquationDomainInfo* pDEDI);
	virtual ~adSetupDomainIteratorNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

public:
	daeDistributedEquationDomainInfo* m_pDEDI;
};

/*********************************************************************************************
	adSetupValueInArrayAtIndexNode
**********************************************************************************************/
class DAE_CORE_API adSetupValueInArrayAtIndexNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adSetupValueInArrayAtIndexNode)
	adSetupValueInArrayAtIndexNode(void);
	adSetupValueInArrayAtIndexNode(const daeDomainIndex& domainIndex, adNodeArrayPtr n);
	virtual ~adSetupValueInArrayAtIndexNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
    virtual bool    IsDifferential(void) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

public:
    daeDomainIndex	m_domainIndex;
    adNodeArrayPtr  node;
};

/*********************************************************************************************
	adSetupVariableNode
**********************************************************************************************/
class DAE_CORE_API adSetupVariableNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adSetupVariableNode)
	adSetupVariableNode(void);
	adSetupVariableNode(daeVariable* pVariable,
	                    std::vector<daeDomainIndex>& arrDomains);
	virtual ~adSetupVariableNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

public:
	daeVariable*			    m_pVariable;
	std::vector<daeDomainIndex>	m_arrDomains;
};

/*********************************************************************************************
	adSetupTimeDerivativeNode
**********************************************************************************************/
class DAE_CORE_API adSetupTimeDerivativeNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adSetupTimeDerivativeNode)
	adSetupTimeDerivativeNode(void);
	adSetupTimeDerivativeNode(daeVariable* pVariable, 
	                          size_t nDegree,
							  std::vector<daeDomainIndex>& arrDomains);
	virtual ~adSetupTimeDerivativeNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;

public:
	daeVariable*			    m_pVariable;
	size_t					    m_nDegree;
	std::vector<daeDomainIndex>	m_arrDomains;
};

/*********************************************************************************************
	adSetupPartialDerivativeNode
**********************************************************************************************/
class DAE_CORE_API adSetupPartialDerivativeNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adSetupPartialDerivativeNode)
	adSetupPartialDerivativeNode(void);
	adSetupPartialDerivativeNode(daeVariable* pVariable, 
	                             size_t nDegree, 
								 std::vector<daeDomainIndex>& arrDomains,
								 daeDomain* pDomain);
	virtual ~adSetupPartialDerivativeNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

public:
	daeVariable*			    m_pVariable;
	daeDomain*				    m_pDomain;
	size_t					    m_nDegree;
	std::vector<daeDomainIndex>	m_arrDomains;
};

/*********************************************************************************************
	adScalarExternalFunctionNode
**********************************************************************************************/
class DAE_CORE_API adScalarExternalFunctionNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adScalarExternalFunctionNode)
	adScalarExternalFunctionNode(daeScalarExternalFunction* externalFunction);
	virtual ~adScalarExternalFunctionNode(void);

public:
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode* Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
    virtual bool    IsDifferential(void) const;
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

public:
	daeScalarExternalFunction* m_pExternalFunction;
};

//These two classes (adSetupFEMatrixItemNode and adSetupFEVectorItemNode) are both setup and runtime at the same time
/*********************************************************************************************
    adFEMatrixItemNode
**********************************************************************************************/
class DAE_CORE_API adFEMatrixItemNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adFEMatrixItemNode)
    adFEMatrixItemNode(void);
    adFEMatrixItemNode(const string& strMatrixName, const dae::daeMatrix<real_t>& matrix, size_t row, size_t column, const unit& units);
    virtual ~adFEMatrixItemNode(void);

public:
    virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNode* Clone(void) const;
    virtual void	Open(io::xmlTag_t* pTag);
    virtual void	Save(io::xmlTag_t* pTag) const;
    virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool	IsLinear(void) const;
    virtual bool	IsFunctionOfVariables(void) const;
    virtual bool    IsDifferential(void) const;
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity GetQuantity(void) const;

public:
    string                          m_strMatrixName;
    const dae::daeMatrix<real_t>&   m_matrix;
    size_t                          m_row;
    size_t                          m_column;
    unit                            m_units;
};

/*********************************************************************************************
    adFEVectorItemNode
**********************************************************************************************/
class DAE_CORE_API adFEVectorItemNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adFEVectorItemNode)
    adFEVectorItemNode(void);
    adFEVectorItemNode(const string& strVectorName, const dae::daeArray<real_t>& array, size_t row, const unit& units);
    virtual ~adFEVectorItemNode(void);

public:
    virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const;
    virtual adNode* Clone(void) const;
    virtual void	Open(io::xmlTag_t* pTag);
    virtual void	Save(io::xmlTag_t* pTag) const;
    virtual string  SaveAsLatex(const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const;
    virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
    virtual bool	IsLinear(void) const;
    virtual bool	IsFunctionOfVariables(void) const;
    virtual bool    IsDifferential(void) const;
    virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    virtual const quantity GetQuantity(void) const;

public:
    string                          m_strVectorName;
    const dae::daeArray<real_t>&    m_vector;
    size_t                          m_row;
    unit                            m_units;
};


inline void FillDomains(const std::vector<daeDomainIndex>& arrDomains, std::vector<string>& strarrDomains)
{
	size_t n = arrDomains.size();
	strarrDomains.resize(n);
	for(size_t i = 0; i < n; i++)
	{
		strarrDomains[i] = arrDomains[i].GetIndexAsString();
		
//		if(arrDomains[i].m_eType == eConstantIndex)
//		{
//			strarrDomains[i] = toString<size_t>(arrDomains[i].m_nIndex);
//		}
//		else if(arrDomains[i].m_eType == eDomainIterator)
//		{
//			if(!arrDomains[i].m_pDEDI)
//				daeDeclareAndThrowException(exInvalidCall);
//			strarrDomains[i] = arrDomains[i].m_pDEDI->GetName();
//		}
//		else if(arrDomains[i].m_eType == eIncrementedDomainIterator)
//		{
//			if(!arrDomains[i].m_pDEDI)
//				daeDeclareAndThrowException(exInvalidCall);
//			strarrDomains[i] = arrDomains[i].m_pDEDI->GetName() + (arrDomains[i].m_iIncrement >= 0 ? "+" : "") + toString<int>(arrDomains[i].m_iIncrement);
//		}
//		else
//		{
//			daeDeclareAndThrowException(exInvalidCall);
//		}
	}
}





}
}

#endif
