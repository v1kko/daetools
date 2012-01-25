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
#ifndef DAE_NODES_ARRAY_H
#define DAE_NODES_ARRAY_H

#include "adouble.h"
#include "nodes.h"

namespace dae 
{
namespace core 
{
inline boost::shared_ptr<adNodeArray> CLONE_NODE_ARRAY(boost::shared_ptr<adNodeArray> n)
{
	if(n)
		return boost::shared_ptr<adNodeArray>(n->Clone());
	else
	{
		daeDeclareAndThrowException(exInvalidCall);
		return boost::shared_ptr<adNodeArray>();
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
	
	virtual void GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
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
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool			IsLinear(void) const;
	virtual bool			IsFunctionOfVariables(void) const;
	virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity	GetQuantity(void) const;

public:
	real_t	m_dValue;
	unit    m_Unit;
};

/*********************************************************************************************
	adRuntimeParameterNodeArray
**********************************************************************************************/
class DAE_CORE_API adRuntimeParameterNodeArray : public adNodeArrayImpl
{
public:
	daeDeclareDynamicClass(adRuntimeParameterNodeArray)
	adRuntimeParameterNodeArray(void);
	virtual ~adRuntimeParameterNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool			IsLinear(void) const;
	virtual bool			IsFunctionOfVariables(void) const;
	virtual const quantity	GetQuantity(void) const;

public:
// Runtime part
	vector<boost::shared_ptr<adNode> >		m_ptrarrParameterNodes;
// Report/GUI part
	daeParameter*							m_pParameter;
	vector<daeArrayRange>					m_arrRanges;
};

/*********************************************************************************************
	adRuntimeVariableNodeArray
**********************************************************************************************/
class DAE_CORE_API adRuntimeVariableNodeArray : public adNodeArrayImpl
{
public:
	daeDeclareDynamicClass(adRuntimeVariableNodeArray)
	adRuntimeVariableNodeArray(void);
	virtual ~adRuntimeVariableNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool			IsLinear(void) const;
	virtual bool			IsFunctionOfVariables(void) const;
	virtual const quantity	GetQuantity(void) const;

public:
// Runtime part
	vector< boost::shared_ptr<adNode> >		m_ptrarrVariableNodes;
// Report/GUI part
	daeVariable*							m_pVariable;
	vector<daeArrayRange>					m_arrRanges;
};

/*********************************************************************************************
	adRuntimeTimeDerivativeNodeArray
**********************************************************************************************/
class DAE_CORE_API adRuntimeTimeDerivativeNodeArray : public adNodeArrayImpl
{
public:
	daeDeclareDynamicClass(adRuntimeTimeDerivativeNodeArray)
	adRuntimeTimeDerivativeNodeArray(void);
	virtual ~adRuntimeTimeDerivativeNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual const quantity	GetQuantity(void) const;

public:
// Runtime part
	vector< boost::shared_ptr<adNode> >	m_ptrarrTimeDerivativeNodes;
	size_t								m_nDegree;
// Report/GUI part
	daeVariable*						m_pVariable;
	vector<daeArrayRange>				m_arrRanges;
};

/*********************************************************************************************
	adRuntimePartialDerivativeNodeArray
**********************************************************************************************/
class DAE_CORE_API adRuntimePartialDerivativeNodeArray : public adNodeArrayImpl
{
public:
	daeDeclareDynamicClass(adRuntimePartialDerivativeNodeArray)
	adRuntimePartialDerivativeNodeArray(void);
	virtual ~adRuntimePartialDerivativeNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool			IsLinear(void) const;
	virtual bool			IsFunctionOfVariables(void) const;
	virtual const quantity	GetQuantity(void) const;

public:
// Runtime part
	vector< boost::shared_ptr<adNode> >	m_ptrarrPartialDerivativeNodes;
	size_t								m_nDegree;
// Report/GUI part
	daeVariable*						m_pVariable;
	daeDomain*							m_pDomain;
	vector<daeArrayRange>				m_arrRanges;
};

/*********************************************************************************************
	adRuntimeSpecialFunctionNode
**********************************************************************************************/
class DAE_CORE_API adRuntimeSpecialFunctionNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adRuntimeSpecialFunctionNode)
	adRuntimeSpecialFunctionNode(void);
	adRuntimeSpecialFunctionNode(daeeSpecialUnaryFunctions eFun, 
					             daeModel* pModel,
								 boost::shared_ptr<adNodeArray> n);
	virtual ~adRuntimeSpecialFunctionNode(void);

public:
	virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode*	Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string	SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual const quantity GetQuantity(void) const;

public:
// Runtime part
	daeModel*						m_pModel;
	boost::shared_ptr<adNodeArray>	node;
// Report/GUI part
	daeeSpecialUnaryFunctions		eFunction;
};

/*********************************************************************************************
	adRuntimeIntegralNode
**********************************************************************************************/
class DAE_CORE_API adRuntimeIntegralNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adRuntimeIntegralNode)
	adRuntimeIntegralNode(void);
	adRuntimeIntegralNode(daeeIntegralFunctions eFun,
						  daeModel* pModel,
						  boost::shared_ptr<adNodeArray> n,
	                      daeDomain* pDomain,
						  const vector<const real_t*>& pdarrPoints);
	virtual ~adRuntimeIntegralNode(void);

public:
	virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode*	Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string	SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual const quantity GetQuantity(void) const;

public:
	boost::shared_ptr<adNodeArray>	node;
	daeModel*						m_pModel;
	daeeIntegralFunctions			eFunction;
	daeDomain*						m_pDomain;
	vector<const real_t*>			m_pdarrPoints;
};

/*********************************************************************************************
	adUnaryNodeArray
**********************************************************************************************/
class DAE_CORE_API adUnaryNodeArray : public adNodeArrayImpl
{
public:
	daeDeclareDynamicClass(adUnaryNodeArray)
	adUnaryNodeArray(void);
	adUnaryNodeArray(daeeUnaryFunctions eFun, boost::shared_ptr<adNodeArray> n);
	virtual ~adUnaryNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool			IsLinear(void) const;
	virtual bool			IsFunctionOfVariables(void) const;
	virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity	GetQuantity(void) const;

public:
	boost::shared_ptr<adNodeArray>	node;
	daeeUnaryFunctions				eFunction;
};

/*********************************************************************************************
	adBinaryNodeArray
**********************************************************************************************/
class DAE_CORE_API adBinaryNodeArray : public adNodeArrayImpl
{
public:
	daeDeclareDynamicClass(adBinaryNodeArray)
	adBinaryNodeArray(void);
	adBinaryNodeArray(daeeBinaryFunctions eFun, boost::shared_ptr<adNodeArray> l, boost::shared_ptr<adNodeArray> r);
	virtual ~adBinaryNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool			IsLinear(void) const;
	virtual bool			IsFunctionOfVariables(void) const;
	virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity	GetQuantity(void) const;

public:
	boost::shared_ptr<adNodeArray>	left;
	boost::shared_ptr<adNodeArray>	right;
	daeeBinaryFunctions				eFunction;
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
					           daeModel* pModel,
							   boost::shared_ptr<adNodeArray> n);
	virtual ~adSetupSpecialFunctionNode(void);

public:
	virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode*	Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string	SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual bool	IsLinear(void) const;
	virtual bool	IsFunctionOfVariables(void) const;
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

public:
	daeModel*						m_pModel;
	boost::shared_ptr<adNodeArray>	node;
	daeeSpecialUnaryFunctions		eFunction;
};

/*********************************************************************************************
	adSetupExpressionDerivativeNode
**********************************************************************************************/
class DAE_CORE_API adSetupExpressionDerivativeNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adSetupExpressionDerivativeNode)
	adSetupExpressionDerivativeNode(void);
	adSetupExpressionDerivativeNode(daeModel* pModel, boost::shared_ptr<adNode> n);
	virtual ~adSetupExpressionDerivativeNode(void);

public:
	virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode*	Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string	SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

protected:
	boost::shared_ptr<adNode> calc_dt(boost::shared_ptr<adNode> n, const daeExecutionContext* pExecutionContext) const;
	
public:
	daeModel*					m_pModel;
	boost::shared_ptr<adNode>	node;
	size_t						m_nDegree;
};

/*********************************************************************************************
	adSetupExpressionPartialDerivativeNode
**********************************************************************************************/
class DAE_CORE_API adSetupExpressionPartialDerivativeNode : public adNodeImpl
{
public:
	daeDeclareDynamicClass(adSetupExpressionPartialDerivativeNode)
	adSetupExpressionPartialDerivativeNode(void);
	adSetupExpressionPartialDerivativeNode(daeModel* pModel, daeDomain* pDomain, boost::shared_ptr<adNode> n);
	virtual ~adSetupExpressionPartialDerivativeNode(void);

public:
	virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode*	Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string	SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

protected:
	boost::shared_ptr<adNode> calc_d(boost::shared_ptr<adNode> n, daeDomain* pDomain, const daeExecutionContext* pExecutionContext) const;
	
public:
	daeModel*					m_pModel;
	daeDomain*					m_pDomain;
	boost::shared_ptr<adNode>	node;
	size_t						m_nDegree;
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
					    daeModel* pModel,
						boost::shared_ptr<adNodeArray> n,
		                daeDomain* pDomain,
						const daeArrayRange& arrayRange);
	virtual ~adSetupIntegralNode(void);

public:
	virtual adouble	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNode*	Clone(void) const;
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual string	SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void	AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void	Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity GetQuantity(void) const;

public:
	daeModel*						m_pModel;
	boost::shared_ptr<adNodeArray>	node;
	daeDomain*						m_pDomain;
	daeeIntegralFunctions			eFunction;
	daeArrayRange					m_ArrayRange;
};

/*********************************************************************************************
	adSingleNodeArray
**********************************************************************************************/
class DAE_CORE_API adSingleNodeArray : public adNodeArrayImpl
{
public:
	daeDeclareDynamicClass(adSingleNodeArray)
	adSingleNodeArray(void);
	adSingleNodeArray(boost::shared_ptr<adNode> n);
	virtual ~adSingleNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity	GetQuantity(void) const;

public:
	boost::shared_ptr<adNode>	node;
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
							  vector<daeArrayRange>& arrRanges);
	virtual ~adSetupParameterNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity	GetQuantity(void) const;

public:
	daeParameter*			m_pParameter;
	vector<daeArrayRange>	m_arrRanges;
};

/*********************************************************************************************
	adSetupVariableNodeArray
**********************************************************************************************/
class DAE_CORE_API adSetupVariableNodeArray : public adNodeArrayImpl
{
public:
	daeDeclareDynamicClass(adSetupVariableNodeArray)
	adSetupVariableNodeArray(void);
	adSetupVariableNodeArray(daeVariable* pVariable,
	                         vector<daeArrayRange>& arrRanges);
	virtual ~adSetupVariableNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity	GetQuantity(void) const;

public:
	daeVariable*			m_pVariable;
	vector<daeArrayRange>	m_arrRanges;
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
	                               size_t nDegree,
							       vector<daeArrayRange>& arrRanges);
	virtual ~adSetupTimeDerivativeNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity	GetQuantity(void) const;

public:
	daeVariable*			m_pVariable;
	size_t					m_nDegree;
	vector<daeArrayRange>	m_arrRanges;
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
	                                  size_t nDegree, 
								      vector<daeArrayRange>& arrRanges,
								      daeDomain* pDomain);
	virtual ~adSetupPartialDerivativeNodeArray(void);

public:
	virtual size_t			GetSize(void) const;
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity	GetQuantity(void) const;

public:
	daeVariable*			m_pVariable;
	daeDomain*				m_pDomain;
	size_t					m_nDegree;
	vector<daeArrayRange>	m_arrRanges;
};


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
	virtual void			GetArrayRanges(vector<daeArrayRange>& arrRanges) const;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const;
	virtual adNodeArray*	Clone(void) const;
	virtual void			Open(io::xmlTag_t* pTag);
	virtual void			Save(io::xmlTag_t* pTag) const;
	virtual string			SaveAsLatex(const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const;
	virtual void			AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed);
	virtual void			Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
	virtual const quantity	GetQuantity(void) const;

public:
	daeVectorExternalFunction* m_pExternalFunction;
};


inline void FillDomains(const vector<daeArrayRange>& arrRanges, vector<string>& strarrDomains)
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
