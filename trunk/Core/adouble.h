#if !defined(ADOLC_ADOUBLE_H)
#define ADOLC_ADOUBLE_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#ifdef DAEDLL
#ifdef MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllexport)
#else // MODEL_EXPORTS
#define DAE_CORE_API __declspec(dllimport)
#endif // MODEL_EXPORTS
#else // DAEDLL
#define DAE_CORE_API
#endif // DAEDLL

#else // WIN32
#define DAE_CORE_API 
#endif // WIN32

// Some M$ macro crap
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include "definitions.h"
#include "io_impl.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <float.h>
#include <stack>

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	daeCondition
**********************************************************************************************/
class condNode;
class adNode;
class adNodeArray;
class daeModel;
class daeExecutionContext;
class DAE_CORE_API daeCondition : public io::daeSerializable
{
public:
	daeDeclareDynamicClass(daeCondition)
	daeCondition(void);
	daeCondition(boost::shared_ptr<condNode> condition);
	virtual ~daeCondition(void);

public:
	virtual void	Open(io::xmlTag_t* pTag);
	virtual void	Save(io::xmlTag_t* pTag) const;
	virtual void	OpenRuntime(io::xmlTag_t* pTag);
	virtual void	SaveRuntime(io::xmlTag_t* pTag) const;

	virtual bool	Evaluate(const daeExecutionContext* pExecutionContext) const;
					operator bool(void);

    daeCondition operator ||(const daeCondition& rCondition) const;
    daeCondition operator &&(const daeCondition& rCondition) const;
    daeCondition operator |(const daeCondition& rCondition) const;
    daeCondition operator &(const daeCondition& rCondition) const;
    daeCondition operator !(void) const;
	
	void   SetEventTolerance(real_t dEventTolerance);
	real_t GetEventTolerance(void);

	void BuildExpressionsArray(const daeExecutionContext* pExecutionContext);
	string SaveNodeAsPlainText(void) const;

protected:
	void   SaveNodeAsMathML(io::xmlTag_t* pTag, const string& strObjectName) const;

public:
	daeModel*									m_pModel;
	boost::shared_ptr<condNode>					m_pConditionNode;
	std::vector< boost::shared_ptr<adNode> >	m_ptrarrExpressions;
	real_t										m_dEventTolerance;
};

/*********************************************************************************************
	adouble
**********************************************************************************************/
class DAE_CORE_API adouble 
{
public:
    adouble(void);
    adouble(const real_t value);
    adouble(const real_t value, real_t derivative);
    adouble(const adouble& a);
    virtual ~adouble();

public:
    const adouble operator -(void) const;
    const adouble operator +(void) const;

    //const adouble operator +(const real_t v) const;
    const adouble operator +(const adouble& a) const;
    friend DAE_CORE_API const adouble operator +(const real_t v, const adouble& a);
    friend DAE_CORE_API const adouble operator +(const adouble& a, const real_t v);

    //const adouble operator -(const real_t v) const;
    const adouble operator -(const adouble& a) const;
    friend DAE_CORE_API const adouble operator -(const real_t v, const adouble& a);
    friend DAE_CORE_API const adouble operator -(const adouble& a, const real_t v);

    //const adouble operator *(const real_t v) const;
    const adouble operator *(const adouble& a) const;
    friend DAE_CORE_API const adouble operator *(const real_t v, const adouble& a);
    friend DAE_CORE_API const adouble operator *(const adouble& a, const real_t v);

    //const adouble operator /(const real_t v) const;
    const adouble operator /(const adouble& a) const;
    friend DAE_CORE_API const adouble operator /(const real_t v, const adouble& a);
    friend DAE_CORE_API const adouble operator /(const adouble& a, const real_t v);

    friend DAE_CORE_API const adouble exp(const adouble &a);
    friend DAE_CORE_API const adouble log(const adouble &a);
    friend DAE_CORE_API const adouble sqrt(const adouble &a);
    friend DAE_CORE_API const adouble sin(const adouble &a);
    friend DAE_CORE_API const adouble cos(const adouble &a);
    friend DAE_CORE_API const adouble tan(const adouble &a);
    friend DAE_CORE_API const adouble asin(const adouble &a);
    friend DAE_CORE_API const adouble acos(const adouble &a);
    friend DAE_CORE_API const adouble atan(const adouble &a);

    friend DAE_CORE_API const adouble pow(const adouble &a, real_t v);
    friend DAE_CORE_API const adouble pow(const adouble &a, const adouble &b);
    friend DAE_CORE_API const adouble pow(real_t v, const adouble &a);
    friend DAE_CORE_API const adouble log10(const adouble &a);

    friend DAE_CORE_API const adouble abs(const adouble &a);
    friend DAE_CORE_API const adouble ceil(const adouble &a);
    friend DAE_CORE_API const adouble floor(const adouble &a);
    friend DAE_CORE_API const adouble max(const adouble &a, const adouble &b);
    friend DAE_CORE_API const adouble max(real_t v, const adouble &a);
    friend DAE_CORE_API const adouble max(const adouble &a, real_t v);
    friend DAE_CORE_API const adouble min(const adouble &a, const adouble &b);
    friend DAE_CORE_API const adouble min(real_t v, const adouble &a);
    friend DAE_CORE_API const adouble min(const adouble &a, real_t v);

    void operator =(const real_t v);
    void operator =(const adouble& a);

    daeCondition operator !=(const adouble&) const;
    daeCondition operator !=(const real_t) const;
    friend daeCondition operator !=(const real_t, const adouble&);

    daeCondition operator ==(const adouble&) const;
    daeCondition operator ==(const real_t) const;
    friend daeCondition operator ==(const real_t, const adouble&);

    daeCondition operator <=(const adouble&) const;
    daeCondition operator <=(const real_t) const;
    friend daeCondition operator <=(const real_t, const adouble&);

    daeCondition operator >=(const adouble&) const;
    daeCondition operator >=(const real_t) const;
    friend daeCondition operator >= (const real_t, const adouble&);

    daeCondition operator >(const adouble&) const;
    daeCondition operator >(const real_t) const;
    friend daeCondition operator >(const real_t, const adouble&);

    daeCondition operator <(const adouble&) const;
    daeCondition operator <(const real_t) const;
    friend daeCondition operator <(const real_t, const adouble&);

	real_t getValue() const 
	{
		return m_dValue;
	}
	void setValue(const real_t v) 
	{
		m_dValue = v;
	}
	real_t getDerivative() const 
	{
		return m_dDeriv;
	}
	void setDerivative(real_t v) 
	{
		m_dDeriv = v;
	}	   
	bool getGatherInfo(void) const
	{
		return m_bGatherInfo;
	}
	void setGatherInfo(bool bGatherInfo)
	{
		m_bGatherInfo = bGatherInfo;
	}

	boost::shared_ptr<adNode>	node;
	//static bool bIsParsing;

private:
    real_t m_dValue;
    real_t m_dDeriv;
	bool   m_bGatherInfo;
};

// Issues with daeModel::min/max
inline const adouble max_(const adouble &a, const adouble &b)
{
	return max(a, b);
}

inline const adouble min_(const adouble &a, const adouble &b)
{
	return min(a, b);
}

/******************************************************************
	adouble_array
*******************************************************************/
class DAE_CORE_API adouble_array
{
public:
	adouble_array(void);
	adouble_array(const adouble_array& a);
	virtual ~adouble_array(void);

public:
	size_t GetSize(void) const;
	void   Resize(size_t n);
	adouble& operator[](size_t n);
	const adouble& operator[](size_t n) const;
	adouble GetItem(size_t n);

	void operator =(const adouble_array& a);
	
	const adouble_array operator -(void) const;
	
    const adouble_array operator +(const adouble_array& a) const;
    const adouble_array operator +(const real_t v) const;
    const adouble_array operator +(const adouble& a) const;
	
    const adouble_array operator -(const adouble_array& a) const;
    const adouble_array operator -(const real_t v) const;
    const adouble_array operator -(const adouble& a) const;
	
    const adouble_array operator *(const adouble_array& a) const;
    const adouble_array operator *(const real_t v) const;
    const adouble_array operator *(const adouble& a) const;
	
    const adouble_array operator /(const adouble_array& a) const;
    const adouble_array operator /(const real_t v) const;
    const adouble_array operator /(const adouble& a) const;
	
	bool getGatherInfo(void) const;
	void setGatherInfo(bool bGatherInfo);

public:
	bool							m_bGatherInfo;
	boost::shared_ptr<adNodeArray>	node;
	std::vector<adouble>			m_arrValues;
};

DAE_CORE_API const adouble_array operator +(const real_t v, const adouble_array& a);
DAE_CORE_API const adouble_array operator -(const real_t v, const adouble_array& a);
DAE_CORE_API const adouble_array operator *(const real_t v, const adouble_array& a);
DAE_CORE_API const adouble_array operator /(const real_t v, const adouble_array& a);

DAE_CORE_API const adouble_array operator +(const adouble& a, const adouble_array& arr);
DAE_CORE_API const adouble_array operator -(const adouble& a, const adouble_array& arr);
DAE_CORE_API const adouble_array operator *(const adouble& a, const adouble_array& arr);
DAE_CORE_API const adouble_array operator /(const adouble& a, const adouble_array& arr);

DAE_CORE_API const adouble_array exp(const adouble_array& a);
DAE_CORE_API const adouble_array sqrt(const adouble_array& a);
DAE_CORE_API const adouble_array log(const adouble_array& a);
DAE_CORE_API const adouble_array log10(const adouble_array& a);
DAE_CORE_API const adouble_array abs(const adouble_array& a);
DAE_CORE_API const adouble_array floor(const adouble_array& a);
DAE_CORE_API const adouble_array ceil(const adouble_array& a);
DAE_CORE_API const adouble_array sin(const adouble_array& a);
DAE_CORE_API const adouble_array cos(const adouble_array& a);
DAE_CORE_API const adouble_array tan(const adouble_array& a);
DAE_CORE_API const adouble_array asin(const adouble_array& a);
DAE_CORE_API const adouble_array acos(const adouble_array& a);
DAE_CORE_API const adouble_array atan(const adouble_array& a);

//DAE_CORE_API const adouble_array pow(const adouble_array& a, real_t v);

/*********************************************************************************************
	daeFPUCommand
**********************************************************************************************/
typedef struct DAE_CORE_API daeFPUCommandInfo_t
{
	unsigned kind:		2;
	unsigned opcode:	6;
	unsigned loperand:	3;
	unsigned roperand:	3;
	unsigned result:	2;
} daeFPUCommandInfo;

typedef struct DAE_CORE_API daeFPUCommand_t
{
	daeFPUCommandInfo	info;
	real_t				lvalue;
	real_t				rvalue;
} daeFPUCommand;

/*********************************************************************************************
	adNode
**********************************************************************************************/
const int Nbinaryfns = 7;
const int Nunaryfns  = 12;

static const string strarrBinaryFns[7]={"plus",
										"minus",
										"times",
										"divide",
										"power",
										"min",
										"max"
									    };
static const string strarrUnaryFns[12]={"minus",
										"sin",
										"cos",
										"tan",
										"arcsin",
										"arccos",
										"arctan",
										"root",
										"exp",
										"ln",
										"log",
										"abs"
										};

class daeSaveAsMathMLContext;
class DAE_CORE_API adNode
{
public:
	virtual ~adNode(void){}

public:
	virtual adNode* Clone(void) const												= 0;
	virtual adouble Evaluate(const daeExecutionContext* pExecutionContext) const	= 0;

	virtual void	Open(io::xmlTag_t* pTag)										= 0;
	virtual void	Save(io::xmlTag_t* pTag) const									= 0;

	virtual string  SaveAsLatex(const daeSaveAsMathMLContext* c) const				= 0;
	virtual string  SaveAsPlainText(const daeSaveAsMathMLContext* c) const			= 0;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, 
		                                const daeSaveAsMathMLContext* c) const		= 0;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, 
		                                     const daeSaveAsMathMLContext* c) const	= 0;

	virtual void	AddVariableIndexToArray(std::map<size_t, size_t>& mapIndexes)	= 0;

	static adNode*	CreateNode(const io::xmlTag_t* pTag);
	static void		SaveNode(io::xmlTag_t* pTag, const string& strObjectName, const adNode* node);
	static adNode*	OpenNode(io::xmlTag_t* pTag, const string& strObjectName, io::daeOnOpenObjectDelegate_t<adNode>* ood = NULL);
	static void		SaveNodeAsMathML(io::xmlTag_t* pTag, 
									 const string& strObjectName, 
									 const adNode* node, 
									 const daeSaveAsMathMLContext* c, 
									 bool bAppendEqualToZero = false);
};

#define CLONE_NODE(NODE, VALUE) (  boost::shared_ptr<adNode>(  (NODE ? NODE->Clone() : new adConstantNode(VALUE))  )  )

/*********************************************************************************************
	adNodeArray
**********************************************************************************************/
class DAE_CORE_API adNodeArray
{
public:
	virtual ~adNodeArray(void){}

public:
	virtual size_t			GetSize(void) const												= 0;
	virtual adNodeArray*	Clone(void) const												= 0;
	virtual adouble_array	Evaluate(const daeExecutionContext* pExecutionContext) const	= 0;

	virtual void	Open(io::xmlTag_t* pTag)												= 0;
	virtual void	Save(io::xmlTag_t* pTag) const											= 0;

	virtual string  SaveAsLatex(const daeSaveAsMathMLContext* c) const						= 0;
	virtual string  SaveAsPlainText(const daeSaveAsMathMLContext* c) const					= 0;
	virtual void	SaveAsContentMathML(io::xmlTag_t* pTag, 
		                                const daeSaveAsMathMLContext* c) const				= 0;
	virtual void	SaveAsPresentationMathML(io::xmlTag_t* pTag, 
		                                     const daeSaveAsMathMLContext* c) const			= 0;

	virtual void	AddVariableIndexToArray(std::map<size_t, size_t>& mapIndexes)			= 0;

	static adNodeArray*	CreateNode(const io::xmlTag_t* pTag);
	static void			SaveNode(io::xmlTag_t* pTag, const string& strObjectName, const adNodeArray* node);
	static adNodeArray*	OpenNode(io::xmlTag_t* pTag, const string& strObjectName, io::daeOnOpenObjectDelegate_t<adNodeArray>* ood = NULL);
	
	static void			SaveRuntimeNodeArrayAsPresentationMathML(io::xmlTag_t* pTag, 
														         const std::vector< boost::shared_ptr<adNode> >& arrNodes, 
														         const daeSaveAsMathMLContext* c);
	static string		SaveRuntimeNodeArrayAsLatex(const std::vector< boost::shared_ptr<adNode> >& arrNodes, 
											        const daeSaveAsMathMLContext* c);
	static string		SaveRuntimeNodeArrayAsPlainText(const std::vector< boost::shared_ptr<adNode> >& arrNodes, 
												        const daeSaveAsMathMLContext* c);
};

/*********************************************************************************************
	condNode
**********************************************************************************************/
class DAE_CORE_API condNode
{
public:
	virtual ~condNode(void){}

public:	
	virtual condNode*		Clone(void) const																= 0;
	virtual bool			Evaluate(const daeExecutionContext* pExecutionContext) const					= 0;
	virtual daeCondition	CreateRuntimeNode(const daeExecutionContext* pExecutionContext) const			= 0;

	virtual void		Open(io::xmlTag_t* pTag)															= 0;
	virtual void		Save(io::xmlTag_t* pTag) const														= 0;

	virtual string		SaveAsLatex(const daeSaveAsMathMLContext* c) const									= 0;
	virtual string		SaveAsPlainText(const daeSaveAsMathMLContext* c) const								= 0;
	virtual void		SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const		= 0;
	virtual void		SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const = 0;

	virtual void		BuildExpressionsArray(std::vector<boost::shared_ptr<adNode> >& ptrarrExpressions, 
		                                      const daeExecutionContext* pExecutionContext,
											  real_t dEventTolerance)										= 0;
	virtual void		AddVariableIndexToArray(std::map<size_t, size_t>& mapIndexes)						= 0;

	static condNode*	CreateNode(const io::xmlTag_t* pTag);
	static void			SaveNode(io::xmlTag_t* pTag, const string& strObjectName, const condNode* node);
	static condNode*	OpenNode(io::xmlTag_t* pTag, const string& strObjectName, io::daeOnOpenObjectDelegate_t<condNode>* ood = NULL);
	static void			SaveNodeAsMathML(io::xmlTag_t* pTag, const string& strObjectName, const condNode* node, const daeSaveAsMathMLContext* c);
};


}
}


#endif
