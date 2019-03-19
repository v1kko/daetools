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
#ifndef DAE_NODES_H
#define DAE_NODES_H

#include "adouble.h"
#include "thermo_package.h"
#include <typeindex>
#include <mutex>

// The most often used node types (in particular in FE models) are:
// Constants, Unary and Binary operators/functions and FloatCoefficientVariableSum.
// These types use a boost memory pool for allocation of setup and runtime nodes.
#include <boost/pool/poolfwd.hpp>
#include <boost/pool/singleton_pool.hpp>
#include <boost/pool/pool_alloc.hpp>
// In large-scale simulations the number of nodes is in millions, therefore
// the size of blocks should also be large to avoid frequent memory allocations
// and memory fragmentation. However, the effect of NEXT_SIZE template argument
// has not been analysed in details.
#define POOL_NEXT_SIZE 1024
#define daePoolAllocator(Class) \
    public: \
        uint8_t memoryPool; \
        struct default_pool_tag{}; \
        struct setup_pool_tag{}; \
        struct runtime_pool_tag{}; \
        static void* operator new(std::size_t sz) \
        { \
            void* ptr; \
            daeeMemoryPool memPool = adNodeImpl::GetMemoryPool(); \
            if(memPool == eSetupNodesPool) \
                ptr = boost::singleton_pool<setup_pool_tag, sizeof(Class), boost::default_user_allocator_new_delete, std::mutex, POOL_NEXT_SIZE>::malloc(); \
            else if(memPool == eRuntimeNodesPool) \
                ptr = boost::singleton_pool<runtime_pool_tag, sizeof(Class), boost::default_user_allocator_new_delete, std::mutex, POOL_NEXT_SIZE>::malloc(); \
            else \
                ptr = ::operator new(sz); \
            Class* c = static_cast<Class*>(ptr); \
            c->memoryPool = static_cast<uint8_t>(memPool); \
            return ptr; \
        } \
        static void operator delete(void* ptr) \
        { \
            Class* c = static_cast<Class*>(ptr); \
            if(c->memoryPool == eSetupNodesPool) \
                boost::singleton_pool<setup_pool_tag, sizeof(Class), boost::default_user_allocator_new_delete, std::mutex, POOL_NEXT_SIZE>::free(ptr); \
            else if(c->memoryPool == eRuntimeNodesPool) \
                boost::singleton_pool<runtime_pool_tag, sizeof(Class), boost::default_user_allocator_new_delete, std::mutex, POOL_NEXT_SIZE>::free(ptr); \
            else \
                ::operator delete(ptr); \
        } \
        static bool release_setup_memory() \
        { \
            return boost::singleton_pool<setup_pool_tag, sizeof(Class), boost::default_user_allocator_new_delete, std::mutex, POOL_NEXT_SIZE>::release_memory(); \
        } \
        static bool purge_setup_memory() \
        { \
            return boost::singleton_pool<setup_pool_tag, sizeof(Class), boost::default_user_allocator_new_delete, std::mutex, POOL_NEXT_SIZE>::purge_memory(); \
        } \
        static bool release_runtime_memory() \
        { \
            return boost::singleton_pool<runtime_pool_tag, sizeof(Class), boost::default_user_allocator_new_delete, std::mutex, POOL_NEXT_SIZE>::release_memory(); \
        } \
        static bool purge_runtime_memory() \
        { \
            return boost::singleton_pool<runtime_pool_tag, sizeof(Class), boost::default_user_allocator_new_delete, std::mutex, POOL_NEXT_SIZE>::purge_memory(); \
        }

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
enum daeeMemoryPool
{
    eHeapMemory = 0,
    eSetupNodesPool,
    eRuntimeNodesPool
};

class DAE_CORE_API adNodeImpl : public adNode
{
public:
    adNodeImpl();
    virtual ~adNodeImpl();

    void Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const;
    void ExportAsLatex(string strFileName);
    bool IsLinear(void) const;
    bool IsFunctionOfVariables(void) const;
    bool IsDifferential(void) const;
    size_t SizeOf(void) const;
    size_t GetHash() const;

    // These two functions are used to keep track of the existing nodes (type and count)
    // and used ONLY for debugging purposes.
    template<typename T>
    static void AddToNodeMap(T* self)
    {
        /*if(adNodeImpl::g_memoryPool != cSetupNodesPool)
            return;
        size_t type_id = typeid(*self).hash_code();
        size_t addr = reinterpret_cast<size_t>(self);
        std::lock_guard<std::mutex> lock(g_mutex);
        std::map<size_t,adNode*>& node_map = adNodeImpl::g_allNodes[type_id];
        node_map[addr] = self;*/
    }
    template<typename T>
    static void RemoveFromNodeMap(T* self)
    {
        /*size_t type_id = typeid(*self).hash_code();
        size_t addr = reinterpret_cast<size_t>(self);
        std::lock_guard<std::mutex> lock(g_mutex);
        std::map<size_t,adNode*>& node_map = adNodeImpl::g_allNodes[type_id];
        node_map.erase(addr);*/
    }

    static void PurgeSetupNodesMemory();
    static void PurgeRuntimeNodesMemory();
    static void ReleaseSetupNodesMemory();
    static void ReleaseRuntimeNodesMemory();

    static void SetMemoryPool(daeeMemoryPool memPool);
    static daeeMemoryPool GetMemoryPool();

    static double HASH_FLOAT_CONSTANT_PRECISION;
    //static std::mutex                                  g_mutex;
    //static std::map<size_t, std::map<size_t,adNode*> > g_allNodes;
private:
    thread_local static daeeMemoryPool g_memoryPool;
};

// boost::hash support
template<typename T>
size_t hash_value(T const& obj)
{
    return obj.GetHash();
}

daeeEquationType DetectEquationType(adNodePtr node);

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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;
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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;

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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;

public:
// Runtime part
    real_t*             m_pdValue;
// Report/GUI part
    daeParameter*       m_pParameter;
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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;

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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;

public:
// Runtime part
    size_t			m_nOverallIndex;
    size_t			m_nBlockIndex;
    bool			m_bIsAssigned;
// Report/GUI part
    daeVariable*        m_pVariable;
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
    virtual quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;
    virtual size_t  SizeOf(void) const;

public:
// Runtime part
    size_t			m_nOverallIndex;
    size_t		    m_nBlockIndex;
// Report/GUI part
    daeVariable*        m_pVariable;
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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;
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
    virtual quantity GetQuantity(void) const;

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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;
    virtual size_t  GetHash() const;

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
    virtual quantity GetQuantity(void) const;

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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;

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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;
    virtual size_t  GetHash() const;

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
    virtual quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;
    virtual size_t  SizeOf(void) const;
    virtual size_t  GetHash() const;

public:
    daeVariable*			    m_pVariable;
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
                                 size_t nOrder,
                                 std::vector<daeDomainIndex>& arrDomains,
                                 daeDomain* pDomain,
                                 daeeDiscretizationMethod eDiscretizationMethod,
                                 const std::map<std::string, std::string>& mapDiscretizationOptions);
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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;

public:
    daeVariable*                        m_pVariable;
    daeDomain*                          m_pDomain;
    size_t                              m_nOrder;
    std::vector<daeDomainIndex>         m_arrDomains;
    daeeDiscretizationMethod            m_eDiscretizationMethod;
    std::map<std::string, std::string>  m_mapDiscretizationOptions;
};

/*********************************************************************************************
    adSetupExpressionDerivativeNode
**********************************************************************************************/
class DAE_CORE_API adSetupExpressionDerivativeNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adSetupExpressionDerivativeNode)
    adSetupExpressionDerivativeNode(void);
    adSetupExpressionDerivativeNode(adNodePtr n);
    virtual ~adSetupExpressionDerivativeNode(void);

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
    virtual quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;
    virtual size_t  SizeOf(void) const;

protected:
    adNodePtr calc_dt(adNodePtr n, const daeExecutionContext* pExecutionContext) const;

public:
    adNodePtr	node;
    size_t		m_nDegree;
};

/*********************************************************************************************
    adSetupExpressionPartialDerivativeNode
**********************************************************************************************/
class DAE_CORE_API adSetupExpressionPartialDerivativeNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adSetupExpressionPartialDerivativeNode)
    adSetupExpressionPartialDerivativeNode(void);
    adSetupExpressionPartialDerivativeNode(daeDomain*                                pDomain,
                                           size_t                                    nOrder,
                                           daeeDiscretizationMethod                  eDiscretizationMethod,
                                           const std::map<std::string, std::string>& mapDiscretizationOptions,
                                           adNodePtr                                 n);
    virtual ~adSetupExpressionPartialDerivativeNode(void);

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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;

    static adNodePtr calc_d(adNodePtr                                 n,
                            daeDomain*                                pDomain,
                            daeeDiscretizationMethod                  eDiscretizationMethod,
                            const std::map<std::string, std::string>& mapDiscretizationOptions,
                            const daeExecutionContext*                pExecutionContext);
    static adNodePtr calc_d2(adNodePtr                                 n,
                             daeDomain*                                pDomain,
                             daeeDiscretizationMethod                  eDiscretizationMethod,
                             const std::map<std::string, std::string>& mapDiscretizationOptions,
                             const daeExecutionContext*                pExecutionContext);

public:
    daeDomain*                          m_pDomain;
    adNodePtr                           node;
    size_t                              m_nOrder;
    daeeDiscretizationMethod            m_eDiscretizationMethod;
    std::map<std::string, std::string>  m_mapDiscretizationOptions;
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
    virtual quantity GetQuantity(void) const;

public:
    daeScalarExternalFunction* m_pExternalFunction;
};


/*********************************************************************************************
    adThermoPhysicalPropertyPackageScalarNode
**********************************************************************************************/
class DAE_CORE_API adThermoPhysicalPropertyPackageScalarNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adThermoPhysicalPropertyPackageScalarNode)
    adThermoPhysicalPropertyPackageScalarNode(daeeThermoPackagePropertyType propType,
                                              const std::string& property_,
                                              daeeThermoPackageBasis basis_,
                                              const std::string& compound_,
                                              const unit& units_,
                                              daeThermoPhysicalPropertyPackage_t* tpp);
    virtual ~adThermoPhysicalPropertyPackageScalarNode(void);

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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;

public:
    daeeThermoPackagePropertyType       propertyType;
    adNodePtr                           pressure;
    adNodePtr                           temperature;
    adNodeArrayPtr                      composition;
    std::string                         phase;
    adNodePtr                           pressure2;
    adNodePtr                           temperature2;
    adNodeArrayPtr                      composition2;
    std::string                         phase2;
    std::string                         property;
    daeeThermoPackageBasis              basis;
    std::string                         compound;
    unit                                units;
    daeThermoPhysicalPropertyPackage_t* thermoPhysicalPropertyPackage;
};

//These two classes (adSetupFEMatrixItemNode and adSetupFEVectorItemNode) are both setup and runtime at the same time
/*********************************************************************************************
    adFEMatrixItemNode
**********************************************************************************************/
/*
class DAE_CORE_API adFEMatrixItemNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adFEMatrixItemNode)
    adFEMatrixItemNode(void);
    adFEMatrixItemNode(const string& strMatrixName, const dae::daeMatrix<adouble>& matrix, size_t row, size_t column, const unit& units);
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
    virtual quantity GetQuantity(void) const;

public:
    string                          m_strMatrixName;
    const dae::daeMatrix<adouble>&  m_matrix;
    size_t                          m_row;
    size_t                          m_column;
    unit                            m_units;
};
*/
/*********************************************************************************************
    adFEVectorItemNode
**********************************************************************************************/
/*
class DAE_CORE_API adFEVectorItemNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adFEVectorItemNode)
    adFEVectorItemNode(void);
    adFEVectorItemNode(const string& strVectorName, const dae::daeArray<adouble>& array, size_t row, const unit& units);
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
    virtual quantity GetQuantity(void) const;

public:
    string                          m_strVectorName;
    const dae::daeArray<adouble>&   m_vector;
    size_t                          m_row;
    unit                            m_units;
};
*/

// Very important!!!
//   The most often used node types (in particular in FE models) use the boost memory pool for memory allocation.
//   They also contain an additional 1 byte long flag memoryPool to keep track about the pool they belong.
//   The default data structure alignment should be changed to prevent much larger memory requirements.
//   In the classes below the largest data members are 8 bytes wide requiring 1 byte wide memoryPool
//   to be 8-byte aligned. This way, a lot of memory is wasted for bytes for padding.
//   This can be addressed by specifying alignment to 1 byte boundary using the #pragma directives
//   available in most of compilers (here, Microsoft VC++ and GNU g++ are of interest).
//   They are not supported by OpenCL but the node classes are never used within OpenCL context.
//   OpenCL supports __attribute__ ((aligned (x))) (where x must be a power of two):
//   typedef struct StructName_
//   {
//   } __attribute__ ((aligned (8))) StructName
#pragma pack(push, 1)

/*********************************************************************************************
    adConstantNode
**********************************************************************************************/
class DAE_CORE_API adConstantNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adConstantNode)
    daePoolAllocator(adConstantNode)
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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;
    virtual size_t  GetHash() const;

public:
    quantity m_quantity;
};

/*********************************************************************************************
    adUnaryNode
**********************************************************************************************/
class DAE_CORE_API adUnaryNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adUnaryNode)
    daePoolAllocator(adUnaryNode)
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
    virtual quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;
    virtual size_t  SizeOf(void) const;
    virtual size_t  GetHash() const;

    adNode* getNodeRawPtr() const;

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
    daePoolAllocator(adBinaryNode)
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
    virtual quantity GetQuantity(void) const;
    virtual bool    IsDifferential(void) const;
    virtual size_t  SizeOf(void) const;
    virtual size_t  GetHash() const;

    adNode* getLeftRawPtr() const;
    adNode* getRightRawPtr() const;

public:
    adNodePtr           left;
    adNodePtr           right;
    daeeBinaryFunctions	eFunction;
};

/*********************************************************************************************
    daeFloatCoefficientVariableProduct
**********************************************************************************************/
class daeFloatCoefficientVariableProduct
{
public:
    daeFloatCoefficientVariableProduct()
    {
        coefficient = 0.0;
        variable    = NULL;
        blockIndex  = ULONG_MAX;
    }

    daeFloatCoefficientVariableProduct(real_t coefficient_, daeVariable* variable_)
    {
        coefficient = coefficient_;
        variable    = variable_;
        blockIndex  = ULONG_MAX;
    }

    size_t GetHash() const
    {
        size_t seed = 0;
        // Important:
        //   When using a hash for equation-templates, it should include only the coefficient and the variable.
        //   The block index is an equation-template argument.
        //   Equations assembled using the FE method produce coefficients that, due to FP rounding, differ
        //   on 12-14th decimal and therefore give different hashes. One way to deal with this is to multiply
        //   with some constant and use only the integer part, i.e. 0.123456789012345*1e12 -> 123456789012.
        long int coeff = (long int)(coefficient*adNodeImpl::HASH_FLOAT_CONSTANT_PRECISION);
        boost::hash_combine(seed, coeff);
        boost::hash_combine(seed, (std::intptr_t)variable);
        //boost::hash_combine(seed, blockIndex);
        return seed;
    }

public:
    real_t       coefficient;
    daeVariable* variable;
    size_t       blockIndex;
};

/*********************************************************************************************
    adFloatCoefficientVariableSumNode
**********************************************************************************************/
// Using 16 chunks in a single allocator block produces the lowest peak RAM requirements and
// lower requirements in individual phases of the simulation initialisation.
#define POOL_ALLOCATOR_NEXT_SIZE 16

class DAE_CORE_API adFloatCoefficientVariableSumNode : public adNodeImpl
{
public:
    daeDeclareDynamicClass(adFloatCoefficientVariableSumNode)
    daePoolAllocator(adFloatCoefficientVariableSumNode)
    adFloatCoefficientVariableSumNode(void);
    adFloatCoefficientVariableSumNode(const adFloatCoefficientVariableSumNode& n);
    virtual ~adFloatCoefficientVariableSumNode(void);

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
    virtual quantity GetQuantity(void) const;
    virtual size_t  SizeOf(void) const;
    virtual size_t  GetHash() const;

    void AddItem(double coefficient, daeVariable* variable, unsigned int variableIndex);
public:
// Runtime part
    typedef boost::fast_pool_allocator< std::pair<const size_t, daeFloatCoefficientVariableProduct>,
                                        boost::default_user_allocator_new_delete,
                                        std::mutex,
                                        POOL_ALLOCATOR_NEXT_SIZE,
                                        0
                                      > allocator;
    typedef boost::singleton_pool< boost::fast_pool_allocator_tag,
                                   sizeof(std::pair<const size_t, daeFloatCoefficientVariableProduct>),
                                   boost::default_user_allocator_new_delete,
                                   std::mutex,
                                   POOL_ALLOCATOR_NEXT_SIZE,
                                   0
                                 > pool_allocator;
    bool                                                    m_bBlockIndexesFound;
    real_t                                                  m_base;
    std::map<size_t, daeFloatCoefficientVariableProduct,
             std::less<size_t>, allocator>                  m_sum;
//    std::map<size_t, daeFloatCoefficientVariableProduct>	m_sum;
};

// Restore the original struct packing
#pragma pack(pop)

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
