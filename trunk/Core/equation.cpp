#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "xmlfunctions.h"
#include <typeinfo>

namespace dae 
{
namespace core 
{
/******************************************************************
	daeEquationExecutionInfo
*******************************************************************/
daeEquationExecutionInfo::daeEquationExecutionInfo(daeEquation* pEquation)
{
    if(!pEquation)
        daeDeclareAndThrowException(exInvalidPointer);
        
	m_dScaling  = 1.0;
    m_pEquation = pEquation;
}

daeEquationExecutionInfo::~daeEquationExecutionInfo()
{
}

void daeEquationExecutionInfo::Open(io::xmlTag_t* pTag)
{
	string strName;

	io::daeSerializable::Open(pTag);

	daeFindBlockByID blockdel(NULL);
	daeFindModelByID modeldel(NULL);
	daeFindEquationByID eqndel(NULL);
	daeFindDomainByID domaindel(NULL);

	m_mapIndexes.clear();
	m_narrDomainIndexes.clear();

//	strName = "Block";
//	m_pBlock = pTag->OpenObjectRef<daeBlock>(strName, &blockdel);

//	strName = "Model";
//	m_pModel = pTag->OpenObjectRef<daeModel>(strName, &modeldel);

//	strName = "Equation";
//	m_pEquation = pTag->OpenObjectRef<daeEquation>(strName, &eqndel);

	strName = "EquationIndexInBlock";
	pTag->Open(strName, m_nEquationIndexInBlock);

	strName = "DomainIndexes";
	pTag->OpenArray(strName, m_narrDomainIndexes);

	strName = "mapIndexes";
	pTag->OpenMap(strName, m_mapIndexes);

//	strName = "m_ptrarrDomains";
//	pTag->OpenObjectRefArray(strName, m_ptrarrDomains, &domaindel);

	strName = "EquationEvaluationNode";
	adNode* node = adNode::OpenNode(pTag, strName);
	m_EquationEvaluationNode.reset(node);
}

void daeEquationExecutionInfo::Save(io::xmlTag_t* pTag) const
{
	string strName;

	io::daeSerializable::Save(pTag);

//	strName = "Block";
//	pTag->SaveObjectRef(strName, m_pBlock);

//	strName = "Model";
//	pTag->SaveObjectRef(strName, m_pModel);

//	strName = "Equation";
//	pTag->SaveObjectRef(strName, m_pEquation);

	strName = "EquationIndexInBlock";
	pTag->Save(strName, m_nEquationIndexInBlock);

	strName = "DomainIndexes";
	pTag->SaveArray(strName, m_narrDomainIndexes);

	strName = "mapIndexes";
	pTag->SaveMap(strName, m_mapIndexes);

//	strName = "m_ptrarrDomains";
//	pTag->SaveObjectRefArray(strName, m_ptrarrDomains);

	strName = "EquationEvaluationNode";
	pTag->SaveObject(strName, m_EquationEvaluationNode.get());
}

void daeEquationExecutionInfo::OpenRuntime(io::xmlTag_t* pTag)
{
}

void daeEquationExecutionInfo::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	io::daeSerializable::Save(pTag);

//	strName = "Model";
//	pTag->SaveObjectRef(strName, m_pModel);

//	strName = "Equation";
//	pTag->SaveObjectRef(strName, m_pEquation);

//	strName = "EquationIndexInBlock";
//	pTag->Save(strName, m_nEquationIndexInBlock);

//	strName = "DomainIndexes";
//	pTag->SaveArray(strName, m_narrDomainIndexes);

//	strName = "mapIndexes";
//	pTag->SaveMap(strName, m_mapIndexes);

//	strName = "m_ptrarrDomains";
//	pTag->SaveObjectRefArray(strName, m_ptrarrDomains);

	daeNodeSaveAsContext c(NULL); //m_pModel);

	strName = "EquationEvaluationNode";
	adNode::SaveNodeAsMathML(pTag, string("MathML"), m_EquationEvaluationNode.get(), &c, true);
	
	strName = "IsLinear";
	pTag->Save(strName, m_EquationEvaluationNode->IsLinear());
}

adNode* daeEquationExecutionInfo::GetEquationEvaluationNodeRawPtr(void) const
{
	return m_EquationEvaluationNode.get();
}

daeeEquationType daeEquationExecutionInfo::GetEquationType(void) const
{
    return DetectEquationType(m_EquationEvaluationNode);
}

daeEquation* daeEquationExecutionInfo::GetEquation(void) const
{
    return m_pEquation;
}

std::string daeEquationExecutionInfo::GetName(void) const
{
    std::string strName = m_pEquation->GetCanonicalName();
    if(!m_narrDomainIndexes.empty())
        strName += "(" + toString(m_narrDomainIndexes, string(",")) + ")";
    return strName;
}

const std::map< size_t, std::pair<size_t, adNodePtr> >& daeEquationExecutionInfo::GetJacobianExpressions() const
{
    return m_mapJacobianExpressions;
}

void daeEquationExecutionInfo::GatherInfo(daeExecutionContext& EC, daeEquation* pEquation, daeModel* pModel)
{
// Here m_pDataProxy->GatherInfo is true!!

// We need to propagate this as a global execution context for it will be used to add variable indexes
	EC.m_pEquationExecutionInfo = this;
	pModel->PropagateGlobalExecutionContext(&EC);

// Get a runtime node by evaluating the setup node
	pEquation->GatherInfo(m_narrDomainIndexes, EC, m_EquationEvaluationNode);

// Restore NULL as a global execution context
	pModel->PropagateGlobalExecutionContext(NULL);
}

void daeEquationExecutionInfo::Residual(daeExecutionContext& EC)
{
#ifdef DAE_DEBUG
	if(!EC.m_pBlock)
		daeDeclareAndThrowException(exInvalidPointer);
#endif

    bool bPrintInfo = m_pEquation->m_pModel->m_pDataProxy->PrintInfo();
    if(bPrintInfo)
    {
        m_pEquation->m_pModel->m_pDataProxy->LogMessage(string("  Residual for equation no. ") + toString(m_nEquationIndexInBlock) + string(": ") + GetName(), 0);
    }

	EC.m_pEquationExecutionInfo = this;

    try
    {
        adouble __ad = m_EquationEvaluationNode->Evaluate(&EC) * m_dScaling;
        EC.m_pBlock->SetResidual(m_nEquationIndexInBlock, __ad.getValue());
    }
    catch(std::exception& exc)
    {
        daeDeclareException(exInvalidCall);
        e << "Exception raised during calculation of the Residual of the equation [" << GetName() << "]; " << exc.what();
        throw e;
    }
}

void daeEquationExecutionInfo::Jacobian(daeExecutionContext& EC)
{
#ifdef DAE_DEBUG
	if(!EC.m_pBlock)
		daeDeclareAndThrowException(exInvalidPointer);
#endif

	EC.m_pEquationExecutionInfo = this;

	adouble __ad;
    double startTime, endTime;

    bool bPrintInfo = m_pEquation->m_pModel->m_pDataProxy->PrintInfo();
    if(bPrintInfo)
    {
        size_t count = m_mapIndexes.size();
        m_pEquation->m_pModel->m_pDataProxy->LogMessage(string("  Jacobian for equation no. ") + toString(m_nEquationIndexInBlock) + string(": ") + GetName(), 0);
        m_pEquation->m_pModel->m_pDataProxy->LogMessage(string("     Map of variable indexes (size = ") + toString(count) + string(")"), 0);
        //m_pEquation->m_pModel->m_pDataProxy->LogMessage(string("     ") + toString(m_mapIndexes), 0);

        startTime = dae::GetTimeInSeconds();
    }

    if(m_mapJacobianExpressions.empty() && m_pEquation->m_bBuildJacobianExpressions)
    {
        daeDeclareException(exInvalidCall);
        e << "JacobianExpressions not built for the equation [" << GetName() << "]";
        throw e;
    }

    if(m_mapJacobianExpressions.empty())
    {
        //std::cout << "  Jacobian of the equation [" << GetName() << "]" << std::endl;
        //std::cout << "    ";

        // Achtung, Achtung!!
        // Evaluation trees must be calculated using the eCalculateJacobian mode
        EC.m_eEquationCalculationMode = eCalculateJacobian;

        // m_mapIndexes<OverallIndex, IndexInBlock>
        for(map<size_t, size_t>::iterator iter = m_mapIndexes.begin(); iter != m_mapIndexes.end(); iter++)
        {
            EC.m_nCurrentVariableIndexForJacobianEvaluation = iter->first;
            try
            {
                __ad = m_EquationEvaluationNode->Evaluate(&EC) * m_dScaling;

                //std::cout << toStringFormatted<real_t>(__ad.getDerivative(), -1, 10, true) << " [" << iter->second << "], ";
            }
            catch(std::exception& exc)
            {
                daeDeclareException(exInvalidCall);
                e << "Exception raised during calculation of the Jacobian of the equation [" << GetName() << "] for the EquationIndexInBlock=" << m_nEquationIndexInBlock
                  << " VariableIndexInBlock=" << iter->second << "; " << exc.what();
                throw e;
            }

            try
            {
                EC.m_pBlock->SetJacobian(m_nEquationIndexInBlock, iter->second, __ad.getDerivative());
            }
            catch(std::exception& exc)
            {
                daeDeclareException(exInvalidCall);
                e << "Cannot set Jacobian item for the equation [" << GetName() << "]: EquationIndexInBlock=" << m_nEquationIndexInBlock
                  << " VariableIndexInBlock=" << iter->second << "; " << exc.what();
                throw e;
            }
        }
        //std::cout << std::endl;
    }
    else
    {
        //std::cout << "  JacobianExpressions of the equation [" << GetName() << "]" << std::endl;
        //std::cout << "    ";

        // Achtung, Achtung!!
        // Jacobian expressions must be calculated using the eCalculate mode (not eCalculateJacobian)
        EC.m_eEquationCalculationMode = eCalculate;

        //map<overallIndex, pair<blockIndex, derivNode>>
        for(map< size_t, std::pair<size_t, adNodePtr> >::iterator iter = m_mapJacobianExpressions.begin(); iter != m_mapJacobianExpressions.end(); iter++)
        {
            EC.m_nCurrentVariableIndexForJacobianEvaluation = -1;
            try
            {
                __ad = iter->second.second->Evaluate(&EC) * m_dScaling;

                //std::cout << toStringFormatted<real_t>(__ad.getValue(), -1, 10, true) << " [" << iter->second.first << "], ";
            }
            catch(std::exception& exc)
            {
                daeDeclareException(exInvalidCall);
                e << "Exception raised during calculation of the Jacobian of the equation [" << GetName() << "] for the EquationIndexInBlock=" << m_nEquationIndexInBlock
                  << " VariableIndexInBlock=" << iter->second.first << "; " << exc.what();
                throw e;
            }

            try
            {
                EC.m_pBlock->SetJacobian(m_nEquationIndexInBlock, iter->second.first, __ad.getValue());
            }
            catch(std::exception& exc)
            {
                daeDeclareException(exInvalidCall);
                e << "Cannot set Jacobian item for the equation [" << GetName() << "]: EquationIndexInBlock=" << m_nEquationIndexInBlock
                  << " VariableIndexInBlock=" << iter->second.first << "; " << exc.what();
                throw e;
            }
        }
        //std::cout << std::endl;
        //std::cout << std::endl;
    }

    if(bPrintInfo)
    {
        endTime = dae::GetTimeInSeconds();
        m_pEquation->m_pModel->m_pDataProxy->LogMessage(string("  Evaluation time = ") + toStringFormatted(endTime - startTime, -1, 10) + string("s"), 0);
    }
}

void daeEquationExecutionInfo::SensitivityResiduals(daeExecutionContext& EC, const std::vector<size_t>& narrParameterIndexes)
{
#ifdef DAE_DEBUG
	if(!EC.m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
#endif

    bool bPrintInfo = m_pEquation->m_pModel->m_pDataProxy->PrintInfo();
    if(bPrintInfo)
        std::cout << "  SensitivityResidual for " << GetName() << std::endl;

	EC.m_pEquationExecutionInfo = this;

	adouble __ad;
	for(size_t i = 0; i < narrParameterIndexes.size(); i++)
	{
		EC.m_nCurrentParameterIndexForSensitivityEvaluation             = narrParameterIndexes[i];
		EC.m_nIndexInTheArrayOfCurrentParameterForSensitivityEvaluation = i;
		
        try
        {
            __ad = m_EquationEvaluationNode->Evaluate(&EC) * m_dScaling;
        }
        catch(std::exception& exc)
        {
            daeDeclareException(exInvalidCall);
            e << "Exception raised during calculation of the SensitivityResiduals of the equation [" << GetName() << "] for the EquationIndexInBlock=" << m_nEquationIndexInBlock
              << " ParameterIndex=" << i << "; " << exc.what();
            throw e;
        }

        try
        {
            EC.m_pDataProxy->SetSResValue(i, m_nEquationIndexInBlock, __ad.getDerivative());
        }
        catch(std::exception& exc)
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot set SensitivityResiduals item for the equation [" << GetName() << "]: EquationIndexInBlock=" << m_nEquationIndexInBlock
              << " ParameterIndex=" << i << "; " << exc.what();
            throw e;
        }
    }
}

// Should not be used anymore, but double-check
void daeEquationExecutionInfo::SensitivityParametersGradients(daeExecutionContext& EC, const std::vector<size_t>& narrParameterIndexes)
{
#ifdef DAE_DEBUG
	if(!EC.m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
#endif

    bool bPrintInfo = m_pEquation->m_pModel->m_pDataProxy->PrintInfo();
    if(bPrintInfo)
        std::cout << "  SensitivityParametersGradient for " << GetName() << std::endl;

	EC.m_pEquationExecutionInfo = this;

	adouble __ad;
	for(size_t i = 0; i < narrParameterIndexes.size(); i++)
	{
		EC.m_nCurrentParameterIndexForSensitivityEvaluation = narrParameterIndexes[i];
		
		__ad = m_EquationEvaluationNode->Evaluate(&EC) * m_dScaling;
		EC.m_pDataProxy->SetSResValue(i, m_nEquationIndexInBlock, __ad.getDerivative());
	}
}

// Operates on adRuntimeNodes
void daeEquationExecutionInfo::BuildJacobianExpressions()
{
    map<size_t, size_t>::iterator iter;

    //map<overallIndex, pair<blockIndex, derivNode>>
    m_mapJacobianExpressions.clear();

    bool bPrintInfo = m_pEquation->m_pModel->m_pDataProxy->PrintInfo();
    if(bPrintInfo)
        m_pEquation->m_pModel->m_pDataProxy->LogMessage(string("    Building Jacobian Expressions for equation ") + GetName() + string(" ..."), 0);

    // m_mapIndexes<OverallIndex, IndexInBlock>
    for(iter = m_mapIndexes.begin(); iter != m_mapIndexes.end(); iter++)
    {
        adJacobian jacob = adNode::Derivative(m_EquationEvaluationNode, iter->first);

        // Achtung!! Jacobian items are with no scaling!
        m_mapJacobianExpressions[iter->first] = std::pair<size_t, adNodePtr>(iter->second, jacob.derivative);
    }
}

void daeEquationExecutionInfo::AddVariableInEquation(size_t nIndex)
{
	pair<size_t, size_t> value(nIndex, 0);
	m_mapIndexes.insert(value);
}

void daeEquationExecutionInfo::GetVariableIndexes(std::vector<size_t>& narrVariableIndexes) const
{
	std::map<size_t, size_t>::const_iterator iter;
	for(iter = m_mapIndexes.begin(); iter != m_mapIndexes.end(); iter++)
		dae_push_back(narrVariableIndexes, iter->first);
}

size_t daeEquationExecutionInfo::GetEquationIndexInBlock(void) const
{
	return m_nEquationIndexInBlock;	
}

adNodePtr daeEquationExecutionInfo::GetEquationEvaluationNode(void) const
{
	return m_EquationEvaluationNode;	
}

/******************************************************************
	daeDistributedEquationDomainInfo
*******************************************************************/
daeDistributedEquationDomainInfo::daeDistributedEquationDomainInfo()
{
	m_nCurrentIndex = ULONG_MAX;
	m_pDomain       = NULL;
	m_pEquation     = NULL;
}

daeDistributedEquationDomainInfo::daeDistributedEquationDomainInfo(daeEquation* pEquation, daeDomain* pDomain, daeeDomainBounds eDomainBounds)
{
	if(!pEquation)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pDomain)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid domain in DEDI, in equation [ " << pEquation->GetCanonicalName() << "]";
		throw e;
	}
	if(eDomainBounds == eDBUnknown || eDomainBounds == eCustomBound)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid domain bounds in DEDI on the domain [ " << m_pDomain->GetCanonicalName() 
		  << "]; must be on of [eOpenOpen, eOpenClosed, eClosedOpen, eClosedClosed, eLowerBound, eUpperBound]";
		throw e;
	}

	m_nCurrentIndex = ULONG_MAX;
	m_pEquation     = pEquation;
	m_pDomain       = pDomain;
	m_pModel		= pDomain->m_pModel;
	m_eDomainBounds = eDomainBounds;
}

daeDistributedEquationDomainInfo::daeDistributedEquationDomainInfo(daeEquation* pEquation, daeDomain* pDomain, const vector<size_t>& narrDomainIndexes)
{
	if(!pEquation)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pDomain)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid domain in DEDI, in equation [ " << pEquation->GetCanonicalName() << "]";
		throw e;
	}
	if(narrDomainIndexes.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points cannot be 0 in DEDI, in equation [ " << pEquation->GetCanonicalName() << "]";
		throw e;
	}

	m_nCurrentIndex    = ULONG_MAX;
	m_pEquation        = pEquation;
	m_pDomain          = pDomain;
	m_pModel		   = pDomain->m_pModel;
	m_narrDomainPoints = narrDomainIndexes;
	m_eDomainBounds    = eCustomBound;
}

daeDistributedEquationDomainInfo::~daeDistributedEquationDomainInfo()
{
}

void daeDistributedEquationDomainInfo::Clone(const daeDistributedEquationDomainInfo& rObject)
{
	m_nCurrentIndex    = rObject.m_nCurrentIndex;
	m_pEquation        = rObject.m_pEquation;
	m_narrDomainPoints = rObject.m_narrDomainPoints;
	m_eDomainBounds    = rObject.m_eDomainBounds;
	
	m_pDomain   = FindDomain(rObject.m_pDomain, m_pModel);
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);
}

void daeDistributedEquationDomainInfo::Initialize(void)
{
	size_t i, iNoPoints;

	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);
	
	iNoPoints = m_pDomain->m_nNumberOfPoints;
	if(iNoPoints == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points in domain [ " << m_pDomain->GetCanonicalName() << "] is 0; it should be at least 1";
		throw e;
	}

	if(m_eDomainBounds != eCustomBound)
		m_narrDomainPoints.clear();
	
	if(m_eDomainBounds == eOpenOpen)
	{
		for(i = 1; i < iNoPoints-1; i++)
			m_narrDomainPoints.push_back(i);
	}
	else if(m_eDomainBounds == eOpenClosed)
	{
		for(i = 1; i < iNoPoints; i++)
			m_narrDomainPoints.push_back(i);
	}
	else if(m_eDomainBounds == eClosedOpen)
	{
		for(i = 0; i < iNoPoints-1; i++)
			m_narrDomainPoints.push_back(i);
	}
	else if(m_eDomainBounds == eClosedClosed)
	{
		for(i = 0; i < iNoPoints; i++)
			m_narrDomainPoints.push_back(i);
	}
	else if(m_eDomainBounds == eLowerBound)
	{
		m_narrDomainPoints.push_back(0);
	}
	else if(m_eDomainBounds == eUpperBound)
	{
		m_narrDomainPoints.push_back(iNoPoints-1);
	}
	else if(m_eDomainBounds == eCustomBound)
	{
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
}

daeEquation* daeDistributedEquationDomainInfo::GetEquation(void) const
{
	return m_pEquation;
}

size_t daeDistributedEquationDomainInfo::GetCurrentIndex(void) const
{
	return m_nCurrentIndex;
}

adouble daeDistributedEquationDomainInfo::operator()(void) const
{
// This function can be called only when in NodeEvaluation mode
// I cant call it if I am in FunctionEvaluation mode
	adouble tmp;
	daeDistributedEquationDomainInfo* pDEDI = const_cast<daeDistributedEquationDomainInfo*>(this);
	adSetupDomainIteratorNode* node = new adSetupDomainIteratorNode(pDEDI);
	tmp.node = adNodePtr(node);
	tmp.setGatherInfo(true);
	return tmp;
}

daeDomainIndex daeDistributedEquationDomainInfo::operator+(size_t increment) const
{
	daeDistributedEquationDomainInfo* pDEDI = const_cast<daeDistributedEquationDomainInfo*>(this);
	return daeDomainIndex(pDEDI, int(increment));
}

daeDomainIndex daeDistributedEquationDomainInfo::operator-(size_t increment) const
{
	daeDistributedEquationDomainInfo* pDEDI = const_cast<daeDistributedEquationDomainInfo*>(this);
	return daeDomainIndex(pDEDI, -int(increment));
}

void daeDistributedEquationDomainInfo::Open(io::xmlTag_t* pTag)
{
	string strName;

	daeObject::Open(pTag);

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeFindPortByID del(m_pModel);

	strName = "Domain";
	pTag->OpenObjectRef(strName, &del);

	strName = "Type";
	OpenEnum(pTag, strName, m_eDomainBounds);

	if(m_eDomainBounds == eCustomBound)
	{
		strName = "Points";
		pTag->OpenArray(strName, m_narrDomainPoints);
	}
}

void daeDistributedEquationDomainInfo::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "Domain";
	pTag->SaveObjectRef(strName, m_pDomain);
	
	strName = "Type";
	SaveEnum(pTag, strName, m_eDomainBounds);

	if(m_eDomainBounds == eCustomBound)
	{
		strName = "Points";
		pTag->SaveArray(strName, m_narrDomainPoints);
	}
}

void daeDistributedEquationDomainInfo::OpenRuntime(io::xmlTag_t* pTag)
{
	string strName;

	daeObject::OpenRuntime(pTag);

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeFindPortByID del(m_pModel);

	strName = "Domain";
	pTag->OpenObjectRef(strName, &del);

	strName = "Type";
	OpenEnum(pTag, strName, m_eDomainBounds);

	if(m_eDomainBounds == eCustomBound)
	{
		strName = "Points";
		pTag->OpenArray(strName, m_narrDomainPoints);
	}
}

void daeDistributedEquationDomainInfo::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::SaveRuntime(pTag);

	strName = "Domain";
	pTag->SaveObjectRef(strName, m_pDomain);

	strName = "Type";
	SaveEnum(pTag, strName, m_eDomainBounds);

	strName = "Points";
	pTag->SaveArray(strName, m_narrDomainPoints);
}

daeDomain_t* daeDistributedEquationDomainInfo::GetDomain(void) const
{
	return m_pDomain;
}
	
daeeDomainBounds daeDistributedEquationDomainInfo::GetDomainBounds(void) const
{
	return m_eDomainBounds;
}
	
void daeDistributedEquationDomainInfo::GetDomainPoints(vector<size_t>& narrDomainPoints) const
{
	narrDomainPoints = m_narrDomainPoints;
}

bool daeDistributedEquationDomainInfo::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;

// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check domain
	if(!m_pDomain)
	{
		strError = "Invalid domain in distributed equation domain info [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
// Check equation
	if(!m_pEquation)
	{
		strError = "Invalid equation in distributed equation domain info [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check number of points
	if(m_narrDomainPoints.empty())
	{
		strError = "Invalid number of points in distributed equation domain info [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check bounds
	if(m_eDomainBounds == eDBUnknown)
	{
		strError = "Invalid domain bounds in distributed equation domain info [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	else if(m_eDomainBounds == eOpenOpen)
	{
		if(m_narrDomainPoints.size() != m_pDomain->GetNumberOfPoints()-2)
		{
			strError = "Invalid number of points in distributed equation domain info [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
	}
	else if(m_eDomainBounds == eOpenClosed)
	{
		if(m_narrDomainPoints.size() != m_pDomain->GetNumberOfPoints()-1)
		{
			strError = "Invalid number of points in distributed equation domain info [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
	}
	else if(m_eDomainBounds == eClosedOpen)
	{
		if(m_narrDomainPoints.size() != m_pDomain->GetNumberOfPoints()-1)
		{
			strError = "Invalid number of points in distributed equation domain info [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
	}
	else if(m_eDomainBounds == eClosedClosed)
	{
		if(m_narrDomainPoints.size() != m_pDomain->GetNumberOfPoints())
		{
			strError = "Invalid number of points in distributed equation domain info [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
	}
	else if(m_eDomainBounds == eLowerBound)
	{
	}
	else if(m_eDomainBounds == eUpperBound)
	{
	}
	else if(m_eDomainBounds == eFunctor)
	{
		daeDeclareAndThrowException(exNotImplemented)
	}
	else if(m_eDomainBounds == eCustomBound)
	{
		if(m_narrDomainPoints.empty())
		{
			strError = "List of indexes cannot be empty in distributed equation domain info [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		else
		{
			size_t nPoints = m_pDomain->GetNumberOfPoints();
			std::vector<size_t>::const_iterator it;
			std::vector<size_t>::const_iterator itbegin = m_narrDomainPoints.begin();
			std::vector<size_t>::const_iterator itend   = m_narrDomainPoints.end();

			for(it = itbegin; it != itend; it++)
			{
				if(*it >= nPoints)
				{
					strError = "Cannot find index [" + toString(*it) + "] in distributed equation domain info [" + GetCanonicalName() + "]";
					strarrErrors.push_back(strError);
					bCheck = false;
				}
	
				if(std::count(itbegin, itend, *it) > 1)
				{
					strError = "Dupplicate index [" + toString(*it) + "] found in distributed equation domain info [" + GetCanonicalName() + "]";
					strarrErrors.push_back(strError);
					bCheck = false;
				}
			}
		}
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
	
	return bCheck;
}

/******************************************************************
	daeEquation
*******************************************************************/
daeEquation::daeEquation()
{
    daeConfig& cfg = daeConfig::GetConfig();
    
    m_bCheckUnitsConsistency    = cfg.Get<bool>("daetools.core.checkUnitsConsistency", true);
    m_bBuildJacobianExpressions = false;
    m_dScaling                  = 1.0;
    m_pParentState              = NULL;
    m_pModel                    = NULL;
}

daeEquation::~daeEquation()
{
}

void daeEquation::Clone(const daeEquation& rObject)
{
//	m_pResidualNode	= rObject.m_pResidualNode;
	
//	for(size_t i = 0; i < rObject.m_ptrarrDistributedEquationDomainInfos.size(); i++)
//	{
//		daeDistributedEquationDomainInfo* pDEDI = new daeDistributedEquationDomainInfo();
//		pDEDI->SetName(rObject.m_ptrarrDistributedEquationDomainInfos[i]->m_strShortName);
//		SetModelAndCanonicalName(pDEDI);
//		dae_push_back(m_ptrarrDistributedEquationDomainInfos, pDEDI);
//		pDEDI->m_pModel    = m_pModel;
//		pDEDI->m_pEquation = this;
//		pDEDI->Clone(*rObject.m_ptrarrDistributedEquationDomainInfos[i]);
//	}
}

void daeEquation::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	m_ptrarrDistributedEquationDomainInfos.EmptyAndFreeMemory();
	m_ptrarrDistributedEquationDomainInfos.SetOwnershipOnPointers(true);
	m_pResidualNode.reset();

	daeObject::Open(pTag);

    strName = "Expression";
    adNode* node = adNode::OpenNode(pTag, strName);
    m_pResidualNode.reset(node);

	strName = "DistributedDomainInfos";
	daeSetModelAndCanonicalNameDelegate<daeDistributedEquationDomainInfo> del(this, m_pModel);
	pTag->OpenObjectArray(strName, m_ptrarrDistributedEquationDomainInfos, &del);
}

void daeEquation::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "EquationType";
	SaveEnum(pTag, strName, GetEquationType());

    strName = "Expression";
    adNode::SaveNode(pTag, strName, m_pResidualNode.get());

    strName = "MathML";
    SaveNodeAsMathML(pTag, strName);

	strName = "DistributedDomainInfos";
	pTag->SaveObjectArray(strName, m_ptrarrDistributedEquationDomainInfos);
}

void daeEquation::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	daeDEDI* pDEDI;
	string strExport, strResidual, strBounds;
	boost::format fmtFile;

	if(c.m_bExportDefinition)
	{
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "eq = self.CreateEquation(\"%1%\", \"%2%\")\n";
			if(!m_ptrarrDistributedEquationDomainInfos.empty())
			{
				for(size_t i = 0; i < m_ptrarrDistributedEquationDomainInfos.size(); i++)
				{
					pDEDI = m_ptrarrDistributedEquationDomainInfos[i];
					if(pDEDI->m_eDomainBounds == eCustomBound)
					{
						strBounds = "[" + toString(pDEDI->m_narrDomainPoints) + "]";
					}
					else
					{
						strBounds = g_EnumTypesCollection->esmap_daeeDomainBounds.GetString(pDEDI->m_eDomainBounds);
					}
					
					strExport += c.CalculateIndent(c.m_nPythonIndentLevel) + 
								 pDEDI->GetStrippedName() + 
								 " = eq.DistributeOnDomain(self." + pDEDI->m_pDomain->GetStrippedName() +
								 ", " + strBounds + ")\n";
				}
			}
			strExport += c.CalculateIndent(c.m_nPythonIndentLevel) + "eq.Residual = %3%\n\n";

			m_pResidualNode->Export(strResidual, eLanguage, c);

			fmtFile.parse(strExport);
			fmtFile % m_strShortName 
					% m_strDescription
					% strResidual;
		}
		else if(eLanguage == eCDAE)
		{
			strExport  = c.CalculateIndent(c.m_nPythonIndentLevel) + "{\n";
			c.m_nPythonIndentLevel++;
			strExport += c.CalculateIndent(c.m_nPythonIndentLevel) + "eq = CreateEquation(\"%1%\", \"%2%\");\n";
			if(!m_ptrarrDistributedEquationDomainInfos.empty())
			{
				for(size_t i = 0; i < m_ptrarrDistributedEquationDomainInfos.size(); i++)
				{
					pDEDI = m_ptrarrDistributedEquationDomainInfos[i];
					if(pDEDI->m_eDomainBounds == eCustomBound)
					{
						strExport += c.CalculateIndent(c.m_nPythonIndentLevel) + 
									 "const size_t domainIndexes[" + toString(pDEDI->m_narrDomainPoints.size()) + "] = {" + 
									 toString(pDEDI->m_narrDomainPoints) + "};\n";
						strBounds = "domainIndexes, " + toString(pDEDI->m_narrDomainPoints.size());
					}
					else
					{
						strBounds = g_EnumTypesCollection->esmap_daeeDomainBounds.GetString(pDEDI->m_eDomainBounds);
					}
					
					strExport += c.CalculateIndent(c.m_nPythonIndentLevel) + "daeDEDI* " + 
								 pDEDI->GetStrippedName() + 
								 " = eq->DistributeOnDomain(" + pDEDI->m_pDomain->GetStrippedName() +
								 ", " + strBounds + ");\n";
				}
			}
			strExport += c.CalculateIndent(c.m_nPythonIndentLevel) + "eq->SetResidual(%3%);\n";
			c.m_nPythonIndentLevel--;
			strExport += c.CalculateIndent(c.m_nPythonIndentLevel) + "}\n\n";

			m_pResidualNode->Export(strResidual, eLanguage, c);

			fmtFile.parse(strExport);
			fmtFile % m_strShortName 
					% m_strDescription
					% strResidual;
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += fmtFile.str();
}

void daeEquation::OpenRuntime(io::xmlTag_t* pTag)
{
//	string strName;

//	if(!m_pModel)
//		daeDeclareAndThrowException(exInvalidPointer);

//	m_ptrarrDistributedEquationDomainInfos.EmptyAndFreeMemory();
//	m_ptrarrDistributedEquationDomainInfos.SetOwnershipOnPointers(true);
//	m_pResidualNode.reset();

//	daeObject::OpenRuntime(pTag);

//	strName = "EquationDefinitionMode";
//	OpenEnum(pTag, strName, m_eEquationType);

//	strName = "EquationEvaluationMode";
//	OpenEnum(pTag, strName, m_eEquationEvaluationMode);

//	if(m_eEquationType == eResidualNode)
//	{
//		strName = "MathML";
//		adNode* node = adNode::OpenNode(pTag, strName);
//		m_pResidualNode.reset(node);
//	}

//	strName = "DistributedDomainInfos";
//	daeSetModelAndCanonicalNameDelegate<daeDistributedEquationDomainInfo> del(this, m_pModel);
//	OpenObjectArrayRuntime(pTag, strName, m_ptrarrDistributedEquationDomainInfos, &del);
}

void daeEquation::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::SaveRuntime(pTag);

//	strName = "EquationDefinitionMode";
//	SaveEnum(pTag, strName, m_eEquationType);

//	strName = "EquationEvaluationMode";
//	SaveEnum(pTag, strName, m_eEquationEvaluationMode);

	strName = "MathML";
	SaveNodeAsMathML(pTag, strName);

//	strName = "DistributedDomainInfos";
//	pTag->SaveRuntimeObjectArray(strName, m_ptrarrDistributedEquationDomainInfos);
			
	strName = "EquationExecutionInfos";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrEquationExecutionInfos);
}

void daeEquation::SaveNodeAsMathML(io::xmlTag_t* pTag, const string& strObjectName) const
{
	string strName, strValue, strDEDI, strDomain;
	daeNodeSaveAsContext c(m_pModel); //, m_ptrarrDistributedEquationDomainInfos);
	adNode* node = m_pResidualNode.get();

	io::xmlTag_t* pChildTag = pTag->AddTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	strName = "math";
	io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);
	if(!pMathMLTag)
		daeDeclareAndThrowException(exXMLIOError);

	strName = "xmlns";
	strValue = "http://www.w3.org/1998/Math/MathML";
	pMathMLTag->AddAttribute(strName, strValue);

	strName = "mrow";
	io::xmlTag_t* mrow = pMathMLTag->AddTag(strName);
	if(!mrow)
		daeDeclareAndThrowException(exXMLIOError);

	node->SaveAsPresentationMathML(mrow, &c);

	strName  = "mo";
	strValue = "=";
	mrow->AddTag(strName, strValue);

	strName  = "mo";
	strValue = "0";
	mrow->AddTag(strName, strValue);

	if(m_ptrarrDistributedEquationDomainInfos.size() > 0)
	{
		string strLeftBracket, strRightBracket, strContent;

		strName  = "mo";
		strValue = ";";
		mrow->AddTag(strName, strValue);

		strName  = "mrow";
		strValue = "";
		io::xmlTag_t* mrow1 = mrow->AddTag(strName, strValue);

		for(size_t i = 0; i < m_ptrarrDistributedEquationDomainInfos.size(); i++)
		{
			daeDEDI* pDedi = m_ptrarrDistributedEquationDomainInfos[i];
			if(!pDedi)
				daeDeclareAndThrowException(exXMLIOError);

			if(i != 0)
			{
				strName  = "mo";
				strValue = ",";
				mrow1->AddTag(strName, strValue);
			}

            strDEDI   = pDedi->GetName();
            strDomain = pDedi->GetDomain()->GetName();
			
            if(pDedi->m_eDomainBounds == eLowerBound ||
			   pDedi->m_eDomainBounds == eUpperBound)
			{
				strName  = "mi";
				xml::xmlPresentationCreator::WrapIdentifier(mrow1, strDEDI);
	
				strName  = "mo";
				strValue = "=";
				mrow1->AddTag(strName, strValue);
	
				if(pDedi->m_eDomainBounds == eLowerBound)
				{
					strName  = "msub";
					strValue = "";
					io::xmlTag_t* msub1 = mrow1->AddTag(strName, strValue);
	
					strName  = "mi";
					xml::xmlPresentationCreator::WrapIdentifier(msub1, strDomain);
	
					strName  = "mi";
					strValue = "0"; 
					msub1->AddTag(strName, strValue);
				}
				else
				{
					strName  = "msub";
					strValue = "";
					io::xmlTag_t* msub1 = mrow1->AddTag(strName, strValue);
	
					strName  = "mi";
					xml::xmlPresentationCreator::WrapIdentifier(msub1, strDomain);
	
					strName  = "mi";
					strValue = "n";
					msub1->AddTag(strName, strValue);
				}
			}
			else
			{
				strName  = "mi";
				strValue = "&ForAll;";
				mrow1->AddTag(strName, strValue);

				strName  = "mi";
				xml::xmlPresentationCreator::WrapIdentifier(mrow1, strDEDI);
	
				strName  = "mo";
				strValue = "&isin;";
				mrow1->AddTag(strName, strValue);
	
				if(pDedi->m_eDomainBounds == eOpenOpen)
				{
					strLeftBracket  = "(";
					strRightBracket = ")";				
				}
				else if(pDedi->m_eDomainBounds == eOpenClosed)
				{
					strLeftBracket  = "(";
					strRightBracket = "]";
				}
				else if(pDedi->m_eDomainBounds == eClosedOpen)
				{
					strLeftBracket  = "[";
					strRightBracket = ")";
				}
				else if(pDedi->m_eDomainBounds == eClosedClosed)
				{
					strLeftBracket  = "[";
					strRightBracket = "]";
				}
				else if(pDedi->m_eDomainBounds == eFunctor)
				{
					daeDeclareAndThrowException(exXMLIOError);
				}
				else if(pDedi->m_eDomainBounds == eCustomBound)
				{
					strLeftBracket  = "{";
					strRightBracket = "}";
				}
				else
				{
					daeDeclareAndThrowException(exXMLIOError);
				}
				
			// Add left bracket
				strName  = "mo";
				strValue = strLeftBracket;
				mrow1->AddTag(strName, strValue);
				
			// Add points
				if(pDedi->m_eDomainBounds == eOpenOpen     ||
				   pDedi->m_eDomainBounds == eOpenClosed   ||
				   pDedi->m_eDomainBounds == eClosedOpen   ||
				   pDedi->m_eDomainBounds == eClosedClosed )
				{
					strName  = "msub";
					strValue = "";
					io::xmlTag_t* msub1 = mrow1->AddTag(strName, strValue);
	
					strName  = "mi";
					xml::xmlPresentationCreator::WrapIdentifier(msub1, strDomain);
	
					strName  = "mi";
					strValue = "0";
					msub1->AddTag(strName, strValue);
	
					strName  = "mi";
					strValue = ",";
					mrow1->AddTag(strName, strValue);
	
					strName  = "msub";
					strValue = "";
					io::xmlTag_t* msub2 = mrow1->AddTag(strName, strValue);
	
					strName  = "mi";
					xml::xmlPresentationCreator::WrapIdentifier(msub2, strDomain);
	
					strName  = "mi";
					strValue = "n";
					msub2->AddTag(strName, strValue);
				}
				else if(pDedi->m_eDomainBounds == eCustomBound)
				{
					vector<size_t> narrPoints;
	
					pDedi->GetDomainPoints(narrPoints);
					for(size_t k = 0; k < narrPoints.size(); k++)
					{
						if(k != 0)
						{
							strName  = "mo";
							strValue = ",";
							mrow1->AddTag(strName, strValue);
						}
						strName  = "mn";
						strValue = toString<size_t>(narrPoints[k]);
						mrow1->AddTag(strName, strValue);
					}
				}
				else if(pDedi->m_eDomainBounds == eFunctor)
				{
					daeDeclareAndThrowException(exXMLIOError);
				}
				else
				{
					daeDeclareAndThrowException(exXMLIOError);
				}
				
			// Add right bracket
				strName  = "mo";
				strValue = strRightBracket;
				mrow1->AddTag(strName, strValue);
			}
		}
	}
}

string daeEquation::GetCanonicalName(void) const
{
	if(m_pParentState)
		return m_pParentState->GetCanonicalName() + '.' + m_strShortName;
	else
		return daeObject::GetCanonicalName();
}

size_t daeEquation::GetNumberOfEquations(void) const
{
	daeDEDI* pDedi;

	size_t nTotalNumberOfEquations = 1;
	for(size_t i = 0; i < m_ptrarrDistributedEquationDomainInfos.size(); i++)
	{
		pDedi = m_ptrarrDistributedEquationDomainInfos[i];
		if(!pDedi)
			daeDeclareAndThrowException(exInvalidPointer); 
		if(pDedi->m_narrDomainPoints.empty())
			daeDeclareAndThrowException(exInvalidCall); 

		nTotalNumberOfEquations *= pDedi->m_narrDomainPoints.size();
	}
	return nTotalNumberOfEquations;
}

void daeEquation::SetResidualValue(size_t nEquationIndex, real_t dResidual, daeBlock* pBlock)
{
	if(!pBlock)
		daeDeclareAndThrowException(exInvalidPointer);
	pBlock->SetResidual(nEquationIndex, dResidual);
}

daeeEquationType daeEquation::GetEquationType(void) const
{
    return DetectEquationType(m_pResidualNode);
}

void daeEquation::SetJacobianItem(size_t nEquationIndex, size_t nVariableIndex, real_t dJacobValue, daeBlock* pBlock)
{
	if(!pBlock)
		daeDeclareAndThrowException(exInvalidPointer);
	pBlock->SetJacobian(nEquationIndex, nVariableIndex, dJacobValue);
}

void daeEquation::InitializeDEDIs(void)
{
// First initialize all DEDIs (initialize points in them)
	daeDistributedEquationDomainInfo* pDEDI;
	for(size_t i = 0; i < m_ptrarrDistributedEquationDomainInfos.size(); i++)
	{
		pDEDI = m_ptrarrDistributedEquationDomainInfos[i];
		if(!pDEDI)
			daeDeclareAndThrowException(exInvalidPointer);
		pDEDI->Initialize();
	}
}

void daeEquation::GatherInfo(const vector<size_t>& narrDomainIndexes, const daeExecutionContext& EC, adNodePtr& node)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nNumberOfDomains = m_ptrarrDistributedEquationDomainInfos.size();
	if(nNumberOfDomains != narrDomainIndexes.size())
	{	
		daeDeclareException(exInvalidCall);
		e << "Illegal number of domains in equation [ " << GetCanonicalName() << "]";
		throw e;
	}
    
    for(size_t i = 0; i < nNumberOfDomains; i++)
        m_ptrarrDistributedEquationDomainInfos[i]->m_nCurrentIndex = narrDomainIndexes[i];

    node = m_pResidualNode->Evaluate(&EC).node;
}

daeDEDI* daeEquation::DistributeOnDomain(daeDomain& rDomain, daeeDomainBounds eDomainBounds, const string& strName)
{
	daeDistributedEquationDomainInfo* pDEDI = new daeDistributedEquationDomainInfo(this, &rDomain, eDomainBounds);
    
    if(strName.empty())
    	pDEDI->SetName(rDomain.GetName());
    else
        pDEDI->SetName(strName);
    
    pDEDI->SetModel(m_pModel);
	//SetModelAndCanonicalName(pDEDI);
	dae_push_back(m_ptrarrDistributedEquationDomainInfos, pDEDI);
	return pDEDI;
}

daeDEDI* daeEquation::DistributeOnDomain(daeDomain& rDomain, const vector<size_t>& narrDomainIndexes, const string& strName)
{
	daeDistributedEquationDomainInfo* pDEDI = new daeDistributedEquationDomainInfo(this, &rDomain, narrDomainIndexes);
    
    if(strName.empty())
    	pDEDI->SetName(rDomain.GetName());
    else
        pDEDI->SetName(strName);
    
    pDEDI->SetModel(m_pModel);
	//SetModelAndCanonicalName(pDEDI);
	dae_push_back(m_ptrarrDistributedEquationDomainInfos, pDEDI);
	return pDEDI;
}

daeDEDI* daeEquation::DistributeOnDomain(daeDomain& rDomain, const size_t* pnarrDomainIndexes, size_t n, const string& strName)
{
	vector<size_t> narrDomainIndexes;
	
	narrDomainIndexes.resize(n);
	for(size_t i = 0; i < n; i++)
		narrDomainIndexes[i] = pnarrDomainIndexes[i];

	return DistributeOnDomain(rDomain, narrDomainIndexes, strName);	
}

void daeEquation::GetDomainDefinitions(vector<daeDistributedEquationDomainInfo_t*>& arrDistributedEquationDomainInfo)
{
	arrDistributedEquationDomainInfo.clear();
	dae_set_vector(m_ptrarrDistributedEquationDomainInfos, arrDistributedEquationDomainInfo);
}

void daeEquation::GetEquationExecutionInfos(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfos) const
{
	ptrarrEquationExecutionInfos.clear();
	dae_set_vector(m_ptrarrEquationExecutionInfos, ptrarrEquationExecutionInfos);
}

daeState* daeEquation::GetParentState() const
{
    return m_pParentState;
}

std::vector<daeDEDI*> daeEquation::GetDEDIs() const
{
    return m_ptrarrDistributedEquationDomainInfos;
}

void daeEquation::SetResidual(adouble res)
{
	if(!res.node)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid residual, equation [ " << GetCanonicalName() << "]";
		throw e;
	}

	m_pResidualNode = res.node;
}

adouble daeEquation::GetResidual(void) const
{
	adouble ad;
    ad.setGatherInfo(true);
	ad.node = m_pResidualNode;
	return ad;
}

real_t daeEquation::GetScaling(void) const
{
	return m_dScaling;
}

void daeEquation::SetScaling(real_t dScaling)
{
	m_dScaling = dScaling;
}

bool daeEquation::GetBuildJacobianExpressions(void) const
{
    return m_bBuildJacobianExpressions;
}

void daeEquation::SetBuildJacobianExpressions(bool bBuildJacobianExpressions)
{
    m_bBuildJacobianExpressions = bBuildJacobianExpressions;
}

bool daeEquation::GetCheckUnitsConsistency(void) const
{
	return m_bCheckUnitsConsistency;
}

void daeEquation::SetCheckUnitsConsistency(bool bCheck)
{
	m_bCheckUnitsConsistency = bCheck;
}

bool daeEquation::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;
	
	dae_capacity_check(m_ptrarrDistributedEquationDomainInfos);
	dae_capacity_check(m_ptrarrEquationExecutionInfos);

// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check residual node	
	if(!m_pResidualNode)
	{
		strError = "Invalid residual in equation [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
// Check unit-consistency of the equation	
	if(m_bCheckUnitsConsistency)
	{
		try
		{
			quantity q = m_pResidualNode->GetQuantity();
			//std::cout << (boost::format("Equation [%s] residual units %s") % GetCanonicalName() % q.getUnits().toString()).str() << std::endl;
		}
		catch(units_error& e)
		{
			strError = "Unit-consistency check failed in equation [" + GetCanonicalName() + "]:";
			strarrErrors.push_back(strError);
			strError = "  " + string(e.what());
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		catch(std::exception& e)
		{
			strError = "Exception occurred while unit-consistency check in equation [" + GetCanonicalName() + "]:";
			strarrErrors.push_back(strError);
			strError = "  " + string(e.what());
			strarrErrors.push_back(strError);
			bCheck = false;
		}
	}

// Check distributed equation domain infos	
	if(m_ptrarrDistributedEquationDomainInfos.size() != 0)
	{
		daeDistributedEquationDomainInfo* pDEDI;	
		for(size_t i = 0; i < m_ptrarrDistributedEquationDomainInfos.size(); i++)
		{
			pDEDI = m_ptrarrDistributedEquationDomainInfos[i];
			if(!pDEDI)
			{
				strError = "Invalid distributed equation domain info in equation [" + GetCanonicalName() + "]";
				strarrErrors.push_back(strError);
				bCheck = false;
				continue;
			}
			
			if(!pDEDI->CheckObject(strarrErrors))
				bCheck = false;
		}
	}

	return bCheck;
}

/*********************************************************************************************
	daePortEqualityEquation
**********************************************************************************************/
daePortEqualityEquation::daePortEqualityEquation(void)
{
}

daePortEqualityEquation::~daePortEqualityEquation(void)
{
}

void daePortEqualityEquation::Initialize(daeVariable* pLeft, daeVariable* pRight)
{
	if(!pLeft)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!pRight)
		daeDeclareAndThrowException(exInvalidPointer); 

	m_pLeft  = pLeft;
	m_pRight = pRight;
}

void daePortEqualityEquation::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeEquation::Open(pTag);

	daeFindVariableByID del(m_pModel);

	strName = "LeftVariable";
	m_pLeft = pTag->OpenObjectRef(strName, &del);
	if(!m_pLeft)
		daeDeclareAndThrowException(exXMLIOError); 

	strName = "RightVariable";
	m_pRight = pTag->OpenObjectRef(strName, &del);
	if(!m_pRight)
		daeDeclareAndThrowException(exXMLIOError); 
}

void daePortEqualityEquation::Save(io::xmlTag_t* pTag) const
{
	string strName;

	if(!m_pLeft)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pRight)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeEquation::Save(pTag);

	strName = "LeftVariable";
	pTag->SaveObjectRef(strName, m_pLeft);

	strName = "RightVariable";
	pTag->SaveObjectRef(strName, m_pRight);
}

void daePortEqualityEquation::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
}

void daePortEqualityEquation::OpenRuntime(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeEquation::OpenRuntime(pTag);

	daeFindVariableByID del(m_pModel);

	strName = "LeftVariable";
	m_pLeft = pTag->OpenObjectRef(strName, &del);
	if(!m_pLeft)
		daeDeclareAndThrowException(exXMLIOError); 

	strName = "RightVariable";
	m_pRight = pTag->OpenObjectRef(strName, &del);
	if(!m_pRight)
		daeDeclareAndThrowException(exXMLIOError); 
}

void daePortEqualityEquation::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	if(!m_pLeft)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pRight)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeEquation::SaveRuntime(pTag);

	strName = "LeftVariable";
	pTag->SaveObjectRef(strName, m_pLeft);

	strName = "RightVariable";
	pTag->SaveObjectRef(strName, m_pRight);
}

bool daePortEqualityEquation::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;
	
	dae_capacity_check(m_ptrarrDistributedEquationDomainInfos);
	dae_capacity_check(m_ptrarrEquationExecutionInfos);

	if(!daeEquation::CheckObject(strarrErrors))
		bCheck = false;

	if(!m_pLeft)
	{
		strError = "Invalid left side port in port connection equation [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
	if(!m_pRight)
	{
		strError = "Invalid right side port in port connection equation [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

	return bCheck;
}

size_t daePortEqualityEquation::GetNumberOfEquations(void) const
{
	return daeEquation::GetNumberOfEquations();
}

}
}
