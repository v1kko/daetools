#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "xmlfunctions.h"

namespace dae 
{
namespace core 
{
/******************************************************************
	daeEquationExecutionInfo
*******************************************************************/
daeEquationExecutionInfo::daeEquationExecutionInfo()
{
	m_pEquation = NULL;
	m_pModel	= NULL;
	m_pBlock	= NULL;
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

	strName = "Model";
	m_pModel = pTag->OpenObjectRef<daeModel>(strName, &modeldel);

	strName = "Equation";
	m_pEquation = pTag->OpenObjectRef<daeEquation>(strName, &eqndel);

	strName = "EquationIndexInBlock";
	pTag->Open(strName, m_nEquationIndexInBlock);

	strName = "DomainIndexes";
	pTag->OpenArray(strName, m_narrDomainIndexes);

	strName = "mapIndexes";
	pTag->OpenMap(strName, m_mapIndexes);

	strName = "m_ptrarrDomains";
	pTag->OpenObjectRefArray(strName, m_ptrarrDomains, &domaindel);

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

	strName = "Model";
	pTag->SaveObjectRef(strName, m_pModel);

	strName = "Equation";
	pTag->SaveObjectRef(strName, m_pEquation);

	strName = "EquationIndexInBlock";
	pTag->Save(strName, m_nEquationIndexInBlock);

	strName = "DomainIndexes";
	pTag->SaveArray(strName, m_narrDomainIndexes);

	strName = "mapIndexes";
	pTag->SaveMap(strName, m_mapIndexes);

	strName = "m_ptrarrDomains";
	pTag->SaveObjectRefArray(strName, m_ptrarrDomains);

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

	daeSaveAsMathMLContext c(m_pModel);

	strName = "EquationEvaluationNode";
	adNode::SaveNodeAsMathML(pTag, string("MathML"), m_EquationEvaluationNode.get(), &c, true);
	
	strName = "IsLinear";
	pTag->Save(strName, m_EquationEvaluationNode->IsLinear());
}

void daeEquationExecutionInfo::GatherInfo(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pEquation)
		daeDeclareAndThrowException(exInvalidPointer);

	daeExecutionContext EC;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_pEquationExecutionInfo		= this;
	EC.m_eEquationCalculationMode	= eGatherInfo;

	m_pModel->PropagateGlobalExecutionContext(&EC);
		m_pEquation->GatherInfo(m_narrDomainIndexes, EC, m_EquationEvaluationNode);
	m_pModel->PropagateGlobalExecutionContext(NULL);

	if(m_pEquation->m_eEquationEvaluationMode == eCommandStackEvaluation)
		daeFPU::CreateCommandStack(m_EquationEvaluationNode.get(), m_ptrarrEquationCommands);
}

void daeEquationExecutionInfo::Residual(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pEquation)
		daeDeclareAndThrowException(exInvalidPointer);

	daeExecutionContext EC;
	EC.m_pBlock						= m_pBlock;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_dInverseTimeStep			= m_pBlock->GetInverseTimeStep();
	EC.m_pEquationExecutionInfo		= this;
	EC.m_eEquationCalculationMode	= eCalculate;

	if(m_pEquation->m_eEquationEvaluationMode == eResidualNodeEvaluation)
	{
		adouble __ad = m_EquationEvaluationNode->Evaluate(&EC);
		m_pBlock->SetResidual(m_nEquationIndexInBlock, __ad.getValue());
		
		/* Old code:
		m_pEquation->SetResidual(m_nEquationIndexInBlock, __ad.getValue(), m_pBlock);

		real_t res;
		daeFPU::fpuResidual(&EC, m_ptrarrEquationCommands, res);
		if(res != __ad.getValue())
			daeDeclareAndThrowException(exInvalidCall);
		*/
	}
	else if(m_pEquation->m_eEquationEvaluationMode == eFunctionEvaluation)
	{
		daeDeclareAndThrowException(exInvalidCall)
				
		/* Old code 
		I think I dont need this evaluation mode anymore
		What about foreign objects??
		
		daeExecutionContext* pEC;
		map<size_t, size_t>::iterator iter;
		for(iter = m_mapIndexes.begin(); iter != m_mapIndexes.end(); iter++)
		{
			pEC = m_pModel->m_pDataProxy->GetExecutionContext((*iter).first);
			pEC->m_pBlock						= m_pBlock;
			pEC->m_dInverseTimeStep				= m_pBlock->GetInverseTimeStep();
			pEC->m_pEquationExecutionInfo		= this;
			pEC->m_eEquationCalculationMode		= eCalculate;
			pEC->m_nCurrentVariableIndexForJacobianEvaluation = ULONG_MAX;
		}
		m_pEquation->Residual(m_narrDomainIndexes, EC);
		*/
	}
	else if(m_pEquation->m_eEquationEvaluationMode == eCommandStackEvaluation)
	{
		daeDeclareAndThrowException(exInvalidCall)

		//real_t res;
		//daeFPU::fpuResidual(&EC, m_ptrarrEquationCommands, res);
		//m_pEquation->SetResidual(m_nEquationIndexInBlock, res, m_pBlock);
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
}

void daeEquationExecutionInfo::Jacobian(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pEquation)
		daeDeclareAndThrowException(exInvalidPointer);

	daeExecutionContext EC;
	EC.m_pBlock						= m_pBlock;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_dInverseTimeStep			= m_pBlock->GetInverseTimeStep();
	EC.m_pEquationExecutionInfo		= this;
	EC.m_eEquationCalculationMode	= eCalculateJacobian;

	if(m_pEquation->m_eEquationEvaluationMode == eResidualNodeEvaluation)
	{
		adouble __ad;
		map<size_t, size_t>::iterator iter;
		size_t nVariableindexInBlock;

		for(iter = m_mapIndexes.begin(); iter != m_mapIndexes.end(); iter++)
		{
			EC.m_nCurrentVariableIndexForJacobianEvaluation = (*iter).first;
			nVariableindexInBlock = (*iter).second;
			__ad = m_EquationEvaluationNode->Evaluate(&EC);
			m_pBlock->SetJacobian(m_nEquationIndexInBlock, nVariableindexInBlock, __ad.getDerivative());
			
			//      This was before:
			//m_pEquation->SetJacobianItem(m_nEquationIndexInBlock, nVariableindexInBlock, __ad.getDerivative(), m_pBlock);

			//real_t jacob;
			//daeFPU::fpuJacobian(&EC, m_ptrarrEquationCommands, jacob);
			//if(jacob != __ad.getDerivative())
			//	daeDeclareAndThrowException(exInvalidCall);
		}
	}
	else if(m_pEquation->m_eEquationEvaluationMode == eFunctionEvaluation)
	{
		daeDeclareAndThrowException(exInvalidCall)
		
		//m_pEquation->Jacobian(m_narrDomainIndexes, m_mapIndexes, EC);
	}
	else if(m_pEquation->m_eEquationEvaluationMode == eCommandStackEvaluation)
	{
		daeDeclareAndThrowException(exInvalidCall)

		//real_t jacob;
		//map<size_t, size_t>::iterator iter;
		//size_t nVariableindexInBlock;

		//for(iter = m_mapIndexes.begin(); iter != m_mapIndexes.end(); iter++)
		//{
		//	EC.m_nCurrentVariableIndexForJacobianEvaluation = (*iter).first;
		//	nVariableindexInBlock = (*iter).second;
		//	daeFPU::fpuJacobian(&EC, m_ptrarrEquationCommands, jacob);
		//	m_pBlock->SetJacobian(m_nEquationIndexInBlock, nVariableindexInBlock, jacob);
		//}
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
}

void daeEquationExecutionInfo::SensitivityResiduals(const std::vector<size_t>& narrParameterIndexes)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pEquation)
		daeDeclareAndThrowException(exInvalidPointer);

	daeExecutionContext EC;
	EC.m_pBlock						= m_pBlock;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_dInverseTimeStep			= m_pBlock->GetInverseTimeStep();
	EC.m_pEquationExecutionInfo		= this;
	EC.m_eEquationCalculationMode	= eCalculateSensitivityResiduals;

	if(m_pEquation->m_eEquationEvaluationMode == eResidualNodeEvaluation)
	{
		adouble __ad;
		for(size_t i = 0; i < narrParameterIndexes.size(); i++)
		{
			EC.m_nCurrentParameterIndexForSensitivityEvaluation = narrParameterIndexes[i];
			//Used to get the S/SD values 
			EC.m_nIndexInTheArrayOfCurrentParameterForSensitivityEvaluation = i;
			__ad = m_EquationEvaluationNode->Evaluate(&EC);
			m_pModel->m_pDataProxy->SetSResValue(i, m_nEquationIndexInBlock, __ad.getDerivative());
		}
	}
	else if(m_pEquation->m_eEquationEvaluationMode == eFunctionEvaluation)
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
	else if(m_pEquation->m_eEquationEvaluationMode == eCommandStackEvaluation)
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
}

void daeEquationExecutionInfo::SensitivityParametersGradients(const std::vector<size_t>& narrParameterIndexes)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pEquation)
		daeDeclareAndThrowException(exInvalidPointer);

	daeExecutionContext EC;
	EC.m_pBlock						= m_pBlock;
	EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
	EC.m_dInverseTimeStep			= m_pBlock->GetInverseTimeStep();
	EC.m_pEquationExecutionInfo		= this;
	EC.m_eEquationCalculationMode	= eCalculateSensitivityParametersGradients;

	if(m_pEquation->m_eEquationEvaluationMode == eResidualNodeEvaluation)
	{
		adouble __ad;
		for(size_t i = 0; i < narrParameterIndexes.size(); i++)
		{
			EC.m_nCurrentParameterIndexForSensitivityEvaluation = narrParameterIndexes[i];
			__ad = m_EquationEvaluationNode->Evaluate(&EC);
			m_pModel->m_pDataProxy->SetSResValue(i, m_nEquationIndexInBlock, __ad.getDerivative());
		}
	}
	else if(m_pEquation->m_eEquationEvaluationMode == eFunctionEvaluation)
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
	else if(m_pEquation->m_eEquationEvaluationMode == eCommandStackEvaluation)
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
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

boost::shared_ptr<adNode> daeEquationExecutionInfo::GetEquationEvaluationNode(void) const
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
		e << "Invalid domain in DEDI, in equation [ " << pEquation->m_strCanonicalName << "]";
		throw e;
	}
	if(eDomainBounds == eDBUnknown || eDomainBounds == eCustomBound)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid domain bounds in DEDI on the domain [ " << m_pDomain->m_strCanonicalName 
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
		e << "Invalid domain in DEDI, in equation [ " << pEquation->m_strCanonicalName << "]";
		throw e;
	}
	if(narrDomainIndexes.size() == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of points cannot be 0 in DEDI, in equation [ " << pEquation->m_strCanonicalName << "]";
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
		e << "Number of points in domain [ " << m_pDomain->m_strCanonicalName << "] is 0; it should be at least 1";
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
	tmp.node = boost::shared_ptr<adNode>(node);
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
	m_pModel = NULL;
	m_eEquationDefinitionMode = eEDMUnknown;
	m_eEquationEvaluationMode = eEEMUnknown;
}

daeEquation::~daeEquation()
{
}

void daeEquation::Clone(const daeEquation& rObject)
{
	m_eEquationDefinitionMode  = rObject.m_eEquationDefinitionMode;
	m_eEquationEvaluationMode  = rObject.m_eEquationEvaluationMode;
	m_pResidualNode			   = rObject.m_pResidualNode;
	
	for(size_t i = 0; i < rObject.m_ptrarrDistributedEquationDomainInfos.size(); i++)
	{
		daeDistributedEquationDomainInfo* pDEDI = new daeDistributedEquationDomainInfo();
		pDEDI->SetName(rObject.m_ptrarrDistributedEquationDomainInfos[i]->m_strShortName);
		SetModelAndCanonicalName(pDEDI);
		dae_push_back(m_ptrarrDistributedEquationDomainInfos, pDEDI);
		pDEDI->m_pModel    = m_pModel;
		pDEDI->m_pEquation = this;
		pDEDI->Clone(*rObject.m_ptrarrDistributedEquationDomainInfos[i]);
	}
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

	strName = "EquationDefinitionMode";
	OpenEnum(pTag, strName, m_eEquationDefinitionMode);

	strName = "EquationEvaluationMode";
	OpenEnum(pTag, strName, m_eEquationEvaluationMode);

	if(m_eEquationDefinitionMode == eResidualNode)
	{
		strName = "Expression";
		adNode* node = adNode::OpenNode(pTag, strName);
		m_pResidualNode.reset(node);
	}

	strName = "DistributedDomainInfos";
	daeSetModelAndCanonicalNameDelegate<daeDistributedEquationDomainInfo> del(this, m_pModel);
	pTag->OpenObjectArray(strName, m_ptrarrDistributedEquationDomainInfos, &del);
}

void daeEquation::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "EquationDefinitionMode";
	SaveEnum(pTag, strName, m_eEquationDefinitionMode);

	strName = "EquationEvaluationMode";
	SaveEnum(pTag, strName, m_eEquationEvaluationMode);

	if(m_eEquationDefinitionMode == eResidualNode)
	{
		strName = "Expression";
		adNode::SaveNode(pTag, strName, m_pResidualNode.get());

		strName = "MathML";
		SaveNodeAsMathML(pTag, strName);
	}

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
//	OpenEnum(pTag, strName, m_eEquationDefinitionMode);

//	strName = "EquationEvaluationMode";
//	OpenEnum(pTag, strName, m_eEquationEvaluationMode);

//	if(m_eEquationDefinitionMode == eResidualNode)
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
//	SaveEnum(pTag, strName, m_eEquationDefinitionMode);

//	strName = "EquationEvaluationMode";
//	SaveEnum(pTag, strName, m_eEquationEvaluationMode);

	if(m_eEquationDefinitionMode == eResidualNode)
	{
		strName = "MathML";
		SaveNodeAsMathML(pTag, strName);
	}

//	strName = "DistributedDomainInfos";
//	pTag->SaveRuntimeObjectArray(strName, m_ptrarrDistributedEquationDomainInfos);
			
	strName = "EquationExecutionInfos";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrEquationExecutionInfos);
}

void daeEquation::SaveNodeAsMathML(io::xmlTag_t* pTag, const string& strObjectName) const
{
	string strName, strValue;
	daeSaveAsMathMLContext c(m_pModel, m_ptrarrDistributedEquationDomainInfos);
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

			if(pDedi->m_eDomainBounds == eLowerBound ||
			   pDedi->m_eDomainBounds == eUpperBound)
			{
				strName  = "mi";
				strValue = pDedi->GetName();
				//mrow1->AddTag(strName, strValue);
				xml::xmlPresentationCreator::WrapIdentifier(mrow1, strValue);
	
				strName  = "mo";
				strValue = "=";
				mrow1->AddTag(strName, strValue);
	
				if(pDedi->m_eDomainBounds == eLowerBound)
				{
					strName  = "msub";
					strValue = "";
					io::xmlTag_t* msub1 = mrow1->AddTag(strName, strValue);
	
					strName  = "mi";
					strValue = pDedi->GetName();
					//msub1->AddTag(strName, strValue);
					xml::xmlPresentationCreator::WrapIdentifier(msub1, strValue);
	
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
					strValue = pDedi->GetName();
					//msub1->AddTag(strName, strValue);
					xml::xmlPresentationCreator::WrapIdentifier(msub1, strValue);
	
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
				strValue = pDedi->GetName();
				//mrow1->AddTag(strName, strValue);
				xml::xmlPresentationCreator::WrapIdentifier(mrow1, strValue);
	
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
					strValue = pDedi->GetName();
					//msub1->AddTag(strName, strValue);
					xml::xmlPresentationCreator::WrapIdentifier(msub1, strValue);
	
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
					strValue = pDedi->GetName();
					//msub2->AddTag(strName, strValue);
					xml::xmlPresentationCreator::WrapIdentifier(msub2, strValue);
	
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

daeeEquationDefinitionMode daeEquation::GetEquationDefinitionMode(void) const
{
	return m_eEquationDefinitionMode;
}

daeeEquationEvaluationMode daeEquation::GetEquationEvaluationMode(void) const
{
	return m_eEquationEvaluationMode;
}

void daeEquation::SetEquationEvaluationMode(daeeEquationEvaluationMode eMode)
{
	if(m_eEquationDefinitionMode == eMemberFunctionPointer)
	{
	// Everything is fine, just anymode can be chosen
	}
	else if(m_eEquationDefinitionMode == eResidualNode)
	{
		if(eMode == eFunctionEvaluation)
			daeDeclareAndThrowException(exInvalidCall);
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
	}

	m_eEquationEvaluationMode = eMode;
}

void daeEquation::SetJacobianItem(size_t nEquationIndex, size_t nVariableIndex, real_t dJacobValue, daeBlock* pBlock)
{
	if(!pBlock)
		daeDeclareAndThrowException(exInvalidPointer);
	pBlock->SetJacobian(nEquationIndex, nVariableIndex, dJacobValue);
}

/*
adouble daeEquation::Calculate()
{
// Should never be called! (must be overloaded in derived classes)
	daeDeclareAndThrowException(exInvalidCall)
	return adouble();
}

adouble daeEquation::Calculate(size_t nDomain1)
{
// Should never be called! (must be overloaded in derived classes)
	daeDeclareAndThrowException(exInvalidCall)
	return adouble();
}

adouble daeEquation::Calculate(size_t nDomain1, size_t nDomain2)
{
// Should never be called! (must be overloaded in derived classes)
	daeDeclareAndThrowException(exInvalidCall)
	return adouble();
}

adouble	daeEquation::Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3)
{
// Should never be called! (must be overloaded in derived classes)
	daeDeclareAndThrowException(exInvalidCall)
	return adouble();
}

adouble	daeEquation::Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4)
{
// Should never be called! (must be overloaded in derived classes)
	daeDeclareAndThrowException(exInvalidCall);
	return adouble();
}

adouble	daeEquation::Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5)
{
// Should never be called! (must be overloaded in derived classes)
	daeDeclareAndThrowException(exInvalidCall)
	return adouble();
}

adouble	daeEquation::Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6)
{
// Should never be called! (must be overloaded in derived classes)
	daeDeclareAndThrowException(exInvalidCall)
	return adouble();
}

adouble	daeEquation::Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7)
{
// Should never be called! (must be overloaded in derived classes)
	daeDeclareAndThrowException(exInvalidCall)
	return adouble();
}

adouble	daeEquation::Calculate(size_t nDomain1, size_t nDomain2, size_t nDomain3, size_t nDomain4, size_t nDomain5, size_t nDomain6, size_t nDomain7, size_t nDomain8)
{
// Should never be called! (must be overloaded in derived classes)
	daeDeclareAndThrowException(exInvalidCall)
	return adouble();
}
*/

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

void daeEquation::GatherInfo(const vector<size_t>& narrDomainIndexes, const daeExecutionContext& EC, boost::shared_ptr<adNode>& node)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	adouble ad;
	size_t nNumberOfDomains = m_ptrarrDistributedEquationDomainInfos.size();
	if(nNumberOfDomains != narrDomainIndexes.size())
	{	
		daeDeclareException(exInvalidCall);
		e << "Illegal number of domains in equation [ " << m_strCanonicalName << "]";
		throw e;
	}

	if(m_eEquationDefinitionMode == eResidualNode)
	{
		for(size_t i = 0; i < nNumberOfDomains; i++)
			m_ptrarrDistributedEquationDomainInfos[i]->m_nCurrentIndex = narrDomainIndexes[i];

		if(m_eEquationEvaluationMode == eResidualNodeEvaluation)
			node = m_pResidualNode->Evaluate(&EC).node;
		else
			daeDeclareAndThrowException(exInvalidCall)
	}
	else if(m_eEquationDefinitionMode == eMemberFunctionPointer)
	{
//		if(nNumberOfDomains == 0)
//		{
//			ad = Calculate();
//		}
//		else if(nNumberOfDomains == 1)
//		{
//			ad = Calculate(narrDomainIndexes[0]);
//		}
//		else if(nNumberOfDomains == 2)
//		{
//			ad = Calculate(narrDomainIndexes[0], 
//						   narrDomainIndexes[1]);
//		}
//		else if(nNumberOfDomains == 3)
//		{
//			ad = Calculate(narrDomainIndexes[0], 
//						   narrDomainIndexes[1], 
//						   narrDomainIndexes[2]);
//		}
//		else if(nNumberOfDomains == 4)
//		{
//			ad = Calculate(narrDomainIndexes[0], 
//						   narrDomainIndexes[1], 
//						   narrDomainIndexes[2], 
//						   narrDomainIndexes[3]);
//		}
//		else if(nNumberOfDomains == 5)
//		{
//			ad = Calculate(narrDomainIndexes[0], 
//						   narrDomainIndexes[1], 
//						   narrDomainIndexes[2], 
//						   narrDomainIndexes[3], 
//						   narrDomainIndexes[4]);
//		}
//		else if(nNumberOfDomains == 6)
//		{
//			ad = Calculate(narrDomainIndexes[0], 
//						   narrDomainIndexes[1], 
//						   narrDomainIndexes[2], 
//						   narrDomainIndexes[3], 
//						   narrDomainIndexes[4], 
//						   narrDomainIndexes[5]);
//		}
//		else if(nNumberOfDomains == 7)
//		{
//			ad = Calculate(narrDomainIndexes[0], 
//						   narrDomainIndexes[1], 
//						   narrDomainIndexes[2], 
//						   narrDomainIndexes[3], 
//						   narrDomainIndexes[4], 
//						   narrDomainIndexes[5], 
//						   narrDomainIndexes[6]);
//		}
//		else if(nNumberOfDomains == 8)
//		{
//			ad = Calculate(narrDomainIndexes[0], 
//						   narrDomainIndexes[1], 
//						   narrDomainIndexes[2], 
//						   narrDomainIndexes[3], 
//						   narrDomainIndexes[4], 
//						   narrDomainIndexes[5], 
//						   narrDomainIndexes[6], 
//						   narrDomainIndexes[7]);
//		}
//		else
//		{	
//			daeDeclareException(exInvalidCall);
//			e << "Illegal number of domains in equation [ " << m_strCanonicalName << "]";
//			throw e;
//		}

//		if(m_eEquationEvaluationMode == eResidualNodeEvaluation)
//			node = ad.node;
//		else if(m_eEquationEvaluationMode == eCommandStackEvaluation)
//			node = ad.node;
	}
	else
	{
		daeDeclareAndThrowException(exInvalidCall)
	}
}

/*
void daeEquation::Residual(const vector<size_t>& narrDomainIndexes, const daeExecutionContext& EC)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nNumberOfDomains = m_ptrarrDistributedEquationDomainInfos.size();
	if(nNumberOfDomains != narrDomainIndexes.size())
	{	
		daeDeclareException(exInvalidCall);
		e << "Illegal number of domains in equation [ " << m_strCanonicalName << "]";
		throw e;
	}

	adouble __ad;

	if(nNumberOfDomains == 0)
	{
		__ad = Calculate();
	}
	else if(nNumberOfDomains == 1)
	{
		__ad = Calculate(narrDomainIndexes[0]);
	}
	else if(nNumberOfDomains == 2)
	{
		__ad = Calculate(narrDomainIndexes[0], 
			             narrDomainIndexes[1]);
	}
	else if(nNumberOfDomains == 3)
	{
		__ad = Calculate(narrDomainIndexes[0], 
			             narrDomainIndexes[1], 
						 narrDomainIndexes[2]);
	}
	else if(nNumberOfDomains == 4)
	{
		__ad = Calculate(narrDomainIndexes[0], 
			             narrDomainIndexes[1], 
						 narrDomainIndexes[2], 
						 narrDomainIndexes[3]);
	}
	else if(nNumberOfDomains == 5)
	{
		__ad = Calculate(narrDomainIndexes[0], 
			             narrDomainIndexes[1], 
						 narrDomainIndexes[2], 
						 narrDomainIndexes[3], 
						 narrDomainIndexes[4]);
	}
	else if(nNumberOfDomains == 6)
	{
		__ad = Calculate(narrDomainIndexes[0], 
			             narrDomainIndexes[1], 
						 narrDomainIndexes[2], 
						 narrDomainIndexes[3], 
						 narrDomainIndexes[4], 
						 narrDomainIndexes[5]);
	}
	else if(nNumberOfDomains == 7)
	{
		__ad = Calculate(narrDomainIndexes[0], 
			             narrDomainIndexes[1], 
						 narrDomainIndexes[2], 
						 narrDomainIndexes[3], 
						 narrDomainIndexes[4], 
						 narrDomainIndexes[5], 
						 narrDomainIndexes[6]);
	}
	else if(nNumberOfDomains == 8)
	{
		__ad = Calculate(narrDomainIndexes[0], 
			             narrDomainIndexes[1], 
						 narrDomainIndexes[2], 
						 narrDomainIndexes[3], 
						 narrDomainIndexes[4], 
						 narrDomainIndexes[5], 
						 narrDomainIndexes[6], 
						 narrDomainIndexes[7]);
	}
	else
	{	
		daeDeclareException(exInvalidCall);
		e << "Maximal number of domains is 8, equation [ " << m_strCanonicalName << "]";
		throw e;
	}

	SetResidualValue(EC.m_pEquationExecutionInfo->m_nEquationIndexInBlock, __ad.getValue(), EC.m_pBlock);
}

void daeEquation::Jacobian(const vector<size_t>& narrDomainIndexes, const map<size_t, size_t>& mapIndexes, daeExecutionContext& EC)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	size_t nNumberOfDomains = m_ptrarrDistributedEquationDomainInfos.size();
	if(nNumberOfDomains != narrDomainIndexes.size())
	{	
		daeDeclareException(exInvalidCall);
		e << "Illegal number of domains in equation [ " << m_strCanonicalName << "]";
		throw e;
	}

	adouble __ad;
	size_t nEquationIndex, nVariableindexInBlock;
	map<size_t, size_t>::iterator iter;
	daeExecutionContext* pEC;
	map<size_t, size_t>::const_iterator iter_out, iter_in;

	nEquationIndex = EC.m_pEquationExecutionInfo->m_nEquationIndexInBlock;
	for(iter_out = mapIndexes.begin(); iter_out != mapIndexes.end(); iter_out++)
	{
		nVariableindexInBlock										= (*iter_out).second;
		EC.m_nCurrentVariableIndexForJacobianEvaluation				= (*iter_out).first;
		EC.m_pBlock->m_nCurrentVariableIndexForJacobianEvaluation	= (*iter_out).first;

		for(iter_in = mapIndexes.begin(); iter_in != mapIndexes.end(); iter_in++)
		{
			pEC = m_pModel->m_pDataProxy->GetExecutionContext((*iter_in).first);
			pEC->m_pBlock										= EC.m_pBlock;
			pEC->m_dInverseTimeStep								= EC.m_dInverseTimeStep;
			pEC->m_pEquationExecutionInfo						= EC.m_pEquationExecutionInfo;
			pEC->m_eEquationCalculationMode						= EC.m_eEquationCalculationMode;
			pEC->m_nCurrentVariableIndexForJacobianEvaluation	= EC.m_nCurrentVariableIndexForJacobianEvaluation;
		}

		if(nNumberOfDomains == 0)
		{
			__ad = Calculate();
		}
		else if(nNumberOfDomains == 1)
		{
			__ad = Calculate(narrDomainIndexes[0]);
		}
		else if(nNumberOfDomains == 2)
		{
			__ad = Calculate(narrDomainIndexes[0], 
				             narrDomainIndexes[1]);
		}
		else if(nNumberOfDomains == 3)
		{
			__ad = Calculate(narrDomainIndexes[0], 
				             narrDomainIndexes[1], 
							 narrDomainIndexes[2]);
		}
		else if(nNumberOfDomains == 4)
		{
			__ad = Calculate(narrDomainIndexes[0], 
				             narrDomainIndexes[1], 
							 narrDomainIndexes[2], 
							 narrDomainIndexes[3]);
		}
		else if(nNumberOfDomains == 5)
		{
			__ad = Calculate(narrDomainIndexes[0], 
				             narrDomainIndexes[1], 
							 narrDomainIndexes[2], 
							 narrDomainIndexes[3], 
							 narrDomainIndexes[4]);
		}
		else if(nNumberOfDomains == 6)
		{
			__ad = Calculate(narrDomainIndexes[0], 
							 narrDomainIndexes[1], 
							 narrDomainIndexes[2], 
							 narrDomainIndexes[3], 
							 narrDomainIndexes[4], 
							 narrDomainIndexes[5]);
		}
		else if(nNumberOfDomains == 7)
		{
			__ad = Calculate(narrDomainIndexes[0], 
							 narrDomainIndexes[1], 
							 narrDomainIndexes[2], 
							 narrDomainIndexes[3], 
							 narrDomainIndexes[4], 
							 narrDomainIndexes[5], 
							 narrDomainIndexes[6]);
		}
		else if(nNumberOfDomains == 8)
		{
			__ad = Calculate(narrDomainIndexes[0], 
							 narrDomainIndexes[1], 
							 narrDomainIndexes[2], 
							 narrDomainIndexes[3], 
							 narrDomainIndexes[4], 
							 narrDomainIndexes[5], 
							 narrDomainIndexes[6], 
							 narrDomainIndexes[7]);
		}
		else
		{	
			daeDeclareException(exInvalidCall);
			e << "Maximal number of domains is 8, equation [ " << m_strCanonicalName << "]";
			throw e;
		}
		SetJacobianItem(nEquationIndex, nVariableindexInBlock, __ad.getDerivative(), EC.m_pBlock);
	}
}
*/

daeDEDI* daeEquation::DistributeOnDomain(daeDomain& rDomain, daeeDomainBounds eDomainBounds)
{
	daeDistributedEquationDomainInfo* pDEDI = new daeDistributedEquationDomainInfo(this, &rDomain, eDomainBounds);
	pDEDI->SetName(/*GetCanonicalName() + "." +*/ rDomain.GetName());
	SetModelAndCanonicalName(pDEDI);
	dae_push_back(m_ptrarrDistributedEquationDomainInfos, pDEDI);
	return pDEDI;
}

daeDEDI* daeEquation::DistributeOnDomain(daeDomain& rDomain, const vector<size_t>& narrDomainIndexes)
{
	daeDistributedEquationDomainInfo* pDEDI = new daeDistributedEquationDomainInfo(this, &rDomain, narrDomainIndexes);
	pDEDI->SetName(/*GetCanonicalName() + "." +*/ rDomain.GetName());
	SetModelAndCanonicalName(pDEDI);
	dae_push_back(m_ptrarrDistributedEquationDomainInfos, pDEDI);
	return pDEDI;
}

daeDEDI* daeEquation::DistributeOnDomain(daeDomain& rDomain, const size_t* pnarrDomainIndexes, size_t n)
{
	vector<size_t> narrDomainIndexes;
	
	narrDomainIndexes.resize(n);
	for(size_t i = 0; i < n; i++)
		narrDomainIndexes[i] = pnarrDomainIndexes[i];

	return DistributeOnDomain(rDomain, narrDomainIndexes);	
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

void daeEquation::SetModelAndCanonicalName(daeObject* pObject)
{
	if(!pObject)
		daeDeclareAndThrowException(exInvalidPointer);
	
	string strCanonicalName;
	strCanonicalName = this->m_strCanonicalName + "." + pObject->GetName();
	pObject->SetCanonicalName(strCanonicalName);
	pObject->SetModel(m_pModel);
}

void daeEquation::SetResidual(adouble res)
{
	if(!res.node)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid residual, equation [ " << m_strCanonicalName << "]";
		throw e;
	}

	m_eEquationDefinitionMode = eResidualNode;
	m_eEquationEvaluationMode = eResidualNodeEvaluation;
	m_pResidualNode           = res.node;
}

adouble daeEquation::GetResidual(void) const
{
	adouble ad;
	ad.node = m_pResidualNode;
	return ad;
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

// Check definition/evaluation modes	
	if(m_eEquationDefinitionMode == eEDMUnknown)
	{
		strError = "Invalid definition mode in equation [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	if(m_eEquationEvaluationMode == eEEMUnknown)
	{
		strError = "Invalid evaluation mode in equation [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	if(m_eEquationDefinitionMode == eResidualNode && m_eEquationEvaluationMode == eFunctionEvaluation)
	{
		strError = "Incompatible definition/evaluation modes in equation [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check residual node	
	if(m_eEquationDefinitionMode == eResidualNode && !m_pResidualNode)
	{
		strError = "Invalid residual in equation [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
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
	m_eEquationDefinitionMode = eEDMUnknown;
	m_eEquationEvaluationMode = eEEMUnknown;
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
	m_eEquationDefinitionMode = eResidualNode;
	m_eEquationEvaluationMode = eResidualNodeEvaluation;
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
