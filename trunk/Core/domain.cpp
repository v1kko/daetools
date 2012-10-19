#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "nodes_array.h"
#include "units_io.h"

namespace dae 
{
namespace core 
{
daeDomain::daeDomain()
{
	m_dLowerBound			= 0;
	m_dUpperBound			= 0;
	m_nNumberOfIntervals	= 0;
	m_nNumberOfPoints		= 0;
	m_eDomainType			= eDTUnknown;
	m_nDiscretizationOrder  = 0;
	m_eDiscretizationMethod	= eDMUnknown;
	m_pParentPort           = NULL;
}

daeDomain::daeDomain(string strName, daeModel* pModel, const unit& units, string strDescription)
{
	m_Unit					= units;
	m_dLowerBound			= 0;
	m_dUpperBound			= 0;
	m_nNumberOfIntervals	= 0;
	m_nNumberOfPoints		= 0;
	m_eDomainType			= eDTUnknown;
	m_nDiscretizationOrder  = 0;
	m_eDiscretizationMethod	= eDMUnknown;
	m_pParentPort           = NULL;

	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	pModel->AddDomain(*this, strName, units, strDescription);
}

daeDomain::daeDomain(string strName, daePort* pPort, const unit& units, string strDescription)
{
	m_Unit					= units;
	m_dLowerBound			= 0;
	m_dUpperBound			= 0;
	m_nNumberOfIntervals	= 0;
	m_nNumberOfPoints		= 0;
	m_eDomainType			= eDTUnknown;
	m_nDiscretizationOrder  = 0;
	m_eDiscretizationMethod	= eDMUnknown;
	m_pParentPort           = pPort;

	if(!pPort)
		daeDeclareAndThrowException(exInvalidPointer);
	pPort->AddDomain(*this, strName, units, strDescription);
}

daeDomain::~daeDomain()
{
}

void daeDomain::Clone(const daeDomain& rObject)
{
	m_dLowerBound			= rObject.m_dLowerBound;
	m_dUpperBound			= rObject.m_dUpperBound;
	m_nNumberOfIntervals	= rObject.m_nNumberOfIntervals;
	m_nNumberOfPoints		= rObject.m_nNumberOfPoints;
	m_eDomainType			= rObject.m_eDomainType;
	m_nDiscretizationOrder  = rObject.m_nDiscretizationOrder;
	m_eDiscretizationMethod	= rObject.m_eDiscretizationMethod;
	m_darrPoints			= rObject.m_darrPoints;
}

void daeDomain::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeObject::Open(pTag);

	strName = "Type";
	OpenEnum(pTag, strName, m_eDomainType);
}

void daeDomain::Save(io::xmlTag_t* pTag) const
{
	string strName, strValue;

	daeObject::Save(pTag);

	strName = "Type";
	SaveEnum(pTag, strName, m_eDomainType);

	strName = "Units";
	units::Save(pTag, strName, m_Unit);
	
	strName = "MathMLUnits";
	io::xmlTag_t* pChildTag = pTag->AddTag(strName);

	strName = "math";
	io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);

	strName = "xmlns";
	strValue = "http://www.w3.org/1998/Math/MathML";
	pMathMLTag->AddAttribute(strName, strValue);
	
	units::SaveAsPresentationMathML(pMathMLTag, m_Unit);
}
	
void daeDomain::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport;
	boost::format fmtFile;

	if(c.m_bExportDefinition)
	{
		if(eLanguage == ePYDAE)
		{
		}
		else if(eLanguage == eCDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "daeDomain %1%;\n";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName();
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	else
	{
		if(eLanguage == ePYDAE)
		{
			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.%1% = daeDomain(\"%2%\", self, \"%3%\")\n";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% m_strShortName 
					% m_strDescription;
		}
		else if(eLanguage == eCDAE)
		{
			strExport = ",\n" + c.CalculateIndent(c.m_nPythonIndentLevel) + "%1%(\"%2%\", this, \"%3%\")";
			fmtFile.parse(strExport);
			fmtFile % GetStrippedName() 
					% m_strShortName 
					% m_strDescription;
		}
		else
		{
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
	
	strContent += fmtFile.str();
}

void daeDomain::OpenRuntime(io::xmlTag_t* pTag)
{
//	daeObject::OpenRuntime(pTag);
}

void daeDomain::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName, strValue;

	daeObject::SaveRuntime(pTag);

	strName = "Type";
	SaveEnum(pTag, strName, m_eDomainType);
	
	strName = "Units";
	units::Save(pTag, strName, m_Unit);
	
	strName = "MathMLUnits";
	io::xmlTag_t* pChildTag = pTag->AddTag(strName);

	strName = "math";
	io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);

	strName = "xmlns";
	strValue = "http://www.w3.org/1998/Math/MathML";
	pMathMLTag->AddAttribute(strName, strValue);
	
	units::SaveAsPresentationMathML(pMathMLTag, m_Unit);

	strName = "NumberOfIntervals";
	pTag->Save(strName, m_nNumberOfIntervals);

	strName = "NumberOfPoints";
	pTag->Save(strName, m_nNumberOfPoints);

	strName = "Points";
	pTag->SaveArray(strName, m_darrPoints);

	strName = "LowerBound";
	pTag->Save(strName, m_dLowerBound);

	strName = "UpperBound";
	pTag->Save(strName, m_dUpperBound);

	strName = "DiscretizationMethod";
	SaveEnum(pTag, strName, m_eDiscretizationMethod);

	strName = "DiscretizationOrder";
	pTag->Save(strName, m_nDiscretizationOrder);
}

void daeDomain::CreateDistributed(daeeDiscretizationMethod eMethod, 
								  size_t nOrder, 
								  size_t nNoIntervals, 
								  real_t dLB, 
								  real_t dUB)
{
	if(nNoIntervals == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of intervals in domain [" << GetCanonicalName() << "] cannot be zero";
		throw e;
	}

	if(nNoIntervals < 2)
	{
		daeDeclareException(exInvalidCall);
		e << "Number of intervals in domain [" << GetCanonicalName() << "] cannot be less than 2";
		throw e;
	}

	if(dLB >= dUB)
	{
		daeDeclareException(exInvalidCall);
		e << "The lower bound is greater than the upper one in domain [" << GetCanonicalName();
		throw e;
	}

	m_dLowerBound			= dLB;
	m_dUpperBound			= dUB;
	m_nNumberOfIntervals	= nNoIntervals;
	m_eDomainType			= eDistributed;
	m_nDiscretizationOrder  = nOrder;
	m_eDiscretizationMethod	= eMethod;

	CreatePoints();
}

void daeDomain::CreateArray(size_t nNoIntervals)
{
	if(nNoIntervals == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Number of intervals in domain [" << GetCanonicalName() << "] cannot be zero";
		throw e;
	}

	m_dLowerBound			= 1;
	m_dUpperBound			= nNoIntervals;
	m_nNumberOfIntervals	= nNoIntervals;
	m_eDomainType			= eArray;
	m_nDiscretizationOrder  = 0;
	m_eDiscretizationMethod	= eDMUnknown;

	CreatePoints();
}

string daeDomain::GetCanonicalName(void) const
{
	if(m_pParentPort)
		return m_pParentPort->GetCanonicalName() + '.' + m_strShortName;
	else
		return daeObject::GetCanonicalName();
}

adouble daeDomain::partial(daePartialDerivativeVariable& pdv) const
{
	if(m_eDomainType != eDistributed)
		daeDeclareAndThrowException(exInvalidCall); 

	if(m_eDiscretizationMethod == eBFDM)
		return pd_BFD(pdv);
	else if(m_eDiscretizationMethod == eFFDM)
		return pd_FFD(pdv);
	else if(m_eDiscretizationMethod == eCFDM)
		return pd_CFD(pdv);
	else if(m_eDiscretizationMethod == eCustomDM)
		return customPartialDerivative(pdv);
	else
		daeDeclareAndThrowException(exInvalidCall); 
}

adouble daeDomain::customPartialDerivative(daePartialDerivativeVariable& /*pdv*/) const
{
	daeDeclareAndThrowException(exNotImplemented)
	return adouble();
}

adouble daeDomain::pd_BFD(daePartialDerivativeVariable& /*pdv*/) const
{
	return adouble();
}

adouble daeDomain::pd_FFD(daePartialDerivativeVariable& /*pdv*/) const
{
	return adouble();
}

adouble daeDomain::pd_CFD(daePartialDerivativeVariable& pdv) const
{
	adouble pardev;
// Index which we are calculating partial derivative for
	const size_t n = pdv.GetPoint();
// Domain which we are calculating partial derivative for
	const daeDomain& d = pdv.GetDomain();
// Number of points in the domain we are calculating partial derivative for
	const size_t N = d.GetNumberOfPoints();

	switch(d.GetDiscretizationOrder())
	{
	case 2:
		if(N < 3)
		{	
			daeDeclareException(exInvalidCall);
			e << "Number of points in domain [" << d.GetCanonicalName() << "] cannot be less than 3";
			throw e;
		}

		if(pdv.GetOrder() == 1)
		{
			if(n == 0) // LEFT BOUND
			{
			//	dV(0)/dD = (-3V[0] + 4V[1] - V[2]) / (D[2] - D[0])
				pardev = (-3*pdv[0] + 4*pdv[1] - pdv[2]) / (d[2] - d[0]);
			}
			else if(n == N-1) // RIGHT BOUND
			{
			//	dV(n)/dD = (3V[n] - 4V[n-1] + V[n-2]) / (D[n] - D[n-2])
				pardev = (3*pdv[n] - 4*pdv[n-1] + pdv[n-2]) / (d[n] - d[n-2]);
			}
			else // INTERIOR POINTs
			{
			//	dV(i)/dD = (V[i+1] - V[i-1]) / (D[i+1] - D[i-1])
				pardev = (pdv[n+1] - pdv[n-1]) / (d[n+1] - d[n-1]);
			}
		}
        else if(pdv.GetOrder() == 2)
		{
			if(n == 0) // LEFT BOUND
			{
			//	dV(0)/dD = (V[0] - 2V[1] + V[2]) / ((D[2] - D[1]) * (D[1] - D[0]))
				pardev = (pdv[0] - 2*pdv[1] + pdv[2]) / ( (d[2] - d[1]) * (d[1] - d[0]) );
			}
			else if(n == N-1) // RIGHT BOUND 
			{
			//	dV(n)/dD = (V[n] - 2V[n-1] + V[n-2]) / ((D[n] - D[n-1]) * (D[n-1] - D[n-2]))
				pardev = (pdv[n] - 2*pdv[n-1] + pdv[n-2]) / ((d[n] - d[n-1]) * (d[n-1] - d[n-2]));
			}
			else // INTERIOR POINTs
			{
			//	d2V(i)/dD2 = (V[i+1] - 2V[i] + V[i-1]) / ((D[i+1] - D[i]) * (D[i] - D[i-1]))
				pardev = (pdv[n+1] - 2*pdv[n] + pdv[n-1]) / ((d[n+1] - d[n]) * (d[n] - d[n-1]));
			}
		}
        else
        {
            daeDeclareAndThrowException(exNotImplemented); 
        }
		break;

	case 4:
		daeDeclareAndThrowException(exNotImplemented); 
		break;

	case 6:
		daeDeclareAndThrowException(exNotImplemented); 
		break;

	default:
		daeDeclareAndThrowException(exNotImplemented); 
	}

	return pardev;
}

void daeDomain::GetPoints(std::vector<real_t>& darrPoints) const
{
	darrPoints = m_darrPoints;
}

void daeDomain::SetPoints(const vector<real_t>& darrPoints)
{
	if(m_eDomainType == eArray)
	{	
		daeDeclareException(exInvalidCall);
		e << "Cannot reset an array, domain [" << GetCanonicalName() << "]";
		throw e;
	}
	if(m_nNumberOfPoints != darrPoints.size())
	{	
		daeDeclareException(exInvalidCall); 
		e << "Invalid number of points in domain [" << GetCanonicalName() << "]";
		throw e;
	}

	m_dLowerBound = darrPoints[0];
	m_dUpperBound = darrPoints[darrPoints.size()-1];
	for(size_t i = 0; i < m_nNumberOfPoints; i++)
		m_darrPoints[i] = darrPoints[i];
}

void daeDomain::CreatePoints()
{
	size_t i;
	real_t dInterval;

	if(m_nNumberOfIntervals == 0)
	{	
		daeDeclareException(exInvalidCall);
		e << "Invalid number of intervals in domain [" << GetCanonicalName() << "]";
		throw e;
	}

	m_darrPoints.clear();

	if(m_eDomainType == eArray)
	{
		m_nNumberOfPoints = m_nNumberOfIntervals;
		m_darrPoints.resize(m_nNumberOfPoints);
		for(i = 0; i < m_nNumberOfPoints; i++)
			m_darrPoints[i] = i + 1;
	}
	else
	{
		switch(m_eDiscretizationMethod)
		{
		case eFFDM:
		case eBFDM:
		case eCFDM:
			m_nNumberOfPoints = m_nNumberOfIntervals+1;
			m_darrPoints.resize(m_nNumberOfPoints);
			dInterval = (m_dUpperBound - m_dLowerBound) / (m_nNumberOfIntervals);
			for(i = 0; i < m_nNumberOfPoints; i++)
				m_darrPoints[i] = m_dLowerBound + i * dInterval;
			break;

		default:
			daeDeclareAndThrowException(exNotImplemented); 
		}
	}
}

void daeDomain::SetType(daeeDomainType eDomainType)
{
	m_eDomainType = eDomainType;
}

unit daeDomain::GetUnits(void) const
{
	return m_Unit;
}

void daeDomain::SetUnits(const unit& units)
{
	m_Unit = units;
}

daeeDomainType daeDomain::GetType(void) const
{
	return m_eDomainType;
}

real_t daeDomain::GetLowerBound(void) const
{
	return m_dLowerBound;
}

real_t daeDomain::GetUpperBound(void) const
{
	return m_dUpperBound;
}

size_t daeDomain::GetNumberOfPoints(void) const
{
	return m_nNumberOfPoints;
}

size_t daeDomain::GetNumberOfIntervals(void) const
{
	return m_nNumberOfIntervals;
}

size_t daeDomain::GetDiscretizationOrder(void) const
{
	return m_nDiscretizationOrder;
}

daeeDiscretizationMethod daeDomain::GetDiscretizationMethod(void) const
{
	return m_eDiscretizationMethod;
}

adouble_array daeDomain::array(void)
{
	daeDeclareAndThrowException(exNotImplemented)

	adouble_array varArray; 
	return varArray;
}

adouble_array daeDomain::array(int start, int end, int step)
{
	daeDeclareAndThrowException(exNotImplemented)

	adouble_array varArray; 

//	if(!m_pModel)
//	{	
//		daeDeclareException(exInvalidCall); 
//		e << "Invalid parent model in domain [" << GetCanonicalName() << "]";
//		throw e;
//	}
//	
//	for(size_t i = 0; i < m_nNumberOfPoints; i++)
//		varArray.m_arrValues.push_back((*this)[i]);
//	
//	if(m_pModel->m_pDataProxy->GetGatherInfo())
//	{
//		adRuntimeVariableNodeArray* node = new adRuntimeVariableNodeArray();
//		varArray.node = adNodeArrayPtr(node);
//		varArray.setGatherInfo(true);
//		node->m_pVariable = const_cast<daeVariable*>(this);
//		
//		size_t size = varArray.m_arrValues.size();
//		if(size == 0)
//			daeDeclareAndThrowException(exInvalidCall); 
//		
//		node->m_ptrarrVariableNodes.resize(size);
//		node->m_arrIndexes.resize(size);
//		for(size_t i = 0; i < size; i++)
//		{
//			node->m_ptrarrVariableNodes[i] = varArray.m_arrValues[i].node;
//			node->m_arrIndexes[i] = 0; // varArray.m_arrValues[i].node->;
//		}
//	}

	return varArray;
}

adouble daeDomain::operator()(size_t nIndex) const
{
    return (*this)[nIndex];
}

adouble daeDomain::operator[](size_t nIndex) const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 

	adouble tmp;
	adDomainIndexNode* node = new adDomainIndexNode(const_cast<daeDomain*>(this), nIndex, const_cast<real_t*>(&(m_darrPoints[nIndex])));
	tmp.node = adNodePtr(node);
	tmp.setGatherInfo(true);

	return tmp;
}

const real_t* daeDomain::GetPoint(size_t nIndex) const
{
	if(nIndex >= m_darrPoints.size())
		daeDeclareAndThrowException(exOutOfBounds); 

	return &m_darrPoints[nIndex];
}

bool daeDomain::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;
	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;
	
// Do the basic tests of no. points
	if(m_nNumberOfIntervals == 0 || m_nNumberOfPoints == 0)
	{
		strError = "Invalid number of intervals/points in domain [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	if(m_nNumberOfPoints != m_darrPoints.size())
	{
		strError = "Number of allocated points not equal to the given number in domain [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Depending on the type, perform some type-dependant tasks
	if(m_eDomainType == eDTUnknown)
	{
		strError = "Invalid domain type in domain [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	else if(m_eDomainType == eDistributed)
	{
		if(m_nNumberOfPoints < 2)
		{
			strError = "Invalid number of points in domain [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		if(m_dLowerBound >= m_dUpperBound)
		{
			strError = "Invalid bounds in domain [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
		if(m_eDiscretizationMethod == eDMUnknown)
		{
			strError = "Invalid discretization method in domain [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
	}
	else if(m_eDomainType == eArray)
	{
		if(m_eDiscretizationMethod != eDMUnknown)
		{
			strError = "Invalid discretization method in domain [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
	}

	return bCheck;
}

/******************************************************************
	Find functions
*******************************************************************/
void FindDomains(const std::vector<daeDomain*>& ptrarrSource, std::vector<daeDomain*>& ptrarrDestination, daeModel* pParentModel)
{
	string strRelativeName;
	daeDomain* pDomain;

	ptrarrDestination.resize(ptrarrSource.size());
	for(size_t i = 0; i < ptrarrSource.size(); i++)
	{
		strRelativeName = ptrarrSource[i]->GetNameRelativeToParentModel();
		pDomain = dynamic_cast<daeDomain*>(pParentModel->FindObjectFromRelativeName(strRelativeName));
		if(!pDomain)
			daeDeclareAndThrowException(exInvalidPointer);
		ptrarrDestination[i] = pDomain;
	}
}

daeDomain* FindDomain(const daeDomain* pSource, daeModel* pParentModel)
{
	string strRelativeName = pSource->GetNameRelativeToParentModel();
	return dynamic_cast<daeDomain*>(pParentModel->FindObjectFromRelativeName(strRelativeName));
}

daeEventPort* FindEventPort(const daeEventPort* pSource, daeModel* pParentModel)
{
	string strRelativeName = pSource->GetNameRelativeToParentModel();
	return dynamic_cast<daeEventPort*>(pParentModel->FindObjectFromRelativeName(strRelativeName));
}

//daeEquation* FindEquation(const daeEquation* pSource, daeModel* pParentModel)
//{
//	string strRelativeName = pSource->GetNameRelativeToParentModel();
//	return dynamic_cast<daeEquation*>(pParentModel->FindObjectFromRelativeName(strRelativeName));
//}

}
}
