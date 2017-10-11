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
    //m_nDiscretizationOrder  = 0;
    //m_eDiscretizationMethod = eDMUnknown;
    m_pParentPort           = NULL;
}

daeDomain::daeDomain(string strName, daeModel* pModel, const unit& units, string strDescription)
{
    if(!pModel)
    {
        daeDeclareException(exInvalidPointer);
        string msg = "Cannot create the domain [%s]: the parent model object is a NULL pointer/reference";
        e << (boost::format(msg) % strName).str();
        throw e;
    }

    m_Unit					= units;
    m_dLowerBound			= 0;
    m_dUpperBound			= 0;
    m_nNumberOfIntervals	= 0;
    m_nNumberOfPoints		= 0;
    m_eDomainType			= eDTUnknown;
    //m_nDiscretizationOrder  = 0;
    //m_eDiscretizationMethod = eDMUnknown;
    m_pParentPort           = NULL;

    pModel->AddDomain(*this, strName, units, strDescription);
}

daeDomain::daeDomain(string strName, daePort* pPort, const unit& units, string strDescription)
{
    if(!pPort)
    {
        daeDeclareException(exInvalidPointer);
        string msg = "Cannot create the domain [%s]: the parent port object is a NULL pointer/reference";
        e << (boost::format(msg) % strName).str();
        throw e;
    }

    m_Unit					= units;
    m_dLowerBound			= 0;
    m_dUpperBound			= 0;
    m_nNumberOfIntervals	= 0;
    m_nNumberOfPoints		= 0;
    m_eDomainType			= eDTUnknown;
    //m_nDiscretizationOrder  = 0;
    //m_eDiscretizationMethod	= eDMUnknown;
    m_pParentPort           = pPort;

    pPort->AddDomain(*this, strName, units, strDescription);
}

daeDomain::~daeDomain()
{
}

void daeDomain::Clone(const daeDomain& rObject)
{
//	m_dLowerBound			= rObject.m_dLowerBound;
//	m_dUpperBound			= rObject.m_dUpperBound;
//	m_nNumberOfIntervals	= rObject.m_nNumberOfIntervals;
//	m_nNumberOfPoints		= rObject.m_nNumberOfPoints;
//	m_eDomainType			= rObject.m_eDomainType;
//	m_nDiscretizationOrder  = rObject.m_nDiscretizationOrder;
//	m_eDiscretizationMethod	= rObject.m_eDiscretizationMethod;
//	m_darrPoints			= rObject.m_darrPoints;
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

/*
    strName = "MathMLUnits";
    io::xmlTag_t* pChildTag = pTag->AddTag(strName);

    strName = "math";
    io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);

    strName = "xmlns";
    strValue = "http://www.w3.org/1998/Math/MathML";
    pMathMLTag->AddAttribute(strName, strValue);

    units::SaveAsPresentationMathML(pMathMLTag, m_Unit);
*/
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
/*
    strName = "MathMLUnits";
    io::xmlTag_t* pChildTag = pTag->AddTag(strName);

    strName = "math";
    io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);

    strName = "xmlns";
    strValue = "http://www.w3.org/1998/Math/MathML";
    pMathMLTag->AddAttribute(strName, strValue);

    units::SaveAsPresentationMathML(pMathMLTag, m_Unit);
*/
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

    //strName = "DiscretizationMethod";
    //SaveEnum(pTag, strName, m_eDiscretizationMethod);

    //strName = "DiscretizationOrder";
    //pTag->Save(strName, m_nDiscretizationOrder);
}

void daeDomain::CreateStructuredGrid(size_t nNoIntervals, quantity qLB, quantity qUB)
{
    real_t dLB = qLB.scaleTo(m_Unit).getValue();
    real_t dUB = qUB.scaleTo(m_Unit).getValue();

    CreateStructuredGrid(nNoIntervals, dLB, dUB);
}

void daeDomain::CreateStructuredGrid(size_t nNoIntervals, real_t dLB, real_t dUB)
{
/*
    if(eMethod != eCFDM)
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot create a uniform grid domain [%s]: the only supported discretization method is [eCFDM]";
        e << (boost::format(msg) % GetCanonicalName()).str();
        throw e;
    }

    if(nOrder != 2)
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot create a uniform grid domain [%s]: the only supported discretization order is 2";
        e << (boost::format(msg) % GetCanonicalName()).str();
        throw e;
    }
*/
    if(nNoIntervals < 2)
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot create a uniform grid domain [%s]: the number of intervals is less than 2 (%d)";
        e << (boost::format(msg) % GetCanonicalName() % nNoIntervals).str();
        throw e;
    }

    if(dLB >= dUB)
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot create a uniform grid domain [%s]: the lower bound is greater than or equal the upper bound (%f >= %f)";
        e << (boost::format(msg) % GetCanonicalName() % dLB % dUB).str();
        throw e;
    }

    m_dLowerBound			= dLB;
    m_dUpperBound			= dUB;
    m_nNumberOfIntervals	= nNoIntervals;
    m_eDomainType			= eStructuredGrid;
    //m_nDiscretizationOrder  = nOrder;
    //m_eDiscretizationMethod	= eMethod;

    CreatePoints();
}

void daeDomain::CreateUnstructuredGrid(const std::vector<daePoint>& coordinates)
{
    m_arrCoordinates        = coordinates;
    m_dLowerBound			= 0;
    m_dUpperBound			= 0;
    m_nNumberOfIntervals	= 0;
    m_nNumberOfPoints       = coordinates.size();
    m_eDomainType			= eUnstructuredGrid;
    //m_nDiscretizationOrder  = 0;
    //m_eDiscretizationMethod	= eDMUnknown;
}

void daeDomain::CreateArray(size_t nNoIntervals)
{
    if(nNoIntervals == 0)
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot create an array domain [%s]: the number of intervals is 0";
        e << (boost::format(msg) % GetCanonicalName()).str();
        throw e;
    }

    m_dLowerBound			= 1;
    m_dUpperBound			= nNoIntervals;
    m_nNumberOfIntervals	= nNoIntervals;
    m_eDomainType			= eArray;
    //m_nDiscretizationOrder  = 0;
    //m_eDiscretizationMethod	= eDMUnknown;

    CreatePoints();
}

string daeDomain::GetCanonicalName(void) const
{
    if(m_pParentPort)
        return m_pParentPort->GetCanonicalName() + '.' + m_strShortName;
    else
        return daeObject::GetCanonicalName();
}

const std::vector<daePoint>& daeDomain::GetCoordinates() const
{
    return m_arrCoordinates;
}

void daeDomain::GetPoints(std::vector<real_t>& darrPoints) const
{
    darrPoints = m_darrPoints;
}

void daeDomain::SetPoints(const vector<real_t>& darrPoints)
{
    if(m_eDomainType == eArray || m_eDomainType == eUnstructuredGrid)
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot reset the points of the domain [%s]: it is not a stuctured grid domain";
        e << (boost::format(msg) % GetCanonicalName()).str();
        throw e;
    }
    if(m_nNumberOfPoints != darrPoints.size())
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot reset the points of the domain [%s]: the number of points is illegal "
                     "(required: %d, sent: %d)";
        e << (boost::format(msg) % GetCanonicalName() % m_darrPoints.size() % darrPoints.size()).str();
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

    m_darrPoints.clear();

    if(m_eDomainType == eArray)
    {
        m_nNumberOfPoints = m_nNumberOfIntervals;
        m_darrPoints.resize(m_nNumberOfPoints);
        for(i = 0; i < m_nNumberOfPoints; i++)
            m_darrPoints[i] = i + 1;
    }
    else if(m_eDomainType == eStructuredGrid)
    {
        m_nNumberOfPoints = m_nNumberOfIntervals + 1;
        m_darrPoints.resize(m_nNumberOfPoints);
        dInterval = (m_dUpperBound - m_dLowerBound) / (m_nNumberOfIntervals);
        for(i = 0; i < m_nNumberOfPoints; i++)
            m_darrPoints[i] = m_dLowerBound + i * dInterval;
        /*
        switch(m_eDiscretizationMethod)
        {
        case eFFDM:
            daeDeclareAndThrowException(exNotImplemented);

        case eBFDM:
            daeDeclareAndThrowException(exNotImplemented);

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
        */
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
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

daePort* daeDomain::GetParentPort(void) const
{
    return m_pParentPort;
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

//size_t daeDomain::GetDiscretizationOrder(void) const
//{
//	return m_nDiscretizationOrder;
//}

//daeeDiscretizationMethod daeDomain::GetDiscretizationMethod(void) const
//{
//	return m_eDiscretizationMethod;
//}

adouble_array daeDomain::array(void)
{
    daeIndexRange indexRange(this);
    daeArrayRange arrayRange(indexRange);

    return daeDomain::array(arrayRange);
}

adouble_array daeDomain::array(const daeIndexRange& indexRange)
{
    daeArrayRange arrayRange(indexRange);

    return daeDomain::array(arrayRange);
}

adouble_array daeDomain::array(const daeArrayRange& arrayRange)
{
    adouble_array varArray;

    if(arrayRange.m_eType == eRangeDomainIndex)
    {
        if(arrayRange.m_domainIndex.m_eType == eDomainIterator ||
           arrayRange.m_domainIndex.m_eType == eIncrementedDomainIterator)
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid argument for the daeDomain::array function (must not be the daeDEDI object)";
            throw e;
        }
    }

    adSetupDomainNodeArray* node = new adSetupDomainNodeArray(this, arrayRange);
    varArray.node = adNodeArrayPtr(node);
    varArray.setGatherInfo(true);

    return varArray;
}

/*
adouble_array daeDomain::array(const std::vector<size_t>& narrIndexes)
{
    daeIndexRange indexRange(this, narrIndexes);
    daeArrayRange arrayRange(indexRange);

    adouble_array varArray;

    adSetupDomainNodeArray* node = new adSetupDomainNodeArray(this, arrayRange);
    varArray.node = adNodeArrayPtr(node);
    varArray.setGatherInfo(true);

    return varArray;
}

adouble_array daeDomain::array(int start, int end, int step)
{
    daeIndexRange indexRange(this, start, end, step);
    daeArrayRange arrayRange(indexRange);

    adouble_array varArray;

    adSetupDomainNodeArray* node = new adSetupDomainNodeArray(this, arrayRange);
    varArray.node = adNodeArrayPtr(node);
    varArray.setGatherInfo(true);

    return varArray;
}
*/

adouble daeDomain::operator()(size_t nIndex) const
{
    return (*this)[nIndex];
}

adouble daeDomain::operator[](size_t nIndex) const
{
    adouble tmp;
    real_t* pdPoint = const_cast<real_t*>(GetPoint(nIndex));
    adDomainIndexNode* node = new adDomainIndexNode(const_cast<daeDomain*>(this), nIndex, pdPoint);
    tmp.node = adNodePtr(node);
    tmp.setGatherInfo(true);

    return tmp;
}

const real_t* daeDomain::GetPoint(size_t nIndex) const
{
    if(m_darrPoints.empty())
    {
        daeDeclareException(exInvalidCall);
        string msg = "Cannot get a point from the domain [%s]: domain not initialized with a CreateArray/CreateDistributed call";
        e << (boost::format(msg) % GetCanonicalName()).str();
        throw e;
    }
    if(nIndex >= m_darrPoints.size())
    {
        daeDeclareException(exOutOfBounds);
        string msg = "Cannot get a point from the domain [%s]: index out of bounds (%d >= %d)";
        e << (boost::format(msg) % GetCanonicalName() % nIndex % m_darrPoints.size()).str();
        throw e;
    }

    return &(m_darrPoints[nIndex]);
}

bool daeDomain::CheckObject(vector<string>& strarrErrors) const
{
    string strError;

    bool bCheck = true;

    if(!daeObject::CheckObject(strarrErrors))
        bCheck = false;

// Do the basic tests of no. points
    if(m_nNumberOfIntervals == 0 && m_nNumberOfPoints == 0)
    {
        strError = "Invalid number of intervals/points in domain [" + GetCanonicalName() + "]";
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
    else if(m_eDomainType == eStructuredGrid)
    {
        if(m_nNumberOfPoints != m_darrPoints.size())
        {
            strError = "Number of allocated points not equal to the given number in domain [" + GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }
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
        /*
        if(m_eDiscretizationMethod == eDMUnknown)
        {
            strError = "Invalid discretization method in domain [" + GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }
        */
    }
    else if(m_eDomainType == eArray)
    {
        /*
        if(m_eDiscretizationMethod != eDMUnknown)
        {
            strError = "Invalid discretization method in domain [" + GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }
        */
    }
    else if(m_eDomainType == eUnstructuredGrid)
    {
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
