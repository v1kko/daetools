#include "stdafx.h"
#include "datareporters.h"
#include "../Core/xmlfile.h"
#include "../Core/helpers.h"
using namespace dae;
using namespace dae::xml;

namespace dae
{
namespace datareporting
{
daeHTMLFileDataReporter::daeHTMLFileDataReporter()
{
}

daeHTMLFileDataReporter::~daeHTMLFileDataReporter()
{
}

void daeHTMLFileDataReporter::WriteDataToFile()
{
	//size_t i, j, k;
	//daeDataReporterDomain*		pDomain;
	//daeDataReporterVariable*	pVariable;
	//daeDataReporterResultSet*	pCurrentResultSet;

	//if(!IsConnected())
	//	return;

	//of << "<html>" << endl;
	//of << "<head>" << endl;
	//of << "</head>" << endl;
	//of << "<body>" << endl;

	//of << "<table border=1 cellspacing=0 cellpadding=2>" << endl;
	//	of << "<tr>" << endl;
	//	of << "<td colspan=3>" << endl;
	//	of << "Domains" << endl;
	//	of << "</td>" << endl;
	//	of << "</tr>" << endl;

	//of << "<tr>" << endl;
	//of << "<td>" << endl;
	//of << "Name";
	//of << "</td>" << endl;

	//of << "<td>" << endl;
	//of << "No. points";
	//of << "</td>" << endl;

	//of << "<td>" << endl;
	//of << "Type";
	//of << "</td>" << endl;
	//of << "</tr>" << endl;
	//for(i = 0; i < m_drProcess.m_ptrarrRegisteredDomains.size(); i++)
	//{
	//	pDomain = m_drProcess.m_ptrarrRegisteredDomains[i];
	//	if(!pDomain)
	//		return;

	//	of << "<tr>" << endl;

	//	of << "<td>" << endl;
	//	of << pDomain->m_strName;
	//	of << "</td>" << endl;

	//	of << "<td>" << endl;
	//	of << pDomain->m_nNumberOfPoints;
	//	of << "</td>" << endl;

	//	of << "<td>" << endl;
	//	of << (pDomain->m_eType == eArray ? "Array" : "Distributed");
	//	of << "</td>" << endl;

	//	of << "</tr>" << endl;
	//}
	//of << "</table>" << endl;
	//of << "<br>" << endl;


	//of << "<table border=1 cellspacing=0 cellpadding=2>" << endl;
	//	of << "<tr>" << endl;
	//	of << "<td colspan=2>" << endl;
	//	of << "Variables" << endl;
	//	of << "</td>" << endl;
	//	of << "</tr>" << endl;

	//of << "<tr>" << endl;
	//of << "<td>" << endl;
	//of << "Name";
	//of << "</td>" << endl;

	//of << "<td>" << endl;
	//of << "Domains";
	//of << "</td>" << endl;
	//of << "</tr>" << endl;
	//for(i = 0; i < m_drProcess.m_ptrarrRegisteredVariables.size(); i++)
	//{
	//	pVariable = m_drProcess.m_ptrarrRegisteredVariables[i];
	//	if(!pVariable)
	//		return;

	//	of << "<tr>" << endl;

	//	of << "<td>" << endl;
	//	of << pVariable->m_strName;
	//	of << "</td>" << endl;

	//	of << "<td>" << endl;
	//	if(pVariable->m_strarrDomains.size() == 0)
	//	{
	//		of << "-" << endl;
	//	}
	//	else
	//	{
	//		for(j = 0; j < pVariable->m_strarrDomains.size(); j++)
	//		{
	//			of << pVariable->m_strarrDomains[j];
	//			if(j < pVariable->m_strarrDomains.size()-1)
	//				of << ", ";
	//		}
	//	}
	//	of << "</td>" << endl;

	//	of << "</tr>" << endl;
	//}
	//of << "</table>" << endl;
	//of << "<br>" << endl;


	//for(i = 0; i < m_drProcess.m_ptrarrValues.size(); i++)
	//{
	//	pCurrentResultSet = m_drProcess.m_ptrarrValues[i];
	//	if(!pCurrentResultSet)
	//		return;

	//of << "<table border=1 cellspacing=0 cellpadding=2>" << endl;
	//	of << "<tr>" << endl;
	//	of << "<td colspan=2>" << endl;
	//	of << "Time: " + toStringFormatted<double>(pCurrentResultSet->m_dTime, -1, 15) << endl;
	//	of << "</td>" << endl;
	//	of << "</tr>" << endl;

	//	for(k = 0; k < pCurrentResultSet->m_ptrarrVariables.size(); k++)
	//	{
	//		pVariable = pCurrentResultSet->m_ptrarrVariables[k];
	//		if(!pVariable)
	//			return;

	//	of << "<tr>" << endl;
	//		of << "<td>" << endl;
	//		of << pVariable->m_strName;
	//		of << "</td>" << endl;

	//		of << "<td>" << endl;
	//			of << "<table border=0 cellspacing=2 cellpadding=2>" << endl;
	//			of << "<tr>" << endl;
	//			for(j = 0; j < pVariable->m_darrValues.size(); j++)
	//			{
	//				of << "<td>" << endl;
	//				of << toStringFormatted<double>(pVariable->m_darrValues[j], 12, 5);
	//				of << "</td>" << endl;
	//			}
	//			of << "</tr>" << endl;
	//			of << "</table>" << endl;
	//		of << "</td>" << endl;
	//	of << "</tr>" << endl;
	//	}

	//of << "</table>" << endl;
	//of << "<br>" << endl;
	//}
	//of << "</body>" << endl;
	//of << "</html>" << endl;
	//of.flush();
}


}
}
