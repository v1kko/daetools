#include "stdafx.h"
#include "datareporters.h"
#include "../Core/helpers.h"

namespace dae
{
namespace datareporting
{
daeTEXTFileDataReporter::daeTEXTFileDataReporter()
{
    m_strName = "TEXTFileDataReporter";
}

daeTEXTFileDataReporter::~daeTEXTFileDataReporter()
{
}

void daeTEXTFileDataReporter::WriteDataToFile()
{
	size_t i, j, k;
	daeDataReceiverDomain *pDomain;
	daeDataReceiverVariable *pVariable;
	daeDataReceiverVariableValue* pVarValue;

	if(!IsConnected())
		return;

	of << "Activity: " << m_drProcess.m_strName <<endl;

	of << "Domains:" << endl;
	for(i = 0; i < m_drProcess.m_ptrarrRegisteredDomains.size(); i++)
	{
		pDomain = m_drProcess.m_ptrarrRegisteredDomains[i];
		if(!pDomain)
			return;

		of << pDomain->m_strName << ", ";
		of << pDomain->m_nNumberOfPoints << ", ";
        if(pDomain->m_eType == eArray)
            of << "Array" << endl;
        else if(pDomain->m_eType == eStructuredGrid)
            of << "StructuredGrid" << endl;
        else if(pDomain->m_eType == eUnstructuredGrid)
            of << "UnstructuredGrid" << endl;
        else
            of << "INVALID_TYPE" << endl;
    }
	of << endl;

	of << "Variables:" << endl;
	daePtrMap<string, daeDataReceiverVariable*>::iterator it;
	for(it = m_drProcess.m_ptrmapRegisteredVariables.begin(); it != m_drProcess.m_ptrmapRegisteredVariables.end(); it++)
	{
		pVariable = it->second;
		if(!pVariable)
			return;

		of << pVariable->m_strName << ", ";
		if(pVariable->m_ptrarrDomains.size() == 0)
		{
			of << "[-], ";
		}
		else
		{
			of << "[";
			for(j = 0; j < pVariable->m_ptrarrDomains.size(); j++)
			{
				pDomain = pVariable->m_ptrarrDomains[j];
				if(j != 0)
					of << ", ";
				of << pDomain->m_strName;
			}
			of << "], ";
		}
		of << pVariable->m_nNumberOfPoints << endl;
		for(j = 0; j < pVariable->m_ptrarrValues.size(); j++)
		{
			pVarValue = pVariable->m_ptrarrValues[j];
			of << "[" << pVarValue->m_dTime << "]: ";
			for(k = 0; k < pVariable->m_nNumberOfPoints; k++)
			{
				if(k != 0)
					of << ", ";
				of << toStringFormatted<real_t>(pVarValue->m_pValues[k], -1, 20);
			}
			of << endl;
		}
		of << endl;
	}
	of << endl;


//	for(i = 0; i < m_drProcess.m_ptrarrValues.size(); i++)
//	{
//		pCurrentResultSet = m_drProcess.m_ptrarrValues[i];
//		if(!pCurrentResultSet)
//			return;
//
//	of << "<table border=1 cellspacing=0 cellpadding=2>" << endl;
//		of << "<tr>" << endl;
//		of << "<td colspan=2>" << endl;
//		of << "Time: " + toStringFormatted<double>(pCurrentResultSet->m_dTime, -1, 15) << endl;
//		of << "</td>" << endl;
//		of << "</tr>" << endl;
//
//		for(k = 0; k < pCurrentResultSet->m_ptrarrVariables.size(); k++)
//		{
//			pVariable = pCurrentResultSet->m_ptrarrVariables[k];
//			if(!pVariable)
//				return;
//
//		of << "<tr>" << endl;
//			of << "<td>" << endl;
//			of << pVariable->m_strName;
//			of << "</td>" << endl;
//
//			of << "<td>" << endl;
//				of << "<table border=0 cellspacing=2 cellpadding=2>" << endl;
//				of << "<tr>" << endl;
//				for(j = 0; j < pVariable->m_darrValues.size(); j++)
//				{
//					of << "<td>" << endl;
//					of << toStringFormatted<double>(pVariable->m_darrValues[j], 12, 5);
//					of << "</td>" << endl;
//				}
//				of << "</tr>" << endl;
//				of << "</table>" << endl;
//			of << "</td>" << endl;
//		of << "</tr>" << endl;
//		}
//
//	of << "</table>" << endl;
//	of << "<br>" << endl;
//	}
//	of << "</body>" << endl;
//	of << "</html>" << endl;
	of.flush();
}


}
}
