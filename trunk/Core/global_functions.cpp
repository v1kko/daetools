#include "stdafx.h"
#include "coreimpl.h"
#include "xmlfile.h"

namespace dae
{
namespace core
{
void daeModel::SaveModelReport(const string& strFileName, bool bRecursively) const
{
	if(strFileName.empty())
		daeDeclareAndThrowException(exInvalidCall); 
	
	xml::xmlFile file;
	file.SetDocumentType(xml::eMathML);
	file.SetXSLTFileName("dae-tools.xsl");
	io::xmlTag_t* pRootTag = file.GetRootTag();
	pRootTag->SetName(string("Model"));
	this->Save(pRootTag);
	file.Save(strFileName);

	if(bRecursively)
	{
		daeModel* pModel;
		for(size_t i = 0; i < m_ptrarrModels.size(); i++)
		{
			pModel = m_ptrarrModels[i];
			pModel->SaveModelReport(pModel->GetName() + ".xml", bRecursively);
		}
	}
}

void daeModel::SaveRuntimeModelReport(const string& strFileName) const
{
	xml::xmlFile file;
	
	if(strFileName.empty())
		daeDeclareAndThrowException(exInvalidCall); 
		
	file.SetDocumentType(xml::eMathML);
	file.SetXSLTFileName("dae-tools-rt.xsl");
	io::xmlTag_t* pRootTag = file.GetRootTag();
	pRootTag->SetName(string("Model"));
	this->SaveRuntime(pRootTag);
	file.Save(strFileName);
}

}
}
