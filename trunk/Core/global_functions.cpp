#include "stdafx.h"
#include "coreimpl.h"
#include "xmlfile.h"

namespace dae
{
namespace core
{
void daeModel::SaveModelReport(const string& strFileName) const
{
    if(strFileName.empty())
        daeDeclareAndThrowException(exInvalidCall);

    xml::xmlFile file;
    file.SetDocumentType(xml::eXML);
    file.SetXSLTFileName("dae-tools.xsl");
    io::xmlTag_t* pRootTag = file.GetRootTag();
    pRootTag->SetName(string("Model"));
    this->Save(pRootTag);
    file.Save(strFileName);
}

void daeModel::SaveRuntimeModelReport(const string& strFileName) const
{
    if(strFileName.empty())
        daeDeclareAndThrowException(exInvalidCall);

    xml::xmlFile file;
    file.SetDocumentType(xml::eXML);
    file.SetXSLTFileName("dae-tools-rt.xsl");
    io::xmlTag_t* pRootTag = file.GetRootTag();
    pRootTag->SetName(string("Model"));
    this->SaveRuntime(pRootTag);
    file.Save(strFileName);
}

}
}
