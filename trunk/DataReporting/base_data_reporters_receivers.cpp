#include "stdafx.h"
#include "base_data_reporters_receivers.h"
#include "datareporters.h"

namespace dae
{
namespace datareporting
{
daeDataReporter_t* daeCreateBlackHoleDataReporter(void)
{
	return new daeBlackHoleDataReporter;
}
	
daeDataReporter_t* daeCreateTextDataReporter(void)
{
	return new daeTEXTFileDataReporter;
}

daeDataReporter_t* daeCreateHTMLDataReporter(void)
{
	return new daeHTMLFileDataReporter;
}

daeDataReporter_t* daeCreateXMLDataReporter(void)
{
	return new daeXMLFileDataReporter;
}

daeDataReporter_t* daeCreateTCPIPDataReporter(void)
{
	return new daeTCPIPDataReporter;
}

//daeDataReceiver_t* daeCreateTCPIPDataReceiverServer(int nPort)
//{
//	return new daeTCPIPDataReceiverServer(nPort);
//}

daeDataReporter_t* daeCreateDelegateDataReporter(const vector<daeDataReporter_t*>& ptrarrDataReporters)
{
	vector<daeDataReporter_t*>::const_iterator it;
	
	daeDelegateDataReporter* pReporter = new daeDelegateDataReporter;
	for(it = ptrarrDataReporters.begin(); it != ptrarrDataReporters.end(); it++)
		pReporter->AddDataReporter(*it);
	
	return pReporter;
}

void daeCreateHybridDataReporterReceiver(daeDataReceiver_t** pReceiver, daeDataReporter_t** pReporter)
{
	daeHybridDataReporterReceiver* _pHybrid = new daeHybridDataReporterReceiver;
	*pReceiver = _pHybrid;
	*pReporter = _pHybrid;
}

//void daeCreateLinkedDataReporterReceiver(daeDataReceiver_t** pReceiver, daeDataReporter_t** pReporter)
//{
//	daeHybridDataReporterReceiver* _pHybrid   = new daeHybridDataReporterReceiver;
//	daeDelegateDataReporter*       _pReporter = new daeDelegateDataReporter;
//	_pReporter->AddDataReporter(_pHybrid);
//
//	*pReceiver = _pHybrid;
//	*pReporter = _pReporter;
//}


}
}
