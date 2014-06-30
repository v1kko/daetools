#ifndef DAE_BASE_DATA_REPORTERS_RECEIVERS_H
#define DAE_BASE_DATA_REPORTERS_RECEIVERS_H

#include "../Core/datareporting.h"

namespace dae
{
namespace datareporting
{
daeDataReportingClassFactory_t* daeCreateDataReportingClassFactory(void);

daeDataReporter_t* daeCreateBlackHoleDataReporter(void);
daeDataReporter_t* daeCreateNoOpDataReporter(void);
daeDataReporter_t* daeCreateTextDataReporter(void);
daeDataReporter_t* daeCreateHTMLDataReporter(void);
daeDataReporter_t* daeCreateTCPIPDataReporter(void);
//daeDataReceiver_t* daeCreateTCPIPDataReceiverServer(int nPort);
daeDataReporter_t* daeCreateDelegateDataReporter(const std::vector<daeDataReporter_t*>& ptrarrDataReporters);

//void daeCreateHybridDataReporterReceiver(daeDataReceiver_t** pReceiver, daeDataReporter_t** pReporter);

}
}

#endif
