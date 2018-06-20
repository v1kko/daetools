#include "python_wraps.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
//#define NO_IMPORT_ARRAY
//#include <noprefix.h>
using namespace std;
using namespace boost;
using namespace boost::python;

namespace daepython
{
boost::python::dict GetCallStats(daeSimulation& self)
{
    std::map<std::string, call_stats::TimeAndCount> stats = self.GetCallStats();
    return getDictFromMapByValue(stats);
}
}
