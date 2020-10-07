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

std::string vector_csNodePtr__str__(std::vector<cs::csNodePtr>& self)
{
    std::string res = "[";
    for(int i = 0; i < self.size(); i++)
        res += (i == 0 ? "" : ", ") + self[i]->ToLatex();
    res += "]";
    return res;
}

}
