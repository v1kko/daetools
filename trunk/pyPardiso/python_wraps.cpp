#include "python_wraps.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
using namespace dae;
using namespace boost::python;

namespace daepython
{
template<typename KEY, typename VALUE>
boost::python::dict getDictFromMapByValue(std::map<KEY,VALUE>& mapItems)
{
    boost::python::dict res;
    typename std::map<KEY,VALUE>::iterator iter;

    for(iter = mapItems.begin(); iter != mapItems.end(); iter++)
    {
        KEY   key = iter->first;
        VALUE val = iter->second;
        res[key] = val;
    }

    return res;
}

boost::python::list get_iparm(daePardisoSolver& self)
{
    boost::python::list l;

    for(size_t i = 0; i < 64; i++)
        l.append(self.iparm[i]);

    return l;
}

void set_iparm(daePardisoSolver& self, boost::python::list l_iparm)
{
    int option;
    boost::python::ssize_t n = boost::python::len(l_iparm);
    if(n != 64)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid size of iparm: " << n << "; it must be 64";
        throw e;
    }

    for(boost::python::ssize_t i = 0; i < n; i++)
        self.iparm[i] = extract<int>(l_iparm[i]);
}

boost::python::dict GetCallStats(daePardisoSolver& self)
{
    std::map<std::string, call_stats::TimeAndCount> stats = self.GetCallStats();
    return getDictFromMapByValue(stats);
}

}
