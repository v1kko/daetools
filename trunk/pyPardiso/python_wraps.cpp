#include "python_wraps.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
using namespace dae;
using namespace boost::python;

namespace daepython
{
boost::python::list daePardisoSolver_get_iparm(daePardisoSolver& self)
{
    boost::python::list l;

    for(size_t i = 0; i < 64; i++)
        l.append(self.iparm[i]);

    return l;
}

void daePardisoSolver_set_iparm(daePardisoSolver& self, boost::python::list l_iparm)
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

}
