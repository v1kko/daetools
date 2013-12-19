#include "python_wraps.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
//#define NO_IMPORT_ARRAY
//#include <noprefix.h>
using namespace std;
using namespace boost;
using namespace boost::python;
  
namespace daepython
{
real_t daeArray_GetItem(daeArray<real_t>& self, size_t index)
{
    return self.GetItem(index);
}

boost::python::list daeArray_GetValues(daeArray<real_t>& self)
{
    boost::python::list l;
   
    for(size_t i = 0; i < self.GetSize(); i++)
        l.append(self.GetItem(i));

    return l;
}

real_t daeDenseMatrix_GetItem(daeDenseMatrix& self, size_t i, size_t j)
{
    return self.GetItem(i, j);
}


}
