#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#define NO_IMPORT_ARRAY
#include <noprefix.h>
using namespace std;
using namespace boost;
using namespace boost::python;
  
// Sets all elements to zero
void clearMatrix(FullMatrix<double>& m)
{
    m = 0;
}

