#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#define NO_IMPORT_ARRAY
#include <noprefix.h>
using namespace std;
using namespace boost;
using namespace boost::python;
  
namespace daepython
{

void SetOptionS(daeBONMINSolver& self, const string& strOptionName, const string& strValue)
{
    self.SetOption(strOptionName, strValue);
}
    
void SetOptionF(daeBONMINSolver& self, const string& strOptionName, real_t dValue)
{
    self.SetOption(strOptionName, dValue);
}

void SetOptionI(daeBONMINSolver& self, const string& strOptionName, int iValue)
{
    self.SetOption(strOptionName, iValue);
}   

}
