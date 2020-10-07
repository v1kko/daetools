#ifndef DAE_FEM_COMMON_H
#define DAE_FEM_COMMON_H

#include "../dae_develop.h"
#include "../IDAS_DAESolver/dae_array_matrix.h"
using dae::core::adouble;
using namespace dae::solver;

typedef daeCSRMatrix<adouble, int> adoubleCSRMatrix;
adouble adouble_(double val);

namespace dae 
{
template<>
std::string toStringFormatted(adouble value, 
							  std::streamsize width, 
							  std::streamsize precision, 
							  bool scientific,
							  bool strip_if_integer);
}


#endif
