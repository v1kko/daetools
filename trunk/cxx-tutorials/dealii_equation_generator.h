#ifndef DAE_DEALII_AUX_H
#define DAE_DEALII_AUX_H

#include "../dae_develop.h"
#include "../variable_types.h"
namespace vt = variable_types;

using units_pool::m;
using units_pool::kg;
using units_pool::K;
using units_pool::J;
using units_pool::W;
using units_pool::s;

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

class modTutorial1 : public daeModel
{
daeDeclareDynamicClass(modTutorial1)
public:
    size_t Np;
    adoubleCSRMatrix&     A;
    std::vector<adouble>& x;
    std::vector<adouble>& b;
    daeDomain fem;
    daeVariable T;

    modTutorial1(size_t _Np, adoubleCSRMatrix& _A, std::vector<adouble>& _x, std::vector<adouble>& _b,
                 string strName, daeModel* pParent = NULL, string strDescription = "") 
      : daeModel(strName, pParent, strDescription),
        Np(_Np),
        A(_A), x(_x), b(_b),
        fem("fem", this, unit(),   "FEM domain"),
        T("T",   vt::no_t, this, "Temperature of the plate, -", &fem)
    {
    }

    void DeclareEquations(void)
    {
        int counter;
        daeEquation* eq;
        adouble res;
        
        //A.Print();
        for(size_t i = 0; i < Np; i++)
        {
            res = 0;
            //std::cout << (boost::format("k0= %1% to %2%") % A.IA[i] % A.IA[i+1]).str() << std::endl;

            counter = 0;
            for(int k = A.IA[i]; k < A.IA[i+1]; k++)
            {
                //std::cout << (boost::format("JA[%1%] = %2%") % k % A.JA[k]).str() << std::endl;
                if(counter == 0)
                    res = A.A[k] * T(A.JA[k]);
                else
                    res = res + A.A[k] * T(A.JA[k]);
                counter++;
            }
            eq = CreateEquation("Element_" + toString(i), "");
            eq->SetResidual( res - b[i] );
        }
    }
};

class simTutorial1 : public daeSimulation
{
public:
	modTutorial1 M;
	
public:
	simTutorial1(size_t _Np, adoubleCSRMatrix& _A, std::vector<adouble>& _x, std::vector<adouble>& _b) 
        : M(_Np, _A, _x, _b, "dealii")
	{
		SetModel(&M);
        M.SetDescription("dealii description");
	}

public:
	void SetUpParametersAndDomains(void)
	{
		M.fem.CreateArray(M.Np);
	}

	void SetUpVariables(void)
	{
	}
};


/*
#include <deal.II/lac/vector.templates.h>//
#include <deal.II/base/numbers.h>
using namespace dealii;
using namespace numbers;

#ifndef DAE_ADOUBLE_CAST_TO_DOUBLE
#define DAE_ADOUBLE_CAST_TO_DOUBLE
#endif

namespace std
{
void printf(const char *format, const adouble& a);
//adouble exp(const adouble &a) ;
//adouble log(const adouble &a);
//adouble sqrt(const adouble &a);
//adouble sin(const adouble &a);
//adouble cos(const adouble &a);
//adouble tan(const adouble &a);
//adouble asin(const adouble &a);
//adouble acos(const adouble &a);
//adouble atan(const adouble &a);

//adouble sinh(const adouble &a);
//adouble cosh(const adouble &a);
//adouble tanh(const adouble &a);
//adouble asinh(const adouble &a);
//adouble acosh(const adouble &a);
//adouble atanh(const adouble &a);
//adouble atan2(const adouble &a, const adouble &b);

////adouble pow(const adouble &a, real_t v);
//adouble pow(const adouble &a, const adouble &b);
//adouble log10(const adouble &a);

//adouble ceil(const adouble &a);
//adouble floor(const adouble &a);

//adouble abs(const adouble &a);
//adouble fabs(const adouble &a);
//adouble max(const adouble &a, adouble &b);
//adouble min(const adouble &a, adouble &b);
}

DEAL_II_NAMESPACE_OPEN
namespace internal
{
    template <>
    void print (const adouble &t,
                const char *format);
}

namespace numbers
{
template <>
struct NumberTraits<adouble>
{
    static const bool is_complex = false;
    typedef adouble real_type;
   
    static adouble conjugate (const adouble &x)
    {
      return adouble();
    }
    static adouble abs_square (const adouble &x)
    {
      return adouble();
    }
    static adouble abs (const adouble &x)
    {
      return adouble();
    }
};



//template <>
//adouble NumberTraits<adouble>::conjugate (const adouble &x)
//{
//   return dae::core::adouble();
//}

//template <>
//typename NumberTraits<adouble>::real_type
//NumberTraits<adouble>::abs (const adouble &x)
//{
//  return abs(x);
//}


//template <>
//typename NumberTraits<adouble>::real_type
//NumberTraits<adouble>::abs_square (const adouble &x)
//{
//  return adouble();
//}

}
DEAL_II_NAMESPACE_CLOSE
*/

#endif // AUX_H
