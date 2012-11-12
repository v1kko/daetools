#include "dealii_equation_generator.h"
#include "../Core/nodes.h"

adouble adouble_(double val)
{
    adouble a(val);
    a.node = adNodePtr(new dae::core::adConstantNode(val));
    a.setGatherInfo(true);
    return a;
}

namespace dae 
{
template<>
std::string toStringFormatted(adouble value, 
							  std::streamsize width, 
							  std::streamsize precision, 
							  bool scientific,
							  bool strip_if_integer)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

}

/*
DEAL_II_NAMESPACE_OPEN
namespace internal
{
    template <>
    void print (const adouble &t,
                const char *format)
    {
      if (format != 0)
        std::printf (format, t.getValue());
      else
        std::printf (" %5.2f", t.getValue());
    }
}
DEAL_II_NAMESPACE_CLOSE

namespace std
{
void printf(const char *format, const adouble& a)
{
    printf(format, a.getValue());
}

//adouble exp(const adouble &a) {return dae::core::exp( adouble(a) );}
//adouble log(const adouble &a) {return dae::core::log( adouble(a) );}
//adouble sqrt(const adouble &a) {return dae::core::sqrt( adouble(a) );}
//adouble sin(const adouble &a) {return dae::core::sin( adouble(a) );}
//adouble cos(const adouble &a) {return dae::core::cos( adouble(a) );}
//adouble tan(const adouble &a) {return dae::core::tan( adouble(a) );}
//adouble asin(const adouble &a) {return dae::core::asin( adouble(a) );}
//adouble acos(const adouble &a) {return dae::core::acos( adouble(a) );}
//adouble atan(const adouble &a) {return dae::core::atan( adouble(a) );}

//adouble sinh(const adouble &a) {return dae::core::sinh( adouble(a) );}
//adouble cosh(const adouble &a) {return dae::core::cosh( adouble(a) );}
//adouble tanh(const adouble &a) {return dae::core::tanh( adouble(a) );}
//adouble asinh(const adouble &a) {return dae::core::asinh( adouble(a) );}
//adouble acosh(const adouble &a) {return dae::core::acosh( adouble(a) );}
//adouble atanh(const adouble &a) {return dae::core::atanh( adouble(a) );}
//adouble atan2(const adouble &a, adouble &b) {return dae::core::atan2(adouble(a), adouble(b));}

////adouble pow(const adouble &a, real_t v) {return dae::core::pow(adouble(a), v);}
//adouble pow(const adouble &a, const adouble &b) {return dae::core::pow(adouble(a), adouble(b));}
//adouble log10(const adouble &a) {return dae::core::log10( adouble(a) );}

//adouble ceil(const adouble &a) {return dae::core::ceil( adouble(a) );}
//adouble floor(const adouble &a) {return dae::core::floor( adouble(a) );}

//adouble abs(const adouble &a) {return dae::core::fabs( adouble(a) );}
//adouble fabs(const adouble &a) {return dae::core::fabs( adouble(a) );}
//adouble max(const adouble &a, const adouble &b) {return dae::core::max(adouble(a), adouble(b));}
//adouble min(const adouble &a, const adouble &b) {return dae::core::min(adouble(a), adouble(b));}

}

*/
