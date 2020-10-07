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

