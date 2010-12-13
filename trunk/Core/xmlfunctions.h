#ifndef DAE_XML_FUNCTIONS_H
#define DAE_XML_FUNCTIONS_H

#include "io.h"
using namespace dae::io;

namespace dae 
{
namespace xml
{
// Supported functions: 
//	Unary: minus, sin, cos, tan, cotan, arcsin, arccos, arctan, root, exp, ln, log, abs
//  Binary: +, -, *, /, power
class xmlContentCreator
{
public:
//	static xmlTag_t* Constant(xmlTag_t* parent, 
//							  real_t value);
//	static xmlTag_t* Variable(xmlTag_t* parent, 
//							  string name, const vector<size_t>& domains, size_t index);
//	static xmlTag_t* TimeDerivative(xmlTag_t* parent, size_t order, string name, const vector<size_t>& domains, size_t index);
//	static xmlTag_t* PartialDerivative(xmlTag_t* parent, size_t order, string name, string domain, const vector<size_t>& domains, size_t index);
//	static xmlTag_t* Function(string fun, xmlTag_t* parent);
	
	static xmlTag_t* Constant(xmlTag_t* parent, 
							  real_t value);
	static xmlTag_t* Variable(xmlTag_t* parent, 
							  string name, 
							  const vector<string>& domains);
	static xmlTag_t* Domain(xmlTag_t* parent, 
							string name, 
							string strIndex);
	static xmlTag_t* TimeDerivative(xmlTag_t* parent, 
									size_t order, 
									string name, 
									const vector<string>& domains);
	static xmlTag_t* PartialDerivative(xmlTag_t* parent,
									   size_t order, 
									   string name, 
									   string domain, 
									   const vector<string>& domains);
};

class xmlPresentationCreator
{
public:
	static xmlTag_t* Constant(xmlTag_t* parent, 
							  real_t value);
	static xmlTag_t* Variable(xmlTag_t* parent, 
							  string name, 
							  const vector<string>& domains);
	static xmlTag_t* Domain(xmlTag_t* parent, 
							string name, 
							string strIndex);
	static xmlTag_t* TimeDerivative(xmlTag_t* parent, 
									size_t order, 
									string name, 
									const vector<string>& domains);
	static xmlTag_t* PartialDerivative(xmlTag_t* parent,
									   size_t order, 
									   string name, 
									   string domain, 
									   const vector<string>& domains);
	static void WrapIdentifier(xmlTag_t* parent, string name);
};

class textCreator
{
public:
	static string Constant(real_t value);
	static string Variable(string name, const vector<string>& domains);
	static string Domain(string name, string strIndex);
	static string TimeDerivative(size_t order, string name, const vector<string>& domains, bool bBracketsAroundName = false);
	static string PartialDerivative(size_t order, string name, string domain, const vector<string>& domains, bool bBracketsAroundName = false);
};

class latexCreator
{
public:
	static string Constant(real_t value);
	static string Variable(string name, const vector<string>& domains);
	static string Domain(string name, string strIndex);
	static string TimeDerivative(size_t order, string name, const vector<string>& domains, bool bBracketsAroundName = false);
	static string PartialDerivative(size_t order, string name, string domain, const vector<string>& domains, bool bBracketsAroundName = false);
};


}
}

#endif
