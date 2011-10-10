#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

#include "parser_objects.h"
#include "units.h"

namespace parser
{
namespace qi    = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
/*
template <typename Iterator>
struct roman : qi::grammar<Iterator>
{
	roman() : roman::base_type(start)
	{
		using qi::eps;
		using qi::lit;
		using qi::_val;
		using qi::_1;
		using ascii::char_;
		
		start = eps [_val = 0] >>
		                          (
		                              +lit('M')       [_val += 1000]
		                              )
		                          ;
//		constant = float_ 
//		p[0] = Number(ConstantNode(int(p[1])))
//		"""constant : LPAREN NUMBER RPAREN"""
//		p[0] = Number(ConstantNode(int(p[2])))
//		"""constant : FLOAT"""
//		p[0] = Number(ConstantNode(float(p[1])))
//		"""constant : LPAREN FLOAT RPAREN"""
//		p[0] = Number(ConstantNode(float(p[2])))
//		"""base_unit : BASE_UNIT  """
//		p[0] = Number(IdentifierNode(p[1]))
	}
	
	qi::rule<Iterator> unit_expression, 
	                   unit,
	                   base_unit,
	                   constant,
	                   start;
};
*/
}

/*
tokens = [ 'BASE_UNIT',
           'NUMBER', 'FLOAT',
           'TIMES', 'DIVIDE', 'EXP',
           'LPAREN','RPAREN'
         ]

t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_EXP     = r'\^'

t_LPAREN  = r'\('
t_RPAREN  = r'\)'

t_NUMBER = r'(\+|-)?\d+'
t_FLOAT  = r'((\d+)(\.\d+)(e(\+|-)?(\d+))? | (\d+)e(\+|-)?(\d+))([lL]|[fF])?'

t_ignore = " \t\n"

t_BASE_UNIT = r'[a-zA-Z_][a-zA-Z_0-9]*'

def t_error(t):
    print("Illegal character '{0}' found while parsing '{1}'".format(t.value[0], t.value))
    #t.lexer.skip(1)

# Parser rules:
def p_expression_1(p):
    """unit_expression : unit"""
    p[0] = p[1]

def p_expression_2(p):
    """unit_expression : unit_expression DIVIDE unit"""
    p[0] = p[1] / p[3]

def p_expression_3(p):
    """unit_expression : unit_expression TIMES unit"""
    p[0] = p[1] * p[3]

def p_unit_1(p):
    """unit :  base_unit"""
    p[0] = p[1]

def p_unit_2(p):
    """unit :  base_unit EXP constant"""
    p[0] = p[1] ** p[3]

def p_unit_3(p):
    """unit :  LPAREN base_unit EXP constant RPAREN"""
    p[0] = p[2] ** p[4]

def p_constant_1(p):
    """constant : NUMBER"""
    p[0] = Number(ConstantNode(int(p[1])))
    
def p_constant_2(p):
    """constant : LPAREN NUMBER RPAREN"""
    p[0] = Number(ConstantNode(int(p[2])))
    
def p_constant_3(p):
    """constant : FLOAT"""
    p[0] = Number(ConstantNode(float(p[1])))
    
def p_constant_4(p):
    """constant : LPAREN FLOAT RPAREN"""
    p[0] = Number(ConstantNode(float(p[2])))
    
def p_base_unit_1(p):
    """base_unit : BASE_UNIT  """
    p[0] = Number(IdentifierNode(p[1]))
    
def p_error(p):
    raise Exception("Syntax error at '%s'" % p.value)
*/

///////////////////////////////////////////////////////////////////////////////
//  Main program
///////////////////////////////////////////////////////////////////////////////
using namespace parser_objects;
using namespace units;
namespace u  = units::units_pool;
namespace bu = units::base_units_pool;

int counter = 1;
#define doTest(EXPRESSION) try{ std::cout << counter++ << ")  "; EXPRESSION;} catch(std::runtime_error& e){std::cout << e.what() << std::endl;}

void test_base_units()
{
	doTest( std::cout << "kg * m = " << (bu::kg * bu::m) << std::endl )
	doTest( std::cout << "kg / m = " << (bu::kg / bu::m) << std::endl )
	doTest( std::cout << "kg ^ 2.3 = " << (bu::kg ^ 2.3) << std::endl )
	
	doTest( std::cout << "0.001 * kg * m = " << (0.001 * bu::kg * bu::m) << std::endl )
	doTest( std::cout << "1000 * kg / m = " << (1000 * bu::kg / bu::m) << std::endl )
	doTest( std::cout << "200 * (kg ^ 2.3) = " << (200 * (bu::kg ^ 2.3)) << std::endl )

	doTest( std::cout << "2 / m = " << (2 / bu::m) << std::endl )
	doTest( std::cout << "m / 2 = " << (bu::m / 2) << std::endl )
	
	doTest( std::cout << "kg ^ m = " << (bu::kg ^ bu::m) << std::endl )

	doTest( std::cout << "kg + m = " << (bu::kg + bu::m) << std::endl )
	
	doTest( std::cout << "kg - m = " << (bu::kg - bu::m) << std::endl )
	
	doTest( std::cout << "1 + m = " << (1 + bu::m) << std::endl )
	
	doTest( std::cout << "m + 1 = " << (bu::m + 1) << std::endl )
	
	doTest( std::cout << "1 - m = " << (1 - bu::m) << std::endl )
	
	doTest( std::cout << "m - 1 = " << (bu::m - 1) << std::endl )
}

void test_units()
{
	// Print the whole pool of base units
	//for(std::map<std::string, base_unit>::iterator iter = unit::__base_units__.begin(); iter != unit::__base_units__.end(); iter++)
	//	std::cout << (*iter).first << " = " << (*iter).second.toString() << std::endl;

	doTest( std::cout << "kg * m * s = " << (u::kg * u::m * u::s ) << std::endl )
	doTest( std::cout << "kg * s / m = " << (u::kg * u::s / u::m) << std::endl )
	doTest( std::cout << "kg ^ 2.3 = " << (u::kg ^ 2.3) << std::endl )
	
	doTest( std::cout << "0.001 * kg * m = " << (0.001 * (u::kg * u::m)) << std::endl )
	doTest( std::cout << "1000 * kg / m = " << (1000 * (u::kg / u::m)) << std::endl )
	doTest( std::cout << "200 * (kg ^ 2.3) = " << (200 * (u::kg ^ 2.3)) << std::endl )

	doTest( std::cout << "2 / u::m = " << (2 / u::m) << std::endl )
	doTest( std::cout << "u::m / 2 = " << (u::m / 2) << std::endl )

	doTest( std::cout << "0.1 * km + 15 * m = " << (0.1 * u::km + 15 * u::m) << std::endl )

	doTest( std::cout << "kg ^ m = " << (u::kg ^ u::m) << std::endl )

	doTest( std::cout << "kg + m = " << (u::kg + u::m) << std::endl )
	
	doTest( std::cout << "kg - m = " << (u::kg - u::m) << std::endl )
	
	doTest( std::cout << "1 + m = " << (1 + u::m) << std::endl )
	
	doTest( std::cout << "m + 1 = " << (u::m + 1) << std::endl )
	
	doTest( std::cout << "1 - m = " << (1 - u::m) << std::endl )
	
	doTest( std::cout << "m - 1 = " << (u::m - 1) << std::endl )
}

void test_quantities()
{
	doTest( std::cout << "1.0 * m   = " << (1.0 * u::m) << std::endl )
	doTest( std::cout << "0.2 * km  = " << (0.2 * u::km) << std::endl )
	doTest( std::cout << "15 * N    = " << (15 * u::N) << std::endl )
	doTest( std::cout << "1.25 * kJ = " << (1.25 * u::kJ) << std::endl )

	doTest( std::cout << "(1.0 * m) * (0.2 * km) = "  << ((1.0 * u::m) * (0.2 * u::km)) << std::endl )
	doTest( std::cout << "(15 * N)  ^ (1.25 * kJ) = " << ((15 * u::N) ^ (1.25 * u::kJ)) << std::endl )
	doTest( std::cout << "(1.0 * m) + (0.2 * km) = "  << ((1.0 * u::m) + (0.2 * u::km)) << std::endl )
	doTest( std::cout << "(15 * N)  - (1.25 * kJ) = " << ((15 * u::N) - (1.25 * u::kJ)) << std::endl )

	doTest( std::cout << "(1.0 * m) * 1.5 = " << ((1.0 * u::m) * 1.5) << std::endl )
	doTest( std::cout << "(15 * N)  ^ 1.5 = " << ((15 * u::N) ^ 1.5)  << std::endl )
	doTest( std::cout << "(1.0 * m) + 1.5 = " << ((1.0 * u::m) + 1.5) << std::endl )
	doTest( std::cout << "(15 * N)  - 1.5 = " << ((15 * u::N) - 1.5)  << std::endl )
	
	doTest( std::cout << "1.5 * (0.2  * km) = " << (1.5 * (0.2 * u::km))  << std::endl )
	doTest( std::cout << "1.5 ^ (1.25 * kJ) = " << (1.5 ^ (1.25 * u::kJ)) << std::endl )
	doTest( std::cout << "1.5 + (0.2  * km) = " << (1.5 + (0.2 * u::km))  << std::endl )
	doTest( std::cout << "1.5 - (1.25 * kJ) = " << (1.5 - (1.25 * u::kJ)) << std::endl )

	doTest( std::cout << "(15 * N) / (1.2 * N) - 1.5 = " << ((15 * u::N) / (1.2 * u::N) - 1.5)  << std::endl )
	doTest( std::cout << "(15 * N) / (1.2 * N) + 1.5 = " << ((15 * u::N) / (1.2 * u::N) + 1.5)  << std::endl )

	doTest( std::cout << "(15 * J) / (1.2 * kJ) - 1.5 = " << ((15 * u::J) / (1.2 * u::kJ) - 1.5)  << std::endl )
	doTest( std::cout << "(15 * J) / (1.2 * kJ) + 1.5 = " << ((15 * u::J) / (1.2 * u::kJ) + 1.5)  << std::endl )

	quantity y = 1.23 * (u::km * u::mV * (u::ms ^ 2)); 
    doTest( std::cout << (boost::format("1.23 * (km * mV * (ms ^ 2)) = %1% (%2%) (%3%)") % y % y.getUnits().getBaseUnit() % y.getValueInSIUnits()).str() << std::endl )
}

int main()
{
	typedef boost::shared_ptr< Node<double> >    node_ptr;
	typedef boost::shared_ptr< Node<base_unit> > base_unit_ptr;
	
//	node_ptr c1(new ConstantNode<double>(5.7));
//	node_ptr c2(new ConstantNode<double>(6.0));
//	node_ptr id(new IdentifierNode<double>("x4"));
//	Number<double> x(c1);
//	Number<double> y(c2);
//	Number<double> z(id);
//	std::cout << (x + y * z).node->toString() << std::endl;

//	std::cout << "unit kg = " << kg.toString() << std::endl;
//	std::cout << (kg / m).toString() << std::endl;
//	std::cout << (kg * (m^-1)).toString() << std::endl;
	
//	base_unit_ptr id_(new IdentifierNode<base_unit>("mV"));
//	Number<base_unit> mV(id_);
//	std::cout << (mV * mV).node->toString() << std::endl;

	std::cout << std::endl << "********************** test_base_units *********************" << std::endl;
	test_base_units();
	std::cout << std::endl << "************************ test_units ************************" << std::endl;
	test_units();
	std::cout << std::endl << "********************** test_quantities *********************" << std::endl;
	test_quantities();
}

