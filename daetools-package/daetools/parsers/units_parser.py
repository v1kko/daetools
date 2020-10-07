
import os, sys, operator, math
import ply.lex as lex
import ply.yacc as yacc
from . import parser_objects
from .parser_objects import Number, BinaryNode, IdentifierNode, ConstantNode, unit
from .expression_parser import ExpressionParser

"""
Loading ALL units into the current module could be done by:
    current_module = sys.modules[__name__]
    supportedUnits = unit.getAllSupportedUnits()
    for key, u in supportedUnits.items():
        setattr(current_module, key, u)
    #print(globals())
ACHTUNG, ACHTUNG: it pollutes the namespace with tens of unit variables.
Therefore, it is better to use 'unit' static members:
    unit.kg
    unt.m
    unit.V
    unit.J
    ...
They are created automatically when the module with the 'unit' class is loaded into the namespace.
If that didn't work, they can be explicitly loaded by calling the class method: unit.init_base_units()
That can be done right after importing the module:
    from parser_objects import unit
    unit.init_base_units()
"""

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

debug = True

def t_error(t):
    print(t)
    print("Illegal character '{0}' found while parsing '{1}'".format(t.value[0], t.value))
    #t.lexer.skip(1)

# Parser rules:
def p_expression_1(p):
    """unit_expression : unit"""
    if debug:
        print('p_expression_1 %s' % str(p[:]))
    p[0] = p[1]

def p_expression_2(p):
    """unit_expression : LPAREN unit_expression RPAREN"""
    if debug:
        print('p_expression_2 %s' % str(p[:]))
    p[0] = p[2]

def p_expression_3(p):
    """unit_expression : unit_expression DIVIDE unit"""
    if debug:
        print('p_expression_3 %s' % str(p[:]))
    p[0] = p[1] / p[3]

def p_expression_4(p):
    """unit_expression : unit_expression TIMES unit"""
    if debug:
        print('p_expression_4 %s' % str(p[:]))
    p[0] = p[1] * p[3]

def p_unit_1(p):
    """unit :  base_unit"""
    if debug:
        print('p_unit_1 %s' % str(p[:]))
    p[0] = p[1]

def p_unit_2(p):
    """unit :  base_unit EXP constant"""
    if debug:
        print('p_unit_2 %s' % str(p[:]))
    p[0] = p[1] ** p[3]

def p_unit_3(p):
    """unit :  LPAREN base_unit EXP constant RPAREN"""
    if debug:
        print('p_unit_3 %s' % str(p[:]))
    p[0] = p[2] ** p[4]

def p_constant_1(p):
    """constant : NUMBER"""
    if debug:
        print('p_constant_1 %s' % str(p[:]))
    p[0] = Number(ConstantNode(int(p[1])))
    
def p_constant_2(p):
    """constant : LPAREN NUMBER RPAREN"""
    if debug:
        print('p_constant_2 %s' % str(p[:]))
    p[0] = Number(ConstantNode(int(p[2])))
    
def p_constant_3(p):
    """constant : FLOAT"""
    if debug:
        print('p_constant_3 %s' % str(p[:]))
    p[0] = Number(ConstantNode(float(p[1])))
    
def p_constant_4(p):
    """constant : LPAREN FLOAT RPAREN"""
    if debug:
        print('p_constant_4 %s' % str(p[:]))
    p[0] = Number(ConstantNode(float(p[2])))
    
def p_base_unit_1(p):
    """base_unit : BASE_UNIT  """
    if debug:
        print('p_base_unit_1 %s' % str(p[:]))
    p[0] = Number(IdentifierNode(p[1]))
    
def p_error(p):
    raise Exception("Syntax error at '%s' (%s)" % (p.value, p))

class UnitsParser:
    """
    The parser supports the following math. operators: *, /, ** in expressions like the following:
       'kg', 'kg * m**2', 'm/s', 'm/s**2' ...
    To evaluate the AST tree objects that support the above operators should be provided in dictBaseUnits.
    Dictionary 'dictBaseUnits' should contain pairs of the following type:
        unit-name:unit-object (unit-objects must define *, /, ** operators)
    Instances of the 'unit' class from 'parser_objects.py' should be used as unit-objects.
    dictBaseUnits with the most common base/derived units can be obtained by calling the 'unit' class-method:
        dictBaseUnits = unit.getAllSupportedUnits()
    """
    def __init__(self, dictUnits = None):
        self.lexer  = lex.lex()
        self.parser = yacc.yacc() #(write_tables = 0)
        self.parseResult = None
        self.dictBaseUnits = dictUnits
        
    def parse(self, expression):
        self.parseResult = self.parser.parse(expression, debug = 0)
        return self.parseResult

    def parse_and_evaluate(self, expression):
        self.parse(expression)
        return self.evaluate()

    def parse_to_latex(self, expression):
        self.parse(expression)
        return self.toLatex()

    def parse(self, expression):
        self.parseResult = self.parser.parse(expression, debug = 0)
        return self.parseResult

    def toLatex(self):
        if self.parseResult is None:
            raise RuntimeError('expression not parsed yet')
        return self.parseResult.toLatex()

    def evaluate(self):
        if self.parseResult is None:
            raise RuntimeError('expression not parsed yet')
        if self.dictBaseUnits is None:
            raise RuntimeError('dictBaseUnits not set')

        node = None
        if isinstance(self.parseResult, Number):
            node = self.parseResult.Node
        else:
            raise RuntimeError('Invalid parse result type')

        result = node.evaluate(self.dictBaseUnits, None)
        return result


def testConsistency(parser, expression, expected_units):
    parse_res    = parser.parse(expression)
    latex_res    = parser.toLatex()
    print('Expression: ' + expression)
    #print('NodeTree:\n', repr(parse_res))
    print('Parse result: ', str(parse_res))
    try:
        eval_res = parser.evaluate()
        if eval_res.units.base_unit != expected_units.base_unit:
            print('Units consistency failed: evaluated {0}; expected {1}\n'.format(eval_res.units, expected_units))
        else:
            print('Units consistency: OK [{0} == {1}]\n'.format(eval_res.units, expected_units))
    except Exception as e:
        print('Units consistency failed: {0}\n'.format(str(e)))
        
def testExpression(parser, expression, do_evaluation = True):
    parse_res    = parser.parse(expression)
    latex_res    = parser.toLatex()
    print('Expression: ' + expression)
    #print('NodeTree:\n', repr(parse_res))
    print('Parse result: ', str(parse_res))
    print('Latex: ', latex_res)
    if do_evaluation:
        eval_res = parser.evaluate()
        print('Evaluate result String: {0}'.format(eval_res.toString()))
        print('Evaluate result Latex: {0}'.format(eval_res.toLatex()))
    print(' ')

    return parse_res, latex_res, eval_res

def testUnitsConsistency():
    dictIdentifiers = {}
    dictFunctions   = {}

    # Define some 'quantity' objects (they have 'value' and 'units')
    y   = 15   * unit.mm
    x1  = 1.0  * unit.m
    x2  = 0.2  * unit.km
    x3  = 15   * unit.N
    x4  = 1.25 * unit.kJ
    print('y  = {0} ({1} {2})'.format(y,  y.value_in_SI_units,  y.units.base_unit.toString()))
    print('x1 = {0} ({1} {2})'.format(x1, x1.value_in_SI_units, x1.units.base_unit.toString()))
    print('x2 = {0} ({1} {2})'.format(x2, x2.value_in_SI_units, x2.units.base_unit.toString()))
    print('x3 = {0} ({1} {2})'.format(x3, x3.value_in_SI_units, x3.units.base_unit.toString()))
    print('x4 = {0} ({1} {2})'.format(x4, x4.value_in_SI_units, x4.units.base_unit.toString()))

    print('x1({0}) == x2({1}) ({2})'.format(x1, x2, x1 == x2))
    print('x1({0}) != x2({1}) ({2})'.format(x1, x2, x1 != x2))
    print('x1({0}) > x2({1}) ({2})'.format(x1, x2, x1 > x2))
    print('x1({0}) >= x2({1}) ({2})'.format(x1, x2, x1 >= x2))
    print('x1({0}) < x2({1}) ({2})'.format(x1, x2, x1 < x2))
    print('x1({0}) <= x2({1}) ({2})'.format(x1, x2, x1 <= x2))
    
    # quantity in [m]
    z = 1 * unit.m
    print(z)
    z.value = 12.4 * unit.mm # set a new value given in [mm]
    print(z)
    z.value = 0.32 * unit.km # set a new value given in [km]
    print(z)
    z.value = 1 # set a new value in units in the quantity object, here in [m]
    print(z)
    
    # Define identifiers for the parser
    dictIdentifiers['pi'] = math.pi
    dictIdentifiers['e']  = math.e
    dictIdentifiers['y']  = y
    dictIdentifiers['x1'] = x1
    dictIdentifiers['x2'] = x2
    dictIdentifiers['x3'] = x3
    dictIdentifiers['x4'] = x4

    # Define math. functions for the parser
    dictFunctions['sin']   = parser_objects.sin
    dictFunctions['cos']   = parser_objects.cos
    dictFunctions['tan']   = parser_objects.tan
    dictFunctions['asin']  = parser_objects.asin
    dictFunctions['acos']  = parser_objects.acos
    dictFunctions['atan']  = parser_objects.atan
    dictFunctions['sinh']  = parser_objects.sinh
    dictFunctions['cosh']  = parser_objects.cosh
    dictFunctions['tanh']  = parser_objects.tanh
    dictFunctions['asinh'] = parser_objects.asinh
    dictFunctions['acosh'] = parser_objects.cosh
    dictFunctions['atanh'] = parser_objects.atanh
    dictFunctions['log10'] = parser_objects.log10
    dictFunctions['log']   = parser_objects.log
    dictFunctions['sqrt']  = parser_objects.sqrt
    dictFunctions['exp']   = parser_objects.exp
    dictFunctions['floor'] = parser_objects.floor
    dictFunctions['ceil']  = parser_objects.ceil
    dictFunctions['pow']   = parser_objects.pow

    #print('Identifiers:\n', dictIdentifiers, '\n')
    #print('Functions:\n', dictFunctions, '\n')

    parser = ExpressionParser(dictIdentifiers, dictFunctions)

    testConsistency(parser, 'x1 * x3', unit.kJ)             # OK
    testConsistency(parser, 'x1 - x3', None)                # Fail
    testConsistency(parser, 'x1 * y', unit.m**2)            # OK
    testConsistency(parser, '1 + x1/x2 + x1*x3/x4', unit()) # OK
    
def testUnitsParser():
    dictBaseUnits = unit.getAllSupportedUnits()
    parser        = UnitsParser(dictBaseUnits)

    testExpression(parser, 'kg*V/mV^2')
    testExpression(parser, 'kg*V/(A^2 * J^3)')

if __name__ == "__main__":
    testUnitsParser()
    #testUnitsConsistency()

