import os, sys, operator
import ply.lex as lex
import ply.yacc as yacc
import parser_objects
from parser_objects import Number, BinaryNode, IdentifierNode, ConstantNode, unit

tokens = [ 'UNIT_NAME',
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

t_ignore = " \t"

t_UNIT_NAME = r'[a-zA-Z_][a-zA-Z_0-9]*'

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_error(t):
    print "Illegal character '%s'" % t.value[0]
    t.lexer.skip(1)

# Parser rules:
# expression:
def p_expression_1(p):
    'expression : unit'
    p[0] = p[1]

def p_expression_2(p):
    'expression : expression DIVIDE unit'
    p[0] = p[1] / p[3]

def p_expression_3(p):
    'expression : expression TIMES unit'
    p[0] = p[1] * p[3]

# unit-expression
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
    """base_unit : UNIT_NAME  """
    p[0] = Number(IdentifierNode(p[1]))
    
def p_error(p):
    raise Exception("Syntax error at '%s'" % p.value)

# Parser class
class UnitParser:
    def __init__(self, dictUnits = None, dictFunctions = None):
        self.lexer  = lex.lex()
        self.parser = yacc.yacc() #(write_tables = 0)
        self.parseResult = None
        self.dictUnits     = dictUnits
        self.dictFunctions = dictFunctions
        
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
        if self.dictUnits is None:
            raise RuntimeError('dictUnits not set')

        node = None
        if isinstance(self.parseResult, Number):
            node = self.parseResult.Node
        else:
            raise RuntimeError('Invalid parse result type')

        result = node.evaluate(self.dictUnits, self.dictFunctions)
        return result

def testExpression(expression, do_evaluation = True):
    parse_res    = parser.parse(expression)
    latex_res    = parser.toLatex()
    print 'Expression: ' + expression
    #print 'NodeTree:\n', repr(parse_res)
    print 'Parse result: ', str(parse_res)
    print 'Latex: ', latex_res
    eval_res = 0
    if do_evaluation:
        eval_res = parser.evaluate()
        print 'Evaluate result: {0}'.format(eval_res)
    print '\n'

    return parse_res, latex_res, eval_res

if __name__ == "__main__":
    # Basic SI units
    kg  = unit(value = 1.0, M = 1, name = 'kg')
    m   = unit(value = 1.0, L = 1, name = 'm')
    s   = unit(value = 1.0, T = 1, name = 's')
    cd  = unit(value = 1.0, C = 1, name = 'cd')
    A   = unit(value = 1.0, A = 1, name = 'A')
    K   = unit(value = 1.0, K = 1, name = 'K')
    mol = unit(value = 1.0, N = 1, name = 'mol')

    # Some derived units
    g = 0.001 * kg
    V = (kg * m**2) / (A * s**(-3))
    mV = 0.001 * V

    dictUnits     = {}
    dictFunctions = {}

    dictUnits['kg']  = kg
    dictUnits['m']   = m
    dictUnits['s']   = s
    dictUnits['cd']  = cd
    dictUnits['A']   = A
    dictUnits['K']   = K
    dictUnits['mol'] = mol
    dictUnits['g']   = g
    dictUnits['V']   = V
    dictUnits['mV']  = mV

    # if we want to evaluate and check units of an expression, we have to provide
    # basic math. functions as well and some other identifiers (e, t, pi ...) 
    """
    dictUnits['t']  = model.time()
    dictUnits['pi'] = math.pi
    dictUnits['e']  = math.e

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
    """

    parser = UnitParser(dictUnits, dictFunctions)

    print repr(g * K)
    print 0.1*kg ** 2
    print kg == 1.02*kg

    testExpression('kg*V/mV^2')
    testExpression('kg*V/A^2')
