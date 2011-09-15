import os, sys, operator
import ply.lex as lex
import ply.yacc as yacc
from math import *
from copy import copy, deepcopy

class DimensionsError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def format(name, U):
    if U == 1:
        return '{0}'.format(name, U)
    elif U != 0:
        return '({0}^{1})'.format(name, U)
    else:
        return None
        
def format_and_append(name, U, l):
    res = format(name, U)
    if res:
        l.append(res)

def latex(name, U):
    if U == 1:
        return '{{{0}}}'.format(name)
    elif U != 0:
        return '{{{{{0}}}^{{{1}}}}}'.format(name, U)
    else:
        return None

def latex_and_append(name, U, l):
    res = latex(name, U)
    if res:
        l.append(res)

class unit:
    def __init__(self, **kwargs):
        self.L = kwargs.get('L', 0) # length, m
        self.M = kwargs.get('M', 0) # mass, kg
        self.T = kwargs.get('T', 0) # time, s
        self.C = kwargs.get('C', 0) # candela, cd
        self.A = kwargs.get('A', 0) # ampere, A
        self.K = kwargs.get('K', 0) # kelvin, K
        self.N = kwargs.get('N', 0) # mole, mol
        self.name      = kwargs.get('name',      None)
        #self.latexName = kwargs.get('latexName', self.name)
        self.value     = kwargs.get('value',     0.0)

    def __eq__(self, other):
        return self.areDimensionsEqual(other)
        
    def __ne__(self, other):
        return not self.areDimensionsEqual(other)

    def isEqualTo(self, other):
        if (self.value == other.value) and self.areDimensionsEqual(other):
           return True
        else:
            return False

    def areDimensionsEqual(self, other):
        if ((self.M == other.M) and \
            (self.L == other.L) and \
            (self.L == other.T) and \
            (self.L == other.C) and \
            (self.L == other.A) and \
            (self.L == other.K) and \
            (self.T == other.N)):
           return True
        else:
            return False
            
    def __neg__(self):
        tmp       = deepcopy(self)
        tmp.value = -self.value
        return tmp

    def __pos__(self):
        return deepcopy(self)

    def __add__(self, other):
        if not self.areDimensionsEqual(other):
            raise DimensionsError('Units not consistent')

        tmp       = deepcopy(self)
        tmp.value = self.value + other.value
        return tmp

    def __sub__(self, other):
        if not self.areDimensionsEqual(other):
            raise DimensionsError('Units not consistent')

        tmp       = deepcopy(self)
        tmp.value = self.value - other.value
        return tmp

    def __mul__(self, other):
        if not isinstance(other, unit):
            val   = float(other)
            other = unit(value = val)

        tmp   = unit()
        tmp.L = self.L + other.L
        tmp.M = self.M + other.M
        tmp.T = self.T + other.T
        tmp.C = self.C + other.C
        tmp.A = self.A + other.A
        tmp.K = self.K + other.K
        tmp.N = self.N + other.N
        tmp.value = self.value * other.value
        return tmp

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if not isinstance(other, unit):
            val   = float(other)
            other = unit(value = val)

        tmp   = unit()
        tmp.L = self.L - other.L
        tmp.M = self.M - other.M
        tmp.T = self.T - other.T
        tmp.C = self.C - other.C
        tmp.A = self.A - other.A
        tmp.K = self.K - other.K
        tmp.N = self.N - other.N
        tmp.value = self.value / other.value
        return tmp

    def __rdiv__(self, other):
        if not isinstance(other, unit):
            val = float(other)
            other = unit(value = val)

        return other.__div__(self)

    def __pow__(self, power):
        tmp   = deepcopy(self)
        tmp.L = self.L * float(power)
        tmp.M = self.M * float(power)
        tmp.T = self.T * float(power)
        tmp.C = self.C * float(power)
        tmp.A = self.A * float(power)
        tmp.K = self.K * float(power)
        tmp.N = self.N * float(power)
        tmp.value = self.value ** float(power)
        return tmp

    def unitsAsString(self):
        if self.name:
            return '{0} [{1}]'.format(self.value, self.name)
        else:
            units = []
            format_and_append('kg',  self.M, units)
            format_and_append('m',   self.L, units)
            format_and_append('s',   self.T, units)
            format_and_append('cd',  self.C, units)
            format_and_append('A',   self.A, units)
            format_and_append('K',   self.K, units)
            format_and_append('mol', self.N, units)
            return '*'.join(units)

    def unitsAsLatex(self):
        if self.name:
            return self.name
        else:
            units = []
            latex_and_append('kg',  self.M, units)
            latex_and_append('m',   self.L, units)
            latex_and_append('s',   self.T, units)
            latex_and_append('cd',  self.C, units)
            latex_and_append('A',   self.A, units)
            latex_and_append('K',   self.K, units)
            latex_and_append('mol', self.N, units)
            return ' \cdot '.join(units)

    def __repr__(self):
        template = 'unit(name = {0}, value = {1}, M = {2}, L = {3}, T = {4}, C = {5}, A = {6}, K = {7}, N = {8})'
        return template.format(self.name, self.value, self.M, self.L, self.T, self.C, self.A, self.K, self.N)

    def __str__(self):
        return '{0} [{1}]'.format(self.value, self.unitsAsString())

tokens = [ 'NAME',
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

dictUnits = {}
kg  = unit(value = 1.0, M = 1, name = 'kg')
m   = unit(value = 1.0, L = 1, name = 'm')
s   = unit(value = 1.0, T = 1, name = 's')
cd  = unit(value = 1.0, C = 1, name = 'cd')
A   = unit(value = 1.0, A = 1, name = 'A')
K   = unit(value = 1.0, K = 1, name = 'K')
mol = unit(value = 1.0, N = 1, name = 'mol')
g = 0.001 * kg

dictUnits['kg'] = kg
dictUnits['m']  = m
dictUnits['s']  = s
dictUnits['cd']  = cd
dictUnits['A']  = A
dictUnits['K']  = K
dictUnits['mol']  = mol
dictUnits['g']  = g

t_NAME = r'[a-zA-Z_][a-zA-Z_0-9]*'

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
    p[0] = int(p[1])

def p_constant_2(p):
    """constant : LPAREN NUMBER RPAREN"""
    p[0] = int(p[2])

def p_constant_3(p):
    """constant : FLOAT"""
    p[0] = float(p[1])

def p_constant_4(p):
    """constant : LPAREN FLOAT RPAREN"""
    p[0] = float(p[2])

def p_base_unit_1(p):
    """base_unit : NAME  """
    if not dictUnits.has_key(p[1]):
        raise RuntimeError('Unit: {0} not found in the dictionary'.format(p[1]))
    
    p[0] = dictUnits[p[1]]

def p_error(p):
    raise Exception("Syntax error at '%s'" % p.value)

# Parser class
class UnitParser:
    def __init__(self):
        self.lexer  = lex.lex()
        self.parser = yacc.yacc() #(write_tables = 0)
        self.parseResult = None

    def parse(self, expression):
        self.parseResult = self.parser.parse(expression, debug = 0)
        return self.parseResult


"""
print repr(kg)
print N
print N.unitsAsLatex()
print g * N * kg
print repr(g * N)

print 0.1*kg ** 2
print kg == 1.02*kg
"""

parser = UnitParser()
res = parser.parse('g*kg*m^(-1)/s^(+2)')
print res.unitsAsLatex()

