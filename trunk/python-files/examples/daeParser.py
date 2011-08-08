#!/usr/bin/env python

"""********************************************************************************
                             daeParser.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2011
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""

import os, sys
import ply.lex as lex
import ply.yacc as yacc
from daetools.pyDAE import *

class Node:
    def evaluate(self, dictNamesValues):
        pass

class ConstantNode(Node):
    def __init__(self, value):
        self.Value = value

    def __repr__(self):
        return str(self.Value)

    def __str__(self):
        return str(self.Value)

    def evaluate(self, dictNamesValues):
        return self.Value

class AssignmentNode(Node):
    def __init__(self, lnode, rnode):
        self.lNode = lnode
        self.rNode = rnode

    def __repr__(self):
        return 'AssignmentNode({0}, =, {1})'.format(repr(self.lNode), repr(self.rNode))

    def __str__(self):
        return '{0} = {1}'.format(str(self.lNode), str(self.rNode))

    def evaluate(self, dictNamesValues):
        return self.rNode.evaluate(dictNamesValues)

class IdentifierNode(Node):
    def __init__(self, name):
        self.Name = name

    def __repr__(self):
        return 'IdentifierNode({0})'.format(repr(self.Name))

    def __str__(self):
        return self.Name

    def evaluate(self, dictNamesValues):
        if self.Name in dictNamesValues:
            return dictNamesValues[self.Name]
        raise RuntimeError('Identifier {0} not found in the Name-Value dictionary'.format(self.Name))

class FunctionNode(Node):
    functions = ['exp', 'sin', 'cos', 'tan', 'log', 'ln', 'sqrt']

    def __init__(self, function, expression):
        if not (function in FunctionNode.functions):
            raise RuntimeError('The function {0} is not supported'.format(function))

        self.Function = function
        self.Node     = expression.Node

    def __repr__(self):
        return 'FunctionNode({0}, {1})'.format(self.Function, repr(self.Node))

    def __str__(self):
        return '{0}({1})'.format(self.Function, str(self.Node))

    def evaluate(self, dictNamesValues):
        if self.Function in dictNamesValues:
            fun = dictNamesValues[self.Function]
            return fun(self.Node.evaluate(dictNamesValues))
        else:
            raise RuntimeError('The function {0} not found in the Name-Value dictionary'.format(self.Function))

class UnaryNode(Node):
    opMinus = '-'
    opPlus  = '+'

    def __init__(self, operator, node):
        self.Node     = node
        self.Operator = operator

    def __repr__(self):
        return 'UnaryNode({0}{1})'.format(self.Operator, repr(self.Node))

    def __str__(self):
        return '({0}{1})'.format(self.Operator, str(self.Node))

    def evaluate(self, dictNamesValues):
        if self.Operator == UnaryNode.opMinus:
            return (-self.Node.evaluate(dictNamesValues))
        elif self.Operator == UnaryNode.opPlus:
            return self.Node.evaluate(dictNamesValues)
        else:
            raise RuntimeError("Not supported unary operator: {0}".format(self.Operator))

class BinaryNode(Node):
    opMinus  = '-'
    opPlus   = '+'
    opMulti  = '*'
    opDivide = '/'
    opPower  = '^'

    def __init__(self, lnode, operator, rnode):
        self.lNode    = lnode
        self.rNode    = rnode
        self.Operator = operator

    def __repr__(self):
        return 'BinaryNode({0}, {1}, {2})'.format(repr(self.lNode), self.Operator, repr(self.rNode))

    def __str__(self):
        return '({0} {1} {2})'.format(str(self.lNode), self.Operator, str(self.rNode))

    def evaluate(self, dictNamesValues):
        if self.Operator == BinaryNode.opPlus:
            return self.lNode.evaluate(dictNamesValues) + self.rNode.evaluate(dictNamesValues)
        elif self.Operator == BinaryNode.opMinus:
            return self.lNode.evaluate(dictNamesValues) - self.rNode.evaluate(dictNamesValues)
        elif self.Operator == BinaryNode.opMulti:
            return self.lNode.evaluate(dictNamesValues) * self.rNode.evaluate(dictNamesValues)
        elif self.Operator == BinaryNode.opDivide:
            return self.lNode.evaluate(dictNamesValues) / self.rNode.evaluate(dictNamesValues)
        elif self.Operator == BinaryNode.opPower:
            return self.lNode.evaluate(dictNamesValues) ** self.rNode.evaluate(dictNamesValues)
        else:
            raise RuntimeError("Not supported binary operator: {0}".format(self.Operator))

class ConditionNode:
    def evaluate(self, dictNamesValues):
        pass

class ConditionUnaryNode(ConditionNode):
    opNot = 'not'

    def __init__(self, operator, node):
        self.Node     = node
        self.Operator = operator

    def __repr__(self):
        return 'ConditionUnaryNode({0}, {1})'.format(self.Operator, repr(self.Node))

    def __str__(self):
        return '({0} {1})'.format(self.Operator, str(self.Node))

    def evaluate(self, dictNamesValues):
        if self.Operator == ConditionUnaryNode.opNot:
            return not self.Node.evaluate(dictNamesValues)
        else:
            raise RuntimeError("Not supported logical unary operator: {0}".format(self.Operator))

class ConditionBinaryNode(ConditionNode):
    opEQ = '=='
    opNE = '!='
    opGT = '>'
    opGE = '>='
    opLT = '<'
    opLE = '<='

    def __init__(self, lnode, operator, rnode):
        self.lNode    = lnode
        self.rNode    = rnode
        self.Operator = operator

    def __repr__(self):
        return 'ConditionBinaryNode({0}, {1}, {2})'.format(repr(self.lNode), self.Operator, repr(self.rNode))

    def __str__(self):
        return '({0} {1} {2})'.format(str(self.lNode), self.Operator, str(self.rNode))

    def evaluate(self, dictNamesValues):
        if self.Operator == ConditionBinaryNode.opEQ:
            return self.lNode.evaluate(dictNamesValues) == self.rNode.evaluate(dictNamesValues)
        elif self.Operator == ConditionBinaryNode.opNE:
            return self.lNode.evaluate(dictNamesValues) != self.rNode.evaluate(dictNamesValues)
        elif self.Operator == ConditionBinaryNode.opLT:
            return self.lNode.evaluate(dictNamesValues) < self.rNode.evaluate(dictNamesValues)
        elif self.Operator == ConditionBinaryNode.opLE:
            return self.lNode.evaluate(dictNamesValues) <= self.rNode.evaluate(dictNamesValues)
        elif self.Operator == ConditionBinaryNode.opGT:
            return self.lNode.evaluate(dictNamesValues) > self.rNode.evaluate(dictNamesValues)
        elif self.Operator == ConditionBinaryNode.opGE:
            return self.lNode.evaluate(dictNamesValues) >= self.rNode.evaluate(dictNamesValues)
        else:
            raise RuntimeError("Not supported logical binary operator: {0}".format(self.Operator))

class ConditionExpressionNode(ConditionNode):
    opAnd = '&'
    opOr  = '|'

    def __init__(self, lnode, operator, rnode):
        self.lNode    = lnode
        self.rNode    = rnode
        self.Operator = operator

    def __repr__(self):
        return 'ConditionExpressionNode({0}, {1}, {2})'.format(repr(self.lNode), self.Operator, repr(self.rNode))

    def __str__(self):
        return '({0} {1} {2})'.format(str(self.lNode), self.Operator, str(self.rNode))

    def evaluate(self, dictNamesValues):
        if self.Operator == ConditionExpressionNode.opAnd:
            return self.lNode.evaluate(dictNamesValues) & self.rNode.evaluate(dictNamesValues)
        elif self.Operator == ConditionExpressionNode.opOr:
            return self.lNode.evaluate(dictNamesValues) | self.rNode.evaluate(dictNamesValues)
        else:
            raise RuntimeError("Not supported logical operator: {0}".format(self.Operator))


class Condition:
    def __init__(self, condNode):
        self.CondNode = condNode

    def __repr__(self):
        return repr(self.CondNode)

    def __str__(self):
        return str(self.CondNode)

    def __not__(self):
        return Condition(ConditionUnaryNode(ConditionUnaryNode.opNot,
                                            self.CondNode
                                           )
                        )

    def __and__(self, cond):
        return Condition(ConditionExpressionNode(self.CondNode,
                                                 ConditionExpressionNode.opAnd,
                                                 cond.CondNode
                                                )
                        )

    def __or__(self, cond):
        return Condition(ConditionExpressionNode(self.CondNode,
                                                 ConditionExpressionNode.opOr,
                                                 cond.CondNode
                                                )
                        )

def getOperand(val):
    operand = None
    if isinstance(val, float):
        operand = Number(ConstantNode(val))
    elif isinstance(val, Number):
        operand = val
    else:
        raise RuntimeError("Invalid operand type")
    return operand

class Number:
    def __init__(self, node):
        if node == None:
            raise RuntimeError("Invalid node")
        self.Node = node

    def __repr__(self):
        return repr(self.Node)

    def __str__(self):
        return str(self.Node)

    def __neg__(self):
        return Number(UnaryNode(UnaryNode.opMinus, self.Node))

    def __pos__(self):
        return Number(UnaryNode(UnaryNode.opPlus, self.Node))

    def __add__(self, val):
        return Number(BinaryNode(self.Node,
                                 BinaryNode.opPlus,
                                 val.Node
                                )
                     )

    def __sub__(self, val):
        return Number(BinaryNode(self.Node,
                                 BinaryNode.opMinus,
                                 val.Node
                                )
                     )

    def __mul__(self, val):
        return Number(BinaryNode(self.Node,
                                 BinaryNode.opMulti,
                                 val.Node
                                )
                     )

    def __div__(self, val):
        return Number(BinaryNode(self.Node,
                                 BinaryNode.opDivide,
                                 val.Node
                                )
                     )

    def __pow__(self, val):
        return Number(BinaryNode(self.Node,
                                 BinaryNode.opPower,
                                 val.Node
                                )
                     )

    def __eq__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opEQ,
                                             val.Node
                                )
                     )

    def __neq__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opNE,
                                             val.Node
                                )
                     )

    def __le__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opLE,
                                             val.Node
                                )
                     )

    def __gt__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opGT,
                                             val.Node
                                )
                     )

    def __ge__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opGE,
                                             val.Node
                                )
                     )

reserved = {'exp' : 'exp',
            'sin' : 'sin'
            }
tokens = [
    'NAME', 'NUMBER', 'FLOAT',
    'PLUS','MINUS','EXP', 'TIMES','DIVIDE','EQUALS',
    'LPAREN','RPAREN',
    'NOT', 'AND', 'OR',
    'LT', 'LE', 'GT', 'GE', 'EQ', 'NE',
    ] + list(reserved.values())

t_PLUS    = r'\+'
t_MINUS   = r'-'
t_EXP     = r'\*\*'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'

t_NOT = r'!'
t_AND = r'&'
t_OR  = r'\|'
t_EQ = r'=='
t_NE = r'!='
t_GT = r'>'
t_GE = r'>='
t_LT = r'<'
t_LE = r'<='

#t_FUNCTIONS = r'sin|cos|tan|exp'
t_EQUALS  = r'='
t_LPAREN  = r'\('
t_RPAREN  = r'\)'

#t_NAME    = r'[a-zA-Z_][a-zA-Z0-9_]*'
def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value,'NAME') # Check for reserved words
    return t

t_NUMBER = r'\d+([uU]|[lL]|[uU][lL]|[lL][uU])?'
t_FLOAT = r'((\d+)(\.\d+)(e(\+|-)?(\d+))? | (\d+)e(\+|-)?(\d+))([lL]|[fF])?'

t_ignore = " \t"

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_error(t):
    print "Illegal character '%s'" % t.value[0]
    t.lexer.skip(1)

# Parsing rules

precedence = (
    ('left','NAME'),
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE'),
    ('left', 'EXP'),
    ('right','UMINUS')
    )

#def p_exp(p):
#    """expression : EXP LPAREN expression RPAREN"""
#    p[0] = Number(FunctionNode(p[1], p[3]))

def p_statement_assign(p):
    """statement : expression EQUALS expression"""
    p[0] = AssignmentNode(p[1], p[3])

def p_statement_expr(p):
    """
    statement : expression
              | expression_logical
              | expression_rel
    """
    p[0] = p[1]

def p_expression_binop(p):
    """
    expression : expression PLUS expression
               | expression MINUS expression
               | expression TIMES expression
               | expression DIVIDE expression
               | expression EXP expression
    """
    if p[2]   == '+' : p[0] = p[1] + p[3]
    elif p[2] == '-' : p[0] = p[1] - p[3]
    elif p[2] == '*' : p[0] = p[1] * p[3]
    elif p[2] == '/' : p[0] = p[1] / p[3]
    elif p[2] == '^' : p[0] = p[1] ** p[3]

def p_expression_logical(p):
    """
    expression_logical  : expression LT expression
                        | expression LE expression
                        | expression GT expression
                        | expression GE expression
                        | expression EQ expression
                        | expression NE expression
    """
    if   p[2] == '<'  : p[0] = p[1] <  p[3]
    elif p[2] == '<=' : p[0] = p[1] <= p[3]
    elif p[2] == '>'  : p[0] = p[1] >  p[3]
    elif p[2] == '>=' : p[0] = p[1] >= p[3]
    elif p[2] == '==' : p[0] = p[1] == p[3]
    elif p[2] == '!=' : p[0] = p[1] != p[3]

def p_expression_rel(p):
    """
    expression_rel : expression_logical AND expression_logical
                   | expression_logical OR  expression_logical
    """
    if p[2]   == '&' : p[0] = p[1] & p[3]
    elif p[2] == '|' : p[0] = p[1] | p[3]

def p_expression_uminus(p):
    """expression : MINUS expression %prec UMINUS"""
    p[0] = - p[2]

def p_expression_not(p):
    """expression_rel : NOT expression_logical"""
    p[0] = (not p[2])

def p_expression_group_1(p):
    """expression : LPAREN expression RPAREN"""
    p[0] = p[2]

def p_expression_group_2(p):
    """expression_rel : LPAREN expression_rel RPAREN"""
    p[0] = p[2]

def p_expression_group_3(p):
    """expression_logical : LPAREN expression_logical RPAREN"""
    p[0] = p[2]

def p_expression_number(p):
    """expression : NUMBER"""
    p[0] = Number(ConstantNode(int(p[1])))

def p_expression_float(p):
    """expression : FLOAT"""
    p[0] = Number(ConstantNode(float(p[1])))

def p_expression_name(p):
    """expression : NAME"""
    p[0] = Number(IdentifierNode(p[1]))

def p_error(p):
    print "Syntax error at '%s'" % p.value
    raise Exception("")

class daeParser:
    def __init__(self):
        self.lexer  = lex.lex()
        self.parser = yacc.yacc()
        self.parseResult = None

    def parse(self, expression):
        self.parseResult = self.parser.parse(expression, debug = 0)
        return self.parseResult

    def evaluate(self, dictNamesValues):
        if self.parseResult is None:
            raise RuntimeError('expression not parsed yet')

        node = None
        if isinstance(self.parseResult, Number):
            node = self.parseResult.Node
        elif isinstance(self.parseResult, Condition):
            node = self.parseResult.CondNode
        elif isinstance(self.parseResult, AssignmentNode):
            node = self.parseResult
        else:
            raise RuntimeError('Invalid parse result type')

        result = node.evaluate(dictNamesValues)
        return result

def getSimpleParserDictionary():
    """
    Dictionary should contain the keys of the following type:
     - identifier-name: value (like 'V' : 1.25)
     - function-of-one-argument : function-object ('exp': math.exp)
       these function keys HAS to be implemented: sin, cos, tan, exp, ln, log, sqrt
    """
    import math
    dictNameValue = {}

    dictNameValue['y']  = 10.0
    dictNameValue['x1'] = 1.0
    dictNameValue['x2'] = 2.0
    dictNameValue['x3'] = 3.0
    dictNameValue['x4'] = 4.0

    dictNameValue['sin']  = math.sin
    dictNameValue['cos']  = math.cos
    dictNameValue['tan']  = math.tan
    dictNameValue['log']  = math.log10
    dictNameValue['ln']   = math.log
    dictNameValue['sqrt'] = math.sqrt
    dictNameValue['exp']  = math.exp

    return dictNameValue

# Some tests...
if __name__ == "__main__":
    parser  = daeParser()

    dictNameValue = getSimpleParserDictionary()
    
    expression = 'y + x2 / x4 + 4.0 - x1'
    parse_res = parser.parse(expression)
    eval_res  = parser.evaluate(dictNameValue)
    print '-------------------------------------------'
    print 'Expression:' + expression
    print '-------------------------------------------'
    print '    NodeTree:', repr(parse_res)
    print '    String:', str(parse_res)
    print '    Evaluate:', str(eval_res)
    print '\n\n'

    #expression = '(exp(y + x2 / x4) + 4.0) - x1'
    #parse_res = parser.parse(expression)
    #print '-------------------------------------------'
    #print 'Expression:' + expression
    #print '-------------------------------------------'
    #print '    NodeTree:', repr(parse_res)
    #print '    String:', str(parse_res)
    #print '\n\n'

    expression = 'y = (x1 + x3)/x4'
    res = parser.parse(expression)
    parse_res = parser.parse(expression)
    print '-------------------------------------------'
    print 'Expression:' + expression
    print '-------------------------------------------'
    print '    NodeTree:', repr(parse_res)
    print '    String:', str(parse_res)
    print '\n\n'

    expression = '(y + 4.0 >= x3 - 3.2e-03) & (y == 3)'
    parse_res = parser.parse(expression)
    print '-------------------------------------------'
    print 'Expression:' + expression
    print '-------------------------------------------'
    print '    NodeTree:', repr(parse_res)
    print '    String:', str(parse_res)
    print '\n\n'

    expression = 'dV/dt = (v_rest - V)/tau_m + (gE*(e_rev_E - V) + gI*(e_rev_I - V) + i_offset)/cm'
    parse_res = parser.parse(expression)
    print '-------------------------------------------'
    print 'Expression:' + expression
    print '-------------------------------------------'
    print '    NodeTree:', repr(parse_res)
    print '    String:', str(parse_res)
    print '\n\n'
