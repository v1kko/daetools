#!/usr/bin/env python

"""********************************************************************************
                             parser.py
                 Copyright (C) Dragan Nikolic, 2011
***********************************************************************************
ExpressionParser is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. It is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this 
software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""

import os, sys, operator
import ply.lex as lex
import ply.yacc as yacc
from math import *

class Node:
    def evaluate(self, dictIdentifiers, dictFunctions):
        pass

class ConstantNode(Node):
    def __init__(self, value):
        self.Value = value

    def __repr__(self):
        return 'ConstantNode({0})'.format(self.Value)

    def __str__(self):
        return str(self.Value)

    def toLatex(self):
        return str(self.Value)

    def evaluate(self, dictIdentifiers, dictFunctions):
        return self.Value

class AssignmentNode(Node):
    def __init__(self, identifier, expression):
        self.identifier = identifier
        self.expression = expression

    def __repr__(self):
        return 'AssignmentNode({0}, =, {1})'.format(repr(self.identifier), repr(self.expression))

    def __str__(self):
        return '{0} = {1}'.format(str(self.identifier), str(self.expression))

    def toLatex(self):
        return '{0} = {1}'.format(self.identifier.toLatex(), self.expression.toLatex())

    def evaluate(self, dictIdentifiers, dictFunctions):
        value = self.expression.Node.evaluate(dictIdentifiers, dictFunctions)
        dictIdentifiers[self.identifier.Node.Name] = value
        return value

class IdentifierNode(Node):
    def __init__(self, name):
        self.Name = name

    def __repr__(self):
        return 'IdentifierNode({0})'.format(repr(self.Name))

    def __str__(self):
        return self.Name

    def toLatex(self):
        return self.Name

    def evaluate(self, dictIdentifiers, dictFunctions):
        if self.Name in dictIdentifiers:
            return dictIdentifiers[self.Name]

        raise RuntimeError('Identifier {0} not found in the identifiers dictionary'.format(self.Name))

class StandardFunctionNode(Node):
    functions = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
                 'sqrt', 'exp', 'log', 'log10', 'ceil', 'floor']

    def __init__(self, function, expression):
        if not (function in StandardFunctionNode.functions):
            raise RuntimeError('The function {0} is not supported'.format(function))

        self.Function = function
        self.Node     = expression.Node

    def __repr__(self):
        return 'StandardFunctionNode({0}, {1})'.format(self.Function, repr(self.Node))

    def __str__(self):
        return '{0}({1})'.format(self.Function, str(self.Node))

    def toLatex(self):
        if self.Function == 'sqrt':
            return '\\sqrt{{{0}}}'.format(self.Node.toLatex())
        elif self.Function == 'exp':
            return 'e ^ {{{0}}}'.format(self.Node.toLatex())
        else:
            return '{0} \\left( {1} \\right)'.format(self.Function, self.Node.toLatex())
            
    def evaluate(self, dictIdentifiers, dictFunctions):
        if self.Function in dictFunctions:
            fun = dictFunctions[self.Function]
            if not callable(fun):
                raise RuntimeError('The function {0} in the dictionary is not a callable object'.format(self.Function))
            argument0 = self.Node.evaluate(dictIdentifiers, dictFunctions)
            return fun(argument0)
        else:
            raise RuntimeError('The function {0} not found in the functions dictionary'.format(self.Function))

class NonstandardFunctionNode(Node):
    def __init__(self, function, argument_list = []):
        self.Function          = function
        self.ArgumentsNodeList = []
        for arg in argument_list:
            self.ArgumentsNodeList.append(arg.Node)

    def __repr__(self):
        argument_list = ''
        for i in range(0, len(self.ArgumentsNodeList)):
            node = self.ArgumentsNodeList[i]
            if i == 0:
                argument_list += repr(node)
            else:
                argument_list += ', ' + repr(node)
        return 'NonstandardFunctionNode({0}, {1})'.format(self.Function, argument_list)

    def __str__(self):
        argument_list = ''
        for i in range(0, len(self.ArgumentsNodeList)):
            node = self.ArgumentsNodeList[i]
            if i == 0:
                argument_list += str(node)
            else:
                argument_list += ', ' + str(node)
        return '{0}({1})'.format(self.Function, argument_list)

    def toLatex(self):
        argument_list = ''
        for i in range(0, len(self.ArgumentsNodeList)):
            node = self.ArgumentsNodeList[i]
            if i == 0:
                argument_list += node.toLatex()
            else:
                argument_list += ', ' + node.toLatex()
        return '{0} \\left( {1} \\right)'.format(self.Function, argument_list)

    def evaluate(self, dictIdentifiers, dictFunctions):
        if self.Function in dictFunctions:
            fun = dictFunctions[self.Function]
            if not callable(fun):
                raise RuntimeError('The function {0} in the dictionary is not a callable object'.format(self.Function))
            argument_list = ()
            for node in self.ArgumentsNodeList:
                argument_list = argument_list + (node.evaluate(dictIdentifiers, dictFunctions), )
            return fun(*argument_list)
        else:
            raise RuntimeError('The function {0} not found in the functions dictionary'.format(self.Function))

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

    def toLatex(self):
        return '{0}{1}'.format(self.Operator, self.encloseNode())

    def enclose(self, doEnclose):
        if doEnclose:
            return '\\left( ' + self.Node.toLatex() + ' \\right)'
        else:
            return self.Node.toLatex()
            
    def encloseNode(self):
        if isinstance(self.Node, ConstantNode):
            return self.enclose(False)

        elif isinstance(self.Node, IdentifierNode):
            return self.enclose(False)

        elif isinstance(self.Node, StandardFunctionNode):
            return self.enclose(False)

        elif isinstance(self.Node, UnaryNode):
            return self.enclose(True)

        elif isinstance(self.Node, BinaryNode):
            if (self.Node.Operator == '+') or (self.Node.Operator == '-'):
                return self.enclose(True)
            elif (self.Node.Operator == '*') or (self.Node.Operator == '/') or (self.Node.Operator == '^'):
                return self.enclose(False)
            else:
                return self.enclose(True)

        else:
            return self.enclose(True)
            
    def evaluate(self, dictIdentifiers, dictFunctions):
        if self.Operator == UnaryNode.opMinus:
            return (-self.Node.evaluate(dictIdentifiers, dictFunctions))
        elif self.Operator == UnaryNode.opPlus:
            return self.Node.evaluate(dictIdentifiers, dictFunctions)
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

    def encloseLeft(self, doEnclose):
        if doEnclose:
            return '\\left( ' + self.lNode.toLatex() + ' \\right)'
        else:
            return self.lNode.toLatex()

    def encloseRight(self, doEnclose):
        if doEnclose:
            return '\\left( ' + self.rNode.toLatex() + ' \\right)'
        else:
            return self.rNode.toLatex()

    def toLatex(self):
        if (self.Operator == '+'):
            # Default behaviour is to not enclose any
            left  = self.encloseLeft(False)
            right = self.encloseRight(False)

            # Right exceptions:
            if isinstance(self.rNode, UnaryNode):
                right = self.encloseRight(True)

            return '{0} + {1}'.format(left, right)

        elif (self.Operator == '-'):
            # Default behaviour is to enclose right
            left  = self.encloseLeft(False)
            right = self.encloseRight(True)

            # Right exceptions:
            if isinstance(self.rNode, ConstantNode):
                right = self.encloseRight(False)
            elif isinstance(self.rNode, IdentifierNode):
                right = self.encloseRight(False)
            elif isinstance(self.rNode, StandardFunctionNode):
                right = self.encloseRight(False)
            elif isinstance(self.rNode, BinaryNode):
                if (self.rNode.Operator == '*') or (self.rNode.Operator == '/') or (self.rNode.Operator == '^'):
                    right = self.encloseRight(False)

            return '{0} - {1}'.format(left, right)

        elif (self.Operator == '*'):
            # Default behaviour is to enclose both
            left  = self.encloseLeft(True)
            right = self.encloseRight(True)

            # Left exceptions:
            if isinstance(self.lNode, ConstantNode):
                left = self.encloseLeft(False)
            elif isinstance(self.lNode, IdentifierNode):
                left = self.encloseLeft(False)
            elif isinstance(self.lNode, StandardFunctionNode):
                left = self.encloseLeft(False)
            elif isinstance(self.lNode, UnaryNode):
                left = self.encloseLeft(False)
            elif isinstance(self.lNode, BinaryNode):
                if (self.lNode.Operator == '*') or (self.lNode.Operator == '/') or (self.lNode.Operator == '^'):
                    left = self.encloseLeft(False)

            # Right exceptions:
            if isinstance(self.rNode, ConstantNode):
                right = self.encloseRight(False)
            elif isinstance(self.rNode, IdentifierNode):
                right = self.encloseRight(False)
            elif isinstance(self.rNode, StandardFunctionNode):
                right = self.encloseRight(False)
            elif isinstance(self.rNode, BinaryNode):
                if (self.rNode.Operator == '*') or (self.rNode.Operator == '/') or (self.rNode.Operator == '^'):
                    right = self.encloseRight(False)

            return '{0} \\cdot {1}'.format(left, right)

        elif (self.Operator == '/'):
            # Default behaviour is to not enclose any
            left  = self.encloseLeft(False)
            right = self.encloseRight(False)

            return '\\frac{{{0}}}{{{1}}}'.format(left, right)

        elif (self.Operator == '^'):
            # Default behaviour is to enclose left
            left  = self.encloseLeft(True)
            right = self.encloseRight(False)

            # Left exceptions:
            if isinstance(self.lNode, ConstantNode):
                left = self.encloseLeft(False)
            elif isinstance(self.lNode, IdentifierNode):
                left = self.encloseLeft(False)

            return '{{{0}}} ^ {{{1}}}'.format(left, right)

        else:
            # Default behaviour is to enclose both
            left  = self.encloseLeft(True)
            right = self.encloseRight(True)

            return '{0} {1} {2}'.format(left, self.Operator, right)
            
    def evaluate(self, dictIdentifiers, dictFunctions):
        if self.Operator == BinaryNode.opPlus:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) + self.rNode.evaluate(dictIdentifiers, dictFunctions)

        elif self.Operator == BinaryNode.opMinus:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) - self.rNode.evaluate(dictIdentifiers, dictFunctions)

        elif self.Operator == BinaryNode.opMulti:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) * self.rNode.evaluate(dictIdentifiers, dictFunctions)

        elif self.Operator == BinaryNode.opDivide:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) / self.rNode.evaluate(dictIdentifiers, dictFunctions)

        elif self.Operator == BinaryNode.opPower:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) ** self.rNode.evaluate(dictIdentifiers, dictFunctions)

        else:
            raise RuntimeError("Not supported binary operator: {0}".format(self.Operator))

class ConditionNode:
    def evaluate(self, dictIdentifiers, dictFunctions):
        pass

"""
class ConditionUnaryNode(ConditionNode):
    opNot = 'not'

    def __init__(self, operator, node):
        self.Node     = node
        self.Operator = operator

    def __repr__(self):
        return 'ConditionUnaryNode({0}, {1})'.format(self.Operator, repr(self.Node))

    def __str__(self):
        return '({0} {1})'.format(self.Operator, str(self.Node))

    def evaluate(self, dictIdentifiers, dictFunctions):
        if self.Operator == ConditionUnaryNode.opNot:
            return operator.not_(self.Node.evaluate(dictIdentifiers, dictFunctions))
        else:
            raise RuntimeError("Not supported logical unary operator: {0}".format(self.Operator))
"""

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

    def toLatex(self):
        if self.Operator == ConditionBinaryNode.opEQ:
            return '\\left( {0} == {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opNE:
            return '\\left( {0} \\neq {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opLT:
            return '\\left( {0} < {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opLE:
            return '\\left( {0} \\leq {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opGT:
            return '\\left( {0} > {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opGE:
            return '\\left( {0} \\geq {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        else:
            raise RuntimeError("Not supported logical binary operator: {0}".format(self.Operator))

    def evaluate(self, dictIdentifiers, dictFunctions):
        if self.Operator == ConditionBinaryNode.opEQ:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) == self.rNode.evaluate(dictIdentifiers, dictFunctions)
        elif self.Operator == ConditionBinaryNode.opNE:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) != self.rNode.evaluate(dictIdentifiers, dictFunctions)
        elif self.Operator == ConditionBinaryNode.opLT:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) < self.rNode.evaluate(dictIdentifiers, dictFunctions)
        elif self.Operator == ConditionBinaryNode.opLE:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) <= self.rNode.evaluate(dictIdentifiers, dictFunctions)
        elif self.Operator == ConditionBinaryNode.opGT:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) > self.rNode.evaluate(dictIdentifiers, dictFunctions)
        elif self.Operator == ConditionBinaryNode.opGE:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) >= self.rNode.evaluate(dictIdentifiers, dictFunctions)
        else:
            raise RuntimeError("Not supported logical binary operator: {0}".format(self.Operator))

class ConditionExpressionNode(ConditionNode):
    opAnd = '&&'
    opOr  = '||'

    def __init__(self, lnode, operator, rnode):
        self.lNode    = lnode
        self.rNode    = rnode
        self.Operator = operator

    def __repr__(self):
        return 'ConditionExpressionNode({0}, {1}, {2})'.format(repr(self.lNode), self.Operator, repr(self.rNode))

    def __str__(self):
        return '({0} {1} {2})'.format(str(self.lNode), self.Operator, str(self.rNode))

    def toLatex(self):
        if self.Operator == ConditionExpressionNode.opAnd:
            return '\\left( {0} \\land {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionExpressionNode.opOr:
            return '\\left( {0} \\lor {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        else:
            raise RuntimeError("Not supported logical binary operator: {0}".format(self.Operator))

    def evaluate(self, dictIdentifiers, dictFunctions):
        if self.Operator == ConditionExpressionNode.opAnd:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) & self.rNode.evaluate(dictIdentifiers, dictFunctions)
        elif self.Operator == ConditionExpressionNode.opOr:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) | self.rNode.evaluate(dictIdentifiers, dictFunctions)
        else:
            raise RuntimeError("Not supported logical operator: {0}".format(self.Operator))


class Condition:
    def __init__(self, condNode):
        self.CondNode = condNode

    def __repr__(self):
        return repr(self.CondNode)

    def __str__(self):
        return str(self.CondNode)

    def toLatex(self):
        return self.CondNode.toLatex()

    #def not_(self):
    #    return Condition(ConditionUnaryNode(ConditionUnaryNode.opNot,
    #                                        self.CondNode
    #                                       )
    #                    )

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

    def toLatex(self):
        return self.Node.toLatex()

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

    def __ne__(self, val):
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

#logical_operator = {'and': 'and',
#                    'or' : 'or'
#                   }

functions = {'exp'  : 'exp',
             'sqrt' : 'sqrt',
             'log'  : 'log',
             'log10': 'log10',
             'sin'  : 'sin',
             'cos'  : 'cos',
             'tan'  : 'tan',
             'asin' : 'asin',
             'acos' : 'acos',
             'atan' : 'atan',
             'sinh' : 'sinh',
             'cosh' : 'cosh',
             'tanh' : 'tanh',
             'asinh': 'asinh',
             'acosh': 'acosh',
             'atanh': 'atanh',
             'ceil' : 'ceil',
             'floor': 'floor'
            }

tokens = [
    'NAME', 'NUMBER', 'FLOAT',
    'PLUS','MINUS','TIMES','DIVIDE','EXP','EQUALS',
    'LPAREN','RPAREN','PERIOD', 'COMMA',
    'LT', 'LE', 'GT', 'GE', 'EQ', 'NE',
    'AND', 'OR'
    ] + list(functions.values()) #+ list(logical_operator.values())

precedence = [
                ('left', 'PLUS', 'MINUS'),
                ('left', 'TIMES', 'DIVIDE'),
                ('left', 'EXP')
             ]


t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_EXP     = r'\*\*'

t_EQ = r'=='
t_NE = r'!='
t_GT = r'>'
t_GE = r'>='
t_LT = r'<'
t_LE = r'<='
t_AND = r'&&'
t_OR  = r'\|\|'

t_COMMA   = r','
t_EQUALS  = r'='
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_PERIOD  = r'\.'

#t_PI = r'pi'

def t_NAME(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    if t.value in functions:
        t.type = functions[t.value]
    #elif t.value in logical_operator:
    #    t.type = logical_operator[t.value]
    else:
        t.type = 'NAME'
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

# Parser rules:
# expression:
def p_expression_1(p):
    'expression : assignment_expression'
    p[0] = p[1]

def p_expression_2(t):
    'expression : expression COMMA assignment_expression'
    p[0] = p[1] + p[3]

# assigment_expression:
def p_assignment_expression_1(p):
    'assignment_expression : conditional_expression'
    p[0] = p[1]

def p_assignment_expression_2(p):
    'assignment_expression : identifier EQUALS assignment_expression'
    p[0] = AssignmentNode(p[1], p[3])
    #'assignment_expression : shift_expression EQUALS assignment_expression'

# conditional-expression
def p_conditional_expression_1(p):
    'conditional_expression : or_expression'
    p[0] = p[1]

# OR-expression
def p_or_expression_1(p):
    'or_expression : and_expression'
    p[0] = p[1]

def p_or_expression_2(p):
    'or_expression : or_expression OR and_expression'
    p[0] = (p[1] | p[3])

# AND-expression
def p_and_expression_1(p):
    'and_expression : equality_expression'
    p[0] = p[1]

def p_and_expression_2(p):
    'and_expression : and_expression AND equality_expression'
    p[0] = (p[1] & p[3])

# equality-expression:
def p_equality_expression_1(p):
    'equality_expression : relational_expression'
    p[0] = p[1]

def p_equality_expression_2(p):
    'equality_expression : equality_expression EQ relational_expression'
    p[0] = (p[1] == p[3])

def p_equality_expression_3(p):
    'equality_expression : equality_expression NE relational_expression'
    p[0] = (p[1] != p[3])

# relational-expression:
def p_relational_expression_1(p):
    'relational_expression : shift_expression'
    p[0] = p[1]

def p_relational_expression_2(p):
    'relational_expression : relational_expression LT shift_expression'
    p[0] = p[1] < p[3]

def p_relational_expression_3(p):
    'relational_expression : relational_expression GT shift_expression'
    p[0] = p[1] > p[3]

def p_relational_expression_4(p):
    'relational_expression : relational_expression LE shift_expression'
    p[0] = p[1] <= p[3]

def p_relational_expression_5(p):
    'relational_expression : relational_expression GE shift_expression'
    p[0] = p[1] >= p[3]

# shift-expression
def p_shift_expression_1(p):
    'shift_expression : additive_expression'
    p[0] = p[1]

# additive-expression
def p_additive_expression_1(p):
    'additive_expression : multiplicative_expression'
    p[0] = p[1]

def p_additive_expression_2(p):
    'additive_expression : additive_expression PLUS multiplicative_expression'
    p[0] = p[1] + p[3]

def p_additive_expression_3(p):
    'additive_expression : additive_expression MINUS multiplicative_expression'
    p[0] = p[1] - p[3]

# multiplicative-expression
def p_multiplicative_expression_1(p):
    'multiplicative_expression : power_expression'
    p[0] = p[1]

def p_multiplicative_expression_2(p):
    'multiplicative_expression : multiplicative_expression DIVIDE power_expression'
    p[0] = p[1] / p[3]

def p_multiplicative_expression_3(p):
    'multiplicative_expression : multiplicative_expression TIMES power_expression'
    p[0] = p[1] * p[3]


def p_power_expression_1(p):
    'power_expression : unary_expression'
    p[0] = p[1]

def p_power_expression_2(p):
    'power_expression : power_expression EXP unary_expression'
    p[0] = p[1] ** p[3]

# unary-expression:
def p_unary_expression_1(p):
    'unary_expression : postfix_expression'
    p[0] = p[1]

def p_unary_expression_2(p):
    'unary_expression : unary_operator'
    p[0] = p[1]

# unary-operator:
def p_unary_operator(p):
    '''
    unary_operator : PLUS  postfix_expression
                   | MINUS postfix_expression
    '''
    if p[1] == '+':
        p[0] = p[2]
    elif p[1] == '-':
        p[0] = - p[2]

# postfix-expression:
def p_postfix_expression_1(p):
    """postfix_expression : primary_expression"""
    p[0] = p[1]

def p_postfix_expression_2(p):
    """
    postfix_expression : sin   LPAREN expression RPAREN
                       | cos   LPAREN expression RPAREN
                       | tan   LPAREN expression RPAREN
                       | asin  LPAREN expression RPAREN
                       | acos  LPAREN expression RPAREN
                       | atan  LPAREN expression RPAREN
                       | sinh  LPAREN expression RPAREN
                       | cosh  LPAREN expression RPAREN
                       | tanh  LPAREN expression RPAREN
                       | asinh LPAREN expression RPAREN
                       | acosh LPAREN expression RPAREN
                       | atanh LPAREN expression RPAREN
                       | exp   LPAREN expression RPAREN
                       | sqrt  LPAREN expression RPAREN
                       | log   LPAREN expression RPAREN
                       | log10 LPAREN expression RPAREN
                       | ceil  LPAREN expression RPAREN
                       | floor LPAREN expression RPAREN
    """
    p[0] = Number(StandardFunctionNode(p[1], p[3]))

def p_postfix_expression_3(p):
    '''postfix_expression : postfix_expression PERIOD NAME'''
    p[0] = Number(IdentifierNode(p[1].Node.Name + '.' + p[3]))

def p_postfix_expression_4(p):
    '''postfix_expression : postfix_expression LPAREN argument_expression_list RPAREN'''
    print str(p[1]) + '(' + str(p[3]) + ')'
    p[0] = Number(NonstandardFunctionNode(str(p[1]), p[3]))

# primary-expression
def p_primary_expression(p):
    '''primary_expression :  identifier
                          |  constant
                          |  LPAREN expression RPAREN'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        p[0] = p[2]

# argument-expression-list:
def p_argument_expression_list(p):
    '''argument_expression_list :  assignment_expression
                                |  argument_expression_list COMMA assignment_expression'''
    arguments = []
    if len(p) == 2:
        arguments.append(p[1])
    elif len(p) == 4:
        for arg in list(p[1]):
            arguments.append(arg)
        arguments.append(p[3])

    p[0] = arguments

def p_constant_1(p):
    """constant : NUMBER"""
    p[0] = Number(ConstantNode(int(p[1])))

def p_constant_2(p):
    """constant : FLOAT"""
    p[0] = Number(ConstantNode(float(p[1])))

def p_identifier(p):
    """identifier : NAME"""
    p[0] = Number(IdentifierNode(p[1]))

def p_error(p):
    print "Syntax error at '%s'" % p.value
    raise Exception("Syntax error at '%s'" % p.value)

# Parser class
class ExpressionParser:
    def __init__(self, dictIdentifiers = None, dictFunctions = None):
        self.lexer  = lex.lex()
        self.parser = yacc.yacc() #(write_tables = 0)
        self.parseResult = None
        self.dictIdentifiers = dictIdentifiers
        self.dictFunctions   = dictFunctions

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
        if self.dictIdentifiers is None:
            raise RuntimeError('dictIdentifiers not set')
        if self.dictFunctions is None:
            raise RuntimeError('dictFunctions not set')

        node = None
        if isinstance(self.parseResult, Number):
            node = self.parseResult.Node
        elif isinstance(self.parseResult, Condition):
            node = self.parseResult.CondNode
        elif isinstance(self.parseResult, AssignmentNode):
            node = self.parseResult
        else:
            raise RuntimeError('Invalid parse result type')

        result = node.evaluate(self.dictIdentifiers, self.dictFunctions)
        return result

def testExpression(expression, expected_res, do_evaluation = True):
    parse_res    = parser.parse(expression)
    latex_res    = parser.toLatex()
    print 'Expression: ' + expression
    #print 'NodeTree:\n', repr(parse_res)
    print 'Parse result: ', str(parse_res)
    print 'Latex: ', latex_res
    eval_res = 0
    if do_evaluation:
        eval_res = parser.evaluate()
        if fabs(eval_res - expected_res) > 0:
            raise RuntimeError('Expression evaluation failed: {0} (evaluated {1}; expected {2})'.format(expression, eval_res, expected_res))
        else:
            print 'Evaluate result: OK (={0})'.format(eval_res)
    print '\n'

    return parse_res, latex_res, eval_res
    
def testLatex(parser):
    operators = ['+', '-', '*', '/', '**']
    l = 'x_1'
    r = 'x_2'
    counter = 0
    for lop in operators:
        for rop in operators:
            for op in operators:
                expression   = l + lop + r + op + l + rop + r
                parse_res    = parser.parse(expression)
                latex_res    = parser.toLatex()
                print '\\begin{verbatim}' + str(counter) + '. ' + expression + '\\end{verbatim}\n'
                print '\\begin{verbatim}Parse result: ' + str(parse_res) + '\\end{verbatim}\n'
                print '$' + latex_res + '$\n\n'
                counter += 1

def Sum(a1, a2, a3):
    return a1 + a2 + a3

def step(t0, amplitude):
    return amplitude

def impulse(t0, amplitude):
    return amplitude

def linear(t0, t1, start, end):
    return end

if __name__ == "__main__":
    """
    The parser supports the following math. functions: 'sqrt', 'exp', 'log', 'log10', 'ceil', 'floor',
                                                       'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                                                       'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'
    Objects that can be used with this parser should support the following math operators: +, -, *, /, **
    and abovementioned math. functions.
    User-defined functions with arbitrary number of arguments that operate on such objects can be defined by the user.
    Dictionary dictIdentifiers should contain pairs of the following type:
      identifier-name: value (for instance 'var' : number,
                              or 'name.var' : object-that-defines-math-operators-and-functions)
    Dictionary dictFunctions should contain pairs of the following type:
      function-name : callable-object (for instance 'exp': math.exp)
    """
    dictIdentifiers = {}
    dictFunctions   = {}

    # Some dummy identifiers:values
    y       = 10.0
    x1      =  1.0
    x2      =  2.0
    x3      =  3.0
    x4      =  4.0
    m1_x    =  2.11
    m1_m2_y = 10.12
    R       =  0.0

    # identifiers
    dictIdentifiers['pi']      = pi
    dictIdentifiers['e']       = e
    dictIdentifiers['y']       = y
    dictIdentifiers['x1']      = x1
    dictIdentifiers['x2']      = x2
    dictIdentifiers['x3']      = x3
    dictIdentifiers['x4']      = x4
    dictIdentifiers['m1.x']    = m1_x
    dictIdentifiers['m1.m2.y'] = m1_m2_y
    dictIdentifiers['R']       = R

    # Standard functions
    dictFunctions['log10']  = log10
    dictFunctions['log']    = log
    dictFunctions['sqrt']   = sqrt
    dictFunctions['exp']    = exp
    dictFunctions['ceil']   = ceil
    dictFunctions['floor']  = floor
    dictFunctions['sin']    = sin
    dictFunctions['cos']    = cos
    dictFunctions['tan']    = tan
    dictFunctions['asin']   = asin
    dictFunctions['acos']   = acos
    dictFunctions['atan']   = atan
    dictFunctions['sinh']   = sinh
    dictFunctions['cosh']   = cosh
    dictFunctions['tanh']   = tanh
    dictFunctions['asinh']  = asinh
    dictFunctions['acosh']  = acosh
    dictFunctions['atanh']  = atanh

    # Nonstandard functions (custom functions)
    dictFunctions['Sum']      = Sum
    dictFunctions['step']     = step
    dictFunctions['impulse']  = impulse
    dictFunctions['linear']   = linear

    print 'Identifiers:\n', dictIdentifiers
    print '\n'
    print 'Functions:\n', dictFunctions
    print '\n'

    parser = ExpressionParser(dictIdentifiers, dictFunctions)

    #testLatex(parser)


    testExpression('step(1.2, 1.2)', 1.2)
    testExpression('impulse(1.2, 1.2)', 1.2)
    testExpression('linear(0.0, 1.0, 5, 10)', 10)
    exit(0)

    testExpression('log10(pi)', log10(pi))
    testExpression('log(pi)',   log(pi))
    testExpression('sqrt(pi)',  sqrt(pi))
    testExpression('exp(pi)',   exp(pi))
    testExpression('ceil(pi)',  ceil(pi))
    testExpression('floor(pi)', floor(pi))

    testExpression('sin(pi)', sin(pi))
    testExpression('cos(pi)', cos(pi))
    testExpression('tan(pi)', tan(pi))
    testExpression('asin(1)', asin(1))
    testExpression('acos(1)', acos(1))
    testExpression('atan(1)', atan(1))

    testExpression('sinh(pi)', sinh(pi))
    testExpression('cosh(pi)', cosh(pi))
    testExpression('tanh(pi)', tanh(pi))
    testExpression('asinh(pi)', asinh(pi))
    testExpression('acosh(pi)', acosh(pi))
    testExpression('atanh(0.1)', atanh(0.1))

    testExpression('x1 + x2', x1 + x2)
    testExpression('x1 - x2', x1 - x2)
    testExpression('x1 * x2', x1 * x2)
    testExpression('x1 / x2', x1 / x2)
    testExpression('x1 ** x2', x1 ** x2)

    testExpression('Sum(pi, 1.5, sqrt(4))', Sum(pi, 1.5, sqrt(4)))
    testExpression('-sqrt(m1.m2.y + 2) / m1.x', -sqrt(m1_m2_y + 2) / m1_x)
    testExpression('(-exp(y + x2 / x4) + 4.0) - x1', (-exp(y + x2 / x4) + 4.0) - x1)
    parse_res, latex_res, eval_res = testExpression('R = sin(x1 + x3)/x4', sin(x1 + x3)/x4)
    print 'Updated dictIdentifiers[R] = {0}\n'.format(dictIdentifiers['R'])
    testExpression('(y + 4.0 >= x3 - 3.2e-03) || (y != 3) && (x1 <= 5)', (y + 4.0 >= x3 - 3.2e-03) or (y != 3) and (x1 <= 5))
    testExpression('(v_rest - V)/tau_m + (gE*(e_rev_E - V) + gI*(e_rev_I - V) + i_offset)/cm', 0, False)
