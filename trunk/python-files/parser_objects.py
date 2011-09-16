#!/usr/bin/env python

"""********************************************************************************
                          parser_objects.py
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

import os, sys, operator, math
from copy import copy, deepcopy

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

def exp(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.exp(unit.value)
    return tmp

def pow(unit, n):
    return unit ** n

def log(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.log(unit.value)
    return tmp

def log10(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.log10(unit.value)
    return tmp

def sqrt(unit):
    return unit ** 0.5

def sin(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.sin(unit.value)
    return tmp

def cos(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.cos(unit.value)
    return tmp

def tan(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.tan(unit.value)
    return tmp

def asin(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.asin(unit.value)
    return tmp

def acos(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.acos(unit.value)
    return tmp

def atan(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.atan(unit.value)
    return tmp

def sinh(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.sinh(unit.value)
    return tmp

def cosh(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.cosh(unit.value)
    return tmp

def tanh(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.tanh(unit.value)
    return tmp

def asinh(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.asinh(unit.value)
    return tmp

def acosh(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.acosh(unit.value)
    return tmp

def atanh(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.atanh(unit.value)
    return tmp

def abs(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.abs(unit.value)
    return tmp

def ceil(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.ceil(unit.value)
    return tmp

def floor(unit):
    tmp       = deepcopy(unit)
    tmp.value = math.floor(unit.value)
    return tmp
