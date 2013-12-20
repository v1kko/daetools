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

import os, sys, operator, math, numbers
from copy import copy, deepcopy

class Node(object):
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
            if not isinstance(fun, collections.Callable):
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

class ConditionNode(object):
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
            return '{0} == {1}'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opNE:
            return '{0} \\neq {1}'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opLT:
            return '{0} < {1}'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opLE:
            return '{0} \\leq {1}'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opGT:
            return '{0} > {1}'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionBinaryNode.opGE:
            return '{0} \\geq {1}'.format(self.lNode.toLatex(), self.rNode.toLatex())
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
            return '\\left( {0} \\right) \\land \\left( {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        elif self.Operator == ConditionExpressionNode.opOr:
            return '\\left( {0} \\right) \\lor \\left( {1} \\right)'.format(self.lNode.toLatex(), self.rNode.toLatex())
        else:
            raise RuntimeError("Not supported logical binary operator: {0}".format(self.Operator))

    def evaluate(self, dictIdentifiers, dictFunctions):
        if self.Operator == ConditionExpressionNode.opAnd:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) & self.rNode.evaluate(dictIdentifiers, dictFunctions)
        elif self.Operator == ConditionExpressionNode.opOr:
            return self.lNode.evaluate(dictIdentifiers, dictFunctions) | self.rNode.evaluate(dictIdentifiers, dictFunctions)
        else:
            raise RuntimeError("Not supported logical operator: {0}".format(self.Operator))


class Condition(object):
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

class Number(object):
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
                                 val.Node))

    def __sub__(self, val):
        return Number(BinaryNode(self.Node,
                                 BinaryNode.opMinus,
                                 val.Node))

    def __mul__(self, val):
        return Number(BinaryNode(self.Node,
                                 BinaryNode.opMulti,
                                 val.Node))

    def __div__(self, val):
        return Number(BinaryNode(self.Node,
                                 BinaryNode.opDivide,
                                 val.Node))

    def __pow__(self, val):
        return Number(BinaryNode(self.Node,
                                 BinaryNode.opPower,
                                 val.Node))

    def __eq__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opEQ,
                                             val.Node))

    def __ne__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opNE,
                                             val.Node))

    def __lt__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opLT,
                                             val.Node))

    def __le__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opLE,
                                             val.Node))

    def __gt__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opGT,
                                             val.Node))

    def __ge__(self, val):
        return Condition(ConditionBinaryNode(self.Node,
                                             ConditionBinaryNode.opGE,
                                             val.Node))

class UnitsError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def to_string(name, exp):
    if exp == float(int(exp)):
        exp = int(exp)
    if exp == 1:
        return '{0}'.format(name)
    elif exp != 0:
        return '{0}^{1}'.format(name, exp)
    else:
        return None

def to_string_and_append(name, exp, l):
    res = to_string(name, exp)
    if res:
        l.append(res)

def to_latex(name, exp):
    if exp == float(int(exp)):
        exp = int(exp)
    if exp == 1:
        return '{{{0}}}'.format(name)
    elif exp != 0:
        return '{{{{{0}}}^{{{1}}}}}'.format(name, exp)
    else:
        return None

def to_latex_and_append(name, U, l):
    res = to_latex(name, U)
    if res:
        l.append(res)

__latex_space__            = '\\, '
__string_unit_delimiter__  = ' '
__latex_unit_delimiter__   = ' \\, '

class base_unit(object):
    """
    Base-unit is the main building block for units and currently uses SI system as its base.
    It consists of a multiplier and 7 fundamental dimansions:
     - L: length, m
     - M: mass, kg
     - T: time, s
     - C: luminous intensity, cd
     - I: el. current, A
     - O: temperature, K
     - N: amount of a substance, mol 
    By changing the multiplier the derived units like km, cm, mm etc can be defined.
    The following basic math. operators (*, /, **) are supported and used to:
        a) create scaled base_units: number * base_unit => scaled 'base_unit' object (ie. km = 1000 * m)
        b) create compound base_units: base_unit1 * base_unit2 / base_unit3**1.2 => 'base_unit' object
           (ie. Volt = kg * m**2 / (A * s**3)
    Comparison operators (!=, ==) can be used to test if the base_units are of the same dimensions.
    Base unit can be created in the following ways:
        1)  kg = base_unit(multiplier = 1.0, M = 1)
        2a) V  = base_unit(M = 1, L = 2, A = -1, T = -3)
        2b) V  = kg * m**2 / (A * s**3) [where 'kg', 'm', 'A' and 's' are 'base_unit' objects]
        2c) mV = base_unit(multiplier = 0.001, M = 1, L = 2, A = -1, T = -3)
        2d) mV = 0.001 * V
        3a) km = base_unit(multiplier = 1000, L = 1)
        3b) km = 1000 * m
    """
    def __init__(self, **kwargs):
        self.L          = float(kwargs.get('L',          0.0)) # length, m
        self.M          = float(kwargs.get('M',          0.0)) # mass, kg
        self.T          = float(kwargs.get('T',          0.0)) # time, s
        self.C          = float(kwargs.get('C',          0.0)) # luminous intensity, cd
        self.I          = float(kwargs.get('I',          0.0)) # el. current, A
        self.O          = float(kwargs.get('K',          0.0)) # temperature, K
        self.N          = float(kwargs.get('N',          0.0)) # amount of a substance, mol
        self.multiplier = float(kwargs.get('multiplier', 1.0))

    def __eq__(self, other):
        """Returns True if all dimensions are equal"""
        return self.areDimensionsEqual(other)

    def __ne__(self, other):
        """Returns True if some of dimensions are not equal"""
        return not self.areDimensionsEqual(other)

    def isEqualTo(self, other):
        if (self.multiplier == other.multiplier) and self.areDimensionsEqual(other):
           return True
        else:
            return False

    def areDimensionsEqual(self, other):
        if ((self.M == other.M) and \
            (self.L == other.L) and \
            (self.T == other.T) and \
            (self.C == other.C) and \
            (self.I == other.I) and \
            (self.O == other.O) and \
            (self.N == other.N)):
           return True
        else:
            return False

    def __mul__(self, other):
        """
        base_unit * base_unit = base_unit
        base_unit * number    = base_unit
        """
        if isinstance(other, base_unit):
            tmp   = base_unit()
            tmp.L = self.L + other.L
            tmp.M = self.M + other.M
            tmp.T = self.T + other.T
            tmp.C = self.C + other.C
            tmp.I = self.I + other.I
            tmp.O = self.O + other.O
            tmp.N = self.N + other.N
            tmp.multiplier = self.multiplier * other.multiplier
            return tmp
        elif isinstance(other, (float, int, long)):
            tmp = deepcopy(self)
            tmp.multiplier *= float(other)
            return tmp
        else:
            raise UnitsError('Invalid right argument type ({0}) for: {1} * {2}'.format(type(other), self, other))

    def __rmul__(self, other):
        """number * base_unit = base_unit"""
        if isinstance(other, (float, int, long)):
            tmp = deepcopy(self)
            tmp.multiplier *= float(other)
            return tmp
        else:
            raise UnitsError('Invalid left argument type ({0}) for: {1} * {2}'.format(type(other), other, self))

    def __div__(self, other):
        """base_unit / base_unit = base_unit"""
        if not isinstance(other, base_unit):
            raise UnitsError('Invalid right argument type ({0}) for: {1} / {2}'.format(type(other), self, other))
        tmp   = base_unit()
        tmp.L = self.L - other.L
        tmp.M = self.M - other.M
        tmp.T = self.T - other.T
        tmp.C = self.C - other.C
        tmp.I = self.I - other.I
        tmp.O = self.O - other.O
        tmp.N = self.N - other.N
        tmp.multiplier = self.multiplier / other.multiplier
        return tmp

    def __pow__(self, exponent):
        """base_unit ** number = base_unit"""
        if (not isinstance(exponent, float)) and (not isinstance(exponent, int)):
            raise UnitsError('Invalid exponent type ({0}) for: {1} ** {2}'.format(type(exponent), self, exponent))
        tmp   = deepcopy(self)
        tmp.L = self.L * exponent
        tmp.M = self.M * exponent
        tmp.T = self.T * exponent
        tmp.C = self.C * exponent
        tmp.I = self.I * exponent
        tmp.O = self.O * exponent
        tmp.N = self.N * exponent
        tmp.multiplier = self.multiplier ** exponent
        return tmp

    def toString(self):
        """Returns the string representation of units (only)"""
        units = []
        to_string_and_append('kg',  self.M, units)
        to_string_and_append('m',   self.L, units)
        to_string_and_append('s',   self.T, units)
        to_string_and_append('cd',  self.C, units)
        to_string_and_append('A',   self.I, units)
        to_string_and_append('K',   self.O, units)
        to_string_and_append('mol', self.N, units)
        res = __string_unit_delimiter__.join(units)
        if len(res) == 0:
            res = 'none'
        return res

    def toLatex(self):
        """Returns the Latex representation of units (only)"""
        units = []
        to_latex_and_append('kg',  self.M, units)
        to_latex_and_append('m',   self.L, units)
        to_latex_and_append('s',   self.T, units)
        to_latex_and_append('cd',  self.C, units)
        to_latex_and_append('A',   self.I, units)
        to_latex_and_append('K',   self.O, units)
        to_latex_and_append('mol', self.N, units)
        res = __latex_unit_delimiter__.join(units)
        if len(res) == 0:
            res = 'none'
        return res

    def __repr__(self):
        template = 'base_unit(multiplier = {0}, M = {1}, L = {2}, T = {3}, C = {4}, A = {5}, K = {6}, N = {7})'
        return template.format(self.multiplier, self.M, self.L, self.T, self.C, self.I, self.O, self.N)

    def __str__(self):
        return '{0}*[{1}]'.format(self.multiplier, self.toString())

class unit(object):
    """
    Units consists of several base units and their exponents (stored in the 'self.units' dictionary).
    Base units are 'base_unit' objects with predefined symbol (name) like kg, m, J, N, F ...
    Therefore, the 'units' dictionary can contain entries like:
        'kg':2, 'V':1, 'J':-2 ...
    The following basic math. operators (*, /, **) are supported and used to:
        a) create quantities: number * unit => 'quantity' object
        b) create compound units: unit1 * unit2 / unit3**1.2 => 'unit' object
    Comparison operators (!=, ==) can be used to test if the units are of the same dimensions.
    Unit can be created in the following ways:
        1)  kg = unit(kg = 1)
        2a) V  = unit(V = 1)
        2b) V  = kg * m**2 / (A * s**3) [where 'kg', 'm', 'A' and 's' are 'unit' objects]
        2d) mV = 0.001 * V [where 'V' is a 'unit' object]
    There is a large set of predefined unit objects available as static data members of the 'unit' class:
        A   : 1.0 [A]
        C   : 1.0 [s A]
        F   : 1.0 [kg^-1 m^-2 s^4 A^2]
        Hz  : 1.0 [s^-1]
        J   : 1.0 [kg m^2 s^-2]
        K   : 1.0 [none]
        MHz : 1000000.0 [s^-1]
        MPa : 1000000.0 [kg m^-1 s^-2]
        N   : 1.0 [kg m s^-2]
        Ohm : 1.0 [kg m^2 s^-3 A^-2]
        P   : 1.0 [kg m^-1 s^-1]
        Pa  : 1.0 [kg m^-1 s^-2]
        St  : 0.0001 [m^2 s^-1]
        V   : 1.0 [kg m^2 s^-3 A^-1]
        W   : 1.0 [kg m^2 s^-3]
        cd  : 1.0 [cd]
        cm  : 0.01 [m]
        day : 43200.0 [s]
        dl  : 0.0001 [m^3]
        dm  : 0.1 [m]
        h   : 3600.0 [s]
        kHz : 1000.0 [s^-1]
        kJ  : 1000.0 [kg m^2 s^-2]
        kPa : 1000.0 [kg m^-1 s^-2]
        kW  : 1000.0 [kg m^2 s^-3]
        kg  : 1.0 [kg]
        km  : 1000.0 [m]
        l   : 0.001 [m^3]
        m   : 1.0 [m]
        mA  : 0.001 [A]
        mV  : 0.001 [kg m^2 s^-3 A^-1]
        min : 60.0 [s]
        mm  : 0.001 [m]
        mol : 1.0 [mol]
        ms  : 0.001 [s]
        nm  : 1e-09 [m]
        s   : 1.0 [s]
        and many more ...
    """
    @classmethod
    def init_base_units(cls):
        # Scaling factors:
        tera  = 1E+12
        giga  = 1E+9
        mega  = 1E+6
        kilo  = 1E+3
        deka  = 1E+1
        deci  = 1E-1
        centi = 1E-2
        mili  = 1E-3
        micro = 1E-6
        nano  = 1E-9
        pico  = 1E-12

        # Base units:
        kg  = base_unit(M = 1)
        m   = base_unit(L = 1)
        s   = base_unit(T = 1)
        cd  = base_unit(C = 1)
        A   = base_unit(I = 1)
        K   = base_unit(O = 1)
        mol = base_unit(N = 1)

        dimless = base_unit()

        # Time:
        ms   = mili  * s
        us   = micro * s
        min  = 60    * s
        hour = 3600  * s
        day  = 43200 * s
        Hz   = s**(-1)
        kHz  = kilo * Hz
        MHz  = mega * Hz

        # Length related:
        km = kilo  * m
        dm = deci  * m
        cm = centi * m
        mm = mili  * m
        um = micro * m
        nm = nano  * m

        # Volume:
        lit = 1E-3 * m**3
        dl  = deci * lit

        # Energy:
        N  = kg * m / (s**2)
        J  = N * m
        kJ = kilo * J
        W  = J / s
        kW = kilo * W

        # Electrical:
        V   = kg * m**2 / (A * s**3) # Volt
        C   = A * s                  # Coulomb
        F   = C / V                  # Farad
        Ohm = J * s / (C**2)
        mV  = mili * V
        mA  = mili * A

        # Pressure:
        Pa = N / m**2
        kPa = kilo * Pa
        MPa = mega * Pa

        # Viscosity
        P  = Pa * s      # Poise
        St = cm**2 / s   # Stoke

        # Base units has to be represented by a single symbol and can be used to create compound units (mV/s, Pa/m^2 ...)
        cls.__base_units__ =  { 'kg'  : kg,
                                'm'   : m,
                                's'   : s,
                                'cd'  : cd,
                                'A'   : A,
                                'K'   : K,
                                'mol' : mol,

                                'ms'  : ms,
                                'us'  : us,
                                'min' : min,
                                'h'   : hour,
                                'day' : day,
                                'Hz'  : Hz,
                                'kHz' : kHz,
                                'MHz' : MHz,

                                'km'  : km,
                                'dm'  : dm,
                                'cm'  : cm,
                                'mm'  : mm,
                                'um'  : um,
                                'nm'  : nm,

                                'l'   : lit,
                                'dl'  : dl,

                                'N'   : N,
                                'J'   : J,
                                'kJ'  : kJ,
                                'W'   : W,
                                'kW'  : kW,

                                'C'   : C,
                                'F'   : F,
                                'Ohm' : Ohm,
                                'V'   : V,
                                'mV'  : mV,
                                'mA'  : mA,

                                'Pa'  : Pa,
                                'kPa' : kPa,
                                'MPa' : MPa,

                                'P'  : P,
                                'St' : St
                              }

        supportedUnits = cls.getAllSupportedUnits()
        for key, u in list(supportedUnits.items()):
            setattr(cls, key, u)
        setattr(cls, 'tera',  1E+12)
        setattr(cls, 'giga',  1E+9)
        setattr(cls, 'mega',  1E+6)
        setattr(cls, 'kilo',  1E+3)
        setattr(cls, 'deka',  1E+1)
        setattr(cls, 'deci',  1E-1)
        setattr(cls, 'centi', 1E-2)
        setattr(cls, 'mili',  1E-3)
        setattr(cls, 'micro', 1E-6)
        setattr(cls, 'nano',  1E-9)
        setattr(cls, 'pico',  1E-12)
        
    @classmethod
    def create(cls, dictUnits):
        if not isinstance(dictUnits, dict):
            raise UnitsError('Invalid argument type ({0}); must be a dictionary'.format(type(dictUnits)))
        return unit(**dictUnits)

    @classmethod
    def getAllSupportedUnits(cls):
        supportedUnits = {}
        for key, baseunit in list(cls.__base_units__.items()):
            supportedUnits[key] = unit.create({key : 1})
        return supportedUnits

    def __init__(self, **kwargs):
        # if not already loaded, load all base units
        #if not hasattr(unit, '__base_units__'):
        #    unit.init_base_units()
        self.units = {}
        for key, exp in list(kwargs.items()):
            if key in unit.__base_units__:
                self.units[key] = exp
            else:
                raise UnitsError('Cannot find the base_unit {0} in the __base_units__ dictionary'.format(key))

    @property
    def base_unit(self):
        """Returns the 'base_unit' object built by joining dimensions from all items in the dictionary 'units'.
        It multiplies base units from the dictionary (risen on the corresponding exponent)."""
        res_unit = base_unit()
        for base_unit_name, exp in list(self.units.items()):
            if not base_unit_name in unit.__base_units__:
                raise UnitsError('Cannot find the base_unit {0} in the __base_units__ dictionary'.format(base_unit_name))
            res_unit = res_unit * (unit.__base_units__[base_unit_name] ** exp)
        return res_unit

    def __repr__(self):
        return 'unit({0})'.format(self.units)

    def __str__(self):
        return self.toString()

    def toString(self):
        units = []
        for base_unit_name, exp in self.units.items():
            to_string_and_append(base_unit_name, exp, units)
        res = __string_unit_delimiter__.join(units)
        return res

    def toLatex(self):
        units = []
        for base_unit_name, exp in self.units.items():
            to_latex_and_append(base_unit_name, exp, units)
        res = __latex_unit_delimiter__.join(units)
        return res

    def __eq__(self, other):
        return (self.base_unit == other.base_unit)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __mul__(self, other):
        # unit * number = quantity
        if isinstance(other, (float, int, long)):
            return quantity(other, self)
        # unit * unit = unit
        elif isinstance(other, unit):
            tmp       = unit()
            # First add all units from self.units
            tmp.units = deepcopy(self.units)
            # Then iterate over other.units and either add it to the tmp.units or increase the exponent
            for key, exp in other.units.items():
                if key in tmp.units:
                    tmp.units[key] += exp
                else:
                    tmp.units[key] = exp
            return tmp
        else:
            raise UnitsError('Invalid right argument type ({0}) for: {1} * {2}'.format(type(other), self, other))

    def __rmul__(self, other):
        # number * unit = quantity
        if isinstance(other, (float, int, long)):
            return quantity(other, self)
        else:
            raise UnitsError('Invalid left argument type ({0}) for: {1} * {2}'.format(type(other), other, self))

    def __div__(self, other):
        # unit / number = quantity
        if isinstance(other, (float, int, long)):
            return quantity(1.0 / other, self)
        # unit / unit = unit
        elif isinstance(other, unit):
            tmp   = unit()
            tmp.units = deepcopy(self.units)
            for key, exp in other.units.items():
                if key in tmp.units:
                    tmp.units[key] -= exp
                else:
                    tmp.units[key] = -exp
            return tmp
        else:
            raise UnitsError('Invalid right argument type ({0}) for: {1} / {2}'.format(type(other), self, other))

    def __rdiv__(self, other):
        # number / unit = quantity
        if isinstance(other, (float, int, long)):
            return quantity(other, self.__pow__(-1))
        else:
            raise UnitsError('Invalid left argument type ({0}) for: {1} / {2}'.format(type(other), other, self))

    def __pow__(self, exponent):
        # exponent must be a number
        if isinstance(exponent, float) or isinstance(exponent, int):
            n = float(exponent)
        else:
            raise UnitsError('Invalid exponent type ({0}) for: {1} ** {2}'.format(type(exponent), self, exponent))

        tmp = unit()
        for key, exp in self.units.items():
            tmp.units[key] = exp * n
        return tmp

# Achtung, achtung!!!
# Fill the __base_units__ dictionary and load all defined units into the 'unit' class
unit.init_base_units()

class quantity(object):
    """
    Quantities consists of the:
     - 'value' (float)
     - 'units' object (of 'unit' class)
    All basic math. operators (+, -, *, /, **) and math. functions are supported.
    Some functions operate only on radians/non dimensional units.
    Comparison operators (<, <=, >, >=, !=, ==) compare values in SI units and
    raise an exception if units are different; this way: 0.2 km == 200 m == 2.0e5 mm
    """
    def __init__(self, value = 0.0, units = unit()):
        if not isinstance(units, unit):
            raise UnitsError('Invalid argument type ({0})'.format(type(units)))
        self._units = units
        self._value = float(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if isinstance(new_value, quantity):
            if self.units != new_value.units:
                raise UnitsError('Cannot set a value given in: {0} to a quantity in: {1}'.format(new_value.units, self.units))
            self._value = float(new_value.scaleTo(self).value)
        elif isinstance(new_value, (float, int, long)):
            self._value = float(new_value)
        else:
            raise UnitsError('Invalid argument type ({0})'.format(type(new_value)))
    
    @property
    def units(self):
        return self._units

    @property
    def value_in_SI_units(self):
        """Returns the value in SI units"""
        return self.value * self.units.base_unit.multiplier
    
    def scaleTo(self, referrer):
        """
        Scales the value to the units given in the argument 'referrer',
        and returns 'quantity' object in units of the 'referrer' object.
        Argument 'referrer' can be 'quantity' or 'unit' object.
        """
        if isinstance(referrer, quantity):
            units = referrer.units
        elif isinstance(referrer, unit):
            units = referrer
        else:
            raise UnitsError('Invalid argument type ({0})'.format(type(referrer)))

        if self.units != units:
            raise UnitsError('Units not consistent: scale from {0} to {1}'.format(self.units, units))

        units = deepcopy(units)
        value = self.value * self.units.base_unit.multiplier / float(units.base_unit.multiplier)
        return quantity(value, units)

    def __eq__(self, other):
        if self.units != other.units:
            raise UnitsError('Units not consistent for comparison: {0} == {1}'.format(self.units, other.units))
        return self.value_in_SI_units == other.value_in_SI_units

    def __ne__(self, other):
        if self.units != other.units:
            raise UnitsError('Units not consistent for comparison: {0} != {1}'.format(self.units, other.units))
        return self.value_in_SI_units != other.value_in_SI_units

    def __lt__(self, other):
        if self.units != other.units:
            raise UnitsError('Units not consistent for comparison: {0} < {1}'.format(self.units, other.units))
        return self.value_in_SI_units < other.value_in_SI_units

    def __le__(self, other):
        if self.units != other.units:
            raise UnitsError('Units not consistent for comparison: {0} <= {1}'.format(self.units, other.units))
        return self.value_in_SI_units <= other.value_in_SI_units

    def __gt__(self, other):
        if self.units != other.units:
            raise UnitsError('Units not consistent for comparison: {0} > {1}'.format(self.units, other.units))
        return self.value_in_SI_units > other.value_in_SI_units

    def __ge__(self, other):
        if self.units != other.units:
            raise UnitsError('Units not consistent for comparison: {0} >= {1}'.format(self.units, other.units))
        return self.value_in_SI_units >= other.value_in_SI_units

    def __neg__(self):
        tmp       = deepcopy(self)
        tmp.value = -self.value
        return tmp

    def __pos__(self):
        return deepcopy(self)

    def __add__(self, other):
        if not isinstance(other, quantity):
            if isinstance(other, (float, int, long)):
                other = quantity(other)
            else:
                raise UnitsError('Invalid right argument type ({0}) for: {1} + {2}'.format(type(other), self, other))
        if self.units != other.units:
            raise UnitsError('Units not consistent: {0} + {1}'.format(self, other))

        tmp       = deepcopy(self)
        tmp.value = self.value + other.scaleTo(self).value
        return tmp

    def __radd__(self, other):
        if not isinstance(other, quantity):
            if isinstance(other, (float, int, long)):
                other = quantity(other)
            else:
                raise UnitsError('Invalid left argument type ({0}) for: {1} + {2}'.format(type(other), other, self))
        return other.__mul__(self)

    def __sub__(self, other):
        if not isinstance(other, quantity):
            if isinstance(other, (float, int, long)):
                other = quantity(other)
            else:
                raise UnitsError('Invalid right argument type ({0}) for: {1} - {2}'.format(type(other), self, other))
        if self.units != other.units:
            raise UnitsError('Units not consistent: {0} - {1}'.format(self, other))

        tmp       = deepcopy(self)
        tmp.value = self.value - other.scaleTo(self).value
        return tmp

    def __rsub__(self, other):
        if not isinstance(other, quantity):
            if isinstance(other, (float, int, long)):
                other = quantity(other)
            else:
                raise UnitsError('Invalid left argument type ({0}) for: {1} - {2}'.format(type(other), other, self))
        return other.__add__(self)

    def __mul__(self, other):
        if not isinstance(other, quantity):
            if isinstance(other, (float, int, long)):
                other = quantity(other)
            else:
                raise UnitsError('Invalid right argument type ({0}) for: {1} * {2}'.format(type(other), self, other))

        units = self.units * other.units
        value = self.value * other.value
        return quantity(value, units)

    def __rmul__(self, other):
        if isinstance(other, (float, int, long)):
            other = quantity(other)
        else:
            raise UnitsError('Invalid left argument type ({0}) for: {1} * {2}'.format(type(other), other, self))
        return other.__mul__(self)

    def __div__(self, other):
        if not isinstance(other, quantity):
            if isinstance(other, (float, int, long)):
                other = quantity(other)
            else:
                raise UnitsError('Invalid right argument type ({0}) for: {1} / {2}'.format(type(other), self, other))

        units = self.units / other.units
        value = self.value / other.value
        return quantity(value, units)

    def __rdiv__(self, other):
        if isinstance(other, (float, int, long)):
            other = quantity(other)
        else:
            raise UnitsError('Invalid left argument type ({0}) for: {1} / {2}'.format(type(other), other, self))
        return other.__div__(self)

    def __pow__(self, exponent):
        if isinstance(exponent, float) or isinstance(exponent, int):
            n = float(exponent)
        else:
            raise UnitsError('Invalid exponent type ({0}) for: {1} ** {2}'.format(type(exponent), self, exponent))
        
        units = self.units ** n
        value = self.value ** n
        return quantity(value, units)

    def toString(self):
        return '{0} {1}'.format(self.value, self.units.toString())

    def toLatex(self):
        return '{0}{1}{2}'.format(self.value, __latex_space__, self.units.toLatex())

    def __repr__(self):
        return 'quantity({0}, {1})'.format(self.value, repr(self.units))

    def __str__(self):
        return self.toString()

def exp(q):
    tmp       = deepcopy(q)
    tmp.value = math.exp(q.value)
    return tmp

def pow(q, n):
    return q ** n

def log(q):
    tmp       = deepcopy(q)
    tmp.value = math.log(q.value)
    return tmp

def log10(q):
    tmp       = deepcopy(q)
    tmp.value = math.log10(q.value)
    return tmp

def sqrt(q):
    return q ** 0.5

def sin(q):
    tmp       = deepcopy(q)
    tmp.value = math.sin(q.value)
    return tmp

def cos(q):
    tmp       = deepcopy(q)
    tmp.value = math.cos(q.value)
    return tmp

def tan(q):
    tmp       = deepcopy(q)
    tmp.value = math.tan(q.value)
    return tmp

def asin(q):
    tmp       = deepcopy(q)
    tmp.value = math.asin(q.value)
    return tmp

def acos(q):
    tmp       = deepcopy(q)
    tmp.value = math.acos(q.value)
    return tmp

def atan(q):
    tmp       = deepcopy(q)
    tmp.value = math.atan(q.value)
    return tmp

def sinh(q):
    tmp       = deepcopy(q)
    tmp.value = math.sinh(q.value)
    return tmp

def cosh(q):
    tmp       = deepcopy(q)
    tmp.value = math.cosh(q.value)
    return tmp

def tanh(q):
    tmp       = deepcopy(q)
    tmp.value = math.tanh(q.value)
    return tmp

def asinh(q):
    tmp       = deepcopy(q)
    tmp.value = math.asinh(q.value)
    return tmp

def acosh(q):
    tmp       = deepcopy(q)
    tmp.value = math.acosh(q.value)
    return tmp

def atanh(q):
    tmp       = deepcopy(q)
    tmp.value = math.atanh(q.value)
    return tmp

def abs(q):
    tmp       = deepcopy(q)
    tmp.value = math.abs(q.value)
    return tmp

def ceil(q):
    tmp       = deepcopy(q)
    tmp.value = math.ceil(q.value)
    return tmp

def floor(q):
    tmp       = deepcopy(q)
    tmp.value = math.floor(q.value)
    return tmp
