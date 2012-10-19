import sys, numpy, traceback
from daetools.pyDAE import *

def fixName(name):
    return name.replace('&', '').replace(';', '')

class daeExpressionFormatter(object):
    def __init__(self):
        # Equation and condition node formatting settings

        # Index base in arrays:
        #  - Modelica, Fortran use 1
        #  - daetools, python, c/c++ use 0
        self.indexBase = 1

        # Use relative names (relative to domains/parameters/variables model) or full canonical names
        # If we are in model root.comp1 then variables' names could be:
        #   if useRelativeNames is True:
        #       name = 'comp2.Var' (relative to parent comp1)
        #   else:
        #       name = 'root.comp1.comp2.Var' (full canonical name)
        self.useRelativeNames = True

        # One- and multi-dimensional parameters/variables and domain points
        self.indexLeftBracket  = '['
        self.indexRightBracket = ']'
        self.indexFormat = 'python_index_style'
        # if format is python_index_style:
        #     var(i1, i2, ..., in)
        # else if indexFormat is c_index_style:
        #     var[i1][i2]...[in]

        # String format for the time derivative
        #    ie. der(variable[1, 2])
        self.derivative = 'der({variable}{indexes})'

        # Operators
        self.AND   = 'and'
        self.OR    = 'or'
        self.NOT   = 'not'

        self.EQ    = '=='
        self.NEQ   = '!='
        self.LT    = '<'
        self.LTEQ  = '<='
        self.GT    = '>'
        self.GTEQ  = '>='

        # Mathematical operators
        self.PLUS   = '+'
        self.MINUS  = '-'
        self.MULTI  = '*'
        self.DIVIDE = '/'
        self.POWER  = '**'

        # Mathematical functions
        self.SIN    = 'sin'
        self.COS    = 'cos'
        self.TAN    = 'tan'
        self.ASIN   = 'asin'
        self.ACOS   = 'acos'
        self.ATAN   = 'atan'
        self.EXP    = 'exp'
        self.SQRT   = 'sqrt'
        self.LOG    = 'log'
        self.LOG10  = 'log10'
        self.FLOOR  = 'floor'
        self.CEIL   = 'ceil'
        self.ABS    = 'abs'
        self.MIN    = 'min'
        self.MAX    = 'max'

        # Current time in simulation
        self.TIME   = 'time'

        # Internal data: model will be set by the analyzer
        self.model = None

    def formatDomain(self, domainCanonicalName, index):
        # python_index_style: domain(0)
        # c_index_style: domain[0]
        # ACHTUNG, ACHTUNG!! Take care of indexing
        if self.useRelativeNames:
            name = daeGetRelativeName(self.model.CanonicalName, domainCanonicalName)
        else:
            name = domainCanonicalName

        if self.indexFormat == 'python_index_style':
            res = '{0}{1}{2}{3}'.format(name, self.indexLeftBracket, index + self.indexBase, self.indexRightBracket)
        elif self.indexFormat == 'c_index_style':
            res = '{0}{1}{2}{3}'.format(name, self.indexLeftBracket, index + self.indexBase, self.indexRightBracket)
        else:
            raise RuntimeError('Unsupported index style')
        return res

    def formatParameter(self, parameterCanonicalName, domainIndexes):
        # python_index_style: parameter(0, 1, 2)
        # c_index_style: parameter[0][1][2]
        # ACHTUNG, ACHTUNG!! Take care of indexing
        if self.useRelativeNames:
            name = daeGetRelativeName(self.model.CanonicalName, parameterCanonicalName)
        else:
            name = parameterCanonicalName

        domainindexes = ''
        if len(domainIndexes) > 0:
            if self.indexFormat == 'python_index_style':
                domainindexes = self.indexLeftBracket + ','.join([str(di+self.indexBase) for di in domainIndexes]) + self.indexRightBracket
            elif self.indexFormat == 'c_index_style':
                domainindexes = ''.join([self.indexLeftBracket + str(di+self.indexBase) + self.indexRightBracket for di in domainIndexes])
            else:
                raise RuntimeError('Unsupported index style')

        res = '{0}{1}'.format(name, domainindexes)
        return res

    def formatVariable(self, variableCanonicalName, domainIndexes):
        # python_index_style: variable(0, 1, 2)
        # c_index_style: variable[0][1][2]
        # ACHTUNG, ACHTUNG!! Take care of indexing
        if self.useRelativeNames:
            name = daeGetRelativeName(self.model.CanonicalName, variableCanonicalName)
        else:
            name = variableCanonicalName

        domainindexes = ''
        if len(domainIndexes) > 0:
            if self.indexFormat == 'python_index_style':
                domainindexes = self.indexLeftBracket + ','.join([str(di+self.indexBase) for di in domainIndexes]) + self.indexRightBracket
            elif self.indexFormat == 'c_index_style':
                domainindexes = ''.join([self.indexLeftBracket + str(di+self.indexBase) + self.indexRightBracket for di in domainIndexes])
            else:
                raise RuntimeError('Unsupported index style')

        res = '{0}{1}'.format(name, domainindexes)
        return res

    def formatTimeDerivative(self, variableCanonicalName, domainIndexes, order):
        # python_index_style: derivative(variable(0, 1, 2))
        # c_index_style: derivative(variable[0][1][2])
        # ACHTUNG, ACHTUNG!! Take care of indexing
        if self.useRelativeNames:
            name = daeGetRelativeName(self.model.CanonicalName, variableCanonicalName)
        else:
            name = variableCanonicalName

        domainindexes = ''
        if len(domainIndexes) > 0:
            if self.indexFormat == 'python_index_style':
                domainindexes = self.indexLeftBracket + ','.join([str(di+self.indexBase) for di in domainIndexes]) + self.indexRightBracket
            elif self.indexFormat == 'c_index_style':
                domainindexes = ''.join([self.indexLeftBracket + str(di+self.indexBase) + self.indexRightBracket for di in domainIndexes])
            else:
                raise RuntimeError('Unsupported index style')

        res = self.derivative.format(variable = name, indexes = domainindexes)
        return res

    def formatQuantity(self, quantity):
        # Formats constants/quantities in equations that have a value and units
        return str(quantity.value) # + ' ' + self.formatUnit(quantity.units)

    def formatUnit(self, unit):
        # Format: m.kg2.s-1 meaning m * kg**2 / s
        res = []
        for u, exp in unit.unitDictionary.items():
            if exp >= 0:
                if exp == 1:
                    res.append('{0}'.format(u))
                elif int(exp) == exp:
                    res.append('{0}{1}'.format(u, int(exp)))
                else:
                    res.append('{0}{1}'.format(u, exp))

        for u, exp in unit.unitDictionary.items():
            if exp < 0:
                if int(exp) == exp:
                    res.append('{0}{1}'.format(u, int(exp)))
                else:
                    res.append('{0}{1}'.format(u, exp))

        return '.'.join(res)

    def formatRuntimeConditionNode(self, node):
        res = ''
        if isinstance(node, condUnaryNode):
            n = '(' + self.formatRuntimeConditionNode(node.Node) + ')'
            if node.LogicalOperator == eNot:
                res = '{0}{1}'.format(self.NOT, n)
            else:
                raise RuntimeError('Not supported unary logical operator')

        elif isinstance(node, condBinaryNode):
            left  = '(' + self.formatRuntimeConditionNode(node.LNode) + ')'
            right = '(' + self.formatRuntimeConditionNode(node.RNode) + ')'

            if node.LogicalOperator == eAnd:
                res = '{0} {1} {2}'.format(left, self.AND, right)
            elif node.LogicalOperator == eOr:
                res = '{0} {1} {2}'.format(left, self.OR, right)
            else:
                raise RuntimeError('Not supported binary logical operator')

        elif isinstance(node, condExpressionNode):
            left  = '(' + self.formatRuntimeNode(node.LNode) + ')'
            right = '(' + self.formatRuntimeNode(node.RNode) + ')'

            if node.ConditionType == eNotEQ: # !=
                res = '{0} {1} {2}'.format(left, self.NEQ, right)
            elif node.ConditionType == eEQ: # ==
                res = '{0} {1} {2}'.format(left, self.EQ, right)
            elif node.ConditionType == eGT: # >
                res = '{0} {1} {2}'.format(left, self.GT, right)
            elif node.ConditionType == eGTEQ: # >=
                res = '{0} {1} {2}'.format(left, self.GTEQ, right)
            elif node.ConditionType == eLT: # <
                res = '{0} {1} {2}'.format(left, self.LT, right)
            elif node.ConditionType == eLTEQ: # <=
                res = '{0} {1} {2}'.format(left, self.LTEQ, right)
            else:
                raise RuntimeError('Not supported condition type')
        else:
            raise RuntimeError('Not supported condition node')

        return res

    def formatRuntimeNode(self, node):
        res = ''
        if isinstance(node, adConstantNode):
            res = self.formatQuantity(node.Quantity)

        elif isinstance(node, adTimeNode):
            res = self.TIME

        elif isinstance(node, adUnaryNode):
            n = '(' + self.formatRuntimeNode(node.Node) + ')'

            if node.Function == eSign:
                res = '{0}{1}'.format(self.MINUS, n)
            elif node.Function == eSqrt:
                res = '{0}({1})'.format(self.SQRT, n)
            elif node.Function == eExp:
                res = '{0}({1})'.format(self.EXP, n)
            elif node.Function == eLog:
                res = '{0}({1})'.format(self.LOG10, n)
            elif node.Function == eLn:
                res = '{0}({1})'.format(self.LOG, n)
            elif node.Function == eAbs:
                res = '{0}({1})'.format(self.ABS, n)
            elif node.Function == eSin:
                res = '{0}({1})'.format(self.SIN, n)
            elif node.Function == eCos:
                res = '{0}({1})'.format(self.COS, n)
            elif node.Function == eTan:
                res = '{0}({1})'.format(self.TAN, n)
            elif node.Function == eArcSin:
                res = '{0}({1})'.format(self.ASIN, n)
            elif node.Function == eArcCos:
                res = '{0}({1})'.format(self.ACOS, n)
            elif node.Function == eArcTan:
                res = '{0}({1})'.format(self.ATAN, n)
            elif node.Function == eCeil:
                res = '{0}({1})'.format(self.CEIL, n)
            elif node.Function == eFloor:
                res = '{0}({1})'.format(self.FLOOR, n)
            else:
                raise RuntimeError('Not supported unary function')

        elif isinstance(node, adBinaryNode):
            left  = '(' + self.formatRuntimeNode(node.LNode) + ')'
            right = '(' + self.formatRuntimeNode(node.RNode) + ')'

            if node.Function == ePlus:
                res = '{0} {1} {2}'.format(left, self.PLUS, right)
            elif node.Function == eMinus:
                res = '{0} {1} {2}'.format(left, self.MINUS, right)
            elif node.Function == eMulti:
                res = '{0} {1} {2}'.format(left, self.MULTI, right)
            elif node.Function == eDivide:
                res = '{0} {1} {2}'.format(left, self.DIVIDE, right)
            elif node.Function == ePower:
                res = '{0} {1} {2}'.format(left, self.POWER, right)
            elif node.Function == eMin:
                res = '{0}({1}, {2})'.format(self.MIN, left, right)
            elif node.Function == eMax:
                res = '{0}({1}, {2})'.format(self.MAX, left, right)
            else:
                raise RuntimeError('Not supported binary function')

        elif isinstance(node, adScalarExternalFunctionNode):
            raise RuntimeError('External functions are not supported')

        elif isinstance(node, adVectorExternalFunctionNode):
            raise RuntimeError('External functions are not supported')

        elif isinstance(node, adDomainIndexNode):
            res = self.formatDomain(fixName(node.Domain.CanonicalName), node.Index)

        elif isinstance(node, adRuntimeParameterNode):
            res = self.formatParameter(fixName(node.Parameter.CanonicalName), node.DomainIndexes)

        elif isinstance(node, adRuntimeVariableNode):
            res = self.formatVariable(fixName(node.Variable.CanonicalName), node.DomainIndexes)

        elif isinstance(node, adRuntimeTimeDerivativeNode):
            res = self.formatTimeDerivative(fixName(node.Variable.CanonicalName), node.DomainIndexes, node.Order)

        else:
            raise RuntimeError('Not supported node')

        return res
