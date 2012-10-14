import sys
from daetools.pyDAE import *

modelTemplate = """class %(model)s
public
// Parameters:
%(parameters)s

// Variables:
%(variables)s

// Components:
%(units)s

equation
%(equations)s

end %(model)s;
"""

def nameFormat(name):
    return name.replace('&', '').replace(';', '')

def unitFormat(unit):
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
    
class daeModelicaExport:
    def __init__(self, model):
        self.model = model
        
    def export(self):
        sParameters = []
        sVariables  = []
        sUnits      = []
        sEquations  = []
        
        for domain in self.model.Domains:
            sParameters.append( '    parameter Integer {0}_np = {1} "Number of points in domain {0}";'.format(nameFormat(domain.Name),
                                                                                                              domain.NumberOfPoints) )
            sParameters.append( '    parameter Real[{0}_np] {1}(unit="{2}") = {{{3}}} "{4}";'.format(nameFormat(domain.Name),
                                                                                                     nameFormat(domain.Name),
                                                                                                     unitFormat(domain.Units),
                                                                                                     ','.join(str(p) for p in domain.Points),
                                                                                                     domain.Description) )
        for parameter in self.model.Parameters:
            values = ''
            if len(parameter.Domains) > 0:
                domains = '[{0}]'.format(','.join(nameFormat(d.Name)+'_np' for d in parameter.Domains))
                _vals = []
                for domain in parameter.Domains:
                    _vals.append('{{{0}}}'.format(','.join(str(p) for p in domain.Points)))
                values = '{{{0}}}'.format(','.join(_vals))
            else:
                domains = ''
                values = str(parameter.GetValue())

            sParameters.append( '    parameter Real{0} {1}(unit="{2}") = {3} "{4}";'.format(domains,
                                                                                            nameFormat(parameter.Name),
                                                                                            unitFormat(parameter.Units),
                                                                                            values,
                                                                                            parameter.Description) )

        for variable in self.model.Variables:
            values = ''
            if len(variable.Domains) > 0:
                domains = '[{0}]'.format(','.join(nameFormat(d.Name)+'_np' for d in variable.Domains))
                #_vals = []
                #for domain in variable.Domains:
                #    _vals.append('{{{0}}}'.format(','.join(str(p) for p in domain.Points)))
                #values = '{{{0}}}'.format(','.join(_vals))
                values = '0.0'
            else:
                domains = ''
                values = str(variable.GetValue())

            sVariables.append( '    Real{0} {1}(start={2}, unit="{3}") "{4}";'.format(domains,
                                                                                      nameFormat(variable.Name),
                                                                                      values,
                                                                                      unitFormat(variable.VariableType.Units),
                                                                                      variable.Description) )
                                                                          
        for equation in self.model.Equations:
            for eeinfo in equation.EquationExecutionInfos:
                node = eeinfo.Node
                sEquations.append('    {0} = 0;'.format(self.processNode(node)))

        dictModel = {
                      'model'      : nameFormat(self.model.Name),
                      'parameters' : '\n'.join(sParameters),
                      'variables'  : '\n'.join(sVariables),
                      'units'      : '\n'.join(sUnits),
                      'equations'  : '\n'.join(sEquations)
                    }
        print modelTemplate % dictModel
        
    def processNode(self, node):
        res = ''
        if isinstance(node, adConstantNode):
            res = '{0}'.format(node.Quantity.value) #, node.Quantity.units)

        elif isinstance(node, adTimeNode):
            res = 'time'

        elif isinstance(node, adUnaryNode):
            n = '(' + self.processNode(node.Node) + ')'

            if node.Function == ePlus:
                res = '-{0}'.format(n)
            else:
                raise RuntimeError('Not supported unary function')

        elif isinstance(node, adBinaryNode):
            left  = '(' + self.processNode(node.LNode) + ')'
            right = '(' + self.processNode(node.RNode) + ')' 

            if node.Function == ePlus:
                res = '{0} + {1}'.format(left, right)
            elif node.Function == eMinus:
                res = '{0} - {1}'.format(left, right)
            elif node.Function == eMulti:
                res = '{0} * {1}'.format(left, right)
            elif node.Function == eDivide:
                res = '{0} / {1}'.format(left, right)
            elif node.Function == ePower:
                res = '{0} ** {1}'.format(left, right)
            elif node.Function == eMin:
                res = 'min({0}, {1})'.format(left, right)
            elif node.Function == eMax:
                res = 'max({0}, {1})'.format(left, right)
            else:
                raise RuntimeError('Not supported binary function')

        elif isinstance(node, adScalarExternalFunctionNode):
            pass

        elif isinstance(node, adVectorExternalFunctionNode):
            pass

        elif isinstance(node, adDomainIndexNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            res = '{0}[{1}]'.format(nameFormat(node.Domain.Name), node.Index+1)

        elif isinstance(node, adRuntimeParameterNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            domainindexes = ''
            if len(node.DomainIndexes) > 0:
                domainindexes = '[{0}]'.format(','.join([str(di + 1) for di in node.DomainIndexes]))

            res = '{0}{1}'.format(nameFormat(node.Parameter.Name), domainindexes)

        elif isinstance(node, adRuntimeVariableNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            domainindexes = ''
            if len(node.DomainIndexes) > 0:
                domainindexes = '[{0}]'.format(','.join([str(di + 1) for di in node.DomainIndexes]))

            res = '{0}{1}'.format(nameFormat(node.Variable.Name), domainindexes)

        elif isinstance(node, adRuntimeTimeDerivativeNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            domainindexes = ''
            if len(node.DomainIndexes) > 0:
                domainindexes = '[{0}]'.format(','.join([str(di + 1) for di in node.DomainIndexes]))

            res = 'der({0}{1})'.format(nameFormat(node.Variable.Name), domainindexes)

        else:
            raise RuntimeError('Not supported node')

        return res

        
"""
print ' '
equations = simulation.m.Equations
eq = equations[0]
eeinfos = eq.EquationExecutionInfos
#print eeinfos

eeinfo = eeinfos[0]
node = eeinfo.EquationEvaluationNode #eq.Residual.Node
print node, type(node)
if isinstance(node, adBinaryNode):
    print node.LNode, node.RNode, node.Function
print ' '
"""