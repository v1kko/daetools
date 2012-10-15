import sys
from daetools.pyDAE import *

portTemplate = """connector %(port)s
public
%(variables)s
end %(port)s;
"""

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
    def __init__(self, simulation):
        self.ports         = {}
        self.models        = {}
        self.initialValues = {}
        self.topLevelModel = simulation.m
        self.simulation    = simulation

        self.models[self.simulation.m.__class__] = self.simulation.m
        self.collectObjects(self.simulation.m)

    def collectObjects(self, model):
        for port in model.Ports:
            if not port.__class__ in self.ports:
                self.ports[port.__class__] = port

        for model in model.Models:
            if not model.__class__ in self.models:
                self.models[model.__class__] = model

        print 'Models collected:', self.models
        print 'Ports collected:', self.ports

    def export(self):
        result = ''
        for port_class, port in self.ports.items():
            result += self.exportPort(port) 
        for model_class, model in self.models.items():
            result += self.exportModel(model)
        print result
        #print '\n'.join('  {0} = {1};'.format(key, value) for key, value in self.initialValues.items())
        print ','.join('{0}={1}'.format(key, value) for key, value in self.initialValues.items())
        
    def exportPort(self, port):
        return ''
    
    def exportModel(self, model):
        sParameters = []
        sVariables  = []
        sUnits      = []
        sEquations  = []

        varTypes   = self.simulation.VariableTypes
        initValues = self.simulation.InitialValues

        # cnNormal = 0, cnDifferential = 1, cnAssigned = 2
        for domain in model.Domains:
            canonicalName = nameFormat(daeGetRelativeName(self.topLevelModel, domain))
            sParameters.append('  parameter Integer {0}_np = {1} "Number of points in domain {0}";'.format(nameFormat(domain.Name),
                                                                                                          domain.NumberOfPoints) )
        
            sParameters.append('  parameter Real[{0}_np] {1}(unit="{2}") = {{{3}}} "{4}";\n'.format(nameFormat(domain.Name),
                                                                                                    nameFormat(domain.Name),
                                                                                                    unitFormat(domain.Units),
                                                                                                    ','.join(str(p) for p in domain.Points),
                                                                                                    domain.Description) )
            self.initialValues[canonicalName+'_np'] = str(domain.NumberOfPoints)
            self.initialValues[canonicalName]       = '{{{0}}}'.format(','.join(str(p) for p in domain.Points))
                                                                                                        
        for parameter in model.Parameters:
            canonicalName = nameFormat(daeGetRelativeName(self.topLevelModel, parameter))
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

            sParameters.append('  parameter Real{0} {1}(unit="{2}") = {3} "{4}";'.format(domains,
                                                                                         nameFormat(parameter.Name),
                                                                                         unitFormat(parameter.Units),
                                                                                         values,
                                                                                         parameter.Description) )
            self.initialValues[canonicalName] = values

        for variable in model.Variables:
            canonicalName = nameFormat(daeGetRelativeName(self.topLevelModel, variable))
            startValue    = ''
            assignedValue = ''
            if len(variable.Domains) > 0:
                domains = '[{0}]'.format(','.join(nameFormat(d.Name)+'_np' for d in variable.Domains))
                isDiff     = False
                isAssigned = False
                isNormal   = False
                for vt in varTypes[variable.OverallIndex : variable.OverallIndex+variable.NumberOfPoints]:
                    if vt == cnDifferential:
                        isDiff = True
                    elif vt == cnAssigned:
                        isAssigned = True
                    else:
                        isNormal = True

                if isDiff and not isAssigned and not isNormal:
                    varType = 'all differential'
                    startValue = 'start = 0.0, '
                elif not isDiff and isAssigned and not isNormal:
                    varType = 'all assigned'
                elif not isDiff and not isAssigned and isNormal:
                    varType = 'all state'
                elif isDiff and not isAssigned and isNormal:
                    varType = 'differential or state'
                    startValue = 'start = 0.0, '
                elif isDiff and isAssigned and not isNormal:
                    varType = 'differential or assigned'
                    startValue = 'start = 0.0, '
                elif not isDiff and isAssigned and isNormal:
                    varType = 'assigned or state'
                else:
                    varType = 'differential, assigned or state'
                    startValue = 'start = 0.0, '                    

            else:
                domains = ''
                if varTypes[variable.OverallIndex] == cnDifferential:
                    varType = 'differential'
                    startValue = 'start = {0}, '.format(variable.GetValue())

                elif varTypes[variable.OverallIndex] == cnAssigned:
                    varType = 'assigned'
                    assignedValue = '= {0}'.format(variable.GetValue())

                else:
                    varType = 'state'

            sVariables.append('  // Type of variable(s): {0}'.format(varType))
            sVariables.append('  Real{0} {1}({2}unit="{3}") {4} "{5}";\n'.format(domains,
                                                                                     nameFormat(variable.Name),
                                                                                     startValue,
                                                                                     unitFormat(variable.VariableType.Units),
                                                                                     assignedValue,
                                                                                     variable.Description) )
            self.initialValues[canonicalName] = assignedValue
                                                                          
        for equation in model.Equations:
            for eeinfo in equation.EquationExecutionInfos:
                node = eeinfo.Node
                sEquations.append('  {0} = 0;'.format(self.processNode(node)))

        dictModel = {
                      'model'             : nameFormat(model.Name),
                      'parameters'        : '\n'.join(sParameters),
                      'variables'         : '\n'.join(sVariables),
                      'units'             : '\n'.join(sUnits),
                      'equations'         : '\n'.join(sEquations)
                    }
        result = modelTemplate % dictModel
        return result
        
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