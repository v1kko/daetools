import sys
from daetools.pyDAE import *

portTemplate = """connector %(port)s
%(variables)s
end %(port)s;

"""

modelTemplate = """class %(model)s
/* Import libs */
  import Modelica.Math.*;
%(parameters)s
%(variables)s
%(ports)s
%(components)s

equation
%(port_connections)s
%(equations)s
%(stns)s
end %(model)s;

"""

wrapperTemplate = """class %(model)s_simulation
  /* WARNINGS:
%(warnings)s
  */

%(model_instance)s

  annotation(experiment(StartTime = %(start_time)s, StopTime = %(end_time)s, Tolerance = %(tolerance)s));
end %(model)s_simulation;

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

class exportNodeContext(object):
    def __init__(self, model):
        self.model = model
        
class daeModelicaExport(object):
    def __init__(self, simulation):
        self.ports                   = {}
        self.models                  = {}
        self.parametersValues        = {}
        self.assignedVariablesValues = {}
        self.initialConditions       = {}
        self.warnings                = []
        self.topLevelModel           = simulation.m
        self.simulation              = simulation
        self.variableTypes           = simulation.VariableTypes
        self.initialVariableValues   = simulation.InitialValues

        self.models[self.simulation.m.__class__.__name__] = self.simulation.m
        self.collectObjects(self.simulation.m)

    def collectObjects(self, model):
        for port in model.Ports:
            if not port.__class__.__name__ in self.ports:
                self.ports[port.__class__.__name__] = port

        for model in model.Components:
            if not model.__class__.__name__ in self.models:
                self.models[model.__class__.__name__] = model

        for model in model.Components:
            self.collectObjects(model)
            
        print 'Models collected:', self.models.keys()
        print 'Ports collected:', self.ports.keys()

    def export(self):
        result = ''
        for port_class, port in self.ports.items():
            result += self.exportPort(port) 
        for model_class, model in self.models.items():
            result += self.exportModel(model)

        self.collectInitialValuesFromModel(self.topLevelModel)

        f = open('%s_export.mo' % nameFormat(self.topLevelModel.Name), "w")
        f.write(result)

        model_instance = '  %s m(' % self.topLevelModel.__class__.__name__
        indent = ' ' * len(model_instance)
        
        params             = ['{0} = {1}'.format(key, value) for key, value in self.parametersValues.items()]
        assigned           = ['{0} = {1}'.format(key, value) for key, value in self.assignedVariablesValues.items()]
        initial_conditions = ['{0}(start = {1})'.format(key, value) for key, value in self.initialConditions.items()]

        arguments = params + assigned + initial_conditions
        arguments.sort(key=str.lower)
        join_format = ',\n%s' % indent
        arguments = join_format.join(arguments)

        model_instance = model_instance + arguments + ');'
        dictModel = {
                       'model'          : self.topLevelModel.__class__.__name__,
                       'model_instance' : model_instance,
                       'warnings'       : '\n'.join(self.warnings),
                       'start_time'     : 0.0,
                       'end_time'       : self.simulation.TimeHorizon,
                       'tolerance'      : daeGetConfig().GetFloat('daetools.IDAS.relativeTolerance', 1e-5)
                    }
        wrapper = wrapperTemplate % dictModel
        f.write(wrapper)
        f.close()
        
    def exportPort(self, port):
        sVariables  = []

        if len(port.Domains) > 0:
            raise RuntimeError('Modelica ports cannot contain domains')

        if len(port.Parameters) > 0:
            raise RuntimeError('Modelica ports cannot contain domains')

        for variable in port.Variables:
            if len(variable.Domains) > 0:
                raise RuntimeError('Modelica ports cannot contain distributed variables')
            else:
                value = variable.GetValue()
                if self.variableTypes[variable.OverallIndex] == cnDifferential:
                    startValue = 'start = 0.0, '

            sVariables.append('  Real {0}(unit="{1}") "{2}";'.format(nameFormat(variable.Name),
                                                                     unitFormat(variable.VariableType.Units),
                                                                     variable.Description) )

        dictPort = {
                     'port'      : nameFormat(port.__class__.__name__),
                     'variables' : '\n'.join(sVariables)
                   }
        result = portTemplate % dictPort
        return result
    
    def exportModel(self, model):
        sParameters      = []
        sVariables       = []
        sPorts           = []
        sComponents      = []
        sPortConnections = []
        sEquations       = []
        sSTNs            = []

        for domain in model.Domains:
            canonicalName = nameFormat(daeGetRelativeName(self.topLevelModel, domain))
            sParameters.append('  parameter Integer {0}_np "Number of points in domain {0}";'.format(nameFormat(domain.Name)))
            sParameters.append('  parameter Real[{0}_np] {1}(unit="{2}") "{3}";\n'.format(nameFormat(domain.Name),
                                                                                          nameFormat(domain.Name),
                                                                                          unitFormat(domain.Units),
                                                                                          domain.Description) )
            self.parametersValues[canonicalName+'_np'] = str(domain.NumberOfPoints)
            self.parametersValues[canonicalName]       = '{{{0}}}'.format(','.join(str(p) for p in domain.Points))
                                                                                                        
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

            sParameters.append('  parameter Real{0} {1}(unit="{2}") "{3}";'.format(domains,
                                                                                   nameFormat(parameter.Name),
                                                                                   unitFormat(parameter.Units),
                                                                                   parameter.Description) )
            self.parametersValues[canonicalName] = values

        for variable in model.Variables:
            canonicalName = nameFormat(daeGetRelativeName(self.topLevelModel, variable))
            startValue    = ''
            if len(variable.Domains) > 0:
                domains = '[{0}]'.format(','.join(nameFormat(d.Name)+'_np' for d in variable.Domains))
                isDiff     = False
                for vt in self.variableTypes[variable.OverallIndex : variable.OverallIndex+variable.NumberOfPoints]:
                    if vt == cnDifferential:
                        isDiff = True

                if isDiff:
                    startValue = 'start = 0.0, '
                    self.warnings.append('    - {0} is differential and distributed and its initial conditions should be double-checked'.format(canonicalName))

                varType = 'distributed differential'

            else:
                domains = ''
                if self.variableTypes[variable.OverallIndex] == cnDifferential:
                    varType = 'differential'
                    startValue = 'start = 0.0, '
                elif self.variableTypes[variable.OverallIndex] == cnAssigned:
                    varType = 'assigned'
                else:
                    varType = 'state'

            sVariables.append('  /* type: {0} */'.format(varType))
            sVariables.append('  Real{0} {1}({2}unit="{3}") "{4}";'.format(domains,
                                                                           nameFormat(variable.Name),
                                                                           startValue,
                                                                           unitFormat(variable.VariableType.Units),
                                                                           variable.Description) )

        context = exportNodeContext(model)
        for equation in model.Equations:
            sEquations.append('\n  /* {0} */'.format(equation.Description))
            for eeinfo in equation.EquationExecutionInfos:
                node = eeinfo.Node
                sEquations.append('  {0} = 0;'.format(self.processNode(node, context)))

        for port_connection in model.PortConnections:
            nameFrom = nameFormat(daeGetRelativeName(model, port_connection.PortFrom))
            nameTo   = nameFormat(daeGetRelativeName(model, port_connection.PortTo))
            sPortConnections.append('  connect({0}, {1});'.format(nameFrom, nameTo))
        
        for port in model.Ports:
            if port.Type == eInletPort:
                portType = 'inlet'
            elif port.Type == eOutletPort:
                portType = 'outlet'
            else:
                portType = 'unknown'
            sPorts.append('  /* type: {0} */'.format(portType))
            sPorts.append('  {0} {1} "{2}";'.format(port.__class__.__name__,
                                                    nameFormat(port.Name),
                                                    port.Description))

        for stn in model.STNs:
            pass

        for component in model.Components:
            sComponents.append('  {0} {1} "{2}";'.format(component.__class__.__name__,
                                                         nameFormat(component.Name),
                                                         component.Description))

        _ports            = ('\n/* Ports */      \n' if len(sPorts)      else '') + '\n'.join(sPorts)
        _parameters       = ('\n/* Parameters */ \n' if len(sParameters) else '') + '\n'.join(sParameters)
        _variables        = ('\n/* Variables */  \n' if len(sVariables)  else '') + '\n'.join(sVariables)
        _components       = ('\n/* Components */ \n' if len(sComponents) else '') + '\n'.join(sComponents)
        _port_connections =                                                         '\n'.join(sPortConnections)
        _equations        =                                                         '\n'.join(sEquations)
        _stns             =                                                         '\n'.join(sSTNs)
        dictModel = {
                      'model'            : model.__class__.__name__,
                      'ports'            : _ports,
                      'parameters'       : _parameters,
                      'variables'        : _variables,
                      'components'       : _components,
                      'port_connections' : _port_connections,
                      'equations'        : _equations,
                      'stns'             : _stns
                    }
        result = modelTemplate % dictModel
        return result

    def collectInitialValuesFromModel(self, model):
        for variable in model.Variables:
            canonicalName = nameFormat(daeGetRelativeName(self.topLevelModel, variable))
            if len(variable.Domains) > 0:
                isDiff     = False
                isAssigned = False
                isNormal   = False
                for vt in self.variableTypes[variable.OverallIndex : variable.OverallIndex+variable.NumberOfPoints]:
                    if vt == cnDifferential:
                        isDiff = True
                    elif vt == cnAssigned:
                        isAssigned = True
                    else:
                        isNormal = True
                value = self.initialVariableValues[variable.OverallIndex]
                # I should do something now

            else:
                value = variable.GetValue()
                if self.variableTypes[variable.OverallIndex] == cnDifferential:
                    self.initialConditions[canonicalName] = value
                elif self.variableTypes[variable.OverallIndex] == cnAssigned:
                    self.assignedVariablesValues[canonicalName] = value

        for port in model.Ports:
            self.collectInitialValuesFromPort(port)

        for component in model.Components:
            self.collectInitialValuesFromModel(component)

    def collectInitialValuesFromPort(self, port):
        for variable in port.Variables:
            canonicalName = nameFormat(daeGetRelativeName(self.topLevelModel, variable))
            if len(variable.Domains) > 0:
                raise RuntimeError('Modelica ports cannot contain distributed variables')
            else:
                value = variable.GetValue()
                if self.variableTypes[variable.OverallIndex] == cnDifferential:
                    self.initialConditions[canonicalName] = value
                elif self.variableTypes[variable.OverallIndex] == cnAssigned:
                    self.assignedVariablesValues[canonicalName] = value
        
    def processNode(self, node, context):
        res = ''
        if isinstance(node, adConstantNode):
            res = '{0}'.format(node.Quantity.value) #, node.Quantity.units)

        elif isinstance(node, adTimeNode):
            res = 'time'

        elif isinstance(node, adUnaryNode):
            n = '(' + self.processNode(node.Node, context) + ')'

            if node.Function == eSign:
                res = '-{0}'.format(n)
            elif node.Function == eSqrt:
                res = 'sqrt({0})'.format(n)
            elif node.Function == eExp:
                res = 'exp({0})'.format(n)
            elif node.Function == eLog:
                res = 'log10({0})'.format(n)
            elif node.Function == eLn:
                res = 'log({0})'.format(n)
            elif node.Function == eAbs:
                res = 'abs({0})'.format(n)
            elif node.Function == eSin:
                res = 'sin({0})'.format(n)
            elif node.Function == eCos:
                res = 'cos({0})'.format(n)
            elif node.Function == eTan:
                res = 'tan({0})'.format(n)
            elif node.Function == eArcSin:
                res = 'asin({0})'.format(n)
            elif node.Function == eArcCos:
                res = 'acos({0})'.format(n)
            elif node.Function == eArcTan:
                res = 'atan({0})'.format(n)
            elif node.Function == eCeil:
                res = 'ceil({0})'.format(n)
            elif node.Function == eFloor:
                res = 'floor({0})'.format(n)
            else:
                raise RuntimeError('Not supported unary function')

        elif isinstance(node, adBinaryNode):
            left  = '(' + self.processNode(node.LNode, context) + ')'
            right = '(' + self.processNode(node.RNode, context) + ')' 

            if node.Function == ePlus:
                res = '{0} + {1}'.format(left, right)
            elif node.Function == eMinus:
                res = '{0} - {1}'.format(left, right)
            elif node.Function == eMulti:
                res = '{0} * {1}'.format(left, right)
            elif node.Function == eDivide:
                res = '{0} / {1}'.format(left, right)
            elif node.Function == ePower:
                res = '{0} ^ {1}'.format(left, right)
            elif node.Function == eMin:
                res = 'min({0}, {1})'.format(left, right)
            elif node.Function == eMax:
                res = 'max({0}, {1})'.format(left, right)
            else:
                raise RuntimeError('Not supported binary function')

        elif isinstance(node, adScalarExternalFunctionNode):
            raise RuntimeError('External functions are not supported')

        elif isinstance(node, adVectorExternalFunctionNode):
            raise RuntimeError('External functions are not supported')

        elif isinstance(node, adDomainIndexNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            name = nameFormat(daeGetRelativeName(context.model, node.Domain))
            res = '{0}[{1}]'.format(name, node.Index+1)

        elif isinstance(node, adRuntimeParameterNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            domainindexes = ''
            if len(node.DomainIndexes) > 0:
                domainindexes = '[{0}]'.format(','.join([str(di + 1) for di in node.DomainIndexes]))

            name = nameFormat(daeGetRelativeName(context.model, node.Parameter))
            res = '{0}{1}'.format(name, domainindexes)

        elif isinstance(node, adRuntimeVariableNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            domainindexes = ''
            if len(node.DomainIndexes) > 0:
                domainindexes = '[{0}]'.format(','.join([str(di + 1) for di in node.DomainIndexes]))

            name = nameFormat(daeGetRelativeName(context.model, node.Variable))
            res = '{0}{1}'.format(name, domainindexes)

        elif isinstance(node, adRuntimeTimeDerivativeNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            domainindexes = ''
            if len(node.DomainIndexes) > 0:
                domainindexes = '[{0}]'.format(','.join([str(di + 1) for di in node.DomainIndexes]))

            name = nameFormat(daeGetRelativeName(context.model, node.Variable))
            res = 'der({0}{1})'.format(name, domainindexes)

        else:
            raise RuntimeError('Not supported node')

        return res
