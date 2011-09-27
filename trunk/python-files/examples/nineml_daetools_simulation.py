#!/usr/bin/env python

from __future__ import print_function
import json, numpy
import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
import os, sys, math
from time import localtime, strftime, time
from daetools.pyDAE import *
from nineml_daetools_bridge import *
from nineml_tex_report import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class daeSimulationInputData:
    def __init__(self):
        self.parser = ExpressionParser()

        # Dictionaries 'canonical/relative name' : floating-point-value
        self.parameters         = {}
        self.initial_conditions = {}
        # Dictionaries: 'canonical/relative name' : 'expression'
        self.analog_port_expressions = {}
        self.event_port_expressions  = {}
        # Dictionary: 'canonical/relative name' : string
        self.active_regimes      = {}
        # Dictionary: 'canonical/relative name' : boolean
        self.variables_to_report = {}

        self.timeHorizon       = 0.0
        self.reportingInterval = 0.0

        #self.daeSolver    = daeIDAS()
        #self.laSolver     = pySuperLU.daeCreateSuperLUSolver()
        #self.log          = daePythonStdOutLog()
        #self.dataReporter = daeTCPIPDataReporter()

    def dumpJSON(self, sort = True, indent = 2):
        data = {}
        data['timeHorizon']               = self.timeHorizon
        data['reportingInterval']         = self.reportingInterval
        data['parameters']                = self.parameters
        data['initial_conditions']        = self.initial_conditions
        data['analog_port_expressions']   = self.analog_port_expressions
        data['event_port_expressions']    = self.event_port_expressions
        data['active_regimes']            = self.active_regimes
        data['variables_to_report']       = self.variables_to_report

        return json.dumps(data, sort_keys = sort, indent = indent)

    def loadJSON(self, jsonContent):
        data = json.loads(jsonContent)

        if 'timeHorizon' in data:
            self.timeHorizon = float(data['timeHorizon'])

        if 'reportingInterval' in data:
            self.reportingInterval = float(data['reportingInterval'])

        if 'parameters' in data:
            temp = data['parameters']
            if isinstance(temp, dict):
                self.parameters = temp

        if 'initial_conditions' in data:
            temp = data['initial_conditions']
            if isinstance(temp, dict):
                self.initial_conditions = temp

        if 'analog_ports_expressions' in data:
            temp = data['analog_ports_expressions']
            if isinstance(temp, dict):
                self.analog_ports_expressions = temp

        if 'event_ports_expressions' in data:
            temp = data['event_ports_expressions']
            if isinstance(temp, dict):
                self.event_ports_expressions = temp

        if 'active_regimes' in data:
            temp = data['active_regimes']
            if isinstance(temp, list):
                self.active_regimes = temp

        if 'variables_to_report' in data:
            temp = data['variables_to_report']
            if isinstance(temp, list):
                self.variables_to_report = temp

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        data = {}
        data['timeHorizon']               = self.timeHorizon
        data['reportingInterval']         = self.reportingInterval
        data['parameters']                = self.parameters
        data['initial_conditions']        = self.initial_conditions
        data['analog_port_expressions']   = self.analog_port_expressions
        data['event_port_expressions']    = self.event_port_expressions
        data['active_regimes']            = self.active_regimes
        data['variables_to_report']       = self.variables_to_report
        return str(data)

class ninemlTesterDataReporter(daeDataReporterLocal):
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def createReportData(self, tmp_folder = '/tmp'):
        fp8  = mpl.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='normal', size=8)
        fp9  = mpl.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='normal', size=9)
        fp11 = mpl.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='normal', size=11)

        font = {'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 8}
        mpl.rc('font', **font)
        params = {'axes.labelsize':  9,
                  'text.fontsize':   8,
                  'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'text.usetex': True}
        mpl.rcParams.update(params)

        plots = []
        for i, var in enumerate(self.Process.Variables):
            fileName   = var.Name
            fileName   = fileName.replace('.', '')
            fileName   = fileName.replace('_', '')
            pngName    = fileName + '.png'
            csvName    = fileName + '.csv'
            pngPath    = tmp_folder + '/' + pngName
            csvPath    = tmp_folder + '/' + csvName
            title      = var.Name.split('.')[-1] + ' = f(t)'
            xAxisLabel = 't'
            yAxisLabel = var.Name
            xPoints    = var.TimeValues
            yPoints    = var.Values.reshape(len(var.Values))

            fig = plt.figure(figsize=(4, 3), dpi=(300))
            ax = fig.add_subplot(111)
            ax.plot(xPoints, yPoints)
            ax.set_title(title)
            ax.set_xlabel(xAxisLabel)
            #ax.set_ylabel(yAxisLabel)
            fig.savefig(pngPath, dpi=(300))
            
            if self.exportCSV(xPoints, yPoints, xAxisLabel, yAxisLabel, csvPath):
                plots.append((var.Name, xPoints, yPoints, pngName, csvName))
            else:
                plots.append((var.Name, xPoints, yPoints, None, csvName))

        return plots

    def exportCSV(self, x, y, xname, yname, filename):
        try:
            n = len(x)
            f = open(filename, "w")
            f.write('{0},{1}\n'.format(xname, yname))
            for i in range(0, n):
                f.write('%.18e,%.18e\n' % (x[i], y[i]))
            f.close()
            return True
        except Exception as e:
            return False

    def Connect(self, ConnectionString, ProcessName):
        return True

    def Disconnect(self):
        return True

    def IsConnected(self):
        return True

class nineml_daetools_simulation(daeSimulation):
    def __init__(self, model, **kwargs):
        daeSimulation.__init__(self)
        self.m = model

        dictIdentifiers, dictFunctions      = getAnalogPortsDictionaries(self.m)
        self.analog_ports_expression_parser = ExpressionParser(dictIdentifiers, dictFunctions)

        # These dictionaries may contain unicode strings (if the input originated from the web form)
        # Therefore, str(...) should be used whenever a string is expected
        self._parameters               = kwargs.get('parameters',               {})
        self._initial_conditions       = kwargs.get('initial_conditions',       {})
        self._active_regimes           = kwargs.get('active_regimes',           {})
        self._variables_to_report      = kwargs.get('variables_to_report',      {})
        self._analog_ports_expressions = kwargs.get('analog_ports_expressions', {})
        self._event_ports_expressions  = kwargs.get('event_ports_expressions',  {})

        self.TimeHorizon               = float(kwargs.get('timeHorizon', 0.0))
        self.ReportingInterval         = float(kwargs.get('reportingInterval', 0.0))
        
        self.intervals = {}
        
        # Initialize reduce ports
        for portName, expression in list(self._analog_ports_expressions.items()):
            portName = str(portName)
            port = getObjectFromCanonicalName(self.m, portName, look_for_ports = True, look_for_reduceports = True)
            if port == None:
                raise RuntimeError('Could not locate port {0}'.format(portName))
            if isinstance(port, ninemlReduceAnalogPort):
                if len(port.Ports) != 0:
                    raise RuntimeError('The reduce port {0} is connected and cannot be set a value'.format(portName))
                a_port = port.addPort()
            elif isinstance(port, ninemlAnalogPort):
                pass

    def SetUpParametersAndDomains(self):
        for paramName, value in list(self._parameters.items()):
            paramName = str(paramName)
            parameter = getObjectFromCanonicalName(self.m, paramName, look_for_parameters = True)
            if parameter == None:
                raise RuntimeError('Could not locate parameter {0}'.format(paramName))
            self.Log.Message('  --> Set the parameter: {0} to: {1}'.format(paramName, value), 0)
            parameter.SetValue(value)
            
    def SetUpVariables(self):
        for varName, value in list(self._initial_conditions.items()):
            varName = str(varName)
            variable = getObjectFromCanonicalName(self.m, varName, look_for_variables = True)
            if variable == None:
                raise RuntimeError('Could not locate variable {0}'.format(varName))
            self.Log.Message('  --> Set the variable: {0} to: {1}'.format(varName, value), 0)
            variable.SetInitialCondition(value)

        for portName, expression in list(self._analog_ports_expressions.items()):
            portName = str(portName)
            if expression == None or expression == '':
                raise RuntimeError('The analog port {0} is not connected and no value has been provided'.format(portName))
            port = getObjectFromCanonicalName(self.m, portName, look_for_ports = True, look_for_reduceports = True)
            if port == None:
                raise RuntimeError('Could not locate port {0}'.format(portName))
            
            value = float(self.analog_ports_expression_parser.parse_and_evaluate(expression))
            if isinstance(port, ninemlAnalogPort):
                port.value.AssignValue(value)
            elif isinstance(port, ninemlReduceAnalogPort):
                for a_port in port.Ports:
                    a_port.value.AssignValue(value)
            else:
                raise RuntimeError('Unknown port object: {0}'.format(portName))
            self.Log.Message('  --> Assign the value of the port variable: {0} to {1} (evaluated value: {2})'.format(portName, expression, value), 0)
        
        for portName, expression in list(self._event_ports_expressions.items()):
            portName = str(portName)
            if expression == None or expression == '':
                continue
            port = getObjectFromCanonicalName(self.m, portName, look_for_eventports = True)
            if port == None:
                raise RuntimeError('Could not locate event port {0}'.format(portName))
            
            str_values = expression.split(',')
            for item in str_values:
                try:
                    value = float(item)
                except ValueError:
                    raise RuntimeError('Cannot convert: {0} to floating point value in the event port expression: {1}'.format(item, expression))
                # At this point self.intervals contain only event emit time points
                if value in self.intervals:
                    data = self.intervals[value]
                else:
                    data = []
                data.append(port)
                self.intervals[value] = data
            self.Log.Message('  --> Event port {0} triggers at: {1}'.format(portName, expression), 0)

        for modelName, stateName in list(self._active_regimes.items()):
            modelName = str(modelName)
            stateName = str(stateName)
            stn = getObjectFromCanonicalName(self.m, modelName + '.' + nineml_daetools_bridge.ninemlSTNRegimesName, look_for_stns = True)
            if stn == None:
                raise RuntimeError('Could not locate STN {0}'.format(nineml_daetools_bridge.ninemlSTNRegimesName))

            self.Log.Message('  --> Set the active state in the model: {0} to: {1}'.format(modelName, stateName), 0)
            stn.ActiveState = stateName

        self.m.SetReportingOn(False)
        for varName, value in list(self._variables_to_report.items()):
            varName = str(varName)
            if value:
                variable = getObjectFromCanonicalName(self.m, varName, look_for_variables = True)
                if variable == None:
                    raise RuntimeError('Could not locate variable {0}'.format(varName))
                self.Log.Message('  --> Report the variable: {0}'.format(varName), 0)
                variable.ReportingOn = True
        
    def Run(self):
        # Add the normal reporting times
        for t in self.ReportingTimes:
            if not t in self.intervals:
                self.intervals[t] = None
        #for t, ports in sorted(self.intervals.items()):
        #    print('%.18e: %s' % (t, str(ports)))
        
        for t, event_ports in sorted(self.intervals.items()):
            # IDA complains when time horizon is too close to the current time 
            if math.fabs(t - self.CurrentTime) < 1E-5:
                self.Log.Message('WARNING: skipping the time point %.18e: too close to the previous time point' % t, 0)
                continue
            
            # Integrate until 't'
            self.Log.Message('Integrating from %.7f to %.7f ...' % (self.CurrentTime, t), 0)
            self.IntegrateUntilTime(t, eDoNotStopAtDiscontinuity)
            
            # Trigger the events (if any) and reinitialize
            if event_ports:
                for event_port in event_ports:
                    event_port.ReceiveEvent(0.0)
                self.Reinitialize()
            
            # Report the data
            self.ReportData(self.CurrentTime)
                    
if __name__ == "__main__":
    parameters = {
        'cobaExcit.q' : 3.0,
        'cobaExcit.tau' : 5.0,
        'cobaExcit.vrev' : 0.0,
        'iaf.cm' : 1,
        'iaf.gl' : 50,
        'iaf.taurefrac' : 0.008,
        'iaf.vreset' : -0.060,
        'iaf.vrest' : -0.060,
        'iaf.vthresh' : -0.040
    }
    initial_conditions = {
        'cobaExcit.g' : 0.0,
        'iaf.V' : -0.060,
        'iaf.tspike' : -1E99
    }
    analog_ports_expressions = {}
    event_ports_expressions = {}
    active_regimes = {
        'cobaExcit' : 'cobadefaultregime',
        'iaf' : 'subthresholdregime'
    }
    variables_to_report = {
        'cobaExcit.I' : True,
        'iaf.V' : True
    }
    timeHorizon       = 1
    reportingInterval = 0.001

    # Load the Component:
    nineml_comp  = TestableComponent('hierachical_iaf_1coba')()
    if not nineml_comp:
        raise RuntimeError('Cannot load NineML component')

    # Create Log, Solver, DataReporter and Simulation object
    log          = daeBaseLog()
    daesolver    = daeIDAS()

    model = nineml_daetools_bridge(nineml_comp.name, nineml_comp, None, '')
    simulation = nineml_daetools_simulation(model, timeHorizon              = timeHorizon,
                                                   reportingInterval        = reportingInterval,
                                                   parameters               = parameters,
                                                   initial_conditions       = initial_conditions,
                                                   active_regimes           = active_regimes,
                                                   analog_ports_expressions = analog_ports_expressions,
                                                   event_ports_expressions  = event_ports_expressions,
                                                   variables_to_report      = variables_to_report)
    datareporter = ninemlTesterDataReporter()

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = reportingInterval
    simulation.TimeHorizon       = timeHorizon

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model reports for all models
    #simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    #iaf  = findObjectInModel(simulation.m, 'iaf', look_for_models = True)
    #iaf.SaveModelReport(iaf.Name + ".xml")
    #coba = findObjectInModel(simulation.m, 'cobaExcit', look_for_models = True)
    #coba.SaveModelReport(coba.Name + ".xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

    inspector = nineml_component_inspector()
    inspector.inspect(nineml_comp)

    log_output = log.JoinMessages('\n')
    plots = datareporter.createReportData('.')

    dictInputs = {}
    dictInputs['parameters']                = parameters
    dictInputs['initial_conditions']        = initial_conditions
    dictInputs['analog_ports_expressions']  = analog_ports_expressions
    dictInputs['event_ports_expressions']   = event_ports_expressions
    dictInputs['active_regimes']            = active_regimes
    dictInputs['variables_to_report']       = variables_to_report
    dictInputs['timeHorizon']               = timeHorizon
    dictInputs['reportingInterval']         = reportingInterval
    
    tests_data = []
    tests_data.append( ('Dummy test', 'Dummy test notes', dictInputs, plots, log_output) )

    createLatexReport(inspector, tests_data, 'nineml-tex-template.tex', 'coba_iaf.tex')

    res = createPDF('coba_iaf.tex')
    subprocess.call(['evince', 'coba_iaf.pdf'], shell=False)
