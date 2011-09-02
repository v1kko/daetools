#!/usr/bin/env python

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation
import os, sys
from time import localtime, strftime, time
from daetools.pyDAE import *
from nineml_daetools_bridge import *

class daeSimulationInputData:
    def __init__(self):
        self.parser = ExpressionParser()

        # Dictionaries 'canonical/relative name' : floating-point-value
        self.parameters        = {}
        self.initialConditions = {}

        # Dictionaries: 'canonical/relative name' : 'expression'
        self.inletPortExpressions      = {}
        self.inletEventPortExpressions = {}

        # List of canonical/relative names
        self.activeStates       = []
        self.variablesToReport  = []

        #self.timeHorizon       = 0.0
        #self.reportingInterval = 0.0

        #self.daeSolver    = daeIDAS()
        #self.laSolver     = pySuperLU.daeCreateSuperLUSolver()
        #self.log          = daePythonStdOutLog()
        #self.dataReporter = daeTCPIPDataReporter()

class nineml_daetools_simulation(daeSimulation):
    def __init__(self, model, **kwargs):
        daeSimulation.__init__(self)
        self.m = model

        dictIdentifiers, dictFunctions = getNineMLDictionaries(self.m)
        self.parser = ExpressionParser(dictIdentifiers, dictFunctions)

        self._parameters               = kwargs.get('parameters',               {})
        self._initial_conditions       = kwargs.get('initial_conditions',       {})
        self._active_states            = kwargs.get('active_states',            [])
        self._variables_to_report      = kwargs.get('variables_to_report',      [])
        self._analog_ports_expressions = kwargs.get('analog_ports_expressions', {})
        self._event_ports_expressions  = kwargs.get('event_ports_expressions',  {})

    def SetUpParametersAndDomains(self):
        for paramName, value in self._parameters.items():
            parameter = getObjectFromCanonicalName(self.m, paramName, look_for_parameters = True)
            if parameter == None:
                raise RuntimeError('Could not locate parameter {0}'.format(paramName))
            print '  --> Set the parameter: {0} to: {1}'.format(paramName, value)
            parameter.SetValue(value)
            
    def SetUpVariables(self):
        for varName, value in self._initial_conditions.items():
            variable = getObjectFromCanonicalName(self.m, varName, look_for_variables = True)
            if variable == None:
                raise RuntimeError('Could not locate variable {0}'.format(paramName))
            print '  --> Set the variable: {0} to: {1}'.format(varName, value)
            variable.SetInitialCondition(value)

        for portName, expression in self._analog_ports_expressions.items():
            if expression == None or expression == '':
                raise RuntimeError('An expression for the value of the analog port {0} cannot be empty'.format(portName))
            port = getObjectFromCanonicalName(self.m, portName, look_for_ports = True, look_for_reduceports = True)
            if port == None:
                raise RuntimeError('Could not locate port {0}'.format(portName))
            print 'port: {0} = {1}'.format(portName, self.parser.parse(expression))

        for activeStateName in self._active_states:
            listNames = activeStateName.split('.')
            if len(listNames) > 1:
                stateName = listNames[-1]
                modelName = '.'.join(listNames[:-1])
            else:
                raise RuntimeError('Invalid initial active state name {0}'.format(activeStateName))

            stn = getObjectFromCanonicalName(self.m, modelName + '.' + nineml_daetools_bridge.ninemlSTNRegimesName, look_for_stns = True)
            if stn == None:
                raise RuntimeError('Could not locate STN {0}'.format(nineml_daetools_bridge.ninemlSTNRegimesName))

            print '  --> Set the active state in the model: {0} to: {1}'.format(modelName, stateName)
            stn.ActiveState = stateName

        self.m.SetReportingOn(False)
        for varName in self._variables_to_report:
            variable = getObjectFromCanonicalName(self.m, varName, look_for_variables = True)
            if variable == None:
                raise RuntimeError('Could not locate variable {0}'.format(paramName))
            print '  --> Report the variable: {0}'.format(varName)
            variable.ReportingOn = True
        
    def Run1(self):
        spikeinput = getObjectFromCanonicalName(self.m, 'iaf_1coba.cobaExcit.spikeinput', look_for_eventports = True)

        dt = 0.1
        while self.CurrentTime < self.TimeHorizon:
            spikeinput.ReceiveEvent(0.0)
            self.Reinitialize()
            self.ReportData(self.CurrentTime)

            targetTime = self.CurrentTime + dt
            if targetTime > self.TimeHorizon:
                targetTime = self.TimeHorizon

            while self.CurrentTime < targetTime:
                t = self.NextReportingTime
                self.Log.Message('Integrating from {0} to {1} ...'.format(self.CurrentTime, t), 0)
                self.IntegrateUntilTime(t, eDoNotStopAtDiscontinuity)
                self.ReportData(self.CurrentTime)
          
if __name__ == "__main__":
    parameters = {
        'cobaExcit.q' : 3.0,
        'cobaExcit.tau' : 5.0,
        'cobaExcit.vrev' : 0.0,
        'iaf.cm' : 1,
        'iaf.gl' : 50,
        'iaf.taurefrac' : 0.008,
        'iaf.vreset' : -60,
        'iaf.vrest' : -60,
        'iaf.vthresh' : -40
    }
    initial_conditions = {
        'cobaExcit.g' : 0.0,
        'iaf.V' : -60,
        'iaf.tspike' : -1E99
    }
    active_states = [
        'cobaExcit.cobadefaultregime',
        'iaf.subthresholdregime'
    ]
    analog_ports_expressions = {
        'cobaExcit.V' : '1.2 * e',
        'iaf.ISyn' : '5 * pi'
    }
    event_ports_expressions = {
    }
    variables_to_report = [
        'cobaExcit.g',
        'iaf.tspike'
    ]

    # Load the Component:
    nineml_comp  = TestableComponent('hierachical_iaf_1coba')()
    if not nineml_comp:
        raise RuntimeError('Cannot load NineML component')

    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()

    model = nineml_daetools_bridge(nineml_comp.name, nineml_comp, None, '')
    simulation = nineml_daetools_simulation(model, parameters               = parameters,
                                                   initial_conditions       = initial_conditions,
                                                   active_states            = active_states,
                                                   analog_ports_expressions = analog_ports_expressions,
                                                   event_ports_expressions  = event_ports_expressions,
                                                   variables_to_report      = variables_to_report)
    datareporter = daeTCPIPDataReporter()

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.001
    simulation.TimeHorizon = 1

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
