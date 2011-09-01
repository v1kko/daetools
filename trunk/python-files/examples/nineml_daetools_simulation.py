#!/usr/bin/env python

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation
import os, sys
from time import localtime, strftime, time
from daetools.pyDAE import *
from nineml_daetools_bridge import *

class nineml_daetools_simulation(daeSimulation):
    def __init__(self, ninemlComponent, parameters, initial_conditions, active_regimes, results_variables):
        daeSimulation.__init__(self)
        self.m = nineml_daetools_bridge(ninemlComponent.name, ninemlComponent, None, '')

        self._parameters         = parameters
        self._initial_conditions = initial_conditions
        self._active_regimes     = active_regimes
        self._results_variables  = results_variables

    def SetUpParametersAndDomains(self):
        for paramName, value in self._parameters.items():
            parameter = getObjectFromCanonicalName(self.m, paramName, look_for_parameters = True)
            if parameter == None:
                raise RuntimeError('Could not locate parameter {0}'.format(paramName))
            print '  Set the parameter: {0} to: {1}'.format(paramName, value)
            parameter.SetValue(value)
            
    def SetUpVariables(self):
        for varName, value in self._initial_conditions.items():
            variable = getObjectFromCanonicalName(self.m, varName, look_for_variables = True)
            if variable == None:
                raise RuntimeError('Could not locate variable {0}'.format(paramName))
            print '  Set the variable: {0} to: {1}'.format(varName, value)
            variable.SetInitialCondition(value)

        for activeStateName in self._active_regimes:
            listNames = activeStateName.split('.')
            if len(listNames) > 1:
                stateName = listNames[-1]
                modelName = '.'.join(listNames[:-1])
            else:
                raise RuntimeError('Invalid initial active state name {0}'.format(activeStateName))

            stn = getObjectFromCanonicalName(self.m, modelName + '.' + nineml_daetools_bridge.ninemlSTNRegimesName, look_for_stns = True)
            if stn == None:
                raise RuntimeError('Could not locate STN {0}'.format(nineml_daetools_bridge.ninemlSTNRegimesName))
            
            print '  Set the active state in the model: {0} to: {1}'.format(modelName, stateName)
            stn.ActiveState = stateName

        self.m.SetReportingOn(False)
        for varName in self._results_variables:
            variable = getObjectFromCanonicalName(self.m, varName, look_for_variables = True)
            if variable == None:
                raise RuntimeError('Could not locate variable {0}'.format(paramName))
            print '  Report the variable: {0}'.format(varName)
            variable.ReportingOn = True
        
    def Run(self):
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
active_regimes = [
    'cobaExcit.cobadefaultregime',
    'iaf.subthresholdregime'
]
results_variables = [
    'cobaExcit.g',
    'iaf.tspike'
]

# Load the Component:
coba1_base = TestableComponent('hierachical_iaf_1coba')
coba1 = coba1_base()

#print 'Component hierachical_iaf_1coba:'
#printComponent(coba1, 'hierachical_iaf_1coba')

# Create Log, Solver, DataReporter and Simulation object
log          = daePythonStdOutLog()
daesolver    = daeIDAS()

start_time = time()
simulation   = nineml_daetools_simulation(coba1, parameters, initial_conditions, active_regimes, results_variables)
elapsed_time = time() - start_time
print 'Time to create component =', elapsed_time

datareporter = daeTCPIPDataReporter()

# Set the time horizon and the reporting interval
simulation.ReportingInterval = 0.001
simulation.TimeHorizon = 1

# Connect data reporter
simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
if(datareporter.Connect("", simName) == False):
    sys.exit()

# Initialize the simulation
start_time = time()
simulation.Initialize(daesolver, datareporter, log)
elapsed_time = time() - start_time
print 'Time to initialize the simulation =', elapsed_time

# Save the model reports for all models
simulation.m.SaveModelReport(simulation.m.Name + ".xml")
iaf  = findObjectInModel(simulation.m, 'iaf', look_for_models = True)
iaf.SaveModelReport(iaf.Name + ".xml")
coba = findObjectInModel(simulation.m, 'cobaExcit', look_for_models = True)
coba.SaveModelReport(coba.Name + ".xml")

# Solve at time=0 (initialization)
simulation.SolveInitial()

# Run
simulation.Run()
simulation.Finalize()
