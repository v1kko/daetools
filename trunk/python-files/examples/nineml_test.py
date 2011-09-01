#!/usr/bin/env python

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation
import os, sys
from time import localtime, strftime, time
from daetools.pyDAE import *
from nineml_daetools_bridge import *

class sim_hierachical_iaf_1coba(daeSimulation):
    def __init__(self, ninemlComponent, parameters, initial_conditions, active_regimes, results_variables):
        daeSimulation.__init__(self)
        self.m = nineml_daetools_bridge(ninemlComponent.name, ninemlComponent)
        self.m.Description = ""

        self._parameters         = parameters
        self._initial_conditions = initial_conditions
        self._active_regimes     = active_regimes
        self._results_variables  = results_variables

    def SetUpParametersAndDomains(self):
        for paramName in self._parameters:
            parameter = getObjectFromCanonicalName(self.m, paramName, look_for_parameters = True)
            print 'parameter = ' + repr(parameter)
            
        iaf  = findObjectInModel(self.m, 'iaf',       look_for_models = True)
        coba = findObjectInModel(self.m, 'cobaExcit', look_for_models = True)

        tau       = findObjectInModel(coba, 'tau',       look_for_parameters = True)
        vrev      = findObjectInModel(coba, 'vrev',      look_for_parameters = True)
        q         = findObjectInModel(coba, 'q',         look_for_parameters = True)
        cm        = findObjectInModel(iaf,  'cm',        look_for_parameters = True)
        gl        = findObjectInModel(iaf,  'gl',        look_for_parameters = True)
        taurefrac = findObjectInModel(iaf,  'taurefrac', look_for_parameters = True)
        vreset    = findObjectInModel(iaf,  'vreset',    look_for_parameters = True)
        vrest     = findObjectInModel(iaf,  'vrest',     look_for_parameters = True)
        vthresh   = findObjectInModel(iaf,  'vthresh',   look_for_parameters = True)

        tau       .SetValue(5.0)
        vrev      .SetValue(0)
        q         .SetValue(3) # was 1
        cm        .SetValue(1)
        gl        .SetValue(50)
        taurefrac .SetValue(0.008) # was 8
        vreset    .SetValue(-60)
        vrest     .SetValue(-60)
        vthresh   .SetValue(-40)
        
    def SetUpVariables(self):
        for varName in self._initial_conditions:
            variable = getObjectFromCanonicalName(self.m, varName, look_for_variables = True)
            print 'variable = ' + repr(variable)

        iaf  = findObjectInModel(self.m, 'iaf',       look_for_models = True)
        coba = findObjectInModel(self.m, 'cobaExcit', look_for_models = True)

        g      = findObjectInModel(coba, 'g',      look_for_variables = True)
        V      = findObjectInModel(iaf,  'V',      look_for_variables = True)
        tspike = findObjectInModel(iaf,  'tspike', look_for_variables = True)

        g     .SetInitialCondition(0)
        V     .SetInitialCondition(-60) # = parameter [iaf_vrest]
        tspike.SetInitialCondition(-1e99)
        
    def Run(self):
        spikeinput = getObjectFromCanonicalName(self.m, 'iaf_1coba.cobaExcit.spikeinput', look_for_eventports = True)

        # Mickey Mouse simulation to test the functionality of the NineML-DAE Tools bridge.
        # We have only one neuron (IAF) and a synapse (COBA) connected to it.
        # Here we trigger an input event on the synapse each 'dt' seconds.
        # At the beginning the Isyn will not be high enough to depolarize the membrane to produce the action potential.
        # However after the repeated events on the synapse it will start emit action potential events.
        # Note: here ReceiveEvent is used just for simulating the input events; it is not used in real applications.
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

# Write the component back out to XML
#nineml.al.writers.XMLWriter.write(coba1, 'TestOut_Coba1.xml')
#nineml.al.writers.DotWriter.write(coba1, 'TestOut_Coba1.dot')
#nineml.al.writers.DotWriter.build('TestOut_Coba1.dot')

#print 'Component hierachical_iaf_1coba:'
#printComponent(coba1, 'hierachical_iaf_1coba')

# Create Log, Solver, DataReporter and Simulation object
log          = daePythonStdOutLog()
daesolver    = daeIDAS()

start_time = time()
simulation   = sim_hierachical_iaf_1coba(coba1, parameters, initial_conditions, active_regimes, results_variables)
elapsed_time = time() - start_time
print 'Time to create component =', elapsed_time

datareporter = daeTCPIPDataReporter()

# Enable reporting of all variables
simulation.m.SetReportingOn(True)

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
