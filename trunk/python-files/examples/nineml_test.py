#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation
import os, sys
from time import localtime, strftime
from daetools.pyDAE.daeParser import daeExpressionParser
from daetools.pyDAE.daeGetParserDictionary import getParserDictionary
from daetools.pyDAE import *

from nineml_daetools_bridge import *

class sim_hierachical_iaf_1coba(daeSimulation):
    def __init__(self, ninemlComponent):
        daeSimulation.__init__(self)
        self.m = nineml_daetools_bridge(ninemlComponent.name, ninemlComponent)
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
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
        q         .SetValue(1)
        cm        .SetValue(1)
        gl        .SetValue(1) # was 50
        taurefrac .SetValue(2)
        vreset    .SetValue(-60)
        vrest     .SetValue(-60)
        vthresh   .SetValue(-40)
        
    def SetUpVariables(self):
        iaf  = findObjectInModel(self.m, 'iaf',       look_for_models = True)
        coba = findObjectInModel(self.m, 'cobaExcit', look_for_models = True)

        g      = findObjectInModel(coba, 'g',      look_for_variables = True)
        V      = findObjectInModel(iaf,  'V',      look_for_variables = True)
        tspike = findObjectInModel(iaf,  'tspike', look_for_variables = True)

        g     .SetInitialCondition(0)
        V     .SetInitialCondition(-60) # = parameter [iaf_vrest]
        tspike.SetInitialCondition(-1e99)
        
    def Run(self):
        spikeoutput = getObjectFromCanonicalName(self.m, 'iaf_1coba.iaf.spikeoutput', look_for_eventports = True)

        self.Log.Message("Integrating for 1 second ... ", 0)
        time = self.IntegrateForTimeInterval(1)
        self.ReportData(self.CurrentTime)

        spikeoutput.SendEvent(0.0)
        self.Reinitialize()

        daeSimulation.Run(self)
        return
        while self.CurrentTime < self.TimeHorizon:
            t = self.NextReportingTime
            self.Log.Message('Integrating from {0} to {1} ...'.format(self.CurrentTime, t), 0)
            self.IntegrateUntilTime(t, eStopAtModelDiscontinuity)
            self.ReportData(self.CurrentTime)
            

# Load the Component:
coba1_base = TestableComponent('hierachical_iaf_1coba')
coba1 = coba1_base()

# Write the component back out to XML
#nineml.al.writers.XMLWriter.write(coba1, 'TestOut_Coba1.xml')
#nineml.al.writers.DotWriter.write(coba1, 'TestOut_Coba1.dot')
#nineml.al.writers.DotWriter.build('TestOut_Coba1.dot')

print 'Component hierachical_iaf_1coba:'
printComponent(coba1, 'hierachical_iaf_1coba')

# Create Log, Solver, DataReporter and Simulation object
log          = daePythonStdOutLog()
daesolver    = daeIDAS()
simulation   = sim_hierachical_iaf_1coba(coba1)
datareporter = daeTCPIPDataReporter()

# Enable reporting of all variables
simulation.m.SetReportingOn(True)

# Set the time horizon and the reporting interval
simulation.ReportingInterval = 250
simulation.TimeHorizon = 250

# Connect data reporter
simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
if(datareporter.Connect("", simName) == False):
    sys.exit()

# Initialize the simulation
simulation.Initialize(daesolver, datareporter, log)

# Save the model report and the runtime model report
simulation.m.SaveModelReport(simulation.m.Name + ".xml")
iaf  = findObjectInModel(simulation.m, 'iaf',       look_for_models = True)
coba = findObjectInModel(simulation.m, 'cobaExcit', look_for_models = True)
iaf.SaveModelReport(iaf.Name + ".xml")
coba.SaveModelReport(coba.Name + ".xml")

simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

# Solve at time=0 (initialization)
simulation.SolveInitial()

# Run
simulation.Run()
simulation.Finalize()




"""
# Simulate the Neuron:
records = [
    RecordValue(what='iaf_V', tag='V', label='V'),
    RecordValue(what='regime', tag='Regime', label='Regime'),
        ]

parameters = nineml.al.flattening.ComponentFlattener.flatten_namespace_dict({
'cobaExcit_tau':5.0,
'cobaExcit_vrev':0,
'iaf_cm':1,
'iaf_gl':50,
'iaf_taurefrac':8,
'iaf_vreset':-60,
'iaf_vrest':-60,
'iaf_vthresh':-40
 })

initial_values = {
        'iaf_V': parameters['iaf_vrest'],
        'tspike': -1e99,
        'regime': 1002,
            }

res = std_pynn_simulation( test_component = coba1,
                    parameters = parameters,
                    initial_values = initial_values,
                    synapse_components = [('cobaExcit','q')],
                    synapse_weights=15.0,
                    records = records,
                    sim_time=250,
                    syn_input_rate=100
                   )
"""