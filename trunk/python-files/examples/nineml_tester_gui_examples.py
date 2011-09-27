#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os, sys, math
from time import localtime, strftime, time
import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from daetools.pyDAE import *
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from nineml_component_inspector import nineml_component_inspector
from nineml_daetools_bridge import *
from nineml_daetools_simulation import *
from daetools.pyDAE.WebViewDialog import WebView

def test_Izhikevich():
    nineml_component = TestableComponent('izhikevich')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    timeHorizon = 1
    reportingInterval = 0.001
    parameters = {
                    "a": 0.02,
                    "b": 0.2,
                    "c": -50.0,
                    "d": 2.0,
                    "theta": 0.03
                 }
    initial_conditions = {
                            "U": 0.0,
                            "V": -0.07
                         }
    analog_ports_expressions = {
                                  "Isyn": "0.1"
                               }
    event_ports_expressions = {}
    active_regimes = {}
    variables_to_report = {}

    inspector = nineml_component_inspector()
    inspector.inspect(nineml_component, timeHorizon              = timeHorizon,
                                        reportingInterval        = reportingInterval,
                                        parameters               = parameters,
                                        initial_conditions       = initial_conditions,
                                        active_regimes           = active_regimes,
                                        analog_ports_expressions = analog_ports_expressions,
                                        event_ports_expressions  = event_ports_expressions,
                                        variables_to_report      = variables_to_report)
    results = inspector.showQtGUI()
    return results, inspector

def test_Hodgkin_Huxley():
    nineml_component = TestableComponent('hh')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    timeHorizon = 1
    reportingInterval = 0.001
    parameters = {
        'C' : 1,
        'ek' : -63,
        'el' : 0.1,
        'ena' : -190,
        'gkbar' : 36,
        'gnabar' : 120,
        'theta' : -60,
        'gl' : 0.3,
        'celsius' : 1
    }
    initial_conditions = {
        'n' : 0.31768,
        'm' : 0.052932,
        'h' : 0.59612,
        'V' : -75
    }
    analog_ports_expressions = {
        'Isyn' : '0.01'
    }
    event_ports_expressions = {}
    active_regimes = {}
    variables_to_report = {}

    inspector = nineml_component_inspector()
    inspector.inspect(nineml_component, timeHorizon              = timeHorizon,
                                        reportingInterval        = reportingInterval,
                                        parameters               = parameters,
                                        initial_conditions       = initial_conditions,
                                        active_regimes           = active_regimes,
                                        analog_ports_expressions = analog_ports_expressions,
                                        event_ports_expressions  = event_ports_expressions,
                                        variables_to_report      = variables_to_report)
    results = inspector.showQtGUI()
    return results, inspector

def test_hierachical_iaf_1coba():
    nineml_component = TestableComponent('hierachical_iaf_1coba')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    timeHorizon = 1
    reportingInterval = 0.001
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
    event_ports_expressions = {
               'cobaExcit.spikeinput' : '0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90'
    }
    active_regimes = {
        'cobaExcit' : 'cobadefaultregime',
        'iaf' : 'subthresholdregime'
    }
    variables_to_report = {
        'cobaExcit.I' : True,
        'iaf.V' : True
    }

    inspector = nineml_component_inspector()
    inspector.inspect(nineml_component, timeHorizon              = timeHorizon,
                                        reportingInterval        = reportingInterval,
                                        parameters               = parameters,
                                        initial_conditions       = initial_conditions,
                                        active_regimes           = active_regimes,
                                        analog_ports_expressions = analog_ports_expressions,
                                        event_ports_expressions  = event_ports_expressions,
                                        variables_to_report      = variables_to_report)
    results = inspector.showQtGUI()
    return results, inspector

def test_coba_synapse():
    nineml_component = TestableComponent('coba_synapse')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    timeHorizon = 1
    reportingInterval = 0.001
    parameters = {
        'q' : 3.0,
        'tau' : 5.0,
        'vrev' : 0.0
    }
    initial_conditions = {
        'g' : 0.0,
    }
    analog_ports_expressions = {
        'V' : -0.050
    }
    event_ports_expressions = {
        'spikeinput' : '0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90'
    }
    active_regimes = {
        'cobaExcit' : 'cobadefaultregime'
    }
    variables_to_report = {
        'I' : True
    }

    inspector = nineml_component_inspector()
    inspector.inspect(nineml_component, timeHorizon              = timeHorizon,
                                        reportingInterval        = reportingInterval,
                                        parameters               = parameters,
                                        initial_conditions       = initial_conditions,
                                        active_regimes           = active_regimes,
                                        analog_ports_expressions = analog_ports_expressions,
                                        event_ports_expressions  = event_ports_expressions,
                                        variables_to_report      = variables_to_report)
    results = inspector.showQtGUI()
    return results, inspector

def test_iaf():
    nineml_component = TestableComponent('iaf')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    timeHorizon = 1
    reportingInterval = 0.001
    parameters = {
        'cm' : 1,
        'gl' : 50,
        'taurefrac' : 0.008,
        'vreset' : -0.060,
        'vrest' : -0.060,
        'vthresh' : -0.040
    }
    initial_conditions = {
        'V' : -0.060,
        'tspike' : -1E99
    }
    analog_ports_expressions = {
        'I' : 1.2
    }
    event_ports_expressions = {}
    active_regimes = {
        'iaf' : 'subthresholdregime'
    }
    variables_to_report = {
        'V' : True
    }

    inspector = nineml_component_inspector()
    inspector.inspect(nineml_component, timeHorizon              = timeHorizon,
                                        reportingInterval        = reportingInterval,
                                        parameters               = parameters,
                                        initial_conditions       = initial_conditions,
                                        active_regimes           = active_regimes,
                                        analog_ports_expressions = analog_ports_expressions,
                                        event_ports_expressions  = event_ports_expressions,
                                        variables_to_report      = variables_to_report)
    results = inspector.showQtGUI()
    return results, inspector

def test_hierachical_iaf_nmda():
    nineml_component = TestableComponent('hierachical_iaf_nmda')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    timeHorizon = 1
    reportingInterval = 0.001
    parameters = {
                    'cobaExcit.q' : 3.0,
                    'cobaExcit.tau' : 5.0,
                    'cobaExcit.vrev' : 0.0,

                    'iaf.cm': 1.0,
                    'iaf.gl': 50.0,
                    'iaf.taurefrac': 5.0,
                    'iaf.vrest': -65.0,
                    'iaf.vreset': -65.0,
                    'iaf.vthresh': -50.0,

                    # NMDA parameters from Gertner's book, pg 53.
                    'nmda.taur': 3.0, # ms
                    'nmda.taud': 40.0, # ms
                    'nmda.gmax': 1.2, #nS
                    'nmda.E': 0.0,
                    'nmda.gamma': 0.062, #1/mV
                    'nmda.mgconc': 1.2, # mM
                    'nmda.beta': 3.57 #mM
                 }
    initial_conditions = {
                            'cobaExcit.g' : 0.0,
                            'iaf.V': parameters['iaf.vrest'],
                            'iaf.tspike': -1e99
                         }
    analog_ports_expressions = {}
    event_ports_expressions = {}
    active_regimes = {}
    variables_to_report = {
                            'iaf.V' : True,
                            'nmda.g' : True,
                            'cobaExcit.g' : True
                          }

    inspector = nineml_component_inspector()
    inspector.inspect(nineml_component, timeHorizon              = timeHorizon,
                                        reportingInterval        = reportingInterval,
                                        parameters               = parameters,
                                        initial_conditions       = initial_conditions,
                                        active_regimes           = active_regimes,
                                        analog_ports_expressions = analog_ports_expressions,
                                        event_ports_expressions  = event_ports_expressions,
                                        variables_to_report      = variables_to_report)
    results = inspector.showQtGUI()
    return results, inspector

if __name__ == "__main__":
    # test_iaf()
    # test_coba_synapse
    # test_hierachical_iaf_nmda
    # test_hierachical_iaf_1coba
    # test_Hodgkin_Huxley
    # test_Izhikevich
    results, inspector = test_coba_synapse()
    if not results:
        exit(0)

    print('Input data from the GUI:')
    inspector.printCollectedData()

    simulation_data = daeSimulationInputData()
    simulation_data.timeHorizon              = results['timeHorizon']
    simulation_data.reportingInterval        = results['reportingInterval']
    simulation_data.parameters               = results['parameters']
    simulation_data.initial_conditions       = results['initial_conditions']
    simulation_data.active_regimes           = results['active_regimes']
    simulation_data.analog_ports_expressions = results['analog_ports_expressions']
    simulation_data.event_ports_expressions  = results['event_ports_expressions']
    simulation_data.variables_to_report      = results['variables_to_report']

    print('JSON data:')
    jsonContent = simulation_data.dumpJSON()
    print('jsonContent:', jsonContent)

    # Create Log, DAESolver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter() #ninemlTesterDataReporter()
    model        = nineml_daetools_bridge(inspector.ninemlComponent.name, inspector.ninemlComponent)
    simulation   = nineml_daetools_simulation(model, timeHorizon              = simulation_data.timeHorizon,
                                                     reportingInterval        = simulation_data.reportingInterval,
                                                     parameters               = simulation_data.parameters,
                                                     initial_conditions       = simulation_data.initial_conditions,
                                                     active_regimes           = simulation_data.active_regimes,
                                                     analog_ports_expressions = simulation_data.analog_ports_expressions,
                                                     event_ports_expressions  = simulation_data.event_ports_expressions,
                                                     variables_to_report      = simulation_data.variables_to_report)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = simulation_data.reportingInterval
    simulation.TimeHorizon       = simulation_data.timeHorizon

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model reports for all models
    #simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    #simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()
