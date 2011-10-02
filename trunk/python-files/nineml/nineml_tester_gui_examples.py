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

def test_Izhikevich():
    nineml_component = TestableComponent('izhikevich')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    timeHorizon = 1
    reportingInterval = 0.001
    parameters = {
                    "Izhikevich.a": 0.02,
                    "Izhikevich.b": 0.2,
                    "Izhikevich.c": -0.05,
                    "Izhikevich.d": 2.0,
                    "Izhikevich.theta": 0.03
                 }
    initial_conditions = {
                            "Izhikevich.U": 0.0,
                            "Izhikevich.V": -0.07
                         }
    analog_ports_expressions = {
                                  "Izhikevich.Isyn": "0.1"
                               }
    event_ports_expressions = {}
    active_regimes = {}
    variables_to_report = {
                             'Izhikevich.U' : True,
                             'Izhikevich.V' : True
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

def test_hierachical_iaf_1coba():
    nineml_component = TestableComponent('hierachical_iaf_1coba')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    timeHorizon =  1.0
    reportingInterval = 0.001 
    initial_conditions = {
        "iaf_1coba.iaf.tspike": -1e+99, 
        "iaf_1coba.iaf.V": -0.06, 
        "iaf_1coba.cobaExcit.g": 0.0
    }
    parameters = {
        "iaf_1coba.iaf.gl": 50.0, 
        "iaf_1coba.cobaExcit.vrev": 0.0, 
        "iaf_1coba.cobaExcit.q": 3.0, 
        "iaf_1coba.iaf.vreset": -0.06, 
        "iaf_1coba.cobaExcit.tau": 5.0, 
        "iaf_1coba.iaf.taurefrac": 0.008, 
        "iaf_1coba.iaf.vthresh": -0.04, 
        "iaf_1coba.iaf.vrest": -0.06, 
        "iaf_1coba.iaf.cm": 1.0
    } 
    variables_to_report = {
        "iaf_1coba.cobaExcit.I": True, 
        "iaf_1coba.iaf.V": True
    } 
    event_ports_expressions = {
        "iaf_1coba.cobaExcit.spikeinput": "0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90"
    } 
    active_regimes = {
        "iaf_1coba.cobaExcit": "cobadefaultregime", 
        "iaf_1coba.iaf": "subthresholdregime"
    } 
    analog_ports_expressions = {}

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
        'CobaSyn.q' : 3.0,
        'CobaSyn.tau' : 5.0,
        'CobaSyn.vrev' : 0.0
    }
    initial_conditions = {
        'CobaSyn.g' : 0.0,
    }
    analog_ports_expressions = {
        'CobaSyn.V' : -0.050
    }
    event_ports_expressions = {
        'CobaSyn.spikeinput' : '0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90'
    }
    active_regimes = {
        'CobaSyn' : 'cobadefaultregime'
    }
    variables_to_report = {
        'CobaSyn.I' : True
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
        'iaf.cm' : 1,
        'iaf.gl' : 50,
        'iaf.taurefrac' : 0.008,
        'iaf.vreset' : -0.060,
        'iaf.vrest' : -0.060,
        'iaf.vthresh' : -0.040
    }
    initial_conditions = {
        'iaf.V' : -0.060,
        'iaf.tspike' : -1E99
    }
    analog_ports_expressions = {
        'iaf.ISyn' : 1.2
    }
    event_ports_expressions = {}
    active_regimes = {
        'iaf' : 'subthresholdregime'
    }
    variables_to_report = {
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

if __name__ == "__main__":
    # test_iaf
    # test_coba_synapse
    # test_hierachical_iaf_1coba
    # test_Izhikevich
    results, inspector = test_hierachical_iaf_1coba()
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
