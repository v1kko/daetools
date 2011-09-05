#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, math
from time import localtime, strftime, time
import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from daetools.pyDAE.parser import ExpressionParser
from daetools.pyDAE import *
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from nineml_tester_gui import *
from nineml_daetools_bridge import *
from nineml_daetools_simulation import *


def test_Izhikevich():
    nineml_component = TestableComponent('izhikevich')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

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
    active_states = []
    variables_to_report = []

    app = QtGui.QApplication(sys.argv)
    s = nineml_tester_gui(nineml_component, parameters               = parameters,
                                            initial_conditions       = initial_conditions,
                                            active_states            = active_states,
                                            analog_ports_expressions = analog_ports_expressions,
                                            event_ports_expressions  = event_ports_expressions,
                                            variables_to_report      = variables_to_report)
    res = s.exec_()
    if res == QtGui.QDialog.Accepted:
        s.printResults()
        return s
    else:
        return None

def test_Hodgkin_Huxley():
    nineml_component = TestableComponent('hh')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

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
    event_ports_expressions = {
    }
    active_states = [
    ]
    variables_to_report = [
    ]

    app = QtGui.QApplication(sys.argv)
    s = nineml_tester_gui(nineml_component, parameters               = parameters,
                                            initial_conditions       = initial_conditions,
                                            active_states            = active_states,
                                            analog_ports_expressions = analog_ports_expressions,
                                            event_ports_expressions  = event_ports_expressions,
                                            variables_to_report      = variables_to_report)
    res = s.exec_()
    if res == QtGui.QDialog.Accepted:
        s.printResults()
        return s
    else:
        return None

def test_hierachical_iaf_1coba():
    nineml_component = TestableComponent('hierachical_iaf_1coba')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

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
    analog_ports_expressions = {
        'cobaExcit.V' : '1.2 * e',
        'iaf.ISyn' : '5 * pi'
    }
    event_ports_expressions = {
    }
    active_states = [
        'cobaExcit.cobadefaultregime',
        'iaf.subthresholdregime'
    ]
    variables_to_report = [
        'cobaExcit.g',
        'iaf.tspike'
    ]

    app = QtGui.QApplication(sys.argv)
    s = nineml_tester_gui(nineml_component, parameters               = parameters,
                                            initial_conditions       = initial_conditions,
                                            active_states            = active_states,
                                            analog_ports_expressions = analog_ports_expressions,
                                            event_ports_expressions  = event_ports_expressions,
                                            variables_to_report      = variables_to_report)
    res = s.exec_()
    if res == QtGui.QDialog.Accepted:
        s.printResults()
        return s
    else:
        return None

def test_hierachical_iaf_nmda():
    nineml_component = TestableComponent('hierachical_iaf_nmda')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

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
    active_states = []
    variables_to_report = [
                            'iaf.V',
                            'nmda.g',
                            'cobaExcit.g'
                          ]

    app = QtGui.QApplication(sys.argv)
    s = nineml_tester_gui(nineml_component, parameters               = parameters,
                                            initial_conditions       = initial_conditions,
                                            active_states            = active_states,
                                            analog_ports_expressions = analog_ports_expressions,
                                            event_ports_expressions  = event_ports_expressions,
                                            variables_to_report      = variables_to_report)
    res = s.exec_()
    if res == QtGui.QDialog.Accepted:
        s.printResults()
        return s
    else:
        return None

try:
    # test_hierachical_iaf_nmda
    # test_hierachical_iaf_1coba
    # test_Hodgkin_Huxley
    # test_Izhikevich
    input_data = test_Izhikevich() 
    if input_data == None:
        exit(0)

    print 'Input data from the GUI:'
    input_data.printResults()

    simulation_data = daeSimulationInputData()

    simulation_data.parameters                 = input_data.parameters
    simulation_data.initialConditions          = input_data.initialConditions
    simulation_data.inletPortExpressions       = input_data.analogPortsExpressions
    simulation_data.inletEventPortExpressions  = input_data.eventPortsExpressions
    simulation_data.activeStates               = input_data.activeStates
    simulation_data.variablesToReport          = input_data.variablesToReport

    print 'JSON data:'
    jsonContent = simulation_data.dumpJSON()
    print jsonContent
    simulation_data.importJSON(jsonContent)
    jsonContent1 = simulation_data.dumpJSON()
    print str(jsonContent1)

    # Create Log, DAESolver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    model        = nineml_daetools_bridge(input_data.ninemlComponent.name, input_data.ninemlComponent)
    simulation   = nineml_daetools_simulation(model, parameters               = input_data.parametersValues,
                                                     initial_conditions       = input_data.initialConditions,
                                                     active_states            = input_data.activeStates,
                                                     analog_ports_expressions = input_data.analogPortsExpressions,
                                                     event_ports_expressions  = input_data.eventPortsExpressions,
                                                     variables_to_report      = input_data.variablesToReport)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.1
    simulation.TimeHorizon = 10

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model reports for all models
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

except Exception, e:
    print e
