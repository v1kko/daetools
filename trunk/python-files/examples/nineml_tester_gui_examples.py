#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation
import os, sys, math
from time import localtime, strftime, time
from daetools.pyDAE.parser import ExpressionParser
from daetools.pyDAE import *
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from nineml_tester_gui import *
from nineml_daetools_bridge import *
from nineml_daetools_simulation import *

def test_Hodgkin_Huxley():
    nineml_component = TestableComponent('hh')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    parameters = {
        'el' : 0.1,
        'C' : 0,
        'ek' : 0,
        'ena' : 0,
        'gkbar' : 0,
        'gnabar' : 0,
        'theta' : 0,
        'gl' : 0,
        'celsius' : 0
    }
    initial_conditions = {
        'n' : 0,
        'm' : 0,
        'h' : 0,
        'V' : 0
    }
    analog_ports_expressions = {
        'Isyn' : '5 * pi'
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
    # test_hierachical_iaf_nmda()
    # test_hierachical_iaf_1coba()
    # test_Hodgkin_Huxley()
    input_data = test_Hodgkin_Huxley() 
    if input_data == None:
        exit(0)

    print 'Input data from the GUI:'
    print input_data.printResults()
    
    #dictIdentifiers = {}
    #dictFunctions   = {}

    #dictIdentifiers['pi'] = math.pi
    #dictIdentifiers['e']  = math.e

    #parser = ExpressionParser(dictIdentifiers, dictFunctions)

    #for portName, expr in input_data.analogPortsExpressions.items():
    #    if expr:
    #        print 'port: {0} = {1}'.format(portName, parser.parse(expr))

    # Create Log, DAESolver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    model        = nineml_daetools_bridge(input_data.ninemlComponent.name, input_data.ninemlComponent)
    simulation   = nineml_daetools_simulation(model, parameters               = input_data.parametersValues,
                                                     initial_conditions       = input_data.initialConditions,
                                                     active_states            = input_data.initialActiveStates,
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
