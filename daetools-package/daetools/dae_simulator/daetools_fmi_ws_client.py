#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""********************************************************************************
                            daetools_fmi_ws_client.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
import os, sys, json, random
from daetools.dae_simulator.web_service_client import daeWebServiceClient

class fmi2Component_ws(daeWebServiceClient):
    def __init__(self, webServiceName = 'daetools_fmi_ws', server = '127.0.0.1', port = 8002):
        daeWebServiceClient.__init__(self, webServiceName, server, port)
        self.simulationID = None

    def fmi2Instantiate(self, instanceName, guid, resourceLocation):
        parameters = {}
        parameters['function']         = 'fmi2Instantiate'
        parameters['instanceName']     = instanceName
        parameters['guid']             = guid
        parameters['resourceLocation'] = resourceLocation
        self.sendRequest(parameters)
        response = self.getResponse()
        self.simulationID  = response['simulationID']
        self.FMI_Interface = response['FMI_Interface']
        self.startTime     = response['startTime']
        self.stopTime      = response['stopTime']
        self.step          = response['step']
        self.tolerance     = response['tolerance']
    
    def fmi2FreeInstance(self):
        parameters = {}
        parameters['function']     = 'fmi2FreeInstance'
        parameters['simulationID'] = self.simulationID
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2SetupExperiment(self, toleranceDefined, tolerance, startTime, stopTimeDefined, stopTime):
        parameters = {}
        parameters['function']         = 'fmi2SetupExperiment'
        parameters['simulationID']     = self.simulationID
        parameters['toleranceDefined'] = toleranceDefined
        parameters['tolerance']        = tolerance
        parameters['startTime']        = startTime
        parameters['stopTimeDefined']  = stopTimeDefined
        parameters['stopTime']         = stopTime
        self.sendRequest(parameters)
        response = self.getResponse()

    def fmi2EnterInitializationMode(self):
        parameters = {}
        parameters['function']     = 'fmi2EnterInitializationMode'
        parameters['simulationID'] = self.simulationID
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2ExitInitializationMode(self):
        parameters = {}
        parameters['function']     = 'fmi2ExitInitializationMode'
        parameters['simulationID'] = self.simulationID
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2DoStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint):
        parameters = {}
        parameters['function']                         = 'fmi2DoStep'
        parameters['simulationID']                     = self.simulationID
        parameters['currentCommunicationPoint']        = currentCommunicationPoint
        parameters['communicationStepSize']            = communicationStepSize
        parameters['noSetFMUStatePriorToCurrentPoint'] = noSetFMUStatePriorToCurrentPoint
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2CancelStep(self):
        parameters = {}
        parameters['function']     = 'fmi2CancelStep'
        parameters['simulationID'] = self.simulationID
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2Terminate(self):
        parameters = {}
        parameters['function']     = 'fmi2Terminate'
        parameters['simulationID'] = self.simulationID
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2Reset(self):
        parameters = {}
        parameters['function']     = 'fmi2Reset'
        parameters['simulationID'] = self.simulationID
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2GetReal(self, valReferences):
        parameters = {}
        parameters['function']      = 'fmi2GetReal'
        parameters['simulationID']  = self.simulationID
        parameters['valReferences'] = json.dumps(valReferences)
        self.sendRequest(parameters)
        response = self.getResponse()
        if 'Values' not in response:
            raise RuntimeError('fmi2GetReal: The response does not contain the Values')
        values_hex = response['Values']
        return [float(v) for v in values_hex] # fromhex
    
    def fmi2GetInteger(self, valReferences):
        parameters = {}
        parameters['function']      = 'fmi2GetInteger'
        parameters['simulationID']  = self.simulationID
        parameters['valReferences'] = json.dumps(valReferences)
        self.sendRequest(parameters)
        response = self.getResponse()
        if 'Values' not in response:
            raise RuntimeError('fmi2GetReal: The response does not contain the Values')
        return response['Values']
    
    def fmi2GetBoolean(self, valReferences):
        parameters = {}
        parameters['function']      = 'fmi2GetBoolean'
        parameters['simulationID']  = self.simulationID
        parameters['valReferences'] = json.dumps(valReferences)
        self.sendRequest(parameters)
        response = self.getResponse()
        if 'Values' not in response:
            raise RuntimeError('fmi2GetReal: The response does not contain the Values')
        return response['Values']
    
    def fmi2GetString(self, valReferences):
        parameters = {}
        parameters['function']      = 'fmi2GetString'
        parameters['simulationID']  = self.simulationID
        parameters['valReferences'] = json.dumps(valReferences)
        self.sendRequest(parameters)
        response = self.getResponse()
        if 'Values' not in response:
            raise RuntimeError('fmi2GetReal: The response does not contain the Values')
        return response['Values']
    
    def fmi2SetReal(self, valReferences, values):
        parameters = {}
        parameters['function']      = 'fmi2SetReal'
        parameters['simulationID']  = self.simulationID
        parameters['valReferences'] = json.dumps(valReferences)
        parameters['values']        = json.dumps(['%.16f' % v for v in values]) # hex
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2SetInteger(self, valReferences, values):
        parameters = {}
        parameters['function']      = 'fmi2SetInteger'
        parameters['simulationID']  = self.simulationID
        parameters['valReferences'] = json.dumps(valReferences)
        parameters['values']        = json.dumps(values)
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2SetBoolean(self, valReferences, values):
        parameters = {}
        parameters['function']      = 'fmi2SetBoolean'
        parameters['simulationID']  = self.simulationID
        parameters['valReferences'] = json.dumps(valReferences)
        parameters['values']        = json.dumps(values)
        self.sendRequest(parameters)
        response = self.getResponse()
    
    def fmi2SetString(self, valReferences, values):
        parameters = {}
        parameters['function']      = 'fmi2SetString'
        parameters['simulationID']  = self.simulationID
        parameters['valReferences'] = json.dumps(valReferences)
        parameters['values']        = json.dumps(values)
        self.sendRequest(parameters)
        response = self.getResponse()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python daetools_fmi_ws_client.py "full_path_to_resource_directory"')
        sys.exit()
    
    name = 'test-%f' % random.random()
    guid = 'dee8cf5e-9df1-11e7-8f29-680715e7b846'
    resourceLocation = sys.argv[1]

    c = fmi2Component_ws()
    c.fmi2Instantiate(name, guid, resourceLocation)
    
    t_current   = c.startTime
    t_step      = c.step
    t_horizon   = c.stopTime
    t_tolerance = c.tolerance
    
    references = []
    names      = []    
    for key, obj in c.FMI_Interface.items():
        if obj['type'] == 'Output' or obj['type'] == 'Local':
            names.append(obj['name'])
            references.append(obj['reference'])

    c.fmi2SetupExperiment(False, t_tolerance, t_current, False, t_horizon)
    c.fmi2EnterInitializationMode()
    c.fmi2ExitInitializationMode()
    
    line = 'time'
    for i in range(len(references)):
        line += ', %s' % names[i]
    print(line)
    
    while t_current < t_horizon:
        c.fmi2DoStep(t_current, t_step, False)
        t_current += t_step
        
        values = c.fmi2GetReal(references)
        
        line = '%.14f' % t_current
        for i in range(len(values)):
            line += ', %.14f' % values[i]
        print(line)

    c.fmi2Terminate()
    c.fmi2FreeInstance()
   
