"""********************************************************************************
                            fmi_interface.py
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
import os, sys, json
from . import auxiliary
from daetools.pyDAE import *
try:
    # python 3
    from urllib.parse import urlparse
except ImportError:
    # python 2
    from urlparse import urlparse

class fmi2Component(object):
    def __init__(self, simulation, instanceName, guid):
        assert simulation != None
        self.simulation    = simulation
        self.instanceName  = instanceName
        self.guid          = guid
        self.FMI_Interface = self.simulation.Model.GetFMIInterface()
        
    @staticmethod
    def fmi2Instantiate(instanceName, guid, resourceLocation):
        url = urlparse(resourceLocation)
        resourceDir = url[2] # item 2 returns path
        if not os.path.isdir(resourceDir):
            raise RuntimeError('Invalid resourceLocation directory specified: %s' % resourceLocation)
        
        settings_path = os.path.join(resourceDir, 'settings.json')
        f = open(settings_path, 'r')
        settings = json.loads(f.read())
        f.close()
        
        simulationFile     = settings['simulationFile']
        callableObjectName = settings['callableObjectName']
        arguments          = settings['arguments']

        simulation = auxiliary.loadSimulation(resourceDir, simulationFile, callableObjectName, arguments)
        c = fmi2Component(simulation, instanceName, guid)

        return c
    
    def fmi2Terminate(self):
        if self.simulation:
            self.simulation.Finalize()
    
    def fmi2FreeInstance(self):
        self.simulation    = None
        self.instanceName  = None
        self.guid          = None
        self.FMI_Interface = {}
        
    def fmi2SetupExperiment(self, toleranceDefined, tolerance, startTime, stopTimeDefined, stopTime):
        if stopTimeDefined:
            self.simulation.TimeHorizon       = float(stopTime)
            self.simulation.ReportingInterval = float(stopTime) / 100

        if toleranceDefined:
            self.simulation.DAESolver.RelativeTolerance = float(tolerance)
            
        self.simulation.SolveInitial()

    def fmi2EnterInitializationMode(self):
        pass
    
    def fmi2ExitInitializationMode(self):
        pass
    
    def fmi2DoStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint):
        stepTimeHorizon = currentCommunicationPoint + communicationStepSize
        self.simulation.IntegrateUntilTime(stepTimeHorizon, eDoNotStopAtDiscontinuity, True)
        self.simulation.ReportData(self.simulation.CurrentTime)
    
    def fmi2CancelStep(self):
        pass
    
    def fmi2Reset(self):
        self.simulation.SetUpVariables()
        self.simulation.Reset()
        self.simulation.SolveInitial()
    
    def fmi2GetReal(self, valReferences):
        values = []
        for i, reference in enumerate(valReferences):
            if not reference in self.FMI_Interface:
                raise RuntimeError('Reference %s not found in the component %s {%s}'% (str(reference), self.instanceName, self.guid))
            
            fmi = self.FMI_Interface[reference]
            if fmi.type == "Parameter":
                values.append( fmi.parameter.GetValue(list(fmi.indexes)) )
            elif fmi.type == "Input" or fmi.type == "Output" or fmi.type == "Local":
                values.append( fmi.variable.GetValue(list(fmi.indexes)) )
            else:
                raise RuntimeError('Cannot get real value for reference that is not a parameter|input|output|local variable: %s' % str(reference))
        return values
    
    def fmi2GetInteger(self, valReferences):
        return []
    
    def fmi2GetBoolean(self, valReferences):
        return []
    
    def fmi2GetString(self, valReferences):
        values = []
        for i, reference in enumerate(valReferences):
            if not reference in self.FMI_Interface:
                raise RuntimeError('Reference %s not found in the component %s {%s}'% (str(reference), self.instanceName, self.guid))
            
            fmi = self.FMI_Interface[reference]
            if fmi.type == "STN":
                values.append(fmi.stn.ActiveState)
            else:
                raise RuntimeError('Cannot set string value for the non-STN reference: %s' % str(reference))
        return values
    
    def fmi2SetReal(self, valReferences, values):
        for i, reference in enumerate(valReferences):
            if not reference in self.FMI_Interface:
                raise RuntimeError('Reference %s not found in the component %s {%s}'% (str(reference), self.instanceName, self.guid))
            
            fmi = self.FMI_Interface[reference]
            if fmi.type == "Parameter":
                fmi.parameter.SetValue(list(fmi.indexes), values[i])
            elif fmi.type == "Input":
                fmi.variable.ReAssignValue(list(fmi.indexes), values[i])
            else:
                raise RuntimeError('Cannot set real value for reference that is not a parameter or input variable: %s' % str(reference))
    
    def fmi2SetInteger(self, valReferences, values):
        pass
    
    def fmi2SetBoolean(self, valReferences, values):
        pass
    
    def fmi2SetString(self, valReferences, values):
        for i, reference in enumerate(valReferences):
            if not reference in self.FMI_Interface:
                raise RuntimeError('Reference %s not found in the component %s {%s}'% (str(reference), self.instanceName, self.guid))
            
            fmi = self.FMI_Interface[reference]
            if fmi.type == "STN":
                fmi.stn.ActiveState = str(values[i])
            else:
                raise RuntimeError('Cannot set string value for the non-STN reference: %s' % str(reference))
