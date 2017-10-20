#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""********************************************************************************
                            daetools_ws.py
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
import os, sys, json, numpy, traceback
import uuid, zipfile, cgitb, importlib, hashlib, threading
from daetools.pyDAE import *
from daetools.dae_simulator.auxiliary import loadSimulation, loadTutorial
from daetools.dae_simulator.web_service import daeWebService, ServerError, BadRequest
    
__WebAppName__  = 'daetools_ws'

def toBool(val):
  return val.lower() in ("yes", "true", "t", "1")

class daeSimulationWebService(daeWebService):
    def __init__(self):
        daeWebService.__init__(self)
        
        self.availableSimulations = {}
    
    def __call__(self, environ, start_response):
        # HTTP request handler function.
        self.noHTTPRequests += 1
        simulationID = None
        try:
            pathInfo, args = self.getQueryArguments(environ) 
            root_dir = os.path.dirname(os.path.realpath(__file__))
            
            # Serve files:
            #filename = os.path.basename(pathInfo)
            #filepath = os.path.join(root_dir, filename)
            #if os.path.exists(filepath):
            #    return self._sendFile(filepath, 'application/octet-stream', start_response)
            
            if pathInfo != ('/%s' % __WebAppName__):
                raise BadRequest('Invalid path %s requested (/%s available)' % (pathInfo, __WebAppName__))
            
            if 'function' not in args:
                raise BadRequest('No function argument specified in the query: %s' % pathInfo)
            else:
                function = args['function'].value
                        
            if function == 'shutdown':
                # Finalize simulations and free resources for all components
                for simulationID,simulation in self.activeObjects.items():
                    try:
                        simulation.fmi2Terminate()
                        simulation.fmi2FreeInstance()
                    except:
                        # We do not care about errors here.
                        pass
                # Empty the storage and set noHTTPRequests to >0 (meaning: stop the server)
                self.activeObjects  = {}
                self.noHTTPRequests = 1
                return self.jsonSuccess(start_response)
            
            elif function == 'clear':
                # Finalize simulations and free resources for all components
                for simulationID,simulation in self.activeObjects.items():
                    try:
                        simulation.fmi2Terminate()
                        simulation.fmi2FreeInstance()
                    except:
                        # We do not care about errors here.
                        pass
                # Empty the storage and set noHTTPRequests to 0 (meaning: keep the server running)
                self.activeObjects  = {}
                self.noHTTPRequests = 0
                return self.jsonSuccess(start_response)
            
            elif function == 'status':
                activeObjects = []
                response = {'ActiveClients': activeObjects}
                for simulationID, simulation in self.activeObjects.items():
                    activeObjects.append({'simulationID': simulationID,
                                          'modelName':    simulation.m.Name})
                return self.jsonResult(response, start_response)
            
            elif function == 'LoadSimulation':
                # Get the function arguments
                pythonFile   = str(args['pythonFile'].value)
                loadCallable = str(args['loadCallable'].value)
                arguments    = str(args['arguments'].value)

                # Load the simulation
                directory, simulationFile = os.path.split(pythonFile)
                try:
                    simulation = loadSimulation(directory, simulationFile, loadCallable, arguments)
                except Exception as e:
                    raise BadRequest(str(e))
                
                # Create model info
                simulation.__ModelInfo__ = {}
                simulation.__ModelInfo__['Domains']    = {}
                simulation.__ModelInfo__['Parameters'] = {}
                simulation.__ModelInfo__['Variables']  = {}
                simulation.__ModelInfo__['STNs']       = {}
                _collectModelInfo(simulation.__ModelInfo__, simulation.m)
                
                # Create an ID for the simulation object (to be used with the subsequent calls)
                simulationID = str(uuid.uuid1())

                # Store the created object in a dictionary
                self.activeObjects[simulationID] = simulation
                
                # Return the component ID
                response = {'simulationID': str(simulationID)}
                return self.jsonResult(response, start_response)
            
            elif function == 'LoadTutorial':
                # Get the function arguments
                tutorialName = str(args['tutorialName'].value)

                # Load the simulation
                loadCallable = 'run'
                arguments    = 'initializeAndReturn=True, datareporter = daeNoOpDataReporter()'
                
                try:
                    simulation = loadTutorial(tutorialName, loadCallable, arguments)
                except Exception as e:
                    raise BadRequest(str(e))
                
                # Create model info
                simulation.__ModelInfo__ = {}
                simulation.__ModelInfo__['Domains']    = {}
                simulation.__ModelInfo__['Parameters'] = {}
                simulation.__ModelInfo__['Variables']  = {}
                simulation.__ModelInfo__['STNs']       = {}
                _collectModelInfo(simulation.__ModelInfo__, simulation.m)
                
                # Create an ID for the simulation object (to be used with the subsequent calls)
                simulationID = str(uuid.uuid1())
                
                # Store the created object in a dictionary
                self.activeObjects[simulationID] = simulation
                
                # Return the component ID
                response = {'simulationID': str(simulationID)}
                return self.jsonResult(response, start_response)
            
            elif function == 'LoadSimulationByName':
                # Get the function arguments
                simulationName = str(args['simulationName'].value)
                arguments      = json.loads(args['arguments'].value)

                if not isinstance(arguments, dict):
                    raise BadRequest('The arguments must be a dictionary object')

                # Load the simulation
                if simulationName not in self.availableSimulations:
                    raise BadRequest('Simulation %s does not exist in the list of available simulations' % simulationName)
                
                simulationCallable = self.availableSimulations[simulationName]
                if not callable(simulationCallable):
                    raise BadRequest('Simulation %s does not provide valid callable for loading' % simulationName)
                
                try:
                    simulation = simulationCallable(**arguments)
                except Exception as e:
                    raise BadRequest(str(e))
                
                # Create model info
                simulation.__ModelInfo__ = {}
                simulation.__ModelInfo__['Domains']    = {}
                simulation.__ModelInfo__['Parameters'] = {}
                simulation.__ModelInfo__['Variables']  = {}
                simulation.__ModelInfo__['STNs']       = {}
                _collectModelInfo(simulation.__ModelInfo__, simulation.m)
                
                # Create an ID for the simulation object (to be used with the subsequent calls)
                simulationID = str(uuid.uuid1())
                
                # Store the created object in a dictionary
                self.activeObjects[simulationID] = simulation
                
                # Return the component ID
                response = {'simulationID': str(simulationID)}
                return self.jsonResult(response, start_response)
                
            elif function == 'AvailableSimulations':
                # Get the function arguments
                # Get the list of simulations
                simNames = list(self.availableSimulations.keys())
                
                # Return the list with the names
                response = {'AvailableSimulations': simNames}
                return self.jsonResult(response, start_response)

            elif function == 'GetModelInfo':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Prepare the model info dictionary
                modelInfo = {}
                modelInfo['Domains']    = [name for name, obj in simulation.__ModelInfo__['Domains'].items()]
                modelInfo['Parameters'] = [name for name, obj in simulation.__ModelInfo__['Parameters'].items()]
                modelInfo['Variables']  = [name for name, obj in simulation.__ModelInfo__['Variables'].items()]
                modelInfo['STNs']       = [name for name, obj in simulation.__ModelInfo__['STNs'].items()]
                
                # Return the model info
                response = {'ModelInfo': modelInfo}
                return self.jsonResult(response, start_response)

            elif function == 'GetName':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Return the top-level model name
                response = {'Name': simulation.m.Name}
                return self.jsonResult(response, start_response)

            elif function == 'GetCurrentTime':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Return the CurrentTime
                response = {'CurrentTime': simulation.CurrentTime} # hex
                return self.jsonResult(response, start_response)
            
            elif function == 'GetTimeHorizon':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Return the TimeHorizon
                response = {'TimeHorizon': simulation.TimeHorizon} # hex
                return self.jsonResult(response, start_response)
            
            elif function == 'SetTimeHorizon':
                # Get the function arguments (if any)                
                timeHorizon = float(args['timeHorizon'].value)
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                simulation.TimeHorizon = timeHorizon

                return self.jsonSuccess(start_response)

            elif function == 'GetReportingInterval':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Return the ReportingInterval
                response = {'ReportingInterval': simulation.ReportingInterval} # hex
                return self.jsonResult(response, start_response)
             
            elif function == 'SetReportingInterval':
                # Get the function arguments (if any)                
                reportingInterval = float(args['reportingInterval'].value)
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                simulation.ReportingInterval = reportingInterval

                return self.jsonSuccess(start_response)
           
            elif function == 'Run':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                simulation.Run()
                
                return self.jsonSuccess(start_response)
            
            elif function == 'Finalize':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                simulation.Finalize()
                del self.activeObjects[simulationID]
                
                # Changed to use keepRunning function.
                # If the number of fmi objects is zero shutdown the server.
                # Otherwise, continue serving the requests from the existing objects. 
                #if len(self.activeObjects) == 0:
                #    threading.Thread(target = httpd.shutdown).start()

                return self.jsonSuccess(start_response)
            
            elif function == 'SolveInitial':
                # Get the function arguments (if any)
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                simulation.SolveInitial()

                return self.jsonSuccess(start_response)
            
            elif function == 'Reinitialize':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                simulation.Reinitialize()

                return self.jsonSuccess(start_response)
                
            elif function == 'Reset':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                simulation.Reset()

                return self.jsonSuccess(start_response)
                
            elif function == 'ReportData':
                # Get the function arguments (if any)                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                simulation.ReportData(simulation.CurrentTime)

                return self.jsonSuccess(start_response)
            
            elif function == 'Integrate':
                # Get the function arguments (if any)
                stopAtDiscontinuity             = toBool(args['stopAtDiscontinuity'].value)
                reportDataAroundDiscontinuities = toBool(args['reportDataAroundDiscontinuities'].value)

                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                stopMode = (eStopAtModelDiscontinuity if stopAtDiscontinuity else eDoNotStopAtDiscontinuity)
                timeReached = simulation.Integrate(stopMode, reportDataAroundDiscontinuities)

                # Return the time reached by simulation
                response = {'TimeReached': timeReached} # hex
                return self.jsonResult(response, start_response)
            
            elif function == 'IntegrateForTimeInterval':
                # Get the function arguments (if any)
                timeInterval                    = float (args['timeInterval'].value)
                stopAtDiscontinuity             = toBool(args['stopAtDiscontinuity'].value)
                reportDataAroundDiscontinuities = toBool(args['reportDataAroundDiscontinuities'].value)

                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                stopMode = (eStopAtModelDiscontinuity if stopAtDiscontinuity else eDoNotStopAtDiscontinuity)
                timeReached = simulation.IntegrateForTimeInterval(timeInterval, stopMode, reportDataAroundDiscontinuities)

                # Return the time reached by simulation
                response = {'TimeReached': timeReached} # hex
                return self.jsonResult(response, start_response)
            
            elif function == 'IntegrateUntilTime':
                # Get the function arguments (if any)
                timeHorizon                     = float (args['time'].value)
                stopAtDiscontinuity             = toBool(args['stopAtDiscontinuity'].value)
                reportDataAroundDiscontinuities = toBool(args['reportDataAroundDiscontinuities'].value)

                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                stopMode = (eStopAtModelDiscontinuity if stopAtDiscontinuity else eDoNotStopAtDiscontinuity)
                timeReached = simulation.IntegrateUntilTime(timeHorizon, stopMode, reportDataAroundDiscontinuities)

                # Return the time reached by simulation
                response = {'TimeReached': timeReached} # hex
                return self.jsonResult(response, start_response)
            
            elif function == 'GetParameterValue':
                # Get the function arguments (if any)  
                parameterName = str(args['name'].value)
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                parameters = simulation.__ModelInfo__['Parameters']
                parameter  = parameters[parameterName]
                values     = parameter.npyValues
                if not isinstance(values, float):
                    values  = values.tolist()
                    
                # Return the values array
                response = {'Values': values} # hex
                return self.jsonResult(response, start_response)
            
            elif function == 'GetVariableValue':
                # Get the function arguments (if any)  
                variableName = str(args['name'].value)
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                variables = simulation.__ModelInfo__['Variables']
                variable  = variables[variableName]
                values    = variable.npyValues
                if not isinstance(values, float):
                    values  = values.tolist()
                    
                # Return the values array
                response = {'Values': values} # hex
                return self.jsonResult(response, start_response)
            
            elif function == 'GetActiveState':
                # Get the function arguments (if any)  
                stnName = str(args['stnName'].value)
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                stns = simulation.__ModelInfo__['STNs']
                stn  = stns[stnName]
                activeState = stn.ActiveState
                    
                # Return the activeState
                response = {'ActiveState': activeState}
                return self.jsonResult(response, start_response)
            
            elif function == 'SetParameterValue':
                # Get the function arguments (if any)  
                parameterName = str(args['name'].value)
                value         = json.loads(str(args['value'].value))
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                parameters = simulation.__ModelInfo__['Parameters']
                parameter  = parameters[parameterName]
                if isinstance(value, float):
                    parameter.SetValue(value)
                else:
                    parameter.SetValues(value)
                    
                return self.jsonSuccess(start_response)
            
            elif function == 'ReAssignValue':
                # Get the function arguments (if any)  
                variableName = str(args['name'].value)
                value        = json.loads(str(args['value'].value))
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                variables = simulation.__ModelInfo__['Variables']
                variable  = variables[variableName]
                if isinstance(value, float):
                    variable.ReAssignValue(value)
                else:
                    variable.ReAssignValues(numpy.array(value))
                    
                return self.jsonSuccess(start_response)
            
            elif function == 'ReSetInitialCondition':
                # Get the function arguments (if any)  
                variableName = str(args['name'].value)
                value        = json.loads(str(args['value'].value))
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                variables = simulation.__ModelInfo__['Variables']
                variable  = variables[variableName]
                if isinstance(value, float):
                    variable.ReSetInitialCondition(value)
                else:
                    variable.ReSetInitialConditions(numpy.array(value))
                    
                return self.jsonSuccess(start_response)
            
            elif function == 'SetActiveState':
                # Get the function arguments (if any)  
                stnName     = str(args['stnName'].value)
                activeState = str(args['activeState'].value)
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                stns = simulation.__ModelInfo__['STNs']
                stn  = stns[stnName]
                stn.ActiveState = activeState
                    
                return self.jsonSuccess(start_response)
            
            elif function == 'DAESolver.GetRelativeTolerance':
                # Get the function arguments (if any)  
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                relativeTolerance = simulation.DAESolver.RelativeTolerance
                    
                # Return the relativeTolerance
                response = {'RelativeTolerance': relativeTolerance}
                return self.jsonResult(response, start_response)
            
            elif function == 'DAESolver.SetRelativeTolerance':
                # Get the function arguments (if any)  
                relativeTolerance = float(args['relativeTolerance'].value)
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Call the function
                simulation.DAESolver.RelativeTolerance = relativeTolerance
                    
                return self.jsonSuccess(start_response)
            
            elif function == 'DataReporter.AllValues':
                # Get the function arguments (if any)  
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                                
                # Get the data
                datareporter = simulation.DataReporter
                process      = datareporter.Process
                variables    = process.dictVariables
                varValues    = process.dictVariableValues
                
                values = {}
                for variableName, (ndarr_values, ndarr_times, l_domains, s_units) in varValues.items():
                    name = variableName      #_removeRootModel(variableName)
                    
                    variable    = variables[variableName]
                    domainNames = [_removeRootModel(d.Name) for d in variable.Domains]
                    
                    value = {'ShortName'     : _nameFromCanonical(variableName),
                             'Name'          : variableName,
                             'Values'        : ndarr_values.tolist(),
                             'Times'         : ndarr_times.tolist(),
                             'DomainNames'   : domainNames,
                             'Domains'       : [d.tolist() for d in l_domains],
                             'Units'         : s_units
                            }
                    values[name] = value
                    
                # Return the values dictionary
                response = {'AllValues': values}
                return self.jsonResult(response, start_response)
                
            elif function == 'DataReporter.Value':
                # Get the function arguments (if any)
                # variableName is full canonical name
                variableName = str(args['variableName'].value)
                
                # Get the simulation object from the dictionary
                simulation, simulationID = self._getSimulation(args)
                
                # Get the data
                datareporter = simulation.DataReporter
                process      = datareporter.Process
                variables    = process.dictVariables
                varValues    = process.dictVariableValues
                
                # Pre-pend the variable name with the top-level model name
                name = variableName     # '%s.%s' % (simulation.m.Name, variableName)
                
                ndarr_values, ndarr_times, l_domains, s_units = varValues[name]
                variable    = variables[name]
                domainNames = [_removeRootModel(d.Name) for d in variable.Domains]
                
                values = {'ShortName'     : _removeRootModel(name),
                          'Name'          : variable.Name,
                          'Values'        : ndarr_values.tolist(),
                          'Times'         : ndarr_times.tolist(),
                          'DomainNames'   : domainNames,
                          'Domains'       : [d.tolist() for d in l_domains],
                          'Units'         : s_units
                         }
                    
                # Return the values dictionary
                response = {'Value': values}
                return self.jsonResult(response, start_response)
                
            else:
                return self.jsonBadRequest('Invalid function specified: %s' % function, start_response, simulationID) 

        except BadRequest as e:
            print(str(e))
            return self.jsonBadRequest(str(e), start_response, simulationID)                
            
        except ServerError as e:
            print(str(e))
            return self.jsonError(str(e), start_response, simulationID)                

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            ltb = traceback.format_exception(exc_type, exc_value, exc_traceback)
            return self.jsonError(''.join(ltb), start_response, simulationID)                

    def _getSimulation(self, args):
        # Returns the simulation object from the dictionary fr the given simulationID.
        # Raises an exception if the simulationID has not been sent or it does not exist.
        if 'simulationID' not in args:
            raise BadRequest('The query arguments do not contain the simulation ID string')   
        
        simulationID = args['simulationID'].value   
        if simulationID not in self.activeObjects:
            raise BadRequest('Simulation %s does not exist or is terminated' % simulationID)
        
        simulation = self.activeObjects[simulationID]
        return simulation, simulationID
    
    @staticmethod
    def runSimulationsAsWebService(availableSimulations, address = '127.0.0.1', port = 8001):
        """
        Starts the daetools web service providing the simulation via their names.
        Argument 'simulations' is a dictionary: {name: callable}
        """
        if not isinstance(availableSimulations, dict):
            raise RuntimeError('The daeRunSimulationsAsWebService function argument availableSimulations must be a dictionary')
        
        application = daeSimulationWebService()
        
        # Populate the dictionary with the inputs from the argument availableSimulations
        for simulationName, loaderFunction in availableSimulations.items():
            if not callable(loaderFunction):
                raise RuntimeError('The loaderFunction for the simulation %s is not a callable' % simulationName)
            application.availableSimulations[simulationName] = loaderFunction
        
        # Start the web service
        daeWebService.startWebService(application, address, port, True)    

def _collectModelInfoFromPort(modelInfo, port):
    for domain in port.Domains:
        lcanonicalName = domain.CanonicalName.split('.')[1:]
        canonicalName  = '.'.join(lcanonicalName)
        modelInfo['Domains'][canonicalName] = domain

    for parameter in port.Parameters:
        lcanonicalName = parameter.CanonicalName.split('.')[1:]
        canonicalName  = '.'.join(lcanonicalName)
        modelInfo['Parameters'][canonicalName] = parameter

    for variable in port.Variables:
        lcanonicalName = variable.CanonicalName.split('.')[1:]
        canonicalName  = '.'.join(lcanonicalName)
        modelInfo['Variables'][canonicalName] = variable

def _collectModelInfo(modelInfo, model):
    for domain in model.Domains:
        lcanonicalName = domain.CanonicalName.split('.')[1:]
        canonicalName  = '.'.join(lcanonicalName)
        modelInfo['Domains'][canonicalName] = domain

    for parameter in model.Parameters:
        lcanonicalName = parameter.CanonicalName.split('.')[1:]
        canonicalName  = '.'.join(lcanonicalName)
        modelInfo['Parameters'][canonicalName] = parameter

    for variable in model.Variables:
        lcanonicalName = variable.CanonicalName.split('.')[1:]
        canonicalName  = '.'.join(lcanonicalName)
        modelInfo['Variables'][canonicalName] = variable

    for port in model.Ports:
        _collectModelInfoFromPort(modelInfo, port)

    for stn in model.STNs:
        lcanonicalName = stn.CanonicalName.split('.')[1:]
        canonicalName  = '.'.join(lcanonicalName)
        modelInfo['STNs'][canonicalName] = stn

    for component in model.Components:
        _collectModelInfo(modelInfo, component)
        
def _removeRootModel(name):
    lnames = daeGetStrippedName(name).split('.')
    return '.'.join(lnames[1:])

def _nameFromCanonical(canonicalName):
    return daeGetStrippedName(canonicalName).split('.')[-1]

if __name__ == "__main__":
    address     = '127.0.0.1'
    port        = 8001
    application = daeSimulationWebService()        
    try:
        daeWebService.startWebService(application, address, port, True)
    except Exception as e:
        pass
