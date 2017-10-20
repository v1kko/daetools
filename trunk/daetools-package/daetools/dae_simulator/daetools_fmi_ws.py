#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""********************************************************************************
                            daetools_fmi_ws.py
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
import uuid, zipfile, cgitb, importlib, hashlib, threading
from daetools.pyDAE import *
from daetools.dae_simulator.fmi_interface import fmi2Component
from daetools.dae_simulator.web_service import daeWebService, ServerError, BadRequest
    
__WebAppName__  = 'daetools_fmi_ws'

def toBool(val):
  return val.lower() in ("yes", "true", "t", "1")

class daeFMI2_CoS_WebService(daeWebService):
    def __init__(self):
        daeWebService.__init__(self)
    
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
                raise BadRequest('No function argument specified in the query: %' % pathInfo)    
            
            function = args['function'].value
                        
            if function == 'shutdown':
                # Finalize simulations and free resources for all components
                for simulationID,c in self.activeObjects.items():
                    try:
                        c.fmi2Terminate()
                        c.fmi2FreeInstance()
                    except:
                        # We do not care about errors here.
                        pass
                # Empty the storage and set noHTTPRequests to >0 (meaning: stop the server)
                self.activeObjects  = {}
                self.noHTTPRequests = 1
                return self.jsonSuccess(start_response)
            
            elif function == 'clear':
                # Finalize simulations and free resources for all components
                for simulationID,c in self.activeObjects.items():
                    try:
                        c.fmi2Terminate()
                        c.fmi2FreeInstance()
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
                for simulationID, c in self.activeObjects.items():
                    activeObjects.append({'simulationID': simulationID,
                                          'instanceName':c.instanceName,
                                          'guid' :       c.guid})
                return self.jsonResult(response, start_response)
            
            elif function == 'fmi2Instantiate':
                # Get the function arguments
                instanceName     = str(args['instanceName'].value)
                guid             = str(args['guid'].value)
                resourceLocation = str(args['resourceLocation'].value)
                
                # Create an ID for the fmi2Component object (to be used with the subsequent calls)
                md5 = hashlib.md5()
                key = '%s-%s' % (instanceName, guid)
                md5.update(key.encode())
                simulationID = md5.hexdigest()

                # We could forbid creation of FMUs with the same name and guid,
                # but we allow it for the time being.
                #if simulationID in self.activeObjects:
                #    simulationID = None
                #    raise BadRequest('FMI component: %s (GUID=%s) already instantiated' % (instanceName, guid))

                # Create fmi2Component object and load the simulation
                try:
                    c = fmi2Component.fmi2Instantiate(instanceName, guid, resourceLocation)
                except Exception as e:
                    raise BadRequest(str(e))
                
                # Store the created fmi2Component object in a dictionary
                self.activeObjects[simulationID] = c
                
                FMI_Interface = {}
                for varReference, obj in c.FMI_Interface.items():
                    item = {}
                    item['name']        = obj.name
                    item['type']        = obj.type
                    item['description'] = obj.description
                    item['units']       = obj.units
                    item['reference']   = obj.reference
                    #item['indexes']     = obj.indexes
                    FMI_Interface[varReference] = item
                startTime = 0.0
                stopTime  = c.simulation.TimeHorizon
                step      = c.simulation.ReportingInterval
                tolerance = c.simulation.DAESolver.RelativeTolerance
                modelName = c.simulation.m.Name
                
                # Return the component ID
                response = {'simulationID': str(simulationID),
                            'modelName': modelName,
                            'startTime': startTime,
                            'stopTime': stopTime,
                            'step': step,
                            'tolerance': tolerance,
                            'FMI_Interface': FMI_Interface}
                return self.jsonResult(response, start_response)
            
            elif function == 'fmi2Terminate':
                # Get the function arguments (if any)                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2Terminate()
                
                return self.jsonSuccess(start_response)
            
            elif function == 'fmi2FreeInstance':
                # Get the function arguments (if any)                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2FreeInstance()
                del self.activeObjects[simulationID]
                
                # Changed to use keepRunning function.
                # If the number of fmi objects is zero shutdown the server.
                # Otherwise, continue serving the requests from the existing objects. 
                #if len(self.activeObjects) == 0:
                #    threading.Thread(target = httpd.shutdown).start()

                return self.jsonSuccess(start_response)
            
            elif function == 'fmi2SetupExperiment':
                # Get the function arguments (if any)
                toleranceDefined = toBool(args['toleranceDefined'].value)
                tolerance        = float (args['tolerance'].value)
                startTime        = float (args['startTime'].value)
                stopTimeDefined  = toBool(args['stopTimeDefined'].value)
                stopTime         = float (args['stopTime'].value)

                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2SetupExperiment(toleranceDefined, tolerance, startTime, stopTimeDefined, stopTime)

                return self.jsonSuccess(start_response)
            
            elif function == 'fmi2EnterInitializationMode':
                # Get the function arguments (if any)                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2EnterInitializationMode()

                return self.jsonSuccess(start_response)
                
            elif function == 'fmi2ExitInitializationMode':
                # Get the function arguments (if any)                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2ExitInitializationMode()

                return self.jsonSuccess(start_response)
            
            elif function == 'fmi2DoStep':
                # Get the function arguments (if any)
                currentCommunicationPoint        = float (args['currentCommunicationPoint'].value)
                communicationStepSize            = float (args['communicationStepSize'].value)
                noSetFMUStatePriorToCurrentPoint = toBool(args['noSetFMUStatePriorToCurrentPoint'].value)

                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2DoStep(currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint)

                return self.jsonSuccess(start_response)
            
            elif function == 'fmi2CancelStep':
                # Get the function arguments (if any)                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2CancelStep()

                return self.jsonSuccess(start_response)
                
            elif function == 'fmi2Reset':
                # Get the function arguments (if any)                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2Reset()

                return self.jsonSuccess(start_response)
            
            elif function == 'fmi2GetReal':
                # Get the function arguments (if any)  
                valReferences_s = str(args['valReferences'].value)
                valReferences = [int(ref) for ref in json.loads(valReferences_s)]
                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                values = c.fmi2GetReal(valReferences)

                # Return the values array
                response = {'Values': values} # hex
                return self.jsonResult(response, start_response)
            
            elif function == 'fmi2GetInteger':
                # Get the function arguments (if any)                
                valReferences_s = str(args['valReferences'].value)
                valReferences = [int(ref) for ref in json.loads(valReferences_s)]
                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                values = c.fmi2GetInteger(valReferences)

                # Return the values array
                response = {'Values': values}
                return self.jsonResult(response, start_response)
            
            elif function == 'fmi2GetBoolean':
                # Get the function arguments (if any)                
                valReferences_s = str(args['valReferences'].value)
                valReferences = [int(ref) for ref in json.loads(valReferences_s)]
                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                values = c.fmi2GetBoolean(valReferences)

                # Return the values array
                response = {'Values': values}
                return self.jsonResult(response, start_response)
            
            elif function == 'fmi2GetString':
                # Get the function arguments (if any)                
                valReferences_s = str(args['valReferences'].value)
                valReferences = [int(ref) for ref in json.loads(valReferences_s)]
                
                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                values = c.fmi2GetString(valReferences)

                # Return the values array
                response = {'Values': values}
                return self.jsonResult(response, start_response)
            
            elif function == 'fmi2SetReal':
                # Get the function arguments (if any)                
                valReferences_s = str(args['valReferences'].value)
                values_s        = str(args['values'].value)
                valReferences = [int(ref)   for ref in json.loads(valReferences_s)]
                values        = [float(val) for val in json.loads(values_s)] # fromhex

                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2SetReal(valReferences, values)

                return self.jsonSuccess(start_response)
            
            elif function == 'fmi2SetInteger':
                # Get the function arguments (if any)                
                valReferences_s = str(args['valReferences'].value)
                values_s        = str(args['values'].value)
                valReferences = [int(ref) for ref in json.loads(valReferences_s)]
                values        = [int(val) for val in json.loads(values_s)]

                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2SetInteger(valReferences, values)

                return self.jsonSuccess(start_response)
            
            elif function == 'fmi2SetBoolean':
                # Get the function arguments (if any)                
                valReferences_s = str(args['valReferences'].value)
                values_s        = str(args['values'].value)
                valReferences = [int(ref)  for ref in json.loads(valReferences_s)]
                values        = [bool(val) for val in json.loads(values_s)]

                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2SetBoolean(valReferences, values)

                return self.jsonSuccess(start_response)
            
            elif function == 'fmi2SetString':
                # Get the function arguments (if any)                
                valReferences_s = str(args['valReferences'].value)
                values_s        = str(args['values'].value)
                valReferences = [int(ref) for ref in json.loads(valReferences_s)]
                values        = [str(val) for val in json.loads(values_s)]

                # Get the FMI component from the dictionary
                c, simulationID = self._getFMIComponent(args)
                
                # Call the function
                c.fmi2SetString(valReferences, values)

                return self.jsonSuccess(start_response)
            
            else:
                return self.jsonBadRequest('Invalid function specified: %s' % function, start_response, simulationID) 

        except BadRequest as e:
            print(str(e))
            return self.jsonBadRequest(str(e), start_response, simulationID)                
            
        except ServerError as e:
            print(str(e))
            return self.jsonError(str(e), start_response, simulationID)                

        except Exception as e:
            print(str(e))
            return self.jsonError(str(e), start_response, simulationID)                

    def _getFMIComponent(self, args):
        # Returns the FMI component from the dictionary fr the given simulationID.
        # Raises an exception if the simulationID has not been sent or it does not exist.
        if 'simulationID' not in args:
            raise BadRequest('The query arguments do not contain the simulation ID string')   
        
        simulationID = args['simulationID'].value   
        if simulationID not in self.activeObjects:
            raise BadRequest('FMI component: %s does not exist or is terminated' % simulationID)
        
        component = self.activeObjects[simulationID]
        return component, simulationID

if __name__ == "__main__":
    address     = '127.0.0.1'
    port        = 8002
    application = daeFMI2_CoS_WebService()    
    try:
        daeWebService.startWebService(application, address, port, True)
    except Exception as e:
        pass
