*******************
DAE Tools Simulator
*******************
..  
    Copyright (C) Dragan Nikolic
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

Overview
=========


Classes
=======
.. py:module:: daetools.dae_simulator.simulator
.. py:currentmodule:: daetools.dae_simulator.simulator

.. autoclass:: daetools.dae_simulator.simulator.daeSimulator
    :members:
    :undoc-members:
 
.. py:module:: daetools.dae_simulator.simulation_explorer
.. py:currentmodule:: daetools.dae_simulator.simulation_explorer

.. autoclass:: daetools.dae_simulator.simulation_explorer.daeSimulationExplorer
    :members:
    :undoc-members:

.. py:module:: daetools.dae_simulator.simulation_inspector
.. py:currentmodule:: daetools.dae_simulator.simulation_inspector

.. autoclass:: daetools.dae_simulator.simulation_inspector.daeSimulationInspector
    :members:
    :undoc-members:

Web service classes
===================
.. autosummary::
    ~daetools.dae_simulator.web_service.daeWebService
    ~daetools.dae_simulator.daetools_ws.daeSimulationWebService
    ~daetools.dae_simulator.daetools_fmi_ws.daeFMI2_CoS_WebService
    
.. autoclass:: daetools.dae_simulator.web_service.BadRequest
    :members:
    :undoc-members:
    
.. autoclass:: daetools.dae_simulator.web_service.ServerError
    :members:
    :undoc-members:
    
.. autoclass:: daetools.dae_simulator.web_service.NoLoggingWSGIRequestHandler
    :members:
    :undoc-members:
    
.. autoclass:: daetools.dae_simulator.web_service.daeWebService
    :members:
    :undoc-members:
    
.. autoclass:: daetools.dae_simulator.daetools_ws.daeSimulationWebService
    :members:
    :undoc-members:
    
.. autoclass:: daetools.dae_simulator.fmi_interface.fmi2Component
    :members:
    :undoc-members:

.. autoclass:: daetools.dae_simulator.daetools_fmi_ws.daeFMI2_CoS_WebService
    :members:
    :undoc-members:

.. autoclass:: daetools.dae_simulator.web_service_client.daeWebServiceClient
    :members:
    :undoc-members:

.. autoclass:: daetools.dae_simulator.daetools_fmi_ws_client.fmi2Component_ws
    :members:
    :undoc-members:


Web service JavaScript classes 
==============================
.. autosummary::
    daeWebService
    daeDataReporter
    daeDAESolver
    daeSimulation
    daeFMI2Simulation    

.. js:class:: daeWebService(address, port, webServiceName, method)

    .. js:attribute:: ServerStatus()
    .. js:function:: ClearServer()
    .. js:function:: onSuccess(httpRequest, path, args) 
    .. js:function:: onError(httpRequest, path, args) 
    .. js:function:: onConnectionFailure(path, error)
    .. js:function:: getResponse(httpRequest) 
    .. js:function:: createHTTPRequest()
    .. js:function:: executeFun(functionName, parameters) 
    

.. js:class:: daeSimulation(webService)

    .. js:function:: LoadSimulation(pythonFile, loadCallable, args) 
    .. js:function:: LoadTutorial(tutorialName) 
    .. js:function:: LoadSimulationByName(simulationName, args) 
    .. js:function:: AvailableSimulations() 
    .. js:function:: Finalize() 
    .. js:attribute:: ModelInfo() 
    .. js:attribute:: Name() 
    .. js:attribute:: DataReporter()
    .. js:attribute:: DAESolver()
    .. js:attribute:: CurrentTime()
    .. js:attribute:: TimeHorizon()
    .. js:attribute:: ReportingInterval()
    .. js:function:: Run() 
    .. js:function:: SolveInitial() 
    .. js:function:: Reinitialize() 
    .. js:function:: Reset() 
    .. js:function:: ReportData() 
    .. js:function:: Integrate(stopAtDiscontinuity, reportDataAroundDiscontinuities) 
    .. js:function:: IntegrateForTimeInterval(timeInterval, stopAtDiscontinuity, reportDataAroundDiscontinuities) 
    .. js:function:: IntegrateUntilTime(time, stopAtDiscontinuity, reportDataAroundDiscontinuities) 
    .. js:function:: GetParameterValue(name) 
    .. js:function:: GetVariableValue(name) 
    .. js:function:: GetActiveState(stnName) 
    .. js:function:: SetParameterValue(name, value) 
    .. js:function:: ReAssignValue(name, value) 
    .. js:function:: ReSetInitialCondition(name, value) 
    .. js:function:: SetActiveState(stnName, activeState)
   
.. js:function:: create_daetools_ws

.. js:class:: daeDataReporter(simulation)

    .. js:function:: Value(variableName)    
    .. js:function:: AllValues()

.. js:class:: daeDAESolver(simulation)

    .. js:attribute:: RelativeTolerance()

.. js:class:: daeFMI2Simulation(webService)

    .. js:function:: fmi2Instantiate(instanceName, guid, resourceLocation) 
    .. js:function:: fmi2Terminate() 
    .. js:function:: fmi2FreeInstance() 
    .. js:function:: fmi2SetupExperiment(toleranceDefined, tolerance, startTime, stopTimeDefined, stopTime) 
    .. js:function:: fmi2EnterInitializationMode() 
    .. js:function:: fmi2ExitInitializationMode() 
    .. js:function:: fmi2Reset() 
    .. js:function:: fmi2DoStep(currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint) 
    .. js:function:: fmi2CancelStep() 
    .. js:function:: fmi2GetReal(valReferences) 
    .. js:function:: fmi2SetReal(valReferences, values) 
    .. js:function:: fmi2GetString(valReferences) 
    .. js:function:: fmi2SetString(valReferences, values) 
    .. js:function:: fmi2GetBoolean(valReferences)
    .. js:function:: fmi2SetBoolean(valReferences, values)
    .. js:function:: fmi2GetInteger(valReferences)
    .. js:function:: fmi2SetInteger(valReferences, values)

    
