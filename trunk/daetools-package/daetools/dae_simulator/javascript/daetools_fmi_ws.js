/***********************************************************************************
                           daetools_fmi_ws.js
                 DAE Tools Project, www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************/
var daetools_fmi_ws_connection_info = {
    address:        'http://127.0.0.1', 
    port:           8002, 
    webServiceName: 'daetools_fmi_ws',
    method:         'POST'
};

function create_daetools_fmi_ws()
{
    var webService_fmi = new daeWebService(daetools_fmi_ws_connection_info.address, 
                                           daetools_fmi_ws_connection_info.port, 
                                           daetools_fmi_ws_connection_info.webServiceName,
                                           daetools_fmi_ws_connection_info.method);
    return webService_fmi;
}

function simulateFMU(fmi_simulation, reportingInterval, stopTime, relTolerance, logFunction, plotFunction)
{   
    if(fmi_simulation == null)
        return;

    var times       = [];
    var values      = [];
    var names       = [];
    var units       = [];
    var references  = [];
    var currentTime = 0.0;
    for(var key in fmi_simulation.FMI_Interface)
    {
        var obj = fmi_simulation.FMI_Interface[key];
        if(obj.type == 'Output' || obj.type == 'Local')
        {
            units.push(obj.units);
            names.push(obj.name);
            references.push(obj.reference);
        }
    }
    
    fmi_simulation.fmi2SetupExperiment(false, relTolerance, currentTime, true, stopTime);
    fmi_simulation.fmi2EnterInitializationMode();
    fmi_simulation.fmi2ExitInitializationMode();
    
    var doStep = function()
    {
        logFunction('Integrating from ' + currentTime.toFixed(2) + ' to ' + (currentTime+reportingInterval).toFixed(2) + '...');
        
        fmi_simulation.fmi2DoStep(currentTime, reportingInterval, false);
        
        currentTime += reportingInterval;
        vals = fmi_simulation.fmi2GetReal(references);
        times.push(currentTime);
        values.push(vals);
        
        var newProgress = Math.ceil(100.0 * currentTime / stopTime)
        setProgress(newProgress);
        
        if(currentTime < stopTime)
        {
            setTimeout(doStep, 0);
        }
        else
        {
            logFunction('The simulation has finished successfuly!');
            logFunction('Preparing the plots...');
            
            if(plotFunction !== null)
                plotFunction(references, values, names, units, times, 'plots');
            
            // Clean up.
            fmi_simulation.fmi2Terminate();
            fmi_simulation.fmi2FreeInstance();
            fmi_simulation = null;
        }        
    }
    
    doStep();
}

class daeFMI2Simulation 
{
    constructor(webService)
    {
        this.webService     = webService;
        this.simulationID   = null;
        this.FMI_Interface  = null;
        this.modelName      = null;
        this.startTime      = null;
        this.stopTime       = null;
        this.tolerance      = null;
    }
    
    fmi2Instantiate(instanceName, guid, resourceLocation) 
    {
        var parameters = {};
        var functionName = 'fmi2Instantiate';
        parameters['instanceName']     = instanceName;
        parameters['guid']             = guid;
        parameters['resourceLocation'] = resourceLocation;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        this.simulationID  = response.result['simulationID'];
        this.FMI_Interface = response.result['FMI_Interface'];
        this.modelName     = response.result['modelName'];
        this.startTime     = response.result['startTime'];
        this.stopTime      = response.result['stopTime'];
        this.step          = response.result['step'];
        this.tolerance     = response.result['tolerance'];
    }
    
    fmi2Terminate() 
    {
        var parameters = {};
        var functionName = 'fmi2Terminate';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    fmi2FreeInstance() 
    {
        var parameters = {};
        var functionName = 'fmi2FreeInstance';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        this.simulationID = null;
    }
    
    fmi2SetupExperiment(toleranceDefined, tolerance, startTime, stopTimeDefined, stopTime) 
    {
        var parameters = {};
        var functionName = 'fmi2SetupExperiment';
        parameters['simulationID']     = this.simulationID;
        parameters['toleranceDefined'] = toleranceDefined;
        parameters['tolerance']        = tolerance;
        parameters['startTime']        = startTime;
        parameters['stopTimeDefined']  = stopTimeDefined;
        parameters['stopTime']         = stopTime;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    fmi2EnterInitializationMode() 
    {
        var parameters = {};
        var functionName = 'fmi2EnterInitializationMode';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    fmi2ExitInitializationMode() 
    {
        var parameters = {};
        var functionName = 'fmi2ExitInitializationMode';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }

    fmi2Reset() 
    {
        var parameters = {};
        var functionName = 'fmi2Reset';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    fmi2DoStep(currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint) 
    {
        var parameters = {};
        var functionName = 'fmi2DoStep';
        parameters['simulationID']                     = this.simulationID;
        parameters['currentCommunicationPoint']        = currentCommunicationPoint
        parameters['communicationStepSize']            = communicationStepSize
        parameters['noSetFMUStatePriorToCurrentPoint'] = noSetFMUStatePriorToCurrentPoint
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    fmi2CancelStep() 
    {
        var parameters = {};
        var functionName = 'fmi2CancelStep';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    fmi2GetReal(valReferences) 
    {
        var parameters = {};
        var functionName = 'fmi2GetReal';
        parameters['simulationID']  = this.simulationID;
        parameters['valReferences'] = JSON.stringify(valReferences)
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['Values'];
    }
    
    fmi2SetReal(valReferences, values) 
    {
        var parameters = {};
        var functionName = 'fmi2SetReal';
        parameters['simulationID']  = this.simulationID;
        parameters['valReferences'] = JSON.stringify(valReferences)
        parameters['values']        = JSON.stringify(values)
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
 
    fmi2GetString(valReferences) 
    {
        var parameters = {};
        var functionName = 'fmi2GetString';
        parameters['simulationID']  = this.simulationID;
        parameters['valReferences'] = JSON.stringify(valReferences)
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['Values'];
    }
    
    fmi2SetString(valReferences, values) 
    {
        var parameters = {};
        var functionName = 'fmi2SetString';
        parameters['simulationID']  = this.simulationID;
        parameters['valReferences'] = JSON.stringify(valReferences)
        parameters['values']        = JSON.stringify(values)
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    fmi2GetBoolean(valReferences)
    {
        var parameters = {};
        var functionName = 'fmi2GetBoolean';
        parameters['simulationID']  = this.simulationID;
        parameters['valReferences'] = JSON.stringify(valReferences)
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['Values'];
    }

    fmi2SetBoolean(valReferences, values)
    {
        var parameters = {};
        var functionName = 'fmi2SetBoolean';
        parameters['simulationID']  = this.simulationID;
        parameters['valReferences'] = JSON.stringify(valReferences)
        parameters['values']        = JSON.stringify(values)
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }

    fmi2GetInteger(valReferences)
    {
        var parameters = {};
        var functionName = 'fmi2GetInteger';
        parameters['simulationID']  = this.simulationID;
        parameters['valReferences'] = JSON.stringify(valReferences)
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['Values'];
    }

    fmi2SetInteger(valReferences, values)
    {
        var parameters = {};
        var functionName = 'fmi2SetInteger';
        parameters['simulationID']  = this.simulationID;
        parameters['valReferences'] = JSON.stringify(valReferences)
        parameters['values']        = JSON.stringify(values)
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
}
