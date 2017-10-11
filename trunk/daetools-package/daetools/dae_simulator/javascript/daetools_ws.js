/***********************************************************************************
                           daetools_ws.js
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
var daetools_ws_connection_info = {
    address:        'http://127.0.0.1', 
    port:           8001, 
    webServiceName: 'daetools_ws',
    method:         'POST'
};
    
function create_daetools_ws()
{
    var webService_daetools = new daeWebService(daetools_ws_connection_info.address, 
                                                daetools_ws_connection_info.port, 
                                                daetools_ws_connection_info.webServiceName,
                                                daetools_ws_connection_info.method);
    return webService_daetools;
}

class daeDataReporter 
{
    constructor(simulation)
    {
        this.simulation = simulation;
    }
    
    Value(variableName)
    {
        var parameters = {};
        var functionName = 'DataReporter.Value';
        parameters['simulationID'] = this.simulation.simulationID;
        parameters['variableName'] = variableName;
        var response = this.simulation.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['Value'];
    }
    
    AllValues()
    {
        var parameters = {};
        var functionName = 'DataReporter.AllValues';
        parameters['simulationID'] = this.simulation.simulationID;
        var response = this.simulation.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['AllValues'];
    }
}

class daeDAESolver 
{
    constructor(simulation)
    {
        this.simulation = simulation;
    }
    
    get RelativeTolerance()
    {
        var parameters = {};
        var functionName = 'DAESolver.GetRelativeTolerance';
        parameters['simulationID'] = this.simulation.simulationID;
        var response = this.simulation.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['RelativeTolerance'];
    }
    
    set RelativeTolerance(relativeTolerance)
    {
        var parameters = {};
        var functionName = 'DAESolver.SetRelativeTolerance';
        parameters['simulationID']      = this.simulation.simulationID;
        parameters['relativeTolerance'] = relativeTolerance;
        var response = this.simulation.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
}

class daeSimulation
{
    constructor(webService)
    {
        this.webService   = webService;
        this.simulationID = null;
        this.datareporter = null;
        this.daesolver    = null;
    }
    
    LoadSimulation(pythonFile, loadCallable, args) 
    {
        var parameters = {};
        var functionName           = 'LoadSimulation';
        parameters['pythonFile']   = pythonFile;
        parameters['loadCallable'] = loadCallable;
        parameters['arguments']    = args;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        this.simulationID = response.result['simulationID'];
        this.datareporter = new daeDataReporter(this);
        this.daesolver    = new daeDAESolver(this);
    }
    
    LoadTutorial(tutorialName) 
    {
        var parameters = {};
        var functionName           = 'LoadTutorial';
        parameters['tutorialName'] = tutorialName;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        this.simulationID = response.result['simulationID'];
        this.datareporter = new daeDataReporter(this);
        this.daesolver    = new daeDAESolver(this);
    }
    
    LoadSimulationByName(simulationName, args) 
    {
        var parameters = {};
        var functionName             = 'LoadSimulationByName';
        parameters['simulationName'] = simulationName;
        parameters['arguments']      = args;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        this.simulationID = response.result['simulationID'];
        this.datareporter = new daeDataReporter(this);
        this.daesolver    = new daeDAESolver(this);
    }
    
    AvailableSimulations() 
    {
        var parameters = {};
        var functionName = 'AvailableSimulations';
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['AvailableSimulations'];
    }
    
    Finalize() 
    {
        var parameters = {};
        var functionName = 'Finalize';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        this.webService   = null;
        this.simulationID = null;
        this.datareporter = null;
    }
    
    get ModelInfo() 
    {
        var parameters = {};
        var functionName = 'GetModelInfo';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['ModelInfo'];
    }
    
    get Name() 
    {
        var parameters = {};
        var functionName = 'GetName';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['Name'];
    }

    get DataReporter()
    {
        return this.datareporter;
    }

    get DAESolver()
    {
        return this.daesolver;
    }
    
    get CurrentTime()
    {
        var parameters = {};
        var functionName = 'GetCurrentTime';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['CurrentTime'];
    }
    
    get TimeHorizon()
    {
        var parameters = {};
        var functionName = 'GetTimeHorizon';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['TimeHorizon'];
    }
    
    set TimeHorizon(timeHorizon)
    {
        var parameters = {};
        var functionName = 'SetTimeHorizon';
        parameters['simulationID'] = this.simulationID;
        parameters['timeHorizon']  = timeHorizon;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    get ReportingInterval()
    {
        var parameters = {};
        var functionName = 'GetReportingInterval';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['ReportingInterval'];
    }
    
    set ReportingInterval(reportingInterval)
    {
        var parameters = {};
        var functionName = 'SetReportingInterval';
        parameters['simulationID']      = this.simulationID;
        parameters['reportingInterval'] = reportingInterval;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }

    Run() 
    {
        var parameters = {};
        var functionName = 'Run';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
        
    SolveInitial() 
    {
        var parameters = {};
        var functionName = 'SolveInitial';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    Reinitialize() 
    {
        var parameters = {};
        var functionName = 'Reinitialize';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    Reset() 
    {
        var parameters = {};
        var functionName = 'Reset';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }

    ReportData() 
    {
        var parameters = {};
        var functionName = 'ReportData';
        parameters['simulationID'] = this.simulationID;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    Integrate(stopAtDiscontinuity, reportDataAroundDiscontinuities) 
    {
        var parameters = {};
        var functionName = 'Integrate';
        parameters['simulationID']                    = this.simulationID;
        parameters['stopAtDiscontinuity']             = stopAtDiscontinuity;
        parameters['reportDataAroundDiscontinuities'] = reportDataAroundDiscontinuities;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['TimeReached'];
    }
    
    IntegrateForTimeInterval(timeInterval, stopAtDiscontinuity, reportDataAroundDiscontinuities) 
    {
        var parameters = {};
        var functionName = 'IntegrateForTimeInterval';
        parameters['simulationID']                    = this.simulationID;
        parameters['timeInterval']                    = timeInterval;
        parameters['stopAtDiscontinuity']             = stopAtDiscontinuity;
        parameters['reportDataAroundDiscontinuities'] = reportDataAroundDiscontinuities;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['TimeReached'];
    }
    
    IntegrateUntilTime(time, stopAtDiscontinuity, reportDataAroundDiscontinuities) 
    {
        var parameters = {};
        var functionName = 'IntegrateUntilTime';
        parameters['simulationID']                    = this.simulationID;
        parameters['time']                            = time;
        parameters['stopAtDiscontinuity']             = stopAtDiscontinuity;
        parameters['reportDataAroundDiscontinuities'] = reportDataAroundDiscontinuities;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['TimeReached'];
    }
    
    GetParameterValue(name) 
    {
        var parameters = {};
        var functionName = 'GetParameterValue';
        parameters['simulationID'] = this.simulationID;
        parameters['name']         = name;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['Values'];
    }
    
    GetVariableValue(name) 
    {
        var parameters = {};
        var functionName = 'GetVariableValue';
        parameters['simulationID'] = this.simulationID;
        parameters['name']         = name;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['Values'];
    }
    
    GetActiveState(stnName) 
    {
        var parameters = {};
        var functionName = 'GetActiveState';
        parameters['simulationID'] = this.simulationID;
        parameters['stnName']      = stnName;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['ActiveState'];
    }
    
    SetParameterValue(name, value) 
    {
        var parameters = {};
        var functionName = 'SetParameterValue';
        parameters['simulationID'] = this.simulationID;
        parameters['name']         = name;
        parameters['value']        = name;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    ReAssignValue(name, value) 
    {
        var parameters = {};
        var functionName = 'ReAssignValue';
        parameters['simulationID'] = this.simulationID;
        parameters['name']         = name;
        parameters['value']        = name;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    ReSetInitialCondition(name, value) 
    {
        var parameters = {};
        var functionName = 'ReSetInitialCondition';
        parameters['simulationID'] = this.simulationID;
        parameters['name']         = name;
        parameters['value']        = name;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    SetActiveState(stnName, activeState) 
    {
        var parameters = {};
        var functionName = 'SetActiveState';
        parameters['simulationID'] = this.simulationID;
        parameters['stnName']      = stnName;
        parameters['activeState']  = activeState;
        var response = this.webService.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
}
