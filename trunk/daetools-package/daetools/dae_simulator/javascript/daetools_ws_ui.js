/***********************************************************************************
                           daetools_ws_ui.js
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
/* Global variables. */
var webService          = create_daetools_ws();
var simulation          = null;
var currentTutorialName = null;

/* Add event listeners: 
    a) When fully loaded (document.DOMContentLoaded)
    b) Before the page is unloaded (window.beforeunload) */
document.addEventListener("DOMContentLoaded", function(event) {
    clearInputs();
    disableInputs(true);
    
    var simulations = availableSimulations();
    var tag_simulationName   = document.getElementById('simulationName');
    tag_simulationName.innerHTML = '';
    for(var i = 0; i < simulations.length; i++)
    {
        var name = simulations[i];
        var tag_option  = document.createElement('option');
        tag_option.value = name;
        tag_option.text = name;
        tag_simulationName.add(tag_option);
    }
});

window.addEventListener("beforeunload", function(event) { 
    /* If not null, finalize the simulation before unloading (it frees the resources on the server).*/
    if(simulation !== null)
    {
        if(simulation.simulationID !== null)
            simulation.Finalize();
    }
});

function availableSimulations() 
{
    var parameters = {};
    var functionName = 'AvailableSimulations';
    var response = webService.executeFun(functionName, parameters);
    if(response.success == false)
        throw response['Reason'];
    return response.result['AvailableSimulations'];
}

function messageBox(msg)
{
    var tag_mesageBox        = document.getElementById('messageBox');
    var tag_mesageBoxContent = document.getElementById('messageBoxContent');
    tag_mesageBoxContent.innerHTML = msg;
    tag_mesageBox.style.display='block';
}

/* Log messages to the specified html tag (here: 'output').*/
function log(message) 
{
    var tag_output = document.getElementById('output');
    content = tag_output.innerHTML;
    tag_output.innerHTML = content + '\n' + message;
    tag_output.scrollTop = tag_output.scrollHeight;
}

/* Set the new progress bar value.*/
function setProgress(newProgress)
{
    var tag_progress = document.getElementById('progress');
    tag_progress.style.visibility = "visible";
    tag_progress.style.width = Math.floor(newProgress) + '%';
    tag_progress.innerHTML = newProgress.toFixed(1) + '%';
}

/* Disable/enable all input fields.*/
function disableInputs(disabled)
{
    var tag_modelName         = document.getElementById('modelName');
    var tag_runButton         = document.getElementById('runButton');
    var tag_timeHorizon       = document.getElementById('timeHorizon');
    var tag_reportingInterval = document.getElementById('reportingInterval');
    tag_modelName.disabled         = disabled;
    tag_runButton.disabled         = disabled;
    tag_timeHorizon.disabled       = disabled;
    tag_reportingInterval.disabled = disabled;
}

/* Sets the current tutorial name. 
   Cleans the resource of the previous simlation (if existing).*/
function setTutorial(tutorialName)
{
    /* Run this function using the setTimeout with the delay set to 0 to improve responsiveness.*/
    currentTutorialName = tutorialName;
    
    var setTutorial_ = function() {
        if(simulation !== null)
        {
            if(simulation.simulationID !== null)
                simulation.Finalize();
            simulation = null;
        }
        simulation = loadTutorial(currentTutorialName, log);
        
        var tag_modelName         = document.getElementById('modelName');
        var tag_timeHorizon       = document.getElementById('timeHorizon');
        var tag_reportingInterval = document.getElementById('reportingInterval');
        
        clearInputs();
        disableInputs(false);

        tag_modelName.value         = simulation.Name;
        tag_timeHorizon.value       = simulation.TimeHorizon.toFixed(2);
        tag_reportingInterval.value = simulation.ReportingInterval.toFixed(2);
    }
    setTimeout(setTutorial_, 0);
}

/* Sets the current simulation name. 
   Works with the simulationsProvided dictionary in daetools_ws. 
   Cleans the resource of the previous simlation (if existing).*/
function setSimulationName()
{
    /* Run this function using the setTimeout with the delay set to 0 to improve responsiveness.*/
    var loadSimulationByName_ = function() {
        if(simulation !== null)
        {
            if(simulation.simulationID !== null)
                simulation.Finalize();
            simulation = null;
        }
        
        clearInputs();
        disableInputs(false);

        var tag_simulationName  = document.getElementById('simulationName');
        var tag_loaderArguments = document.getElementById('loaderArguments');
        var simulationName = tag_simulationName.value;
        var args_s         = tag_loaderArguments.value;
        var args = {};
        if(args_s !== '')
        {
            try 
            {
                args = JSON.parse(args_s);
            } 
            catch (e)
            {
                var msg = 'Error parsing the loader function argument: ' + args_s + '\n';
                msg += e;
                alert(msg); 
                return;
            }
        }
        
        if(simulationName == '')
            return;
        
        simulation = loadSimulationByName(simulationName, args, log);
    
        currentTutorialName = simulationName;
        
        var tag_modelName         = document.getElementById('modelName');
        var tag_timeHorizon       = document.getElementById('timeHorizon');
        var tag_reportingInterval = document.getElementById('reportingInterval');

        tag_modelName.value         = simulation.Name;
        tag_timeHorizon.value       = simulation.TimeHorizon.toFixed(2);
        tag_reportingInterval.value = simulation.ReportingInterval.toFixed(2);
    }
    setTimeout(loadSimulationByName_, 0);
}

/* Clears all input fields.*/               
function clearInputs()
{
    var tag_progress          = document.getElementById('progress');
    var tag_output            = document.getElementById('output');
    var tag_plots             = document.getElementById('plots');
    var tag_modelName         = document.getElementById('modelName');
    var tag_timeHorizon       = document.getElementById('timeHorizon');
    var tag_reportingInterval = document.getElementById('reportingInterval');

    tag_progress.style.visibility = "hidden";
    tag_output.innerHTML          = '';
    tag_plots.innerHTML           = '';
    tag_modelName.value           = '';
    tag_timeHorizon.value         = 0;
    tag_reportingInterval.value   = 0;
}

/* Runs the currently loaded tutorial.
    This function is run using the setTimeout (delay = 0) to improve responsiveness. */               
function runTutorial()
{
    if(simulation == null)
        return;                

    // Validate inputs
    var tag_timeHorizon       = document.getElementById('timeHorizon');
    var tag_reportingInterval = document.getElementById('reportingInterval');

    var timeHorizon       = parseFloat(tag_timeHorizon.value);
    var reportingInterval = parseFloat(tag_reportingInterval.value);
    
    if(isNaN(timeHorizon))
    {
        alert('Invalid time horizon value specified');
        return;
    }
    if(isNaN(reportingInterval))
    {
        alert('Invalid reporting interval value specified');
        return;
    }
    
    var runTutorial_ = function() {    
        simulation.TimeHorizon       = timeHorizon;
        simulation.ReportingInterval = reportingInterval;
        disableInputs(true);
        simulateTutorial(simulation, currentTutorialName, 'plots', log, setProgress);
    };
    setTimeout(runTutorial_, 0);
}
