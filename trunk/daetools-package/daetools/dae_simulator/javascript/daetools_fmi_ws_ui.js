/***********************************************************************************
                           daetools_fmi_ws_ui.js
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
var webService = create_daetools_fmi_ws();
var fmi_simulation = null;

document.addEventListener("DOMContentLoaded", function(event) {
    clearInputs();
    disableInputs(true);
});

window.addEventListener("beforeunload", function(event) { 
    /* If not null, finalize the simulation before unloading (it frees the resources on the server).*/
    if(fmi_simulation !== null)
    {
        if(fmi_simulation.simulationID !== null)
        {
            fmi_simulation.fmi2Terminate();
            fmi_simulation.fmi2FreeInstance();
        }
        fmi_simulation = null;
    }
});

function uuidv4() 
{
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c => (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16));
}

function log(message) 
{
    var tag_output = document.getElementById('output');
    content = tag_output.innerHTML;
    tag_output.innerHTML = content + '\n' + message;
    tag_output.scrollTop = tag_output.scrollHeight;
}

function loadDAEToolsFMU()
{
    if(fmi_simulation !== null)
    {
        if(fmi_simulation.simulationID !== null)
        {
            fmi_simulation.fmi2Terminate();
            fmi_simulation.fmi2FreeInstance();
        }
        fmi_simulation = null;
    }

    var tag_resourceDir = document.getElementById('resourceDir');
    var tag_modelName   = document.getElementById('modelName');
    var tag_startTime   = document.getElementById('startTime');
    var tag_stopTime    = document.getElementById('stopTime');
    var tag_step        = document.getElementById('step');
    var tag_tolerance   = document.getElementById('tolerance');
    
    // Clear the data
    clearInputs();

    var instanceName     = 'fmu_test';
    var guid             = uuidv4();
    var resourceLocation = tag_resourceDir.value;
    
    if(resourceLocation == '')
        return;
    
    fmi_simulation = new daeFMI2Simulation(webService);
    fmi_simulation.fmi2Instantiate(instanceName, guid, resourceLocation);
    
    tag_modelName.value = fmi_simulation.modelName;
    tag_startTime.value = fmi_simulation.startTime;
    tag_stopTime.value  = fmi_simulation.stopTime;
    tag_step.value      = fmi_simulation.step;
    tag_tolerance.value = fmi_simulation.tolerance;
    
    disableInputs(false);
}

function setProgress(newProgress)
{
    var tag_progress = document.getElementById('progress');
    tag_progress.style.visibility = "visible";
    tag_progress.style.width = Math.floor(newProgress) + '%';
    tag_progress.innerHTML = newProgress.toFixed(1) + '%';
}
        
function plot_2D(data, x_label, y_label, plot_title, htmlTagID)
{   
    var x_tickformat = '.1f';
    var y_tickformat = '.2f';
    var layout = {
        showlegend: true,
        legend: {x: 1.0, 
                 y: 1.1,
                 xanchor: 'right',
                 orientation: 'h'
        },
        autosize: true,
        title: plot_title,
        xaxis: {
            zeroline: false,
            showline: true,
            showgrid: true,
            showticklabels: true,
            tickformat: x_tickformat,
            title: x_label
        },
        yaxis: {
            zeroline: false,
            showline: true,
            showgrid: true,
            showticklabels: true,
            tickformat: y_tickformat,
            title: y_label
        }
    };

    Plotly.newPlot(htmlTagID, data, layout);
    
    window.addEventListener('resize', function() { 
        var d3 = Plotly.d3;
        var gd3 = d3.select("div[id='" + htmlTagID + "']");
        var gd = gd3.node();
        Plotly.Plots.resize(gd);
    });
}

function clearInputs()
{
    var tag_plots       = document.getElementById('plots');
    var tag_progress    = document.getElementById('progress');
    var tag_output      = document.getElementById('output');
    var tag_modelName   = document.getElementById('modelName');
    var tag_startTime   = document.getElementById('startTime');
    var tag_stopTime    = document.getElementById('stopTime');
    var tag_step        = document.getElementById('step');
    var tag_tolerance   = document.getElementById('tolerance');
    
    // Clear the data
    tag_progress.style.visibility = "hidden";
    tag_plots.innerHTML = '';
    tag_output.innerHTML= '';
    tag_modelName.value = '';
    tag_startTime.value = 0.0;
    tag_stopTime.value  = 0.0;
    tag_step.value      = 0.0;
    tag_tolerance.value = 0.0;
}

function disableInputs(disabled)
{
    var tag_buttonRun = document.getElementById('runButton');
    tag_buttonRun.disabled = disabled;
}

function plotVariables(references, values, names, units, times, plotsTagID)
{
    // Plot up to 20 plots.
    var no_plots = Math.min(20, references.length);
    
    // Clear the previous plots (if any).
    var tag_plots = document.getElementById(plotsTagID);
    tag_plots.innerHTML = '';
    
    for(var i = 0; i < no_plots; i++)
    {
        var name = names[i];
        var unit = units[i];
        var y = [];
        for(var t = 0; t < times.length; t++)
            y.push(values[t][i]);
        var data = { x: times,
                        y: y,
                        mode: 'lines+markers',
                        name: name};
        var tag_div  = document.createElement('div');
        tag_div.id = 'plot-' + i;
        tag_plots.appendChild(tag_div);
        plot_2D([data], 'time [s]', name + ' [' + unit + ']', 'Plot ' + (i+1) + ': ' + name, tag_div.id);
    }
}

function runSimulation(fmi_simulation, logFunction, plotFunction)
{   
    if(fmi_simulation == null)
        return;

    var tag_output = document.getElementById('output');
    tag_output.innerHTML = '';
    
    disableInputs(true);
    
    var modelName = fmi_simulation.modelName;
    var startTime = fmi_simulation.startTime;
    var step      = fmi_simulation.step;
    var stopTime  = fmi_simulation.stopTime;
    var tolerance = fmi_simulation.tolerance;
    
    simulateFMU(fmi_simulation, step, stopTime, tolerance, logFunction, plotFunction);
}
