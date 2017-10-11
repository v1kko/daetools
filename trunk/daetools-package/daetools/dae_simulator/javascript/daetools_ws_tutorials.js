/***********************************************************************************
                           daetools_ws_tutorials.js
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
var x_tickformat = '.1f';
var y_tickformat = '.1f';
var z_tickformat = '.1f';

function plot_2D(data, x_label, y_label, plot_title, htmlTagID)
{   
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

function plot_3D(surfaceData, x_label, y_label, z_label, plot_title, htmlTagID)
{   
    var layout = {
        showlegend: true,
        autosize: true,
        title: plot_title,
        scene: {
            xaxis: {
                tickmode: 'auto',
                nticks: 10,
                tickformat: x_tickformat,
                title: x_label
            },
            yaxis: {
                tickmode: 'auto',
                nticks: 10,
                tickformat: y_tickformat,
                title: y_label
            },
            zaxis: {
                tickformat: z_tickformat,
                title: z_label
            }
        },
    };
    
    Plotly.newPlot(htmlTagID, [surfaceData], layout);
    
    window.addEventListener('resize', function() { 
        var d3 = Plotly.d3;
        var gd3 = d3.select("div[id='" + htmlTagID + "']");
        var gd = gd3.node();
        Plotly.Plots.resize(gd);
    });
}

function loadTutorial(tutorialName, logFunction)
{   
    var simulation = new daeSimulation(webService);
    simulation.LoadTutorial(tutorialName);
    
    return simulation;
}

function loadSimulationByName(simulationName, args, logFunction)
{   
    var simulation = new daeSimulation(webService);
    simulation.LoadSimulationByName(simulationName, JSON.stringify(args));
    
    return simulation;
}
    
function simulateTutorial(simulation, tutorialName, plotTagID, logFunction, setProgressFunction)
{    
    /* Run simulation in a recursive way using the function setTimeout with the delay set to 0 to improve responsiveness.
       The function doStep will call itself again if the current time is lower than the time horizon.
       After every call to setTimeout the javascript runtime has a chance to do other tasks (including GUI).
       Once the time horizon has been reached the plots are prepared, the simulation.Finalize called
       and the function exits.*/
    if(simulation == null)
        return;
    
    simulation.SolveInitial();    
    
    var reportingInterval = simulation.ReportingInterval;
    var timeHorizon       = simulation.TimeHorizon;
    var currentTime       = simulation.CurrentTime;
    
    var doStep = function() {
        // Get the time step (based on the TimeHorizon and the ReportingInterval).
        // Do not allow to get past the TimeHorizon.
        var t = currentTime + reportingInterval;
        if(t > timeHorizon)
            t = timeHorizon;

        // If a discontinuity is found, loop until the end of the integration period.
        // The data will be reported around discontinuities!
        while(t > currentTime)
        {
            if(logFunction)
                logFunction('Integrating from ' + currentTime.toFixed(2) + ' to ' + t.toFixed(2) + ' ...');
            currentTime = simulation.IntegrateUntilTime(t, true, true);
        }
        
        // After the integration period, report the data. 
        simulation.ReportData()
        
        // Set the simulation progress.
        var newProgress = Math.ceil(100.0 * currentTime / timeHorizon)
        if(setProgressFunction)
            setProgressFunction(newProgress);
        
        if(currentTime < timeHorizon)
        {
            setTimeout(doStep, 0);
        }
        else
        {
            if(logFunction)
            {
                logFunction('The simulation has finished successfuly!');
                logFunction('Preparing the plots...');
            }

            // Make some plots (if specified).
            plotTutorial(simulation, tutorialName, plotTagID);
           
            // Clean up.
            simulation.Finalize();
        }        
    };
    
    doStep();
}

function makePlotlyData_2D(variable) {
    return {
        x: variable.Times,
        y: variable.Values,
        mode: 'lines+markers',
        name: variable.ShortName
    };
};

function makePlotlyData_3D(variable) {
    return {
        x: variable.Domains[0],
        y: variable.Domains[1],
        z: variable.Values[variable.Values.length-1],
        type: 'surface',
        name: variable.ShortName
        //colorscale: [[0.00, 'rgb(0,0,255)'], 
        //             [1.00, 'rgb(255,0,0)']]
    };
};

function plotTutorial(simulation, tutorialName, plotTagID)
{
    x_tickformat = '.1f';
    y_tickformat = '.1f';
   
    if(tutorialName == 'tutorial1')
    {
        var T  = simulation.DataReporter.Value('tutorial1.T');
        var T_surface =  makePlotlyData_3D(T);
        
        var tag_plots = document.getElementById(plotTagID);
        var tag_Tdiv  = document.createElement('div');
        tag_Tdiv.id = 'Tplots';
        tag_plots.innerHTML = '';
        tag_plots.appendChild(tag_Tdiv);
        
        x_tickformat = '.2f';
        y_tickformat = '.2f';
        plot_3D(T_surface, 'x', 'y', 'T [' + T.Units + ']', 'Tutorial 1 Temperature Plot (at t = ' + simulation.CurrentTime.toFixed(1) + 's)', tag_Tdiv.id);
    }
    else if(tutorialName == 'tutorial4')
    {
        var T   = simulation.DataReporter.Value('tutorial4.T');
        var Qin = simulation.DataReporter.Value('tutorial4.Q_in');
        
        var T_line   = makePlotlyData_2D(T);
        var Qin_line = makePlotlyData_2D(Qin);
        
        var tag_plots = document.getElementById(plotTagID);
        var tag_Qdiv  = document.createElement('div');
        var tag_Tdiv  = document.createElement('div');
        tag_Qdiv.id = 'Qplots';
        tag_Tdiv.id = 'Tplots';
        tag_plots.innerHTML = '';
        tag_plots.appendChild(tag_Qdiv);
        tag_plots.appendChild(tag_Tdiv);        
        plot_2D([T_line],   't [s]', 'T [' + T.Units + ']',     'Tutorial 4 Temperature Plot', tag_Tdiv.id);
        plot_2D([Qin_line], 't [s]', 'Qin [' + Qin.Units + ']', 'Tutorial 4 Input Power Plot', tag_Qdiv.id);
    }
    else if(tutorialName == 'tutorial5')
    {
        var T   = simulation.DataReporter.Value('tutorial5.T');
        var Qin = simulation.DataReporter.Value('tutorial5.Q_in');
        
        var T_line   = makePlotlyData_2D(T);
        var Qin_line = makePlotlyData_2D(Qin);
        
        var tag_plots = document.getElementById(plotTagID);
        var tag_Qdiv  = document.createElement('div');
        var tag_Tdiv  = document.createElement('div');
        tag_Qdiv.id = 'Qplots';
        tag_Tdiv.id = 'Tplots';
        tag_plots.innerHTML = '';
        tag_plots.appendChild(tag_Qdiv);
        tag_plots.appendChild(tag_Tdiv);
        
        plot_2D([T_line],   't [s]', 'T [' + T.Units + ']',     'Tutorial 5 Temperature Plot', tag_Tdiv.id);
        plot_2D([Qin_line], 't [s]', 'Qin [' + Qin.Units + ']', 'Tutorial 5 Input Power Plot', tag_Qdiv.id);
    }
    else if(tutorialName == 'tutorial14')
    {
        var T    = simulation.DataReporter.Value('tutorial14.T');
        var Heat = simulation.DataReporter.Value('tutorial14.Heat');
        
        var T_line    = makePlotlyData_2D(T);
        var Heat_line = makePlotlyData_2D(Heat);
        
        var tag_plots = document.getElementById(plotTagID);
        var tag_Tdiv    = document.createElement('div');
        var tag_Heatdiv = document.createElement('div');
        tag_Tdiv.id    = 'Tplots';
        tag_Heatdiv.id = 'Heatplots';
        tag_plots.innerHTML = '';
        tag_plots.appendChild(tag_Heatdiv);
        tag_plots.appendChild(tag_Tdiv);
        
        plot_2D([T_line],    't [s]', 'T [' + T.Units + ']',       'Tutorial 14 Temperature Plot', tag_Tdiv.id);
        plot_2D([Heat_line], 't [s]', 'Heat [' + Heat.Units + ']', 'Tutorial 14 Heat Plot',        tag_Heatdiv.id);
    }
    else if(tutorialName == 'tutorial_che_1')
    {
        var T  = simulation.DataReporter.Value('tutorial_che_1.T');
        var Tk = simulation.DataReporter.Value('tutorial_che_1.T_k');
        var Ca = simulation.DataReporter.Value('tutorial_che_1.Ca');
        var Cb = simulation.DataReporter.Value('tutorial_che_1.Cb');
        var Cc = simulation.DataReporter.Value('tutorial_che_1.Cc');
        var Cd = simulation.DataReporter.Value('tutorial_che_1.Cd');
        
        var T_line =  makePlotlyData_2D(T);
        var Tk_line = makePlotlyData_2D(Tk);
        var Ca_line = makePlotlyData_2D(Ca);
        var Cb_line = makePlotlyData_2D(Cb);
        var Cc_line = makePlotlyData_2D(Cc);
        var Cd_line = makePlotlyData_2D(Cd);

        var tag_plots = document.getElementById(plotTagID);
        var tag_Cdiv  = document.createElement('div');
        var tag_Tdiv  = document.createElement('div');
        tag_Cdiv.id = 'Cplots';
        tag_Tdiv.id = 'Tplots';
        tag_plots.innerHTML = '';
        tag_plots.appendChild(tag_Cdiv);
        tag_plots.appendChild(tag_Tdiv);
        
        plot_2D([Ca_line, Cb_line, Cc_line, Cd_line], 't [s]', 'Concentration [' + Ca.Units + ']', 'Tutorial Che. 1 Concentration Plots', tag_Cdiv.id);
        plot_2D([T_line, Tk_line],                    't [s]', 'Temperature [' + T.Units + ']',    'Tutorial Che. 1 Temperature Plots',   tag_Tdiv.id);
    }
    else if(tutorialName == 'tutorial_che_9')
    {
        var u1 = simulation.DataReporter.Value('tutorial_che_9.u1');
        var u3 = simulation.DataReporter.Value('tutorial_che_9.u3');
        var u4 = simulation.DataReporter.Value('tutorial_che_9.u4');
        var u6 = simulation.DataReporter.Value('tutorial_che_9.u6');
        var u8 = simulation.DataReporter.Value('tutorial_che_9.u8');
        
        var u1_line = makePlotlyData_2D(u1);
        var u3_line = makePlotlyData_2D(u3);
        var u4_line = makePlotlyData_2D(u4);
        var u6_line = makePlotlyData_2D(u6);
        var u8_line = makePlotlyData_2D(u8);

        var tag_plots = document.getElementById(plotTagID);
        var tag_u1div = document.createElement('div');
        var tag_u2div = document.createElement('div');
        tag_u1div.id = 'u1plots';
        tag_u2div.id = 'u2plots';
        tag_plots.innerHTML = '';
        tag_plots.appendChild(tag_u1div);
        tag_plots.appendChild(tag_u2div);
        
        y_tickformat = '.3f';        
        plot_2D([u1_line, u3_line, u4_line], 't [s]', 'Concentration [' + u1.Units + ']', 'Tutorial Che. 9 Concentration Plots (a)', tag_u1div.id);
        plot_2D([u6_line, u8_line],          't [s]', 'Concentration [' + u6.Units + ']', 'Tutorial Che. 9 Concentration Plots (b)', tag_u2div.id);
    }
    else
    {
        var values = simulation.DataReporter.AllValues();
        var tag_plots = document.getElementById(plotTagID);
        var tag_pre  = document.createElement('pre');
        var tag_code = document.createElement('code');
        tag_pre.setAttribute('style', 'max-height:300px; overflow: auto; overflow-y:visible;');
        tag_pre.setAttribute('class', 'w3-code w3-border w3-small');
        tag_pre.appendChild(tag_code);
        tag_plots.appendChild(tag_pre);
        tag_code.textContent = 'Plots not available, displaying the raw data: ' + JSON.stringify(values, null, 4);
    }
}
