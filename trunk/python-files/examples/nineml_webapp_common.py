import os, sys, traceback

_css = """
    <style type="text/css">
        body 
        {
            padding:0;
            margin:20px;
            color:#333;
            background: white;
            font:14px sans-serif Tahoma;
            margin-left: auto; 
            margin-right: auto; 
            width: 700px;
        }
        form 
        {
            margin:0;
            padding:0;
        }
        fieldset 
        { 
            margin:1em 0;
            border:none;
            border-top:1px solid #ccc;
        }
        textarea
        {
             border: 1px solid LightGray;
        }
        input
        {
             border: 1px solid LightGray;
        }
        input:hover 
        {
            border: 1px solid Gray;
            background: white;
        }
        textarea:hover 
        {
            border: 1px solid Gray;
            background: white;
        }
        select
        {
            color: #202020; 
            background-color: White;
            width: 200px;
        }
        /*
        input:focus,textarea:focus 
        {
            background:#efefef;
            color:#000;
        }
        */
        
        form fieldset 
        {
            margin-bottom: 10px;
        }
        form legend 
        {
            padding: 0 2px;
            font-weight: bold;
        }
        form label 
        {
            display: inline-block;
            line-height: 1.8;
            vertical-align: top;
        }
        form fieldset ol 
        {
            margin: 0;
            padding: 0;
        }
        form fieldset li 
        {
            list-style: disc;
            padding: 0px;
            margin: 0;
        }
        form fieldset fieldset 
        {
            border: none;
            margin: 3px 0 0;
        }
        form fieldset fieldset legend 
        {
            padding: 0 0 5px;
            font-weight: normal;
        }
        form fieldset fieldset label 
        {
            display: block;
            width: auto;
        }
        form em 
        {
            font-weight: bold;
            font-style: normal;
            color: #f00;
        }
        form label 
        {
            width: 150px; /* Width of labels */
        }
        form fieldset fieldset label 
        {
            margin-left: 153px; /* Width plus 3 (html space) */
        }
    </style>
"""

def getSelectComponentPage():
    html = """
    <html>
        <head>
            CSS_STYLES
        </head>
        <body>
            <form action="nineml-webapp" method="post">
                <p>
                    NineML testable component name:<br/>
                    <input type="text" name="TestableComponent" value="hierachical_iaf_1coba"/>
                </p>
                <p>
                    Add a test to the repport (optional).
                </p>
                <p>
                    Initial values (in JSON format):<br/>
                    <textarea name="InitialValues" rows="40" cols="80">
{
    "timeHorizon" : 1.0,
    "reportingInterval" : 0.001,
    "parameters" : {
        "cobaExcit.q" : 3.0,
        "cobaExcit.tau" : 5.0,
        "cobaExcit.vrev" : 0.0,
        "iaf.cm" : 1,
        "iaf.gl" : 50,
        "iaf.taurefrac" : 0.008,
        "iaf.vreset" : -0.060,
        "iaf.vrest" : -0.060,
        "iaf.vthresh" : -0.040
    },
    "initial_conditions" : {
        "cobaExcit.g" : 0.0,
        "iaf.V" : -0.060,
        "iaf.tspike" : -1E99
    },
    "analog_ports_expressions" : {},
    "event_ports_expressions" : {
        "cobaExcit.spikeinput" : "0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90"
    },
    "active_regimes" : {
        "cobaExcit" : "cobadefaultregime",
        "iaf" : "subthresholdregime"
    },
    "variables_to_report" : {
        "cobaExcit.I" : true,
        "iaf.V" : true
    }
}
                </textarea>
                </p>
                <input type="hidden" name="__NINEML_WEBAPP_ACTION__" value="setupData"/>
                <br/>
                <input type="submit" value="Submit" />
            </form>
        </body>
        </html>
    """
    return html.replace('CSS_STYLES', _css)

def getSetupDataForm():
    html = """
    <form action="nineml-webapp" method="post">
        <h1>Test NineML component: {0}</h1>
        {1}
        <input type="hidden" name="__NINEML_WEBAPP_ID__" value="{2}"/>
        <input type="hidden" name="__NINEML_WEBAPP_ACTION__" value="runSimulation"/>
        <br/>
        
        <input type="submit" value="Submit" />
    </form>
    """
    return html

def createResultPage(content):
    html =  """
    <html>
        <head>
            {0}
        </head>
        <body>
            {1}
        </body>
    </html>
    """
    return html.format(_css, content)

def createSetupDataPage(content):
    html =  """
    <html>
        <head>
            {0}
        </head>
        <body>
            {1}
        </body>
    </html>
    """
    return html.format(_css, content)

def createErrorPage(error, trace_back, additional_data = ''):
    html =  """
    <html>
        <head>
            {0}
        </head>
        <body>
            <pre>Error occurred:\n  {1}</pre>
            <pre>{2}</pre>
            <pre>{3}</pre>
        </body>
    </html>
    """
    errorDescription = ''
    messages = traceback.format_tb(trace_back)
    for msg in messages:
        errorDescription += msg
        
    return html.format(_css, error, errorDescription, additional_data)
