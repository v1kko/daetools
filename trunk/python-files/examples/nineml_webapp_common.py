import os, sys, traceback

_css = """
    <style type="text/css">
        form{margin:0;padding:0;}
        fieldset{margin:1em 0;border:none;border-top:1px solid #ccc;}
        textarea{border-top:1px solid #555;border-left:1px solid #555;border-bottom:1px solid #ccc;border-right:1px solid #ccc;padding:1px;color:#333;}
        input:focus,textarea:focus{background:#efefef;color:#000;}
        body{padding:0;margin:20px;color:#333;background:#fff;font:14px arial,verdana,sans-serif;}

        form fieldset {
            margin-bottom: 10px;
        }
        form legend {
            padding: 0 2px;
            font-weight: bold;
        }
        form label {
            display: inline-block;
            line-height: 1.8;
            vertical-align: top;
        }
        form fieldset ol {
            margin: 0;
            padding: 0;
        }
        form fieldset li {
            list-style: disc;
            padding: 0px;
            margin: 0;
        }
        form fieldset fieldset {
            border: none;
            margin: 3px 0 0;
        }
        form fieldset fieldset legend {
            padding: 0 0 5px;
            font-weight: normal;
        }
        form fieldset fieldset label {
            display: block;
            width: auto;
        }
        form em {
            font-weight: bold;
            font-style: normal;
            color: #f00;
        }
        form label {
            width: 120px; /* Width of labels */
        }
        form fieldset fieldset label {
            margin-left: 123px; /* Width plus 3 (html space) */
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
                    <textarea name="InitialValues" rows="30" cols="50">{
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
            "event_ports_expressions" : {},
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
    <html><body style="margin-left: auto; margin-right: auto; width: 70%;" >
    {0}
    </body></html>
    """
    return html.format(content)

def createSetupDataPage(content):
    html =  """
    <html>
        <head>
            {0}
        </head>
        <body style="margin-left: auto; margin-right: auto; width: 600px;" >
            {1}
        </body>
    </html>
    """
    return html.format(_css, content)

def createErrorPage(error, trace_back, additional_data = ''):
    html =  """
    <html><body>
    <pre>Error occurred:\n  {0}</pre>
    <pre>{1}</pre>
    <pre>{2}</pre>
    </body></html>
    """
    errorDescription = ''
    messages = traceback.format_tb(trace_back)
    for msg in messages:
        errorDescription += msg
        
    return html.format(error, errorDescription, additional_data)
