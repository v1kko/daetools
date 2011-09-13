import os, sys, traceback

def getSelectComponentPage():
    html = """
    <html><body>
    <form action="nineml-webapp" method="post">
        <p>
            NineML testable component name:<br/>
            <input type="text" name="TestableComponent" value="hierachical_iaf_1coba"/>
        </p>
        <p>
            Initial values:<br/>
            <textarea name="InitialValues" rows="30" cols="50"></textarea>
        </p>
        <input type="hidden" name="__NINEML_WEBAPP_ACTION__" value="setupData"/>
        <br/>
        <input type="submit" value="Submit" />
    </form>
    </body></html>
    """
    return html

def getSetupDataForm():
    html = """
    <form action="nineml-webapp" method="post">
    {0}
    <input type="hidden" name="__NINEML_WEBAPP_ID__" value="{1}"/>
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
    <html><body style="margin-left: auto; margin-right: auto; width: 70%;" >
    {0}
    </body></html>
    """
    return html.format(content)

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
