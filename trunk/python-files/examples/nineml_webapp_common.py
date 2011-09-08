import os, sys

def getSelectComponentPage():
    html = """
    <html><body>
    <form action="nineml-webapp-setup-data" method="post">
        <p>
            NineML testable component name:<br/>
            <input type="text" name="TestableComponent" value="hierachical_iaf_1coba"/>
        </p>
        <p>
            Initial values:<br/>
            <textarea name="InitialValues" rows="30" cols="100"></textarea>
        </p>
        <input type="submit" value="Submit" />
    </form>
    </body></html>
    """
    return html

def getSetupDataForm():
    html = """
    <form action="nineml-webapp-run-simulation" method="post">
    {0}
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

def createErrorPage(content):
    html =  """
    <html><body>
    {0}
    </body></html>
    """
    return html.format(content)
