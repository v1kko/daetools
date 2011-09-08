from pprint import pformat
import os, sys, math, json, traceback
from time import localtime, strftime, time
import urlparse
import cgitb
cgitb.enable()

___import_exception___ = None
try:
    sys.path.append("/home/ciroki/Data/daetools/trunk/python-files/examples")

    import nineml
    from nineml.abstraction_layer import readers
    from nineml.abstraction_layer.testing_utils import TestableComponent
    from nineml.abstraction_layer import ComponentClass
    from nineml_tester_gui import nineml_tester_htmlGUI
    from nineml_webapp_common import createErrorPage, getSetupDataForm, createSetupDataPage

except Exception, e:
    ___import_exception___ = str(e)

"""
Sample json input data:
{
    "parameters" : {
        "cobaExcit.q" : 3.0,
        "cobaExcit.tau" : 5.0,
        "cobaExcit.vrev" : 0.0,
        "iaf.cm" : 1,
        "iaf.gl" : 50,
        "iaf.taurefrac" : 0.008,
        "iaf.vreset" : -60,
        "iaf.vrest" : -60,
        "iaf.vthresh" : -40
    },
    "initial_conditions" : {
        "cobaExcit.g" : 0.0,
        "iaf.V" : -60,
        "iaf.tspike" : -1E99
    },
    "analog_ports_expressions" : {
        "cobaExcit.V" : "1.2 * e",
        "iaf.ISyn" : "5 * pi"
    },
    "event_ports_expressions" : {
    },
    "active_regimes" : {
        "cobaExcit" : "cobadefaultregime",
        "iaf" : "subthresholdregime"
    },
    "variables_to_report" : {
        "cobaExcit.g" : true,
        "iaf.tspike" : true
    }
}
"""

def application(environ, start_response):
    html = ''
    try:
        if not ___import_exception___:
            if environ['REQUEST_METHOD'] == 'POST':
                content = ''
                content += '<pre>' + pformat(environ) + '</pre>'
                content_length = int(environ['CONTENT_LENGTH'])
                if content_length == 0:
                    raise RuntimeError('No input NineML component has been specified')

                arguments = pformat(environ['wsgi.input'].read(content_length))
                arguments = arguments.strip(' \'')
                content += '<p>arguments = {0}</p>'.format(arguments)
                dictFormData = urlparse.parse_qs(arguments)
                content += str(dictFormData)
                
                if not dictFormData.has_key('TestableComponent'):
                    raise RuntimeError('No input NineML component has been specified')
                compName = dictFormData['TestableComponent'][0]

                nineml_component = TestableComponent(compName)()
                if not nineml_component:
                    raise RuntimeError('Cannot load the specified NineML component')

                parameters = {}
                initial_conditions = {}
                analog_ports_expressions = {}
                event_ports_expressions = {}
                active_regimes = {}
                variables_to_report = {}

                if dictFormData.has_key('InitialValues'):
                    data = json.loads(dictFormData['InitialValues'][0])
                    if isinstance(data, dict):
                        if data.has_key('parameters'):
                            temp = data['parameters']
                            if isinstance(temp, dict):
                                parameters = temp
                            else:
                                raise RuntimeError('parameters argument must be a dictionary')

                        if data.has_key('initial_conditions'):
                            temp = data['initial_conditions']
                            if isinstance(temp, dict):
                                initial_conditions = temp
                            else:
                                raise RuntimeError('initial_conditions argument must be a dictionary')

                        if data.has_key('analog_ports_expressions'):
                            temp = data['analog_ports_expressions']
                            if isinstance(temp, dict):
                                analog_ports_expressions = temp
                            else:
                                raise RuntimeError('analog_ports_expressions argument must be a dictionary')

                        if data.has_key('event_ports_expressions'):
                            temp = data['event_ports_expressions']
                            if isinstance(temp, dict):
                                event_ports_expressions = temp
                            else:
                                raise RuntimeError('event_ports_expressions argument must be a dictionary')

                        if data.has_key('active_regimes'):
                            temp = data['active_regimes']
                            if isinstance(temp, dict):
                                active_regimes = temp
                            else:
                                raise RuntimeError('active_regimes argument must be a dictionary')

                        if data.has_key('variables_to_report'):
                            temp = data['variables_to_report']
                            if isinstance(temp, dict):
                                variables_to_report = temp
                            else:
                                raise RuntimeError('variables_to_report argument must be a dictionary')

                content += 'Parsed arguments:<br/><pre>'
                content += 'parameters:<br/>  {0}<br/>'.format(parameters)
                content += 'initial_conditions:<br/>  {0}<br/>'.format(initial_conditions)
                content += 'active_regimes:<br/>  {0}<br/>'.format(active_regimes)
                content += 'analog_ports_expressions:<br/>  {0}<br/>'.format(analog_ports_expressions)
                content += 'event_ports_expressions:<br/>  {0}<br/>'.format(event_ports_expressions)
                content += 'variables_to_report:<br/>  {0}<br/>'.format(variables_to_report)
                content += '</pre>'

                s = nineml_tester_htmlGUI(nineml_component, parameters               = parameters,
                                                            initial_conditions       = initial_conditions,
                                                            active_regimes           = active_regimes,
                                                            analog_ports_expressions = analog_ports_expressions,
                                                            event_ports_expressions  = event_ports_expressions,
                                                            variables_to_report      = variables_to_report)
                formTemplate = getSetupDataForm()
                content     += formTemplate.format(s.generateHTMLForm())
                html         = createSetupDataPage(content)
        else:
            html = createErrorPage('<p>Error occurred: {0}</p>'.format(___import_exception___))

    except Exception, e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        html = createErrorPage('<p>Error occurred: {0}<br/><pre>{1}</pre></p>'.format(str(e), repr(traceback.format_tb(exc_traceback))))

    output = []
    output.append(html)
    output_len = sum(len(line) for line in output)
    start_response('200 OK', [('Content-type', 'text/html'),
                              ('Content-Length', str(output_len))])
    return output