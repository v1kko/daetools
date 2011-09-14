from pprint import pformat
import os, sys, math, json, traceback
from time import localtime, strftime, time
import urlparse
import cgitb
cgitb.enable()

___import_exception___ = None
___import_exception_traceback___ = None
try:
    sys.path.append("/home/ciroki/Data/daetools/trunk/python-files/examples")

    import nineml
    from nineml.abstraction_layer import readers
    from nineml.abstraction_layer.testing_utils import TestableComponent
    from nineml.abstraction_layer import ComponentClass
    from nineml_component_inspector import nineml_component_inspector
    from nineml_webapp_common import createErrorPage, getSetupDataForm, createSetupDataPage

except Exception, e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    ___import_exception___           = str(e)
    ___import_exception_traceback___ = exc_traceback

"""
Sample json input data:
{
    "timeHorizon" : 10.0,
    "reportingInterval" : 0.01,
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
"""

def application(environ, start_response):
    try:
        html = ''
        content = ''
        raw_arguments = ''
        parameters = {}
        initial_conditions = {}
        analog_ports_expressions = {}
        event_ports_expressions = {}
        active_regimes = {}
        variables_to_report = {}

        if not ___import_exception___:
            if environ['REQUEST_METHOD'] == 'POST':
                content_length = int(environ['CONTENT_LENGTH'])
                if content_length == 0:
                    raise RuntimeError('No input NineML component has been specified')

                raw_arguments = pformat(environ['wsgi.input'].read(content_length))
                raw_arguments = raw_arguments.strip(' \'')
                dictFormData  = urlparse.parse_qs(raw_arguments)

                if not dictFormData.has_key('TestableComponent'):
                    raise RuntimeError('No input NineML component has been specified')

                compName = dictFormData['TestableComponent'][0]
                nineml_component = TestableComponent(compName)()
                if not nineml_component:
                    raise RuntimeError('The specified component: {0} could not be loaded'.format(compName))
                
                if dictFormData.has_key('InitialValues'):
                    data = json.loads(dictFormData['InitialValues'][0])
                    if not isinstance(data, dict):
                        raise RuntimeError('InitialValues argument must be a dictionary in JSON')

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

                inspector = nineml_component_inspector()
                inspector.inspect(nineml_component, parameters               = parameters,
                                                    initial_conditions       = initial_conditions,
                                                    active_regimes           = active_regimes,
                                                    analog_ports_expressions = analog_ports_expressions,
                                                    event_ports_expressions  = event_ports_expressions,
                                                    variables_to_report      = variables_to_report)

                applicationID = compName + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
                if not environ.has_key('nineml_process_data'):
                    environ['nineml_process_data'] = {}
                environ['nineml_process_data'][applicationID] = inspector

                raise RuntimeError('nineml_process_data')
                formTemplate  = getSetupDataForm()
                content      += formTemplate.format(inspector.generateHTMLForm(), applicationID)
                html          = createSetupDataPage(content)

        else:
            html = 'Error occurred:\n{0}\n{1}'.format(___import_exception___, ___import_exception_traceback___)
            
    except Exception, e:
        content = 'Application environment:\n' + pformat(environ) + '\n\n'
        content += 'Form arguments:\n  {0}\n\n'.format(raw_arguments)
        content += 'Form input:\n'
        content += '  parameters:  {0}\n'.format(parameters)
        content += '  initial_conditions:  {0}\n'.format(initial_conditions)
        content += '  active_regimes:  {0}\n'.format(active_regimes)
        content += '  analog_ports_expressions:  {0}\n'.format(analog_ports_expressions)
        content += '  event_ports_expressions:  {0}\n'.format(event_ports_expressions)
        content += '  variables_to_report:  {0}\n'.format(variables_to_report)

        exc_type, exc_value, exc_traceback = sys.exc_info()
        html = createErrorPage(str(e), exc_traceback, content)

    output_len = len(html)
    start_response('200 OK', [('Content-type', 'text/html'),
                              ('Content-Length', str(output_len))])
    return [html]
    