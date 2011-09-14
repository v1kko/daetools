from pprint import pformat
import os, sys, math, json, traceback
from time import localtime, strftime, time
import urlparse
import cgitb
cgitb.enable()

___import_exception___ = None
___import_exception_traceback___ = None
try:
    os.environ['HOME'] = "/tmp"
    sys.path.append("/home/ciroki/Data/daetools/trunk/python-files/examples")

    import nineml
    from nineml.abstraction_layer import readers
    from nineml.abstraction_layer.testing_utils import TestableComponent
    from nineml.abstraction_layer import ComponentClass

    from daetools.pyDAE import pyCore, pyActivity, pyDataReporting, pyIDAS, daeLogs
    from nineml_component_inspector import nineml_component_inspector
    from nineml_daetools_bridge import nineml_daetools_bridge
    from nineml_daetools_simulation import daeSimulationInputData, nineml_daetools_simulation, ninemlTesterDataReporter
    from nineml_webapp_common import createErrorPage, getSetupDataForm, createSetupDataPage, getSelectComponentPage, createResultPage

except Exception, e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    ___import_exception___           = str(e)
    ___import_exception_traceback___ = exc_traceback

class nineml_webapp:
    nineml_process_data = {}

    def __init__(self):
        pass

    def initial_page(self, environ, start_response):
        html = getSelectComponentPage()
        output_len = len(html)
        start_response('200 OK', [('Content-type', 'text/html'),
                                ('Content-Length', str(output_len))])
        return [html]
        
    def setup_data(self, dictFormData, environ, start_response):
        try:
            html = ''
            content = ''
            raw_arguments = ''
            timeHorizon = 0.0
            reportingInterval = 0.0
            parameters = {}
            initial_conditions = {}
            analog_ports_expressions = {}
            event_ports_expressions = {}
            active_regimes = {}
            variables_to_report = {}

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

                if data.has_key('timeHorizon'):
                    timeHorizon = float(data['timeHorizon'])
                if data.has_key('reportingInterval'):
                    reportingInterval = float(data['reportingInterval'])

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
            inspector.inspect(nineml_component, timeHorizon              = timeHorizon,
                                                reportingInterval        = reportingInterval,
                                                parameters               = parameters,
                                                initial_conditions       = initial_conditions,
                                                active_regimes           = active_regimes,
                                                analog_ports_expressions = analog_ports_expressions,
                                                event_ports_expressions  = event_ports_expressions,
                                                variables_to_report      = variables_to_report)

            applicationID = compName + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
            nineml_webapp.nineml_process_data[applicationID] = inspector

            formTemplate  = getSetupDataForm()
            content      += formTemplate.format(inspector.generateHTMLForm(), applicationID)
            html          = createSetupDataPage(content)
        
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

    def run_simulation(self, dictFormData, environ, start_response):
        try:
            html = ''
            raw_arguments = ''
            content = ''
            log = None
            success = False
            timeHorizon = 0.0
            reportingInterval = 0.0
            parameters = {}
            initial_conditions = {}
            analog_ports_expressions = {}
            event_ports_expressions = {}
            active_regimes = {}
            variables_to_report = {}

            if not dictFormData.has_key('__NINEML_WEBAPP_ID__'):
                raise RuntimeError('No application ID has been specified')

            applicationID = dictFormData['__NINEML_WEBAPP_ID__'][0]
            if not nineml_webapp.nineml_process_data.has_key(applicationID):
                return self.initial_page(environ, start_response)

            inspector = nineml_webapp.nineml_process_data[applicationID]
            if not inspector:
                raise RuntimeError('Invalid inspector object')

            # Remove inspector from the map since I finished with it
            del nineml_webapp.nineml_process_data[applicationID]
            
            if dictFormData.has_key('timeHorizon'):
                timeHorizon = float(dictFormData['timeHorizon'][0])
            if dictFormData.has_key('reportingInterval'):
                reportingInterval = float(dictFormData['reportingInterval'][0])

            for key, values in dictFormData.items():
                names = key.split('.')
                if len(names) > 0:
                    canonicalName = '.'.join(names[1:])

                    if names[0] == nineml_component_inspector.categoryParameters:
                        parameters[canonicalName] = float(values[0])

                    elif names[0] == nineml_component_inspector.categoryInitialConditions:
                        initial_conditions[canonicalName] = float(values[0])

                    elif names[0] == nineml_component_inspector.categoryActiveStates:
                        active_regimes[canonicalName] = values[0]

                    elif names[0] == nineml_component_inspector.categoryAnalogPortsExpressions:
                        analog_ports_expressions[canonicalName] = values[0]

                    elif names[0] == nineml_component_inspector.categoryEventPortsExpressions:
                        event_ports_expressions[canonicalName] = values[0]

                    elif names[0] == nineml_component_inspector.categoryVariablesToReport:
                        if values[0] == 'on':
                            variables_to_report[canonicalName] = True

            nineml_component = inspector.ninemlComponent
            if not nineml_component:
                raise RuntimeError('Cannot load NineML component')

            simulation_data = daeSimulationInputData()
            simulation_data.timeHorizon              = timeHorizon
            simulation_data.reportingInterval        = reportingInterval
            simulation_data.parameters               = parameters
            simulation_data.initial_conditions       = initial_conditions
            simulation_data.analog_ports_expressions = analog_ports_expressions
            simulation_data.event_ports_expressions  = event_ports_expressions
            simulation_data.active_regimes           = active_regimes
            simulation_data.variables_to_report      = variables_to_report

            # Create Log, DAESolver, DataReporter and Simulation object
            log          = daeLogs.daeBaseLog()
            daesolver    = pyIDAS.daeIDAS()
            datareporter = ninemlTesterDataReporter()
            model        = nineml_daetools_bridge(nineml_component.name, nineml_component)
            simulation   = nineml_daetools_simulation(model, timeHorizon              = timeHorizon,
                                                             reportingInterval        = reportingInterval,
                                                             parameters               = simulation_data.parameters,
                                                             initial_conditions       = simulation_data.initial_conditions,
                                                             active_regimes           = simulation_data.active_regimes,
                                                             analog_ports_expressions = simulation_data.analog_ports_expressions,
                                                             event_ports_expressions  = simulation_data.event_ports_expressions,
                                                             variables_to_report      = simulation_data.variables_to_report)

            # Set the time horizon and the reporting interval
            simulation.ReportingInterval = simulation_data.reportingInterval
            simulation.TimeHorizon       = simulation_data.timeHorizon

            # Connect data reporter
            simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
            if(datareporter.Connect("", simName) == False):
                raise RuntimeError('Cannot connect a TCP/IP datareporter; did you forget to start daePlotter?')

            # Initialize the simulation
            simulation.Initialize(daesolver, datareporter, log)

            # Save the model reports for all models
            #simulation.m.SaveModelReport(simulation.m.Name + ".xml")
            #simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

            # Solve at time=0 (initialization)
            simulation.SolveInitial()

            # Run
            simulation.Run()

            log_output = '<pre>{0}</pre>'.format(log.JoinMessages('\n'))
            content += log_output
            simulation.Finalize()

            html = createResultPage(content)
            success = True

        except Exception, e:
            content += 'Application environment:\n' + pformat(environ) + '\n\n'
            content += 'Form arguments:\n  {0}\n\n'.format(raw_arguments)
            content += 'Simulation input:\n'
            content += '  parameters:  {0}\n'.format(parameters)
            content += '  initial_conditions:  {0}\n'.format(initial_conditions)
            content += '  active_regimes:  {0}\n'.format(active_regimes)
            content += '  analog_ports_expressions:  {0}\n'.format(analog_ports_expressions)
            content += '  event_ports_expressions:  {0}\n'.format(event_ports_expressions)
            content += '  variables_to_report:  {0}\n'.format(variables_to_report)
            if log:
                log_output = 'Log output:\n{0}'.format(log.JoinMessages('\n'))
                content += '\n' + log_output

            exc_type, exc_value, exc_traceback = sys.exc_info()
            html = createErrorPage(str(e), exc_traceback, content)

        if success:
            boundary = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            pdf = open("/home/ciroki/Data/daetools/trunk/python-files/examples/coba_iaf.pdf", "rb").read()
            part1 = '--{0}\r\nContent-Type: text/html\r\n\r\n{1}\n'.format(boundary, html)
            part2 = '--{0}\r\nContent-Disposition: attachment; filename=Dummy-model-report.pdf\r\n\r\n{1}\n'.format(boundary, pdf)
            output = part1 + part2
            output_len = len(output)
            start_response('200 OK', [('Content-type', 'multipart/mixed; boundary={0}'.format(boundary)),
                                    ('Content-Length', str(output_len))])
            return [output]

        else:
            output_len = len(html)
            start_response('200 OK', [('Content-type', 'text/html'),
                                    ('Content-Length', str(output_len))])
            return [html]
        
    def __call__(self, environ, start_response):
        try:
            html = ''
            if not ___import_exception___:
                if environ['REQUEST_METHOD'] == 'GET':
                    return self.initial_page(environ, start_response)

                else:
                    content_length = int(environ['CONTENT_LENGTH'])
                    raw_arguments = pformat(environ['wsgi.input'].read(content_length))
                    raw_arguments = raw_arguments.strip(' \'')
                    dictFormData  = urlparse.parse_qs(raw_arguments)

                    if not dictFormData.has_key('__NINEML_WEBAPP_ACTION__'):
                        raise RuntimeError('Phase argument must be specified')

                    phase = dictFormData['__NINEML_WEBAPP_ACTION__'][0]
                    if phase == 'setupData':
                        return self.setup_data(dictFormData, environ, start_response)

                    elif phase == 'runSimulation':
                        return self.run_simulation(dictFormData, environ, start_response)

                    else:
                        raise RuntimeError('Invalid phase argument specified; ' + str(dictFormData))
            else:
                html = 'Error occurred:\n{0}\n{1}'.format(___import_exception___, ___import_exception_traceback___)

        except Exception, e:
            content = 'Application environment:\n' + pformat(environ) + '\n\n'
            exc_type, exc_value, exc_traceback = sys.exc_info()
            html = createErrorPage(e, exc_traceback, content)

        output_len = len(html)
        start_response('200 OK', [('Content-type', 'text/html'),
                                ('Content-Length', str(output_len))])
        return [html]

application = nineml_webapp()