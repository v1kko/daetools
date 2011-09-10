from pprint import pformat
import os, sys, math, traceback
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
    from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
    from nineml.abstraction_layer import ComponentClass
    from nineml_tester_gui import nineml_tester_htmlGUI

    from daetools.pyDAE import pyCore, pyActivity, pyDataReporting, pyIDAS, daeLogs
    from nineml_daetools_bridge import nineml_daetools_bridge
    from nineml_daetools_simulation import daeSimulationInputData, nineml_daetools_simulation, ninemlTesterDataReporter
    from nineml_webapp_common import createResultPage, createErrorPage

except Exception, e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    ___import_exception___           = str(e)
    ___import_exception_traceback___ = exc_traceback

def application(environ, start_response):
    try:
        html = ''
        raw_arguments = ''
        content = ''
        log = None
        success = False
        parameters = {}
        initial_conditions = {}
        analog_ports_expressions = {}
        event_ports_expressions = {}
        active_regimes = {}
        variables_to_report = {}

        if not ___import_exception___:
            if environ['REQUEST_METHOD'] == 'POST':
                content_length = int(environ['CONTENT_LENGTH'])

                if content_length > 0:
                    raw_arguments = pformat(environ['wsgi.input'].read(content_length))
                    raw_arguments = raw_arguments.strip(' \'')
                    dictFormData  = urlparse.parse_qs(raw_arguments)

                    for key, values in dictFormData.items():
                        names = key.split('.')
                        if len(names) > 0:
                            canonicalName = '.'.join(names[1:])

                            if names[0] == nineml_tester_htmlGUI.categoryParameters:
                                parameters[canonicalName] = float(values[0])

                            elif names[0] == nineml_tester_htmlGUI.categoryInitialConditions:
                                initial_conditions[canonicalName] = float(values[0])

                            elif names[0] == nineml_tester_htmlGUI.categoryActiveStates:
                                active_regimes[canonicalName] = values[0]

                            elif names[0] == nineml_tester_htmlGUI.categoryAnalogPortsExpressions:
                                analog_ports_expressions[canonicalName] = values[0]

                            elif names[0] == nineml_tester_htmlGUI.categoryEventPortsExpressions:
                                event_ports_expressions[canonicalName] = values[0]

                            elif names[0] == nineml_tester_htmlGUI.categoryVariablesToReport:
                                if values[0] == 'on':
                                    variables_to_report[canonicalName] = True

                            else:
                                raise RuntimeError('Unknown argument category: {0} in the argument: {1}'.format(names[0], key))
                                
                        else:
                            raise RuntimeError('Cannot process argument: ' + key)

                else:
                    html = '<p>No arguments available</p>'# createResultPage('<p>No arguments available</p>')

                nineml_component = TestableComponent('hierachical_iaf_1coba')()
                if not nineml_component:
                    raise RuntimeError('Cannot load NineML component')

                simulation_data = daeSimulationInputData()

                simulation_data.parameters               = parameters
                simulation_data.initial_conditions       = initial_conditions
                simulation_data.analog_ports_expressions = analog_ports_expressions
                simulation_data.event_ports_expressions  = event_ports_expressions
                simulation_data.active_regimes           = active_regimes
                simulation_data.variables_to_report      = variables_to_report

                # Create Log, DAESolver, DataReporter and Simulation object
                log          = daeLogs.daeStringListLog()
                daesolver    = pyIDAS.daeIDAS()
                datareporter = ninemlTesterDataReporter()
                model        = nineml_daetools_bridge(nineml_component.name, nineml_component)
                simulation   = nineml_daetools_simulation(model, parameters               = simulation_data.parameters,
                                                                 initial_conditions       = simulation_data.initial_conditions,
                                                                 active_regimes           = simulation_data.active_regimes,
                                                                 analog_ports_expressions = simulation_data.analog_ports_expressions,
                                                                 event_ports_expressions  = simulation_data.event_ports_expressions,
                                                                 variables_to_report      = simulation_data.variables_to_report)

                # Set the time horizon and the reporting interval
                simulation.ReportingInterval = 0.1
                simulation.TimeHorizon = 10

                # Connect data reporter
                simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
                if(datareporter.Connect("", simName) == False):
                    raise RuntimeError('Cannot connect a TCP/IP datareporter; did you forget to strat daePlotter?')

                # Initialize the simulation
                simulation.Initialize(daesolver, datareporter, log)

                # Save the model reports for all models
                #simulation.m.SaveModelReport(simulation.m.Name + ".xml")
                #simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

                # Solve at time=0 (initialization)
                simulation.SolveInitial()

                # Run
                simulation.Run()
                simulation.Finalize()

                success = True
                log_output = '<pre>{0}</pre>'.format('\n'.join(log.messages))
                content += log_output
                
                html = createResultPage(content)

        else:
            html = 'Error occurred:\n{0}\n{1}'.format(___import_exception___, ___import_exception_traceback___)
            
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
            log_output = 'Log output:\n{0}'.format('\n'.join(log.messages))
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
    