from pprint import pformat
import os, sys, math, traceback
from time import localtime, strftime, time
import urlparse
import cgitb
cgitb.enable()

___import_exception___ = None
try:
    sys.path.append("/home/ciroki/Data/daetools/trunk/python-files/examples")

    import nineml
    from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
    from nineml.abstraction_layer import ComponentClass
    from nineml_tester_gui import nineml_tester_htmlGUI

    from daetools.pyDAE import pyCore, pyActivity, pyDataReporting, pyIDAS, daeLogs
    from nineml_daetools_bridge import nineml_daetools_bridge
    from nineml_daetools_simulation import daeSimulationInputData, nineml_daetools_simulation
    from nineml_webapp_common import createResultPage, createErrorPage

except Exception, e:
    ___import_exception___ = str(e)

def application(environ, start_response):
    html = ''
    try:
        if not ___import_exception___:
            if environ['REQUEST_METHOD'] == 'POST':
                content_length = int(environ['CONTENT_LENGTH'])

                parameters = {}
                initial_conditions = {}
                analog_ports_expressions = {}
                event_ports_expressions = {}
                active_regimes = {}
                variables_to_report = {}

                content = ''
                if content_length > 0:
                    arguments = pformat(environ['wsgi.input'].read(content_length))
                    arguments = arguments.strip(' \'')

                    #content += '<p>FORM DATA:<br/>{0}</p>'.format(arguments)

                    dictFormData = urlparse.parse_qs(arguments)

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
                    html = createPage('<p>No arguments available</p>')

                content += '<pre>'
                content += 'parameters:<br/>  {0}<br/>'.format(parameters)
                content += 'initial_conditions:<br/>  {0}<br/>'.format(initial_conditions)
                content += 'active_regimes:<br/>  {0}<br/>'.format(active_regimes)
                content += 'analog_ports_expressions:<br/>  {0}<br/>'.format(analog_ports_expressions)
                content += 'event_ports_expressions:<br/>  {0}<br/>'.format(event_ports_expressions)
                content += 'variables_to_report:<br/>  {0}<br/>'.format(variables_to_report)
                content += '</pre>'
                
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
                datareporter = pyDataReporting.daeTCPIPDataReporter()
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
                    sys.exit()

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

                content += '<pre>'
                for line in log.messages:
                    content += line + '<br/>'
                content += '</pre>'
                
                html = createResultPage(content)

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