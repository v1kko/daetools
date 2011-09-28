from __future__ import print_function
from pprint import pformat
import os, sys, math, json, traceback, os.path, tempfile, shutil
import cPickle as pickle
from time import localtime, strftime, time
import urlparse
import zipfile
import cgitb
cgitb.enable()

___import_exception___ = None
___import_exception_traceback___ = None
try:
    baseFolder = '/home/ciroki/Data/daetools/trunk/python-files/nineml'
    sys.path.append(baseFolder)
    os.environ['HOME'] = tempfile.gettempdir()
    #print(os.environ, file=sys.stderr)

    import nineml
    from nineml.abstraction_layer import readers
    from nineml.abstraction_layer.testing_utils import TestableComponent
    from nineml.abstraction_layer import ComponentClass

    from daetools.pyDAE import pyCore, pyActivity, pyDataReporting, pyIDAS, daeLogs
    from nineml_component_inspector import nineml_component_inspector
    from nineml_daetools_bridge import nineml_daetools_bridge
    from nineml_tex_report import createLatexReport, createPDF
    from nineml_daetools_simulation import daeSimulationInputData, nineml_daetools_simulation, ninemlTesterDataReporter
    from nineml_webapp_common import createErrorPage, getSetupDataForm, createSetupDataPage, getSelectComponentPage, createResultPage

except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    ___import_exception___           = str(e)
    ___import_exception_traceback___ = exc_traceback

class nineml_webapp:
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

            # Create tmpFolder
            tmpFolder = tempfile.mkdtemp(prefix='nineml-webapp-', suffix='-tmp')
            os.chmod(tmpFolder, 0777)

            applicationID   = os.path.split(tmpFolder)[1]
            pickle_filename = os.path.join(tmpFolder, 'webapp.pickle')
            
            f_pickle = open(pickle_filename, 'wb')
            pickle.dump(inspector, f_pickle, pickle.HIGHEST_PROTOCOL)
            f_pickle.close()
            
            formTemplate  = getSetupDataForm()
            content      += formTemplate.format(nineml_component.name, inspector.generateHTMLForm(), applicationID)
            html          = createSetupDataPage(content)
        
        except Exception as e:
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
            pdf = None
            zip = None
            html = ''
            texReport = ''
            pdfReport = ''
            tmpFolder = ''
            raw_arguments = ''
            content = ''
            tests_content = ''
            log_output = ''
            log = None
            timeHorizon = 0.0
            reportingInterval = 0.0
            parameters = {}
            initial_conditions = {}
            analog_ports_expressions = {}
            event_ports_expressions = {}
            active_regimes = {}
            variables_to_report = {}
            tests_data = []
            applicationID = ''
            inspector = None
            
            success = False
            
            if not dictFormData.has_key('__NINEML_WEBAPP_ID__'):
                raise RuntimeError('No application ID has been specified')

            applicationID   = dictFormData['__NINEML_WEBAPP_ID__'][0]
            tmpFolder       = os.path.join(tempfile.gettempdir(), applicationID)
            pickle_filename = os.path.join(tempfile.gettempdir(), applicationID, 'webapp.pickle')
            if (applicationID == '') or (not os.path.isfile(pickle_filename)) or (not os.path.isdir(tmpFolder)):
                return self.initial_page(environ, start_response)
            
            f_pickle  = open(pickle_filename)
            inspector = pickle.load(f_pickle)
            f_pickle.close()
            if not inspector:
                raise RuntimeError('Invalid inspector object')
            nineml_component = inspector.ninemlComponent
            if not nineml_component:
                raise RuntimeError('Cannot load NineML component')

            # Copy the logo image to tmp folder
            shutil.copy2(baseFolder + '/logo.png', tmpFolder + '/logo.png')

            isOK, results = self.run_test(nineml_component, dictFormData, tmpFolder)
            if isOK:
                testName, testDescription, dictInputs, plots, log_output = results
                tests_data.append( (testName, testDescription, dictInputs, plots, log_output) )
                content += 'Test status: {0} [SUCCEEDED]'.format(testName)
            else:
                #error    = results[5]
                testName, testDescription, dictInputs, plots, log_output, error = results
                content += 'Test status: {0} [FAILED]'.format(testName)
                content += error
                
            texReport = '{0}/{1}.tex'.format(tmpFolder, applicationID)
            pdfReport = '{0}/{1}.pdf'.format(tmpFolder, applicationID)

            # Generate Tex report
            createLatexReport(inspector, tests_data, os.path.join(baseFolder, 'nineml-tex-template.tex'), texReport, tmpFolder)

            # Generate PDF report
            createPDF = os.path.join(tmpFolder, 'createPDF.sh')
            createPDFfile = open(createPDF, "w")
            createPDFfile.write('cd {0}\n'.format(tmpFolder))
            # Run it twice because of the problems with the Table Of Contents (we need two passes)
            createPDFfile.write('/usr/bin/pdflatex -interaction=nonstopmode {0}\n'.format(texReport))
            createPDFfile.write('/usr/bin/pdflatex -interaction=nonstopmode {0}\n'.format(texReport))
            createPDFfile.close()
            os.system('sh {0}'.format(createPDF))

            zipReport = '{0}/{1}.zip'.format(tmpFolder, applicationID)
            self.pack_tests_data(zipReport, tests_data, tmpFolder)
            
            # Read the contents of the report into a variable (to be sent together with the .html part)
            if os.path.isfile(pdfReport):
                pdf = open(pdfReport, "rb").read()
            if os.path.isfile(zipReport):
                zip = open(zipReport, "rb").read()
            
            success = True
            html = createResultPage(content)

        except Exception as e:
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

        # Remove temporary directory
        if os.path.isdir(tmpFolder):
            shutil.rmtree(tmpFolder)
            
        if success:
            part1 = ''
            part2 = ''
            part3 = ''
            boundary = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            part1 = '--{0}\r\nContent-Type: text/html\r\n\r\n{1}\n'.format(boundary, html)
            if pdf:
                part2 = '--{0}\r\nContent-Disposition: attachment; filename=model-report.pdf\r\n\r\n{1}\n'.format(boundary, pdf)
            if zip:
                part3 = '--{0}\r\nContent-Disposition: attachment; filename=report-data.zip\r\n\r\n{1}\n'.format(boundary, zip)
            output = part1 + part2 + part3
            output_len = len(output)
            start_response('200 OK', [('Content-type', 'multipart/mixed; boundary={0}'.format(boundary)),
                                    ('Content-Length', str(output_len))])
            return [output]

        else:
            output_len = len(html)
            start_response('200 OK', [('Content-type', 'text/html'),
                                    ('Content-Length', str(output_len))])
            return [html]

    def pack_tests_data(self, zipReport, tests_data, tmpFolder):
        try:
            if len(tests_data) == 0:
                return
                
            zip = zipfile.ZipFile(zipReport, "w")
            
            for i, test_data in enumerate(tests_data):
                testName, testDescription, dictInputs, plots, log_output = test_data
                testFolder = 'test-no.{0}/'.format(i+1, testName) 
                
                # Write log file contents
                logName = '/log_output.txt'
                f = open(tmpFolder + '/' + logName, "w")
                f.write(log_output)
                f.close()
                zip.write(tmpFolder + '/' + logName, testFolder + logName)
                
                # Write .png and .csv files
                for plot in plots:
                    varName, xPoints, yPoints, pngName, csvName = plot
                    zip.write(tmpFolder + '/' + pngName, testFolder + pngName)
                    zip.write(tmpFolder + '/' + csvName, testFolder + csvName)
            
            zip.close()
            
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(str(e), '\n'.join(traceback.format_tb(exc_traceback)), file=sys.stderr)
    
    def run_test(self, nineml_component, dictFormData, tmpFolder):
        try:
            testName = 'Test name'
            testDescription = 'Test description'
            timeHorizon = 0.0
            reportingInterval = 0.0
            parameters = {}
            initial_conditions = {}
            analog_ports_expressions = {}
            event_ports_expressions = {}
            active_regimes = {}
            variables_to_report = {}
            log_output = ''
            log          = None
            daesolver    = None
            datareporter = None
            model        = None
            simulation   = None
            dictInputs = {}

            if dictFormData.has_key('testName'):
                testName = str(dictFormData['testName'][0])
            if dictFormData.has_key('testDescription'):
                testDescription = str(dictFormData['testDescription'][0])
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
            simulation.Finalize()

            log_output = log.JoinMessages('\n')
            
            dictInputs['parameters']                = parameters
            dictInputs['initial_conditions']        = initial_conditions
            dictInputs['analog_ports_expressions']  = analog_ports_expressions
            dictInputs['event_ports_expressions']   = event_ports_expressions
            dictInputs['active_regimes']            = active_regimes
            dictInputs['variables_to_report']       = variables_to_report
            dictInputs['timeHorizon']               = timeHorizon
            dictInputs['reportingInterval']         = reportingInterval
            plots = datareporter.createReportData(tmpFolder)

            return True, (testName, testDescription, dictInputs, plots, log_output)
            
        except Exception as e:
            if log:
                log_output = '<pre>{0}</pre>'.format(log.JoinMessages('\n'))
            return False, (testName, testDescription, dictInputs, None, log_output, str(e))
        
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

        except Exception as e:
            content = 'Application environment:\n' + pformat(environ) + '\n\n'
            exc_type, exc_value, exc_traceback = sys.exc_info()
            html = createErrorPage(e, exc_traceback, content)

        output_len = len(html)
        start_response('200 OK', [('Content-type', 'text/html'),
                                ('Content-Length', str(output_len))])
        return [html]

application = nineml_webapp()