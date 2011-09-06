from pprint import pformat
from cgi import parse_qs, escape
import os, sys, math
from time import localtime, strftime, time

def application(environ, start_response):
    output = []

    #output.append('<pre>')
    #output.append(pformat(environ))
    #output.append('</pre>')

    if environ['REQUEST_METHOD'] == 'GET':
        try:
            sys.path.append("/home/ciroki/Data/daetools/trunk/python-files/examples")
            import nineml
            from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
            from nineml.abstraction_layer import ComponentClass
            from nineml_tester_gui import nineml_tester_htmlGUI

            nineml_component = TestableComponent('hierachical_iaf_1coba')()
            if not nineml_component:
                form = 'Cannot load NineML component'

            parameters = {
                'cobaExcit.q' : 3.0,
                'cobaExcit.tau' : 5.0,
                'cobaExcit.vrev' : 0.0,
                'iaf.cm' : 1,
                'iaf.gl' : 50,
                'iaf.taurefrac' : 0.008,
                'iaf.vreset' : -60,
                'iaf.vrest' : -60,
                'iaf.vthresh' : -40
            }
            initial_conditions = {
                'cobaExcit.g' : 0.0,
                'iaf.V' : -60,
                'iaf.tspike' : -1E99
            }
            analog_ports_expressions = {
                'cobaExcit.V' : '1.2 * e',
                'iaf.ISyn' : '5 * pi'
            }
            event_ports_expressions = {
            }
            active_states = [
                'cobaExcit.cobadefaultregime',
                'iaf.subthresholdregime'
            ]
            variables_to_report = [
                'cobaExcit.g',
                'iaf.tspike'
            ]

            s = nineml_tester_htmlGUI(nineml_component, parameters               = parameters,
                                                        initial_conditions       = initial_conditions,
                                                        active_states            = active_states,
                                                        analog_ports_expressions = analog_ports_expressions,
                                                        event_ports_expressions  = event_ports_expressions,
                                                        variables_to_report      = variables_to_report)
            form = s.generateHTMLForm()

            output.append('<html>')
            output.append('<body style="margin-left: auto; margin-right: auto; width: 70%;" >')
            output.append(form)
            output.append('</body>')
            output.append('</html>')

        except Exception, e:
            output.append('Exception: ' + str(e))
        
    elif environ['REQUEST_METHOD'] == 'POST':
        # show form data as received by POST:
        output.append('<h1>FORM DATA</h1>')
        content_length = int(environ['CONTENT_LENGTH'])
        if content_length > 0:
            query_string = parse_qs(pformat(environ['wsgi.input'].read(content_length)))
            output.append('<p>' + str(query_string) + '</p>')
        else:
            output.append('<p>No data</p>')

    # send results
    output_len = sum(len(line) for line in output)
    start_response('200 OK', [('Content-type', 'text/html'),
                              ('Content-Length', str(output_len))])
    return output