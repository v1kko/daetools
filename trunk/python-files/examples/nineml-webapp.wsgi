import os, sys, math
import cgitb
cgitb.enable()

___import_exception___ = None
try:
    sys.path.append("/home/ciroki/Data/daetools/trunk/python-files/examples")

    from nineml_webapp_common import createErrorPage, getSelectComponentPage

except Exception, e:
    ___import_exception___ = str(e)

def application(environ, start_response):
    html = ''
    try:
        html = getSelectComponentPage()
            
    except Exception, e:
        html = createErrorPage('<p>Error occurred: {0}</p>'.format(str(e)))

    output = []
    output.append(html)
    output_len = sum(len(line) for line in output)
    start_response('200 OK', [('Content-type', 'text/html'),
                              ('Content-Length', str(output_len))])
    return output