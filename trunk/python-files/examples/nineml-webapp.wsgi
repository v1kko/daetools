import os, sys
import cgitb
cgitb.enable()

___import_exception___ = None
___import_exception_traceback___ = None
try:
    sys.path.append("/home/ciroki/Data/daetools/trunk/python-files/examples")

    from nineml_webapp_common import createErrorPage, getSelectComponentPage

except Exception, e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    ___import_exception___           = str(e)
    ___import_exception_traceback___ = exc_traceback

def application(environ, start_response):
    try:
        html = ''
        if not ___import_exception___:
            html = getSelectComponentPage()
        else:
            html = createErrorPage(___import_exception___, ___import_exception_traceback___)
            
    except Exception, e:
        content = 'Application environment:\n' + pformat(environ) + '\n\n'
        exc_type, exc_value, exc_traceback = sys.exc_info()
        html = createErrorPage(e, exc_traceback, content)

    output_len = len(html)
    start_response('200 OK', [('Content-type', 'text/html'),
                              ('Content-Length', str(output_len))])
    return [html]
    