"""********************************************************************************
                            web_service.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
import os, sys, json, traceback, cgi
import uuid, zipfile, cgitb, hashlib, threading
import wsgiref.simple_server
from daetools.pyDAE import *

try:
    # python 3
    import http.client as http_client
    from urllib.parse import urlparse
except ImportError:
    # python 2
    import httplib as http_client
    from urlparse import urlparse

# cgi traceback
logdir = os.path.join(os.path.expanduser('~'), 'daetools_ws')
cgitb.enable(display = 0, logdir = logdir, format = 'html') 

class daeWebService(object):
    # The base class for web applications/web services.
    # Performs the low level tasks such as parsing the query to get its arguments
    # and sending responses with results to the clients.
    def __init__(self):
        self.noHTTPRequests = 0
        self.activeObjects = {}
        self.httpServer    = None
    
    @staticmethod
    def startWebService(application, address, port, serveForever = False):
        # Start the web application.
        http_port    = int(port)
        http_address = str(address)
        
        httpd = wsgiref.simple_server.make_server(http_address, http_port, application, handler_class=NoLoggingWSGIRequestHandler)
        application.httpServer = httpd
        if serveForever:
            # Serve even if there is no active clients.
            httpd.serve_forever(poll_interval = 5.0)
        else:
            # Serve as long as the function keepRunning returns True.
            # keepRunning will return False if all clients freed their objects and exited.
            while application.keepRunning():
                httpd.handle_request()

    @staticmethod
    def tryConnectWebService(address, port, timeout = 10):
        # Try to connect to the above server (timeout is 10s).
        # If successful, the server has successfully been started
        # and the subsequent clients can connect to it.
        c = http_client.HTTPConnection(address, port, timeout)
    
    def keepRunning(self):
        # Server will stop if at least one request has been processed and no activeObjects exist
        # (all active objects finished with the simulation).
        # Therefore, the server can serve multiple clients but will shutdown after all objects 
        # freed their resources and the number of active objects goes to zero.
        return len(self.activeObjects) > 0 or self.noHTTPRequests == 0
    
    def getQueryArguments(self, environ):
        # Returns the query arguments' values as a dictionary object.
        # Can process both GET and POST requests (double check this).
        arguments   = cgi.FieldStorage(fp = environ['wsgi.input'], environ = environ) 
        pathInfo    = environ['PATH_INFO']
        return pathInfo, arguments
        
    def jsonError(self, reason, start_response, simulationID):
        # Returns the error status to the client.
        # The argument reason is a string with the exception description.
        # In case of errors always deletes the current object identified by its simulationID.
        if simulationID and simulationID in self.activeObjects:
            del self.activeObjects[simulationID]
        obj = {'Status': 'Error',
               'Reason': reason,
               'Result': None}
        return self._sendResponse_json(obj, '500 Internal Server Error', start_response)
        
    def jsonSuccess(self, start_response):
        # Returns the success status to the client and no results.
        obj = {'Status': 'Success',
               'Reason': None,
               'Result': None}
        return self._sendResponse_json(obj, '200 OK', start_response)

    def jsonResult(self, result, start_response):
        # Returns the success status and the data to the client.
        obj = {'Status': 'Success',
               'Reason': None,
               'Result': result}
        return self._sendResponse_json(obj, '200 OK', start_response)
    
    def _sendResponse_json(self, obj, status, start_response):
        # Low level function that sends the HTTP response in JSON format.
        # The argument obj contains the data to be sent as JSON message.
        json_b = json.dumps(obj, indent = 2).encode() # default is utf-8
        start_response(status, [('Access-Control-Allow-Origin', '*'),
                                ('Access-Control-Allow-Methods', 'POST, GET, OPTIONS'),
                                ('Content-type', 'application/json'),
                                ('Content-Length', str(len(json_b)))])
        return [json_b]

    def _sendResponse_html(self, html_content, status, start_response):
        # Low level function that sends the HTTP response in HTML format.
        # The argument html_content contains the HTML message.
        html_b = html_content.encode() # default is utf-8
        start_response(status, [('Access-Control-Allow-Origin', '*'),
                                ('Access-Control-Allow-Methods', 'POST, GET, OPTIONS'),
                                ('Content-type', 'text/html'),
                                ('Content-Length', str(len(html_b)))])
        return [html_b]

    def _sendFile(self, filePath, contentType, start_response):
        # Low level function that sends a file as the HTTP response.
        # The argument html_content contains the HTML message.
        try:
            f = open(filePath, 'rb')
            content     = f.read()
            filename    = os.path.basename(filePath)
            content_len = len(content)
            start_response('200 OK', [('Access-Control-Allow-Origin', '*'),
                                      ('Access-Control-Allow-Methods', 'POST, GET, OPTIONS'),
                                      ('Content-type', contentType),
                                      ('Content-Disposition', 'attachment; filename=%s' % filename),
                                      ('Content-Length', str(content_len))])
        finally:
            f.close()
        return [content]

class NoLoggingWSGIRequestHandler(wsgiref.simple_server.WSGIRequestHandler):
    # Request handler that suppresses logging to the stderr.
    def log_message(self, format, *args):
        pass
