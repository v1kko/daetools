"""********************************************************************************
                            web_service_client.py
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
import os, sys, json
try:
    # python 3
    import http.client as http_client
    from urllib.parse import urlencode
except ImportError:
    # python 2
    import httplib as http_client
    from urllib import urlencode

class daeWebServiceClient(object):
    # The base class for implementing web services accessing daetools web applications.
    # Performs the low level tasks such as encoding the query arguments
    # sending the GET/POST requests and receiving the responses with results to the clients.
    def __init__(self, webServiceName, address, port):
        self.http_connection = http_client.HTTPConnection(address, port)
        self.webServiceName  = webServiceName
       
    def sendRequest(self, args, method = 'POST'):
        # Low level function that sends the HTTP GET or POST query in JSON format.
        #  - HTTP GET:  the query arguments will be encoded into the query url.
        #  - HTTP POST: the query arguments will be sent as JSON content.
        if method == 'GET':
            headers = {'Content-type': 'application/x-www-form-urlencoded', 
                       'User-Agent'  : 'DAE Tools WebService Application/1.0',
                       'Accept'      : 'application/json'}
            parameters = urlencode(args)
            self.http_connection.request('GET', '/%s?%s' % (self.webServiceName, parameters), headers = headers)
        else:
            headers = {'Content-type': 'application/x-www-form-urlencoded', 
                       'User-Agent'  : 'DAE Tools WebService Application/1.0',
                       'Accept'      : 'application/json'}
            parameters = urlencode(args)
            self.http_connection.request('POST', '/%s' % self.webServiceName, body = parameters, headers = headers)
        
    def getResponse(self):
        # Returns the query response from the web app as a dictionary object.
        response     = self.http_connection.getresponse()
        response_str = response.read().decode()
        json_data    = json.loads(response_str)

        # All responses where any of the following conditions is satisfied is an error:
        #  - status code that is not '200 OK' 
        #  - 'Status' field does not exist
        #  - 'Status' field is not equal to 'Success'.
        if (response.status != 200) or ('Status' not in json_data) or (json_data['Status'] != 'Success'):
            reason = ''
            if 'Reason' in json_data:
                reason = json_data['Reason']
            raise RuntimeError('The http request failed: %s %s\nDescription: %s' % (response.status, response.reason, reason))
        
        # Accept only responses in JSON format.
        content_type = response.getheader('Content-type', '')
        if content_type != 'application/json':
            raise RuntimeError('Invalid content type received: %s' % content_type)
        
        # If the response does not contain the 'Result' field something went wrong. 
        if 'Result' not in json_data:
            raise RuntimeError('The JSON response does not contain the Result field:\n%s' % json_data)
            
        # Return the response results as a python object (it is function dependent).
        # Clients should know how to interpret it.
        return json_data['Result']
