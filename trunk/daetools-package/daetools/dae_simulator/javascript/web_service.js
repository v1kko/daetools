/***********************************************************************************
                           web_service.js
                 DAE Tools Project, www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************/
class daeWebService
{
    constructor(address, port, webServiceName, method)
    {
        this.address        = address;
        this.port           = port;
        this.webServiceName = webServiceName;
        this.method         = method;
        this.asyncRequests  = false;
    }
    
    get ServerStatus()
    {
        var parameters = {};
        var functionName = 'status';
        var response = this.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
        return response.result['ActiveClients'];
    }

    ClearServer()
    {
        var parameters = {};
        var functionName = 'clear';
        var response = this.executeFun(functionName, parameters);
        if(response.success == false)
            throw response['Reason'];
    }
    
    onSuccess(httpRequest, path, args) 
    {
    }
    
    onError(httpRequest, path, args) 
    {
        var msg  = 'Request failed';
        var msg1 = 'Query: ' + path + '?' + args.toString();
        var msg2 = 'Status: ' + httpRequest.status.toString() + ' ' + httpRequest.statusText;
        var msg3 = 'Headers: ' + httpRequest.getAllResponseHeaders();
        var msg4 = 'Response: \n' + httpRequest.responseText;
        console.log(msg);
        console.log(msg1);
        console.log(msg2);
        console.log(msg3);
        console.log(msg4);
        alert(msg + '\n' + msg1 + '\n' + msg2 + '\n' + msg3 + '\n' + msg4 + '\n\n');
    }
    
    onConnectionFailure(path, error)
    {
        var msg = '';
        if(error !== null)
            msg = error;
        alert(error + '\n\n' + 
              'Connection failure to the server: ' + 
              path + '\n' +
              'The web service has not been started.\nStart it using: python -m daetools.dae_simulator.' + this.webServiceName);
    }
    
    getResponse(httpRequest) 
    {
        var res = {success:false, 
                   reason: "", 
                   result: null};
                
        if (httpRequest.readyState == 4)
        {
            if(httpRequest.status == 200) 
            {
                // Success: '200 OK'.
                var response = JSON.parse(httpRequest.responseText);
                if(response["Status"] == "Success")
                    res.success = true;
                res.result = response["Result"];
            }
            else if(httpRequest.status == 400)
            {
                // Error in client request: '400 BAD REQUEST'.
                var response = JSON.parse(httpRequest.responseText);
                res.reason = response["Reason"];
            }
            else if(httpRequest.status == 500)
            {
                // Server error: '500 INTERNAL SERVER ERROR'.
                var response = JSON.parse(httpRequest.responseText);
                res.reason = response["Reason"];
            }
            else
            {
                res.reason = "Unknown status received: " + httpRequest.status.toString() + " " + httpRequest.statusText;
            }
        } 
        
        return res;
    }
    
    createHTTPRequest()
    {
        if (window.XMLHttpRequest)
            return new XMLHttpRequest();
        else
            return new ActiveXObject("Microsoft.XMLHTTP");
    };
    
    executeFun(functionName, parameters) 
    {
        var webApp = this;
        var httpRequest = this.createHTTPRequest();
        var args = 'function=' + functionName;
        for(var paramName in parameters) 
            args += '&' + String(paramName) + '=' + String(parameters[paramName]);
        var path = this.address + ':' + this.port.toString() + '/' + this.webServiceName;
        httpRequest.open(this.method, path, this.asyncRequests);
        httpRequest.setRequestHeader('Accept', 'application/json');
        httpRequest.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        httpRequest.addEventListener('load', function(text) { console.log(text); });
        httpRequest.onreadystatechange = function() {
            // This function is used for asynchronous request
            if(this.readyState == 4)
            {
                if(httpRequest.status == 200) 
                {
                    // Success: '200 OK'.
                    var response = webApp.getResponse(httpRequest);
                    if(response.success == true)
                        webApp.onSuccess(httpRequest, path, args);
                    else
                        webApp.onError(httpRequest, path, args);
                }
                else if(httpRequest.status == 400)
                {
                    // Error in client request: '400 BAD REQUEST'.
                    webApp.onError(httpRequest, path, args);
                }
                else if(httpRequest.status == 500)
                {
                    // Server error: '500 INTERNAL SERVER ERROR'.
                    webApp.onError(httpRequest, path, args);
                }
                else if(httpRequest.status == 0)
                {
                    // Connection failure
                }
                else
                {
                    webApp.onError(httpRequest, path, args);
                }
            }
        };
        
        /* Send the request with the parameters uri encoded. */
        try
        {
            httpRequest.send(encodeURI(args));
        }
        catch(exception)
        {
            webApp.onConnectionFailure(path, exception);
            throw exception;
        }
        
        /* If the request sent was synchronous then return the response.
           Otherwise, return nothing. */
        if(!this.asyncRequests)
        {
            var response = this.getResponse(httpRequest);
            if(response.success == false)
                throw response['Reason'];
            return response;
        }
        else
            return null;
    }
}

/* Gets the server status.
   This function is run using the setTimeout (delay = 0) to improve responsiveness. */               
function getServerStatus(ws)
{
    var getServerStatus_ = function() {
        if(ws == null)
            return;
        
        var path   = ws.address + ':' + ws.port.toString() + '/' + ws.webServiceName;
        var status = ws.ServerStatus;
        var msg = 'Web service: ' + path + '\n';
        msg += 'Number of active clients: ' + status.length;
        alert(msg);
    };
    setTimeout(getServerStatus_, 0);
}

/* Clears the server (removes all clients; use with care).
   This function is run using the setTimeout (delay = 0) to improve responsiveness. */               
function clearServer(ws)
{
    var clearServer_ = function() {
        if(ws == null)
            return;
        
        ws.ClearServer();
        clearInputs();
        simulation = null;
    };
    setTimeout(clearServer_, 0);
}
