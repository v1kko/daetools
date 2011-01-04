"""
daetools.pyDAE.datareporting module consists of two parts:
 1) DataReporter
 2) DataReceiver

DataReporter is used by the running activity to send the information (a client). 
There are two different types of data reporters: Local and Remote. Local do not need the server side (data receiver) and are capable to process 
the received data internally (to save them into a file, for instance). Remote need the server to send the data. 
The main class is daeDataReporter_t which defines interface used to perform the following tasks:
 - Connect/Disconnect to the corresponding DataReceiver (if any). A good example is a tcpip client which connects and sends data to a tcpip server.
 - Send the info about domains and variables in the activity
 - Send the results for specified variables at the given time intervals

DataReceiver is used to receive the information from the activity (a server). The most often it is a tcp/ip server but in general any type of a client/server 
communication can be implemented (pipes, shared memory, etc).
The main class is daeDataReceiver_t which defines an interface used to access the received data.
"""

class daeDataReporterDomain:
    """
    Class used in daeDataReporter_t::RegisterDomain() function to register a domain.
    Contains the information about a domain in the running activity.
    PROPERTIES:
     - Name: string (read-only)
     - Type: daeeDomainType (read-only)
     - NumberOfPoints: unsigned int (read-only)
     - Points: list (read-only)
    """
    def __init__(self, Name, Type, NumberOfPoints):
        """
        ARGUMENTS:
         - Name: string
         - Type: daeeDomainType
         - NumberOfPoints: unsigned int
        """
        pass
    
    def __getitem__(self, Index):
        """
        ARGUMENTS:
         - Index: unsigned int
        RETURNS:
          float
        """
        pass
    
    def __setitem__(self, Index, Value):
        """
        ARGUMENTS:
         - Index: unsigned int
         - Value: float
        RETURNS:
           Nothing
        """
        pass
        
class daeDataReporterVariable:
    """
    Class used in daeDataReporter_t::RegisterVariable() function to register a variable.
    Contains the information about a variable in the running activity.
    PROPERTIES:
     - Name: string (read-only)
     - NumberOfPoints: unsigned int (read-only)
     - NumberOfDomains: unsigned int (read-only)
     - Domains: list of strings (read-only)
    """
    def __init__(self, Name, NumberOfPoints):
        """ 
        ARGUMENTS:
         - Name: string
         - NumberOfPoints: unsigned int
        """
        pass

    def AddDomain(self, Name):
        """ 
        ARGUMENTS:
         - Name: string
        RETURNS:
           Nothing
        """
        pass
        
class daeDataReporterVariableValue:
    """
    Class used in daeDataReporter_t::SendVariable() function to send variable values.
    Contains the information about variable values at the specified time in the running activity.
    PROPERTIES:
     - Name: string (read-only)
     - NumberOfPoints: unsigned int (read-only)
     - Values: list of floats (read-only)
    """
    def __init__(self, Name, NumberOfPoints):
        """ 
        ARGUMENTS:
         - Name: string
         - NumberOfPoints: unsigned int
        """
        pass
        
    def __getitem__(self, Index):
        """
        ARGUMENTS:
         - Index: unsigned int
        RETURNS:
          float
        """
        pass
    
    def __setitem__(self, Index, Value):
        """
        ARGUMENTS:
         - Index: unsigned int
         - Value: float
        RETURNS:
           Nothing
        """
        pass

class daeDataReporter_t:
    """
    The base data reporter class (abstract).
    """
    def Connect(self, ConnectionString, ProcessName):
        """
        (Abstract)
        ARGUMENTS:
         - ConnectionString: string
         - ProcessName: string
        RETURNS:
           bool
        """
        pass

    def Disconnect(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           bool
        """
        pass
    
    def IsConnected(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           bool
        """
        pass
    
    def StartRegistration(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           bool
        """
        pass
    
    def RegisterDomain(self, Domain):
        """
        (Abstract)
        ARGUMENTS:
         - Domain: daeDataReporterDomain
        RETURNS:
           bool
        """
        pass
    
    def RegisterVariable(self, Variable):
        """
        (Abstract)
        ARGUMENTS:
         - Variable: daeDataReporterVariable
        RETURNS:
           bool
        """
        pass
    
    def EndRegistration(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           bool
        """
        pass
    
    def StartNewResultSet(self, Time):
        """
        (Abstract)
        ARGUMENTS:
         - Time: float
        RETURNS:
           bool
        """
        pass
    
    def EndOfData(self):
        """
        (Abstract)
        ARGUMENTS:
           None
        RETURNS:
           bool
        """
        pass
    
    def SendVariable(self, Variable):
        """
        (Abstract)
        ARGUMENTS:
         - Variable: daeDataReporterVariableValue
        RETURNS:
           bool
        """
        pass

class daeDelegateDataReporter(daeDataReporter_t):
    """
    Class which delegates all method calls to the given list of data reporters. 
    For instance, it can be useful for simultaneous sending data to a server and data saving into a file.
    The number of data reporters is unlimited.
    """
    def AddDataReporter(self, DataReporter):
        """
        ARGUMENTS:
         - DataReporter: daeDataReporter_t
        RETURNS:
           Nothing
        """
        pass

class daeDataReporterLocal(daeDataReporter_t):
    """
    This class does not send the data received but stores them into a daeDataReporterProcess object.
    It implements all methods from the daeDataReporter_t interface and it is typically used to process the data 
    internally (to save them into a file, for example). 
    It defines only one new method used to obtain the daeDataReporterProcess object.
    PROPERTIES:
     - Process: daeDataReporterProcess (read-only)
    """

class daeDataReporterFile(daeDataReporterLocal):
    """
    Abstract class derived from daeDataReporterLocal which defines a method to write the data from daeDataReporterProcess into a file.
    WriteDataToFile() method must be implemented in derived classes.
    """
    def WriteDataToFile(self, Filename):
        """
        (Abstract)
        ARGUMENTS:
         - Filename: string
        RETURNS:
           Nothing
        """
        pass

class daeTEXTFileDataReporter(daeDataReporterFile):
    """
    Auxiliary class derived from daeDataReporterFile to save the data in a simple format into a text file.:
    """
    def WriteDataToFile(self, Filename):
        """
        ARGUMENTS:
         - Filename: string
        RETURNS:
           Nothing
        """
        pass

class daeDataReporterRemote(daeDataReporter_t):
    """
    Class which does not keep the data received but sends them immediately to the appropriate server (daeDataReceiver_t).
    It implements all methods from the daeDataReporter_t interface and it is typically used to send the data to a server.
    It defines one new method to send the received data. The data are formatted by the internal message formatter (in the binary format).
    Derived classes must implement the following methods:
     - Connect
     - Disconnect
     - IsConnected
     - SendMessage
    """
    def SendMessage(self, Message):
        """
        (Abstract)
        ARGUMENTS:
         - Message: string
        RETURNS:
           bool
        """
        pass

class daeTCPIPDataReporter(daeDataReporterRemote):
    """
    Class used to send the received data to a server by tcp/ip protocol.
    """
    def __init__(self, ConnectionString, ProcessName):
        """ 
        ARGUMENTS:
         - ConnectionString: string
           The connection string is given in the following form: TCPIPAddress:PORT (192.168.0.1:50000 for instance)
         - ProcessName: string
        """
        pass

class daeDataReceiverDomain:
    """
    PROPERTIES:
     - Name: string (read-only)
     - Type: daeeDomainType (read-only)
     - NumberOfPoints: unsigned int (read-only)
     - Points: one-dimensional numpy array (read-only)
    """
    def __init__(self, Name, Type, NumberOfPoints):
        """ 
        ARGUMENTS:
         - Name: string
         - Type: daeeDomainType
         - NumberOfPoints: unsigned int
        """
        pass
    
    def __getitem__(self, Index):
        """
        ARGUMENTS:
         - Index: unsigned int
        RETURNS:
          float
        """
        pass
    
    def __setitem__(self, Index, Value):
        """
        ARGUMENTS:
         - Index: unsigned int
         - Value: float
        RETURNS:
           Nothing
        """
        pass

class daeDataReceiverVariableValue:
    """
    PROPERTIES:
     - Time: float (read-only)
    """
    def __init__(self, Time, NumberOfPoints):
        """ 
        ARGUMENTS:
         - Name: string
         - NumberOfPoints: unsigned int
        """
        pass

    def __getitem__(self, Index):
        """
        ARGUMENTS:
         - Index: unsigned int
        RETURNS:
          float
        """
        pass
    
    def __setitem__(self, Index, Value):
        """
        ARGUMENTS:
         - Index: unsigned int
         - Value: float
        RETURNS:
           Nothing
        """
        pass
    
class daeDataReceiverVariable:
    """
    PROPERTIES:
     - Name: string (read-only)
     - NumberOfPoints: unsigned int (read-only)
     - Domains: list of daeDataReceiverDomains (read-only)
     - TimeValues: multi-dimensional numpy array (read-only)
                   Array of available points in time that the data are available for
     - Values: multi-dimensional numpy array (read-only)
               Array of values for all time points
    """
    def __init__(self, Name, NumberOfPoints):
        """ 
        ARGUMENTS:
         - Name: string
         - NumberOfPoints: unsigned int
        """
        pass
    
    def AddDomain(self, Domain):
        """
        ARGUMENTS:
         - Domain: daeDataReceiverDomain
        RETURNS:
           Nothing
        """
        pass
    
    def AddVariableValue(self, VariableValue):
        """
        ARGUMENTS:
         - VariableValue: daeDataReceiverVariableValue
        RETURNS:
           Nothing
        """
        pass

class daeDataReporterProcess:
    """
    PROPERTIES:
     - Name: string (read-only)
     - Domains: list of daeDataReceiverDomains (read-only)
     - Variables: list of daeDataReceiverVariables (read-only)
    """
    def RegisterDomain(self, Domain):
        """
        ARGUMENTS:
         - Domain: daeDataReceiverDomain
        RETURNS:
           Nothing
        """
        pass
    
    def RegisterVariable(self, Variable):
        """
        ARGUMENTS:
         - Variable: daeDataReceiverVariable
        RETURNS:
           Nothing
        """
        pass
    
    def FindVariable(self, Name):
        """
        ARGUMENTS:
         - Name: string
        RETURNS:
           daeDataReceiverVariable object
        """
        pass


class daeDataReceiver_t:
    """
    Abstract class. Defines DataReceiver interface
    """
    def Start(self):
        """
        (Abstract)
        ARGUMENTS:
          None
        RETURNS:
          bool
        """
        pass
    
    def Stop(self):
        """
        (Abstract)
        ARGUMENTS:
          None
        RETURNS:
          bool
        """
        pass
    
    def GetProcess(self):
        """
        (Abstract)
        ARGUMENTS:
          None
        RETURNS:
          daeDataReporterProcess
        """
        pass

class daeTCPIPDataReceiverServer:
    """
    PROPERTIES:
     - NumberOfProcesses: unsigned int (read-only)
     - NumberOfDataReceivers: unsigned int (read-only)
    """
    def __init__(self, Port):
        """ 
        ARGUMENTS:
         - Port: unsigned int
        """
        pass

    def Start(self):
        """
        ARGUMENTS:
          None
        RETURNS:
          bool
        """
        pass
    
    def Stop(self):
        """
        ARGUMENTS:
          None
        RETURNS:
          bool
        """
        pass
    
    def GetProcess(self, Index):
        """
        ARGUMENTS:
         - Index: unsigned int
        RETURNS:
          daeDataReporterProcess
        """
        pass

    def GetDataReceiver(self, Index):
        """
        ARGUMENTS:
         - Index: unsigned int
        RETURNS:
          daeDataReceiver_t
        """
        pass

