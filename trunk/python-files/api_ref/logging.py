"""
dae.logging module is used to log messages from different parts of the pyDAE framework (core, solver, simulation).
All classes are derived from the base daeLog_t class which contains only one method used for message reporting.
"""

class daeLog_t:
    """
    The base log class (abstract).
    Properties:
     - Enabled: boolean
     - Indent: unsigned integer
     - IndentString: string (inserted before every message)
    """
    def Message(self, Message, Priority):
        """
        (Abstract)
        ARGUMENTS:
         - Message: string
         - Priority: integer (currently not in use)
        RETURNS:
           Nothing
        """
        pass
    
    def IncreaseIndent(self, offset):
        """
        """
        pass
    
    def DecreaseIndent(self, offset):
        """
        """
        pass

class daeBaseLog(daeLog_t):
    """
    The implementation of the simple log class.
    """
    def Message(self, Message, Priority):
        """
        """
        pass
    
    def IncreaseIndent(self, offset):
        """
        """
        pass
    
    def DecreaseIndent(self, offset):
        """
        """
        pass

class daeFileLog(daeBaseLog):
    """
    The log class used to save received messages to the given text file.
    """
    def __init__(self, Filename):
        """
        ARGUMENTS:
         - Filename: string
        """
        pass

class daeStdOutLog(daeBaseLog):
    """
    The log class used to redirect received messages to the standard output.
    """
    pass

class daeTCPIPLog(daeBaseLog):
    """
    The log class used to send messages to daeTCPIPLogServer by TCPIP protocol.
    """
    def __init__(self, TCPIPAddress, Port):
        """
        ARGUMENTS:
         - TCPIPAddress: string
         - Port: unsigned int
        """
        pass

class daeTCPIPLogServer:
    """
    The abstract server class used to receive messages sent by daeTCPIPLog log class.
    User has to implement the method MessageReceived()
    """
    def __init__(self, Port):
        """
        ARGUMENTS:
         - Port: unsigned int
        """
        pass
    
    def MessageReceived(self, Message):
        """
        (Abstract)
        ARGUMENTS:
         - Message: string
        RETURNS:
           Nothing
        """
        pass
