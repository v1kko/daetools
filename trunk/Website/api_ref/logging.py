"""
dae.logging module is used to log messages from different parts of the pyDAE framework (core, solver, simulation).
All classes are derived from the base daeLog_t class which contains only one method used for message reporting.
"""

class daeLog_t:
    """
    The base log class (abstract).
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

class daeFileLog(daeLog_t):
    """
    The log class used to save received messages to the given text file.
    """
    def __init__(self, Filename):
        """
        ARGUMENTS:
         - Filename: string
        """
        pass

class daeStdOutLog(daeLog_t):
    """
    The log class used to redirect received messages to the standard output.
    """
    pass

class daeTCPIPLog(daeLog_t):
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
