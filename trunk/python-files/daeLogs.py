from pyCore import *

class daePythonStdOutLog(daeStdOutLog):
    def __init__(self):
        daeStdOutLog.__init__(self)

    def Message(self, message, severity):
        print message


