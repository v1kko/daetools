"""********************************************************************************
                             daeLogs.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software 
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
import sys
from pyCore import *

class daePythonStdOutLog(daeStdOutLog):
    def __init__(self):
        daeStdOutLog.__init__(self)

    def Message(self, message, severity):
        print '{0:30s}'.format(self.IndentString + message)
        print ' {0} {1}'.format(self.PercentageDone, self.ETA), "\r",
        sys.stdout.flush()
