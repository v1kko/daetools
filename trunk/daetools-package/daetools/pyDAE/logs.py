"""********************************************************************************
                               logs.py
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
python_major = sys.version_info[0]

class daePythonStdOutLog(daeStdOutLog):
    def __init__(self):
        daeStdOutLog.__init__(self)

    @property
    def Name(self):
        return 'PythonStdOutLog'

    def Message(self, message, severity):
        if self.Enabled:
            if self.PrintProgress:
                sys.stdout.write('{0:30s}\n'.format(self.IndentString + message))
                sys.stdout.write(' {0} {1}'.format(self.PercentageDone, self.ETA))
                sys.stdout.write('\r')
            else:
                print(self.IndentString + message)
            sys.stdout.flush()
