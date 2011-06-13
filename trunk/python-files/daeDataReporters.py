"""********************************************************************************
                            daeDataReporters.py
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
from pyCore import *
from pyDataReporting import *
try:
    import scipy
    from scipy.io import savemat, loadmat
except ImportError, e:
    print 'Cannot load scipy.io.savemat module', str(e)

class daeMatlabMATFileDataReporter(daeDataReporterLocal):
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def Connect(self, ConnectionString, ProcessName):
        self.ProcessName      = ProcessName
        self.ConnectionString = ConnectionString
        return True

    def IsConnected(self):
        return True

    def Disconnect(self):
        self.Write()
        return True

    def Write(self):
        mdict = {}
        variables = self.Process.Variables
        for var in variables:
            values  = var.Values
            domains = var.Domains
            times   = var.TimeValues
            varName = var.Name
            mdict[varName] = values

        scipy.io.savemat(self.ConnectionString,
                         mdict,
                         appendmat=False,
                         format='5',
                         long_field_names=False,
                         do_compression=False,
                         oned_as='row')

        mat = scipy.io.loadmat(self.ConnectionString)
        print mat
