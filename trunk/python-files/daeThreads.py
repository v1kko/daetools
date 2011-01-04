"""********************************************************************************
                             daeThreads.py
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
import threading
from threading import Thread

from libdaepython import *
from PyQt4 import QtCore

class runSimulationThread(threading.Thread):
    def __init__(self, simulation, solver, datareporter, log):
        threading.Thread.__init__(self)
        self.simulation   = simulation
        self.solver       = solver
        self.datareporter = datareporter
        self.log          = log

    def run(self):
        # Initialize simulation object
        self.simulation.Initialize(self.solver, self.datareporter, self.log)
        #daeSaveModel(sim.m, "Conduction.xml")

        # Solve at time=0 (initialization)
        self.simulation.SolveInitial()
        
        # Finally, start the simulation
        self.simulation.Run()

class runTextEditLogThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # Create application and activity window
        self.app = QtGui.QApplication(sys.argv)
        self.log = TextEditLog() #daeStdOutLog()
        self.act = activity.Activity()
        self.act.setModal(True)
        self.log.object.InsertMessage.connect(self.act.InsertMessage)

    def run(self):
        self.act.exec_()
