"""********************************************************************************
                             daeProcess.py
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
#from multiprocessing import Process
from pyDAE import *
from PyQt4 import QtCore

def runSimulation(simulation, solver, datareporter, log):
    # Initialize simulation object
    simulation.Initialize(solver, datareporter, log)
    #daeSaveModel(sim.m, "Conduction.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    #print "uja2"
    # Finally, start the simulation
    simulation.Run()

#def runTextEditLog(log):
#    app = QtGui.QApplication(sys.argv)
#    log = TextEditLog() #daeStdOutLog()
#    act = activity.Activity()
#    act.setModal(True)
#    log.object.InsertMessage.connect(self.act.InsertMessage)
#    act.exec_()

