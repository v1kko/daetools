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

