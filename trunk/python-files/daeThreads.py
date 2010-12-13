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

