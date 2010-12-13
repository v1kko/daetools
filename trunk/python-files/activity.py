from pyDAE import *
from PyQt4 import QtCore, QtGui
from Queue import *
from ui_activity import Ui_Activity

class logObject(QtCore.QObject):
	InsertMessage = QtCore.pyqtSignal(QtCore.QString)

class daeActivityLogServer(daeTCPIPLogServer):
    def __init__(self, port):
        daeTCPIPLogServer.__init__(self, port)
        self.object = logObject()
        
    def MessageReceived(self, message):
        print message
        #self.object.InsertMessage.emit(message)

class daeTextEditLog(daeStdOutLog):
    def __init__(self):
        daeStdOutLog.__init__(self)
        self.object = logObject()

    def Message(self, message, severity):
        super(daeTextEditLog, self).Message(message, severity)               
        self.object.InsertMessage.emit(message)

class Activity(QtGui.QMainWindow):
    def __init__(self, port):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_Activity()
        self.ui.setupUi(self)
        self.logServer = daeActivityLogServer(port)
        self.logServer.object.InsertMessage.connect(self.InsertMessage)
        #self.log = texteditlog
        #self.log.object.InsertMessage.connect(self.InsertMessage)

    @QtCore.pyqtSlot()
    def InsertMessage(self, msg):
        self.ui.textEditLog.append(msg)
        if self.ui.textEditLog.isVisible() == True:
        	self.ui.textEditLog.update()
        #self.ui.textEditLog.ensureCursorVisible()

    @QtCore.pyqtSlot()
    def slotPauseThread(self):
        #simulation.Pause()
        pass

    @QtCore.pyqtSlot()
    def slotResumeThread(self):
        pass

    @QtCore.pyqtSlot()
    def slotStopThread(self):
        pass
        

