import sys
from time import localtime, strftime

try:
    from daetools.pyDAE import *
except ImportError, e:
    print '[daeRunExamples]: Cannot pyDAE module', str(e)

try:
    from PyQt4 import QtCore, QtGui, QtWebKit
except ImportError, e:
    print '[daeRunExamples]: Cannot load pyQt4 modules', str(e)

try:
    from RunExamples_ui import Ui_RunExamplesDialog
    from WebView_ui import Ui_WebViewDialog
    import webbrowser
except ImportError, e:
    print '[daeRunExamples]: Cannot load ui modules', str(e)

try:
    import whats_the_time, tutorial1, tutorial2, tutorial3, tutorial4, tutorial5, tutorial6, tutorial7, tutorial8, tutorial9, tutorial10
except ImportError, e:
    print '[daeRunExamples]: Cannot load tutorial modules', str(e)


class daeTextEditLog(daeStdOutLog):
    def __init__(self, TextEdit, App):
        daeStdOutLog.__init__(self)
        self.TextEdit = TextEdit
        self.App      = App

    def Message(self, message, severity):
        self.TextEdit.append(message)
        if self.TextEdit.isVisible() == True:
            self.TextEdit.update()
        self.App.processEvents()

class WebView(QtGui.QDialog):
    def __init__(self, url):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_WebViewDialog()
        self.ui.setupUi(self)
        self.ui.webView.load(url)

class RunExamples(QtGui.QDialog):
    def __init__(self, app):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_RunExamplesDialog()
        self.ui.setupUi(self)
        self.app = app
        
        self.connect(self.ui.toolButtonRun,                QtCore.SIGNAL('clicked()'), self.slotRunTutorial)
        self.connect(self.ui.toolButtonCode,               QtCore.SIGNAL('clicked()'), self.slotShowCode)
        self.connect(self.ui.toolButtonModelReport,        QtCore.SIGNAL('clicked()'), self.slotShowModelReport)
        self.connect(self.ui.toolButtonRuntimeModelReport, QtCore.SIGNAL('clicked()'), self.slotShowRuntimeModelReport)
        
    #@QtCore.pyqtSlot()
    def slotShowCode(self):
        simName = str(self.ui.comboBoxExample.currentText())
        if simName == "tutorial1":
            url   = QtCore.QUrl("tutorial1.html")
            title = "tutorial1.py"
        elif simName == "tutorial2":
            url   = QtCore.QUrl("tutorial2.html")
            title = "tutorial2.py"
        elif simName == "tutorial3":
            url   = QtCore.QUrl("tutorial3.html")
            title = "tutorial3.py"
        elif simName == "tutorial4":
            url   = QtCore.QUrl("tutorial4.html")
            title = "tutorial4.py"
        elif simName == "tutorial5":
            url   = QtCore.QUrl("tutorial5.html")
            title = "tutorial5.py"
        elif simName == "tutorial6":
            url   = QtCore.QUrl("tutorial6.html")
            title = "tutorial6.py"
        elif simName == "tutorial7":
            url   = QtCore.QUrl("tutorial7.html")
            title = "tutorial7.py"
        elif simName == "tutorial8":
            url   = QtCore.QUrl("tutorial8.html")
            title = "tutorial8.py"
        elif simName == "tutorial9":
            url   = QtCore.QUrl("tutorial9.html")
            title = "tutorial9.py"
        elif simName == "tutorial10":
            url   = QtCore.QUrl("tutorial10.html")
            title = "tutorial10.py"
        else:
            url   = QtCore.QUrl("whats_the_time.html")
            title = "whats_the_time.py"
        wv = WebView(url)
        wv.setWindowTitle(title)
        wv.exec_()
    
    #@QtCore.pyqtSlot()
    def slotShowModelReport(self):
        simName = str(self.ui.comboBoxExample.currentText())
        if simName == "tutorial1":
            url   = "Tutorial_1.xml"
        elif simName == "tutorial2":
            url   = "Tutorial_2.xml"
        elif simName == "tutorial3":
            url   = "Tutorial_3.xml"
        elif simName == "tutorial4":
            url   = "Tutorial_4.xml"
        elif simName == "tutorial5":
            url   = "Tutorial_5.xml"
        elif simName == "tutorial6":
            url   = "Tutorial_6.xml"
        elif simName == "tutorial7":
            url   = "Tutorial_7.xml"
        elif simName == "tutorial8":
            url   = "Tutorial_8.xml"
        elif simName == "tutorial9":
            url   = "Tutorial_9.xml"
        elif simName == "tutorial10":
            url   = "Tutorial_10.xml"
        else:
            url   = "WhatsTheTime.xml"
        webbrowser.open_new(url)

    #@QtCore.pyqtSlot()
    def slotShowRuntimeModelReport(self):
        simName = str(self.ui.comboBoxExample.currentText())
        if simName == "tutorial1":
            url   = "Tutorial_1-rt.xml"
        elif simName == "tutorial2":
            url   = "Tutorial_2-rt.xml"
        elif simName == "tutorial3":
            url   = "Tutorial_3-rt.xml"
        elif simName == "tutorial4":
            url   = "Tutorial_4-rt.xml"
        elif simName == "tutorial5":
            url   = "Tutorial_5-rt.xml"
        elif simName == "tutorial6":
            url   = "Tutorial_6-rt.xml"
        elif simName == "tutorial7":
            url   = "Tutorial_7-rt.xml"
        elif simName == "tutorial8":
            url   = "Tutorial_8-rt.xml"
        elif simName == "tutorial9":
            url   = "Tutorial_9-rt.xml"
        elif simName == "tutorial10":
            url   = "Tutorial_10-rt.xml"
        else:
            url   = "WhatsTheTime-rt.xml"
        webbrowser.open_new(url)

    #@QtCore.pyqtSlot()
    def slotRunTutorial(self):
        simName = str(self.ui.comboBoxExample.currentText())
        TimeHorizon       = 1000
        ReportingInterval = 10
        if simName == "tutorial1":
            simulation = tutorial1.simTutorial()
        elif simName == "tutorial2":
            simulation = tutorial2.simTutorial()
        elif simName == "tutorial3":
            simulation = tutorial3.simTutorial()
            TimeHorizon       = 200
            ReportingInterval = 5
        elif simName == "tutorial4":
            simulation = tutorial4.simTutorial()
            TimeHorizon       = 500
            ReportingInterval = 10
        elif simName == "tutorial5":
            simulation = tutorial5.simTutorial()
            TimeHorizon       = 500
            ReportingInterval = 2
        elif simName == "tutorial6":
            simulation = tutorial6.simTutorial()
            TimeHorizon       = 100
            ReportingInterval = 10
        elif simName == "tutorial7":
            simulation = tutorial7.simTutorial()
            TimeHorizon       = 500
            ReportingInterval = 10
        elif simName == "tutorial8":
            simulation = tutorial8.simTutorial()
            TimeHorizon       = 100
            ReportingInterval = 10
        elif simName == "tutorial9":
            simulation = tutorial9.simTutorial()
        elif simName == "tutorial10":
            simulation = tutorial10.simTutorial()
        else:
            simulation = whats_the_time.simTutorial()
            TimeHorizon       = 500
            ReportingInterval = 10
        
        simulation.ReportingInterval = ReportingInterval
        simulation.TimeHorizon       = TimeHorizon
        simulation.m.SetReportingOn(True)
        
        try:
            if simName == "tutorial8":
                filename = str(QtGui.QFileDialog.getSaveFileName(None, "Insert the file name for text data reporter", "tutorial8.out", "Text files (*.txt)"))
                if filename == "":
                    return
                datareporter = daeDelegateDataReporter()
                dr1 = tutorial8.MyDataReporter()
                dr2 = daeTCPIPDataReporter()
                datareporter.AddDataReporter(dr1)
                datareporter.AddDataReporter(dr2)
                simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
                if(dr1.Connect(filename, simName) == False):
                    QtGui.QMessageBox.warning(None, "DAE Tools Examples", "Cannot open tutorial8.out for writing")
                    raise RuntimeError("Cannot connect MyDataReporter")
                if(dr2.Connect("", simName) == False):
                    QtGui.QMessageBox.warning(None, "DAE Tools Examples", "Cannot connect data reporter!\nDid you forget to start daePlotter?")
                    raise RuntimeError("Cannot connect daeTCPIPDataReporter")
                simulator = daeSimulator(self.app, simulation, datareporter)
                simulator.exec_()
                dr1.Write()
                QtGui.QMessageBox.information(None, "Tutorial 8", "Now check daePlotter and file: [" + filename + "] for the results.\nThey should both contain the same data.")
            else:
                simulator = daeSimulator(self.app, simulation)
                simulator.exec_()
        except RuntimeError:
            pass
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = RunExamples(app)
    main.show()
    app.exec_()
