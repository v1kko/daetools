#!/usr/bin/env python

import sys
from time import localtime, strftime

try:
    from PyQt4 import QtCore, QtGui
except Exception, e:
    print '[daePlotter]: Cannot load PyQt4 modules\n Error: ', str(e)
    sys.exit()

try:
    import numpy
except Exception, e:
    print '[daePlotter]: Cannot load numpy module\n Error: ', str(e)
    sys.exit()

try:
    from daetools.pyDAE import *
except Exception, e:
    print '[daePlotter]: Cannot load daetools.pyDAE module\n Error: ', str(e)
    sys.exit()

try:
    from RunExamples_ui import Ui_RunExamplesDialog
    from daetools.pyDAE.WebViewDialog import WebView
    #from daetools.WebView_ui import Ui_WebViewDialog
    import webbrowser
except Exception, e:
    print '[daePlotter]: Cannot load UI modules\n Error: ', str(e)
    sys.exit()

try:
    import whats_the_time, tutorial1, tutorial2, tutorial3, tutorial4, tutorial5, tutorial6, tutorial7, tutorial8, tutorial9, tutorial10
    import opt_tutorial1, opt_tutorial2
except Exception, e:
    print '[daePlotter]: Cannot load Tutorials modules\n Error: ', str(e)


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

#class WebView(QtGui.QDialog):
#    def __init__(self, url):
#        QtGui.QDialog.__init__(self)
#        self.ui = Ui_WebViewDialog()
#        self.ui.setupUi(self)
#        self.ui.webView.load(url)

class RunExamples(QtGui.QDialog):
    def __init__(self, app):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_RunExamplesDialog()
        self.ui.setupUi(self)
        self.app = app
        
        self.setWindowTitle("DAE Tools Tutorials v" + daeVersion(True))

        self.connect(self.ui.toolButtonRun,                QtCore.SIGNAL('clicked()'), self.slotRunTutorial)
        self.connect(self.ui.toolButtonCode,               QtCore.SIGNAL('clicked()'), self.slotShowCode)
        self.connect(self.ui.toolButtonModelReport,        QtCore.SIGNAL('clicked()'), self.slotShowModelReport)
        self.connect(self.ui.toolButtonRuntimeModelReport, QtCore.SIGNAL('clicked()'), self.slotShowRuntimeModelReport)
        
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(0, "whats_the_time")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(1, "tutorial1")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(2, "tutorial2")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(3, "tutorial3")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(4, "tutorial4")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(5, "tutorial5")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(6, "tutorial6")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(7, "tutorial7")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(8, "tutorial8")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(9, "tutorial9")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(10, "tutorial10")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(11, "opt_tutorial1")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(12, "opt_tutorial2")
        
    #@QtCore.pyqtSlot()
    def slotShowCode(self):
        simName = str(self.ui.comboBoxExample.currentText())
        url   = QtCore.QUrl(simName + ".html")
        title = simName + ".py"
        wv = WebView(url)
        wv.setWindowTitle(title)
        wv.exec_()
    
    #@QtCore.pyqtSlot()
    def slotShowModelReport(self):
        simName = str(self.ui.comboBoxExample.currentText())
        url     = simName + ".xml"
        webbrowser.open_new(url)

    #@QtCore.pyqtSlot()
    def slotShowRuntimeModelReport(self):
        simName = str(self.ui.comboBoxExample.currentText())
        url     = simName + "-rt.xml"
        webbrowser.open_new(url)

    #@QtCore.pyqtSlot()
    def slotRunTutorial(self):
        simName = str(self.ui.comboBoxExample.currentText())
        #TimeHorizon       = 1000
        #ReportingInterval = 10
        
        #simulation   = None
        #optimization = None
        #if simName == "tutorial1":
            #simulation = tutorial1.simTutorial()
        #elif simName == "tutorial2":
            #simulation = tutorial2.simTutorial()
        #elif simName == "tutorial3":
            #simulation = tutorial3.simTutorial()
            #TimeHorizon       = 200
            #ReportingInterval = 5
        #elif simName == "tutorial4":
            #simulation = tutorial4.simTutorial()
            #TimeHorizon       = 500
            #ReportingInterval = 10
        #elif simName == "tutorial5":
            #simulation = tutorial5.simTutorial()
            #TimeHorizon       = 500
            #ReportingInterval = 2
        #elif simName == "tutorial6":
            #simulation = tutorial6.simTutorial()
            #TimeHorizon       = 100
            #ReportingInterval = 10
        #elif simName == "tutorial7":
            #simulation = tutorial7.simTutorial()
            #TimeHorizon       = 500
            #ReportingInterval = 10
        #elif simName == "tutorial8":
            #simulation = tutorial8.simTutorial()
            #TimeHorizon       = 100
            #ReportingInterval = 10
        #elif simName == "tutorial9":
            #simulation = tutorial9.simTutorial()
        #elif simName == "tutorial10":
            #simulation = tutorial10.simTutorial()
        #elif simName == "opt_tutorial1":
            #simulation   = opt_tutorial1.simTutorial()
            #optimization = daeOptimization()
            #TimeHorizon       = 1
            #ReportingInterval = 1
        #else:
            #simulation = whats_the_time.simTutorial()
            #TimeHorizon       = 500
            #ReportingInterval = 10
        
        #simulation.ReportingInterval = ReportingInterval
        #simulation.TimeHorizon       = TimeHorizon
        #simulation.m.SetReportingOn(True)
        
        try:
            if simName == "tutorial1":
                tutorial1.guiRun(self.app)
            elif simName == "tutorial2":
                tutorial2.guiRun(self.app)
            elif simName == "tutorial3":
                tutorial3.guiRun(self.app)
            elif simName == "tutorial4":
                tutorial4.guiRun(self.app)
            elif simName == "tutorial5":
                tutorial5.guiRun(self.app)
            elif simName == "tutorial6":
                tutorial6.guiRun(self.app)
            elif simName == "tutorial7":
                tutorial7.guiRun(self.app)
            elif simName == "tutorial8":
                tutorial8.guiRun(self.app)
            elif simName == "tutorial9":
                tutorial9.guiRun(self.app)
            elif simName == "tutorial10":
                tutorial10.guiRun(self.app)
            elif simName == "tutorial11":
                tutorial11.guiRun(self.app)
            elif simName == "tutorial1":
                opt_tutorial1.guiRun(self.app)
            elif simName == "tutorial2":
                opt_tutorial2.guiRun(self.app)
            elif simName == "whats_the_time":
                whats_the_time.guiRun(self.app)
            elif simName == "opt_tutorial1":
                opt_tutorial1.guiRun(self.app)
            elif simName == "opt_tutorial2":
                opt_tutorial2.guiRun(self.app)
            else:
                pass
        
            #if simName == "tutorial8":
                #filename = str(QtGui.QFileDialog.getSaveFileName(None, "Insert the file name for text data reporter", "tutorial8.out", "Text files (*.txt)"))
                #if filename == "":
                    #return
                #datareporter = daeDelegateDataReporter()
                #dr1 = tutorial8.MyDataReporter()
                #dr2 = daeTCPIPDataReporter()
                #datareporter.AddDataReporter(dr1)
                #datareporter.AddDataReporter(dr2)
                #simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
                #if(dr1.Connect(filename, simName) == False):
                    #QtGui.QMessageBox.warning(None, "DAE Tools Examples", "Cannot open tutorial8.out for writing")
                    #raise RuntimeError("Cannot connect MyDataReporter")
                #if(dr2.Connect("", simName) == False):
                    #QtGui.QMessageBox.warning(None, "DAE Tools Examples", "Cannot connect data reporter!\nDid you forget to start daePlotter?")
                    #raise RuntimeError("Cannot connect daeTCPIPDataReporter")
                #simulator = daeSimulator(self.app, simulation, datareporter)
                #simulator.exec_()
                #dr1.Write()
                #QtGui.QMessageBox.information(None, "Tutorial 8", "Now check daePlotter and file: [" + filename + "] for the results.\nThey should both contain the same data.")
            #elif simName == "opt_tutorial1":
                #simulator = daeSimulator(self.app, simulation=simulation, optimization=optimization)
                #simulator.exec_()
            #elif simName == "tutorial1":
                #simulator = daeSimulator(self.app, simulation=simulation, optimization=optimization)
                #simulator.exec_()
            #else:
                #simulator = daeSimulator(self.app, simulation=simulation)
                #simulator.exec_()
        except RuntimeError, e:
            print "Exception: ", str(e)
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = RunExamples(app)
    main.show()
    app.exec_()
