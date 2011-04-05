#!/usr/bin/env python
"""********************************************************************************
                             daeRunExamples.py
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
    import whats_the_time, tutorial1, tutorial2, tutorial3, tutorial4, tutorial5, tutorial6, tutorial7, tutorial8, tutorial9, tutorial10, tutorial11, tutorial12
    import opt_tutorial1, opt_tutorial2, opt_tutorial3
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
        self.ui.comboBoxExample.setItemText(11, "tutorial11")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(12, "tutorial12")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(13, "opt_tutorial1")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(14, "opt_tutorial2")
        self.ui.comboBoxExample.addItem("")
        self.ui.comboBoxExample.setItemText(15, "opt_tutorial3")

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
            elif simName == "tutorial12":
                tutorial12.guiRun(self.app)
            elif simName == "whats_the_time":
                whats_the_time.guiRun(self.app)
            elif simName == "opt_tutorial1":
                opt_tutorial1.guiRun(self.app)
            elif simName == "opt_tutorial2":
                opt_tutorial2.guiRun(self.app)
            elif simName == "opt_tutorial3":
                opt_tutorial3.guiRun(self.app)
            else:
                pass
        except RuntimeError, e:
            print "Exception: ", str(e)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = RunExamples(app)
    main.show()
    app.exec_()
