#!/usr/bin/env python
"""
***********************************************************************************
                           run_examples.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2013
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************
"""

import os, sys, subprocess, webbrowser, traceback
from os.path import join, realpath, dirname
from StringIO import StringIO
from time import localtime, strftime
from os.path import join, realpath, dirname

try:
    from PyQt4 import QtCore, QtGui
except Exception, e:
    print '[daeRunExamples]: Cannot load PyQt4 modules\n Error: ', str(e)
    sys.exit()

try:
    import numpy
except Exception, e:
    print '[daeRunExamples]: Cannot load numpy module\n Error: ', str(e)
    sys.exit()

try:
    from daetools.pyDAE import *
except Exception, e:
    print '[daeRunExamples]: Cannot load daetools.pyDAE module\n Error: ', str(e)
    sys.exit()

try:
    from RunExamples_ui import Ui_RunExamplesDialog
    from daetools.pyDAE.web_view_dialog import daeWebView
except Exception, e:
    print '[daeRunExamples]: Cannot load UI modules\n Error: ', str(e)

try:
    import whats_the_time, tutorial1, tutorial2, tutorial3, tutorial4, tutorial5, tutorial6
    import tutorial7, tutorial8, tutorial9, tutorial10, tutorial11, tutorial12, tutorial13
    import tutorial14, tutorial15, tutorial16, tutorial17, tutorial18, tutorial19
except Exception, e:
    exc_traceback = sys.exc_info()[2]
    print '\n'.join(traceback.format_tb(exc_traceback))
    print '[daeRunExamples]: Cannot load Tutorials modules\n Error: ', str(e)

try:
    import opt_tutorial1
except Exception, e:
    exc_traceback = sys.exc_info()[2]
    print '\n'.join(traceback.format_tb(exc_traceback))
    print '[daeRunExamples]: Cannot load opt_tutorial1 module\n Error: ', str(e)

try:
    import opt_tutorial2
except Exception, e:
    exc_traceback = sys.exc_info()[2]
    print '\n'.join(traceback.format_tb(exc_traceback))
    print '[daeRunExamples]: Cannot load opt_tutorial2 module\n Error: ', str(e)

try:
    import opt_tutorial3
except Exception, e:
    exc_traceback = sys.exc_info()[2]
    print '\n'.join(traceback.format_tb(exc_traceback))
    print '[daeRunExamples]: Cannot load opt_tutorial3 module\n Error: ', str(e)

try:
    import opt_tutorial4
except Exception, e:
    exc_traceback = sys.exc_info()[2]
    print '\n'.join(traceback.format_tb(exc_traceback))
    print '[daeRunExamples]: Cannot load opt_tutorial4 module\n Error: ', str(e)

try:
    import opt_tutorial5
except Exception, e:
    exc_traceback = sys.exc_info()[2]
    print '\n'.join(traceback.format_tb(exc_traceback))
    print '[daeRunExamples]: Cannot load opt_tutorial5 module\n Error: ', str(e)

try:
    import opt_tutorial6
except Exception, e:
    exc_traceback = sys.exc_info()[2]
    print '\n'.join(traceback.format_tb(exc_traceback))
    print '[daeRunExamples]: Cannot load opt_tutorial6 module\n Error: ', str(e)

try:
    import opt_tutorial7
except Exception, e:
    exc_traceback = sys.exc_info()[2]
    print '\n'.join(traceback.format_tb(exc_traceback))
    print '[daeRunExamples]: Cannot load opt_tutorial7 module\n Error: ', str(e)

try:
    _examples_dir = dirname(__file__)
except:
    # In case we are running the module on its own (i.e. as __main__)
    _examples_dir = realpath(dirname(sys.argv[0]))

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

try:
    images_dir = dirname(__file__)
except:
    # In case we are running the module on its own (i.e. as __main__)
    images_dir = realpath(dirname(sys.argv[0]))

class RunExamples(QtGui.QDialog):
    def __init__(self, app):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_RunExamplesDialog()
        self.ui.setupUi(self)
        self.app = app

        self.setWindowTitle("DAE Tools Tutorials v" + daeVersion(True))
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.connect(self.ui.toolButtonRun,                QtCore.SIGNAL('clicked()'), self.slotRunTutorial)
        self.connect(self.ui.toolButtonCode,               QtCore.SIGNAL('clicked()'), self.slotShowCode)
        self.connect(self.ui.toolButtonModelReport,        QtCore.SIGNAL('clicked()'), self.slotShowModelReport)
        self.connect(self.ui.toolButtonRuntimeModelReport, QtCore.SIGNAL('clicked()'), self.slotShowRuntimeModelReport)
        self.connect(self.ui.comboBoxExample,              QtCore.SIGNAL("currentIndexChanged(int)"), self.slotTutorialChanged)

        self.ui.comboBoxExample.addItem("whats_the_time", QtCore.QVariant(whats_the_time))
        self.ui.comboBoxExample.addItem("tutorial1", QtCore.QVariant(tutorial1))
        self.ui.comboBoxExample.addItem("tutorial2", QtCore.QVariant(tutorial2))
        self.ui.comboBoxExample.addItem("tutorial3", QtCore.QVariant(tutorial3))
        self.ui.comboBoxExample.addItem("tutorial4", QtCore.QVariant(tutorial4))
        self.ui.comboBoxExample.addItem("tutorial5", QtCore.QVariant(tutorial5))
        self.ui.comboBoxExample.addItem("tutorial6", QtCore.QVariant(tutorial6))
        self.ui.comboBoxExample.addItem("tutorial7", QtCore.QVariant(tutorial7))
        self.ui.comboBoxExample.addItem("tutorial8", QtCore.QVariant(tutorial8))
        self.ui.comboBoxExample.addItem("tutorial9", QtCore.QVariant(tutorial9))
        self.ui.comboBoxExample.addItem("tutorial10", QtCore.QVariant(tutorial10))
        self.ui.comboBoxExample.addItem("tutorial11", QtCore.QVariant(tutorial11))
        self.ui.comboBoxExample.addItem("tutorial12", QtCore.QVariant(tutorial12))
        self.ui.comboBoxExample.addItem("tutorial13", QtCore.QVariant(tutorial13))
        self.ui.comboBoxExample.addItem("tutorial14", QtCore.QVariant(tutorial14))
        self.ui.comboBoxExample.addItem("tutorial15", QtCore.QVariant(tutorial15))
        self.ui.comboBoxExample.addItem("tutorial16", QtCore.QVariant(tutorial16))
        self.ui.comboBoxExample.addItem("tutorial17", QtCore.QVariant(tutorial17))
        self.ui.comboBoxExample.addItem("tutorial18", QtCore.QVariant(tutorial18))
        self.ui.comboBoxExample.addItem("tutorial19", QtCore.QVariant(tutorial19))
        self.ui.comboBoxExample.addItem("opt_tutorial1", QtCore.QVariant(opt_tutorial1))
        self.ui.comboBoxExample.addItem("opt_tutorial2", QtCore.QVariant(opt_tutorial2))
        self.ui.comboBoxExample.addItem("opt_tutorial3", QtCore.QVariant(opt_tutorial3))
        self.ui.comboBoxExample.addItem("opt_tutorial4", QtCore.QVariant(opt_tutorial4))
        self.ui.comboBoxExample.addItem("opt_tutorial5", QtCore.QVariant(opt_tutorial5))
        self.ui.comboBoxExample.addItem("opt_tutorial6", QtCore.QVariant(opt_tutorial6))
        self.ui.comboBoxExample.addItem("opt_tutorial7", QtCore.QVariant(opt_tutorial7))

    def slotTutorialChanged(self, index):
        module = self.ui.comboBoxExample.itemData(index).toPyObject()
        self.ui.docstringEdit.setText(module.__doc__)

    #@QtCore.pyqtSlot()
    def slotShowCode(self):
        simName = str(self.ui.comboBoxExample.currentText())
        try:
            url   = QtCore.QUrl(join(_examples_dir, simName + ".html"))
            title = simName + ".py"
            wv = daeWebView(url)
            wv.setWindowTitle(title)
            wv.exec_()
        except Exception as e:
            webbrowser.open_new_tab(simName + ".html")
            
    #@QtCore.pyqtSlot()
    def slotShowModelReport(self):
        simName = str(self.ui.comboBoxExample.currentText())
        url     = join(_examples_dir, simName + ".xml")
        webbrowser.open_new_tab(url)

    #@QtCore.pyqtSlot()
    def slotShowRuntimeModelReport(self):
        simName = str(self.ui.comboBoxExample.currentText())
        url     = join(_examples_dir, simName + "-rt.xml")
        webbrowser.open_new_tab(url)

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
            elif simName == "tutorial13":
                tutorial13.guiRun(self.app)
            elif simName == "tutorial14":
                tutorial14.guiRun(self.app)
            elif simName == "tutorial15":
                tutorial15.guiRun(self.app)
            elif simName == "tutorial16":
                tutorial16.guiRun(self.app)
            elif simName == "tutorial17":
                tutorial17.guiRun(self.app)
            elif simName == "tutorial18":
                tutorial18.guiRun(self.app)
            elif simName == "tutorial19":
                tutorial19.guiRun(self.app)
            elif simName == "whats_the_time":
                whats_the_time.guiRun(self.app)
            elif simName == "opt_tutorial1":
                opt_tutorial1.guiRun(self.app)
            elif simName == "opt_tutorial2":
                opt_tutorial2.guiRun(self.app)
            elif simName == "opt_tutorial3":
                opt_tutorial3.guiRun(self.app)
            elif simName == "opt_tutorial4":
                self.consoleRunAndShowResults(opt_tutorial4)
            elif simName == "opt_tutorial5":
                self.consoleRunAndShowResults(opt_tutorial5)
            elif simName == "opt_tutorial6":
                self.consoleRunAndShowResults(opt_tutorial6)
            elif simName == "opt_tutorial7":
                opt_tutorial7.guiRun(self.app)
            else:
                pass
        
        except RuntimeError as e:
            QtGui.QMessageBox.warning(self, "daeRunExamples", "Exception raised: " + str(e))

    def consoleRunAndShowResults(self, module):
        try:
            output = StringIO()
            saveout = sys.stdout
            sys.stdout = output
            module.run()
            sys.stdout = saveout
            message = '<pre>{0}</pre>'.format(output.getvalue())
            try:
                view = daeWebView(message)
                view.resize(700, 500)
                view.setWindowTitle('Console execution results')
                view.exec_()
            except Exception as e:
                f, path = tempfile.mkstemp(suffix='.html', prefix='')
                f.write(message)
                f.close()
                webbrowser.open_new_tab(path)
        except:
            sys.stdout = saveout

def daeRunExamples():
    app = QtGui.QApplication(sys.argv)
    main = RunExamples(app)
    main.show()
    app.exec_()

if __name__ == "__main__":
    daeRunExamples()
	