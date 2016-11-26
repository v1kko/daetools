#!/usr/bin/env python
"""
***********************************************************************************
                           run_examples.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2016
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

import os, platform, sys, subprocess, webbrowser, traceback, numpy
from os.path import join, realpath, dirname
from time import localtime, strftime
from os.path import join, realpath, dirname
from PyQt4 import QtCore, QtGui

python_major = sys.version_info[0]
python_minor = sys.version_info[1]
python_build = sys.version_info[2]
if python_major == 2:
    from StringIO import StringIO
elif python_major == 3:
    from io import StringIO
    
from daetools.pyDAE import *
from .RunExamples_ui import Ui_RunExamplesDialog

tutorial_modules = []
tutorial_modules.append(('whats_the_time', []))
for i in range(1, 19):
    tutorial_modules.append(('tutorial%d' % i, []))
for i in range(1, 5):
    tutorial_modules.append(('tutorial_adv_%d' % i, []))
for i in range(1, 9):
    tutorial_modules.append(('tutorial_che_%d' % i, []))
for i in range(1, 7):
    tutorial_modules.append(('tutorial_dealii_%d' % i, []))
for i in range(1, 8):
    tutorial_modules.append(('opt_tutorial%d' % i, []))

for m_name, data  in tutorial_modules:
    try:
        import importlib
        module = importlib.import_module('.'+m_name, 'daetools.examples')
        #module = __import__(m_name, globals(), locals(), [])
        doc    = module.__doc__
        data.append(module)
        data.append(doc)
    except Exception as e:
        exc_traceback = sys.exc_info()[2]
        print('\n'.join(traceback.format_tb(exc_traceback)))
        print('[daeRunExamples]: Cannot load Tutorials modules\n Error: ', str(e))

try:
    _examples_dir = dirname(__file__)
except:
    # In case we are running the module on its own (i.e. as __main__)
    _examples_dir = realpath(dirname(sys.argv[0]))

try:
    # Try to get Firefox webbrowser instance
    # If can't find it - get the default one
    firefox = webbrowser.get('firefox')
except:
    firefox = webbrowser.get()

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

        font = QtGui.QFont()
        if platform.system() == 'Linux':
            font.setFamily("Monospace")
            font.setPointSize(9)
        elif platform.system() == 'Darwin':
            font.setFamily("Monaco")
            font.setPointSize(10)
        else:
            font.setFamily("Courier New")
            font.setPointSize(9)
        self.ui.docstringEdit.setFont(font)
        
        self.setWindowTitle("DAE Tools Examples v%s [py%d.%d]" % (daeVersion(True), python_major, python_minor))
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.connect(self.ui.toolButtonRun,                QtCore.SIGNAL('clicked()'), self.slotRunTutorial)
        self.connect(self.ui.toolButtonCode,               QtCore.SIGNAL('clicked()'), self.slotShowCode)
        self.connect(self.ui.toolButtonModelReport,        QtCore.SIGNAL('clicked()'), self.slotShowModelReport)
        self.connect(self.ui.toolButtonRuntimeModelReport, QtCore.SIGNAL('clicked()'), self.slotShowRuntimeModelReport)
        self.connect(self.ui.comboBoxExample,              QtCore.SIGNAL("currentIndexChanged(int)"), self.slotTutorialChanged)

        for m_name, data  in tutorial_modules:
            if len(data) == 2:
                module = data[0]
                doc    = data[1]
                if module:
                    self.ui.comboBoxExample.addItem(m_name, (module, doc))
        
    def slotTutorialChanged(self, index):
        data = self.ui.comboBoxExample.itemData(index)
        if isinstance(data, QtCore.QVariant):
            module, doc = data.toPyObject()
        else:
            module, doc = data
        self.ui.docstringEdit.setText(str(doc))

    #@QtCore.pyqtSlot()
    def slotShowCode(self):
        simName = str(self.ui.comboBoxExample.currentText())
        address = join(_examples_dir, simName + ".html")
        try:
            from daetools.pyDAE.web_view_dialog import daeWebView
            url   = QtCore.QUrl(address)
            title = simName + ".py"
            wv = daeWebView(url)
            wv.setWindowTitle(title)
            wv.exec_()
        except Exception as e:
            firefox.open_new_tab(address)
            
    #@QtCore.pyqtSlot()
    def slotShowModelReport(self):
        simName = str(self.ui.comboBoxExample.currentText())
        url     = join(_examples_dir, simName + ".xml")
        firefox.open_new_tab(url)

    #@QtCore.pyqtSlot()
    def slotShowRuntimeModelReport(self):
        simName = str(self.ui.comboBoxExample.currentText())
        url     = join(_examples_dir, simName + "-rt.xml")
        firefox.open_new_tab(url)

    #@QtCore.pyqtSlot()
    def slotRunTutorial(self):
        m_name = str(self.ui.comboBoxExample.currentText())
        data   = self.ui.comboBoxExample.itemData(self.ui.comboBoxExample.currentIndex())
        if isinstance(data, QtCore.QVariant):
            module, doc = data.toPyObject()
        else:
            module, doc = data
        
        try:
            if m_name in ["opt_tutorial4", "opt_tutorial5", "opt_tutorial6"]:
                self.consoleRunAndShowResults(module)
            else:
                module.guiRun(self.app)

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
                from daetools.pyDAE.web_view_dialog import daeWebView
                view = daeWebView(message)
                view.resize(700, 500)
                view.setWindowTitle('Console execution results')
                view.exec_()
            except Exception as e:
                f, path = tempfile.mkstemp(suffix='.html', prefix='')
                f.write(message)
                f.close()
                firefox.open_new_tab(path)
        except:
            sys.stdout = saveout

def daeRunExamples():
    app = QtGui.QApplication(sys.argv)
    main = RunExamples(app)
    main.show()
    app.exec_()

if __name__ == "__main__":
    daeRunExamples()
    
