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

import os, platform, sys, subprocess, webbrowser, traceback, importlib
from os.path import join, realpath, dirname
from time import localtime, strftime
from os.path import join, realpath, dirname
python_major = sys.version_info[0]
if python_major == 2:
    from StringIO import StringIO
elif python_major == 3:
    from io import StringIO
    
try:
    from PyQt4 import QtCore, QtGui
except Exception as e:
    print('[daeRunExamples]: Cannot load PyQt4 modules\n Error: ', str(e))
    sys.exit()

try:
    import numpy
except Exception as e:
    print('[daeRunExamples]: Cannot load numpy module\n Error: ', str(e))
    sys.exit()

try:
    from daetools.pyDAE import *
except Exception as e:
    print('[daeRunExamples]: Cannot load daetools.pyDAE module\n Error: ', str(e))
    sys.exit()

try:
    from .RunExamples_ui import Ui_RunExamplesDialog
    from daetools.pyDAE.web_view_dialog import daeWebView
except Exception as e:
    print('[daeRunExamples]: Cannot load UI modules\n Error: ', str(e))

tutorial_modules = []
tutorial_modules.append(('whats_the_time', []))
for i in range(1, 20):
    tutorial_modules.append(('tutorial%d' % i, []))
for i in range(1, 3):
    tutorial_modules.append(('tutorial_dealii_%d' % i, []))
for i in range(1, 8):
    tutorial_modules.append(('opt_tutorial%d' % i, []))

for m_name, data  in tutorial_modules:
    try:
        module = __import__(m_name, globals(), locals(), [], -1)
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
        
        self.setWindowTitle("DAE Tools Tutorials v" + daeVersion(True))
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
    
