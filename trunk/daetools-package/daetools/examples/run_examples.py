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

import os, sys, subprocess, webbrowser, traceback, importlib
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

print(tutorial_modules)
"""
try:
    from . import tutorial7, tutorial8, tutorial9, tutorial10, tutorial11, tutorial12, tutorial13
except Exception as e:
    exc_traceback = sys.exc_info()[2]
    print('\n'.join(traceback.format_tb(exc_traceback)))
    print('[daeRunExamples]: Cannot load Tutorials modules\n Error: ', str(e))

try:
    from . import tutorial14, tutorial15, tutorial16, tutorial17, tutorial18, tutorial19
except Exception as e:
    exc_traceback = sys.exc_info()[2]
    print('\n'.join(traceback.format_tb(exc_traceback)))
    print('[daeRunExamples]: Cannot load Tutorials modules\n Error: ', str(e))

try:
    from . import opt_tutorial1
except Exception as e:
    exc_traceback = sys.exc_info()[2]
    print('\n'.join(traceback.format_tb(exc_traceback)))
    print('[daeRunExamples]: Cannot load opt_tutorial1 module\n Error: ', str(e))

try:
    from . import opt_tutorial2
except Exception as e:
    exc_traceback = sys.exc_info()[2]
    print('\n'.join(traceback.format_tb(exc_traceback)))
    print('[daeRunExamples]: Cannot load opt_tutorial2 module\n Error: ', str(e))

try:
    from . import opt_tutorial3
except Exception as e:
    exc_traceback = sys.exc_info()[2]
    print('\n'.join(traceback.format_tb(exc_traceback)))
    print('[daeRunExamples]: Cannot load opt_tutorial3 module\n Error: ', str(e))

try:
    from . import opt_tutorial4
except Exception as e:
    exc_traceback = sys.exc_info()[2]
    print('\n'.join(traceback.format_tb(exc_traceback)))
    print('[daeRunExamples]: Cannot load opt_tutorial4 module\n Error: ', str(e))

try:
    from . import opt_tutorial5
except Exception as e:
    exc_traceback = sys.exc_info()[2]
    print('\n'.join(traceback.format_tb(exc_traceback)))
    print('[daeRunExamples]: Cannot load opt_tutorial5 module\n Error: ', str(e))

try:
    from . import opt_tutorial6
except Exception as e:
    exc_traceback = sys.exc_info()[2]
    print('\n'.join(traceback.format_tb(exc_traceback)))
    print('[daeRunExamples]: Cannot load opt_tutorial6 module\n Error: ', str(e))

try:
    from . import opt_tutorial7
except Exception as e:
    exc_traceback = sys.exc_info()[2]
    print('\n'.join(traceback.format_tb(exc_traceback)))
    print('[daeRunExamples]: Cannot load opt_tutorial7 module\n Error: ', str(e))
"""

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

        for m_name, data  in tutorial_modules:
            if len(data) == 2:
                module = data[0]
                doc    = data[1]
                if module:
                    self.ui.comboBoxExample.addItem(m_name, (module, doc))
        """
        if 'whats_the_time' in sys.modules:
            self.ui.comboBoxExample.addItem("whats_the_time", whats_the_time)
        if 'tutorial1' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial1", tutorial1)
        if 'tutorial2' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial2", tutorial2)
        if 'tutorial3' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial3", tutorial3)
        if 'tutorial4' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial4", tutorial4)
        if 'tutorial5' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial5", tutorial5)
        if 'tutorial6' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial6", tutorial6)
        if 'tutorial7' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial7", tutorial7)
        if 'tutorial8' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial8", tutorial8)
        if 'tutorial9' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial9", tutorial9)
        if 'tutorial10' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial10", tutorial10)
        if 'tutorial11' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial11", tutorial11)
        if 'tutorial12' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial12", tutorial12)
        if 'tutorial13' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial13", tutorial13)
        if 'tutorial14' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial14", tutorial14)
        if 'tutorial15' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial15", tutorial15)
        if 'tutorial16' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial16", tutorial16)
        if 'tutorial17' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial17", tutorial17)
        if 'tutorial18' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial18", tutorial18)
        if 'tutorial19' in sys.modules:
            self.ui.comboBoxExample.addItem("tutorial19", tutorial19)
        if 'opt_tutorial1' in sys.modules:
            self.ui.comboBoxExample.addItem("opt_tutorial1", opt_tutorial1)
        if 'opt_tutorial2' in sys.modules:
            self.ui.comboBoxExample.addItem("opt_tutorial2", opt_tutorial2)
        if 'opt_tutorial3' in sys.modules:
            self.ui.comboBoxExample.addItem("opt_tutorial3", opt_tutorial3)
        if 'opt_tutorial4' in sys.modules:
            self.ui.comboBoxExample.addItem("opt_tutorial4", opt_tutorial4)
        if 'opt_tutorial5' in sys.modules:
            self.ui.comboBoxExample.addItem("opt_tutorial5", opt_tutorial5)
        if 'opt_tutorial6' in sys.modules:
            self.ui.comboBoxExample.addItem("opt_tutorial6", opt_tutorial6)
        if 'opt_tutorial7' in sys.modules:
            self.ui.comboBoxExample.addItem("opt_tutorial7", opt_tutorial7)
        """
        
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

            """
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
            """
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
    
