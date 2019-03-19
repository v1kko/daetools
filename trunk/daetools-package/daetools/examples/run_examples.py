#!/usr/bin/env python
"""
***********************************************************************************
                           run_examples.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic
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
from PyQt5 import QtCore, QtGui, QtWidgets

python_major = sys.version_info[0]
python_minor = sys.version_info[1]
python_build = sys.version_info[2]
if python_major == 2:
    from StringIO import StringIO
elif python_major == 3:
    from io import StringIO
    
from daetools.pyDAE import *
try:
    # Uses WebKit
    from .RunExamples_ui import Ui_RunExamplesDialog
except:
    # Uses WebEngine (for MacOS, no webkit there)
    from .RunExamples_ui_webengine import Ui_RunExamplesDialog
    
tutorial_modules = []
tutorial_modules.append(('whats_the_time', []))
for i in range(1, 21 + 1):
    tutorial_modules.append(('tutorial%d' % i, []))
for i in range(1, 4 + 1):
    tutorial_modules.append(('tutorial_adv_%d' % i, []))
for i in range(1, 11 + 1):
    tutorial_modules.append(('tutorial_cv_%d' % i, []))
for i in range(1, 9 + 1):
    tutorial_modules.append(('tutorial_che_%d' % i, []))
for i in range(1, 3 + 1):
    tutorial_modules.append(('tutorial_sa_%d' % i, []))
for i in range(1, 3 + 1):
    tutorial_modules.append(('tutorial_opencs_dae_%d' % i, []))
for i in range(1, 3 + 1):
    tutorial_modules.append(('tutorial_opencs_ode_%d' % i, []))
for i in range(1, 6 + 1):
    tutorial_modules.append(('tutorial_che_opt_%d' % i, []))
for i in range(1, 8 + 1):
    tutorial_modules.append(('tutorial_dealii_%d' % i, []))
for i in range(1, 7 + 1):
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
        print('[daeRunExamples]: Cannot load Tutorial module %s\n Error: %s' % (m_name, str(e)))
        print('\n'.join(traceback.format_tb(exc_traceback)))
        print('------------------------------------------------------------')

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

class RunExamples(QtWidgets.QDialog):
    def __init__(self, app):
        QtWidgets.QDialog.__init__(self)
        self.ui = Ui_RunExamplesDialog()
        self.ui.setupUi(self)
        self.app = app

        sg = app.desktop().screenGeometry()
        screenHeight = sg.height()
        screenWidth  = sg.width()
        #print("The screen resolution is %d x %d" % (screenWidth, screenHeight))
        if screenWidth <= 1024:
            self.formWidth = screenWidth - 100
        else:
            self.formWidth = 850
        if screenHeight <= 1024:
            self.formHeight = screenHeight - 100
        else:
            self.formHeight = 900
        self.resize(self.formWidth, self.formHeight)
        
        try:
            from PyQt5 import QtWebKit
            settings = QtWebKit.QWebSettings.globalSettings()
            font = QtGui.QFont()
            if platform.system() == 'Linux':
                font.setFamily("Monospace")
                font.setPointSize(9)
                settings.setFontFamily(QtWebKit.QWebSettings.FixedFont, 'Monospace')
                settings.setFontSize(QtWebKit.QWebSettings.DefaultFontSize, 9)
            elif platform.system() == 'Darwin':
                font.setFamily("Monaco")
                font.setPointSize(10)
                settings.setFontFamily(QtWebKit.QWebSettings.FixedFont, 'Monaco')
                settings.setFontSize(QtWebKit.QWebSettings.DefaultFontSize, 10)
            else:
                font.setFamily("Courier New")
                font.setPointSize(9)
                settings.setFontFamily(QtWebKit.QWebSettings.FixedFont, 'Courier New')
                settings.setFontSize(QtWebKit.QWebSettings.DefaultFontSize, 10)
            #self.ui.docstringWeb.setFont(font)
        except:
            pass
        
        self.setWindowTitle("DAE Tools Examples v%s [py%d.%d]" % (daeVersion(True), python_major, python_minor))
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.ui.toolButtonRun.clicked.connect(self.slotRunTutorial)
        self.ui.toolButtonCode.clicked.connect(self.slotShowCode)
        self.ui.toolButtonModelReport.clicked.connect(self.slotShowModelReport)
        self.ui.toolButtonRuntimeModelReport.clicked.connect(self.slotShowRuntimeModelReport)
        self.ui.comboBoxExample.currentIndexChanged.connect(self.slotTutorialChanged)

        for m_name, data  in tutorial_modules:
            if len(data) == 2:
                module = data[0]
                doc    = data[1]
                if module:
                    self.ui.comboBoxExample.addItem(m_name, (module, doc))
        
        self.setTutorialLink("all-tutorials")
        
    def setTutorialLink(self, module_name):
        address = join(_examples_dir, "..", "docs", "html", "tutorials-all.html#%s" % module_name)
        address = "file:///" + os.path.normpath(address)
        address = address.replace('\\', '/')
        # The url contains bookmarks (i.e. tutorials-all.html#tutorial1.html) - can't use QUrl.fromLocalFile()
        
        # Perhaps it is better to use the file from the daetools website so the docs do not need to be included
        #address = 'http://www.daetools.com/docs-%s/tutorials-all.html#%s' % (daeVersion(True), module_name)
        #address = 'http://www.daetools.com/docs/tutorials-all.html#%s' % module_name
        url = QtCore.QUrl(address)
        #print(url)
        self.ui.docstringWeb.load(url)
        self.ui.docstringWeb.show()
        
    #@QtCore.pyqtSlot()
    def slotTutorialChanged(self, index):
        data = self.ui.comboBoxExample.itemData(index)
        if isinstance(data, QtCore.QVariant):
            module, doc = data.toPyObject()
        else:
            module, doc = data
        module_name = module.__name__.split('.')[-1]
        module_name = module_name.replace(' ', '-')
        module_name = module_name.replace('_', '-')
        try:
            self.setTutorialLink(module_name)
        except:
            self.ui.docstringWeb.setHtml("<code><pre>%s</pre></code>" % str(doc))
            self.ui.docstringWeb.show()

    #@QtCore.pyqtSlot()
    def slotShowCode(self):
        simName = str(self.ui.comboBoxExample.currentText())
        address = join(_examples_dir, simName + ".html")
        address = os.path.normpath(address)
        try:
            from daetools.pyDAE.web_view_dialog import daeWebView
            url   = QtCore.QUrl.fromLocalFile(address)
            title = simName + ".py"
            wv = daeWebView(url)
            wv.setWindowTitle(title)
            wv.resize(self.formWidth, self.formHeight)
            wv.exec_()
        except Exception as e:
            print(str(e))
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
            if m_name in ["tutorial_che_9", "opt_tutorial4", "opt_tutorial5", "opt_tutorial6",
                          "tutorial_cv_1", "tutorial_cv_2", "tutorial_cv_3", "tutorial_cv_4", 
                          "tutorial_cv_5", "tutorial_cv_6", "tutorial_cv_6", "tutorial_cv_7",
                          "tutorial_cv_8", "tutorial_cv_9", "tutorial_cv_10", "tutorial_cv_11",
                          "tutorial_sa_1", "tutorial_sa_2", "tutorial_sa_3",
                          "tutorial_opencs_dae_1", "tutorial_opencs_dae_2", "tutorial_opencs_dae_3"
                          "tutorial_opencs_ode_1", "tutorial_opencs_ode_2", "tutorial_opencs_ode_3"]:
                self.consoleRunAndShowResults(module)
            else:
                module.run(guiRun = True, qtApp = self.app)
        except RuntimeError as e:
            QtGui.QMessageBox.warning(self, "daeRunExamples", "Exception raised: " + str(e))

    def consoleRunAndShowResults(self, module):
        try:
            output = StringIO()
            saveout = sys.stdout
            sys.stdout = output
            #import multiprocessing
            #t = multiprocessing.Process(target=module.run, args=(True, ))
            #t.start()
            #t.join()
            module.run(guiRun = False, qtApp = self.app)
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
        except Exception as e:
            print(str(e))
            sys.stdout = saveout

def daeRunExamples():
    app = QtWidgets.QApplication(sys.argv)
    main = RunExamples(app)
    main.show()
    app.exec_()

if __name__ == "__main__":
    daeRunExamples()
    
