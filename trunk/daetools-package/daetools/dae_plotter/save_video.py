"""********************************************************************************
                             save_video.py
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
********************************************************************************"""
import sys, numpy
from os.path import join, realpath, dirname
from PyQt4 import QtCore, QtGui
from daetools.pyDAE import *
from .save_video_ui import Ui_SaveVideo

images_dir = join(dirname(__file__), 'images')

class daeSavePlot2DVideo(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_SaveVideo()
        self.ui.setupUi(self)

        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)
        self.connect(self.ui.buttonFilename, QtCore.SIGNAL("clicked()"), self.slotOpenFilename)

    def slotOpenFilename(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save video/sequence of images", self.ui.lineeditFilename.text(),
                                                     "Videos (*.avi *.flv *.mp4 *.mpg *.ogv *.webm *.wmv);;Images (*.png *.jpeg *.tiff *.bmp);;All Files (*)")
        if filename:
            self.ui.lineeditFilename.setText(filename)

