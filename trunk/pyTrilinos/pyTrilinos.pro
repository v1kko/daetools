#************************************************************************************
#                 DAE Tools Project: www.daetools.com
#                 Copyright (C) Dragan Nikolic, 2013
#************************************************************************************
# DAE Tools is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 3 as published by the Free Software
# Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with the
# DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
#************************************************************************************

include(../dae.pri)
QT -= core gui
TARGET  = pyTrilinos
TEMPLATE = lib
CONFIG += shared plugin

unix::QMAKE_CXXFLAGS += -std=c++11 

INCLUDEPATH +=  $${BOOSTDIR} \
				$${PYTHON_INCLUDE_DIR} \
				$${PYTHON_SITE_PACKAGES_DIR} \
				$${SUNDIALS_INCLUDE} \
				$${TRILINOS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${SOLIBS_RPATH}
LIBS += $${DAE_TRILINOS_SOLVER_LIB} \
        $${DAE_CONFIG_LIB} \
        $${TRILINOS_LIBS} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${BLAS_LAPACK_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp

HEADERS += stdafx.h \
    docstrings.h

#######################################################
#                Install files
#######################################################
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${SOLVERS_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# Rename libpyModule.so into pyModule.so
install_rename_module.commands = $${MOVE_FILE} \
                                 $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                                 $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}
QMAKE_EXTRA_TARGETS += install_rename_module

# Install into daetools-dev
install_python_module.depends += install_rename_module
install_python_module.path     = $${DAE_INSTALL_PY_MODULES_DIR}
install_python_module.files    = $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

# Install into daetools-package
install_python_module2.depends += install_rename_module
install_python_module2.path     = $${SOLVERS_DIR}
install_python_module2.files    = $${DAE_DEST_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

INSTALLS += install_python_module install_python_module2
