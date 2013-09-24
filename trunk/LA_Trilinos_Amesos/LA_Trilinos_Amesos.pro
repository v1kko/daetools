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
TARGET  = cdaeTrilinos_LASolver
TEMPLATE = lib
CONFIG += staticlib

INCLUDEPATH +=  $${BOOSTDIR} \
				$${PYTHON_INCLUDE_DIR} \
				$${PYTHON_SITE_PACKAGES_DIR} \
				$${SUNDIALS_INCLUDE} \
				$${TRILINOS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${BLAS_LIBS} \
		$${LAPACK_LIBS} \
		$${TRILINOS_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    trilinos_amesos_la_solver.cpp

HEADERS += stdafx.h \
    base_solvers.h \
    trilinos_amesos_la_solver.h

#######################################################
#                Install files
#######################################################
#win32{
#QMAKE_POST_LINK = copy /y  $${TARGET}.lib $${STATIC_LIBS_DIR}
#}

#unix{
#QMAKE_POST_LINK = cp -f lib$${TARGET}.a $${STATIC_LIBS_DIR}
#}

trilinos_headers.path  = $${HEADERS_DIR}/LA_SuperLU
trilinos_headers.files = base_solvers.h

trilinos_libs.path         = $${STATIC_LIBS_DIR}
win32::trilinos_libs.files = $${DAE_DEST_DIR}/$${TARGET}.lib
unix::trilinos_libs.files  = $${DAE_DEST_DIR}/lib$${TARGET}.a

INSTALLS += trilinos_headers trilinos_libs
