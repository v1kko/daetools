#************************************************************************************
#                 DAE Tools Project: www.daetools.com
#                 Copyright (C) Dragan Nikolic, 2010
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
QT -= core \
    gui
TARGET = TrilinosAmesos
TEMPLATE = lib

####################################################################################
#                       Suppress some warnings
####################################################################################
#unix::QMAKE_CXXFLAGS += -ansi -pedantic
unix::QMAKE_CXXFLAGS += -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable
unix::QMAKE_CFLAGS   += -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable

# ####################################################################################
#                       TRILINOS Amesos solvers
# ####################################################################################
SUPERLU_DIR      = ../superlu
TRILINOS_DIR     = ../trilinos/build
TRILINOS_INCLUDE = $${TRILINOS_DIR}/include

win32-msvc2008::BLAS_LAPACK_LIBDIR = ../clapack/LIB/Win32
linux-g++::BLAS_LAPACK_LIBDIR      = /usr/lib/atlas
linux-g++-64::BLAS_LAPACK_LIBDIR   = /usr/lib/atlas

win32-msvc2008::TRILINOS_LIBS = -L$${TRILINOS_DIR}/lib -L$${BLAS_LAPACK_LIBDIR} -L$${SUPERLU_DIR}/lib \
                                BLAS_nowrap.lib clapack_nowrap.lib libf2c.lib \
                                superlu.lib \
                                amesos.lib epetra.lib teuchos.lib aztecoo.lib

linux-g++::TRILINOS_LIBS  =   -L$${TRILINOS_DIR}/lib -L$${BLAS_LAPACK_LIBDIR} -L$${SUPERLU_DIR}/lib \
							  -lblas -llapack \
							  -lcdaesuperlu \
							  -lumfpack -lamd \
							  -laztecoo -lamesos -lepetra -lepetraext -lteuchos

linux-g++-64::TRILINOS_LIBS = -L$${TRILINOS_DIR}/lib -L$${BLAS_LAPACK_LIBDIR} -L$${SUPERLU_DIR}/lib \
							  -lblas -llapack \
							  -lcdaesuperlu \
							  -lumfpack -lamd \
							  -laztecoo -lml -lifpack -lamesos -lepetra -lepetraext -lteuchos

INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR} \
    $${SUNDIALS_INCLUDE} \
    $${TRILINOS_INCLUDE}
QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS += $${BOOST_PYTHON_LIB} \
    $${BLAS_LIBS} \
    $${LAPACK_LIBS} \
    $${TRILINOS_LIBS}
SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    trilinos_amesos_la_solver.cpp
HEADERS += stdafx.h \
    trilinos_amesos_la_solver.h
win32-msvc2008::QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/TrilinosAmesos1.dll \
    $${DAE_DEST_DIR}/pyTrilinosAmesos.pyd
win32-g++::QMAKE_POST_LINK = move \
    /y \
    $${DAE_DEST_DIR}/TrilinosAmesos1.dll \
    $${DAE_DEST_DIR}/pyTrilinosAmesos.pyd
unix::QMAKE_POST_LINK = cp \
    -f \
    $${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
    $${DAE_DEST_DIR}/pyTrilinosAmesos.so
