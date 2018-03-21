#************************************************************************************
#                 DAE Tools Project: www.daetools.com
#                 Copyright (C) Dragan Nikolic
#************************************************************************************
# DAE Tools is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 3 as published by the Free Software
# Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with the
# DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
#************************************************************************************
QT -= core gui
TARGET = daetools_mpi_simulator
TEMPLATE = app

CONFIG(debug, debug|release) {
    OBJECTS_DIR = debug
}

CONFIG(release, debug|release) {
    OBJECTS_DIR = release
}
DESTDIR = bin

# MPI Settings
QMAKE_CXX  = mpicxx
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC   = mpicc

#unix::QMAKE_CFLAGS   += $$system(mpicc --showme:compile)
#unix::QMAKE_LFLAGS   += $$system(mpicxx --showme:link)
#unix::QMAKE_CXXFLAGS += $$system(mpicxx --showme:compile)

unix::QMAKE_CXXFLAGS += -std=c++11

unix::QMAKE_CFLAGS_RELEASE   -= -O2
unix::QMAKE_CXXFLAGS_RELEASE -= -O2

unix::QMAKE_CFLAGS           += -std=c99
unix::QMAKE_CFLAGS_WARN_ON    = -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable
unix::QMAKE_CXXFLAGS_WARN_ON  = -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable

# Path to daetools/trunk directory
BOOST_DIR = ../boost
IDAS_DIR  = ../idas

INCLUDEPATH  += $${BOOST_DIR} \
                $${IDAS_DIR}/build/include

QMAKE_LIBDIR += $${BOOST_DIR}/stage/lib \
                $${IDAS_DIR}/build/lib

LIBS += -lsundials_idas -lsundials_nvecparallel -lboost_mpi -lboost_serialization -lboost_filesystem -lboost_system

SOURCES += auxiliary.cpp \
           daesolver.cpp \
           simulation.cpp \
           model.cpp \
           compute_stack_openmp.cpp \
           lasolver.cpp \
           main.cpp

HEADERS += typedefs.h \
           auxiliary.h \
           daesolver.h \
           simulation.h \
           runtime_information.h \
           model.h \
           compute_stack.h \
           lasolver.h \
           compute_stack_openmp.h
