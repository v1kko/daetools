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
TARGET = daetools_simulation
TEMPLATE = app

# MPI Settings
QMAKE_CXX = mpicxx
QMAKE_CXX_RELEASE = $$QMAKE_CXX
QMAKE_CXX_DEBUG = $$QMAKE_CXX
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC = mpicc

QMAKE_CFLAGS += $$system(mpicc --showme:compile)
QMAKE_LFLAGS += $$system(mpicxx --showme:link)
QMAKE_CXXFLAGS += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK
QMAKE_CXXFLAGS_RELEASE += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK

unix::QMAKE_CXXFLAGS += -std=c++11

unix::QMAKE_CFLAGS_RELEASE   -= -O2
unix::QMAKE_CXXFLAGS_RELEASE -= -O2

unix::QMAKE_CFLAGS           += -std=c99 -pedantic
unix::QMAKE_CFLAGS_WARN_ON    = -Wall -Wextra \
                                -Wno-unused-parameter \
                                -Wno-unused-variable \
                                -Wno-unused-but-set-variable
unix::QMAKE_CXXFLAGS_WARN_ON  = -Wall -Wextra \
                                -Wno-unused-parameter \
                                -Wno-unused-variable \
                                -Wno-unused-but-set-variable

INCLUDEPATH  += ../../idas/build/include
QMAKE_LIBDIR += ../../idas/build/lib

LIBS += -lsundials_idas -lsundials_nvecparallel -lblas -llapack -lboost_mpi -lboost_serialization

SOURCES += auxiliary.cpp adouble.cpp daesolver.cpp simulation.cpp model.cpp main.cpp
HEADERS += typedefs.h auxiliary.h adouble.h daesolver.h simulation.h mpi_sync.h runtime_information.h model.h
