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

unix::QMAKE_CXXFLAGS += -std=c++11 -fopenmp

unix::QMAKE_CFLAGS_RELEASE   -= -O2
unix::QMAKE_CFLAGS_RELEASE   -= -O2
unix::QMAKE_CFLAGS_RELEASE   += -O3
unix::QMAKE_CXXFLAGS_RELEASE += -O3

unix::QMAKE_CFLAGS           += -std=c99
unix::QMAKE_CFLAGS_WARN_ON    = -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-sign-compare
unix::QMAKE_CXXFLAGS_WARN_ON  = -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-sign-compare

# Valgrind profilig
#unix::QMAKE_CXXFLAGS += -g

# Path to daetools/trunk directory
BOOST_DIR       = ../boost
IDAS_DIR        = ../idas-mpi

INCLUDEPATH  += $${BOOST_DIR} \
                $${IDAS_DIR}/build/include \
                ../trilinos/build/include

QMAKE_LIBDIR += $${BOOST_DIR}/stage/lib \
                $${IDAS_DIR}/build/lib

# Trilinos related libraries
UMFPACK_LIBPATH = ../umfpack/build/lib
TRILINOS_DIR    = ../trilinos/build
SUPERLU_PATH    = ../superlu/build
SUPERLU_LIBPATH = $${SUPERLU_PATH}/lib
BLAS_LAPACK_LIBDIR = ../lapack/lib
BLAS_LAPACK_LIBS = $${BLAS_LAPACK_LIBDIR}/liblapack.a $${BLAS_LAPACK_LIBDIR}/libblas.a -lgfortran -lm
SUPERLU_LIBS = -L$${SUPERLU_LIBPATH} -lsuperlu
UMFPACK_LIBS = $${UMFPACK_LIBPATH}/libumfpack.a \
               $${UMFPACK_LIBPATH}/libcholmod.a \
               $${UMFPACK_LIBPATH}/libamd.a \
               $${UMFPACK_LIBPATH}/libcamd.a \
               $${UMFPACK_LIBPATH}/libcolamd.a \
               $${UMFPACK_LIBPATH}/libccolamd.a \
               $${UMFPACK_LIBPATH}/libsuitesparseconfig.a
TRILINOS_LIBS = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
                           -laztecoo -lml -lifpack \
                           -lamesos -lepetraext -ltriutils -lepetra \
                           -lteuchosremainder -lteuchosnumerics -lteuchoscomm \
                           -lteuchosparameterlist -lteuchoscore \
                            $${UMFPACK_LIBS} \
                            $${SUPERLU_LIBS}
LIBS += -L../release -lcdaeTrilinos_LASolver $${TRILINOS_LIBS} $${UMFPACK_LIBS} $${BLAS_LAPACK_LIBS}

# IDAS + BOOST libraries
LIBS += -lsundials_idas -lsundials_nvecparallel -lboost_mpi -lboost_serialization -lboost_filesystem -lboost_system

LIBS += -lgomp

SOURCES += auxiliary.cpp \
           config.cpp \
           daesolver.cpp \
           simulation.cpp \
           model.cpp \
           compute_stack_openmp.cpp \
           preconditioner_jacobi.cpp \
           preconditioner_ifpack.cpp \
           preconditioner_ml.cpp \
           lasolver.cpp \
           main.cpp

HEADERS += typedefs.h \
           auxiliary.h \
           runtime_information.h \
           compute_stack.h \
           idas_la_functions.h \
           compute_stack_openmp.h
