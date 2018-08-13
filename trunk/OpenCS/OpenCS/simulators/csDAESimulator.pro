#************************************************************************************
#                 OpenCS Project: www.daetools.com
#                 Copyright (C) Dragan Nikolic
#************************************************************************************
# OpenCS is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 3 as published by the Free Software
# Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with the
# OpenCS software; if not, see <http://www.gnu.org/licenses/>.
#************************************************************************************
QT -= core gui
TARGET = csDAESimulator
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

# Library paths
BOOST_DIR       = ../../boost
IDAS_DIR        = ../../idas
IDAS_INCLUDE    = $${IDAS_DIR}/build/include
IDAS_LIB_DIR    = $${IDAS_DIR}/build/lib

CVODES_DIR        = ../../cvodes
CVODES_INCLUDE    = $${CVODES_DIR}/build/include
CVODES_LIB_DIR    = $${CVODES_DIR}/build/lib

OPEN_CS_DIR     = ../..
OPEN_CS_INCLUDE = $${OPEN_CS_DIR}
OPEN_CS_LIBS    = -L$${OPEN_CS_DIR}/lib -lOpenCS_Simulators -lOpenCS_Models -lOpenCS_Evaluators

# Trilinos related libraries
TRILINOS_DIR    = ../../trilinos/build
BLAS_LAPACK_LIBS = -llapack -lblas -lgfortran -lm
TRILINOS_INCLUDE = ../../trilinos/build/include
TRILINOS_LIBS    = -L$${TRILINOS_DIR}/lib -L$${SUPERLU_PATH}/lib \
                           -lml -lifpack -laztecoo \
                           -lamesos -lepetraext -ltriutils -lepetra \
                           -lteuchosremainder -lteuchosnumerics -lteuchoscomm \
                           -lteuchosparameterlist -lteuchoscore \
                            $${UMFPACK_LIBS} \
                            $${SUPERLU_LIBS}

# OpenCL
unix::INTEL_OPENCL_DIR     = /opt/intel/opencl
unix::INTEL_OPENCL_INCLUDE = $${INTEL_OPENCL_DIR}/include
unix::INTEL_OPENCL_LIBS    = -L$${INTEL_OPENCL_DIR} -lOpenCL

unix::SOLIBS_RPATH = -Wl,-rpath,\'\$$ORIGIN\',-z,origin

INCLUDEPATH  += $${OPEN_CS_INCLUDE} \
                $${BOOST_DIR} \
                $${CVODES_DIR} \
                $${CVODES_INCLUDE} \
                $${IDAS_DIR} \
                $${IDAS_INCLUDE} \
                $${TRILINOS_INCLUDE} \
                $${INTEL_OPENCL_INCLUDE}

#QMAKE_LIBDIR += $${IDAS_LIB_DIR}

LIBS += $${SOLIBS_RPATH}
LIBS += $${OPEN_CS_LIBS}

#LIBS += $${INTEL_OPENCL_LIBS}
#LIBS += -L../release -lcdaeTrilinos_LASolver
#LIBS += $${TRILINOS_LIBS} $${UMFPACK_LIBS} $${BLAS_LAPACK_LIBS}
# IDAS + BOOST libraries
#LIBS += -lsundials_idas -lsundials_nvecparallel
#LIBS += -lboost_mpi -lboost_serialization -lboost_system

LIBS += -lgomp -lmpi -lstdc++fs

HEADERS += ../../examples/advection_diffusion_2d.h \
           ../../examples/brusselator_2d.h \
           ../../examples/burgers_equations_2d.h \
           ../../examples/chemical_kinetics.h \
           ../../examples/diurnal_kinetics_2d.h \
           ../../examples/roberts.h \
           ../../examples/heat_conduction_2d.h

SOURCES += cs_dae_simulator.cpp \
           auxiliary.cpp \
           config.cpp \
           odeisolver.cpp \
           daesolver.cpp \
           daemodel.cpp \
           daesimulation.cpp \
           preconditioner_jacobi.cpp \
           preconditioner_ifpack.cpp \
           preconditioner_ml.cpp \
           cs_simulators.cpp \
           cs_ode_simulator.cpp \
           cs_dae_simulator.cpp \
           ../models/cs_dae_model.cpp \
           ../models/cs_model_builder.cpp \
           ../models/cs_model_io.cpp \
           ../models/cs_nodes.cpp \
           ../models/cs_number.cpp \
           ../models/cs_partitioner_metis.cpp \
           ../models/cs_partitioner_simple.cpp \
           ../evaluators/cs_evaluator_opencl_factory.cpp \
           ../evaluators/cs_evaluator_opencl_multidevice.cpp \
           ../evaluators/cs_evaluator_opencl.cpp \
           ../evaluators/cs_evaluator_openmp.cpp \
           ../evaluators/cs_evaluator_sequential.cpp \
           ../../examples/dae_example_1.cpp \
           ../../examples/dae_example_2.cpp \
           ../../examples/dae_example_3.cpp \
           ../../examples/ode_example_1.cpp \
           ../../examples/ode_example_2.cpp \
           ../../examples/ode_example_3.cpp

OTHER_FILES += ../evaluators/cs_machine_kernels.cl
