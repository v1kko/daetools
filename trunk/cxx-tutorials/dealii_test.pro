include(../dae.pri)
QT -= gui core
TARGET = dealii_test
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app
DESTDIR = bin
INCLUDEPATH += $${BOOSTDIR}

dealii_dir = ../deal.II
tbb_dir    = ../deal.II/contrib/tbb/tbb30_104oss

INCLUDEPATH    += $${dealii_dir}/include $${tbb_dir}/include
QMAKE_LIBDIR   += $${dealii_dir}/lib
LIBS           += -ldeal_II -ltbb -lz -lblas -lgfortran
QMAKE_CXXFLAGS += -std=c++0x

#QMAKE_CXXFLAGS += -DHAVE_CONFIG_H -DHAVE_ISNAN -ggdb  -DBOOST_NO_HASH -DBOOST_NO_SLIST -DDEBUG -Wall -W -Wpointer-arith -Wwrite-strings -Wsynth -Wsign-compare -Wswitch -Wno-long-long -std=c++0x -Wa,--compress-debug-sections -pthread -D_REENTRANT -fPIC

LIBS += $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_SUPERLU_SOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${DAE_UNITS_LIB} \
        $${SUPERLU_LIBS} \
        $${BLAS_LAPACK_LIBS} \
        $${RT} -lbtf

SOURCES += step-4.cpp fem_common.cpp dealii_transient_diffusion_test.cpp

HEADERS += fem_common.h dealii_transient_diffusion.h
