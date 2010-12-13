include(../dae.pri)
QT -= core \
    gui
TARGET = Solver
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
    $${SUNDIALS_INCLUDE}
SOURCES += stdafx.cpp \
    ida_solver.cpp \
    daesolver.cpp \
    class_factory.cpp \
    base_solvers.cpp
HEADERS += stdafx.h \
    solver_class_factory.h \
    ida_solver.h \
    base_solvers.h \
    dae_array_matrix.h \
    dae_solvers.h \
    ida_la_solver_interface.h

SOURCES += $${SUNDIALS}/src/ida/ida_sptfqmr.c \
    $${SUNDIALS}/src/ida/ida_spils.c \
    $${SUNDIALS}/src/ida/ida_spgmr.c \
    $${SUNDIALS}/src/ida/ida_spbcgs.c \
    $${SUNDIALS}/src/ida/ida_io.c \
    $${SUNDIALS}/src/ida/ida_ic.c \
    $${SUNDIALS}/src/ida/ida_direct.c \
    $${SUNDIALS}/src/ida/ida_dense.c \
    $${SUNDIALS}/src/ida/ida_lapack.c \
    $${SUNDIALS}/src/ida/ida_bbdpre.c \
    $${SUNDIALS}/src/ida/ida_band.c \
    $${SUNDIALS}/src/ida/ida.c \
    $${SUNDIALS}/src/sundials/sundials_direct.c \
    $${SUNDIALS}/src/sundials/sundials_iterative.c \
    $${SUNDIALS}/src/sundials/sundials_dense.c \
    $${SUNDIALS}/src/sundials/sundials_band.c \
    $${SUNDIALS}/src/sundials/sundials_sptfqmr.c \
    $${SUNDIALS}/src/sundials/sundials_spgmr.c \
    $${SUNDIALS}/src/sundials/sundials_spbcgs.c \
    $${SUNDIALS}/src/sundials/sundials_nvector.c \
    $${SUNDIALS}/src/sundials/sundials_math.c \
    $${SUNDIALS}/src/nvec_ser/nvector_serial.c
HEADERS += $${SUNDIALS}/src/ida/ida_spils_impl.h \
    $${SUNDIALS}/src/ida/ida_impl.h \
    $${SUNDIALS}/src/ida/ida_direct_impl.h \
    $${SUNDIALS}/src/ida/ida_bbdpre_impl.h \
    $${SUNDIALS}/include/ida/ida_spils.h \
    $${SUNDIALS}/include/ida/ida_spgmr.h \
    $${SUNDIALS}/include/ida/ida_spbcgs.h \
    $${SUNDIALS}/include/ida/ida_dense.h \
    $${SUNDIALS}/include/ida/ida_lapack.h \
    $${SUNDIALS}/include/ida/ida_direct.h \
    $${SUNDIALS}/include/ida/ida_bbdpre.h \
    $${SUNDIALS}/include/ida/ida_band.h \
    $${SUNDIALS}/include/ida/ida.h \
    $${SUNDIALS}/include/ida/ida_sptfqmr.h \
    $${SUNDIALS}/include/sundials/sundials_types.h \
    $${SUNDIALS}/include/sundials/sundials_direct.h \
    $${SUNDIALS}/include/sundials/sundials_dense.h \
    $${SUNDIALS}/include/sundials/sundials_config.h \
    $${SUNDIALS}/include/sundials/sundials_band.h \
    $${SUNDIALS}/include/sundials/sundials_iterative.h \
    $${SUNDIALS}/include/sundials/sundials_lapack.h \
    $${SUNDIALS}/include/sundials/sundials_math.h \
    $${SUNDIALS}/include/sundials/sundials_nvector.h \
    $${SUNDIALS}/include/sundials/sundials_spbcgs.h \
    $${SUNDIALS}/include/sundials/sundials_spgmr.h \
    $${SUNDIALS}/include/sundials/sundials_sptfqmr.h
