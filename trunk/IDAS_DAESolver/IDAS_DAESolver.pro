include(../dae.pri)
QT -= core \
    gui
TARGET = cdaeIDAS_DAESolver
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
               $${SUNDIALS_INCLUDE} \
               $${MPI_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR}
LIBS +=	$${SUNDIALS_LIBS} \
        $${MPI_LIBS}

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

