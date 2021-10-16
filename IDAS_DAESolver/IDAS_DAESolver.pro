include(../dae.pri)
QT -= core \
    gui
TARGET = cdaeIDAS_DAESolver$${SHARED_LIB_POSTFIX}
TEMPLATE = lib
CONFIG += shared plugin

INCLUDEPATH += $${BOOSTDIR} \
               $${SUNDIALS_INCLUDE}

win32::QMAKE_CXXFLAGS += -DIDAS_EXPORTS

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR}

LIBS += $${SOLIBS_RPATH_SL}
LIBS +=	$${DAE_UNITS_LIB} \
        $${DAE_CONFIG_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${BLAS_LAPACK_LIBS}

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

#######################################################
#                Install files
#######################################################
QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}.$${SHARED_LIB_EXT} \
                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}.$${SHARED_LIB_EXT}

include(../dae_install_library.pri)
