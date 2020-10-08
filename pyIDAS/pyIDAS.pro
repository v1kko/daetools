include(../dae.pri)
QT -= core \
    gui
TARGET = pyIDAS
TEMPLATE = lib
CONFIG += shared plugin

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${SUNDIALS_INCLUDE} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                $${SUNDIALS_LIBDIR}

LIBS += $${SOLIBS_RPATH}

LIBS +=	$${DAE_IDAS_SOLVER_LIB} \
        $${DAE_UNITS_LIB} \
        $${DAE_CONFIG_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${BLAS_LAPACK_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    docstrings.h \
    python_wraps.h

#######################################################
#                Install files
#######################################################
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${PYDAE_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

include(../dae_install_py_module.pri)
QMAKE_CXXFLAGS += -fpermissive
