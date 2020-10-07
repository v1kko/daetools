include(../dae.pri)
QT -= core \
    gui
TARGET = pyOpenCS
TEMPLATE = lib
CONFIG += shared plugin

unix::QMAKE_CXXFLAGS  += -std=c++11
unix::QMAKE_LFLAGS    += -std=c++11

OPEN_CS_DIR     = ../OpenCS/build
OPEN_CS_INCLUDE = $${OPEN_CS_DIR}/include
OPEN_CS_LIBS    = -L$${OPEN_CS_DIR}/lib -lOpenCS_Evaluators -lOpenCS_Models -lOpenCS_Simulators

unix::MPI_LIBS  = -lmpi_cxx -lmpi

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${OPEN_CS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${SOLIBS_RPATH}

LIBS +=	$${OPEN_CS_LIBS} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}

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
#                  $${SOLVERS_DIR}/$${TARGET}.$${PYTHON_EXTENSION_MODULE_EXT}

include(../dae_install_py_module.pri)
