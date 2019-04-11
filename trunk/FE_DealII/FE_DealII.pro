include(../dae.pri)
QT -= core gui
TARGET = cdaeDealII_FESolver
TEMPLATE = lib
CONFIG += staticlib

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${NUMPY_INCLUDE_DIR} \
               $${DEALII_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                $${DEALII_LIB_DIR}

unix::QMAKE_CXXFLAGS  += -std=c++11
unix::QMAKE_LFLAGS    += -std=c++11
win32::QMAKE_CXXFLAGS += /std:c++11
win32::QMAKE_LFLAGS   += /std:c++11

LIBS += $${DEALII_LIBS} \
        $${DAE_CORE_LIB} \
        $${DAE_UNITS_LIB} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${RT}

SOURCES += stdafx.cpp \
           dllmain.cpp

HEADERS += stdafx.h \
           dealii_common.h \
           dealii_cell_context.h \
           dealii_template_inst.h \
           dealii_datareporter.h \
           dealii_omp_work_stream.h \
           dealii_fe_system.h

#######################################################
#                Install files
#######################################################
