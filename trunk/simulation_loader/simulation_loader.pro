include(../dae.pri)

QT -= core
QT -= gui

TEMPLATE = lib
# Add "plugin" to avoid creation of symlinks
CONFIG += shared plugin

TARGET  = cdaeSimulationLoader-py$${PYTHON_MAJOR}$${PYTHON_MINOR}

SOURCES += dllmain.cpp \
           simulation_loader.cpp \
           simulation_loader_c.cpp

HEADERS += stdafx.h \
           simulation_loader.h \
           simulation_loader_c.h \
           simulation_loader_common.h

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${SOLIBS_RPATH}

LIBS += $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}

QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}.$${SHARED_LIB_EXT}
