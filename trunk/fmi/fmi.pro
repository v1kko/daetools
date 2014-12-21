include(../dae.pri)

QT -= core
QT -= gui

TEMPLATE = lib
# Add "plugin" to avoid creation of symlinks
CONFIG += shared plugin

TARGET = cdaeFMU_CS-py$${PYTHON_MAJOR}$${PYTHON_MINOR}

unix::QMAKE_CXXFLAGS += -fvisibility=hidden

SOURCES += dllmain.cpp \
           daetools_fmi_cs.cpp

HEADERS += stdafx.h \
           daetools_fmi_cs.h

INCLUDEPATH += $${BOOSTDIR}

LIBS += $${DAE_SIMULATION_LOADER_LIB}

# Achtung, Achtung!!
# It uses daetools/solibs for linking
QMAKE_LIBDIR += $${SOLIBS_DIR}

QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                  $${FMI_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}.$${SHARED_LIB_EXT}
