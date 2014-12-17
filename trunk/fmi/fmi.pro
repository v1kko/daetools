include(../dae.pri)

QT -= core
QT -= gui

TEMPLATE = lib
CONFIG += shared

TARGET = daetools_fmi_cs-$${DAE_SYSTEM}_$${DAE_MACHINE}

SOURCES += dllmain.cpp \
           daetools_fmi_cs.cpp

HEADERS += stdafx.h \
           daetools_fmi_cs.h

INCLUDEPATH += $${BOOSTDIR}

LIBS += -lcdaeSimulationLoader
# Achtung, Achtung!!
# It uses daetools/solibs for linking
QMAKE_LIBDIR += $${SOLIBS_DIR}

QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_APPEND} \
                  $${FMI_DIR}/$${TARGET}.$${SHARED_LIB_APPEND}
