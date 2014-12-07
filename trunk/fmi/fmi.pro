include(../dae.pri)

QT -= core
QT -= gui

TEMPLATE = lib
CONFIG += shared

TARGET  = daetools_fmi_cs

SOURCES += dllmain.cpp \
           daetools_fmi_cs.cpp

HEADERS += stdafx.h \
           daetools_fmi_cs.h

INCLUDEPATH += $${BOOSTDIR}

LIBS += -lcdaeSimulationLoader

unix::QMAKE_POST_LINK = $${COPY_FILE} \
                        $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}* \
                        $${SOLIBS_DIR}
win32::QMAKE_POST_LINK = $${COPY_FILE} \
                         $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}* \
                         $${SOLIBS_DIR}/$${TARGET}.$${SHARED_LIB_APPEND}
