include(../dae.pri)

CONFIG += c++11
CONFIG -= app_bundle
CONFIG -= qt

QT -= core
QT -= gui

TEMPLATE = lib
# Add "plugin" to avoid creation of symlinks
CONFIG += shared plugin

TARGET = cdaeFMU_CS_WS

unix::QMAKE_CXXFLAGS += -fvisibility=hidden

SOURCES += dllmain.cpp \
           auxiliary.cpp \
           fmi_component.cpp \
           daetools_fmi_cs.cpp

HEADERS += stdafx.h \
           fmi_component.h \
           daetools_fmi_cs.h

INCLUDEPATH += ../boost

LIBS += -L../boost/stage/lib -lboost_thread \
                             -lboost_system \
                             -lboost_regex \
                             -lpthread

QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                  $${FMI_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}
