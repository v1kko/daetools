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

unix::LIBS += -L../boost/stage/lib -lboost_thread \
                                   -lboost_system \
                                   -lboost_regex \
                                   -lboost_filesystem \
                                   -lpthread
win32-msvc2015::LIBS += -L../boost/stage/lib libboost_thread.lib \
                                             libboost_system.lib \
                                             libboost_regex.lib \
                                             libboost_filesystem.lib

#######################################################
#                Install files
#######################################################
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${FMI_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

# Install headers and libs into daetools-dev
DAE_PROJECT_NAME = $$basename(PWD)

install_headers.path  = $${DAE_INSTALL_HEADERS_DIR}/$${DAE_PROJECT_NAME}
install_headers.files = *.h

install_libs.path  = $${DAE_INSTALL_LIBS_DIR}
install_libs.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

# Install into daetools-package
install_py_solib.path  = $${FMI_DIR}
install_py_solib.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

INSTALLS += install_headers install_libs install_py_solib
