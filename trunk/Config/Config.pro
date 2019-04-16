include(../dae.pri)
QT -= core
QT -= gui

TARGET = cdaeConfig-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
TEMPLATE = lib
CONFIG += shared plugin

win32::QMAKE_CXXFLAGS += -DCONFIG_EXPORTS

INCLUDEPATH += $${BOOSTDIR}

LIBS +=	$${BOOST_LIBS}

SOURCES += config.cpp \
    stdafx.cpp \
    dllmain.cpp

HEADERS += ../config.h \
    stdafx.h

#######################################################
#                Install files
#######################################################
#QMAKE_POST_LINK = $${COPY_FILE} \
#                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
#                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

# Install headers and libs into daetools-dev
include(../dae_install_library.pri)
include(../dae_install_other.pri)
