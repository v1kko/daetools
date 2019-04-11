include(../dae.pri)
QT -= core
QT -= gui

TARGET = cdaeConfig-py$${PYTHON_MAJOR}$${PYTHON_MINOR}
TEMPLATE = lib
CONFIG += shared plugin

win32-msvc2015::QMAKE_CXXFLAGS += -DDAE_DLL_EXPORTS

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
DAE_PROJECT_NAME = $$basename(PWD)

install_headers.path  = $${DAE_INSTALL_HEADERS_DIR}/$${DAE_PROJECT_NAME}
install_headers.files = *.h

install_headers2.path  = $${DAE_INSTALL_HEADERS_DIR}
install_headers2.files = ../*.h

install_libs.path  = $${DAE_INSTALL_LIBS_DIR}
install_libs.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

# Install into daetools-package
install_py_solib.path  = $${SOLIBS_DIR}
install_py_solib.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

INSTALLS += install_headers install_headers2 install_libs install_py_solib
