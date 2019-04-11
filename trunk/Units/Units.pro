include(../dae.pri)
QT -= core gui

TARGET = cdaeUnits
TEMPLATE = lib
CONFIG += shared plugin

# There is a problem compiling Units under MacOS X
# All optimization flags should be removed!!
macx-g++::QMAKE_CFLAGS_RELEASE   -= -O1 -O2 -O3
macx-g++::QMAKE_CXXFLAGS_RELEASE -= -O1 -O2 -O3

INCLUDEPATH += $${BOOSTDIR} 

LIBS += $${SOLIBS_RPATH}
LIBS +=	$${BOOST_LIBS}

SOURCES += dllmain.cpp \
    units.cpp \
    stdafx.cpp \
    quantity.cpp

HEADERS += units.h \
    stdafx.h \
    units_pool.h

#######################################################
#                Install files
#######################################################
QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

# Install headers and libs into daetools-dev
DAE_PROJECT_NAME = $$basename(PWD)

install_headers.path  = $${DAE_INSTALL_HEADERS_DIR}/$${DAE_PROJECT_NAME}
install_headers.files = *.h

install_libs.path  = $${DAE_INSTALL_LIBS_DIR}
install_libs.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

# Install into daetools-package
install_py_solib.path  = $${SOLIBS_DIR}
install_py_solib.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

INSTALLS += install_headers install_libs install_py_solib
