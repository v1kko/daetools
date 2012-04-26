include(../dae.pri)
QT -= core gui

# There is a problem compiling Units under MacOS X
# All optimization flags should be removed!!
macx-g++::QMAKE_CFLAGS_RELEASE   -= -O1 -O2 -O3
macx-g++::QMAKE_CXXFLAGS_RELEASE -= -O1 -O2 -O3

# TARGET = units
# CONFIG += console
# CONFIG -= app_bundle
# TEMPLATE = app

TARGET = cdaeUnits
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR}
QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS += $${BOOST_PYTHON_LIB} \
    $${BOOST_LIBS}
SOURCES += dllmain.cpp \
    units.cpp \
    stdafx.cpp \
    quantity.cpp

    # main.cpp

HEADERS += units.h \
    stdafx.h \
    units_pool.h


#win32{
#QMAKE_POST_LINK = copy /y  $${TARGET}.lib $${STATIC_LIBS_DIR}
#}

#unix{
#QMAKE_POST_LINK = cp -f  lib$${TARGET}.a $${STATIC_LIBS_DIR}
#}

#INSTALL_HEADERS = $$system($${COPY_FILES} units.h       $${HEADERS_DIR}/Units)
#INSTALL_HEADERS = $$system($${COPY_FILES} units_pool.h  $${HEADERS_DIR}/Units)

#######################################################
#                Install files
#######################################################
units_headers.path  = $${HEADERS_DIR}/Units
units_headers.files = units.h \
                      units_pool.h

units_libs.path         = $${STATIC_LIBS_DIR}
win32::units_libs.files = $${DAE_DEST_DIR}/$${TARGET}.lib
unix::units_libs.files  = $${DAE_DEST_DIR}/lib$${TARGET}.a

INSTALLS += units_headers units_libs
