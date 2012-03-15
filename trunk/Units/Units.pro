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
