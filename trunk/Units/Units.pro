include(../dae.pri)

QT -= core gui

#TARGET = units
#CONFIG += console
#CONFIG -= app_bundle
#TEMPLATE = app

TARGET = cdaeUnits
TEMPLATE = lib
CONFIG += staticlib

INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS +=	$${BOOST_PYTHON_LIB} $${BOOST_LIBS}

SOURCES += dllmain.cpp \
    units.cpp \
	stdafx.cpp
#    main.cpp
    

HEADERS += units.h \
    stdafx.h
