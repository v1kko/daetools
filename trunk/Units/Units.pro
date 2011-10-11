include(../dae.pri)

QT -= core gui

#TARGET = units
#CONFIG += console
#CONFIG -= app_bundle
#TEMPLATE = app

TARGET = pyUnits
TEMPLATE = lib

#TARGET = cdaeUnits
#TEMPLATE = lib
#CONFIG += staticlib

INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS +=	$${BOOST_PYTHON_LIB} $${BOOST_LIBS}

SOURCES += units.cpp \
    dae_python.cpp
#    main.cpp
    

HEADERS += \
    parser_objects.h \
    units.h \
    stdafx.h \
    python_wraps.h

win32{
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyUnits.dll \
	$${DAE_DEST_DIR}/pyUnits.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/pyUnits.so
}
