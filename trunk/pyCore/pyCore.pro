include(../dae.pri)
QT -= core \
	gui
TARGET = pyCore
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS +=	$${DAE_CORE_LIB} $${DAE_UNITS_LIB} $${BOOST_PYTHON_LIB} $${BOOST_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32{
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyCore1.dll \
	$${DAE_DEST_DIR}/pyCore.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/pyCore.so
}
