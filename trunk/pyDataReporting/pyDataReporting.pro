include(../dae.pri)
QT -= core \
	gui
TARGET = pyDataReporting
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS +=	$${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${DAE_DATAREPORTERS_LIB}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32-msvc2008::QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyDataReporting1.dll \
	$${DAE_DEST_DIR}/pyDataReporting.pyd

unix::QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/pyDataReporting.so
