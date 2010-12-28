include(../dae.pri)
QT -= core \
	gui
TARGET = pyIDAS
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${SUNDIALS_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                $${SUNDIALS_LIBDIR}

LIBS +=	$${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${DAE_SOLVER_LIB} \
        $${SUNDIALS_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32-msvc2008::QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyIDAS1.dll \
	$${DAE_DEST_DIR}/pyIDAS.pyd

unix::QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/pyIDAS.so
