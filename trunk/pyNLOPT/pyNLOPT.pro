include(../dae.pri)
QT -= core gui
TARGET = pyNLOPT
TEMPLATE = lib

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${NLOPT_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${PYTHON_LIB_DIR} \
                $${NLOPT_LIBDIR}

LIBS += $${BOOST_PYTHON_LIB} \
        $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_NLOPT_SOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${NLOPT_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32{
QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyNLOPT1.dll \
	$${DAE_DEST_DIR}/pyNLOPT.pyd
}

unix{
QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/$${TARGET}.so
}
