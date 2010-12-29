include(../dae.pri)
QT -= core \
	gui
TARGET = pyBONMIN
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR} \
               $${PYTHON_SITE_PACKAGES_DIR} \
               $${NLPSOLVER_INCLUDE}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR} \
                $${NLPSOLVER_LIBDIR}

LIBS += $${DAE_NLPSOLVER_LIB} \
        $${DAE_DATAREPORTERS_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_SIMULATION_LIB} \
        $${DAE_SOLVER_LIB} \
        $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS} \
        $${NLPSOLVER_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32-msvc2008::QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyBONMIN1.dll \
	$${DAE_DEST_DIR}/pyBONMIN.pyd

unix::QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/pyBONMIN.so
