include(../dae.pri)
QT -= core \
	gui
TARGET = pyActivity
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR}

win32-msvc2008::LIBS += Core.lib \
						Simulation.lib \
						DataReporters.lib \
						Solver.lib \

win32-g++::LIBS +=  -lCore \
					-lSimulation \
					-lDataReporters \
					-lSolver

unix::LIBS += -lCore \
              -lSimulation \
              -lDataReporters \
              -lSolver 

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}
LIBS +=	$${BOOST_PYTHON_LIB} $${BOOST_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32-msvc2008::QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/pyActivity1.dll \
	$${DAE_DEST_DIR}/pyActivity.pyd

win32-g++::QMAKE_POST_LINK =  move /y \
	$${DAE_DEST_DIR}/$${TARGET}1.dll \
	$${DAE_DEST_DIR}/pyActivity.pyd

unix::QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/pyActivity.so
