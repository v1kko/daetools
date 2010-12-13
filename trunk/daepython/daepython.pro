include(../dae.pri)
QT -= core \
	gui
TARGET = daepython
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR} \
    $${PYTHON_INCLUDE_DIR} \
    $${PYTHON_SITE_PACKAGES_DIR} \
    $${SUNDIALS_INCLUDE}

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

LIBPATH += $${PYTHON_LIB_DIR}
LIBS +=	$${BOOST_PYTHON_LIB} $${BOOST_LIBS}

SOURCES += stdafx.cpp \
    dllmain.cpp \
    dae_python.cpp \
    python_wraps.cpp

HEADERS += stdafx.h \
    python_wraps.h

win32-msvc2008::QMAKE_POST_LINK = move /y \
	$${DAE_DEST_DIR}/daepython1.dll \
	$${DAE_DEST_DIR}/pyDAE.pyd

win32-g++::QMAKE_POST_LINK =  move /y \
	$${DAE_DEST_DIR}/$${TARGET}1.dll \
	$${DAE_DEST_DIR}/pyDAE.pyd

unix::QMAKE_POST_LINK = cp -f \
	$${DAE_DEST_DIR}/lib$${TARGET}.so.$${VERSION} \
	$${DAE_DEST_DIR}/pyDAE.so
