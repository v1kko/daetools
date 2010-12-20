include(../dae.pri)
QT -= gui core
TARGET = TestPrograms
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app
INCLUDEPATH += $${BOOSTDIR} 

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR}

win32-msvc2008:LIBS +=	Core.lib \
						Simulation.lib \
						DataReporters.lib \
						Solver.lib \
						$${SUNDIALS_LIBS}
unix:LIBS += -lCore \
			 -lSimulation \
			 -lDataReporters \
			 -lSolver \
		     $${SUNDIALS_LIBS} 

LIBS +=	$${BOOST_LIBS}

SOURCES += main.cpp \
    stdafx.cpp
HEADERS += stdafx.h \
    ../Examples/variable_types.h \
    ../Examples/test_models.h \
    ../Examples/straight_fin1.h \
    ../Examples/py_example.h \
    ../Examples/heat_cond1.h \
    ../Examples/first.h \
    ../Examples/cstr.h \
    ../Examples/conduction.h
