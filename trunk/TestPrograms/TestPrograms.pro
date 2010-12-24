include(../dae.pri)
QT -= gui core
TARGET = TestPrograms
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app
INCLUDEPATH += $${BOOSTDIR} \
               $${IPOPT_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${IPOPT_LIBDIR} 

LIBS += $${DAE_SIMULATION_LIB} \
        $${DAE_DATAREPORTERS_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_SOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${IPOPT_LIBS}

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
