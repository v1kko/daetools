include(../dae.pri)
QT -= gui core
TARGET = cdae_whats_the_time
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app
INCLUDEPATH += $${BOOSTDIR}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR}

LIBS += $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${DAE_UNITS_LIB} \
        $${RT}

SOURCES += cdae_whats_the_time.cpp
