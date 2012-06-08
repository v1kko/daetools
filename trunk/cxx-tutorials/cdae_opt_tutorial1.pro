include(../dae.pri)
QT -= gui core
TARGET = cdae_opt_tutorial1
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app
INCLUDEPATH += $${BOOSTDIR} \
               $${IPOPT_INCLUDE}
DESTDIR = bin

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${IPOPT_LIBDIR} \
                $${BONMIN_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_UNITS_LIB} \
        $${DAE_IPOPT_SOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${BOOST_LIBS} \
        $${IPOPT_LIBS} \
        $${MUMPS_LIBS} \
        $${BLAS_LAPACK_LIBS} \
        $${RT}

SOURCES += cdae_opt_tutorial1.cpp
