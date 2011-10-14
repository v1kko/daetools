include(../dae.pri)
QT -= gui core
TARGET = Tutorials
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app
INCLUDEPATH += $${BOOSTDIR} \
               $${IPOPT_INCLUDE}

QMAKE_LIBDIR += $${SUNDIALS_LIBDIR} \
                $${IPOPT_LIBDIR} \
                $${MUMPS_LIBDIR}

LIBS += $${DAE_ACTIVITY_LIB} \
        $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
		#$${DAE_IPOPT_SOLVER_LIB} \
        $${SUNDIALS_LIBS} \
        $${SUPERLU_LIBS} \
        #$${DAE_SUPERLU_SOLVER_LIB} \
        $${BOOST_LIBS} \
        #$${IPOPT_LIBS} \
        $${DAE_UNITS_LIB} \
        $${MUMPS_LIBS} -lrt -lgfortran

SOURCES += main.cpp \
    stdafx.cpp
HEADERS += stdafx.h \
    variable_types.h \
	cdae_whats_the_time.h \
    cdae_tutorial1.h \
    cdae_tutorial2.h \
    cdae_tutorial3.h \
    cdae_tutorial4.h \
    cdae_tutorial5.h \
    cdae_tutorial6.h \
#    cdae_tutorial7.h \
#    cdae_tutorial8.h \
#    cdae_tutorial9.h \
#    cdae_opt_tutorial1.h \
#    cdae_opt_tutorial2.h \
#    cdae_opt_tutorial3.h \
