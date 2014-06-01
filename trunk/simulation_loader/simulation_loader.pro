include(../dae.pri)

QT       -= core
QT       -= gui

TARGET  = simulation_loader
#CONFIG  += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
           simulation_loader.cpp

HEADERS += simulation_loader.h

message($$system(python2.7-config --cflags))
message($$system(python2.7-config --ldflags))
QMAKE_CXXFLAGS += $$system(python2.7-config --cflags)

#win32::EMBED_BOOST_PYTHON_LIB = $${BOOST_PYTHON_LIB_NAME}.lib $$system(python2.7-config --ldflags)
#unix::EMBED_BOOST_PYTHON_LIB  = -L$${BOOSTLIBPATH} -l$${BOOST_PYTHON_LIB_NAME} $$system(python2.7-config --ldflags)

INCLUDEPATH += $${BOOSTDIR} \
               $${PYTHON_INCLUDE_DIR}

QMAKE_LIBDIR += $${PYTHON_LIB_DIR}

LIBS += $${BOOST_PYTHON_LIB} \
        $${BOOST_LIBS}

#$${DAE_ACTIVITY_LIB} \
#$${DAE_DATAREPORTING_LIB} \
#$${DAE_CORE_LIB} \
#$${DAE_IDAS_SOLVER_LIB} \
#$${DAE_UNITS_LIB} \
#$${SUNDIALS_LIBS} \
#$${BOOST_PYTHON_LIB} \
#$${BOOST_LIBS} \
#$${BLAS_LAPACK_LIBS}


