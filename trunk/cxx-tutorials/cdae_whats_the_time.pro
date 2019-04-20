QT -= gui core
TARGET = cdae_tutorial1
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app

DESTDIR = ./bin

INCLUDEPATH += ../../boost \
               ../../OpenCS/build/include

QMAKE_LIBDIR += ../Linux_x86_64/lib

unix::QMAKE_CXXFLAGS  += -std=c++11
unix::QMAKE_LFLAGS    += -std=c++11

unix::DAE_CONFIG_LIB        = -lcdaeConfig
unix::DAE_CORE_LIB          = -lcdaeCore
unix::DAE_DATAREPORTING_LIB = -lcdaeDataReporting
unix::DAE_ACTIVITY_LIB      = -lcdaeActivity
unix::DAE_IDAS_SOLVER_LIB   = -lcdaeIDAS_DAESolver
unix::DAE_UNITS_LIB         = -lcdaeUnits

linux-g++::SOLIBS_RPATH = -Wl,-rpath,\'\$$ORIGIN/../lib\',-z,origin
macx-g++::SOLIBS_RPATH  = -Wl,-rpath,\'@loader_path/../lib\'
LIBS += $${SOLIBS_RPATH}

LIBS += $${DAE_DATAREPORTING_LIB} \
        $${DAE_CORE_LIB} \
        $${DAE_IDAS_SOLVER_LIB} \
        $${DAE_UNITS_LIB} \
        $${DAE_ACTIVITY_LIB}

SOURCES += cdae_whats_the_time.cpp
