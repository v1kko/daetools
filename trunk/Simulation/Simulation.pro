include(../dae.pri)

QT -= core gui
TARGET = Simulation
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
               $${BONMIN_INCLUDE}

LIBS += $${DAE_CORE_LIB} \
        $${BONMIN_LIBS}

HEADERS += stdafx.h \
    dyn_simulation.h \
    activity_class_factory.h \
    base_activities.h \
    optimization.h
SOURCES += stdafx.cpp \
    dyn_simulation.cpp \
    dllmain.cpp \
    class_factory.cpp \
    base_activities.cpp \
    optimization.cpp
