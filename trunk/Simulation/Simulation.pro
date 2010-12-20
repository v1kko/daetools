include(../dae.pri)

QT -= core gui
TARGET = Simulation
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR}

LIBS += DAE_CORE_LIB

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
