include(../dae.pri)

QT -= core gui
TARGET = Simulation
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR}

win32-msvc2008:LIBS += Core.lib
win32-g++:LIBS      += -lCore
unix::LIBS          += -lCore

HEADERS += stdafx.h \
    dyn_simulation.h \
    activity_class_factory.h \
    base_activities.h
SOURCES += stdafx.cpp \
    dyn_simulation.cpp \
    dllmain.cpp \
    class_factory.cpp \
    base_activities.cpp
