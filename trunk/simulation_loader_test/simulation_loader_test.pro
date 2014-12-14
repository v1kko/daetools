TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp

LIBS += -ldaetools_fmi_cs-Linux_x86_64
       #-lcdaeSimulationLoader

QMAKE_LIBDIR += ../release

