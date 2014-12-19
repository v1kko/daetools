TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp

LIBS += -lcdaeFMU_CS-py27
       #-lcdaeSimulationLoader-py27

QMAKE_LIBDIR += ../release

