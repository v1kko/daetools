include(../dae.pri)

TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp

QMAKE_CXXFLAGS += $$system($${PYTHON}-config --cflags)

QMAKE_LIBDIR += ../release

LIBS += -lcdaeSimulationLoader
