TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp

LIBS += -ldaetools_fmi_cs-Linux_x86_64

QMAKE_LIBDIR += ../release

