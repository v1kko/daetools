QT -= core gui
TARGET = daetools_ansic_simulation
TEMPLATE = app

QMAKE_LIBDIR += 
LIBS +=	

SOURCES += adouble.cpp main.cpp
HEADERS += adouble.h matrix.h daetools_model.h
