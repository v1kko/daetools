TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
           ../auxiliary.cpp \
           ../fmi_component.cpp \
           ../daetools_fmi_cs.cpp

HEADERS += ../fmi_component.h

INCLUDEPATH += .. \
               ../../boost

LIBS += -L../../boost/stage/lib -lboost_thread \
                                -lboost_system \
                                -lboost_regex \
                                -lpthread
