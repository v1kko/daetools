include(../dae.pri)

QT -= core gui

TARGET = units
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app

#TARGET = cdaeUnits
#TEMPLATE = lib
#CONFIG += staticlib


SOURCES += main.cpp \
    units.cpp

HEADERS += \
    parser_objects.h \
    units.h
