include(../dae.pri)
QT -= core \
    gui
TARGET = cdaeDataReporting
CONFIG += staticlib
TEMPLATE = lib
INCLUDEPATH += $${BOOSTDIR}
SOURCES += text_file_reporter.cpp \
    stdafx.cpp \
    html_file_reporter.cpp \
    filereporter.cpp \
    dllmain.cpp \
    data_reporter_local.cpp \
    class_factory.cpp \
    base_data_reporters_receivers.cpp \
    hybrid_data_reporter_receiver.cpp \
    xml_file_reporter.cpp \
    delegate_data_reporter.cpp \
    tcpip_data_reporter.cpp \
    data_reporter_remote.cpp \
    tcpip_data_receiver.cpp
HEADERS += stdafx.h \
    datareporting_class_factory.h \
    base_data_reporters_receivers.h \
    datareporters.h
