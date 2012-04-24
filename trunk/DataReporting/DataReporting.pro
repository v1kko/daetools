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


win32{
QMAKE_POST_LINK = copy /y  $${TARGET}.lib $${STATIC_LIBS_DIR}
}

unix{
QMAKE_POST_LINK = cp -f  lib$${TARGET}.a $${STATIC_LIBS_DIR}
}

INSTALL_HEADERS = $$system($${COPY_FILES} datareporters.h                  $${HEADERS_DIR}/DataReporting)
INSTALL_HEADERS = $$system($${COPY_FILES} base_data_reporters_receivers.h  $${HEADERS_DIR}/DataReporting)
