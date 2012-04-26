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


#win32{
#QMAKE_POST_LINK = copy /y  $${TARGET}.lib $${STATIC_LIBS_DIR}
#}

#unix{
#QMAKE_POST_LINK = cp -f  lib$${TARGET}.a $${STATIC_LIBS_DIR}
#}

#INSTALL_HEADERS = $$system($${COPY_FILES} datareporters.h                  $${HEADERS_DIR}/DataReporting)
#INSTALL_HEADERS = $$system($${COPY_FILES} base_data_reporters_receivers.h  $${HEADERS_DIR}/DataReporting)

#######################################################
#                Install files
#######################################################
datareporting_headers.path   = $${HEADERS_DIR}/DataReporting
datareporting_headers.files  = datareporters.h \
                               base_data_reporters_receivers.h

datareporting_libs.path         = $${STATIC_LIBS_DIR}
win32::datareporting_libs.files = $${DAE_DEST_DIR}/$${TARGET}.lib
unix::datareporting_libs.files  = $${DAE_DEST_DIR}/lib$${TARGET}.a

INSTALLS += datareporting_headers datareporting_libs
