include(../dae.pri)
QT -= core \
    gui
TARGET = cdaeDataReporting
TEMPLATE = lib
CONFIG += shared plugin

INCLUDEPATH += $${BOOSTDIR}

LIBS += $${SOLIBS_RPATH}
LIBS +=	$${DAE_UNITS_LIB} \
        $${DAE_CONFIG_LIB} \
        $${BOOST_LIBS}

SOURCES += text_file_reporter.cpp \
    stdafx.cpp \
    html_file_reporter.cpp \
    filereporter.cpp \
    dllmain.cpp \
    data_reporter_local.cpp \
    class_factory.cpp \
    base_data_reporters_receivers.cpp \
    hybrid_data_reporter_receiver.cpp \
    delegate_data_reporter.cpp \
    tcpip_data_reporter.cpp \
    data_reporter_remote.cpp \
    tcpip_data_receiver.cpp
HEADERS += stdafx.h \
    datareporting_class_factory.h \
    base_data_reporters_receivers.h \
    datareporters.h

#######################################################
#                Install files
#######################################################
QMAKE_POST_LINK = $${COPY_FILE} \
                  $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT} \
                  $${SOLIBS_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

DAE_PROJECT_NAME = $$basename(PWD)

install_headers.path  = $${DAE_INSTALL_HEADERS_DIR}/$${DAE_PROJECT_NAME}
install_headers.files = *.h

install_libs.path  = $${DAE_INSTALL_LIBS_DIR}
install_libs.files = $${DAE_DEST_DIR}/$${SHARED_LIB_PREFIX}$${TARGET}$${SHARED_LIB_POSTFIX}.$${SHARED_LIB_EXT}

INSTALLS += install_headers install_libs
