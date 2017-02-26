include(../dae.pri)

QT -= core gui
TARGET = cdaeActivity
TEMPLATE = lib
CONFIG += staticlib
INCLUDEPATH += $${BOOSTDIR} \
	           $${MPI_INCLUDE}

LIBS += $${DAE_CONFIG_LIB} \
        $${DAE_CORE_LIB} \
        $${MPI_LIBS}

HEADERS += stdafx.h \
    simulation.h \
    activity_class_factory.h \
    base_activities.h 
SOURCES += stdafx.cpp \
    simulation.cpp \
    dllmain.cpp \ 
    class_factory.cpp \
    base_activities.cpp \
    optimization.cpp


#win32{
#QMAKE_POST_LINK = copy /y  $${TARGET}.lib $${STATIC_LIBS_DIR}
#}

#unix{
#QMAKE_POST_LINK = cp -f  lib$${TARGET}.a $${STATIC_LIBS_DIR}
#}

#INSTALL_HEADERS = $$system($${COPY_FILES} simulation.h $${HEADERS_DIR}/Activity)

#######################################################
#                Install files
#######################################################
activity_headers.path  = $${HEADERS_DIR}/Activity
activity_headers.files = simulation.h

activity_libs.path         = $${STATIC_LIBS_DIR}
win32::activity_libs.files = $${DAE_DEST_DIR}/$${TARGET}.lib
unix::activity_libs.files  = $${DAE_DEST_DIR}/lib$${TARGET}.a

INSTALLS += activity_headers activity_libs
